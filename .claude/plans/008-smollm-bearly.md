# 008 — SmolLM2-135M-Instruct on Bearly ML 25 (`c2c-demos/bearly-smollm`)

**Status (2026-08-15):** built, validated on the host, and **the RISC-V binary validated in Spike**;
not yet run on silicon. The exported int8 model reproduces HuggingFace fp32 greedy decoding token
for token, the device's own tokenizer C matches HF on 523/523 corpus lines, the device's own forward
C matches the numpy reference on all 32 stages, and the actual cross-compiled binary — RVV kernel
included — reproduces those 32 stages in Spike to ≤2.4e-6.

Goal: the `bearly25-demos/borai` TinyStories demo, but with a real instruction-tuned 135M model —
type a question over UART, get an answer — using an `llm` linker variant for bearly25 modelled on
the DSP one, with the stack placed in DRAM that actually exists.

## What was built

| Piece | Path |
|---|---|
| Linker variant | `platform/bearly25/bearly25-llm.ld` |
| Board glue (PLL/UART, stack paint, chat loop) | `c2c-demos/bearly-smollm/src/main.c` |
| Forward pass, no chip dependencies | `c2c-demos/bearly-smollm/src/model.c` |
| Byte-level BPE, no chip dependencies | `c2c-demos/bearly-smollm/src/tokenizer.c` |
| Exporter | `scripts/export_smollm.py` |
| Python model of the tokenizer + HF diff | `scripts/tokenizer_ref.py` |
| Numpy reference + golden dump | `scripts/ref_runq.py` |
| **Device C** tokenizer vs HF, on the host | `scripts/check_c_tokenizer.py`, `test/host_tokenizer_test.c` |
| **Device C** forward vs the reference, on the host | `scripts/check_c_forward.py`, `test/host_forward_test.c` |
| Blobs (untracked, 144 MB) | `c2c-demos/bearly-smollm/model/{model_q80.bin,tokenizer.bin}` |

The `main.c` / `model.c` / `tokenizer.c` split exists for one reason: the two files that contain all
the arithmetic have no chip dependencies, so **the exact source that runs on silicon is compiled for
the host and diffed against the reference** in seconds instead of after a 26-minute flash. That is
what caught the `softmax` linkage break (below) and is what the whole validation table rests on.

Build: `make build CHIP=bearly25 PLATFORM=CHIP TARGET=bearly-smollm EXTRA_CMAKE_ARGS="-DLINKER=llm"`.
Full instructions and the option table are in the demo's README.

## The four things that were not obvious

### 1. The embedding table cannot be dequantized up front

`runq.c` (and therefore `dsp25-demos/tinyllama`) dequantizes the whole token embedding table at
load: `malloc(vocab_size * dim * sizeof(float))`. For SmolLM's 49152 × 576 vocab that is **113 MB
of float** — on top of a 143 MB model, in 256 MB of DRAM — to hold something the forward pass reads
576 floats of per token. `embed_token()` dequantizes one row on demand instead. This is the single
change that makes a 135M model fit at all.

### 2. `llama2.c/export.py --hf` gets SmolLM wrong in four ways

Hence `scripts/export_smollm.py`:

- `load_hf_model()` sets `config.n_kv_heads = num_attention_heads`, silently discarding SmolLM's
  9-query/3-KV grouped attention;
- it reads `hf_dict['lm_head.weight']`, which does not exist for a tied model (transformers omits
  it from the state dict entirely);
- RoPE theta is hardcoded to 10000 in both the exporter's world and `runq.c`; **SmolLM uses
  100000**. This one produces fluent-looking garbage rather than an obvious failure, so it is worth
  checking first if output is ever wrong;
- `permute_reverse` (which de-interleaves HF's split-half RoPE layout into llama2.c's
  adjacent-pairs convention) must be called with `n_kv_heads`/`kv_dim` for `wk`, not the query-head
  defaults.

Also, transformers 5 moved `rope_theta` into a `rope_parameters` dict, so `getattr(cfg,
"rope_theta", 10000.0)` silently returns the **default** on a current install. The exporter reads
both and asserts it found one.

The blob keeps llama2.c's `--version 2` layout, with `rope_theta` / `norm_eps` / `head_dim` written
into the unused part of the 256-byte header behind an `'SMLM'` magic — so a stock `runq.c` still
parses it, and the device notices when the extension is absent instead of assuming llama2 defaults.

### 3. The tokenizer is byte-level BPE, not SentencePiece

llama2.c's `tokenizer.bin` (score + string per token, merge by best score) cannot represent GPT-2
byte-level BPE correctly: merging by a *token's* score picks the wrong merge whenever two different
pairs produce the same string. The export emits the merge list as `(a, b, ab)` **id triples in rank
order** and the device hashes pair → rank, which is exact BPE and needs no string compares in the
merge loop. Decoding uses a stored raw-bytes piece per token, so the device never manipulates the
`Ġ`-alphabet strings at all.

Pre-tokenization is the only approximation: GPT-2's ByteLevel regex is reimplemented over byte
classes, with every byte ≥ 0x80 treated as a letter. Two findings from the differential test:

- SmolLM's `Digits(individual_digits)` pre-tokenizer needs **no** special case on the device. It
  applied when the tokenizer was trained, so no vocab token holds two ASCII digits and plain BPE
  leaves digit runs split regardless. Modelling it explicitly was *wrong* — it merged `"  1"` into
  one two-space token where HF emits two single spaces.
- 21 bytes (some control codes, and `0xC0`/`0xC1`/`0xF1`-`0xFF`, which never start valid UTF-8) have
  no token in this vocab and no byte fallback exists. They are dropped on encode, matching HF.

`scripts/tokenizer_ref.py --check` is the regression test: 715/715 corpus lines and the chat
template, exact.

### 4. Stack placement — the reason for a new linker script

`platform/bearly25/bearly25-llm.ld` differs from `bearly25.ld` in exactly two ways:

- **heap and stack are bare `PROVIDE()` symbols, not `NOLOAD` sections.** A `.heap` section extends
  the LOAD segment's `p_memsz` to its end and fesvr/uart_tsi zero-fills `[filesz, memsz]` over the
  UART — ~100 MB of transmitted zeros on top of an already 26-minute load. Verified on the built
  ELF: `filesz 0x8989868`, `memsz 0x898cba0` (the difference is just `.bss`).
- **DRAM keeps its real 256 MB length.** `dsp25-llm.ld` declares 2048 MB so a 1.1 GB TinyLlama blob
  fits inside medany's ±2 GiB PC-relative range, then derives `__stack_start` from that fictitious
  region end — ~1.9 GB past the last backed address, where a 4 KiB stack corrupts silently and a
  256 KiB one hangs the core (plan 007). Nothing here needs more than 256 MB, so the region stays
  honest and the stack is inside real memory *by construction*. The demo still pins it explicitly
  with `--defsym __stack_start=0x8FE00000` and paints it at boot.

## Validation done (host)

| Check | Result |
|---|---|
| `tokenizer_ref.py --check --corpus-file CLAUDE.md` | 547/547 exact vs HF |
| `tokenizer_ref.py --check --corpus-file .claude/plans/007-citrinet-dsp.md` | 715/715 exact, chat template MATCH |
| `check_c_tokenizer.py --corpus CLAUDE.md` (**the device C**) | 523/523 exact vs HF |
| `ref_runq.py --check` (int8 device arithmetic vs HF fp32 greedy) | `'The capital of France is Paris.'`, 7/7 tokens identical |
| `check_c_forward.py` (**the device C** vs the reference) | prompt identical, 32/32 stages within 2.3e-6, PASS |
| Q8_0 quantization | worst per-group error 0.036 |
| ELF layout | blob at `0x800115c0`, `__end 0x8898dce0`, `__heap_end`/`__stack_start` `0x8fe00000`, stacks end `0x8ff40000` < `0x90000000`, `memsz - filesz` = 15 KB (`.bss` only) |
| RVV codegen | `vwmul`/`vwredsum` present in `matmul` |
| **Spike**, the cross-compiled ELF with RVV | all 32 golden stages within **2.4e-6**; `"Hi"` → `"Hello"`; stack high-water **1976 B of 256 KiB**, 486/494 words dense |

Spike run (note the `PLATFORM=SIMS` requirement — and that it takes minutes per token, so use a
short prompt and read the golden lines rather than waiting for text):

```bash
cmake -S ./ -B ./build-smollm-spike -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=./riscv-gcc.cmake -DCHIP=bearly25 -DPLATFORM=SIMS -DLINKER=llm \
  -DSMOLLM_DEBUG_GOLDEN=ON -DSMOLLM_PROMPT=Hi -DSMOLLM_MAX_NEW_TOKENS=1
cmake --build ./build-smollm-spike --target bearly-smollm
spike --isa=rv64gcv_zicntr_zfh_zvl256b build-smollm-spike/c2c-demos/bearly-smollm/bearly-smollm.elf
```

**`init_test()` must be skipped under `PLATFORM=SIMS`.** Spike models neither the PLL nor the UART,
so programming them faults before the first character is printed — the first Spike attempt here
produced *zero output for 25 minutes* and looked exactly like a hang. `app_init()` now calls it only
under `TERMINAL_DEVICE_UART0`; the SIMS console is HTIF.

The numpy reference is an exact oracle for a single position: int8 products and their 64-wide group
sums are exactly representable in float32, so its einsum reproduces the device's int32 accumulation
bit for bit.

### Two things this found that would otherwise have been on-silicon mysteries

**1. `softmax` linked only by luck.** Splitting the arithmetic out of `main.c` left `sample_token()`
calling a `softmax` that is now `static` in `model.c`. It still linked — because at
`SMOLLM_TEMPERATURE=0.0f` the compiler folds the whole sampling branch away — so the break would
only have appeared the first time someone built with a non-zero temperature. Now exported as
`smollm_softmax()`, and both temperature settings are build-tested.

**2. The pipeline is chaotic, so only position 0 is reproducible.** The C and the numpy reference
generate different text after 7 tokens, which looked like a bug and is not. Traced with the new
per-op layer trace: every op agrees to ~1e-6 through layer 13's attention at position 1, and then
the int8 activations feeding `wo` sum to **-45** in C and **-44** in numpy — one activation flipped
one quantization step because activation quantization is a step function and the two implementations
accumulate float sums in different orders (sequential vs numpy's pairwise). By layer 28 they are 6%
apart, and generated token 8 is a near-tie (`<|im_end|>` 16.31 vs `" It"` 16.08) that tips the other
way. Ruled out along the way: reciprocal-vs-divide in `quantize` (no effect, though the code now
divides like llama2.c), attention summation order (no effect), and compiler FP freedom (`-O2` and
`-O3 -ffast-math` agree exactly).

Consequence for silicon: **judge the chip by the position-0 golden check and by whether the text
makes sense**, never by equality with the reference's token stream. Position 0 is exact precisely
because attention has one term there and nothing amplifies.

## Optimization results on silicon (2026-08-15)

Decode went 8,641 -> 4,782 Mcyc/token (~6.4 s at 750 MHz) and prefill 265,620 -> 10,548 Mcyc.
What actually mattered, in order:

| change | effect |
|---|---|
| **Batched prefill** (one weight pass for the whole prompt, not one per token) | prefill **22x** |
| **Full-cache-line loads** (`e8m2`, 64 B) instead of 32 B | decode **1.43x** |
| Skipping the classifier for all but the last prompt token | part of the prefill win |
| Dual-core (row split across harts, probe-gated) | prefill 1.10x, decode **1.07x** |
| int4 (Q4_1 layers) | decode 1.15x, but **+30% perplexity** — NOT shipped |

**Decode is now at 1.11x of the measured memory floor** (136 MB x 31.6 cyc/byte = 4,298 Mcyc), so
kernel work is finished. The ceiling is the memory system: ~31.6 cyc/byte with full-line loads, one
line-fill in flight, and no benefit from concurrency (see the RTL list, entry 3).

### Things that were measured and rejected — do not re-derive these

- **int4.** Q4_1 layers: 1.15x decode for perplexity 44.5 -> 57.9. Q4_1 everything: ~1.36x for 67.8.
  Q4_0 is strictly dominated by Q4_1 (same or worse quality at every size). The classifier is the
  most quantization-sensitive tensor — Q4_0 on it doubled perplexity. Exporter supports all of them
  (`--quant q41-layers` etc.); int8 remains the default because the demo is about answering well.
- **Speculative decoding — structurally dead on this hardware.** Verification is cheap here (one
  weight pass for K+1 tokens = ~1.11 normal decodes, and greedy speculative decoding is EXACT, so
  it costs no quality). But every draft that touches DRAM costs a full memory pass: drafting with
  the Q4_1 model would cost 3,053 Mcyc/token against a 4,782 Mcyc decode. The draft must therefore
  fit in the 256 KB cache, and measured acceptance for such drafts is far below the ~10% break-even:
  distilled bigram 6.0%, 2-gram history lookup 1.7%, combined 7.7% — i.e. **0.92-0.98x, a slowdown**.
  Reproduce with `scripts/measure_draft.py`. It would take a TRAINED 4-8 MB draft model (~60%
  acceptance -> ~2x) to make this work, which is a distillation project, not a port.
- **Enabling L2 ways.** `WayEnable` reads 0 and ignores writes, but this is the open-source SiFive
  cache where all ways are always active. Not a defect, not a lever.

## Open / next

- **Run it on silicon.** Nothing here has touched the chip. Expected first-light output is in the
  README. If the answer is wrong, build with `-DSMOLLM_DEBUG_GOLDEN=ON` — the device prints the
  residual-stream sum after all 32 stages of the first forward pass next to the host's, so a
  divergence names a layer.
- **Speed is unmeasured on silicon.** Spike reports ~111 Mcyc per generated token and ~3312 Mcyc for
  a 31-token prefill, but it charges one cycle per instruction and models no memory system (citrinet
  measured it undercounting silicon ~13x), so that is an instruction count, not a prediction.
  Each token streams all 143 MB of weights, so this is bandwidth-bound;
  the useful next optimization is a transposed weight layout that accumulates across output rows in
  vector lanes and drops the per-group reduction (what `borai`'s `TRANSPOSED_WEIGHTS` does for the
  per-tensor-scale case). Do not tune from Spike — it undercounts silicon cycles by ~13x (plan 007).
- **Flash time is not a constraint — use the SPI loader.** ~144 MB is ~40 s at 30 MHz SPI versus
  ~26 minutes over uart_tsi at 921600 baud. The two-stage preload alternative (weights-only ELF
  loaded once, small program ELF per iteration) is therefore unnecessary; iterate freely and bisect
  on hardware rather than bundling every diagnostic into one image.
- `seq_len` is clamped to 512 at export (`--seq-len`), which is what sizes the 23.6 MB KV cache.
  There is DRAM headroom for ~1024 if longer conversations are wanted.
