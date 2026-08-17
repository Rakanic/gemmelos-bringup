# bearly-smollm — SmolLM2-135M-Instruct on Bearly ML 25

A 135M-parameter instruction-tuned LLM running baremetal on the BML chip: type a question over
the UART console, get an answer. Same lineage as `bearly25-demos/borai` (llama2.c runq int8), but
with a real 135M model instead of the 260K TinyStories one, so it needed a different memory plan,
a different tokenizer, and a linker script that puts the stack somewhere that exists. The whole model rides inside the ELF,
so every flash moves ~144 MB: seconds over the 30 MHz SPI loader, ~26 minutes over uart_tsi.

```
You: What is the capital of France?
SmolLM:
The capital of France is Paris.
[smollm] prompt 37 tok in 5120 Mcyc, generated 8 tok in 1104 Mcyc (1.087 tok/s)
```

## Build and run

```bash
# 1. once: fetch SmolLM2-135M-Instruct and write model/model_q80.bin (143 MB) + model/tokenizer.bin
.venv/bin/python c2c-demos/bearly-smollm/scripts/export_smollm.py

# 2. build (LINKER=llm is required — see "Memory" below)
make build CHIP=bearly25 PLATFORM=CHIP TARGET=bearly-smollm EXTRA_CMAKE_ARGS="-DLINKER=llm"

# 3. flash. The model travels inside the ELF, so this moves ~144 MB — the load path decides how
#    long that takes: uart_tsi at 921600 baud is ~26 min, the 30 MHz SPI loader ~40 s. Use SPI.
make tsi-run TTY=<tty> BINARY=build/c2c-demos/bearly-smollm/bearly-smollm.elf
```

Useful build options (all go inside `EXTRA_CMAKE_ARGS`, and new CMake options need a clean
reconfigure):

| Option | Default | What it does |
|---|---|---|
| `-DSMOLLM_TEMPERATURE=0.8f` | `0.0f` | 0 = greedy (deterministic, matches the host reference); higher = livelier |
| `-DSMOLLM_TOPK=40` | 40 | sample from the k most likely tokens when the temperature is non-zero |
| `-DSMOLLM_MAX_NEW_TOKENS=200` | 200 | cap on answer length |
| `-DSMOLLM_PROMPT="What is a transistor?"` | — | generate once and halt instead of reading the UART (required for Spike) |
| `-DSMOLLM_DEBUG_GOLDEN=ON` | OFF | check every layer's output against the host reference (see below) |
| `-DSMOLLM_USE_RVV=OFF` | ON | scalar int8 dot product; bit-identical, just slower |
| `-DSMOLLM_TARGET_FREQ_HZ=500000000` | 750 MHz | PLL target |

## The model

SmolLM2-135M-Instruct: 30 layers, dim 576, hidden 1536, 9 query heads over 3 KV heads (GQA),
head_dim 64, vocab 49152, tied classifier, RoPE theta **100000**, RMSNorm eps 1e-5. Quantized to
grouped Q8_0 (group 64), which is 143.0 MB — int8 values plus one fp32 scale per 64 weights. Worst
per-group quantization error is 0.036, and the quantized model reproduces HuggingFace's fp32 greedy
output token for token on the prompts tested.

`scripts/export_smollm.py` writes the blob rather than `llama2.c/export.py --hf`, which would get
four things wrong for this model: it hardcodes `n_kv_heads = n_heads` (breaking GQA), assumes an
untied `lm_head`, drops `rope_theta` (SmolLM is 100000, llama2 is 10000), and its tokenizer export
is SentencePiece-only. The blob keeps llama2.c's `--version 2` layout so the structure is familiar,
with rope_theta / norm_eps / head_dim added in the unused part of the 256-byte header.

## The tokenizer

SmolLM uses GPT-2 style **byte-level BPE**, not SentencePiece, so `tokenizer.bin` is a different
format from llama2.c's: a byte→id table, the merge list as `(a, b, ab)` id triples in rank order,
and one raw-bytes piece per token for decoding. The device merges the lowest-rank adjacent pair via
a hash of pair→rank, which is *exact* BPE with no string comparisons in the merge loop.

Pre-tokenization (GPT-2's ByteLevel regex) is the one approximation — the C classifies bytes rather
than Unicode categories, treating every byte ≥ 0x80 as a letter. `scripts/tokenizer_ref.py --check`
models the C exactly and diffs it against HuggingFace: **715/715 corpus lines exact**, including the
chat template. Rerun it after touching `chunk_end()`:

```bash
.venv/bin/python c2c-demos/bearly-smollm/scripts/tokenizer_ref.py --check --corpus-file CLAUDE.md
```

## Memory (this is the part that bites)

DRAM is 256 MB at `0x8000_0000`.

```
0x80000000  .text                          ~70 KB
0x80010c80  .rodata: model blob           143.0 MB   .incbin, read in place
0x88877280           tokenizer blob          1.1 MB
0x8898cba0  __end -> heap                 ~117 MB available, ~26 MB used
                     KV cache               23.6 MB   30 layers x 512 pos x 192 x 2 x 4 B
                     tokenizer merge table   2.0 MB
                     logits, RunState        ~0.3 MB
0x8fe00000  stacks                        5 x 256 KiB, pinned by --defsym
0x90000000  top of DRAM
```

Two things make this fit, and both are easy to get wrong:

1. **The embedding table is dequantized one token at a time.** `runq.c` dequantizes all of it up
   front; for a 49152 × 576 vocab that is a **113 MB float array** — nearly half of DRAM to hold
   something the forward pass reads 576 floats of per token.

2. **`LINKER=llm` (`platform/bearly25/bearly25-llm.ld`) declares the heap and stack as bare
   symbols, not `NOLOAD` sections.** A `.heap` section extends the LOAD segment's `p_memsz` to its
   end, and fesvr/uart_tsi zero-fills `[filesz, memsz]` over the serial link — that would add ~100 MB
   of transmitted zeros to an already 26-minute load. As symbols, `p_memsz` stops just past `.bss`
   (verified: `filesz 0x8989868`, `memsz 0x898cba0`) and `sbrk` grows into the gap at runtime.

That linker script also keeps DRAM at its **real** 256 MB, unlike `dsp25-llm.ld`, which declares
2048 MB so a 1.1 GB TinyLlama blob fits in medany's ±2 GiB range and then derives the stack from
that fictitious end — landing it ~1.9 GB past the last backed address, where a 4 KiB stack corrupts
silently and a 256 KiB one hangs the core. The demo also pins the stack explicitly with `--defsym`
and paints it at boot, reporting the high-water mark after the first answer:

```
[smollm] stack: hart0 [0x8fe00000, 0x8fe40000) = 256 KB, sp=0x8fe3fff0
[smollm] stack: high-water 1104 of 262144 bytes (0%), 274/276 words disturbed
```

## Validating without a chip

`src/model.c` and `src/tokenizer.c` have **no chip dependencies**, so the code that runs on silicon
can be compiled for the host and checked directly. That is the whole point of the split — `main.c`
is only the board glue (PLL/UART bring-up, stack painting, the chat loop).

```bash
# 1. the exported model vs HuggingFace fp32 greedy decoding, through a numpy model of the device
#    arithmetic (exact: int8 products and their 64-wide group sums are exact in float32)
.venv/bin/python c2c-demos/bearly-smollm/scripts/ref_runq.py --check --steps 24
#   hf   fp32: 'The capital of France is Paris.'   dev q8_0: same, 7/7 tokens identical

# 2. the DEVICE tokenizer C, compiled for the host, vs HuggingFace over a corpus
.venv/bin/python c2c-demos/bearly-smollm/scripts/check_c_tokenizer.py --corpus CLAUDE.md
#   C tokenizer: 523/523 lines exact vs HF

# 3. the DEVICE forward pass C, compiled for the host, vs the numpy reference
.venv/bin/python c2c-demos/bearly-smollm/scripts/check_c_forward.py
#   prompt: 37 tokens, identical
#   stages: 32 compared, worst relative difference 2.265e-06
#   C output: 'The capital of France is Paris. It is a city that has been a beacon'
#   PASS

# 4. per-layer golden sums for the on-chip check
.venv/bin/python c2c-demos/bearly-smollm/scripts/ref_runq.py --golden \
    c2c-demos/bearly-smollm/include/smollm_golden.h
```

Regenerate `include/smollm_golden.h` whenever the model blob changes, then build with
`-DSMOLLM_DEBUG_GOLDEN=ON`; the device prints `GOLDEN <stage> got … want … rel …` for all 32 stages
of the first forward pass and flags any stage more than 2% off. If a stage diverges, narrow it to a
single op with `smollm_set_trace_layer(l)` (the host test exposes it as `SMOLLM_TRACE_LAYER=13`),
which reports the norm, q/k/v, RoPE, attention output, `wo`, SwiGLU and `w2` inside that layer.

### Spike

The cross-compiled binary — RVV kernel included — runs the whole demo in Spike: all 32 golden stages
within 2.4e-6 of the reference, and `"Hi"` answered with `"Hello"`. Build with `PLATFORM=SIMS`
(which switches the console to HTIF) and keep the prompt short; Spike takes minutes per token.

```bash
cmake -S ./ -B ./build-smollm-spike -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=./riscv-gcc.cmake -DCHIP=bearly25 -DPLATFORM=SIMS -DLINKER=llm \
  -DSMOLLM_DEBUG_GOLDEN=ON -DSMOLLM_PROMPT=Hi -DSMOLLM_MAX_NEW_TOKENS=1
cmake --build ./build-smollm-spike --target bearly-smollm
spike --isa=rv64gcv_zicntr_zfh_zvl256b build-smollm-spike/c2c-demos/bearly-smollm/bearly-smollm.elf
```

`app_init()` skips `init_test()` unless `TERMINAL_DEVICE_UART0` is defined: Spike models neither the
PLL nor the UART, and programming them faults before the first character is printed — which looks
exactly like a hang (it silently ate 25 minutes here before the cause was found).

### Expect the token stream to drift, and check position 0 instead

int8 activation quantization is a **step function**, so this pipeline is chaotic: a last-bit float
difference eventually pushes one activation across a rounding boundary, and from there the two runs
are different (but equally valid) evaluations of the same model. Measured between the C and the
numpy reference on `"What is the capital of France?"`:

| | |
|---|---|
| position 0, all 32 stages | agree to **2.3e-6** |
| position 1, every op through layer 13's attention | agree to ~1e-6 |
| position 1, the int8 activations feeding layer 13's `wo` | sum to **-45** in C, **-44** in numpy — one value off by one |
| position 1, layer 28 | 6% apart |
| generated token 8 | reference picks `<|im_end|>` at logit 16.31 over `" It"` at 16.08; the C picks `" It"` |

Position 0 is the exact one because attention has a single term there and nothing amplifies — which
is why the golden check runs at position 0 by default (`smollm_set_stage_pos()` moves it). Judge
generated text by whether it makes sense, not by equality with the reference.

## Performance notes

**Silicon speed is unmeasured.** The Spike run reports ~111 Mcyc per generated token, but Spike
charges one cycle per instruction and models no memory system — `dsp-citrinet` measured it
undercounting silicon by ~13x — so treat that as an instruction count, not a prediction. Each token
reads all 143 MB of weights, so expect this to be bandwidth-bound.

The int8 dot product uses RVV (`vwmul` + `vwredsum` per group of 64) and is bit-identical to the
scalar path.
RoPE `cos`/`sin` are computed once per token rather than per layer (30× fewer transcendentals).
The obvious next step if more speed is wanted is a transposed weight layout so the matmul
accumulates across output rows in vector lanes and drops the per-group reduction entirely — that is
what `bearly25-demos/borai`'s `TRANSPOSED_WEIGHTS` path does for the per-tensor-scale case.
