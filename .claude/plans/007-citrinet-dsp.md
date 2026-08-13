# Plan 007 — Citrinet-256 CTC speech-to-text on DSP 25 (`c2c-demos/dsp-citrinet`)

Status: **PASSES ON SILICON** (2026-08-12) — full correct transcript, all 24 stage fingerprints match
the golden, 9/9 tokens, 8,422 ms for 3 s of audio (6.32 G cycles @ 750 MHz, RVV) ≈ 7.9 cycles/MAC.
The "flaky on silicon" phase was a **12x STACK OVERFLOW**, not the chip — see below. Remaining open
item: runs are not yet repeatable back-to-back (run 1 after a flash passes; a subsequent run can
hang at the return from `cn_model_load`).

## The silicon non-determinism was a stack overflow (found 2026-08-12)

Symptom: the same ELF gave different results on different silicon runs, while Spike passed every
time. Several red herrings were chased first (blob corruption, .text corruption, DRAM instability at
750 MHz, cache eviction) and each got its own diagnostic — all of which came back clean, because
none of them was the problem.

**Root cause.** `cn_run_validate` declared `cn_model_t m` as a LOCAL. `sizeof(cn_model_t)` is
**17,272 bytes** (it holds `blk[32]`, each 536 B), giving the function an **18,800-byte frame**:

```
cn_run_validate:
  lui  t0,0xffffb        ; t0 = -18640
  addi sp,sp,-160
  addi t0,t0,1328
  add  sp,sp,t0          ; frame = 18,800 B
```

`dsp-citrinet/CMakeLists.txt` never set `__stack_size`, so it inherited the **linker default of
4 KiB**. Peak demand was ~20 KB against a 4 KiB stack. Both sibling ports this was derived from do
set it — `dsp-moonshine/CMakeLists.txt:108` and `dsp-whisper/CMakeLists.txt:39`, each
`-Wl,--defsym=__stack_size=0x10000`. The port simply dropped that line.

**Why Spike never showed it.** The overflow ran from `0xFFFFC000` down to about `0xFFFF6F90`.
Spike maps 2 GiB of zeroed RAM at `0x80000000`, so that whole region is real, unused memory and the
overflow is harmless there — forever. On silicon it is past the 256 MiB the stock `dsp25.ld`
declares, so stores did not reliably read back. **This is the general shape to watch for on this
platform: a stack overflow here is silent and intermittent, never a fault.**

**Why it corrupted only *some* values.** Only automatics live on the stack; anything the compiler
kept in a register was fine. That asymmetry produced the single most confusing observation in the
whole investigation, recorded in `citrinet.c`: a header parse returning `n_fft=1216` /
`blank=0x80B7A400` / `max_c=0` **while the whole-blob checksum over those same bytes matched the
host exactly**. `hdr[13]` is a stack local in `cn_model_load` (called from the already-overflowed
frame, ~15 KiB out of bounds); the checksum accumulator `h` is a register. Reading a stale DRAM
pointer (`0x80B7A400`) back out of a stack slot is the fingerprint of a store that did not stick.

**Measured on silicon, and it is worse than the static analysis suggested.** With a 64 KiB stack the
high-water mark came back as **50,872 bytes (77%)** — so the original 4 KiB default was a **12x**
overflow, not the 5x implied by `cn_run_validate`'s frame alone. The extra demand does not appear in
a static `addi sp,sp,-N` scan: newlib's `%e` path allocates a multi-KiB `buf[BUF]` in `_vfprintf_r`
(sized for long-double range), and GCC's RVV spill frames are sized at runtime from `vlenb`. Stack
is now **256 KiB**. `cn_stack_report` also reports how many words in the high-water span were
actually disturbed, so a genuine deep stack (dense: Spike shows 438/444) can be told apart from a
stray write into the stack region (sparse) — they need opposite fixes.

**...and the size was only half of it. The PLACEMENT was the real problem.** Raising the stack to
256 KiB made things *worse* — the demo died instantly, on the very first write of the stack-paint
loop. `dsp25-llm.ld` derives the stack from the END of its declared 2 GiB DRAM region, but the chip's
DRAM ends at `0x8FFFFFFF`, so the stack sits ~1.9 GB past real memory in a band that is only
partially backed:

| `__stack_start` | size | behaviour on silicon |
|---|---|---|
| `0xFFFFB000` | 4 KiB | frames run off the bottom, stores do not stick — **the original flakiness** |
| `0xFFFB0000` | 64 KiB | works; reads below ~`0xFFFB3900` return garbage (which is what the bogus 50,872-byte "high-water" actually was) |
| `0xFFEC0000` | 256 KiB | **write hangs the core** |

So "make the stack bigger" walks off the bottom of the usable band. The stack must be **pinned into
real DRAM** instead — the symbols are `PROVIDE()`d, so `--defsym` overrides them. This also means the
50,872-byte reading was an artifact, not real depth; with the stack in DRAM it measures **1,696 bytes,
420/424 words disturbed (dense = genuine)**.

**Fix** (all four, so correctness does not hinge on one link flag):
1. Stack pinned into real DRAM at 256 KiB/hart:
   `--defsym=__stack_size=0x40000`, `--defsym=__stack_start=0x8FE00000`,
   `--defsym=__heap_end=0x8FE00000`. (The sibling ports' 64 KiB is not the issue — *where* it points
   is.)
2. The big objects in `cn_run_validate` (`cn_model_t`, `stats[]`, `s2[]`, `toks[]`) are now `static`.
   Frame: 18,800 -> **240 bytes**; largest frame anywhere in the image is now 1,232 B.
3. `-Wstack-usage=8192` so the next oversized frame is a build error, not a flaky chip.
4. `cn_stack_paint()` / `cn_stack_report()` — paint at boot, report the high-water mark. Measured
   **1,728 of 65,536 bytes (2%)**. Leave both calls in; this is the cheap detector the platform needs.

Also closed while here: the embedded **audio blob had no integrity check** (191 KB that fully
determines the transcript — the model blob has a checksum in its header and `.text` is hashed, but
this had nothing). It now prints an FNV-1a-64 that `text_checksum.py <elf> model/citrinet_audio.bin`
reproduces host-side.

## Why Citrinet instead of Moonshine

Measured on silicon (see the moonshine notes), Moonshine tiny spends **~170 MB of weight traffic per
inference** for a 27 MB model, because 76% of its byte budget is the decoder + tied classifier, both
re-read **once per output token** as n=1 matvecs with zero reuse. That is why tiling, RVV, int4 and
`-O3` all failed to move it off 20–60 cycles/MAC: the wall is bytes, not the kernel.

Citrinet is a **CTC conv encoder — non-autoregressive**. Every weight is read exactly once per
utterance and the cost is a fixed function of audio length, independent of transcript length.

| | Moonshine tiny | QuartzNet 15x5 | **Citrinet-256-γ0.25** |
|---|---|---|---|
| params | 27.1 M | 19.0 M | **9.87 M** |
| MACs @3 s | ~986 M | 2767 M | **804 M** |
| weight bytes moved | **~170 MB @0.6 s** | 18.9 MB | **9.7 MB** |
| largest layer working set | — | 512 KB (exceeds cache) | **160 KB (fits 256 KB)** |
| output layer | 32768 BPE, per token | 28 chars | 1025 classes, once |

QuartzNet was the initial candidate but is strictly worse: 2× the size, **3.3× the MACs** (it strides
only 2×, so 17 blocks run at full time resolution vs Citrinet's 8×), and its widest layer overflows
the cache. It is also no longer shipped in NVIDIA-NeMo/Speech — the architecture support exists
(`ConvASREncoder` + `JasperBlock`) but the configs and checkpoints were dropped; only Citrinet and
Jasper10x5dr are registered. Checkpoints come from **NGC**, not HF.

## Architecture as implemented

23 blocks, all separable + squeeze-excite, 8× total stride, 640-d output, 1025 CTC classes
(1024 SentencePiece pieces + blank).

```
block:  [ depthwise conv -> pointwise matmul + bias -> ReLU ] x R    (ReLU omitted on the last sub-block)
        -> SqueezeExcite(global mean over frames)
        -> += BN(1x1 conv(block input, stride=block stride))          ('stride_add' residual)
        -> ReLU
```
Stride lives on the **depthwise** conv, and only in the last sub-block (`stride_last: true`).
Padding is `dilation*(k-1)/2`.

Front-end: preemph 0.97 → Hann-400 centred in a 512 FFT, hop 160, `center=True` with **zero**
padding → power spectrum → slaney mel (80) → `log(x + 2^-24)` → **per-feature** mean/var norm
(ddof=1, std += 1e-5).

## Gotchas found (in order of how much time they cost)

1. **NeMo's BatchNorm uses `eps=1e-3`, not torch's 1e-5** (`jasper.py:971`). `running_var` in this
   checkpoint goes down to `5.6e-45`, so eps *sets* the BN scale on dead channels. Using 1e-5
   exploded activations ~100× per block into inf/NaN by block 16. This is the single most important
   constant in the port.
2. **SE scratch overflow.** `2*max_c + 64` floats is NOT enough: block 22 is `c_out=640,
   se_hidden=80` → needs 1360, allocated 1344. The 16-float overrun corrupted the heap and surfaced
   under Spike as `bad syscall #2159557928` (= `0x80B42BE8`, a DRAM address written into `tohost`).
   Bound is `3*max_c`.
3. Citrinet checkpoints are on **NGC**; `huggingface.co/nvidia/stt_en_citrinet_256` returns 401
   because it does not exist there (HF returns 401, not 404, for unknown repos).

## Blob format ("cn01")

256 B header → `n_blocks × 12×i32` descriptor table → Hann window → mel filterbank → per-block
weights → decoder. A matrix is stored **Q8_0 iff `in % gs == 0`**, else fp32; both the exporter and
`cn_model_load` derive that predicate independently, so no per-tensor flag is needed. The only fp32
matrix in this model is block 0's pointwise (`in = n_mels = 80`). BatchNorm is folded at export into
the preceding pointwise as a row rescale + bias, so the runtime never sees a BN.
Result: 11.82 MB, 97.1% int8, and the Q8_0 round-trip reproduces the fp32 tokens exactly.

## Host tooling (`dsp25-tests/citrinet-test/scripts/`)

- `citrinet_ref.py` — forward pass straight from `model_config.yaml` + `model_weights.ckpt`, no NeMo
  install. The executable spec.
- `export_citrinet.py` — → `model/citrinet_256_q80.bin`.
- `dump_citrinet_reference.py` — parses the **blob** the way the C loader does, runs it in numpy,
  emits `citrinet_reference.h`. Doing it from the blob makes it a test of the format and of the
  quantization, and means the golden already contains the same quantization error the C will see.
- `gen_citrinet_tokenizer.py` — → `citrinet_vocab.h` (1024 pieces, `▁` pre-converted to space).

Validated: `whisper/tests/jfk.flac` → *"and so my fellow americans asked not what your country can do
for you as what you can do for your country"*. The 3 s demo clip → *"and so my fellow americans"*,
matching Moonshine's golden on the same audio. `stt_en_citrinet_256` (non-γ) is visibly worse
("americas as"), confirming γ0.25 as the right variant.

## Build

```bash
# regenerate assets (only after changing the model)
PYTHONNOUSERSITE=1 .venv/bin/python -s dsp25-tests/citrinet-test/scripts/export_citrinet.py \
    --model build-scratch/nemo-models/stt_en_citrinet_256_gamma_0_25 \
    --out c2c-demos/dsp-citrinet/model/citrinet_256_q80.bin

make build CHIP=dsp25 PLATFORM=SIMS TARGET=dsp-citrinet EXTRA_CMAKE_ARGS="-DLINKER=llm"
spike --isa=rv64gcv_zicntr_zfh_zvl256b build/c2c-demos/dsp-citrinet/dsp-citrinet.elf
```
`LINKER=llm` is required (11.8 MB `.rodata` + a few MB of heap; stock `dsp25.ld` gives 1 MB) and the
Makefile does **not** forward `LINKER=`, so it must go inside `EXTRA_CMAKE_ARGS`.

## Performance (open work, started 2026-08-12)

Silicon baseline, RVV on, 750 MHz: **6.256 G cycles = 8,340 ms for 3 s of audio** (~2.8x slower than
real time). Against 804 M MACs that is **7.8 cycles/MAC**.

**Do not tune from the Spike profile.** Spike reports 482 M cycles for the same run — **13x fewer
than silicon** — because it does not model cache misses or vector-unit latency. Spike's numbers are
an instruction-count guide only; the silicon breakdown must come from a `CITRINET_PROFILE=ON` build
on the board.

MAC budget (computed from the descriptor table, so this part is exact):

| stage | MACs | share |
|---|---|---|
| pointwise | 647 M | 81% |
| residual 1x1 | 113 M | 14% |
| decoder | 25 M | 3% |
| depthwise | ~15 M | 2% |

Blocks 1-6 alone are ~335 M MACs (42%) and hold the largest activation working sets
(256 x 300 x 4 = 307 KB at block 1, i.e. over the 256 KB cache).

**Measured silicon profile** (`CITRINET_PROFILE=ON`, 6.262 G cycles total):

| stage | cycles | share | MACs | cycles/MAC |
|---|---|---|---|---|
| pointwise | 2,951 M | 50.5% | 647 M | 4.6 |
| **residual** | **2,019 M** | **34.5%** | 113 M | **17.9** |
| depthwise | 474 M | 8.1% | ~15 M | 31.6 |
| frontend | 147 M | 2.5% | — | — |
| se | 144 M | 2.5% | — | — |
| decoder | 111 M | 1.9% | 25 M | 4.4 |

Note how badly Spike mispredicts the *ranking*, not just the magnitude: it put depthwise at 25% (real:
8%) and the residual at 7.8% (real: 34.5%). Spike models no memory system, so any kernel whose cost
is addressing rather than arithmetic is invisible to it.

### Done (2026-08-12), both bit-identical to the golden

1. **`residual_add_rvv` was issuing `vlse32` for stride-1 blocks.** 18 of the 21 residual blocks have
   stride 1 (only 1, 7, 14 stride) = 85% of residual MACs, all walking CONTIGUOUS memory with a
   strided-load instruction. That is why the residual cost 17.9 cycles/MAC against the pointwise's
   4.6 for identical arithmetic — this chip evidently cracks `vlse32` into per-element accesses. The
   load is now chosen outside the accumulation loops (`CN_RES_ACC` macro, so the two paths cannot
   drift). **Invisible on Spike** (37.6 M -> 37.8 M) precisely because Spike has no memory model.
2. **`conv_dw` vectorized** over `t` for stride 1 (unit-stride `vle32` per tap; the k-1 edge outputs
   whose kernel window is clipped stay scalar, so the accumulation order over `j` is unchanged).
   Spike: dw 121.3 M -> **14.3 M instructions (8.5x)**.

Spike total 482.6 M -> 375.8 M. All 24 stage fingerprints byte-for-byte unchanged.

**On silicon both barely moved: 8,349 -> 8,239 ms (1.3%).** dw 474 M -> 394 M (17%, not the 8.5x its
instruction count suggested); res 2,019 M -> 2,024 M (**zero** — the vlse32 theory was simply wrong).
The negative result is the useful part: cost here is not instruction count.

### The cost model (derived from the failed predictions)

Cycles per byte of activation LOADED, for the two kernels:

| kernel | vec ops | cycles/vec-op | bytes/vec-op | cycles per byte |
|---|---|---|---|---|
| pointwise (MR=4) | 25.3 M | 117 | 25.6 | **4.57** |
| residual (MR=1) | 7.1 M | 287 | 64.0 | **4.48** |

Two kernels with different weights, strides and shapes land on the same constant. The depthwise fits
too (~60 MB loaded -> ~272 M predicted vs 394 M measured). **Activation traffic is the cost.** The
residual's 4x-worse cycles/MAC is its missing MR blocking (re-reads x once per output row, 256 times,
vs the pointwise's 64), not its load instruction.

Counter-evidence for a *pure* bandwidth model: `CITRINET_USE_RVV=OFF` runs at ~28 s vs 8.24 s, so the
vector unit is worth 3.4x. Issue rate matters as well as traffic.

### 3. Activation tiling over t (done 2026-08-12, silicon result pending)

Loop interchange in both RVV kernels: the t loop moves OUTSIDE the output-row loop, so one activation
slice (`Ci * TILE * 4` bytes) stays resident across all output rows and the weights stream past once
per tile instead. At TILE=32, Ci=Co=256: 32 KB x + ~73 KB weights + 32 KB y = ~137 KB, inside the
256 KB cache. Traffic for a T=150 sub-block: ~10 MB -> ~0.5 MB. Pure interchange, so bit-identical.
Tile width is `CITRINET_TILE_T` (CMake) for sweeping without a source edit.

**Spike calls this a 21% REGRESSION** (pw 237.8 M -> 289.2 M instructions) because tiling trades
instructions for locality and Spike models no memory system. Sharpest example yet of why Spike cannot
rank optimizations for this chip — it had the sign wrong.

**Silicon: 8,239 -> 5,118 ms (1.61x).** Sweep confirms TILE=32:

| tile | latency | pointwise | residual |
|---|---|---|---|
| none | 8,239 ms | 2,940 M | 2,024 M |
| 16 | 5,579 ms | 2,211 M | 801 M |
| **32** | **5,118 ms** | **1,870 M** | **779 M** |
| 64 | 5,138 ms | 1,854 M | 792 M |

16 is clearly worse (weight re-streaming dominates) while 32 and 64 tie within noise (cache capacity
reached), so 32 is the measured optimum, not a guess.

### 4. MR=4 blocking on the residual (done 2026-08-12, silicon result pending)

Tiling made the residual's loads HIT cache but did not remove them — it still issued one activation
load per output row. `residual_add_rvv` now shares each load across 4 rows like the pointwise
(4 m4 accumulators + 1 m4 load = 20 of 32 vector registers). Relevant because RVV-off runs 3.4x
slower, i.e. issue rate is a real cost alongside traffic. Spike: res 38.1 M -> 27.9 M instructions.
Bit-identical.

### 5. Dual-core (done 2026-08-12, silicon result pending)

**There is one vector unit PER CORE on this chip** (user-confirmed), so both halves vectorize at full
rate — this is not the shared-unit case that would have made dual-core pointless.

Uses thread-lib's hthread with the ordering from `bearly25-bmarks/rvv-matmul` and `dsp-whisper`:
```
init_test() -> hthread_init() -> hthread_issue(1, nop) -> hthread_join(1)     /* once, in app_init */
fence; hthread_issue(1, worker, &arg); worker(own half); hthread_join(1); fence   /* per kernel */
```
Splits are over **output rows** (pointwise, residual) and **channels** (depthwise) — the harts write
disjoint destination rows and only share reads, so there is no write sharing to reason about. ~320
forks per utterance; no flush anywhere.

Gotchas that cost time, all now encoded in the CMakeLists:
- **`hthread_init()` must be called** or hart 1 ignores every task forever (`__main` gates on a
  cookie that only `hthread_init` publishes) and `hthread_join` hangs. `bearly25-bmarks/rvv-matmul`
  does NOT call it. This is almost certainly the moonshine "hart 1 never picked up a task" symptom.
- **citrinet.c must not define `__main`** in dual-core builds — hthread.c owns it.
- **`thread-lib` must precede `bmark-lib` on the include path** (`BEFORE`): bmark-lib ships a
  different `hthread.h` with `N_HARTS 4` and no `hthread_init`.
- Descriptors live in BSS, not hart 0's stack (following dsp-whisper).

Spike `-p2`: hart 1 comes up, RESULT PASS, **stage fingerprints bit-identical**. Spike's dual-core
cycle counts are not a usable concurrency model — silicon decides.

**ABANDONED ON SILICON (2026-08-12). Default OFF. Do not restart without fixing the hang first.**
- `CITRINET_MC_SPLIT_PCT=50` hung. `55` ran once (4,609 -> **4,025 ms**, 1.15x) then hung on a rerun.
  So it is an intermittent fork/join race, not a split-ratio effect — the two configs differ only in
  where the row boundary falls. Not debugged; the payoff did not justify it.
- **The measurement is the valuable part.** With two INDEPENDENT vector units, the parallelised
  kernels scaled only ~1.2x (pw 1857->1554 M, dw 434->372 M, res 413->336 M). A second core with its
  own vector unit should have given ~2x if per-core issue rate were the limit. It did not, so the
  bottleneck is **shared**: the memory path. This is the third independent confirmation of the
  4.5-cycles-per-byte model, and it means **only reducing bytes moved will help**. Kernel-level
  micro-optimisation and more cores are both dead ends here.

Note for the record: moonshine's "plain cross-hart data is not coherent" finding was collected while
`__stack_start` was 0xFFFB0000, i.e. hart 0's stack was in the partially-backed region documented
above. That is a plausible alternative explanation for hart 1 reading a stale all-zero descriptor off
hart 0's stack, and it no longer applies now the stack is in real DRAM.

### 6. MR=6 register blocking in the pointwise (done 2026-08-12, silicon result pending)

Follows directly from the dual-core finding. Activation traffic in the pointwise is
`(Co/MR) * Ci * T * 4` bytes — every load re-read once per output-row block, ~9.8 MB per T=150
sub-block. Tiling moved that from DRAM to L2 but did not reduce it, and a second core did not help,
so the only lever is reuse per load. MR=6 shares each `vle32` across 6 `vfmacc`s instead of 4:
**1.5x fewer activation loads**, identical arithmetic, bit-identical results.

6 is the practical ceiling: 6 m4 accumulators + 1 m4 load = 28 of 32 vector registers. vec-nn's
7-row variant needs exactly 32 and moonshine found GCC 13 spills and mangles vsetvl at that
pressure. Verified in the emitted code: **zero vector spill/reload instructions in the binary, no
`vsetvli` inside the inner loop**, one `vle32.v` feeding six `vfmacc.vf`.

Co is 256 and 640, both ≡ 4 (mod 6), so the existing 4-row tier cleans up exactly one block and the
single-row tier never runs. Spike: pw 289.2 M -> 255.5 M instructions.

**MR=6 RESULT: SLOWER (4,609 -> 4,721 ms; pw 1,857 -> 1,937 M). Reverted, gated behind
`CITRINET_MR6=OFF`.** 1.5x fewer activation loads made it worse, which **refutes "the pointwise is
bound by activation load traffic"** once tiling is in place. Best explanation: MR=6 raises concurrent
memory streams from 9 (4 weight rows + 1 activation + 4 output rows) to 13, and a low-associativity
L1 thrashes on stream count even while total bytes fall. MR=4 is a real local optimum.

Correction to the model, since it has now mispredicted twice: this machine is **not** simply
bandwidth-bound. Tiling (locality) won 1.6x, a second vector unit won only 1.2x, and reducing load
count outright LOST. That points at per-access latency and stream/associativity behaviour rather
than raw bytes/s. Optimisations that ELIMINATE work are reliable here; ones that trade one memory
behaviour for another are not, and must be measured on silicon before being believed.

### 7. ReLU fused into the pointwise / residual store (done 2026-08-12, silicon result pending)

`relu_inplace` was a SCALAR read-modify-write pass over the whole activation buffer, run after almost
every pointwise and after every residual — ~130 extra full passes per utterance, all of it pure
memory traffic. Now folded into the vector store that was happening anyway (`cn_relu_m4`, a
`vfmax_vf` before `vse32`); the flag is loop-invariant so the branch hoists out of the t loop. Blocks
0 and 22 have no residual branch and keep the standalone pass for the block-final ReLU.

This is the "eliminate work" category rather than "trade one access pattern for another", so it
should not be able to backfire the way MR=6 did. `vfmax(x,0)` differs from `x<0?0:x` only for -0.0
and NaN, neither of which changes a stage sum or absmax — verified **bit-identical** on Spike.

### 8. Activation-tile packing — MEASURED SLOWER, reverted (`CITRINET_PACK=OFF`)

4,380 -> 4,641 ms. Packed x into [c][tw] so the inner loop reads contiguously instead of striding
600 B per input channel. Telling detail: **the DEPTHWISE also got worse (434 -> 542 M)** although
packing does not touch it — the 80 KB `g_pack` buffer evicted its working set. Adding a buffer costs
more than the access pattern it fixes.

### 9. Dequant staging to kill register spills — MEASURED SLOWER, reverted

Disassembly showed the int8 pointwise inner loop carrying **~13 stack spill/reload instructions per
4 vfmaccs** (GCC 13 out of scalar registers with 4 q + 4 s + x + 4 y pointers live) — about 40% of
the loop, including "load an int8 weight, store it to the stack, load it straight back". Staging the
dequantized weights removed **every** spill (inner loop 28 -> 15 instructions, verified in the
emitted code) and was **SLOWER: 4,380 -> 4,762 ms**, pw 1,857 -> 2,149 M. The staging buffer adds
~2.5 MB of traffic per sub-block; the spills were hitting hot L1 stack slots and were nearly free.

Also re-learned the hard way: the first version put the dequant loop **between the vsetvl and the
vfmaccs** and tripped the GCC 13.2 vsetvl miscompile documented in the moonshine notes — wrong but
plausible results, drifting slightly more each block, CTC tokens still 9/9. Only the stage
fingerprints caught it. Both findings are recorded as comments in `pointwise_rvv`.

### Optimisation scoreboard — what this machine actually rewards

| change | nature | result |
|---|---|---|
| activation tiling | cuts working set 306 -> 137 KB | **1.61x** |
| residual MR=4 | cuts loads, no new buffer | **1.11x** |
| ReLU fused into the store | removes ~130 full buffer passes | **1.05x** |
| MR=6 register blocking | fewer loads, +4 memory streams | 0.98x |
| dual-core | 2nd vector unit, shared memory path | 1.15x but HANGS |
| activation packing | +80 KB buffer | 0.94x |
| dequant staging | removes all spills, +2.5 MB/sub-block | 0.92x |

**Rule: only changes that reduce total memory traffic or working set win. Instruction count,
register spills and vector-unit utilisation are all nearly free by comparison.** Four consecutive
"restructure the access pattern" attempts lost. Do not attempt another blind kernel optimisation —
the next step needs real hardware counters (mhpmcounter: cache misses, stalls) to say what the
machine is actually waiting on.

### 10. DMA + 64 KB scratchpad staging — DEAD END, measured (2026-08-12)

`dsp25-bmarks/dma-bmarks` reports PASS, but that does NOT mean DMA->scratchpad works: its COLD/WARM
cases run a 256 KB eviction walk before the core reads the destination, and HOT_REPEAT re-sends
identical data, so a stale cached copy passes too. The numbers in its table are the DRAM->DRAM case
(64 KB in 413 K cycles = 0.158 B/cycle).

A direct probe — contiguous, `logw=6`/`inc_r=inc_w=64` (the configuration dma-bmarks and dsp-whisper
both use), read back with NO eviction, two passes with different payloads:

```
pass0: 16384 B in 7,452,316 cycles, 2048/4096 words bad — first bad word 4: got 0x00000000
pass1: 16384 B in 7,406,195 cycles, 2048/4096 words bad — first bad word 4: got 0xffffffff
```

`dst[4] == src[0]` in both passes: the first 16 bytes land, then the source restarts — exactly 50%
corruption in a regular pattern. The engine does not perform the programmed transfer into the
scratchpad. Independently disqualifying: **0.0022 B/cycle, 72x slower per byte than DRAM->DRAM**, so
even a correct transfer could not pay. A strided gather (independent `inc_r`/`inc_w`, to pack
activations) is worse still — `logw=7` is unsupported and corrupts 6464/8192 words.

**This corroborates the whisper (2026-08-04) and moonshine (2026-08-06) findings with a concrete
signature rather than "it hangs". Do not revisit without an uncached alias for 0x08000000.** Kept
behind `CITRINET_DMA_STAGE`/`CITRINET_DMA_PROBE`, both default OFF.

### 11. The chunk-loop refactor fixed the register spills for free (keep this one)

Restructuring the output-row loop into chunks (needed for staging, harmless without it) replaced four
`W->q + (o+r)*Ci` expressions with one `qbase`, which freed enough scalar registers that GCC stopped
spilling. Inner loop **33 -> 22 instructions, ~13 -> 1 stack accesses**; Spike pw **283.5 -> 174.8 M**.
Bit-identical. This is the same win section 9's staging buffer was built to get — achieved with no
buffer, i.e. in the only category that reliably wins here.

### 12. CONV1D accelerator for the depthwise — HANGS on silicon, reverted (2026-08-12)

Motivation was sound: the depthwise is the worst kernel here (31 cycles/MAC vs 2.9 for the pointwise)
purely because it re-reads each input row k times, and the engine streams each element exactly once,
structurally removing that amplification.

It is **not** parallelism, and cannot be: `CONV_INPUT_ADDR` is a fixed FIFO data port, not an address
register (`hal_conv.c: conv_stream_input_batch` writes every packet to the same address), so the core
hand-feeds the engine and is fully occupied. Amdahl ceiling was ~15% regardless, since only the
depthwise is expressible as a 1D conv — the pointwise is a 256x256 channel matmul.

Implementation (kept behind `CITRINET_CONV1D`, default OFF): prepend `pad` zeros per row to turn the
engine's left-aligned `out[t] = sum_j k[j]*in[t+j]` into our `y[t] = sum_j w[j]*x[t-pad+j]`; zero-pad
the kernel to the required 8 or 16 taps (ours are 3,5,7,9,11); one preconfigured session per
sub-block; graceful fallback to the RVV kernel on any geometry it cannot express or any
ERROR/INVALID status. **Result: hangs on silicon.** Not debugged — the ceiling did not justify it.

Note `dsp25-tests/dsp-1d-conv-tests/simple-test` is marked "OUTDATED AND WON'T COMPILE", so there is
no working reference for this engine in the tree; `hal_conv.h` doc comments were the only spec.

### Standing profile after tiling (silicon, TILE=32, total 3,474 M profiled / 3,839 M measured)

| stage | cycles | share | cycles/MAC |
|---|---|---|---|
| pointwise | 1,870 M | 53.8% | 2.9 |
| residual | 779 M | 22.4% | 6.9 |
| depthwise | 435 M | 12.5% | 29 |
| frontend | 147 M | 4.2% | — |
| se | 133 M | 3.8% | — |
| decoder | 110 M | 3.2% | 4.4 |

~365 M cycles (9.5%) fall outside every PROF region — `relu_inplace`, the residual `memcpy`, and
`stat_of` in `cn_encode`. `stat_of` is validate-only diagnostic cost that the mic/FROM_AUDIO path
does not pay.

Superseded ranking below, kept for the reasoning. The earlier Spike-derived profile
(`frontend=65.5M dw=121.3M pw=237.8M se=9.1M res=37.6M dec=11.3M`)
implies cycles/MAC of ~0.37 for the vectorized pointwise, ~0.33 residual, ~0.45 decoder — but **8.1
for the depthwise**, which is 2% of the MACs and 25% of the instructions. Candidates, in order:

1. **`conv_dw` has no RVV path at all** — it is contiguous in `t`, exactly the shape the pointwise
   kernel already exploits. Biggest ratio win available.
2. **The log-mel front-end is entirely scalar** (radix-2 FFT + an 80x257 mel projection per frame,
   300 frames) — 13.6% of Spike instructions for ~1% of the MACs. The mel projection is a matvec.
3. **`matvec` (squeeze-excite) has no RVV path.**
4. **Cache blocking for the pointwise.** The kernel re-streams the whole `x` buffer once per
   4-row output block (Co/MR = 64 times). Fine while `x` fits in 256 KB (T<=150), thrashing at
   T=300. Tiling over `t` would bound the working set — this is the one most likely to explain the
   13x Spike-to-silicon gap, and the profile should confirm before any work is done.
5. `-march=rv64gcv_zfh` implies **zvl128b** while the hardware measures **VLEN=256**
   (`cn_report_vlen()` at boot). Telling GCC `zvl256b` is free to try.

## Next

- RVV: the pointwise inner loop is already a scalar×vector accumulate over `t` in channel-major
  layout — the shape a vector kernel wants. Vectorize over `t` (contiguous), which avoids the
  reduction-and-vtype-churn problem that made moonshine's int4 kernel slower than int8.
- Then the depthwise conv (also contiguous in `t`).
- Mic mode on silicon (`-DDSP_CITRINET_USE_MIC=ON`), reusing the proven VAD capture.
