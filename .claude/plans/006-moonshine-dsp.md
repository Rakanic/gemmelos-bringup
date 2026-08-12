# Plan 006 — Moonshine tiny speech-to-text on DSP 25 (raw-audio STT, offline transcribe-on-command)

**Status:** P0–P3 DONE (2026-08-04) — full float engine **host-validated AND Spike-validated (RVV)**;
the mic build (PLATFORM=CHIP) compiles/links (pending on-silicon run). The demo transcribes the jfk
3 s clip to "And so my fellow Americans." with 8/8 encoder stage sums and 7/7 greedy tokens matching
the HF golden, on host gcc AND on Spike with the RVV matvec kernel.
- Sibling of plan 005 (Whisper). Same offline transcribe-on-command shape, same Q8_0 + float
  pipeline and RVV kernels, but a **different model**: Moonshine tiny (`UsefulSensors/moonshine-tiny`).
- The demo lives at `c2c-demos/dsp-moonshine/` (mirrors `dsp-whisper/`). Host tooling in
  `dsp25-tests/moonshine-test/scripts/`.

## Why Moonshine differs from Whisper (and why the whisper.c engine was ~80% reusable)

Moonshine is an encoder–decoder transformer like Whisper, so `whisper.c` was the template. The
architecture deltas (all confirmed from `transformers.models.moonshine.modeling_moonshine`, not memory):

| | Whisper tiny.en | Moonshine tiny |
|---|---|---|
| front-end | log-mel DFT (80 mel) → 2× conv k3 | **raw 16 kHz audio** → 3× conv (`tanh(conv1 k127/s64)` → GroupNorm(1,C) → `gelu(conv2 k7/s3)` → `gelu(conv3 k3/s2)`); NO mel, NO positional embed |
| position | learned/sinusoidal additive PE | **interleaved partial RoPE** (rotary_dim 32 of head_dim 36, θ=1e4) on enc+dec self-attn |
| norms | LayerNorm **with** bias | LayerNorm **bias-free** |
| enc MLP | GELU MLP | GELU MLP (same) |
| dec MLP | GELU MLP | **SwiGLU** (`fc1`→2·ff, `silu(gate)*up`, `fc2`) |
| classifier | tied | tied |
| dims | 384, 4+4 layers, 6 heads, vocab 51865 | 288, 6+6 layers, 8 heads (head_dim 36), vocab 32768 |
| tokenizer | byte-level GPT-2 BPE | **LLaMA/SentencePiece BPE** (▁=space, byte-fallback, strip 1 leading space) |

Key simplifications vs Whisper: **no mel front-end** (the conv stem IS the front-end, on raw audio),
no attention biases at all, natively variable-length. New work: RoPE (interleaved partial), SwiGLU
decoder MLP, GroupNorm(1,C), a general valid (no-pad) conv1d, and a SentencePiece detokenizer.

**head_dim padding is a no-op here:** the model pads head_dim 36→40 with zeros for kernel alignment,
but zeros are numerically inert in QKᵀ and ·V, so the C engine uses head_dim=36, scale=1/√36.

## Files

Host tooling (`dsp25-tests/moonshine-test/scripts/`, run in repo `.venv` with `PYTHONNOUSERSITE=1`):
- `export_moonshine.py` — HF safetensors → **`ms01`** Q8_0 blob (gs=**32**, divides 288 & 1152).
  Norms/GroupNorm/fc-biases/conv stem stay fp32; every matmul + tied embed is Q8_0. 35 MB.
- `dump_moonshine_reference.py` — runs the HF model on a clip; dumps encoder per-stage
  (sum,absmax,mean) fingerprints + pure-greedy token ids → `moonshine_reference.h`, and the raw
  float32 audio → `moonshine_audio.bin`.
- `gen_moonshine_tokenizer.py` — bakes the SentencePiece detok into `moonshine_vocab.h` (id→output
  bytes: ▁→space, `<0xHH>`→raw byte, specials→empty); the C side concatenates + strips 1 lead space.
- `host_test.c` — gcc harness: reads the blob + audio, runs `ms_run_validate`. **The fast inner loop
  — validate on host before the baremetal build.**

Demo (`c2c-demos/dsp-moonshine/`):
- `src/moonshine.c` / `include/moonshine.h` — the engine (loader, conv stem, groupnorm, RoPE,
  bias-free LN, MHA self+cross, SwiGLU, KV-cache greedy decode, tied classifier, detok, validate).
- `src/moonshine_rvv.S` — the Q8_0 matvec kernel (identical to `whisper_rvv.S`, symbols renamed;
  the Q8_0 layout is the same so it drops in). Gated by `MS_USE_RVV`.
- `src/main.c` — default validate / `DSP_MOONSHINE_FROM_AUDIO` / `DSP_MOONSHINE_USE_MIC` modes.
- `model/moonshine_tiny_q80.bin`, `model/moonshine_audio.bin` — `.incbin`'d into DRAM (LINKER=llm).

## Build / run

```bash
# host validation (fastest inner loop):
gcc -O2 -I c2c-demos/dsp-moonshine/include dsp25-tests/moonshine-test/scripts/host_test.c \
    c2c-demos/dsp-moonshine/src/moonshine.c -lm -o /tmp/ms_host
/tmp/ms_host c2c-demos/dsp-moonshine/model/moonshine_tiny_q80.bin \
             c2c-demos/dsp-moonshine/model/moonshine_audio.bin

# Spike (embedded audio -> validate):
make build CHIP=dsp25 PLATFORM=SIMS TARGET=dsp-moonshine LINKER=llm EXTRA_CMAKE_ARGS="-DMOONSHINE_USE_RVV=ON"
spike --isa=rv64gcv_zicntr build/c2c-demos/dsp-moonshine/dsp-moonshine.elf

# real silicon mic (transcribe on command):
make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-moonshine LINKER=llm \
  EXTRA_CMAKE_ARGS="-DMOONSHINE_USE_RVV=ON -DDSP_MOONSHINE_USE_MIC=ON"
make tsi-run TTY=<tty> BINARY=build/c2c-demos/dsp-moonshine/dsp-moonshine.elf
```

## Validation status (against the HF transformers golden, jfk 3 s clip)

- **HOST (gcc, scalar): RESULT PASS.** All 8 encoder stage sums rel ≤ 0.85 %; greedy tokens
  `1126 577 590 10404 23035 29889 2` = **"And so my fellow Americans."** — 7/7 exact match.
- Note the encoder outlier (enc5 absmax ≈ 2328, like Whisper's enc_block3 ≈ 631) — a transformer
  outlier feature; **irrelevant to the float pipeline** (matvec inputs are post-LN/GELU, bounded).
  Flagged for the future int8-activation phase (per-tensor scale would zero the normal activations).

## Staged execution

- **P0/P1 — float engine, host-validated.** DONE (above). Single pure-C `moonshine.c`.
- **P2 — Spike + RVV. DONE.** `PLATFORM=SIMS` build, `spike --isa=rv64gcv_zicntr`: `matvec=RVV`,
  all 8 stage sums ok, 7/7 tokens, **RESULT PASS**. The Q8_0 matvec asm (`moonshine_rvv.S`) is the
  whisper kernel verbatim (renamed symbols) — the Q8_0 layout is identical, and gs=32 works because
  the kernel is VLEN/gs-agnostic (vl = min(remaining, VLMAX) per group).
  - **Bug fixed (alignment):** the header's `gs`/`theta` sit at odd offsets 57/61 (after the tied
    u8), so a direct `*(int*)` load is **misaligned and traps on this silicon** ("Access exception
    ... tohost=0x120"). Read them with `memcpy` (whisper.c already did). Misaligned scalar loads are
    a real trap here — a recurring dsp25 gotcha. Also `setvbuf(stdout, _IONBF)` so a fault doesn't
    swallow buffered output on Spike.
- **P3 — I2S mic + VAD. Build DONE (compiles/links PLATFORM=CHIP); pending on-silicon run.** Reuses
  the proven `dsp-whisper` mic capture verbatim (`DSP_MOONSHINE_USE_MIC`): VAD-gated utterance →
  `ms_transcribe` with the anti-loop guards on. Speak a phrase; it prints the transcript in a loop.
- **P4 (optional) — speed.** RVV conv1d (conv2 dominates the stem), RVV attention, dual-core matmul —
  all portable from the whisper engine. int8 activations only if needed (keep the float fallback).

## Risks / watch-items
- **int8 activations** — deferred (float-first, per the chip's int8-conv2 history). The enc5 outlier
  makes a naive per-tensor activation scale risky; keep the float pipeline as the default/fallback.
- **Front-end fidelity** — no mel to drift, but the conv stem + GroupNorm must match; host stage-sum
  `pre` matched to rel 0.0000, so the stem is exact.
- **Tokenizer** — SentencePiece detok is baked into the vocab table; verified the C detok reproduces
  the HF transcript on the golden clip.
