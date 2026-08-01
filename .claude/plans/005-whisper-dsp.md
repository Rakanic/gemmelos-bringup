# Plan 005 — Whisper `tiny.en` speech-to-text on DSP 25 (I2S mic → full free-form English transcript)

**Status:** P0 tooling validated on host (2026-07-31); C engine not started.
- `dsp25-tests/whisper-test/scripts/export_whisper.py` — Q8_0 weight export (`wh01` header). Written; not yet run against real weights.
- `dsp25-tests/whisper-test/scripts/dump_whisper_reference.py` — host golden dump. **RAN CLEAN** on
  `whisper/tests/jfk.flac`: transcript ` And so my fellow Americans ask not what`, mel 3000 frames →
  1500 enc positions, per-stage sums emitted to `whisper_reference.h`.
- **Env gotchas (documented so we don't re-hit them):** this box has **no ffmpeg** and **broken
  torchaudio/`.local` torch**. Use the repo `.venv` with **`PYTHONNOUSERSITE=1`** (else `.local`
  shadows the venv's good `torch 2.13.0+cpu` with a `torch.nn`-less broken one). Audio is loaded via
  **`soundfile`** (pip wheel bundles libsndfile — no ffmpeg) + a numpy FFT resample to 16 kHz mono;
  the loader is self-contained in `dump_whisper_reference.py::load_audio_16k_mono`.

### First real finding (from the host reference)
**The last encoder block has a large activation outlier: `enc_block3` absmax ≈ 631** vs mean ≈ 0.45
and blocks 0–2 absmax ≈ 7–10. This is a transformer outlier-feature. **Implication for P2:** a naive
per-tensor int8 activation scale in that block would be dominated by the outlier and zero out the
normal activations — the exact failure family as the tinyspeech conv2 `M2 ≈ INT32_MAX` bug. Plan for
**group / per-channel activation quant (or float fallback) in the encoder's last block** from the
start; validate that block's stage sum against the golden first.
**Scope decision:** **full free-form English** (real Whisper `tiny.en`: GPT-2 BPE tokenizer +
open-vocab greedy decode). Not a constrained command grammar.
**Audio source decision (confirmed 2026-07-31):** **VAD-gated utterance**, not a fixed 30 s window.
Capture is gated by the existing VAD onset detector; the encoder runs on the *trimmed* frame count
(≤1500), which cuts the dominant quadratic attention cost. Requires a **variable-length encoder** —
see §5 P3 and the positional-embedding slicing note.
**Chip:** DSP 25 only (single-die demo; no C2C link needed — this is a standalone `dsp-*` target,
unlike the KWS demos).

---

## 1. Verdict & framing

Doable as an **offline, transcribe-on-command** demo (not streaming / not real-time). We do NOT port
OpenAI's PyTorch Whisper; we write a small C inference engine in the exact style of the existing
`llama2.c/runq.c` int8 port, because **Whisper is just an encoder–decoder transformer** and we
already run int8 transformers with RVV kernels on this chip (`bearly25-demos/borai/int8`, TinyLlama).

### Memory — the easy axis
- Whisper `tiny.en` = 39M params. int8 group-quantized (our existing format) ≈ **~39 MB weights**.
- Largest single tensor: vocab embed + output proj = 51865 × 384 ≈ **20M params (~half the model)**.
- `platform/dsp25/dsp25-llm.ld` already provides a **2 GiB DRAM region** (`0x80000000`, LENGTH 2048M)
  with the model `.incbin`'d into `.rodata` — this is exactly how TinyLlama-1.1B (~1.1 GiB) already
  loads. 39 MB is trivial. **The "1 GB RAM" worry is a non-issue.** Build with `LINKER=llm` (see the
  TinyLlama demo).

### Compute — the real cost (set expectations)
- Encoder runs a fixed 30 s window = 1500 frames, 4 layers:
  - linear layers ≈ 10.6 G MAC, attention (1500×1500) ≈ 6.9 G MAC → **encoder ≈ ~17 G MAC / clip**.
- Decoder is autoregressive; per emitted token the **384×51865 output projection ≈ 20 M MAC
  dominates**. A ~20-token transcript ≈ 0.4 G MAC total — small next to the encoder.
- Expect **tens of seconds up to ~1 minute per utterance** on this core. Fine for a demo; not live.
- **Cheap win:** don't always feed the full 30 s. Pad/trim to the VAD-gated capture length; the
  1500² attention term shrinks quadratically with sequence length.

---

## 2. What we reuse (concrete file/symbol map)

| Whisper needs | Reuse from | Notes |
|---|---|---|
| int8 group-quant matmul + `quantize`/`dequantize` | `llama2.c/runq.c` (`matmul`, `quantize`, `QuantizedTensor`, `GS`) and the **RVV-optimized** version in `bearly25-demos/borai/int8/src/main.c` | This is the core FLOP path. Whisper's QKV/out/FFN matmuls are the same op. `dim=384`, `head_size=64` are RVV-friendly. |
| log-mel front-end (FFT + mel filterbank) | `mfcc-lib/` (`mfcc_driver_*`, NMSIS `riscv_mfcc_*`, `riscv_rfft_*`) | Reuse the FFT + mel machinery; **remove the DCT + int8 MFCC quant**. Whisper wants **log-mel**, not cepstral. See §4. |
| 16 kHz mic capture, VAD gating | `dsp-i2s-test`, `KWS_DSP_ROLLING_USE_MIC` path + VAD knobs | Same 16 kHz clock. Reuse verbatim. |
| model `.incbin` into DRAM + serial prompt/print scaffolding | TinyLlama dsp demo (`LINKER=llm`, `dsp25-llm.ld`), `simple_setup.h` | Weights + tokenizer from fixed DRAM `.rodata`. |
| softmax / layernorm / GELU / attention loop | `runq.c` `forward()` / `softmax` (note: borai renames to `llama_softmax` to avoid collision) | Kernel swaps below. |

### Architecture deltas vs our decoder-only llama port
- **RMSNorm → LayerNorm** (Whisper uses LayerNorm *with bias*).
- **SwiGLU/SiLU FFN → GELU** FFN (`w_fc` then GELU then `w_proj`, no gate).
- **RoPE → learned additive positional embeddings** (encoder: sinusoidal fixed; decoder: learned).
- **Decoder-only → encoder + decoder:** decoder has BOTH causal self-attention (KV-cached) AND
  **cross-attention** into the frozen encoder output (cross K/V computed once from encoder states).
- **Conv1d stem:** two Conv1d (kernel 3; conv1 stride 1, conv2 stride 2 → 3000→1500 frames) + GELU,
  before the encoder blocks. Only non-transformer op; do it as im2col→(reuse matmul) or a small RVV
  conv. It's tiny; don't over-optimize.

---

## 3. RVV mapping strategy

The whole model reduces to the matmul/attention kernels we already vectorize for llama. We are
**extending existing kernels to a few new shapes**, not mapping Whisper from scratch.

1. **`matmul` int8×int8→int32 + group dequant** — already vectorized in borai. 90% of FLOPs.
   Vectorize the reduction (`vsetvl` over the contraction dim, `vmacc`/`vwmacc`, reduce). Whisper
   shapes: `384×384` (QKV/out), `384×1536` & `1536×384` (FFN), `384×51865` (vocab head).
2. **Attention** `QKᵀ` (scale by 1/√64) → softmax → `·V` — reuse matmul kernel + vectorized softmax
   (max-reduce, `exp`, sum-reduce), same pattern as llama. Cross-attention: K/V come from encoder
   states (compute once, cache per layer).
3. **Conv1d stem** — im2col to reuse the matmul kernel, or a small dedicated RVV conv over 80/384
   channels. Low priority.
4. **LayerNorm / GELU** — element-wise vectorized passes; cheap. GELU: tanh approx is fine.

---

## 4. The genuinely NEW work (where the time actually goes)

1. **Weight export script** (`scripts/export_whisper.py`, mirror `llama2.c/export.py`): load
   `openai-whisper` `tiny.en`, group-quantize each matmul weight to int8 in our `QuantizedTensor`
   layout, emit a single `.bin` with a header (dims, n_layers, n_mels, n_vocab, GS) + all tensors +
   the LayerNorm/bias/positional/conv weights in fp32. **Most of the effort is here.**
2. **Log-mel front-end** (adapt `mfcc-lib`): 80 mel bins, `n_fft=400`, `hop=160`, `win=400` Hann,
   `sample_rate=16000`; **log10 of mel energies, then Whisper's normalization**
   (`log_spec = max(log_spec, log_spec.max()-8); (log_spec+4)/4`). Reuse the NMSIS rFFT + mel
   filterbank; add a Whisper mel config (the current `mfcc_specialized.h` is hard-wired to
   TinySpeech's 23 mel / 12 DCT — add an 80-mel, no-DCT specialization or a standalone whisper
   frontend .c).
3. **Tokenizer + decoding loop:** GPT-2-style BPE (~51865 tokens) + special tokens
   (`<|startoftranscript|>`, `<|en|>` — implicit for `.en`, `<|transcribe|>`, `<|notimestamps|>`),
   greedy decode until `<|endoftext|>`. Bigger/different from our SentencePiece llama tokenizer.
   Ship the vocab/merges as a `.bin` alongside weights (like `llama2.c/tokenizer.bin`).
4. **Encoder→decoder plumbing:** run encoder once, cache its output; decoder loop with self + cross
   attention and KV cache for self-attention only.

---

## 5. Staged execution (validate each stage against host `openai-whisper` on the same clip)

- **P0 — encoder-only, FLOAT, host-validated.** log-mel (adapt mfcc-lib) → conv stem → 4 encoder
  layers → dump encoder states; **diff per-layer stage sums vs host** (same discipline as
  `tinyspeech_reference.h`/`ref_stage_sums`). Proves front-end + attention before int8 or decoder.
  - **DONE (front-end): log-mel validated on host AND Spike.** `c2c-demos/dsp-whisper` P0 scaffold
    computes the Whisper log-mel in C (direct 400-pt DFT, since n_fft=400 isn't power-of-two) and
    diffs an embedded golden testvec. Host gcc: `max_abs_diff 2e-4`. **Spike** (`rv64gcv_zicntr`,
    VLEN=256, `PLATFORM=SIMS` build): `RESULT PASS max_abs_diff=0.000199`, clean exit code 0. Scalar
    DFT is ~1.0M cycles/frame (RVV target later). Encoder blocks are the remaining P0 work.
- **P1 — add greedy decoder + BPE tokenizer, FLOAT.** Produce a correct transcript on an embedded
  test clip (`.incbin` a WAV). No mic yet.
  - **DONE (float engine, host-validated): full encoder + decoder + greedy decode in `whisper.c`.**
    Single pure-C file (loader for the `wh01` Q8_0 blob, matvec-with-per-group-dequant, layernorm,
    exact-erf GELU, multi-head attn with causal/cross variants, conv1d stem, tied classifier).
    Validated on host against the PyTorch golden (trim-200 jfk testcase): **all 8 encoder stage sums
    rel ≤ 9e-4, all 8 greedy tokens exact match** (`843 523 616 5891 10162 290 523 616`), RESULT
    PASS. `dsp-whisper` now `.incbin`s the 44 MB model + golden mel and runs `whisper_run_validate`;
    builds for `PLATFORM=SIMS` (44 MB ELF). **Remaining P1:** BPE tokenizer for human-readable text
    (correctness is already proven via token-id match, so this is display-only).
  - **CONFIRMED ON SPIKE (2026-07-31):** `spike --isa=rv64gcv_zicntr dsp-whisper.elf` → all 8 stage
    sums `ok`, 8/8 greedy tokens match, `RESULT PASS`. Cost: **~12.6 B cycles** for the trim-200
    case (scalar float) — this is the motivation for the RVV phase (matvec dominates). Gotcha fixed:
    default linker heap is 1 MiB; the engine needs ~2 MiB of activation buffers → bumped to 128 MiB
    via `-Wl,--defsym=__heap_size` in the demo CMakeLists (else malloc-NULL fault spins the bare
    trap handler). `xmalloc` now fails loudly instead of hanging.
- **P2a — RVV-FLOAT matvec (DONE, Spike-validated 2026-07-31).** `whisper_rvv.S` hand-written kernel
  `whisper_matrow_q8` (int8 weight × float activation, per-group scale folded in, one f32 reduction
  per row). Behind CMake `WHISPER_USE_RVV`. **12.6 B → 2.76 B cycles (4.56×)** on the trim-200 case,
  8/8 tokens still match, stage sums unchanged. **Gotcha (documented):** GCC 13.2's vsetvl-insertion
  miscompiles the *intrinsic* widening — emits `vsext.vf4` under an e8 vtype (illegal) at every -O
  level → access fault on the first vector op. Fixed by writing the kernel in asm holding one e32,m4
  vtype across the inner loop. Remaining scalar hotspots now: attention dot-products (`mha_core`),
  conv1d, GELU/erf, softmax/exp, layernorm — candidates for the next RVV pass.
- **P2b — RVV int8 activations + RVV conv1d (DONE, Spike-validated 2026-07-31).**
  - int8 activations (`WHISPER_INT8_ACT`, kernel `whisper_matrow_i8`): correct (host + Spike, 8/8
    tokens) but only ~1% faster on Spike — matvec is already RVV so Amdahl caps it; the enc_block3
    outlier was a non-issue (matvec inputs are post-LN/GELU, bounded). Kept for real-silicon bandwidth.
  - **Profiling** (`WHISPER_PROFILE`) of the 2.76 B float-RVV run: matmul 41%, **conv 33% (scalar!)**,
    attn 14%, gelu 5%, ln 0.5%. → vectorized conv1d.
  - **RVV conv1d** (`conv1d_rvv`, f32 `vfmacc.vf` over output positions, `vlse32` for conv2 stride-2;
    idiom from `dsp25-bmarks/simple-conv-bmark`; float intrinsics = no GCC widening bug). **conv 899 M
    → 20.5 M (44×); total 2.76 B → 1.88 B (−32%)**, conv sums exact, 8/8 tokens. New floor: matmul 60%,
    attn 20%, gelu 8%. Note: DSP also has a CONV1D hardware accelerator (`perform_convolution_1D`,
    real-silicon only — Spike can't sim it) as a future offload.
- **P2 — int8 + RVV kernels.** Swap in the borai matmul; **validate per-layer** (int8 conv2 already
  bit us once on this silicon — see CLAUDE.md "int8 TinySpeech conv2 kernel". Don't trust int8
  kernels without a per-layer host diff).
- **P3a — full on-device audio->transcript DONE (Spike 2026-07-31).** `whisper_logmel_full` (front-end:
  center reflect-pad + framing + global-norm; host-validated to whisper mel at max_abs 1.7e-5) +
  `whisper_transcribe`. `DSP_WHISPER_FROM_AUDIO` build transcribes an embedded clip through the whole
  pipeline: Spike prints `transcript: " And so my fellow mirror and"`, 1.74 B cycles. Needed two fixes:
  **KV-cache decoder** (was O(max_new²·model), recomputed the whole sequence each step → 96-token
  decode on a never-ending clip = ~72 B cycles = 1.5 h Spike hang; now O(N), 1-token/step w/ per-layer
  K/V caches, still 8/8 tokens) and a **repetition guard** (truncated audio loops w/o emitting eot;
  trim the repeated cycle + stop). Also: CMake option cache is sticky — pass the full flag set (a
  stale USE_MIC=ON leaked init_test into a SIMS build → "bad syscall" under Spike).
- **P3 — I2S mic + VAD-gated variable-length encoder (committed).** Reuse the KWS VAD onset
  detector (`KWS_DSP_ROLLING_VAD_*`) to capture just the utterance. Feed the encoder `T ≤ 1500`
  mel frames (T = ceil(captured_samples/160), after the stride-2 conv → T/2 encoder positions) and
  **slice `encoder.positional_embedding[:n_pos]`** — the export ships the full 1500-row table for
  exactly this. Guard a minimum (~a few frames) and clamp to 1500. Validate that slicing matches
  host `whisper` run on the same trimmed clip (openai-whisper pads to 30 s by default, so compare
  against a host run that is *also* trimmed, or accept small edge differences). This is where the
  real-time-ish feel comes from: a 2 s command runs the encoder on ~100 positions, not 1500.

---

## 6. Risks / watch-items
- **int8 RVV matmul correctness on silicon** — proven-risky here (conv2 garbage `M2≈INT32_MAX`).
  Keep a FLOAT fallback build option, exactly like `KWS_BEARLY_ROLLING_USE_FLOAT_PIPELINE`.
- **Speed** — if the full-30s encoder is intolerably slow for a live demo, trim window aggressively
  and/or consider dropping to `tiny` encoder only for shorter clips.
- **Front-end fidelity** — Whisper is sensitive to the exact mel/log/normalization recipe; validate
  the mel spectrogram against `whisper.log_mel_spectrogram` numerically before trusting the encoder.
- **Tokenizer size** — merges table + vocab add a few hundred KB of `.rodata`; fine in the 2 GiB region.

---

## 7. Build sketch (target `dsp-whisper`, to add to `c2c-demos/CMakeLists.txt` or a new `dsp25-demos/`)
```bash
# offline embedded-clip build (P1/P2):
make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-whisper LINKER=llm \
  EXTRA_CMAKE_ARGS="-DWHISPER_USE_FLOAT_PIPELINE=ON"
# mic build (P3): add -DWHISPER_DSP_USE_MIC=ON
```
