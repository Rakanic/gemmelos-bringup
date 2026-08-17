#ifndef C2C_DSP_CITRINET_CONFIG_H
#define C2C_DSP_CITRINET_CONFIG_H

#include <stdio.h>
#include "chip_config.h"

/* ------------------------------------------------------------------------------------------------
 * dsp-citrinet — NeMo Citrinet-256 CTC speech-to-text on DSP 25. Standalone (no C2C link).
 * Same demo shape as dsp-moonshine, different model: a non-autoregressive CTC conv encoder, so the
 * cost is a fixed function of audio length and does NOT grow with the length of the transcript.
 * Three run modes:
 *   default                 : transcribe the embedded audio blob and VALIDATE against the golden
 *                             (per-stage fingerprints + CTC token ids). Spike-testable.
 *   DSP_CITRINET_FROM_AUDIO : transcribe the embedded audio blob, no compare. Spike-testable.
 *   DSP_CITRINET_USE_MIC    : capture live I2S mic audio (VAD-gated) and transcribe in a loop.
 *                             Real silicon only (Spike has no I2S). Build PLATFORM=CHIP.
 * ---------------------------------------------------------------------------------------------- */

#ifndef DSP_CITRINET_LOG_ENABLE
#define DSP_CITRINET_LOG_ENABLE 1
#endif
#if DSP_CITRINET_LOG_ENABLE
#define DSP_CITRINET_LOG(...) do { printf(__VA_ARGS__); } while (0)
#else
#define DSP_CITRINET_LOG(...) do { } while (0)
#endif

#ifndef DSP_CITRINET_FROM_AUDIO
#define DSP_CITRINET_FROM_AUDIO 0
#endif
#ifndef DSP_CITRINET_USE_MIC
#define DSP_CITRINET_USE_MIC 0
#endif

/* C2C mode (target `dsp-citrinet-c2c`): after transcribing live mic audio, forward the transcript
 * across the chip-to-chip link to Bearly ML 25, which answers it with SmolLM2-135M-Instruct and
 * hands the answer back. Implies DSP_CITRINET_USE_MIC. The link itself lives entirely in
 * c2c-demos/dsp-citrinet-c2c/; everything it touches in this demo is behind this flag. */
#ifndef DSP_CITRINET_C2C
#define DSP_CITRINET_C2C 0
#endif
#if DSP_CITRINET_C2C && !DSP_CITRINET_USE_MIC
#error "DSP_CITRINET_C2C requires DSP_CITRINET_USE_MIC (the prompt comes from the microphone)."
#endif

/* Max CTC tokens per utterance. The encoder emits ~13 frames/s (8x stride on a 10 ms hop) and CTC
 * can emit at most one token per frame, so this is a hard bound, not a heuristic. */
#ifndef DSP_CITRINET_MAX_TOKENS
#define DSP_CITRINET_MAX_TOKENS 256
#endif

/* Detokenized transcript buffer. In C2C mode this must not exceed the link's STT_PROMPT_MAX_BYTES
 * (256) or the prompt is truncated on the way out; dsp-citrinet-c2c's CMakeLists keeps the two in
 * step and a _Static_assert there enforces it. */
#ifndef DSP_CITRINET_TRANSCRIPT_MAX
#define DSP_CITRINET_TRANSCRIPT_MAX 256
#endif

/* Operating frequency the PLL is set to by init_test (mic build). UART divisor tracks it. */
#ifndef DSP_CITRINET_TARGET_FREQ_HZ
#define DSP_CITRINET_TARGET_FREQ_HZ 750000000ULL
#endif

/* ---- I2S mic capture (mirrors the proven dsp-moonshine / dsp-whisper mic path) ------------------ */
#ifndef DSP_CITRINET_MIC_CHANNEL
#define DSP_CITRINET_MIC_CHANNEL 0
#endif
#ifndef DSP_CITRINET_MIC_SAMPLE_RATE_HZ
#define DSP_CITRINET_MIC_SAMPLE_RATE_HZ 16000u
#endif
#ifndef DSP_CITRINET_MIC_BITDEPTH
#define DSP_CITRINET_MIC_BITDEPTH 32u
#endif
/* Total capture buffer in samples; the pre-roll is copied in before post-onset capture starts, so
 * recordable speech is (NUM_SAMPLES - PREROLL). 48000 (3 s) + 3200 pre-roll (0.2 s) = 51200. */
#ifndef DSP_CITRINET_MIC_NUM_SAMPLES
#define DSP_CITRINET_MIC_NUM_SAMPLES 51200u
#endif
#ifndef DSP_CITRINET_MIC_SAMPLE_SHIFT
#define DSP_CITRINET_MIC_SAMPLE_SHIFT 8            /* 24-bit sample in top of the 32-bit slot */
#endif
#ifndef DSP_CITRINET_MIC_FULLSCALE
#define DSP_CITRINET_MIC_FULLSCALE 8388608.0f      /* 2^23 */
#endif
/* ---- Two-microphone array (optional) -----------------------------------------------------------
 * Both mics go on ONE I2S channel (the L and R slots of the same frame), NOT on two channels.
 * Each channel has its own clock generator (`clkgen_en` + its own I2S_CLKDIV), so two channels are
 * two independent, drifting time bases. Any two-mic processing worth doing — sum, difference,
 * beamforming — depends on a STABLE sample-to-sample phase relationship between the mics, and
 * independent clocks destroy exactly that. Two mics sharing one BCLK/WS are sample-locked by
 * construction.
 *
 * Wiring: both mics share BCLK, LRCLK/WS and SDIN. The L/R SEL pin picks the slot — mic A SEL->GND
 * (left), mic B SEL->VDD (right). I2S mics tri-state their output during the other mic's slot, so
 * the single shared data line is correct, not a bodge.
 *
 * No clock reconfiguration is needed: set_I2S_sample_freq already computes
 * mclk = rate * bits * 2, where the *2 IS the stereo frame. Both slots are already on the wire and
 * being clocked today — the right one is simply read and thrown away. `rx_force_left` must stay 0
 * (it is), or both slots collapse into the left FIFO.
 *
 * NOTE: the right RX FIFO has never carried real data on this silicon (the only I2S_RIGHT read in
 * the tree is a discard in dsp-i2s-test). mic_probe_right() bounds-checks it at boot and falls back
 * to mono rather than hanging — read_I2S_rx() spins forever on an empty FIFO. */
#ifndef DSP_CITRINET_MIC_STEREO
#define DSP_CITRINET_MIC_STEREO 0
#endif
/* How the two mics are combined:
 *   0 = primary (left) only — capture both, use one. The A/B control for measuring whether the
 *       second mic helps at all, with everything else held identical.
 *   1 = average. Speech is coherent across two closely-spaced mics so it adds in amplitude, while
 *       UNCORRELATED noise (mic self-noise, ADC noise) adds in power: ~3 dB SNR. It does NOT touch
 *       correlated noise — room tone, reverb, a distant talker — which is most real-room noise.
 *   2 = difference. A first-order gradient (cardioid-ish) response: rejects sound arriving equally
 *       at both mics, i.e. distant/diffuse sources, while near-field speech survives because the
 *       amplitude difference is large. Strong rejection, but it high-pass filters speech and is
 *       very sensitive to mic gain mismatch. Try it only if mode 1 disappoints.
 *   3 = aligned sum. Like 1, but the inter-mic delay is estimated per utterance by cross-
 *       correlation and compensated before summing, so the speech stays coherent when the talker
 *       is off-axis. At 16 kHz one sample = 2.1 cm of path difference, so this only does something
 *       when the mics are >~2 cm apart; below that it degenerates to mode 1. */
#ifndef DSP_CITRINET_MIC_COMBINE
#define DSP_CITRINET_MIC_COMBINE 1
#endif
/* Lag search range in samples for combine mode 3. 4 samples = 8.6 cm of path difference at 16 kHz,
 * which comfortably covers any sane desk-mic spacing. */
#ifndef DSP_CITRINET_MIC_MAX_LAG
#define DSP_CITRINET_MIC_MAX_LAG 4
#endif
/* Bounded spin used to decide whether the right slot is alive, so an unwired second mic degrades to
 * mono instead of hanging the core in read_I2S_rx's unbounded empty-wait. */
#ifndef DSP_CITRINET_MIC_PROBE_SPINS
#define DSP_CITRINET_MIC_PROBE_SPINS 2000000u
#endif

/* VAD onset gate (short-frame AC energy) + pre-roll, identical scheme to dsp-moonshine. */
#ifndef DSP_CITRINET_VAD_ENABLE
#define DSP_CITRINET_VAD_ENABLE 1
#endif
#ifndef DSP_CITRINET_VAD_FRAME_SAMPLES
#define DSP_CITRINET_VAD_FRAME_SAMPLES 320u
#endif
#ifndef DSP_CITRINET_VAD_PREROLL_SAMPLES
#define DSP_CITRINET_VAD_PREROLL_SAMPLES 3200u
#endif
#ifndef DSP_CITRINET_VAD_THRESHOLD
#define DSP_CITRINET_VAD_THRESHOLD 1.5e-3f
#endif
#ifndef DSP_CITRINET_VAD_LOG_EVERY
#define DSP_CITRINET_VAD_LOG_EVERY 25u
#endif
#ifndef DSP_CITRINET_VAD_END_ENABLE
#define DSP_CITRINET_VAD_END_ENABLE 1
#endif
/* End-of-speech gate. A frame whose AC energy is below this counts as silence; HANGOVER_FRAMES
 * consecutive silent frames end the capture. RAISING this makes the detector fire SOONER (more
 * frames read as silence) — which on its own would clip the low-energy tail of the last word
 * (unvoiced endings: "s", "th", "f" sit well below a voiced vowel). TAIL_MS below is what pays for
 * that: the kept audio extends TAIL_MS past the point energy first dropped, so the quiet tail of
 * the final word stays in the recording even though the detector already counted it as silence.
 * Tune the two together — a higher END_THRESHOLD wants a longer TAIL_MS.
 * Note this is deliberately BELOW the onset threshold (hysteresis): a frame loud enough to start a
 * capture should not be quiet enough to end one. */
#ifndef DSP_CITRINET_VAD_END_THRESHOLD
#define DSP_CITRINET_VAD_END_THRESHOLD 1.2e-3f
#endif
/* Milliseconds of audio KEPT after the last frame that was still above END_THRESHOLD. The capture
 * always runs HANGOVER_FRAMES past the end of speech (that is how it knows speech ended); this
 * controls how much of that trailing run survives the trim rather than being thrown away.
 * Must be <= the hangover (800 ms), else there is nothing to trim and the full hangover is kept. */
#ifndef DSP_CITRINET_VAD_TAIL_MS
#define DSP_CITRINET_VAD_TAIL_MS 300u
#endif
#ifndef DSP_CITRINET_VAD_HANGOVER_FRAMES
#define DSP_CITRINET_VAD_HANGOVER_FRAMES 40u       /* ~800 ms of silence at 320-sample frames */
#endif
/* TAIL_MS expressed in VAD frames — the unit the trim actually works in. 300 ms at 16 kHz with
 * 320-sample (20 ms) frames = 15 frames. */
#define DSP_CITRINET_VAD_TAIL_FRAMES                                        \
  ((DSP_CITRINET_VAD_TAIL_MS * (DSP_CITRINET_MIC_SAMPLE_RATE_HZ / 1000u)) / \
   DSP_CITRINET_VAD_FRAME_SAMPLES)
#ifndef DSP_CITRINET_VAD_MIN_SAMPLES
#define DSP_CITRINET_VAD_MIN_SAMPLES 8000u         /* >= 0.5 s before end-detect can fire */
#endif

#endif /* C2C_DSP_CITRINET_CONFIG_H */
