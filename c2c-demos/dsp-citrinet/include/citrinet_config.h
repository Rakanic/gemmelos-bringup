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

/* Max CTC tokens per utterance. The encoder emits ~13 frames/s (8x stride on a 10 ms hop) and CTC
 * can emit at most one token per frame, so this is a hard bound, not a heuristic. */
#ifndef DSP_CITRINET_MAX_TOKENS
#define DSP_CITRINET_MAX_TOKENS 256
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
#ifndef DSP_CITRINET_VAD_END_THRESHOLD
#define DSP_CITRINET_VAD_END_THRESHOLD 8.0e-4f
#endif
#ifndef DSP_CITRINET_VAD_HANGOVER_FRAMES
#define DSP_CITRINET_VAD_HANGOVER_FRAMES 40u       /* ~800 ms of silence at 320-sample frames */
#endif
#ifndef DSP_CITRINET_VAD_MIN_SAMPLES
#define DSP_CITRINET_VAD_MIN_SAMPLES 8000u         /* >= 0.5 s before end-detect can fire */
#endif

#endif /* C2C_DSP_CITRINET_CONFIG_H */
