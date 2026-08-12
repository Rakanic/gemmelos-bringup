#ifndef C2C_DSP_MOONSHINE_CONFIG_H
#define C2C_DSP_MOONSHINE_CONFIG_H

#include <stdio.h>
#include "chip_config.h"

/* ------------------------------------------------------------------------------------------------
 * dsp-moonshine — Moonshine tiny speech-to-text on DSP 25 (plan 006). Standalone (no C2C link).
 * Same shape as dsp-whisper but a DIFFERENT model: Moonshine consumes RAW 16 kHz audio directly
 * through a learned conv preprocessor (NO log-mel front-end), uses RoPE + SwiGLU, and is natively
 * variable-length. Three run modes:
 *   default            : validate against the embedded golden (encoder stage sums + greedy tokens).
 *                        Spike-testable. Transcribes the embedded audio blob and diffs the reference.
 *   DSP_MOONSHINE_FROM_AUDIO : transcribe the embedded audio blob (same as default but no compare).
 *   DSP_MOONSHINE_USE_MIC    : capture live I2S mic audio (VAD-gated) and transcribe in a loop.
 *                              Real silicon only (Spike has no I2S). Build PLATFORM=CHIP.
 * See .claude/plans/006-moonshine-dsp.md.
 * ---------------------------------------------------------------------------------------------- */

#ifndef DSP_MOONSHINE_LOG_ENABLE
#define DSP_MOONSHINE_LOG_ENABLE 1
#endif
#if DSP_MOONSHINE_LOG_ENABLE
#define DSP_MOONSHINE_LOG(...) do { printf(__VA_ARGS__); } while (0)
#else
#define DSP_MOONSHINE_LOG(...) do { } while (0)
#endif

#ifndef DSP_MOONSHINE_FROM_AUDIO
#define DSP_MOONSHINE_FROM_AUDIO 0
#endif
#ifndef DSP_MOONSHINE_USE_MIC
#define DSP_MOONSHINE_USE_MIC 0
#endif

/* Max decoded tokens per utterance (moonshine decoder max_position_embeddings is 194). */
#ifndef DSP_MOONSHINE_MAX_TOKENS
#define DSP_MOONSHINE_MAX_TOKENS 96
#endif

/* Operating frequency the PLL is set to by init_test (mic build). UART divisor tracks it. */
#ifndef DSP_MOONSHINE_TARGET_FREQ_HZ
#define DSP_MOONSHINE_TARGET_FREQ_HZ 750000000ULL
#endif

/* ---- I2S mic capture (mirrors the proven dsp-whisper / dsp-kws-rolling mic path) --------------- */
#ifndef DSP_MOONSHINE_MIC_CHANNEL
#define DSP_MOONSHINE_MIC_CHANNEL 0
#endif
#ifndef DSP_MOONSHINE_MIC_SAMPLE_RATE_HZ
#define DSP_MOONSHINE_MIC_SAMPLE_RATE_HZ 16000u
#endif
#ifndef DSP_MOONSHINE_MIC_BITDEPTH
#define DSP_MOONSHINE_MIC_BITDEPTH 32u
#endif
/* Capture buffer, in samples. This is the TOTAL, and the pre-roll is copied into it before the
 * post-onset capture starts, so the speech you can actually record is
 * (DSP_MOONSHINE_MIC_NUM_SAMPLES - DSP_MOONSHINE_VAD_PREROLL_SAMPLES).
 * Sized for a full 3 s of speech: 48000 (3 s @ 16 kHz) + 3200 pre-roll (0.2 s) = 51200.
 * The VAD end-detect still trims to the actual utterance, so short prompts stay cheap — this only
 * raises the ceiling. Cost scales with length and the encoder is superlinear in it: 3 s gives
 * enc_pos=123 vs 23 for 600 ms, roughly 3.4x the total work. */
#ifndef DSP_MOONSHINE_MIC_NUM_SAMPLES
#define DSP_MOONSHINE_MIC_NUM_SAMPLES 51200u
#endif
#ifndef DSP_MOONSHINE_MIC_SAMPLE_SHIFT
#define DSP_MOONSHINE_MIC_SAMPLE_SHIFT 8            /* 24-bit sample in top of the 32-bit slot */
#endif
#ifndef DSP_MOONSHINE_MIC_FULLSCALE
#define DSP_MOONSHINE_MIC_FULLSCALE 8388608.0f      /* 2^23 */
#endif
/* VAD onset gate (short-frame AC energy) + pre-roll, identical scheme to dsp-whisper. */
#ifndef DSP_MOONSHINE_VAD_ENABLE
#define DSP_MOONSHINE_VAD_ENABLE 1
#endif
#ifndef DSP_MOONSHINE_VAD_FRAME_SAMPLES
#define DSP_MOONSHINE_VAD_FRAME_SAMPLES 320u
#endif
#ifndef DSP_MOONSHINE_VAD_PREROLL_SAMPLES
#define DSP_MOONSHINE_VAD_PREROLL_SAMPLES 3200u
#endif
#ifndef DSP_MOONSHINE_VAD_THRESHOLD
#define DSP_MOONSHINE_VAD_THRESHOLD 1.5e-3f
#endif
#ifndef DSP_MOONSHINE_VAD_LOG_EVERY
#define DSP_MOONSHINE_VAD_LOG_EVERY 25u
#endif
#ifndef DSP_MOONSHINE_VAD_END_ENABLE
#define DSP_MOONSHINE_VAD_END_ENABLE 1
#endif
#ifndef DSP_MOONSHINE_VAD_END_THRESHOLD
#define DSP_MOONSHINE_VAD_END_THRESHOLD 8.0e-4f
#endif
#ifndef DSP_MOONSHINE_VAD_HANGOVER_FRAMES
#define DSP_MOONSHINE_VAD_HANGOVER_FRAMES 40u       /* ~800 ms of silence at 320-sample frames */
#endif
#ifndef DSP_MOONSHINE_VAD_MIN_SAMPLES
#define DSP_MOONSHINE_VAD_MIN_SAMPLES 8000u         /* >= 0.5 s before end-detect can fire */
#endif

#endif /* C2C_DSP_MOONSHINE_CONFIG_H */
