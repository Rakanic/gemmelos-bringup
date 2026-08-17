#ifndef WAKE_CONFIG_H
#define WAKE_CONFIG_H

/* ------------------------------------------------------------------------------------------------
 * "Hey Marvin" wake gate — tunables.
 *
 * The gate is dsp-kws-rolling's proven capture path with a classifier on the end: wait for an energy
 * onset (cheap), capture one ~1 s window with pre-roll, run the CNN once, and wake ONLY if it says
 * `marvin`. Anything else — a cough, a door, someone else's sentence — is classified, rejected, and
 * the gate goes straight back to listening without costing anything downstream.
 *
 * Every knob is #ifndef-guarded so CMake (or a -D) can override without editing this file.
 * The values that are NOT knobs (94 frames, 12 coefficients, 160-sample frame hop, 1 s window) are
 * fixed by the trained model and by the MFCC recipe it was trained on; changing one without
 * retraining silently feeds the CNN features that mean something else.
 * ---------------------------------------------------------------------------------------------- */

#include "mfcc_driver.h"   /* MFCC_DRIVER_FFT_LEN / _NUM_DCT */

/* ---- fixed by the model (see dsp25-tests/tinyspeech-test/scripts/train_wakeword.py) ------------- */
#define WAKE_MFCC_DIM        12u
#define WAKE_FRAMES          94u
#define WAKE_FRAME_HOP       160u   /* MFCC frame advance, samples (10 ms at 16 kHz) */
#define WAKE_SAMPLE_RATE_HZ  16000u
#define WAKE_CLASS           0      /* class 0 is `marvin`; 1..5 are reject buckets */
#define WAKE_NUM_CLASSES     6

/* Samples the 94-frame window spans. The last frame starts at 93*160 and the driver consumes a full
 * FFT length from there, so this is what must be captured before the window can be computed. */
#define WAKE_WINDOW_SAMPLES  (((WAKE_FRAMES - 1u) * WAKE_FRAME_HOP) + MFCC_DRIVER_FFT_LEN)

/* ---- onset gate ---------------------------------------------------------------------------------
 * Short-frame AC energy, the same test dsp-kws-rolling and dsp-citrinet use to trigger capture.
 * Nothing but this runs while the room is quiet, which is almost all of the time.
 *
 * Deliberately well BELOW dsp-citrinet's capture onset threshold (1.5e-3): a missed wake word is far
 * worse than an extra inference on a door closing, because a non-keyword onset costs one classify
 * and is then dropped. The idle energy is logged (DSP_WAKE_LOG_EVERY) — tune from what the chip
 * actually sees in your room. */
#ifndef DSP_WAKE_VAD_THRESHOLD
#define DSP_WAKE_VAD_THRESHOLD 2.0e-4f
#endif
#ifndef DSP_WAKE_VAD_FRAME_SAMPLES
#define DSP_WAKE_VAD_FRAME_SAMPLES 320u   /* 20 ms at 16 kHz; must be even (FIFO reads are pairs) */
#endif

/* Audio kept from BEFORE the onset. Two jobs: it stops the word's attack being clipped (energy
 * crosses the threshold part-way into the first syllable), and it puts the word roughly where it
 * sits in a Speech Commands clip — so the single look this gate takes is a well-aligned one, which
 * matters because training only jittered word position by +-100 ms. */
#ifndef DSP_WAKE_PREROLL_SAMPLES
#define DSP_WAKE_PREROLL_SAMPLES 3200u    /* 200 ms */
#endif

/* ---- detection ---------------------------------------------------------------------------------
 * The decision is `logit[marvin] - max(other logits) > MARGIN`, on the one window captured per
 * onset. Measured on the held-out test split (195 marvin vs 1280 negatives), PER WINDOW:
 *     margin > 0   recall 0.939   false accept 1.09%
 *     margin > 1   recall 0.877   false accept 0.31%
 *     margin > 2   recall 0.821   false accept 0.078%
 *     margin > 3   recall 0.718   false accept 0.00%
 *
 * 2.0 is the default because the false-accept rate here is PER ONSET, not per window of a sliding
 * detector: the CNN only ever sees audio that crossed the energy gate, so 0.078% applies to sounds
 * you actually made, not to several windows every second of silence. That is what buys the higher
 * recall.
 *
 * Raise it if the gate wakes on other words; lower it if it will not wake for you. Every onset logs
 * its margin, so tune from the console rather than from this table. */
#ifndef DSP_WAKE_MARGIN
#define DSP_WAKE_MARGIN 2.0f
#endif

/* ---- housekeeping -------------------------------------------------------------------------------
 * How often the C2C link is serviced while listening, in 20 ms monitor frames. This is what lets
 * the DSP collect an answer to an earlier question — or replace one Bearly is still generating —
 * without ever stopping listening. stt_dsp_poll() rate-limits itself as well, so this only has to
 * be often enough not to add latency. */
#ifndef DSP_WAKE_POLL_EVERY_FRAMES
#define DSP_WAKE_POLL_EVERY_FRAMES 12u    /* ~240 ms */
#endif

/* Print the idle energy every N monitor frames (0 = never). 250 frames = ~5 s. This is the number
 * to set DSP_WAKE_VAD_THRESHOLD from: watch the floor in a quiet room, then watch it while talking. */
#ifndef DSP_WAKE_LOG_EVERY
#define DSP_WAKE_LOG_EVERY 250u
#endif

/* Dump the raw logits on every Nth classified onset (0 = off). The margin alone says how confident
 * the model is; this says which class it actually preferred, which is what you want when it is
 * rejecting a word you think it should accept. */
#ifndef DSP_WAKE_DEBUG_FEATURES
#define DSP_WAKE_DEBUG_FEATURES 0u
#endif

/* ---- microphone ---------------------------------------------------------------------------------
 * Shared with the command-capture path in dsp-citrinet's main.c, so the wake window and the
 * utterance that follows it are scaled identically. The gate reads the PRIMARY mic only — array
 * combining exists to help transcription, and a detector gains nothing from it. */
#ifndef DSP_WAKE_MIC_CHANNEL
#define DSP_WAKE_MIC_CHANNEL DSP_CITRINET_MIC_CHANNEL
#endif

#endif /* WAKE_CONFIG_H */
