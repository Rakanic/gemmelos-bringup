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
 * RAISED, AND THE REFERENCE MICROPHONE CHANGED — read this before tuning it. Two independent
 * factors moved this number when the array went in:
 *
 *  1. The onset is now measured on the ARRAY REFERENCE element (E0 = ch1 left), not on
 *     DSP_CITRINET_MIC_CHANNEL's left slot (ch0 left) as before. Those are different physical mics
 *     and ch0's is **8.2 dB hotter** (measured broadside — it is the same number as
 *     DSP_WAKE_ARRAY_GAIN_DB_E2). 8.2 dB is 6.6x in energy, so the old 2.0e-4 on the loud mic is
 *     only ~3.0e-5 of like-for-like sensitivity here.
 *  2. On top of that it is deliberately raised, so quiet and distant sources never reach the
 *     classifier at all. 1.0e-4 is ~3.3x (5 dB) above the like-for-like equivalent.
 *
 * WHAT THIS GATE CAN AND CANNOT DO. It rejects things that are QUIET. It cannot by itself reject
 * things that are to the SIDE, because a loud side talker produces more energy than a quiet front
 * one — direction and loudness are not separable from one number. The gate that actually
 * discriminates direction is DSP_WAKE_ARRAY_TAU_MIN below. Use this one to set how far away the
 * system responds from, and that one to set where.
 *
 * Every onset logs its energy and every idle stretch logs the floor (DSP_WAKE_LOG_EVERY), so tune
 * from the console, not from this comment. */
#ifndef DSP_WAKE_VAD_THRESHOLD
#define DSP_WAKE_VAD_THRESHOLD 1.0e-4f
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
 * Used only when DSP_WAKE_ARRAY is 0. Shared with the command-capture path in dsp-citrinet's main.c,
 * so the wake window and the utterance that follows it are scaled identically. */
#ifndef DSP_WAKE_MIC_CHANNEL
#define DSP_WAKE_MIC_CHANNEL DSP_CITRINET_MIC_CHANNEL
#endif

/* ================================================================================================
 * ENDFIRE MICROPHONE ARRAY — directional wake gate
 *
 * Three colinear elements 2.15 cm apart, beam looking out past element 0. PHYSICAL SETUP: point
 * element 0 (ch1 left) at the user; elements 1 and 2 sit behind it along one straight line.
 *
 * WHAT IT BUYS, measured on silicon (dsp-mic-bench, 2026-08-20, 17 captures): the endfire sum is
 * +2.78 dB on axis (tight, +-0.15 dB over 6 captures) and +0.85 dB at 57 deg off. So the array
 * separates front from side by about **2 dB**. That is the physics of a 4.3 cm aperture, not a
 * limitation of this code: below ~1.3 kHz the array is smaller than a quarter wavelength and has
 * almost no directivity at all, which is exactly where most speech energy sits.
 *
 * TWO GATES, AND ONLY ONE OF THEM IS ROBUST. Both are logged on every onset so one run tunes both:
 *   - TAU_MIN gates on the measured DIRECTION. tau is a LAG RATIO, so it does not move with how loud
 *     the talker is, how far away they are, or whether the gain constants below are stale. This is
 *     the gate that actually means "in front of the array".
 *   - GAIN_DB_MIN gates on the array's ENERGY gain. It is the more intuitive knob and it is real,
 *     but it moves with all three of those things, so its default is a starting point rather than a
 *     value to trust. If the console shows front-facing speech being rejected on gain while its
 *     direction reads correct, lower it — do not lower TAU_MIN instead.
 * ============================================================================================== */

/* Master switch. 0 = the original single-microphone gate, unchanged, and none of the rest of this
 * section applies. Keep this available: it is the A/B control that says whether the array is helping
 * or whether something about it is hurting. */
#ifndef DSP_WAKE_ARRAY
#define DSP_WAKE_ARRAY 1
#endif

/* I2S channels. The physical board is E0 = ch1 left, E1 = ch1 right, E2 = ch0 left (ch0 right is
 * dead), so the primary channel here is 1 — NOT dsp-citrinet's DSP_CITRINET_MIC_CHANNEL of 0, which
 * addresses what this file calls E2. Getting these backwards points the beam behind the user and
 * produces a plausible-looking result, so check them against the ALIVE/SILENT lines at boot. */
#ifndef DSP_WAKE_ARRAY_CHANNEL
#define DSP_WAKE_ARRAY_CHANNEL 1
#endif
#ifndef DSP_WAKE_ARRAY_CHANNEL_B
#define DSP_WAKE_ARRAY_CHANNEL_B 0
#endif

/* Element spacing along the array axis, cm. 2.15 cm is not arbitrary: at 16 kHz it makes one gap
 * exactly 1.000 sample, so every endfire steering delay is an integer and needs no interpolation.
 * It is also the hard ceiling on a believable lag — a per-gap lag beyond 1.00 cannot be sound. */
#ifndef DSP_WAKE_ARRAY_SPACING_CM
#define DSP_WAKE_ARRAY_SPACING_CM 2.15f
#endif

/* Steering delay per element gap, samples. 1 matches the spacing above; anything else steers the
 * beam off the array axis, which is not what this gate wants. */
#ifndef DSP_WAKE_ARRAY_ENDFIRE_DELAY
#define DSP_WAKE_ARRAY_ENDFIRE_DELAY 1
#endif

/* Per-element sensitivity relative to element 0, dB. Measured BROADSIDE at range on the soldered
 * board across three sessions (E1 reproducible to 0.14 dB; E2 to 0.6 dB).
 *
 * MEASURE THESE WITH THE SOURCE FAR AWAY. At 4-5 cm the same calibration asks for E2 = -15.5 dB on
 * axis, -12.5 at 57 deg and -8.2 at broadside — that spread is inverse-square over 4.3 cm of path
 * difference, i.e. distance leaking into a level ratio, and the broadside value is the true one.
 * Baking in a near-field number makes the rear element dominate the sum and costs about 1 dB of
 * front/side separation. Use dsp-mic-bench with -DMICB_CAL_REPORT=ON, source broadside and >= 30 cm. */
#ifndef DSP_WAKE_ARRAY_GAIN_DB_E1
#define DSP_WAKE_ARRAY_GAIN_DB_E1 -0.29f
#endif
#ifndef DSP_WAKE_ARRAY_GAIN_DB_E2
#define DSP_WAKE_ARRAY_GAIN_DB_E2 -8.21f
#endif

/* ---- the direction gate (the one that matters) --------------------------------------------------
 * Accept only sources whose measured per-gap acoustic lag is at least this, in samples. Since one gap
 * is 1.000 sample, tau IS cos(angle-off-axis):
 *     tau 1.00 = 0 deg (on axis, front)      tau 0.50 = 60 deg
 *     tau 0.87 = 30 deg                      tau 0.00 = 90 deg (broadside)
 *     tau 0.70 = 45 deg                      tau < 0  = behind the array
 * 0.70 accepts a +-46 deg cone in front. The measured front position read tau 0.97-1.02 and the
 * measured side position read 0.49-0.58, so this sits cleanly between them.
 *
 * Widen it (lower the number) if the user has to stand too precisely; tighten it toward 0.87 for a
 * narrower cone. Set it to -1.1 to disable the direction gate entirely. */
#ifndef DSP_WAKE_ARRAY_TAU_MIN
#define DSP_WAKE_ARRAY_TAU_MIN 0.70f
#endif

/* ---- the energy gate ---------------------------------------------------------------------------
 * Minimum array gain (beamformed RMS over element-0 RMS, dB) for an onset to reach the classifier.
 * Measured: +2.78 on axis, +0.85 at 57 deg, -1.25 from behind. 1.80 sits in the gap.
 *
 * CAVEAT that decides whether this default works for you: the absolute value shifts with how well
 * the gain constants match. Those +2.78/+0.85 numbers came from a NEAR-FIELD run whose rear element
 * was ~7 dB hot, which lifts every reading; with well-matched gains at range a coherent on-axis sum
 * tends toward 0 dB and an incoherent one toward -4.8 dB. Same ~2 dB separation, different offset.
 * So if the console shows correct directions being rejected on gain, this number is the reason —
 * lower it. Set it to -100.0 to disable the energy gate and rely on direction alone. */
#ifndef DSP_WAKE_ARRAY_GAIN_DB_MIN
#define DSP_WAKE_ARRAY_GAIN_DB_MIN 1.80f
#endif

/* ---- alignment internals ------------------------------------------------------------------------
 * Lag search half-range, samples, PER ELEMENT KIND. Element 1 shares a channel with element 0, so
 * its only instrument term is a +-2 sample FIFO slip and 4 is ample — a wider search there can only
 * find an aliased correlation peak in periodic speech and report it as a lag. Element 2 crosses
 * channels and carries the stream offset, measured at 7.4-12.9 samples and re-randomised every boot,
 * so its window must comfortably contain that. */
#ifndef DSP_WAKE_ARRAY_MAX_LAG_SAME
#define DSP_WAKE_ARRAY_MAX_LAG_SAME 4
#endif
#ifndef DSP_WAKE_ARRAY_MAX_LAG_CROSS
#define DSP_WAKE_ARRAY_MAX_LAG_CROSS 24
#endif

/* Correlation quality below which a lag is not believed and the element is dropped from the sum
 * (rather than summed unaligned, which would raise the output level with no directional meaning). */
#ifndef DSP_WAKE_ARRAY_MIN_GAMMA
#define DSP_WAKE_ARRAY_MIN_GAMMA 0.70f
#endif

/* Level difference, dB, below which element 1 is treated as equally loud and the source as being
 * near broadside. This is what breaks the front/back ambiguity: at 2.15 cm, "on axis in front" and
 * "on axis behind" differ by exactly one whole sample and so are indistinguishable from the FIFO
 * slip by timing alone — the nearer mic being louder is the only thing that separates them. Too
 * small and mic mismatch reads as direction; too large and real front/back cues are ignored. */
#ifndef DSP_WAKE_ARRAY_LEVEL_DEADBAND
#define DSP_WAKE_ARRAY_LEVEL_DEADBAND 0.40f
#endif

/* Samples used for the correlation, centred in the captured window. The full window would work and
 * cost ~4x more; this is plenty of speech to correlate and keeps the alignment far below the
 * classifier's own cost. */
#ifndef DSP_WAKE_ARRAY_CORR_SAMPLES
#define DSP_WAKE_ARRAY_CORR_SAMPLES 8192u
#endif

#endif /* WAKE_CONFIG_H */
