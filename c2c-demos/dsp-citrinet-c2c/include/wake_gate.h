#ifndef WAKE_GATE_H
#define WAKE_GATE_H

/*
 * "Hey Marvin" wake-word gate for DSP 25 — the always-on front door of the voice assistant.
 *
 * Runs entirely on this chip: I2S mic -> rolling 1 s MFCC window -> TinySpeech CNN -> is that the
 * wake word? Only once it fires does the expensive part happen (Citrinet transcription, then the
 * C2C link to Bearly). Bearly is not involved and does not need to be awake.
 *
 * The model is the same TinySpeech topology the KWS demos use, retrained as a DETECTOR rather than
 * a classifier: class 0 is `marvin`, classes 1..5 are reject buckets covering the rest of the
 * Speech Commands vocabulary plus silence/room noise. See
 * dsp25-tests/tinyspeech-test/scripts/train_wakeword.py for why that shape, and why six classes.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bring up the MFCC front-end and the TinySpeech runtime. Call from app_init (local peripherals and
 * memory only — no cross-link traffic). Returns 0 on success. */
int wake_gate_init(void);

/* Listen until the wake word is heard, then return.
 *
 * Blocks reading the microphone, running one inference per hop. `poll` is invoked once per hop so
 * the caller can service the C2C link (collect an answer, retransmit) while listening — this is why
 * the DSP can keep hearing you during the minutes Bearly spends generating. The signature matches
 * stt_dsp_poll() so it can be passed directly; pass NULL for none.
 */
void wake_gate_listen(int (*poll)(void));

/* Cycles spent in the last inference, and the last window's wake margin — for tuning
 * DSP_WAKE_MARGIN from real observations rather than from the training report. */
uint64_t wake_gate_last_cycles(void);
float    wake_gate_last_margin(void);

#ifdef __cplusplus
}
#endif

#endif /* WAKE_GATE_H */
