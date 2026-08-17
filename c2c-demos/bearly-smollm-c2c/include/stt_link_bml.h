#ifndef STT_LINK_BML_H
#define STT_LINK_BML_H

/*
 * Bearly ML 25 side of the voice-assistant C2C link: receive a Citrinet transcript from DSP 25, run
 * SmolLM on it, hand the answer back. Protocol and rationale in c2c-demos/common/stt_link_proto.h.
 *
 * These five calls are all that bearly-smollm's main.c sees; its `answer()` — the whole model
 * pipeline — is untouched.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Boot the link: clear our own control block, arm the wake sources, then announce readiness to the
 * DSP (which is blocked waiting for exactly that). MUST be called from app_main and only once the
 * model and KV cache are up — the announcement is what permits the DSP to write our spad, and a
 * cross-link write into a still-booting chip kills it (/CLAUDE.md). */
void stt_bml_link_init(void);

/* Sleep (wfi, timer-paced) until the DSP grants us a NEW prompt; copy it into `buf` NUL-terminated
 * and return its prompt_index. Absorbs duplicate grants by re-acking the previous answer without
 * re-generating, and a torn payload by asking for a retransmit. Never returns 0. */
uint32_t stt_bml_wait_prompt(char *buf, int cap);

/* Answer text capture. reset() before generating; capture() is called once per emitted token piece
 * with the RAW bytes (no CRLF expansion — that is a console concern). Overflow is recorded, not
 * fatal: the full answer always reaches the BML console regardless. */
void stt_bml_capture_reset(void);
void stt_bml_capture(const uint8_t *bytes, int len);

/* Has the DSP replaced the question we are answering with a newer one?
 *
 * The DSP no longer blocks while we generate — it keeps listening for the wake word, and a new
 * command simply bumps prompt_index and re-publishes, which is physically the same cross-link write
 * as the self-heal re-grant that already happens during generation. Poll this between tokens and
 * abandon the answer when it returns 1; a token costs ~10 s, so the flush this does is free.
 *
 * On abort do NOT call stt_bml_answer_done() — the abandoned prompt is never acked (the DSP is not
 * waiting for it), and stt_bml_wait_prompt() will hand back the newer one immediately. */
int stt_bml_preempted(void);

/* Did the DSP restart since this was last asked? Latching, and cleared by the call.
 *
 * The KV cache outlives the peer: reflashing or resetting the DSP leaves this chip mid-conversation
 * with a session that no longer exists, and SmolLM keeps answering as if the earlier exchanges were
 * still relevant — which reads as the model going strange rather than as stale context. In
 * multi-turn mode the caller should reset the conversation when this returns 1. */
int stt_bml_take_restart(void);

/* Publish the captured answer and hand the turn back to the DSP. */
void stt_bml_answer_done(uint32_t prompt_index, uint32_t tokens);

#ifdef __cplusplus
}
#endif

#endif /* STT_LINK_BML_H */
