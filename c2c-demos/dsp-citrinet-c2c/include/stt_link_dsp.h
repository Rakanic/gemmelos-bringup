#ifndef STT_LINK_DSP_H
#define STT_LINK_DSP_H

/*
 * DSP 25 side of the voice-assistant C2C link: send a Citrinet transcript to Bearly ML 25, collect
 * SmolLM's answer. Protocol and rationale in c2c-demos/common/stt_link_proto.h.
 *
 * The API is NON-BLOCKING, which is the whole point of it. An answer takes minutes, and the DSP has
 * to keep listening for the wake word throughout — both so a new question can interrupt an answer
 * nobody wants any more, and because a chip that stops listening for two minutes is not a voice
 * assistant. So publishing a prompt returns immediately and the caller drives progress by calling
 * stt_dsp_poll() from its microphone loop.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Boot the link: clear our own control block, wait for BML's boot barrier, arm the wake sources.
 * MUST be called from app_main, never app_init — it performs cross-link writes, and a cross-link
 * write into a still-booting chip kills that chip (/CLAUDE.md). Blocks until BML says it is up. */
void stt_dsp_link_init(void);

/* Hand `text` to BML as a question and return immediately with the new prompt index.
 *
 * If an earlier prompt is still unanswered it is REPLACED: the index advances and the new payload
 * is published over the old one. BML notices between tokens, abandons the answer in flight and
 * picks this one up. That is safe because it is physically the same cross-link write as the
 * self-heal re-grant that already runs while BML generates.
 *
 * Returns 0 (and does nothing) for empty text. */
uint32_t stt_dsp_publish_prompt(const char *text, uint32_t enc_frames, uint32_t audio_samples);

/* Drive the link. Call this often from the listening loop — it rate-limits itself to one spad touch
 * per STT_DSP_POLL_INTERVAL_MS, because each touch is a full cache-evict walk. It retransmits when
 * needed and prints the answer when it arrives. Returns 1 on the call where an answer was printed. */
int stt_dsp_poll(void);

/* 1 while a question is outstanding (published, not yet answered). */
int stt_dsp_busy(void);

/* The answer BML returned for the last completed prompt (NUL-terminated; "" if none). */
const char *stt_dsp_last_answer(void);

#ifdef __cplusplus
}
#endif

#endif /* STT_LINK_DSP_H */
