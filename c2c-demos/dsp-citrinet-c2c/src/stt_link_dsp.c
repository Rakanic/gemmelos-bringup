/*
 * DSP 25 side of the voice-assistant C2C link.
 *
 * Role: PRODUCER and initiator. It listens, transcribes, publishes a question, and grants the turn
 * to BML — but it does NOT then sleep waiting for the answer. An answer takes minutes, and a voice
 * assistant that stops listening for minutes is not one. So publishing returns immediately and the
 * microphone loop drives the link by calling stt_dsp_poll(); a question asked while an earlier one
 * is still being answered simply REPLACES it (see stt_dsp_publish_prompt).
 *
 * Every rule this file obeys comes from the silicon, not from taste (see /CLAUDE.md):
 *   - reads only its OWN spad (0xC0000000), and only after a full cache flush;
 *   - writes BML's spad through the cross-link address 0x1_D000_0000, repeated, via c2c_shm;
 *   - 32-bit accesses only (c2c_shm's block helpers assemble words — never memcpy a spad);
 *   - no cross-link write until BML's boot barrier has been seen.
 */

#include <stdint.h>
#include <string.h>

#include "citrinet_config.h"   /* DSP_CITRINET_LOG */
#include "c2c_shm.h"
#include "c2c_turnsync.h"
#include "stt_link_proto.h"
#include "stt_link_dsp.h"

/* The transcript buffer feeds this link directly, so a transcript longer than the payload would be
 * silently clipped on the way out. CMake keeps the two equal; this makes a divergence a build
 * error rather than a truncated question. */
_Static_assert(DSP_CITRINET_TRANSCRIPT_MAX <= (int)STT_PROMPT_MAX_BYTES,
               "the Citrinet transcript buffer is larger than the link's prompt payload");

/* Milliseconds -> core cycles. Timing here is measured with rdcycle rather than the CLINT timer
 * because this side no longer sleeps: it is reading the microphone continuously, and progress on
 * the link is made from inside that loop. (rdcycle is also the one clock on this chip that means
 * what it says — see the mtime entry in /CLAUDE.md.) */
#define STT_DSP_MS_TO_CYCLES(ms) \
  (((uint64_t)(DSP_CITRINET_TARGET_FREQ_HZ) / 1000ull) * (uint64_t)(ms))

/* How often stt_dsp_poll() is allowed to actually touch the scratchpad. The mic loop calls it far
 * more often than this; each real touch is a 256 KiB cache-evict walk (~0.8 ms), so at 200 ms this
 * costs well under 1% of the core while still noticing an answer promptly. */
#ifndef STT_DSP_POLL_INTERVAL_MS
#define STT_DSP_POLL_INTERVAL_MS 200u
#endif

/* Own spad: local reads + our own turn register. Peer spad: cross-link writes only. */
static stt_link_dsp_spad_t *const g_dsp =
    (stt_link_dsp_spad_t *)(uintptr_t)STT_LINK_DSP_SPAD_BASE;
static stt_link_bml_spad_t *const g_bml =
    (stt_link_bml_spad_t *)(uintptr_t)STT_LINK_BML_SPAD_PEER;

/* Padded, fixed-size staging buffers. The checksum covers the FULL buffer (not just the used
 * bytes), so both sides agree on exactly which bytes are being verified. */
static char     g_prompt_pad[STT_PROMPT_MAX_BYTES];
static char     g_answer[STT_ANSWER_MAX_BYTES + 1];
static uint32_t g_prompt_index;      /* last published */

/* Outstanding-question state, driven by stt_dsp_poll() from the microphone loop. */
static int      g_pending;           /* a question is published and unanswered */
static int      g_received;          /* BML has acknowledged receipt of it */
static uint64_t g_sent_cycle;        /* when it was published (for the latency report) */
static uint64_t g_last_poll_cycle;   /* rate-limits the spad touches */
static uint64_t g_last_regrant_cycle;
static uint32_t g_regrants;
static uint32_t g_enc_frames;        /* echoed on a retransmit */
static uint32_t g_audio_samples;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
}

const char *stt_dsp_last_answer(void) { return g_answer; }

/* Identity into BML's spad. Cross-link write -> only after the boot barrier. */
static void publish_identity(void) {
  c2c_remote_write_u32(&g_bml->magic, STT_LINK_MAGIC_BML);
  c2c_remote_write_u32(&g_bml->version, STT_LINK_PROTO_VERSION);
}

/* Publish the staged prompt, then flip the turn. Order is the commit discipline: payload first, the
 * turn register last, so when BML sees its turn the text is already resident in its spad. */
static void handoff_to_bml(uint32_t idx, uint32_t enc_frames, uint32_t audio_samples) {
  const uint64_t tx_cycle = rdcycle64();

  c2c_remote_write_block(g_bml->prompt_text, g_prompt_pad, (uint32_t)STT_PROMPT_MAX_BYTES);
  c2c_remote_write_u32(&g_bml->prompt_checksum,
                       c2c_checksum(g_prompt_pad, (uint32_t)STT_PROMPT_MAX_BYTES));
  c2c_remote_write_u32(&g_bml->prompt_bytes, (uint32_t)strlen(g_prompt_pad));
  c2c_remote_write_u32(&g_bml->enc_frames, enc_frames);
  c2c_remote_write_u32(&g_bml->audio_samples, audio_samples);
  c2c_remote_write_block(&g_bml->dsp_tx_cycle, &tx_cycle, sizeof(tx_cycle));
  c2c_remote_write_u32(&g_bml->prompt_index, idx);

  c2c_remote_write_u32(&g_bml->turn, C2C_TURN_BML); /* commit: BML's turn, in BML's spad */
  c2c_local_write_u32(&g_dsp->turn, C2C_TURN_BML);  /* our own spad: no longer our turn */
  c2c_wake_peer();
}

/* Boot barrier: poll our LOCAL spad for bml_ready. No cross-link write may happen before this
 * returns — a write into a chip that is still booting kills it (observed on silicon, both ways). */
static void wait_for_bml_ready(void) {
  uint32_t loops = 0u;

  /* Wipe a stale flag from a previous run: spad SRAM survives a chip-only reset, so without this a
   * second run could "see" a BML that has not booted yet. Local write — safe before the barrier. */
  c2c_local_write_u32(&g_dsp->bml_ready, 0u);

  for (;;) {
    const uint32_t ready = c2c_local_read_u32(&g_dsp->bml_ready);
    if (ready == STT_LINK_READY_MAGIC) {
      DSP_CITRINET_LOG("[stt-link] bml_ready seen after %u polls\n", (unsigned)loops);
      return;
    }
    /* Each poll is a full 256 KiB cache-evict walk, so this is roughly one line every few seconds —
     * enough to show the DSP is alive and waiting rather than hung, which on this platform look the
     * same. If it never advances, Bearly either has not been started or was started BEFORE this
     * chip and its one-shot announcement has already been missed (see the note in stt_link_bml.c). */
    if ((loops % 20000u) == 0u) {
      DSP_CITRINET_LOG("[stt-link] waiting for bml_ready (own spad 0x%08lx = 0x%08lx) polls=%u\n",
                       (unsigned long)STT_LINK_DSP_SPAD_BASE, (unsigned long)ready, (unsigned)loops);
    }
    loops++;
  }
}

void stt_dsp_link_init(void) {
  g_prompt_index = 0u;
  g_answer[0] = '\0';

  /* Local boot-clear of our OWN control block: the turn starts as BML's so a spurious wake before
   * we have anything to send is ignored, and any stale receipt/ack from a previous run is cleared
   * (otherwise ack_index could already be >= our first prompt_index and we would never wait). */
  c2c_local_write_u32(&g_dsp->turn, C2C_TURN_BML);
  c2c_local_write_u32(&g_dsp->ack_index, 0u);
  c2c_local_write_u32(&g_dsp->rx_index, 0u);
  c2c_local_write_u32(&g_dsp->answer_bytes, 0u);

  DSP_CITRINET_LOG("[stt-link] waiting for the BML boot barrier before touching the peer spad\n");
  wait_for_bml_ready();

  /* Arm MSIP + timer wake before the first wfi. mstatus.MIE stays 0: the interrupts WAKE us, they
   * are never taken as traps, so there is no handler to write. */
  c2c_arm_wake();
  publish_identity();

  DSP_CITRINET_LOG("[stt-link] link up: own=0x%08lx peer=0x%09llx prompt<=%u B answer<=%u B\n",
                   (unsigned long)STT_LINK_DSP_SPAD_BASE,
                   (unsigned long long)STT_LINK_BML_SPAD_PEER,
                   (unsigned)STT_PROMPT_MAX_BYTES, (unsigned)STT_ANSWER_MAX_BYTES);
}

#if STT_LINK_RETURN_ANSWER
/* Read the answer BML wrote into our spad. Safe to call once ack_index has advanced: the answer
 * bytes/checksum are written BEFORE ack_index, so the ack is the release. */
static void fetch_answer(void) {
  const uint32_t nbytes = c2c_local_read_u32(&g_dsp->answer_bytes);
  const uint32_t cksum = c2c_local_read_u32(&g_dsp->answer_checksum);

  g_answer[0] = '\0';
  if (nbytes == 0u) {
    return;
  }
  if (!c2c_local_read_block_verify(g_answer, g_dsp->answer_text, (uint32_t)STT_ANSWER_MAX_BYTES,
                                   cksum)) {
    /* The answer is cosmetic on this side — BML already printed the real thing on its own console —
     * so a torn read is reported, not retried into a deadlock. */
    DSP_CITRINET_LOG("[stt-link] answer verify FAILED (%u bytes, checksum 0x%08lx) — see the BML console\n",
                     (unsigned)nbytes, (unsigned long)cksum);
    g_answer[0] = '\0';
    return;
  }
  g_answer[(nbytes < (uint32_t)STT_ANSWER_MAX_BYTES) ? nbytes : (uint32_t)STT_ANSWER_MAX_BYTES] = '\0';
}
#endif

int stt_dsp_busy(void) { return g_pending; }

uint32_t stt_dsp_publish_prompt(const char *text, uint32_t enc_frames, uint32_t audio_samples) {
  if ((text == NULL) || (text[0] == '\0')) {
    return 0u;
  }

  /* Stage the text zero-padded to the full buffer: the payload and its checksum are fixed-size, so
   * a shorter prompt can never leave the tail of a longer previous one visible to the checksum. */
  memset(g_prompt_pad, 0, sizeof(g_prompt_pad));
  {
    size_t n = strlen(text);
    if (n > (size_t)(STT_PROMPT_MAX_BYTES - 1u)) {
      n = (size_t)(STT_PROMPT_MAX_BYTES - 1u);
      DSP_CITRINET_LOG("[stt-link] prompt truncated to %u bytes\n",
                       (unsigned)STT_PROMPT_MAX_BYTES - 1u);
    }
    memcpy(g_prompt_pad, text, n);
  }

  if (g_pending) {
    /* Replace the question in flight. BML sees the higher index between tokens and abandons the
     * answer it was producing; we never receive an ack for the old index and never wait for one. */
    DSP_CITRINET_LOG("[stt-link] REPLACING unanswered prompt %u with a newer question\n",
                     (unsigned)g_prompt_index);
  }

  g_prompt_index++;
  g_enc_frames = enc_frames;
  g_audio_samples = audio_samples;
  g_answer[0] = '\0';
  g_pending = 1;
  g_received = 0;
  g_regrants = 0u;
  g_sent_cycle = rdcycle64();
  g_last_poll_cycle = g_sent_cycle;
  g_last_regrant_cycle = g_sent_cycle;

  DSP_CITRINET_LOG("[stt-link] prompt %u -> BML: \"%s\"\n",
                   (unsigned)g_prompt_index, g_prompt_pad);
  handoff_to_bml(g_prompt_index, enc_frames, audio_samples);
  return g_prompt_index;
}

int stt_dsp_poll(void) {
  uint64_t now;

  if (!g_pending) {
    return 0;
  }

  now = rdcycle64();
  if ((now - g_last_poll_cycle) < STT_DSP_MS_TO_CYCLES(STT_DSP_POLL_INTERVAL_MS)) {
    return 0;   /* called from the mic loop far more often than the spad should be touched */
  }
  g_last_poll_cycle = now;

  if (c2c_local_read_u32(&g_dsp->ack_index) >= g_prompt_index) {
    const uint64_t cyc = now - g_sent_cycle;
    const uint32_t tokens = c2c_local_read_u32(&g_dsp->answer_tokens);
    g_pending = 0;
    DSP_CITRINET_LOG("[stt-link] answered prompt %u in %lu ms (%u tokens, %u re-grants)\n",
                     (unsigned)g_prompt_index,
                     (unsigned long)(cyc / (uint64_t)(DSP_CITRINET_TARGET_FREQ_HZ / 1000u)),
                     (unsigned)tokens, (unsigned)g_regrants);
#if STT_LINK_RETURN_ANSWER
    fetch_answer();
    if (g_answer[0] != '\0') {
      DSP_CITRINET_LOG("[stt-link] SmolLM: %s%s\n", g_answer,
                       c2c_local_read_u32(&g_dsp->answer_truncated) ? " ...[truncated]" : "");
    }
#endif
    return 1;
  }

  /* Retransmit cadence: brisk until BML acknowledges receipt (a dropped grant means both chips are
   * waiting on each other), then rare — every one of these is a cross-link write into a chip that
   * is busy generating, and it exists only so a dropped FINAL hand-back cannot strand the pair. */
  if (!g_received) {
    g_received = (c2c_local_read_u32(&g_dsp->rx_index) >= g_prompt_index);
    if (g_received) {
      g_last_regrant_cycle = now;
      DSP_CITRINET_LOG("[stt-link] prompt %u received by BML; answer in flight\n",
                       (unsigned)g_prompt_index);
    }
  }
  {
    const uint64_t window = g_received ? STT_DSP_MS_TO_CYCLES(STT_LINK_REGRANT_IDLE_MS)
                                       : STT_DSP_MS_TO_CYCLES(STT_LINK_REGRANT_FAST_MS);
    if ((now - g_last_regrant_cycle) >= window) {
      g_last_regrant_cycle = now;
      g_regrants++;
      DSP_CITRINET_LOG("[stt-link] re-grant prompt %u (%s) [self-heal, #%u]\n",
                       (unsigned)g_prompt_index,
                       g_received ? "received, still generating" : "no receipt yet",
                       (unsigned)g_regrants);
      handoff_to_bml(g_prompt_index, g_enc_frames, g_audio_samples);
    }
  }
  return 0;
}
