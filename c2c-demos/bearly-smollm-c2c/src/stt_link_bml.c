/*
 * Bearly ML 25 side of the voice-assistant C2C link.
 *
 * Role: CONSUMER. The DSP owns the turn while it listens and transcribes; we sleep in wfi until it
 * grants us a prompt, generate an answer, and hand the turn back with the text.
 *
 * The rules here are the silicon's, not taste (see /CLAUDE.md):
 *   - read only our OWN spad (0xD0000000), and only after a full cache flush;
 *   - write the DSP's spad through the cross-link address 0x1_C000_0000, repeated, via c2c_shm;
 *   - 32-bit accesses only (c2c_shm's block helpers assemble words — never memcpy a spad);
 *   - announce readiness only once we are fully booted, because that announcement is what lets the
 *     DSP start writing into us.
 *
 * One thing is genuinely different from the KWS consumer this is modelled on: an answer takes
 * MINUTES, not milliseconds. So a receipt (`rx_index`) is written as soon as the prompt verifies,
 * before generation starts, which lets the DSP stop retransmitting and go quiet for the duration.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "c2c_shm.h"
#include "c2c_turnsync.h"
#include "smollm_config.h"   /* SMOLLM_QUIET */
#include "stt_link_proto.h"
#include "stt_link_bml.h"

/* Two log levels, because the two kinds of link output have opposite audiences.
 *
 * STT_BML_LOG is ANOMALY output — a retransmit, a duplicate grant, a failed checksum, a producer
 * restart. It prints unconditionally, even on a quiet demo console: those lines are the difference
 * between "the link is healing itself" and "the chip has hung", which on this platform look
 * identical from the outside.
 *
 * STT_BML_BOOT_LOG is the steady-state narration (addresses, readiness, per-prompt telemetry). It
 * follows SMOLLM_QUIET, so the demo console shows the conversation and nothing else. */
#define STT_BML_LOG(...) do { printf(__VA_ARGS__); } while (0)
#if SMOLLM_QUIET
#define STT_BML_BOOT_LOG(...) do { } while (0)
#else
#define STT_BML_BOOT_LOG(...) do { printf(__VA_ARGS__); } while (0)
#endif

/* Cycles to idle before announcing readiness. The announcement is a cross-link write, and a
 * cross-link write into a chip that is still BOOTING kills that chip (observed on silicon, both
 * directions) — so this is the window in which a DSP started after us can finish coming up.
 *
 * In practice it is already covered: this chip's own boot (143 MB of weights verified, KV cache
 * allocated) takes far longer than the DSP's, so the DSP is normally waiting long before we get
 * here. The grace is the margin for the reverse order, and 750 Mcyc = ~1 s at 750 MHz.
 *
 * It does NOT make the start order free: the announcement is sent ONCE (repeatedly poking a chip
 * that might be booting is exactly the thing that kills it), so a DSP that starts more than this
 * long after us never sees it and waits forever. Start the DSP first, or together. */
#ifndef STT_BML_STARTUP_GRACE_CYCLES
#define STT_BML_STARTUP_GRACE_CYCLES 750000000ull
#endif

/* Own spad: local reads + our own turn register. Peer spad: cross-link writes only. */
static stt_link_bml_spad_t *const g_bml =
    (stt_link_bml_spad_t *)(uintptr_t)STT_LINK_BML_SPAD_BASE;
static stt_link_dsp_spad_t *const g_dsp =
    (stt_link_dsp_spad_t *)(uintptr_t)STT_LINK_DSP_SPAD_PEER;

static char     g_prompt_pad[STT_PROMPT_MAX_BYTES];   /* verified copy of the received payload */
static char     g_answer_pad[STT_ANSWER_MAX_BYTES];   /* padded staging buffer, checksummed whole */
static uint32_t g_answer_used;
static uint32_t g_answer_truncated;
static uint32_t g_answer_tokens;
static uint32_t g_last_consumed;
static int      g_restarted;    /* the DSP restarted: the conversation so far is from a dead session */

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
}

/* ---- answer capture ---------------------------------------------------------------------------- */

void stt_bml_capture_reset(void) {
  memset(g_answer_pad, 0, sizeof(g_answer_pad));
  g_answer_used = 0u;
  g_answer_truncated = 0u;
  g_answer_tokens = 0u;
}

void stt_bml_capture(const uint8_t *bytes, int len) {
  /* One byte of headroom is reserved so the payload is always NUL-terminated inside the buffer,
   * which lets the DSP treat it as a C string without trusting the length field. */
  const uint32_t cap = (uint32_t)STT_ANSWER_MAX_BYTES - 1u;
  for (int i = 0; i < len; ++i) {
    if (g_answer_used >= cap) {
      g_answer_truncated = 1u;
      return;
    }
    g_answer_pad[g_answer_used++] = (char)bytes[i];
  }
}

/* ---- link ------------------------------------------------------------------------------------- */

/* Publish the answer and flip the turn. Commit discipline: payload, then bytes/checksum, then
 * ack_index, then the turn register — so the DSP never sees an ack for a half-written answer. */
static void handoff_to_dsp(uint32_t ack_idx) {
  const uint64_t rx_cycle = rdcycle64();

#if STT_LINK_RETURN_ANSWER
  c2c_remote_write_block(g_dsp->answer_text, g_answer_pad, (uint32_t)STT_ANSWER_MAX_BYTES);
  c2c_remote_write_u32(&g_dsp->answer_checksum,
                       c2c_checksum(g_answer_pad, (uint32_t)STT_ANSWER_MAX_BYTES));
  c2c_remote_write_u32(&g_dsp->answer_bytes, g_answer_used);
  c2c_remote_write_u32(&g_dsp->answer_truncated, g_answer_truncated);
#else
  c2c_remote_write_u32(&g_dsp->answer_bytes, 0u);
#endif
  c2c_remote_write_u32(&g_dsp->answer_tokens, g_answer_tokens);
  c2c_remote_write_block(&g_dsp->bml_rx_cycle, &rx_cycle, sizeof(rx_cycle));
  c2c_remote_write_u32(&g_dsp->ack_index, ack_idx);

  c2c_remote_write_u32(&g_dsp->turn, C2C_TURN_DSP); /* commit: DSP's turn, in the DSP's spad */
  c2c_local_write_u32(&g_bml->turn, C2C_TURN_DSP);  /* our own spad: no longer our turn */
  c2c_wake_peer();
}

void stt_bml_link_init(void) {
  g_last_consumed = 0u;
  stt_bml_capture_reset();

  /* Local boot-clear of our OWN control block BEFORE announcing readiness, so it lands before the
   * DSP (which is blocked on the announcement) can write anything into us. This is also what
   * defeats stale spad SRAM from a previous run — the scratchpads survive a chip-only reset. */
  c2c_local_write_u32(&g_bml->turn, C2C_TURN_DSP);
  c2c_local_write_u32(&g_bml->prompt_index, 0u);

  /* Arm MSIP + timer wake before the first wfi. mstatus.MIE stays 0: the interrupts WAKE us, they
   * are never taken as traps, so there is no handler to write. */
  c2c_arm_wake();

  {
    const uint64_t t0 = rdcycle64();
    while ((rdcycle64() - t0) < (uint64_t)STT_BML_STARTUP_GRACE_CYCLES) {
      __asm__ volatile("nop");
    }
  }

  STT_BML_BOOT_LOG("[stt-link] announcing bml_ready -> DSP spad (own=0x%08lx peer=0x%09llx)\n",
                   (unsigned long)STT_LINK_BML_SPAD_BASE,
                   (unsigned long long)STT_LINK_DSP_SPAD_PEER);
  c2c_remote_write_u32(&g_dsp->magic, STT_LINK_MAGIC_DSP);
  c2c_remote_write_u32(&g_dsp->ack_index, 0u);
  c2c_remote_write_u32(&g_dsp->rx_index, 0u);
  c2c_remote_write_u32(&g_dsp->bml_ready, STT_LINK_READY_MAGIC);
  STT_BML_BOOT_LOG("[stt-link] link up — waiting for the DSP to hear something\n");
}

uint32_t stt_bml_wait_prompt(char *buf, int cap) {
  for (;;) {
    uint32_t turn;
    uint32_t idx;

    c2c_full_flush();
    turn = g_bml->turn;
    idx = g_bml->prompt_index;

    if (turn == C2C_TURN_BML) {
      /* prompt_index going BACKWARDS means the DSP restarted (its counter begins at 1 again) while
       * we kept running. Without this the new prompt 1 would look like a duplicate of a prompt we
       * already answered and the pair would re-ack each other forever, which reads as a dead link.
       * Only a genuine decrease triggers it — a stale read can show the value we last saw (handled
       * as a duplicate below), never an older one, because every read is flush-first. */
      if ((idx != 0u) && (idx < g_last_consumed)) {
        STT_BML_LOG("[stt-link] prompt index went %u -> %u: the DSP restarted; resyncing\n",
                    (unsigned)g_last_consumed, (unsigned)idx);
        g_last_consumed = 0u;
        g_restarted = 1;
      }

      if (idx > g_last_consumed) {
        const uint32_t cksum = g_bml->prompt_checksum;
        uint32_t nbytes = g_bml->prompt_bytes;

        if (!c2c_local_read_block_verify(g_prompt_pad, g_bml->prompt_text,
                                         (uint32_t)STT_PROMPT_MAX_BYTES, cksum)) {
          /* Torn payload after retries. Hand back the PREVIOUS index so the DSP sees no ack for the
           * prompt it sent and retransmits it — the same self-heal the KWS demo relies on. */
          STT_BML_LOG("[stt-link] prompt %u failed verify (checksum 0x%08lx) — requesting a retransmit\n",
                      (unsigned)idx, (unsigned long)cksum);
          handoff_to_dsp(g_last_consumed);
          continue;
        }

        if (nbytes >= (uint32_t)STT_PROMPT_MAX_BYTES) {
          nbytes = (uint32_t)STT_PROMPT_MAX_BYTES - 1u;
        }
        g_prompt_pad[nbytes] = '\0';
        if (cap > 0) {
          int n = (int)strlen(g_prompt_pad);
          if (n > cap - 1) {
            n = cap - 1;
          }
          memcpy(buf, g_prompt_pad, (size_t)n);
          buf[n] = '\0';
        }
        g_last_consumed = idx;

        /* RECEIPT — the reason this protocol differs from KWS. Written without flipping the turn,
         * so the DSP can stop retransmitting and leave the link quiet for the minutes of decode
         * that follow, while we still hold the turn. */
        c2c_remote_write_u32(&g_dsp->rx_index, idx);

        STT_BML_BOOT_LOG("[stt-link] prompt %u received (%u bytes): \"%s\"\n",
                         (unsigned)idx, (unsigned)nbytes, g_prompt_pad);
        return idx;
      }

      /* Duplicate grant: our previous hand-back was dropped. Re-ack (which re-sends the answer
       * still staged in g_answer_pad) and do NOT generate again. */
      STT_BML_LOG("[stt-link] duplicate grant for prompt %u (already answered) — re-acking\n",
                  (unsigned)idx);
      handoff_to_dsp(g_last_consumed);
    }

    /* Idle cadence, not the turnsync default. We are waiting for a human to say something, which
     * can be minutes, and the DSP's grant MSIP wakes us instantly regardless — the timer only
     * bounds how late we notice a grant whose wake was dropped. At the default ~3.3 ms this loop
     * would spend the entire wait doing 256 KiB cache-evict walks (one per c2c_full_flush, two per
     * iteration) instead of sleeping. Same reasoning as the DSP's idle phase; see stt_link_proto.h. */
    c2c_sleep_ticks(STT_LINK_MS_TO_TICKS(STT_LINK_POLL_IDLE_MS, SMOLLM_TARGET_FREQUENCY_HZ));
  }
}

int stt_bml_preempted(void) {
  /* g_last_consumed is the prompt we are answering, so anything larger is a replacement. A flush is
   * ~0.6 Mcyc against a ~7.5 Gcyc token, i.e. under 0.01% — cheap enough to check every token and
   * every prefill pass. Monotonic, so a stale read can only delay noticing, never invent one. */
  return (c2c_local_read_u32(&g_bml->prompt_index) > g_last_consumed);
}

int stt_bml_take_restart(void) {
  const int r = g_restarted;
  g_restarted = 0;
  return r;
}

void stt_bml_answer_done(uint32_t prompt_index, uint32_t tokens) {
  g_answer_tokens = tokens;
  STT_BML_BOOT_LOG("[stt-link] answered prompt %u: %u tokens, %u bytes returned%s\n",
                   (unsigned)prompt_index, (unsigned)tokens, (unsigned)g_answer_used,
                   g_answer_truncated ? " (truncated for the link)" : "");
  handoff_to_dsp(prompt_index);
}
