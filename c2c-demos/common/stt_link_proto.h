#ifndef C2C_STT_LINK_PROTO_H
#define C2C_STT_LINK_PROTO_H

/*
 * stt_link_proto — two-scratchpad protocol for the voice-assistant C2C demo:
 *
 *     [DSP 25]  I2S mic -> Citrinet-256 CTC -> transcript TEXT
 *          |  C2C link (this protocol)
 *          v
 *     [Bearly ML 25]  SmolLM2-135M-Instruct prefill + decode -> answer TEXT
 *          |  C2C link (answer returned so the DSP console shows the round trip)
 *          v
 *     [DSP 25]  prints the answer next to the transcript
 *
 * It is the KWS streaming protocol (common/kws_stream_proto.h) with a TEXT payload instead of an
 * MFCC map, and one addition forced by the difference in workload: SmolLM takes MINUTES per answer,
 * where TinySpeech took milliseconds. See "Re-grant policy" below.
 *
 * Synchronization is unchanged — the turn-taking pattern proven on silicon (see /CLAUDE.md
 * "Reliable C2C turn-taking synchronization" and common/c2c_turnsync.h):
 *   1. a TURN REGISTER at offset 0x20 of each spad says whose turn it is (correctness),
 *   2. a CLINT timer wakes a sleeper periodically so a dropped MSIP costs latency, not liveness,
 *   3. every cross-link store goes through c2c_shm (repeated + flushed) (delivery).
 *
 * Hardware access rule (see /CLAUDE.md): a chip may READ only its own adjacent spad and may WRITE to
 * both. To WRITE the peer's spad use the cross-link address (peer-local address with a leading 1).
 *
 *   0xC0000000    DSP-adjacent spad   -> DSP reads locally; BML writes it at 0x1_C000_0000
 *   0xD0000000    BML-adjacent spad   -> BML reads locally; DSP writes it at 0x1_D000_0000
 *
 * ---- Re-grant policy (the one real difference from KWS) ----------------------------------------
 * KWS's producer re-grants the outstanding case on EVERY idle timer tick (~every few ms) until it is
 * acked. That is correct there because an inference is ~16 Mcyc. Here an answer is minutes of
 * weight-bandwidth-bound decode, and re-granting every few milliseconds for minutes would hammer a
 * busy chip with cross-link writes — exactly the "keep the link quiet" rule in /CLAUDE.md.
 *
 * So the ack is split in two:
 *   - `rx_index`  — a RECEIPT. BML writes it as soon as it has read and checksum-verified the
 *                   prompt, WITHOUT giving up the turn. It closes the window in which a dropped
 *                   grant needs retransmitting.
 *   - `ack_index` — the answer is finished and the turn is being handed back.
 * The DSP re-grants briskly (STT_LINK_REGRANT_TICKS_PRE_RX) while rx_index < n, and only rarely
 * (STT_LINK_REGRANT_TICKS_POST_RX) once the receipt has landed — the rare poll is still needed
 * because a dropped final hand-back would otherwise leave both chips asleep forever. A duplicate
 * grant is idempotent: BML re-acks (re-sending the answer already in its buffer) without
 * re-generating, exactly as KWS re-acks without re-inferring.
 */

#include <stdint.h>
#include <stddef.h>   /* offsetof */

#ifdef __cplusplus
extern "C" {
#endif

#define STT_LINK_PROTO_VERSION 1u

/* Local scratchpad bases (used for LOCAL reads / local turn writes). */
#ifndef STT_LINK_DSP_SPAD_BASE
#define STT_LINK_DSP_SPAD_BASE 0xC0000000UL
#endif
#ifndef STT_LINK_BML_SPAD_BASE
#define STT_LINK_BML_SPAD_BASE 0xD0000000UL
#endif

/* Cross-link peer bases (used for REMOTE writes into the other chip's spad — leading 1). */
#ifndef STT_LINK_DSP_SPAD_PEER
#define STT_LINK_DSP_SPAD_PEER 0x1C0000000ULL /* BML writes DSP's spad here */
#endif
#ifndef STT_LINK_BML_SPAD_PEER
#define STT_LINK_BML_SPAD_PEER 0x1D0000000ULL /* DSP writes BML's spad here */
#endif

#define STT_LINK_MAGIC_BML 0x53545444u /* 'STTD' — BML spad (DSP -> BML, the prompt) */
#define STT_LINK_MAGIC_DSP 0x53545443u /* 'STTC' — DSP spad (BML -> DSP, the answer) */

/* Boot barrier: BML writes this into the DSP spad once it is fully booted (model loaded, heap
 * allocated). The DSP must NOT write BML's spad until it sees this — a cross-link write into a
 * still-booting chip kills that chip (see /CLAUDE.md). BML's boot is long here (~143 MB of weights
 * already in DRAM, but the KV cache allocation and the memory probe still take seconds), so the DSP
 * will genuinely wait. */
#define STT_LINK_READY_MAGIC 0x52454459u /* 'REDY' */

/* Payload sizes. Both are 4-byte multiples because the spads are 32-bit-access-only, and both are
 * written IN FULL (zero-padded) every time so the checksum covers a fixed-size buffer.
 *   PROMPT — a Citrinet transcript of a few seconds of speech; ~40 characters in practice.
 *   ANSWER — SMOLLM_MAX_NEW_TOKENS (200) tokens of English, ~4 bytes/token. Truncated, not
 *            corrupted, if the model runs longer: the BML console always has the full text. */
#ifndef STT_PROMPT_MAX_BYTES
#define STT_PROMPT_MAX_BYTES 256u
#endif
#ifndef STT_ANSWER_MAX_BYTES
#define STT_ANSWER_MAX_BYTES 1024u
#endif

_Static_assert((STT_PROMPT_MAX_BYTES % 4u) == 0u, "prompt buffer must be a 4-byte multiple");
_Static_assert((STT_ANSWER_MAX_BYTES % 4u) == 0u, "answer buffer must be a 4-byte multiple");

/* BML-adjacent spad @ 0xD0000000 : DSP -> BML prompt path (DSP remote-writes, BML local-reads). */
typedef struct __attribute__((packed)) {
  volatile uint32_t magic;            /* 0x00  STT_LINK_MAGIC_BML */
  volatile uint32_t version;          /* 0x04  STT_LINK_PROTO_VERSION */
  volatile uint32_t prompt_bytes;     /* 0x08  used length of prompt_text (excludes the NUL pad) */
  volatile uint32_t prompt_checksum;  /* 0x0C  c2c_checksum over the FULL padded prompt_text */
  volatile uint32_t prompt_index;     /* 0x10  ++ per utterance; monotonic, guards double-answering */
  volatile uint32_t enc_frames;       /* 0x14  Citrinet encoder frames (telemetry) */
  volatile uint64_t dsp_tx_cycle;     /* 0x18  rdcycle at publish */
  volatile uint32_t turn;             /* 0x20  C2C_TURN_*: whose turn it is (the commit) */
  volatile uint32_t audio_samples;    /* 0x24  captured mic samples (telemetry) */
  volatile uint32_t reserved0[6];     /* 0x28..0x3F */
  volatile char     prompt_text[STT_PROMPT_MAX_BYTES]; /* 0x40 */
} stt_link_bml_spad_t;

/* DSP-adjacent spad @ 0xC0000000 : BML -> DSP ack + answer path (BML remote-writes, DSP local-reads). */
typedef struct __attribute__((packed)) {
  volatile uint32_t magic;            /* 0x00  STT_LINK_MAGIC_DSP */
  volatile uint32_t ack_index;        /* 0x04  last prompt_index fully ANSWERED (turn handed back) */
  volatile uint32_t rx_index;         /* 0x08  last prompt_index RECEIVED+verified (receipt only) */
  volatile uint32_t answer_bytes;     /* 0x0C  used length of answer_text */
  volatile uint32_t bml_ready;        /* 0x10  STT_LINK_READY_MAGIC once BML has booted */
  volatile uint32_t answer_checksum;  /* 0x14  c2c_checksum over the FULL padded answer_text */
  volatile uint64_t bml_rx_cycle;     /* 0x18  rdcycle at ack */
  volatile uint32_t turn;             /* 0x20  C2C_TURN_*: whose turn it is (the commit) */
  volatile uint32_t answer_tokens;    /* 0x24  tokens generated (telemetry) */
  volatile uint32_t answer_truncated; /* 0x28  1 = the console has more text than fitted here */
  volatile uint32_t reserved0[5];     /* 0x2C..0x3F */
  volatile char     answer_text[STT_ANSWER_MAX_BYTES]; /* 0x40 */
} stt_link_dsp_spad_t;

/* The offsets are load-bearing: the two chips are separately compiled binaries that agree only by
 * way of this header, and `turn` in particular must sit at 0x20 in BOTH spads (the turn-taking
 * layer is shared with the KWS demos, which use the same offset). */
_Static_assert(offsetof(stt_link_bml_spad_t, prompt_index) == 0x10u, "prompt_index must be at 0x10");
_Static_assert(offsetof(stt_link_bml_spad_t, turn) == 0x20u, "bml turn register must be at 0x20");
_Static_assert(offsetof(stt_link_bml_spad_t, prompt_text) == 0x40u, "prompt_text must be at 0x40");
_Static_assert(sizeof(stt_link_bml_spad_t) == (0x40u + STT_PROMPT_MAX_BYTES),
               "stt_link_bml_spad_t layout drifted from the documented offsets");
_Static_assert(offsetof(stt_link_dsp_spad_t, ack_index) == 0x04u, "ack_index must be at 0x04");
_Static_assert(offsetof(stt_link_dsp_spad_t, rx_index) == 0x08u, "rx_index must be at 0x08");
_Static_assert(offsetof(stt_link_dsp_spad_t, bml_ready) == 0x10u, "bml_ready must be at 0x10");
_Static_assert(offsetof(stt_link_dsp_spad_t, turn) == 0x20u, "dsp turn register must be at 0x20");
_Static_assert(offsetof(stt_link_dsp_spad_t, answer_text) == 0x40u, "answer_text must be at 0x40");
_Static_assert(sizeof(stt_link_dsp_spad_t) == (0x40u + STT_ANSWER_MAX_BYTES),
               "stt_link_dsp_spad_t layout drifted from the documented offsets");

/* Both structs must fit inside a 16 KiB scratchpad. */
_Static_assert(sizeof(stt_link_bml_spad_t) <= 16384u, "prompt block overflows the 16 KiB spad");
_Static_assert(sizeof(stt_link_dsp_spad_t) <= 16384u, "answer block overflows the 16 KiB spad");

/* ---- Timing tunables, in MILLISECONDS ----------------------------------------------------------
 * Expressed in ms and converted per-chip with STT_LINK_MS_TO_TICKS below, rather than as raw tick
 * counts, because CLINT mtime on this silicon is derived from the CORE clock (mtime_hz = core/1000
 * — see the mtime entry in /CLAUDE.md). A tick count therefore means a different wall-clock time at
 * every PLL setting: c2c_turnsync's C2C_POLL_INTERVAL_TICKS is "~50 ms" at the nominal 50 MHz but
 * ~3.3 ms at the 750 MHz these demos actually run at. Milliseconds mean the same thing at 500 MHz
 * and 750 MHz, which matters here because dropping both chips to 500 is the standing first move if
 * the link ever misbehaves.
 *
 * There are two polling PHASES, because the DSP is waiting for two very different things:
 *
 *   FAST (grant sent, no receipt yet) — BML should verify the prompt within milliseconds. A dropped
 *     grant here means both chips are waiting on each other, so poll briskly and retransmit soon.
 *
 *   IDLE (receipt landed, answer in flight) — BML is generating for MINUTES (~10 s/token, the whole
 *     143 MB streamed per token). Polling at the fast cadence here is not merely wasteful: every
 *     wake costs at least one c2c_full_flush (a 256 KiB buffer walk, ~12k line touches), and four
 *     of those do not fit in 3.3 ms, so the DSP ends up walking its cache CONTINUOUSLY for the
 *     entire generation instead of sleeping. The MSIP from BML's hand-back still wakes us
 *     instantly, so a long interval costs nothing in the normal path — it only bounds how late we
 *     notice a hand-back whose wake was dropped.
 */

/* FAST phase: how often to wake, and how long without a receipt before retransmitting the prompt. */
#ifndef STT_LINK_POLL_FAST_MS
#define STT_LINK_POLL_FAST_MS 5u
#endif
#ifndef STT_LINK_REGRANT_FAST_MS
#define STT_LINK_REGRANT_FAST_MS 30u
#endif

/* IDLE phase: how often to wake while the peer is busy. Also used by BML while it waits for the
 * user to say something, which is the same situation seen from the other end. 1 s bounds the
 * worst-case extra latency from a dropped wake at ~10% of a single token. */
#ifndef STT_LINK_POLL_IDLE_MS
#define STT_LINK_POLL_IDLE_MS 1000u
#endif

/* IDLE phase: how long with a receipt but no answer before the DSP pokes again. This poll exists
 * ONLY so a dropped final hand-back cannot deadlock the pair; every one of them is a cross-link
 * write into a chip that is busy computing, so it is deliberately rare. */
#ifndef STT_LINK_REGRANT_IDLE_MS
#define STT_LINK_REGRANT_IDLE_MS 10000u
#endif

/* Milliseconds -> CLINT mtime ticks at a given core frequency (mtime_hz = core_hz / 1000).
 * 750 MHz x 60 s = 4.5e13, comfortably inside u64. */
#define STT_LINK_MS_TO_TICKS(ms, core_hz) \
  (((uint64_t)(core_hz) * (uint64_t)(ms)) / 1000000ull)

/* Return the generated answer to the DSP so the console you SPEAK at also shows what came back.
 * Costs one extra ~1 KB cross-link block write per turn, inside the turn the DSP is already waiting
 * on. Set to 0 to keep the link minimal — BML still prints the full answer on its own console. */
#ifndef STT_LINK_RETURN_ANSWER
#define STT_LINK_RETURN_ANSWER 1
#endif

#ifdef __cplusplus
}
#endif

#endif /* C2C_STT_LINK_PROTO_H */
