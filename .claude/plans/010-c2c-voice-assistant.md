# 010 — C2C voice assistant: Citrinet (DSP 25) → SmolLM (Bearly ML 25)

**Status:** **WORKS ON SILICON (2026-08-16)** — speak at the DSP mic, SmolLM answers on Bearly and
the answer comes back to the DSP console.
**Targets:** `c2c-demos/dsp-citrinet-c2c`, `c2c-demos/bearly-smollm-c2c`
**Protocol:** `c2c-demos/common/stt_link_proto.h`

Speak at the DSP's microphone. Citrinet-256 turns it into text on DSP 25, the text crosses the C2C
link, SmolLM2-135M-Instruct answers it on Bearly ML 25, and the answer comes back to the DSP
console. This is plan 007 (Citrinet) plus plan 008 (SmolLM) joined by the link pattern plans 001/002
proved with KWS.

```
  [DSP 25]  I2S mic ─VAD─► Citrinet-256 CTC ─► "what is a transistor"
                                                     │  C2C: DSP writes 0x1_D000_0000, turn := BML
                                                     ▼
  [Bearly ML 25]  prompt ─► SmolLM2-135M-Instruct prefill + decode ─► answer text
                                                     │  C2C: BML writes 0x1_C000_0000, turn := DSP
                                                     ▼
  [DSP 25]  prints the answer under the transcript
```

## What was actually built

Nothing about either model was rewritten. Both new targets **reuse their standalone demo's sources**
(the `bearly-kws-llama` precedent) so the transcription and generation paths cannot drift:

| new target | reuses | adds | flag |
|---|---|---|---|
| `dsp-citrinet-c2c` | `dsp-citrinet/src/{main,citrinet,model_data}` | `src/stt_link_dsp.c` | `-DDSP_CITRINET_C2C=1` (implies `USE_MIC`) |
| `bearly-smollm-c2c` | `bearly-smollm/src/{main,model,tokenizer,dualcore,uncore_probe,blob}` | `src/stt_link_bml.c` | `-DSMOLLM_C2C=1` |

Edits to the shared sources are small and behind those flags:

- `dsp-citrinet/src/main.c` — a `DSP_CITRINET_C2C` branch in `app_main`; `transcribe_audio()` now
  also fills a text buffer and returns it (one detokenization path for console and link).
- `dsp-citrinet/src/citrinet.c` — new `cn_tokens_to_text()`; `cn_print_tokens_text()` unchanged.
- `bearly-smollm/src/main.c` — a `SMOLLM_C2C` branch in `app_main`, a capture hook in
  `emit_piece()`, and `g_last_generated` for telemetry. `answer()` itself is untouched.

The standalone `dsp-citrinet` and `bearly-smollm` builds are byte-for-byte unaffected (both rebuilt
clean, no new warnings).

## Protocol

Same shape as `kws_stream_proto.h` — turn register at offset `0x20` in both spads, monotonic index,
checksum over a fixed-size payload, commit written last — with a **text** payload:

```
BML spad 0xD0000000  (DSP writes, BML reads)     DSP spad 0xC0000000  (BML writes, DSP reads)
  0x00 magic 'STTD'                                0x00 magic 'STTC'
  0x08 prompt_bytes                                0x04 ack_index      answer finished
  0x0C prompt_checksum                             0x08 rx_index       prompt RECEIVED  ← new
  0x10 prompt_index                                0x0C answer_bytes
  0x20 turn            ← the commit                0x10 bml_ready
  0x40 prompt_text[256]                            0x14 answer_checksum
                                                   0x20 turn           ← the commit
                                                   0x40 answer_text[1024]
```

### The one real design difference from KWS

KWS's producer re-grants the outstanding case on **every** idle timer tick until it is acked, which
is right when an inference is ~16 Mcyc. Here an answer is **minutes** of weight-bandwidth-bound
decode (~10 s/token at 750 MHz — all 143 MB streamed per token). Re-granting every ~3.3 ms for
minutes would pour cross-link writes into a chip that is busy computing, against the standing
"keep the link quiet" rule.

So the ack is split in two:

- **`rx_index` (receipt)** — BML writes it the moment the prompt verifies, *without giving up the
  turn*. It closes the window where a dropped grant needs retransmitting.
- **`ack_index`** — the answer is done and the turn is handed back.

The DSP retransmits briskly while `rx_index < n` (`STT_LINK_REGRANT_FAST_MS`, 30 ms) and only rarely
afterwards (`STT_LINK_REGRANT_IDLE_MS`, 10 s). The rare poll is still needed: a dropped *final*
hand-back would otherwise leave both chips asleep forever. Duplicate grants stay idempotent — BML
re-acks, re-sending the answer already staged in its buffer, without regenerating.

### Polling cadence: the waiter must not burn the chip it is waiting on

The receipt also gates the DSP's **own** poll interval, and that turned out to matter far more than
the retransmit rate. Every wake costs at least one `c2c_full_flush` — a 256 KiB buffer walk, 4096
lines × 3 passes — and the original loop did **four** per wake (three `c2c_local_read_u32` plus the
one inside `c2c_clear_own_msip`) at the turnsync default interval of ~3.3 ms at 750 MHz. Four walks
do not fit in 3.3 ms, so for the entire multi-minute generation the DSP was not sleeping at all: it
was walking its cache continuously.

Two changes, both keyed on the receipt:

- **Interval.** `c2c_turnsync.h` gained `c2c_sleep_ticks(n)` (the old `c2c_sleep_until_tick()` is now
  a wrapper, so KWS is untouched). Once the receipt lands the DSP backs off to
  `STT_LINK_POLL_IDLE_MS` = **1 s**. This costs nothing in the normal path — BML's hand-back MSIP
  wakes the DSP instantly regardless — it only bounds how late a *dropped* wake is noticed, at ~10%
  of one token.
- **Per-wake cost.** One spad read per wake instead of three: `rx_index` is monotonic, so once the
  receipt has arrived, re-reading it can only confirm what is already known.

Measured in wake counts for a 120 s answer: **36,036 wakes × 4 flushes → 120 × 2**, about 600× less.
BML's idle wait got the same treatment — waiting for a human to speak is the identical situation
seen from the other end.

All the timing knobs are now in **milliseconds**, converted per-chip by `STT_LINK_MS_TO_TICKS()`.
Tick counts were the wrong unit: CLINT `mtime` is derived from the core clock here (`core/1000`), so
a tick count means a different wall-clock time at every PLL setting — and dropping both chips to
500 MHz is the standing first move if the link misbehaves.

### Failure modes and what covers them

| dropped | recovered by |
|---|---|
| grant (payload / prompt_index / turn) | DSP retransmit, fast path (no receipt yet) |
| wake MSIP in either direction | CLINT timer (`c2c_sleep_until_tick`) — latency, not deadlock |
| receipt | DSP keeps retransmitting on the fast path; BML absorbs the dup |
| final hand-back / ack | DSP's slow re-grant → BML sees a duplicate → re-acks with the same answer |
| torn payload | checksum fails → BML hands back the *previous* index → DSP retransmits |
| DSP restarts alone (index resets to 1) | BML detects the decrease and resyncs `g_last_consumed` |

**Not covered: start order.** The readiness announcement is sent once, because repeatedly poking a
chip that may still be booting is exactly what kills it. Start the DSP first (it waits indefinitely
and issues no cross-link write before the barrier), or start both together. BML also idles
`STT_BML_STARTUP_GRACE_CYCLES` (~1 s) before announcing.

## Build & run

```bash
# once, if the SmolLM blobs are not exported yet
.venv/bin/python c2c-demos/bearly-smollm/scripts/export_smollm.py

# Bearly ML 25 — 143 MB, ~40 s over the 30 MHz SPI loader (~26 min over uart_tsi)
make build CHIP=bearly25 PLATFORM=CHIP TARGET=bearly-smollm-c2c EXTRA_CMAKE_ARGS="-DLINKER=llm"
make tsi-run TTY=<bml-tty> BINARY=build/c2c-demos/bearly-smollm-c2c/bearly-smollm-c2c.elf

# DSP 25
make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-citrinet-c2c EXTRA_CMAKE_ARGS="-DLINKER=llm"
make tsi-run TTY=<dsp-tty> BINARY=build/c2c-demos/dsp-citrinet-c2c/dsp-citrinet-c2c.elf
```

Start the **DSP first**, then Bearly. The DSP prints `waiting for bml_ready ...` every few seconds
until Bearly announces itself, then `ready — ask a question at the mic.`

Both chips are built for **750 MHz**. The KWS demos that proved this link ran at **500 MHz**; if the
link misbehaves, dropping both (`-DCN_C2C_TARGET_FREQ_HZ=500000000`,
`-DSMOLLM_C2C_TARGET_FREQ_HZ=500000000`) is the first thing to try. Keep them equal: the retransmit
periods are counted in CLINT ticks, and `mtime` is derived from the core clock on this silicon.

## Expected console

```
DSP:  [dsp-citrinet] mic: listening (VAD thresh=1500/1e6; speak now)...
      [dsp-citrinet] mic: onset energy=... -> capturing
      [dsp-citrinet] transcript: "what is a transistor"
      [stt-link] prompt 1 -> BML: "what is a transistor"
      ... (minutes) ...
      [stt-link] answered prompt 1 in NNNN Mcyc (NNNNN ms @ 750 MHz), 64 tokens, 0 re-grants
      [stt-link] SmolLM: A transistor is a semiconductor device that ...

BML:  You (spoken): what is a transistor         <- quiet console: the conversation only
      ..                                        <- one dot per prefill pass
      SmolLM: A transistor is a semiconductor device that ...
```

Build with `-DSMOLLM_C2C_QUIET=OFF` to get the boot banner, the per-answer timing and the link's
per-prompt telemetry back.

## First silicon run (2026-08-16) — works, and exposed one build bug

The round trip ran end to end on the first try. Two things were wrong, and they had the **same
root cause**: reusing a sibling demo's CMake cache-variable names.

`option()` and `set(... CACHE ...)` create **global** cache entries, and a second declaration of an
existing name is a **no-op** — the first one wins. `c2c-demos/CMakeLists.txt` adds `dsp-citrinet`
and `bearly-smollm` *before* these two directories, so every knob reusing their names silently
resolved to **their** defaults:

| variable | intended here | actually got | consequence |
|---|---|---|---|
| `CITRINET_USE_RVV` | ON | **OFF** | scalar encoder — ~3.4x slower (28 s vs 8.2 s on the golden clip) |
| `SMOLLM_MEM_PROBE` | OFF | **ON** | a screenful of boot output, ~2 s of probing, 16 MB of transient heap |
| `SMOLLM_MAX_NEW_TOKENS` | 64 | **200** | answers up to 3x longer than intended |
| `SMOLLM_CHECK_BLOB` | ON | OFF | (benign) |

Nothing in the build log said so — the `message(STATUS ...)` lines printed the values I *thought* I
had set, because they interpolated the same stale cache entries. Plan 007 already warned about
CMake caching options; this is the same trap one level up.

**Fix, and the rule:** every cache variable in a demo that reuses another demo's sources is now
prefixed with that target's own name (`CN_C2C_*`, `SMOLLM_C2C_*`). Verified: the C2C build's
`citrinet.c.obj` is now **byte-identical in size and vector-instruction count** to the known-good
standalone RVV build. Conclusions that should never be overridable (RVV on; CONV1D, MR6, PACK,
DMA staging, dual-core off — all measured losers or hangs in plan 007) are now plain
`target_compile_definitions`, not options, so no cache state can undo them.

The link's payload geometry (`STT_PROMPT_MAX_BYTES`, `STT_ANSWER_MAX_BYTES`,
`STT_LINK_RETURN_ANSWER`) is declared **once** in `c2c-demos/CMakeLists.txt`, because those three
genuinely must be identical on both chips and two competing defaults would be resolved by
subdirectory order.

### Demo console

`SMOLLM_C2C_QUIET` defaults **ON** (the standalone demo's default is OFF). The console now shows the
conversation and nothing else. The link log split in two to match: `STT_BML_LOG` (anomalies —
retransmit, duplicate grant, failed checksum, producer restart) always prints, because those lines
are what distinguish self-healing from a hang; `STT_BML_BOOT_LOG` (addresses, readiness, per-prompt
telemetry) follows `SMOLLM_QUIET`. What still prints when quiet: the spoken transcript, the token
stream, one dot per prefill pass, and any anomaly.

## Wake word: "marvin", all on the DSP (2026-08-16)

The DSP no longer transcribes everything it hears. An always-on TinySpeech detector gates the
expensive path: mic -> rolling 1 s MFCC window -> CNN -> only on "marvin" does Citrinet run and a
prompt cross the link. Bearly is not involved and can be mid-answer or idle.

**Why `marvin`.** Speech Commands v0.02 is a fixed 35-word vocabulary with nothing on bears or
Berkeley. `marvin` and `sheila` are the only names in it, and they are there precisely as wake-word
candidates; `marvin` has more clips (2,101) and a stronger onset. Everything else is either a
one-syllable command word likely to occur inside a question ("how do I *stop*...") or has too few
clips.

**It is a detector, not a classifier** — `scripts/train_wakeword.py`, separate from the 6-word
`rebuild_weights_simplecnn.py`, which is untouched. Class 0 is `marvin`; classes 1-5 are reject
buckets (digits / commands / short words / long words incl. `sheila` / background chopped from
`_background_noise_`), so background noise always has somewhere to go instead of being forced into a
keyword. Negatives outnumber positives ~5:1 and training augments time shift, gain and noise mixing,
because the model is judged on a live mic at distance, not on clean clips.

Two runtime constraints shaped it, both discovered by reading the shared runtime:

1. **Six classes, not three.** `fc_layer()` in `tinyspeech-mc/src/modules.c` hardcodes
   `output_features == 6` for its fp16 RVV path *and* that path's scalar fallback. Six keeps this
   model on the exact code path already proven on silicon and means the shared runtime needs no edit.
2. **`bias=False` on the classifier.** The header writer exports `FC_WEIGHT` and no bias, and
   `fc_layer(input, weights)` takes none — so a trained `nn.Linear` bias is silently dropped on the
   way to the chip. For argmax you mostly get away with it; for a detector with an absolute
   threshold you do not, because the threshold measured on the host would not be the one running on
   the DSP. **The shipped 6-word model has this bug** (see "Known Code Bugs" in CLAUDE.md).

### Measured (held-out test split: 195 marvin vs 1,280 negatives)

| wake margin = `logit[marvin] - max(other)` | recall | false accept **per window** |
|---|---|---|
| > 0 | 0.939 | 1.09% |
| > 1 | 0.877 | 0.31% |
| > 2 (**default**) | 0.821 | 0.078% |
| > 3 | 0.718 | 0.00% |

A per-window rate is not the rate you experience: at ~6 windows/s, 1% is a false wake every 17 s.
**`DSP_WAKE_CONSECUTIVE=2` is what makes it usable** — a real "marvin" spans several overlapping
windows and fires repeatedly; an isolated false accept does not repeat.

That also fixes the hop. Training augments word position by ±100 ms, so only a reasonably centred
window fires — the hop must be small enough that some window lands centred. **160 ms**
(`DSP_WAKE_HOP_FRAMES=16`) gives ~4 looks at a ~600 ms utterance.

### Shape: gated one-shot, NOT a sliding window (corrected on silicon 2026-08-16)

The first version slid the 1 s window forward every 160 ms and classified every step. That invented
a real-time budget the KWS demo never had — `dsp-kws-rolling` does not run a CNN on the DSP at all,
it streams features to Bearly — and scalar inference measured **151% of the hop**, so the microphone
kept running while the core was busy and the "1 second window" quietly became a second of audio with
holes in it.

The gate now copies the demo's actual shape: monitor 20 ms frames for an energy onset (cheap, no
MFCC, no CNN), capture ONE window with 200 ms of pre-roll, compute 94 frames, classify once, and
wake only if it says `marvin`. Everything else — a cough, a door, someone else's sentence — is
classified, rejected, and listening resumes. There is no real-time budget because nothing is being
tracked while the CNN runs.

It is also more accurate: the pre-roll puts the word roughly where it sits in a Speech Commands
clip, so the single look is a well-aligned one, whereas a blind sliding window mostly produces badly
centred ones and training only jittered position by ±100 ms. And the false-accept rate is now **per
onset** rather than per window of a free-running detector, which is why the margin can sit at 2.0
(recall 0.82) instead of being pushed up to buy back a rate that silence was inflating.

### The RVV kernels are broken on dsp25 — found by a golden self-test

Live margins read −88 when the model's own range on real audio is [−12.6, +9.5], and no input
bounded by ±127 can produce that (uniform random ±127 only reaches −20). Rather than keep guessing
at the microphone, `DSP_WAKE_GOLDEN_CHECK` runs fixed feature maps with host-computed logits — mic,
MFCC and quantization all bypassed. All 8 cases failed, every one picking the same class with the
same ranking and 10–100× magnitudes. Rebuilt with `-march=rv64imafd` on the TinySpeech units only:
**max_diff = 0 on all 8**, and live margins became sane.

Full entry in CLAUDE.md's bug log. Two things worth carrying forward:
- **A wrong theory costs a flash; a bisect costs the same flash and ends the argument.** DC offset
  was a plausible cause, matched the symptom's direction, and was wrong — proven wrong on the host
  in two minutes by injecting DC and watching the margin move only within [−6, +5].
- **Copy the golden check before trusting any NN kernel on a chip that has not run it before.**

### Cost

MFCC + scalar CNN is ~240 ms per detected onset. That is latency on a sound you actually made, not
a duty cycle: a quiet room costs only the 20 ms energy frames.

### Barge-in

A question asked while Bearly is answering **replaces** the one in flight: the DSP bumps
`prompt_index` and re-publishes (physically the same cross-link write as the self-heal re-grant that
already runs during generation), and Bearly checks between tokens — and between prefill passes,
since a long prompt is several 10 s passes — then abandons and picks up the newer one. The
abandoned prompt is deliberately never acked; the DSP stopped waiting on it the moment it published
the replacement.

This is why `stt_dsp_send_prompt()` became `stt_dsp_publish_prompt()` + `stt_dsp_poll()`. The DSP no
longer blocks on an answer at all — `wake_gate_listen()` takes `stt_dsp_poll` as a callback and
services the link once per hop, so an answer to an earlier question arrives on the console *while*
you are being listened to.

## Tunables worth knowing

| knob | default | why |
|---|---|---|
| `SMOLLM_C2C_MAX_NEW_TOKENS` | **64** (200 standalone) | ~10 s/token; 64 ≈ a paragraph and the DSP is holding the turn the whole time |
| `SMOLLM_C2C_MULTI_TURN` | 1 | the KV cache persists, so a spoken conversation has memory |
| `SMOLLM_C2C_QUIET` | **ON** (OFF standalone) | demo console: the conversation and anomalies, nothing else |
| `SMOLLM_C2C_MEM_PROBE` | **OFF** (ON standalone) | ~2 s and 16 MB of heap during a window the DSP is already blocked on |
| `SMOLLM_C2C_CHECK_BLOB` | OFF | turn ON when something looks wrong: separates a bad load from bad arithmetic |
| `STT_LINK_RETURN_ANSWER` | 1 | costs one ~1 KB block write inside a turn the DSP is already waiting on |
| `STT_PROMPT_MAX_BYTES` / `STT_ANSWER_MAX_BYTES` | 256 / 1024 | must match on BOTH chips — they agree only through the header |
| `DSP_CITRINET_VAD_*` (`-D` direct) | as `dsp-citrinet` | onset/end thresholds, pre-roll, tail; tune from the logged per-frame energies |

## Open / next

1. **Watch for the re-grant counter.** A non-zero `re-grants` in the DSP's per-answer line is the
   link self-healing; a steady stream of them means the cadence constants want tuning.
2. **Empty transcripts** are filtered on the DSP (a cough must not cost minutes of decode). If the
   VAD fires often on room noise, raise `DSP_CITRINET_VAD_THRESHOLD` rather than touching the link.
3. **Prompt quality.** Citrinet emits lowercase, unpunctuated text. SmolLM handles it, but if
   answers are poor, wrapping the transcript in a carrier sentence on the BML side is a one-line
   change in `build_turn`'s caller.
4. **Latency.** The whole round trip is dominated by decode. `SMOLLM_MAX_BATCH` already collapses
   prefill to one weight pass; the only remaining lever is reading fewer bytes (int4 — the
   `model_q4*.bin` blobs are already exported).
