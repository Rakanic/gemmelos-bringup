# CLAUDE.md

Working notes for Claude when developing on this repo. This is the **baremetal bringup
environment** for two SP25 chips: **Bearly ML 25** and **DSP 25**. Primary current focus
is the **chip-to-chip (C2C) test suite** under `c2c-demos/`.

Read the "Known Chip Bugs & Quirks" section before writing or changing any C2C code — it is
the living record of hardware behavior we've discovered on silicon. Respect it.

### Current focus

- **C2C VOICE ASSISTANT WORKS ON SILICON (2026-08-16) — Citrinet (DSP) -> SmolLM (BML) over the
  link (plan `.claude/plans/010-c2c-voice-assistant.md`).** Speak at the DSP mic -> `dsp-citrinet-c2c` transcribes with Citrinet-256 -> the TEXT
  crosses the C2C link -> `bearly-smollm-c2c` answers it with SmolLM2-135M-Instruct -> the answer
  comes back and prints on the DSP console too. This is plan 007 + plan 008 joined by the plan
  001/002 link pattern.
  - **Neither model was rewritten.** Both targets REUSE their standalone demo's sources (the
    `bearly-kws-llama` precedent) behind `-DDSP_CITRINET_C2C=1` / `-DSMOLLM_C2C=1`, so the
    transcription and generation paths cannot drift. Each adds exactly one file — `src/stt_link_dsp.c`
    / `src/stt_link_bml.c` — over the shared protocol `c2c-demos/common/stt_link_proto.h`.
    Standalone `dsp-citrinet` and `bearly-smollm` are unaffected.
  - **The one protocol change that matters: the ack is split in two.** KWS's producer re-grants on
    EVERY idle tick (~3.3 ms at 750 MHz) until acked — fine for a 16 Mcyc inference, wrong for a
    SmolLM answer that takes MINUTES, because it would pour cross-link writes into a busy chip. So
    BML writes a **receipt** (`rx_index`) as soon as the prompt verifies, WITHOUT giving up the turn;
    the DSP then retransmits briskly only until the receipt lands (`STT_LINK_REGRANT_TICKS_PRE_RX`)
    and rarely afterwards (`STT_LINK_REGRANT_IDLE_MS`, 10 s) — the slow poll still exists because a
    dropped FINAL hand-back would otherwise deadlock the pair. **Use this shape for any C2C payload
    whose consumer is slow.**
  - **A waiter's poll interval is not free — budget it against `c2c_full_flush`.** Every wake costs
    at least one 256 KiB evict walk (4096 lines x 3 passes), and the first version did FOUR per wake
    at the turnsync default (~3.3 ms at 750 MHz), which does not fit in the interval: the DSP spent
    the entire multi-minute generation walking its cache instead of sleeping. Fixed by
    `c2c_sleep_ticks(n)` (new in `c2c_turnsync.h`; `c2c_sleep_until_tick()` is now a wrapper, so KWS
    is unchanged) plus reading one word per wake instead of three — 36,036 wakes x 4 flushes -> 120
    x 2 over a 120 s answer. The timer is only a SAFETY NET; the peer's MSIP still wakes you
    instantly, so a long interval costs nothing but the latency of a *dropped* wake.
  - Link timing knobs are in **milliseconds** (`STT_LINK_MS_TO_TICKS`), not ticks: `mtime` is
    derived from the core clock, so a tick count means something different at every PLL setting.
  - **The first run cost a lesson worth more than the demo: CMake cache-variable names are global.**
    Reusing `dsp-citrinet`'s / `bearly-smollm`'s option names here silently inherited THEIR defaults
    — the C2C Citrinet shipped scalar (`CITRINET_USE_RVV=OFF`, ~3.4x slower) and SmolLM shipped with
    the memory probe on and a 200-token answer limit, with the build log printing the values that
    were *intended*. All cache variables are now prefixed `CN_C2C_*` / `SMOLLM_C2C_*`, and the
    measured conclusions from plan 007 (RVV on; CONV1D/MR6/PACK/DMA-staging/dual-core off) are plain
    `target_compile_definitions` so no cache state can undo them. See the rule in
    "`c2c-demos/` structure & conventions".
  - **WAKE WORD "marvin", entirely on the DSP (2026-08-16, built, pending silicon).** An always-on
    TinySpeech detector gates the expensive path — mic -> rolling 1 s MFCC window -> CNN -> only on
    `marvin` does Citrinet run and a prompt cross the link. The TinySpeech runtime is pure C + RVV
    (no `hal_ope`), so it compiles for dsp25 unchanged. Trained by the NEW
    `dsp25-tests/tinyspeech-test/scripts/train_wakeword.py` as a DETECTOR: class 0 = `marvin`,
    classes 1-5 = reject buckets (incl. background chopped from `_background_noise_`), so noise is
    never forced into a keyword. Weights live in `dsp-citrinet-c2c/include/wake/weights.h` and win by
    include ORDER over the shared 6-word `weights.h` — get that order wrong and the binary listens
    for "tree". Decision = `logit[marvin] - max(other) > DSP_WAKE_MARGIN` (2.0) for
    `DSP_WAKE_CONSECUTIVE` (2) windows; per-window false accept at margin 2 is 0.078%, and requiring
    consecutive windows is what turns that into a usable rate. Hop is 160 ms because training
    augments position by only ±100 ms. **Tune `DSP_WAKE_MARGIN` from the logged per-window margin**,
    exactly as the VAD threshold is tuned.
  - **Barge-in: a new question REPLACES one being answered.** The DSP bumps `prompt_index` and
    re-publishes; BML checks between tokens *and between prefill passes* and abandons. The abandoned
    prompt is never acked — the DSP stopped waiting the moment it published the replacement. This is
    why the DSP API is now `stt_dsp_publish_prompt()` + `stt_dsp_poll()` (non-blocking), with
    `wake_gate_listen()` calling `poll` once per hop so answers arrive while you are being listened
    to.
  - Console: `SMOLLM_C2C_QUIET` defaults ON (the conversation and link anomalies only);
    `-DSMOLLM_C2C_QUIET=OFF` restores the banner and telemetry.
  - Build: `make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-citrinet-c2c EXTRA_CMAKE_ARGS="-DLINKER=llm"`
    and `make build CHIP=bearly25 PLATFORM=CHIP TARGET=bearly-smollm-c2c EXTRA_CMAKE_ARGS="-DLINKER=llm"`.
    **Start the DSP first** (the readiness announcement is one-shot by design). Both chips run at
    750 MHz; the KWS demos proved the link at 500 — if it misbehaves, drop BOTH together.
- **FULL C2C KWS DEMO WORKS END-TO-END ON SILICON (2026-07-22).** `dsp-kws-rolling` computes MFCC on
  DSP → streams the case over the C2C link → `bearly-kws-rolling` runs TinySpeech and correctly
  predicts `yes` on the embedded `yes_test_005` sample. This is the payoff of plans 001+002.
- **Sync = the proven turn-taking pattern (plan 002 DONE).** Reused from `hello-wfi` via
  `c2c-demos/common/c2c_turnsync.h`: turn register (spad `0x20`, 0=DSP/1=BML) + CLINT-MSIP wake +
  CLINT-timer safety net + `ack_index`/`case_index`-keyed **self-heal retransmit**. Streams
  indefinitely; a dropped cross-link write self-heals (see "Reliable C2C turn-taking synchronization"
  and the bug log). This is the template for all C2C sync.
- **Two accuracy fixes were needed on top of sync** (see "KWS accuracy / TinySpeech" section):
  1. **MFCC layout transpose** — DSP now writes the case **coeff-major** (`idx = coeff*94 + frame`)
     to match the model input `{1,1,12,94}`; it was frame-major (scrambled features). FIXED.
  2. **int8 conv2 kernel is broken on silicon** (garbage activation max `M2 ≈ INT32_MAX` poisons
     calibration → wrong + non-deterministic). **Workaround: run the FLOAT pipeline**
     (`TINYSPEECH_INT8_PIPELINE=0`, CMake option `KWS_BEARLY_ROLLING_USE_FLOAT_PIPELINE=ON`). int8
     root-cause is deferred (suspect the RVV conv2 microkernel).
- **NEXT: validate other sample recordings** (plan `.claude/plans/003-kws-multi-testcase.md`) — the
  demo is proven on one `yes` clip; add more test inputs / cross-check against the 100-case
  reference labels. Then optionally: match the MFCC quant scale exactly, and fix the int8 conv2 bug.
- **The shared region / two scratchpads are non-negotiable** — they stay the transport.
- **LIVE I2S MIC AUDIO SOURCE WIRED (plan 001 P1, 2026-07-25) — built, pending on-silicon
  validation.** `dsp-kws-rolling` gained a `KWS_DSP_ROLLING_USE_MIC` CMake option: instead of
  embedded waveforms it captures ~1 s of live audio from the DSP I2S mic (proven in `dsp-i2s-test`),
  then runs the *same* MFCC → quantize → C2C-stream pipeline unchanged. Mic is clocked at **16 kHz
  directly** via `set_I2S_sample_freq(ch, target_freq, 16000, 32)` (matches the MFCC front-end — no
  resampling). Capture is **VAD-gated** (short-frame AC energy onset + pre-roll; knobs
  `KWS_DSP_ROLLING_VAD_THRESHOLD`/`_FRAME_SAMPLES`/`_PREROLL_SAMPLES`/`_ENABLE` in the config header).
  Live cases carry `expected_label = -1` → BML reports the prediction as `expected=unknown (not
  scored)`. **CONFIRMED WORKING ON SILICON (2026-07-25)** — speak at the mic, BML prints the keyword.
  Build DSP:
  `make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-kws-rolling
  EXTRA_CMAKE_ARGS="-DBUILD_MFCC_LIB=ON -DKWS_DSP_ROLLING_USE_MIC=ON -DKWS_DSP_ROLLING_MULTI_SIGNAL=OFF"`;
  BML: `... CHIP=bearly25 TARGET=bearly-kws-rolling EXTRA_CMAKE_ARGS="-DKWS_BEARLY_ROLLING_USE_FLOAT_PIPELINE=ON -DKWS_BEARLY_ROLLING_DEBUG_INPUT_COMPARE=0"`
  (INPUT-CMP is meaningless for live audio). Two quality knobs added after first-light:
  - **VAD energy threshold** `KWS_DSP_ROLLING_VAD_THRESHOLD` (DSP), default now **5e-4** (was 1e-4) —
    rejects quiet room noise; tune from the logged per-frame `energy` lines (each capture also prints
    an `absmean`/min/max stats line).
  - **Confidence gate** `KWS_BEARLY_ROLLING_MIN_SCORE` (BML, default **2.0**, was 3.0) — softmax is OFF
    so the score is the top raw **logit**; when it's not `> MIN_SCORE` the RESULT line prints
    `pred=no word` (and it's not captured as a keyword). Filters low-confidence non-speech.
  Mutually exclusive with `KWS_DSP_ROLLING_MULTI_SIGNAL`.
- **DUAL-CORE KWS + LLAMA — DYNAMIC STORY MODE (plan `.claude/plans/004-kws-llama-dualcore.md`,
  updated 2026-07-27).** Combined BML target **`c2c-demos/bearly-kws-llama`**: hart 0 runs the C2C KWS
  receiver + controller, hart 1 runs TinyLlama (borai int8) **continuously**. hart 0 collects the next
  `KWS_BEARLY_LLAMA_KEYWORDS_PER_STORY` (=3) confident keywords (top-logit > `KWS_BEARLY_ROLLING_MIN_SCORE`,
  now **2.0**); on the 3rd it builds a story prompt (`"Once upon a time, there was a <w0>, a <w1> and a
  <w2>."`), publishes it via a seqlock (`g_llama_prompt`/`g_llama_prompt_ver`), and raises `g_llama_stop`
  so hart 1 cuts off the current story and starts a new one about those words; then it collects the next
  3. (Earlier START/STOP-by-keyword-polarity design is superseded.) It reuses, unmodified in behavior,
  `bearly-kws-rolling/src/main.c`
  (`-DKWS_BEARLY_LLAMA`) and `bearly25-demos/borai/int8/src/main.c` (`-DKWS_LLAMA_COMBINED`) — both
  guarded so their standalone builds (`bearly-kws-rolling`, `boraiq`) are unchanged. hart-1 dispatch
  = a strong `__main` that (after `g_llama_ready`) enters `llama_run_forever()` and generates
  continuously; the prompt is swapped live via a seqlock (NOT the C2C spad; intra-die is coherent,
  fences order it); SMP-safe malloc via newlib `__malloc_lock`. Build:
  `make build CHIP=bearly25 PLATFORM=CHIP TARGET=bearly-kws-llama EXTRA_CMAKE_ARGS="-DBUILD_VECNN=ON"`
  (DSP side unchanged: the mic `dsp-kws-rolling`). Verify-on-silicon: UART interleave (add a print
  lock if it corrupts), shared-vs-per-hart RVV unit. See the plan for the full collision-resolution
  writeup.

- **SMOLLM2-135M-INSTRUCT ANSWERS QUESTIONS ON BEARLY ML 25 SILICON (plan
  `.claude/plans/008-smollm-bearly.md`, 2026-08-15).** `c2c-demos/bearly-smollm` is the `borai`
  TinyStories demo grown up: type a question over UART, get an answer from a real 135M instruct
  model — `"hello"` -> `"hello! how can i help you today?"`. 143 MB of grouped-Q8_0 weights
  `.incbin`'d into .rodata (flash it over the 30 MHz SPI loader, ~40 s; uart_tsi would be ~26 min),
  ~24 MB of heap for the KV cache, stack pinned into real DRAM.
  **Two silicon findings came out of it, both bigger than the demo:**
  - **RVV `vwredsum` returns garbage** (see the bug log) — the first working flash produced fluent
    nonsense until the int8 reduction was restructured to keep products in lanes and reduce in float.
  - **It is 100% weight-bandwidth bound: ~55 cyc/byte to read the model, 61 to read AND multiply.**
    143 MB/token = ~8.5 Gcyc = ~10 s/token at 750 MHz. Kernel choice barely matters; only reading
    FEWER bytes does. Hence batched prefill (one weight pass for the whole prompt instead of one per
    prompt token, `SMOLLM_MAX_BATCH`) — the single biggest win available, and the classifier is
    skipped for all but the last prompt token. Needs the new **`platform/bearly25/bearly25-llm.ld`**
  (`EXTRA_CMAKE_ARGS="-DLINKER=llm"`), which unlike `dsp25-llm.ld` keeps DRAM at its real 256 MB so the
  stack cannot land past the end of memory, and declares heap/stack as bare symbols so uart_tsi does
  not zero-fill ~100 MB of heap over the serial link.
  Three things generalise beyond this demo:
  1. **Keep the arithmetic in chip-free translation units.** `src/model.c` + `src/tokenizer.c` have no
     chip dependencies, so `scripts/check_c_forward.py` / `check_c_tokenizer.py` compile *the exact
     source that runs on silicon* for the host and diff it against a numpy reference and HuggingFace —
     seconds per iteration instead of a 26-minute flash. This caught a latent link break and a
     tokenizer question that inspection would not have.
  2. **int8 inference is chaotic; only position 0 is reproducible.** A last-bit float difference
     eventually flips one activation by one quantization step (measured: identical to ~1e-6 through
     layer 13, then 6% apart by layer 28, then a different token). Judge a run by the position-0
     golden check and by whether the text makes sense — never by equality with a reference token
     stream.
  3. **`PLATFORM=SIMS` must skip `init_test()`.** Spike models neither the PLL nor the UART, so
     programming them faults before the first character prints and looks exactly like a hang.

---

## The two chips

Both are RISC-V SoCs built from the Chipyard ecosystem. Board bringup is done over UART.

| | **Bearly ML 25** | **DSP 25** |
|---|---|---|
| `CHIP=` (build) | `bearly25` | `dsp25` |
| Platform dir | `platform/bearly25/` | `platform/dsp25/` |
| Accelerator | `CONV2D` (2D conv, `hal_ope`) | `CONV1D`/DMA/I2S (`hal_conv`, `hal_dma`, `hal_i2s`) |
| Role in C2C demos | **receiver / consumer** (bearly-*) | **producer / transmitter** (dsp-*) |
| `SYS_CLK_FREQ` | 50 MHz (nominal) | 50 MHz (nominal) |
| Linker scripts | `bearly25.ld`, `bearly25-maxheap.ld`, `bearly25-llm.ld` | `dsp25.ld`, `dsp25-flash.ld`, `dsp25-scratch.ld`, `dsp25-llm.ld` |
| OpenOCD cfg | `platform/bearly25/bearly25.cfg` | `platform/dsp25/dsp25.cfg` |

`platform/c2c25/chip_config.h` is nearly identical to `dsp25` and is where the C2C link
address window notes live (`// #define C2C_BASE 0x180000000U`).

Timing note: demos set a `TARGET_FREQUENCY_HZ` (e.g. 500 MHz in
`c2c_transfer_dsp_config.h`) that differs from the nominal `SYS_CLK_FREQ` (50 MHz). All
in-demo timing is measured in **core cycles via `rdcycle`**, not wall-clock, so treat cycle
counts as the source of truth and confirm the true operating frequency before converting to
seconds.

---

## C2C link & shared-memory model

- **Two scratchpads, one adjacent to each chip:**
  - **`0xC0000000` — DSP-adjacent scratchpad.** (`*_SHARED_BASE_ADDR`, 16 KiB.)
  - **`0xD0000000` — BML-adjacent scratchpad.**
- **Golden access rule — you may READ only your own adjacent spad; you may WRITE to both.**
  - DSP: reads `0xC0000000` (local); writes `0xC0000000` **and** `0xD0000000`.
  - BML: reads `0xD0000000` (local); writes `0xD0000000` **and** `0xC0000000`.
  - **Neither chip can read across the link.** To send data you *write into the other chip's
    spad*; that chip then reads it locally from its own spad. Never issue a load to the remote
    spad — it is not readable.
- **Cross-spad (remote) writes are the unstable ones** — repeat every write into the *other*
  chip's spad several times to make it stick (see unstable-access bug). Writes to your own spad
  are local/stable. Every read of your own spad still needs a **full cache flush first**,
  because the remote wrote it behind your cache's back.
- **Reaching the peer across the link (addressing).** You cannot READ the remote spad, but to
  WRITE the peer's spad or ring its MSIP, take the peer-local address and **prepend a leading `1`
  (bit 32)**:
  - Peer spad: from BML, DSP's `0xC000_0000` is reached at **`0x1_C000_0000`**; from DSP, BML's
    `0xD000_0000` is reached at **`0x1_D000_0000`**.
  - Peer CLINT MSIP: own MSIP is `0x0200_0000` (CLINT base, hart 0); the peer's is
    **`0x1_0200_0000`**. Writing `1` there raises a machine software interrupt on the peer.
- **Cross-chip wake = CLINT MSIP + `wfi`.** A sleeping core waits in `wfi` with `mie.MSIE` set and
  `mstatus.MIE=0` (so the interrupt wakes it but is NOT taken as a trap — no handler; execution
  resumes after `wfi`). The peer wakes it by writing its MSIP across the link. CLINT layout: MSIP
  @ `+0x0000`, `mtimecmp` @ `+0x4000`, `mtime` @ `+0xBFF8`; `MTIME_FREQ = 50 kHz` (20 us/tick).
- **The cache flush must be force-eviction.** Writing `1` to the cache-controller flush register
  (`0x02010200`) does **NOT** evict on this silicon — always use the 256 KiB buffer-walk
  (`cache_evict_all` / `hwfi_cache_flush`).
- **DSP writes, Bearly reads (data direction unchanged).** DSP is the producer.
- **Handshake** (finalized): DSP writes the case **into `0xD0000000` (BML's spad)** and sets a
  `data_ready` flag there. BML polls `0xD0000000` locally, reads the case, runs inference, then
  **clears `data_ready` in `0xD0000000`** (local write) and **sets a `rx_ready` flag by writing
  into `0xC0000000` (DSP's spad)**. DSP polls `0xC0000000` locally for `rx_ready` before sending
  the next case. (Note: this is the *physically forced* mapping — a receiver can only read its
  own spad, so payload for BML must live in BML's spad `0xD0000000`.)
- **Coherence is not automatic across the link, and access is unreliable.** See the
  **cache-manipulation** and **unstable-access** entries in the bug log — those two hardware
  facts dictate the shape of all shared-memory code. In short: every access (read *or* write,
  including a poll) must be followed by a **full cache flush**, and because individual
  reads/writes are not stable, intended data may need to be **written several times**.
  Demos implement this with:
  - `*_fence_rw()` → `fence rw, rw` around every shared-memory access.
  - `cache_evict_all()` / `cache_writeback_pressure()` — walks a large (`256 KiB`) aligned
    scratch buffer, touching one byte per 64-byte line for several passes, to force the entire
    cache out. This is the current stand-in for a real full-cache-flush instruction.
- Protocol headers live in `c2c-demos/common/`:
  - `transfer_proto.h` — single 64-bit cycle word, DSP overwrites with its `rdcycle` (latency
    measurement).
  - `simpletest_proto.h` — mailbox + ring of message slots with `commit_seq` handshake.
  - `kws_proto.h` — KWS streaming mailbox + ring slots / fast case slots (`commit_seq`
    guards a partially-filled slot: 0 while filling, N when case N committed).
  - `kws_rolling_proto.h` — rolling-window KWS variant built on `kws_proto.h`.
  - `stt_link_proto.h` — voice-assistant text link (transcript out, answer back). Same turn-taking
    layout as `kws_stream_proto.h`, plus the **split ack** (`rx_index` receipt / `ack_index` done)
    that keeps the link quiet while the consumer spends minutes generating.
- **Commit-sequence handshake pattern** (used across simpletest/kws): producer fills a slot,
  fences, then writes `commit_seq = N` last; consumer waits for `commit_seq != 0` / expected N
  before reading the payload. Never reorder the commit write before the payload write.

---

## Reliable C2C turn-taking synchronization (proven on silicon 2026-07-12)

The robust bidirectional sync we converged on after several failed attempts. Proven end-to-end in
`c2c-demos/hello-wfi` (single-chip `wfi`/MSIP sanity) and `c2c-demos/dsp-hello-wfi` +
`c2c-demos/bearly-hello-wfi` (two-chip interrupt ping-pong that counts a shared "baton" back and
forth forever). Shared implementation: **`c2c-demos/common/hello_wfi_link.h`**. **Reuse this
template for all future C2C sync.** It has been ported to the KWS demos (plan 002, DONE) via a
factored header **`c2c-demos/common/c2c_turnsync.h`** (same three layers, built on `c2c_shm` so it
shares one flush + write-repeat implementation). KWS uses turn-register offset **`0x20`** (not
`0x00`) because the KWS spad control block already uses `0x00`.

Why the earlier attempts failed: a one-shot MSIP edge into a core asleep in `wfi` is
**unrecoverable** if that single cross-link write drops — and cross-link writes drop
non-deterministically. No amount of "write it more times" removes the tail; you also need the
receiver to re-check independently, and a way to ignore wakes that aren't for it.

Three layers, each covering a distinct failure mode:

1. **Turn register (correctness / who-goes-next).** One word at **offset `0x00` of each spad**:
   `0 = DSP's turn`, `1 = BML's turn`. A chip reads it from its OWN spad (local, flush-first) and
   runs ONLY when the value equals its own id; any other value → back to `wfi`. Every wake is
   self-checking, so spurious / duplicate / early wakes never cause double-processing.
2. **CLINT timer (liveness / dropped-wake recovery).** Both cores arm a periodic machine-timer
   interrupt (`mie.MTIE`) alongside `MSIE`, with `mstatus.MIE=0` (wakes, never traps). So a
   sleeper re-checks its turn register at least every `HELLO_WFI_POLL_INTERVAL_TICKS` (~50 ms)
   even if the wake MSIP was dropped — a dropped MSIP costs latency, not liveness.
3. **Hardened cross-link writes (delivery).** Every cross-link store (data + turn register + MSIP)
   is repeated `HELLO_WFI_WRITE_REPEATS` times (fenced) then flushed, to fight the
   unstable-remote-write quirk.

**Spad layout:** turn register @ `0x00`, baton/data @ `0x04` (both 32-bit; spads are
32-bit-access-only).

**Handoff order (commit discipline):** write **data** into peer spad → set **turn register** in
peer spad (the commit; data is resident before the peer sees its turn) → set turn register in OWN
spad to the peer's id (so a later spurious wake of ours reads "not my turn") → raise peer **MSIP**.

**Wait path:** `wfi`; on every wake (MSIP or timer) clear own MSIP, flush, re-read own turn
register; proceed only when it is our turn.

**Boot:** each chip clears its own spad (baton + turn) and sets its own turn register to the
PEER's id, so it stays asleep until explicitly handed to (also defeats stale spad SRAM across a
chip-only reset).

**Residual / operational notes:**
- In plain `hello-wfi`, the one thing that must land is the **turn-register write** (hardened by
  repeats); the timer recovers a dropped MSIP but not a dropped turn write.
- **KWS added self-heal retransmit** (closes that residual, keyed on the monotonic counters):
  after granting case `N`, the producer waits for `ack_index >= N` and, on each idle timer tick with
  no ack, **re-grants `N`** (re-writes payload/`case_index` + turn + wake). The consumer only infers
  when `case_index > last_consumed`; a **duplicate** grant (its ack was lost) → it **re-acks without
  re-inferring**. A dropped grant, dropped ack, OR dropped wake all self-heal, and the monotonic
  guard prevents double-inference. On silicon this recovered mid-stream drops (you see occasional
  `re-grant`/`dup grant` log lines and it keeps going). **This is the recommended shape for any
  payload-carrying C2C sync.**
- **Start DSP first (or together).** boot-init sets the turn to "peer"; the boot barrier (`bml_ready`)
  ensures BML clears its spad before DSP's first write lands, removing the boot-order stale hazard.

---

## KWS accuracy / TinySpeech (findings 2026-07-22)

Getting correct predictions (not just a working link) took three separate discoveries. Once sync
worked, the model still mispredicted; debugging split cleanly into **link → MFCC front-end → inference**.

**The TinySpeech runtime lives in `bearly25-bmarks/tinyspeech-mc/`.** Classes (index order, since the
2026-07-27 **6-word REAL-AUDIO retrain** — `TINYSPEECH_NUM_CLASSES=6`): `0=go, 1=bird, 2=cat, 3=dog,
4=happy, 5=tree`. Trained on **real Google Speech Commands audio** (not TTS) via
`dsp25-tests/tinyspeech-test/scripts/rebuild_weights_simplecnn.py` (edit `CLASS_NAMES`), CPU, 25 epochs
→ **val 96% / test 93%**. Front-end = **torchaudio Hann-480** (win 480/30 ms, n_fft 1024, hop 160,
23 mel, 12 MFCC) with **per-case peak int8 quant** (`127/max|x|`); the DSP must match =
Hann-480 `mfcc_driver` + `KWS_DSP_ROLLING_MFCC_NORMALIZE=1` (both restored). Integrated by copying the
generated `weights.h` into the shared lib, `NUM_CLASSES=6`, and updating the label arrays. Consumed by
both `bearly-kws-rolling` and `bearly-kws-llama`. *(This replaced the 2026-07-25 8-word Piper-**TTS**
model hot/cold/chip/dale/apple/pear/messi/ronaldo, which classified everything as `messi` on the real
mic — a synthetic-vs-real-voice domain gap. Real-audio training fixed that.)* The `tinyspeech_inputs.h`
/ `tinyspeech_reference.h` goldens in the lib are **stale** (still 8-word) and now unused by the demos
(`KWS_BEARLY_ROLLING_DEBUG_INPUT_COMPARE` default is 0); regenerate them only if you need the
standalone benchmark/INPUT-CMP. Two golden references shipped in that lib:
- `include/tinyspeech_inputs.h` — 100 cases, each an **int8 MFCC map (12×94=1128 B) + expected_label**.
  This is the *exact* input Spike ran on (BML now includes it for debug/compare/calibration).
- `include/tinyspeech_reference.h` — per-case expected/predicted labels, probs, logits, and 12
  per-layer **stage sums** (great for localizing which layer diverges on-chip).

**1. MFCC layout was transposed (FIXED).** The model input is `{1,1,H=12,W=94}` → it reads
`g_case[coeff*94 + frame]` (**coefficient-major**). DSP originally wrote **frame-major**
(`g_case[frame*12 + coeff]`) → scrambled features. `compute_full_case()` in `dsp-kws-rolling` now
writes coeff-major. Confirmed via the `INPUT-CMP` diagnostic: the received coeff-0 row now ramps
like the reference.

**2. MFCC values still differ by scale/recipe (OPEN, but model is robust enough).** DSP computes
MFCC on-chip (`mfcc_driver`, FFT/mel/DCT + `quantize_mfcc` with `KWS_DSP_ROLLING_MFCC_QUANT_SCALE=4.0`,
`_ZERO=0.0`), which does NOT match the reference extractor bit-for-bit (window/mel/log/DCT + a
different int8 scale). After the transpose, `INPUT-CMP max_abs_diff ≈ 72` (coeff-0 spans ≈[-89,+40]
vs ref ≈[-127,+17]) — but the **float model still predicts `yes` correctly** on DSP's features. If a
future sample mispredicts, tune `QUANT_SCALE`/`_ZERO` to minimize `INPUT-CMP` `max_abs_diff`.

**3. The int8 pipeline is broken on silicon (WORKAROUND: float).** Calibration reduces to **three
integers** — the per-conv activation maxima `g_calib_max1/2/3` (everything else — requant
multipliers `mulN_q31`, scales `sN_fixed`, biases — is derived). On BML, `M2` (conv2) came back as
**deterministic garbage ≈ `INT32_MAX`** while `M1`/`M3` were sane, poisoning calibration → wrong +
non-deterministic predictions. It is **not** multicore-specific (identical single-core), so the
suspect is the **RVV conv2 microkernel** (a vector tail/reduction or accumulation-overflow bug the
Spike reference never exercises). **Workaround in use: the FLOAT model** (`TINYSPEECH_INT8_PIPELINE=0`),
which the reference validates and which bypasses the int8 conv2 path. Float is slower
(~16M cycles/inference vs ~7.7M int8) but correct.
- Also learned: our demo originally calibrated int8 on a **single** sample (degenerate scales); the
  standalone benchmark calibrates over all 100 cases then freezes. Added `tinyspeech_int8_calib_get_max`
  / `set_max` to `tinyspeech_int8.{h,c}` so a computed calibration (just the 3 maxima) can be **baked
  in** and loaded with no on-chip pass — relevant once the int8 conv2 bug is fixed.

**Debug harness (in `bearly-kws-rolling`, all `#ifndef`-guarded config flags):**
- `KWS_BEARLY_ROLLING_DEBUG_INPUT_COMPARE` (=1): prints `INPUT-CMP` once — received `g_case` vs
  `g_tinyspeech_test_inputs[REF_CASE_INDEX]` (mismatch count, max_abs_diff, both `[0:16]` previews).
- `KWS_BEARLY_ROLLING_USE_GOLDEN_INPUT` (=0 normally): infer on the reference bytes instead of the
  received case — isolates model/inference from the MFCC front-end + link. Infer log tags `[GOLDEN]`
  vs `[dsp]` so you always know which input was used.
- `KWS_BEARLY_ROLLING_REF_CASE_INDEX` (=5): which reference case (`yes_test_005`) to compare/use.
- `KWS_BEARLY_ROLLING_CALIBRATE_FULL` (int8 only): calibrate over all 100 ref cases at boot + print
  `CALIB_MAX M1/M2/M3`.
- CMake options in `c2c-demos/bearly-kws-rolling/CMakeLists.txt`:
  `KWS_BEARLY_ROLLING_USE_FLOAT_PIPELINE` (ON → `TINYSPEECH_INT8_PIPELINE=0`) and
  `KWS_BEARLY_ROLLING_FORCE_SINGLECORE_INT8` (int8 debug). New CMake options need a clean reconfigure.

**Current known-good config for the working demo:** float pipeline ON, golden OFF, transpose in
place, DSP pacing `KWS_DSP_ROLLING_INTER_CASE_QUIET_CYCLES=500M` (~1 s between predictions). Start
DSP first, then BML.

---

## Building & running

Build uses CMake driven by the top-level `Makefile`. Pick `CHIP` and a `TARGET`:

```bash
# DSP side (producer)
make build CHIP=dsp25    TARGET=c2c-transfer-dsp
# Bearly side (receiver)
make build CHIP=bearly25 TARGET=c2c-transfer-bearly
```

C2C demo targets (`c2c-demos/CMakeLists.txt`): `membw`, `bearly-smollm`, `dsp-citrinet-c2c` +
`bearly-smollm-c2c` (the voice assistant pair — mic -> Citrinet -> link -> SmolLM -> answer; both
need `-DLINKER=llm`), `dsp-kws`, `bearly-kws`, `dsp-kws-rolling`,
`bearly-kws-rolling`, `bearly-kws-llama` (dual-core KWS + TinyLlama voice control; bearly25 +
`-DBUILD_VECNN=ON`), `dsp-i2s-test`, `dsp-simpletest`, `bearly-simpletest`, `c2c-measure`,
`c2c-transfer-dsp`, `c2c-transfer-bearly`, `hello-wfi` (single-chip `wfi`/MSIP sanity — build
with either `CHIP`), `dsp-hello-wfi`, `bearly-hello-wfi` (two-chip turn-taking ping-pong; the
reference sync implementation). Each `<chip>-*` target must be built with the matching `CHIP=`.
Output ELF lands in `build/c2c-demos/<target>/<target>.elf`.

Flash / run on real silicon over UART (see `Makefile`):

```bash
make tsi-run   TTY=<tty> BINARY=build/c2c-demos/<target>/<target>.elf   # load & run via uart_tsi
make checktsi  TTY=<tty>                                                # sanity poke a scratch addr
```

Simulation path (VCS) exists via `make run CONFIG=... BINARY=...` but chip bringup is the
primary use case here.

Because a C2C run needs **both** chips: build+flash the DSP producer and the Bearly consumer
onto their respective boards, then start them. The producer uses a `STARTUP_DELAY_CYCLES`
grace period so the consumer can be up and polling first.

`PLATFORM=CHIP` (default) selects on-chip linker variants; `PLATFORM=SIMS` selects sim
linkers. `LINKER` can override the linker-script variant.

---

## `c2c-demos/` structure & conventions

Each demo is a standalone `app_init()` / `app_main()` executable:
- `src/main.c` — `app_init()` sets up, `app_main()` runs the loop then `wfi`s forever.
- `include/<name>_config.h` — all tunables as `#ifndef`-guarded `#define`s (packet counts,
  intervals, cache-evict geometry, log enable). Prefer adding a new guarded `#define` over
  hardcoding.
- `include/main.h` — pulls in config + `simple_setup.h` (from `bmark-lib/`).
- Logging via `<NAME>_LOG(...)` macro (wraps `printf`, gated by `<NAME>_LOG_ENABLE`).
- `_Static_assert` the cache-evict geometry invariants (power-of-two line size, evict bytes a
  multiple of a line).

When writing new C2C code, match the existing idiom: guarded config `#define`s, `*_LOG`
macros, `fence` + `cache_evict_all` around every shared-memory touch, `commit_seq`-style
handshakes for multi-word payloads, and structured single-line log records
(`key=value` pairs) so runs can be parsed after the fact.

**NEVER reuse another demo's CMake cache-variable name.** `option()` and `set(... CACHE ...)` create
**global** entries, and a second declaration of an existing name is a **NO-OP** — whichever
`add_subdirectory` runs first wins. When a demo reuses a sibling's sources (the `bearly-kws-llama` /
`dsp-citrinet-c2c` / `bearly-smollm-c2c` pattern), copying its option names silently inherits the
sibling's DEFAULTS, and the `message(STATUS ...)` line prints the stale value back at you so the
build log looks correct. This shipped `dsp-citrinet-c2c` with `CITRINET_USE_RVV=OFF` (scalar
encoder, ~3.4x slower) and `bearly-smollm-c2c` with `SMOLLM_MEM_PROBE=ON` — neither visible anywhere
(found 2026-08-16). Prefix every cache variable with the target's own name (`CN_C2C_*`,
`SMOLLM_C2C_*`). Better still: anything that is a measured **conclusion** rather than a preference
(RVV on; CONV1D / MR6 / PACK / DMA-staging / dual-core off) belongs in
`target_compile_definitions`, not an option — then no cache state, stale or fresh, can undo it.
A value two chips must agree on (e.g. the C2C payload sizes) belongs in the PARENT
`c2c-demos/CMakeLists.txt`, declared once.

---

## Planning docs & skills

- Longer design/planning docs for C2C work go in `.claude/plans/` (e.g. one file per feature
  or investigation). Keep CLAUDE.md itself lean; link out to plans. Current plans:
  `001-full-c2c-kws-stream.md` (KWS streaming design), `002-kws-turn-taking-sync.md` (DONE — sync
  ported + demo works end-to-end), `003-kws-multi-testcase.md` (active next: validate more sample
  recordings, then MFCC-scale match / int8 conv2 fix), `004-kws-llama-dualcore.md` (dual-core KWS +
  TinyLlama voice control — built, pending silicon), `005-whisper-dsp.md`, `006-moonshine-dsp.md`,
  `007-citrinet-dsp.md` (Citrinet STT — passes on silicon), `008-smollm-bearly.md`
  (SmolLM2-135M-Instruct on bearly25 — validated in Spike, pending silicon),
  `010-c2c-voice-assistant.md` (Citrinet -> C2C -> SmolLM voice assistant — built, pending silicon;
  read its "one real design difference from KWS" section before writing any C2C protocol whose
  consumer is slow).
- `.claude/plans/009-silicon-rtl-bug-list.md` collects the defects worth reproducing in RTL
  simulation (RVV `vwredsum`, cross-hart non-coherence, the DRAM read-throughput anomaly, silent
  hangs on bad accesses, the C2C link quirks, DMA strided gather, I2S watermark), each with the
  discriminating evidence and a directed test. Add to it when a new silicon-only defect turns up.
- If a repeatable workflow emerges (e.g. "bring up a new C2C demo", "parse a transfer log"),
  capture it as a skill rather than re-explaining each time.

---

## Known Code Bugs / TODO fixes

Software defects to fix later (distinct from the hardware quirks below).

### TinySpeech exports no FC bias — the deployed model is not the model that was trained
- **Where:** `dsp25-tests/tinyspeech-test/scripts/rebuild_weights_simplecnn.py` (`_model_to_weight_dict`)
  and `bearly25-bmarks/tinyspeech-mc/src/modules.c` (`fc_layer`).
- **Issue:** the model trains with `nn.Linear(96, n)`, which HAS a bias, but the weight dict exports
  `FC_WEIGHT` only and the runtime's `fc_layer(input, weights)` takes no bias argument. So the
  classifier bias is silently dropped between PyTorch and the chip: on-silicon logits differ from the
  trained model's by a per-class constant. Argmax survives it often enough that the 6-word KWS demo
  works, which is why it went unnoticed.
- **Why it matters more than it looks:** any decision with an ABSOLUTE threshold (a wake word, a
  confidence gate) is calibrated on numbers the chip does not reproduce.
- **Workaround:** the wake-word model (`scripts/train_wakeword.py`) trains with `bias=False`, so host
  and chip are exactly equivalent. Fix the 6-word path by doing the same, or by exporting an FC_BIAS
  tensor and teaching `fc_layer` to add it.
- **Status:** avoided in the wake-word model; open for `rebuild_weights_simplecnn.py`.

### `fc_layer` is hardcoded to 6 output classes
- **Where:** `bearly25-bmarks/tinyspeech-mc/src/modules.c` — both the fp16 RVV path and its scalar
  fallback are inside an `output_features == 6` branch, and `tinyspeech_prepack_fc96x6_weights`
  likewise. A model with a different class count silently leaves the optimized, silicon-proven path.
- **Rule:** keep TinySpeech models at 6 classes unless you are prepared to validate the generic path.
  The wake-word detector uses 1 wake class + 5 reject buckets for exactly this reason — which also
  happens to be a better detector than one lumped "not the wake word" class.
- **Status:** worked around by construction.

### int8 TinySpeech conv2 kernel produces garbage on silicon (use the float pipeline)
- **Where:** `bearly25-bmarks/tinyspeech-mc/src/tinyspeech_int8.c` — conv2 path (suspect the RVV
  microkernel `conv3x3_pool2x2_acc_c_rvv` / accumulation).
- **Issue:** the conv2 activation maximum `M2` comes back as **deterministic garbage ≈ `INT32_MAX`**
  (`M1`/`M3` sane), poisoning int8 calibration → wrong + non-deterministic predictions. Not
  multicore-specific (identical single-core). The Spike reference (host, no RVV) never exercises it.
- **Workaround:** run the **float pipeline** (`TINYSPEECH_INT8_PIPELINE=0`, CMake option
  `KWS_BEARLY_ROLLING_USE_FLOAT_PIPELINE=ON`). Correct, ~2× slower.
- **Action:** root-cause the RVV conv2 kernel (dump per-stage sums vs `tinyspeech_reference.h`
  `ref_stage_sums` to find the diverging layer). See "KWS accuracy / TinySpeech".
- **Status:** worked-around (float); int8 fix deferred.

### [DSP 25 / LINKER=llm] `dsp25-llm.ld` puts the stack ~1.9 GB PAST the end of real DRAM  (discovered 2026-08-12, measured on silicon)
- **Symptom:** three different stack sizes, three different failure modes, same source:
  | `__stack_start` | `__stack_size` | behaviour on silicon |
  |---|---|---|
  | `0xFFFFB000` | 4 KiB (default) | frames run off the bottom; stores do not stick -> **silent, intermittent corruption** |
  | `0xFFFB0000` | 64 KiB | works, but reads below ~`0xFFFB3900` return non-pattern garbage |
  | `0xFFEC0000` | 256 KiB | **writing there hangs the core outright** (no trap, no print) |
- **Cause:** `dsp25-llm.ld` declares `DRAM` as **2048M** so a large `.incbin`'d model fits inside
  medany's ±2 GiB PC-relative range, then derives the stack from the region END
  (`__stack_start = ORIGIN + LENGTH - 5*__stack_size`, i.e. just under `0x1_0000_0000`). The chip's
  DRAM actually ends at `0x8FFF_FFFF` (stock `dsp25.ld` says 256M). The stack therefore lands in a
  narrow, undocumented band near the top of the 32-bit space that is only PARTIALLY backed — and
  *growing* the stack walks off the bottom of that band into address space that hangs on write.
- **Rule:** with `LINKER=llm`, **pin the stack into real DRAM** rather than inheriting it from the
  fictitious region end. The symbols are `PROVIDE()`d, so `--defsym` overrides them:
  ```
  -Wl,--defsym=__stack_size=0x40000
  -Wl,--defsym=__stack_start=0x8FE00000   # + 5 harts * 256 KiB = 0x8FF40000, under the 0x90000000 top
  -Wl,--defsym=__heap_end=0x8FE00000      # so sbrk cannot grow the heap into the stacks
  ```
  See `c2c-demos/dsp-citrinet/CMakeLists.txt` for the worked example.
- **Still exposed:** `dsp-moonshine`, `dsp-whisper` and the tinyllama target all use `LINKER=llm`
  with 64 KiB stacks in that fragile window. They work today; they are one stack-size change away
  from not working.
- **Avoided by construction on bearly25:** `platform/bearly25/bearly25-llm.ld` (added 2026-08-15 for
  `c2c-demos/bearly-smollm`) declares DRAM as the REAL 256M, so a stack derived from the region end
  is inside real memory. Copy that approach rather than dsp25-llm.ld's 2048M when a new chip needs an
  `llm` variant; only declare an oversized region if a blob genuinely needs more than medany's ±2 GiB
  reach, and then pin the stack with `--defsym`.
- **Status:** fixed in `dsp-citrinet`, avoided in `bearly25-llm.ld`; open in the other `LINKER=llm`
  demos.

### (RESOLVED, but read this before porting a demo) A stack overflow on this platform is SILENT and looks exactly like a flaky chip
- **Where:** `c2c-demos/dsp-citrinet` — hit 2026-08-12; see `.claude/plans/007-citrinet-dsp.md`.
- **Issue:** `dsp-citrinet/CMakeLists.txt` never set `__stack_size`, inheriting the linker default of
  **4 KiB**, while `cn_run_validate` had an **18,800-byte frame** (it held a 17,272-byte `cn_model_t`
  as a local). Same ELF, different result every silicon run.
- **Why it hides:** with `LINKER=llm` the stack sits at the top of a declared-2 GiB DRAM region, so an
  overflow runs off the bottom into address space the SoC does not back — **stores stop reading back
  instead of faulting**. Only AUTOMATICS go wrong; register-held values stay correct, so a function
  can compute a correct checksum over bytes it simultaneously parses into garbage. Spike maps real
  zeroed RAM across that whole range, so the same overflow passes in simulation forever.
- **Rule:** every demo using `LINKER=llm` must set `-Wl,--defsym=__stack_size=0x10000` (as
  `dsp-moonshine` and `dsp-whisper` do), keep multi-KB objects `static` rather than automatic, and
  build with `-Wstack-usage=`. `dsp-citrinet` now paints its stack at boot and prints the high-water
  mark (`cn_stack_paint`/`cn_stack_report`) — copy that into any new demo on this platform.
- **Status:** fixed.

### (RESOLVED) Consumer read path write-then-flush
- The turn-taking rewrite (plan 002) replaced the old `refresh_shared`/`poll_next_frame` path. BML
  now reads its own spad via `c2c_local_read_block_verify` (full flush, then read, then FNV checksum,
  bounded retries) only when the turn register grants it the turn — proven correct on silicon
  (checksum passes every case). No dummy-write-before-read was needed. Historical note kept for
  context; the "full cache flush every access" quirk below still governs all reads.

## Known Chip Bugs & Quirks

> **This is the living log.** Add an entry every time we discover something about how the
> silicon actually behaves — especially anything that constrains how C2C code must be written.
> Newest first. Keep each entry concrete: what we observed, on which chip, and what the code
> must do about it.

**Entry template:**

```
### [<CHIP>] Short title  (discovered YYYY-MM-DD)
- **Symptom:** what we observed.
- **Scope:** which chip(s), which demo/peripheral, conditions to reproduce.
- **Workaround / rule:** what C2C (or other) code must do because of it.
- **Status:** open / worked-around / fixed-in-hw / under-investigation.
```

### [BOTH] CLINT `mtime` is derived from the core clock — `MTIME_FREQ` is only true at 50 MHz  (measured 2026-08-16, `membw` on dsp25)
- **Symptom:** counting `rdcycle` over CLINT `mtime` ticks and scaling by `MTIME_FREQ` (50 kHz)
  reports **49.99 MHz** on a chip whose PLL was programmed for 750 MHz. The measured ratio is
  **exactly 1000 cycles/tick = `SYS_CLK_FREQ/MTIME_FREQ`**, and it does not change when the PLL does.
- **What it means:** `mtime` ticks at core/1000, so its real frequency is **50 kHz x the PLL ratio**
  (750 kHz at 750 MHz). It is NOT an independent time base and cannot measure the clock that drives
  it. The PLL is fine — the console being legible proves it, since `init_test()` derives `UART0->DIV`
  from the same target.
- **Consequences, which are not cosmetic:**
  - Every `msleep()` / `sleep_ms_blocking()` / CLINT-timer interval in this tree is **short by the
    PLL ratio** at the operating clock. The KWS turn-sync "~50 ms" `HELLO_WFI_POLL_INTERVAL_TICKS`
    safety net is really ~3.3 ms at 750 MHz. It still works (it is a liveness backstop, and firing
    15x too often only costs wakeups), but nothing derived from `MTIME_FREQ` means what it says.
  - Any ns / MB/s figure computed from the naive measurement is **15x wrong**, and it is wrong in the
    direction that flips conclusions: a 3207-cycle DRAM RTT is 64 us at 50 MHz (needing ~33 lines in
    flight to reach 33 MB/s) but 4.3 us at 750 MHz (needing ~2).
- **Rule:** resolve the clock with **`c2c-demos/common/clock_probe.h`** — sample cycles-per-tick once
  before `init_test()` and once after, and read `PLL->RATIO`/`PLLEN`/`BYPASS` and the
  `RCC_CLOCK_SELECTOR` mux back. That distinguishes the three cases (mtime independent / mtime tracks
  the core / PLL genuinely not engaged) which otherwise produce identical timing evidence. Prefer
  metrics that are **ratios of cycle counts** (cyc/byte, overlap, lines-in-flight) — they are immune.
  The only independent time reference on this chip is a stopwatch; `membw` prints its own total
  runtime for exactly that cross-check.
- **Status:** open (software must work around it); worth an RTL/`chip_config.h` fix.

### [DSP 25] The RVV TinySpeech FLOAT kernels return garbage — build them scalar  (discovered 2026-08-16, measured on silicon)
- **Symptom:** running TinySpeech on dsp25 with the RVV kernels, fed FIXED feature maps whose logits
  were computed on the host from the same weights, every case was wrong: all 8 golden cases picked
  the SAME class, the class RANKING was identical across completely different inputs (only the
  magnitude moved), and magnitudes were 10-100x the reference (`max_diff` up to 245.56 against
  logits that should span about +-10). Live wake margins read -85..-119 where the model's own range
  on real audio is [-12.6, +9.5]. Rebuilt scalar, the same golden test passes with **max_diff = 0**
  on all 8, and live margins immediately became sane.
- **Scope:** dsp25, `bearly25-bmarks/tinyspeech-mc` FLOAT path (`TINYSPEECH_INT8_PIPELINE=0`) — i.e.
  the path the KWS demos rely on, but they run it on **bearly25**, where it is fine. The DSP had
  never run this CNN before `dsp-citrinet-c2c`; it only ever computed MFCC and streamed features.
- **What the signature says:** a fixed output ranking that is independent of the input, scaled by
  input magnitude, is what you get when the convolutions produce a spatially near-uniform map — so
  the GAP vector is a constant profile and each logit collapses to `const * rowsum(FC[c])`. Suspect
  the packed conv path (`tinyspeech_prepack_conv_weights`, only compiled under `__riscv_vector`).
- **Workaround / rule:** compile ONLY the TinySpeech translation units with `-march=rv64imafd` via
  `set_source_files_properties(... TARGET_DIRECTORY <tgt> PROPERTIES COMPILE_OPTIONS ...)`. That
  undefines `__riscv_vector` and selects the scalar paths the project's RVV_TYPE=0 mode already
  supports; the ABI is unchanged so it links normally, and other units (Citrinet's own RVV kernels)
  keep their vectorization. ~240 ms per inference scalar, which is fine for a gated one-shot.
- **How to detect it:** `DSP_WAKE_GOLDEN_CHECK` in `c2c-demos/dsp-citrinet-c2c` — fixed inputs,
  host-computed logits parsed from the deployed `weights.h`, got-vs-want printed at boot. **Copy
  this pattern before trusting any NN kernel on a chip that has not run it before.** It separated
  "the features are wrong" from "the maths is wrong" in a single flash, after a day of plausible
  and entirely incorrect theories about the microphone.
- **Status:** worked around (scalar); RTL/kernel root-cause open. Add to
  `.claude/plans/009-silicon-rtl-bug-list.md`.

### [BEARLY ML 25] ccache `WayEnable` reads 0 and ignores writes — this is NOT a problem  (checked 2026-08-15)
- **What it looks like:** the cache controller at `0x02010000` reports `Config=0x06080802` (2 banks,
  8 ways, 256 sets, 64 B lines = 256 KB) with `WayEnable=0`, and writing 7 reads back 0 — i.e.
  apparently one way enabled.
- **Reality:** this is the OPEN-SOURCE SiFive cache, where **all ways are always active** and
  `WayEnable` is not the gating control it is on the commercial IP. The full 256 KB is in use.
  Recorded here only so the "most of the L2 is disabled" reading does not get re-derived — it is
  wrong, and the slow memory path (below) has nothing to do with cache capacity.

### [BEARLY ML 25] Memory streams at ~31.6 cyc/byte and one line-fill at a time  (measured 2026-08-15)
- **Symptom:** reading DRAM costs ~55 cyc/byte with 32-byte loads and ~31.6 with 64-byte (full-line)
  loads; wider loads gain nothing, and 2/4/8 interleaved streams get monotonically WORSE
  (66/122/205 cyc/byte). Heap and `.rodata` are identical, so it is not a region property.
- **Rule for any streaming kernel:** issue loads of exactly one cache line (`e8m2` at VLEN=256) and
  use ONE sequential stream. In `bearly-smollm` that alone was worth **1.43x** on decode.
- **Consequence for LLM work:** a model of M bytes costs ~M x 31.6 cycles per token, whatever the
  kernel does. Only shrinking the model (int4) moves it further.
- **CAVEAT on the multi-stream numbers above:** they were taken with a loop that consumes each load
  with the very next instruction, so on an in-order core the misses cannot overlap *whatever* the
  MSHR count is — the monotonic degradation is at least partly the loop. `membw` now reports the
  dependent and independent forms side by side; use the independent one.
- **How to re-measure** (`membw`, added 2026-08-15, reworked 2026-08-16):
  `make build CHIP=<chip> PLATFORM=CHIP TARGET=membw EXTRA_CMAKE_ARGS="-DLINKER=chip"` then
  `make tsi-run TTY=<tty> BINARY=build/c2c-demos/membw/membw.elf`. Reports the RESOLVED core clock
  (see the `mtime` entry above — the naive measurement is 15x wrong), a pointer-chase latency curve,
  **memory-level parallelism from N independent chases** (the gate: flat cycles-per-round as N rises
  means the core overlaps N misses; linear scaling means one at a time, and then no wire clock can
  saturate an off-chip link from the core), streaming read/write/copy at every RVV LMUL, dependent-vs-
  independent stream sweeps, a stride sweep, and a **Little's Law verdict** (lines in flight sustained
  vs lines required for a target MB/s, tabulated at both candidate clocks). Shared primitives in
  `c2c-demos/common/mem_probe.h` + `clock_probe.h`, host-tested by
  `c2c-demos/membw/test/host_mem_probe_test.c`; `bearly-smollm` prints the same numbers at boot
  (`SMOLLM_MEM_PROBE=1`, default) plus **cyc/model-byte** per answer, which unlike tok/s does not
  move with prompt or answer length. See `c2c-demos/membw/README.md`.
- **No address translation is in play:** nothing in `glossy/`, `platform/` or `bmark-lib/` ever
  writes `satp` — this is bare M-mode. So no knee in the latency curve is a TLB or page-walk effect,
  and huge pages are not an available lever. The 256 KB -> 1 MB cliff is the LLC boundary.
- **Status:** open; RTL entry 3.

### [BEARLY ML 25] RVV `vwredsum.vs` (integer widening reduction) returns garbage — use a float-lane reduction  (discovered 2026-08-15, measured on silicon)
- **Symptom:** an int8 matmul whose inner loop is `vwmul.vv` -> `vwredsum.vs` -> `vmv.x.s` produces
  results unrelated to the correct answer. In `c2c-demos/bearly-smollm` it turned
  `"hello! how can i help you today?"` into `"atimstakingstaking elephstakingarxiv kilow"` — fluent
  garbage, not an obvious failure. Its on-chip benchmark, comparing kernels against a scalar
  reference computed in the same boot, reports `matmul rowdot ... max_diff 2147483647/1e6` (a
  saturated cast: the difference is enormous) while `matmul lane` and `matmul transp` report
  `max_diff 0`.
- **Scope:** bearly25 silicon. Not reproducible in Spike, which executes the same binary correctly —
  so anything validated only in simulation will miss it.
- **What it is NOT:** `vwmul.vv`, `vle8.v`, `vadd.vv`, `vfredusum.vs` and `vfmv.f.s` are all
  exercised by the kernels that measure `max_diff 0`. The failing and passing kernels differ by
  exactly `vwredsum.vs` and `vmv.x.s`. Float reductions are fine; the INTEGER widening reduction (or
  the scalar readout right after it) is not.
- **Almost certainly the same defect as the TinySpeech int8 conv2 bug below** (conv2 activation max
  `M2 ~ INT32_MAX` out of an RVV microkernel, worked around with the float pipeline): same
  signature, same instruction family, different demo.
- **Workaround / rule:** never reduce int8 products with `vwredsum`. Keep the products IN LANES and
  accumulate into a float vector, reducing once per output row with `vfredusum` — see
  `dot_row_lane()` in `c2c-demos/bearly-smollm/src/model.c` (CMake: `SMOLLM_MATMUL_LANEFLOAT=1`,
  the default). It is also no slower: both are weight-bandwidth bound anyway.
- **Status:** worked around; RTL investigation queued in `.claude/plans/009-silicon-rtl-bug-list.md`.

### [C2C link] Cross-chip wake via CLINT MSIP works, but a single wake can drop and a dropped wake into a sleeping core is unrecoverable  (discovered 2026-07-12, proven on silicon)
- **Symptom:** a chip can wake a `wfi`-sleeping peer by writing the peer's CLINT MSIP across the
  link (`0x1_0200_0000` — own MSIP `0x0200_0000` with a leading 1). But a **single** MSIP write
  drops non-deterministically; when it does, the sleeping peer never wakes and the exchange
  deadlocks (nothing re-drives an edge into a sleeping core). Repeating the write helped (1 → a
  few exchanges) but never reached 100%.
- **Scope:** any cross-chip wake; both directions. Same unstable-write family as remote-spad writes.
- **Workaround / rule:** the **turn-register + timer** pattern (see "Reliable C2C turn-taking
  synchronization"). Wait in `wfi` with `mie.MSIE` + `mie.MTIE` set and `mstatus.MIE=0`; a
  periodic machine timer re-wakes the sleeper so it re-reads a persistent **turn register** in its
  own spad — recovering any dropped MSIP within one interval instead of deadlocking. Harden the
  turn-register + MSIP writes with repeats. Start DSP first (no retransmit; boot-init marks the
  turn as the peer's). CLINT: MSIP `+0x0000`, `mtimecmp` `+0x4000`, `mtime` `+0xBFF8`, MTIME_FREQ 50 kHz.
- **Status:** worked-around (proven reliable in `hello-wfi` / `*-hello-wfi`).

### [C2C link] Writing the cache-controller flush register does NOT evict — must force-evict  (discovered 2026-07-12, observed on silicon)
- **Symptom:** writing `1` to the cache-controller flush register (`0x02010200`) did not make a
  peer's cross-link write visible; the reader kept seeing stale data.
- **Scope:** every own-spad read after a remote write; both chips.
- **Workaround / rule:** keep using the **force-eviction buffer walk** (touch one byte per 64-byte
  line across a 256 KiB aligned buffer, several passes, + `fence rw,rw`) — `cache_evict_all` /
  `hwfi_cache_flush`. The register write is not a substitute.
- **Status:** worked-around (hardware behavior).

### [C2C link] A cross-link write to an absent/wedged peer hangs the writer  (discovered 2026-07-05, observed on silicon)
- **Symptom:** a chip hangs on boot (needs an FPGA reset to recover) if it writes the peer's spad
  while the peer is powered off or the link is wedged. BML hung rebooting with DSP off because its
  `app_init` did a cross-link write (the boot-clear) to `0xC` before anything confirmed DSP was up.
- **Scope:** any store to the *other* chip's spad. Local reads/writes of your own spad are safe.
- **Workaround / rule:** **do no cross-link (peer-spad) writes in `app_init`** — only local setup
  there. Defer every write to the peer's spad into `app_main`, after the core has fully booted and
  printed. (DSP's identity/epoch write and BML's ack are now the first peer-spad writes, both in
  `app_main`.) A powered, booted-but-idle peer is fine to write to; a powered-off/wedged one is not.
- **Status:** worked-around (moved all peer-spad writes out of init).

### [C2C link] A cross-link write into a chip while it is BOOTING kills that chip  (discovered 2026-07-05, observed on silicon)
- **Symptom:** whichever chip comes up **second** dies — prints nothing, and the FPGA has to be
  reset. It is **not** poll-vs-write contention and **not** about *continuous* writes: even a
  **single** write from the first chip into the second chip's spad, landing during the second's
  boot, kills it. Fully symmetric (DSP-first kills BML; BML-first kills DSP). Confirmed: DSP did
  one `publish` to `0xD` while BML was still booting (not yet polling) and BML's FPGA died.
- **Scope:** both chips; any store into the peer's spad while the peer has not finished booting.
- **Workaround / rule:** **boot barrier.** No chip writes the peer's spad until the peer signals
  it has fully booted. Implemented as: BML boots, waits a grace, then writes `bml_ready`
  (`KWS_STREAM_READY_MAGIC`) into the DSP spad; DSP polls its **local** `0xC` for `bml_ready` and
  does not touch `0xD` until it sees it. Also keep steady-state traffic **link-quiet** (write the
  peer's spad only in brief, infrequent bursts; poll your own spad in between —
  `KWS_DSP_ROLLING_ACK_POLL_BUDGET`, `KWS_BEARLY_ROLLING_REACK_EVERY`). Prefer starting **DSP
  first** (it waits indefinitely for `bml_ready`), then BML.
- **Status:** worked-around with a boot barrier. *(Open: what exactly the peer must reach before
  it can safely receive a write — is `bml_ready` after runtime prep enough, or is a longer settle
  needed?)*

### [C2C link] Scratchpads are 32-bit-access-only — byte/half accesses hang  (discovered 2026-07-05, confirmed on silicon)
- **Symptom:** a **byte-granular** store to a scratchpad (`0xC`/`0xD`) hangs the core. BML hung on
  the first byte of a `memcpy`-style block write to `0xC`; switching that path to 32-bit word
  stores fixed it and BML progressed all the way to inference.
- **Scope:** all spad accesses on both chips. Word (32-bit) accesses work; byte/half do not.
- **Workaround / rule:** only ever access the spads with **aligned 32-bit** loads/stores.
  `c2c_shm`'s block helpers assemble/disassemble bytes and move whole words; callers pass
  4-aligned addresses and 4-byte-multiple lengths. Never `memcpy` to/from a spad.
- **Status:** worked-around (hardware constraint). Likely also explains earlier "DSP published but
  BML never received" — the byte-wise payload write was mangling data sub-word.

### [C2C link] Shared-region access requires a full cache flush every time  (discovered 2026-07-05)
- **Symptom:** data written to / read from the `0xC0000000` shared region is not made visible
  across the die by an ordinary load/store. Stale data is returned otherwise.
- **Scope:** every access to the shared region on either chip — reads, writes, and polling
  loops alike. All C2C demos.
- **Workaround / rule:** any time we touch the shared region for any purpose, we must **write
  to the address and then flush the entire cache**. In code this is the `cache_evict_all()` /
  `cache_writeback_pressure()` full-cache buffer walk plus `fence rw, rw`, done after every
  access. Do not treat a plain load/store to `0xC0000000` as sufficient.
- **Status:** worked-around (hardware behavior; the flush is mandatory, not an optimization).

### [C2C link] Cross-spad (remote) writes are unstable — repeat them  (discovered 2026-07-05)
- **Symptom:** a single write into the *other* chip's scratchpad across the link is not reliable
  — it may not "take."
- **Scope:** writes into the remote spad (DSP→`0xD0000000`, BML→`0xC0000000`); all C2C demos.
  Writes to your own adjacent spad are local and stable.
- **Workaround / rule:** **repeat every remote-spad write several times** so it sticks. A value
  is not "sent" until it has been written repeatedly. Sync protocols must be idempotent
  (flags/counters that tolerate duplicate writes). Local reads of your own spad still need a full
  cache flush first (remote wrote it behind your cache).
- **Status:** open / worked-around. *(Quantify how many repeats are needed in practice.)*
