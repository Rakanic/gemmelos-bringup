#ifndef SMOLLM_CONFIG_H
#define SMOLLM_CONFIG_H

/* ------------------------------------------------------------------------------------------------
 * bearly25 baremetal SmolLM2-135M-Instruct (llama2.c runq grouped-Q8_0 int8) — tunables.
 *
 * Every knob here is `#ifndef`-guarded so CMake (or a -D on the command line) can override it
 * without editing this file; see CMakeLists.txt for the options that are wired up.
 *
 * Memory (DRAM is 256 MB at 0x8000_0000; see platform/bearly25/bearly25-llm.ld):
 *   .text + .rodata   ~143.1 MB   the .incbin'd model_q80.bin + tokenizer.bin (src/blob.S)
 *   heap              ~24 MB      KV cache (n_layers*seq_len*kv_dim*2*4 = 23.6 MB at seq_len 512),
 *                                 logits, the tokenizer merge table, RunState buffers
 *   stack             5 * 256 KiB pinned just under the top of DRAM by --defsym (CMakeLists.txt)
 * ---------------------------------------------------------------------------------------------- */

/* The model + tokenizer blobs are baked into the ELF (src/blob.S) and read in place out of
 * .rodata — no filesystem, no preload, no copy into the heap. */
extern const unsigned char g_smollm_model[];
extern const unsigned char g_smollm_model_end[];
extern const unsigned char g_smollm_tokenizer[];
extern const unsigned char g_smollm_tokenizer_end[];

/* Operating frequency (PLL target passed to init_test). 750 MHz is what boraiq runs at; 1 GHz did
 * not lock on dsp25 silicon. Lower it if the chip is unstable. */
#ifndef SMOLLM_TARGET_FREQUENCY_HZ
#define SMOLLM_TARGET_FREQUENCY_HZ 750000000ULL
#endif

/* Sampling. 0.0f = greedy/deterministic, which is what scripts/ref_runq.py reproduces exactly —
 * keep it at 0 while validating against the host reference, raise it for livelier demo output. */
#ifndef SMOLLM_TEMPERATURE
#define SMOLLM_TEMPERATURE 0.0f
#endif
/* Sample from the k most likely tokens when the temperature is non-zero (0 = the whole vocab). */
#ifndef SMOLLM_TOPK
#define SMOLLM_TOPK 40
#endif

/* Max tokens generated per answer (the model also stops on <|im_end|>). */
#ifndef SMOLLM_MAX_NEW_TOKENS
#define SMOLLM_MAX_NEW_TOKENS 200
#endif

/* Stop as soon as a token ends a sentence ('.', '!' or '?'), instead of running to <|im_end|> or
 * MAX_NEW_TOKENS. Decode is ~10 s PER TOKEN here (the whole model is streamed for each one), so on
 * a spoken-conversation demo the difference between a one-sentence answer and a four-sentence one
 * is minutes of staring at a console. Unlike lowering MAX_NEW_TOKENS, this ends on a complete
 * sentence rather than mid-word.
 *
 * OFF by default: the standalone demo is judged against a host reference token stream, and an early
 * stop would truncate it. The C2C voice demo turns it on.
 *
 * NOTE: the sentence-ending token is emitted but NOT fed back through the model, which saves the
 * forward pass that produced nothing (~10 s). In multi-turn mode the KV cache therefore omits that
 * one token — in practice a period — so the model's memory of its own last word is missing its
 * punctuation. That is the whole cost. */
#ifndef SMOLLM_STOP_AT_SENTENCE
#define SMOLLM_STOP_AT_SENTENCE 0
#endif

/* Minimum tokens before SMOLLM_STOP_AT_SENTENCE is allowed to fire, so an answer opening with an
 * abbreviation or a decimal ("3.14", "e.g.") is not cut off at its first period. */
#ifndef SMOLLM_MIN_NEW_TOKENS
#define SMOLLM_MIN_NEW_TOKENS 6
#endif

/* Tokens processed per prefill pass over the weights. The forward pass is entirely weight-
 * bandwidth bound (measured on silicon: 55 cyc/byte to read the model, 61 to read it AND multiply),
 * so prefilling one token at a time re-streams all 143 MB per prompt token. Batching reads the
 * weights once for the whole prompt: 31 tokens went from 31 passes to 1. Costs ~19 KB of scratch
 * per token. */
#ifndef SMOLLM_MAX_BATCH
#define SMOLLM_MAX_BATCH 32
#endif

/* Max bytes of one typed prompt line. */
#ifndef SMOLLM_PROMPT_MAX
#define SMOLLM_PROMPT_MAX 512
#endif

/* C2C mode (target `bearly-smollm-c2c`): take the prompt from the chip-to-chip link instead of the
 * console — DSP 25 listens on its microphone, transcribes with Citrinet, and sends the text here;
 * the generated answer goes back the same way. The console still shows everything. The link lives
 * entirely in c2c-demos/bearly-smollm-c2c/; everything it touches in this demo is behind this flag.
 * Mutually exclusive with SMOLLM_PROMPT (the compile-time autorun prompt). */
#ifndef SMOLLM_C2C
#define SMOLLM_C2C 0
#endif
#if SMOLLM_C2C && defined(SMOLLM_PROMPT)
#error "SMOLLM_C2C and SMOLLM_PROMPT both supply the prompt — pick one."
#endif

/* Submit the typed line after this many milliseconds of silence even if no CR/LF ever arrives.
 * GUI serial senders often transmit a message with no line terminator, and `uart_receive()` cannot
 * time out (its `return TIMEOUT` is commented out in the driver), so without this the console waits
 * forever and looks exactly like a hung chip. Set to 0 to require a real newline. */
#ifndef SMOLLM_LINE_IDLE_MS
#define SMOLLM_LINE_IDLE_MS 1200
#endif

/* System prompt, identical to the one in SmolLM2's chat template (so the on-chip prompt token
 * stream matches apply_chat_template — checked by scripts/tokenizer_ref.py --check). */
#ifndef SMOLLM_SYSTEM_PROMPT
#define SMOLLM_SYSTEM_PROMPT "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
#endif

/* Multi-turn: keep the KV cache across turns so the model remembers the conversation. When the
 * context is exhausted the conversation is reset (announced on the console). Set to 0 to start
 * every prompt from a clean context. */
#ifndef SMOLLM_MULTI_TURN
#define SMOLLM_MULTI_TURN 1
#endif

/* ------------------------------------------------------------------------------------------------
 * Few-shot examples: two complete user/assistant exchanges emitted right after the system block.
 *
 * MEASURED WORSE ON SILICON, and off by default — read this before turning it on. The theory was
 * that a 135M model follows FORM where it ignores instructions. What it actually does is treat the
 * examples as CONVERSATION CONTEXT and keep talking about their subject matter: with an example
 * mentioning transistors, "he" was answered with "A transistor is a fundamental..." and "hall how
 * are you" with "Hall is a transistor." The content leaked into every reply.
 *
 * Emitted once per conversation (inside the g_pos == 0 block), so with SMOLLM_MULTI_TURN=1 the cost
 * is a few dozen prompt tokens on the first question only, batched into the existing prefill passes.
 * With MULTI_TURN=0 it is paid per question — worth knowing before turning that off.
 *
 * Keep the answers SHORT and in the register you want back: they are the style specification, and
 * anything long here teaches the model to be long.
 * ---------------------------------------------------------------------------------------------- */
#ifndef SMOLLM_FEWSHOT
#define SMOLLM_FEWSHOT 0
#endif

/* Capitalize the ASR transcript and give it a terminal '.' or '?' before it becomes the prompt.
 * OFF by default — it was added alongside the few-shot experiment and reverted with it. */
#ifndef SMOLLM_POLISH_PROMPT
#define SMOLLM_POLISH_PROMPT 0
#endif
#ifndef SMOLLM_FEWSHOT_1_Q
#define SMOLLM_FEWSHOT_1_Q "Hello, how are you?"
#endif
#ifndef SMOLLM_FEWSHOT_1_A
#define SMOLLM_FEWSHOT_1_A "Good, thanks. What can I do for you?"
#endif
#ifndef SMOLLM_FEWSHOT_2_Q
#define SMOLLM_FEWSHOT_2_Q "What is a transistor?"
#endif
#ifndef SMOLLM_FEWSHOT_2_A
#define SMOLLM_FEWSHOT_2_A "A switch that uses a small voltage to control a larger current."
#endif

/* Use the RVV grouped-int8 dot product in matmul(). Falls back to scalar automatically when the
 * toolchain is built without the vector extension. */
#ifndef SMOLLM_USE_RVV
#define SMOLLM_USE_RVV 1
#endif

/* Print a per-token timing line (cycles, tok/s). */
#ifndef SMOLLM_LOG_TIMING
#define SMOLLM_LOG_TIMING 1
#endif

/* ------------------------------------------------------------------------------------------------
 * Memory-system figure of merit
 *
 * Decode here is 100% weight-bandwidth bound: every token streams all 143 MB of .rodata, so the
 * whole demo is a very slow ammeter for the memory bus. When a bitstream changes the bus clock, the
 * question is "by how much", and tok/s alone cannot answer it — it also moves with the core clock,
 * the prompt length and the number of tokens generated.
 *
 * These two knobs add the numbers that CAN be compared directly across bitstreams:
 *   MEM_PROBE       — a ~2 s boot sweep (c2c-demos/common/mem_probe.h): measured core clock, random
 *                     dependent-load latency in L1 and in DRAM, and streaming read bandwidth over
 *                     both the heap and the model blob itself, in cyc/byte and real MB/s.
 *   LOG_TIMING      — per answer, prefill and decode are also reported as cycles per MODEL BYTE and
 *                     as wall-clock seconds per token against the MEASURED clock. cyc/model-byte is
 *                     the invariant: it is ~61 on the old bitstream regardless of prompt or length.
 * ---------------------------------------------------------------------------------------------- */
#ifndef SMOLLM_MEM_PROBE
#define SMOLLM_MEM_PROBE 1
#endif

/* Heap scratch for the probe, taken and released before the KV cache is allocated. Must be several
 * times the 256 KB last-level cache or the "DRAM" latency point is really an L2 hit. */
#ifndef SMOLLM_MEM_PROBE_BYTES
#define SMOLLM_MEM_PROBE_BYTES (16u << 20)
#endif

/* Compare the residual-stream sums after every stage of the first forward() against the host
 * reference in include/smollm_golden.h, which localizes a diverging layer instead of leaving you
 * with "the output is wrong". Regenerate the header with
 *   scripts/ref_runq.py --golden include/smollm_golden.h
 * whenever the model blob changes. */
#ifndef SMOLLM_DEBUG_GOLDEN
#define SMOLLM_DEBUG_GOLDEN 0
#endif

/* Paint the stack at boot and report the high-water mark after the first answer. On this platform
 * a stack overflow does not trap — it silently corrupts automatics or hangs the core — so the
 * measurement is the only cheap way to know how close the margin is (see the LINKER=llm entry in
 * CLAUDE.md and .claude/plans/007-citrinet-dsp.md). */
#ifndef SMOLLM_STACK_PAINT
#define SMOLLM_STACK_PAINT 1
#endif
#ifndef SMOLLM_STACK_PAINT_BYTES
#define SMOLLM_STACK_PAINT_BYTES (128 * 1024)
#endif

/* Non-interactive autorun prompt: generate once from a compile-time prompt, then halt. Spike has
 * no UART input, so a Spike run needs this. Build with
 *   EXTRA_CMAKE_ARGS="-DSMOLLM_PROMPT=What is the capital of France?"
 * and keep SMOLLM_TEMPERATURE at 0 to diff the token stream against ref_runq.py. */
/* #define SMOLLM_PROMPT "What is the capital of France?" */

/* Quiet demo console: suppress every boot banner, configuration dump and per-answer statistic, so
 * the chip comes up straight to a prompt. ERRORS still print — the model/tokenizer loaders use
 * printf directly, so a bad blob or an exhausted heap still says so. */
#ifndef SMOLLM_QUIET
#define SMOLLM_QUIET 0
#endif

#if SMOLLM_QUIET
#define SMOLLM_LOG(...) do { } while (0)
#else
#define SMOLLM_LOG(...) printf(__VA_ARGS__)
#endif

/* The probe's measurement helpers print with plain printf (they are shared with the standalone
 * `membw` target, which has no notion of a quiet console), so silencing SMOLLM_LOG is not enough —
 * a demo console would still get a screenful of cyc/byte. Turn the whole probe off instead. */
#if SMOLLM_QUIET && SMOLLM_MEM_PROBE
#undef SMOLLM_MEM_PROBE
#define SMOLLM_MEM_PROBE 0
#endif

#endif /* SMOLLM_CONFIG_H */
