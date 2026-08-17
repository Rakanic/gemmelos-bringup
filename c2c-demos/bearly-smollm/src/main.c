/*
 * SmolLM2-135M-Instruct on Bearly ML 25, int8 (grouped Q8_0) forward pass.
 *
 * Derived from llama2.c's runq.c (Andrej Karpathy, MIT) by way of dsp25-demos/tinyllama, with the
 * changes SmolLM and this board need:
 *
 *   - the model + tokenizer are .incbin'd into .rodata and read IN PLACE (src/blob.S); there is no
 *     filesystem, no mmap, and no copy into the heap;
 *   - the token embedding is dequantized ONE TOKEN AT A TIME. runq.c dequantizes the whole table up
 *     front, which for SmolLM's 49152 x 576 vocab would be a 113 MB float array — most of DRAM, to
 *     hold something the forward pass touches 576 floats of per token;
 *   - RoPE theta comes from the model header (SmolLM uses 100000, llama2 uses a hardcoded 10000),
 *     as does the RMSNorm epsilon;
 *   - grouped-attention (9 query heads over 3 KV heads) sizing follows the header;
 *   - the tokenizer is GPT-2 style byte-level BPE, not SentencePiece. tokenizer.bin carries a
 *     byte->id table plus the merge list as (a, b, ab) id triples in rank order, so encoding is
 *     exact BPE via a hash of pair -> rank, with no string compares in the merge loop;
 *   - the prompt is built with SmolLM2's chat template out of <|im_start|>/<|im_end|>.
 *
 * The host reference in scripts/ref_runq.py runs this same arithmetic in numpy and agrees with
 * HuggingFace fp32 greedy decoding token for token, so a disagreement here is a bug in this file.
 * Build and run instructions are in README.md.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "chip_config.h"
#include "uart.h"
#include "simple_setup.h"   /* init_test(): portable PLL + UART bring-up */
#include "smollm_config.h"
#include "smollm_model.h"
#include "smollm_tokenizer.h"
#include "mem_probe.h"      /* c2c-demos/common: memory latency / bandwidth, shared with membw */
#include "clock_probe.h"    /* ... and what the core clock ACTUALLY is (mtime cannot tell you) */

void smollm_dualcore_init(void);   /* src/dualcore.c */
void smollm_probe_uncore(void);    /* src/uncore_probe.c */
#if SMOLLM_C2C
#include "stt_link_bml.h"          /* c2c-demos/bearly-smollm-c2c: prompt in / answer out over C2C */
#endif
#if SMOLLM_DEBUG_GOLDEN
#include "smollm_golden.h"
#endif
#if SMOLLM_CHECK_BLOB
#include "smollm_blob_check.h"
#endif

uint64_t target_frequency = SMOLLM_TARGET_FREQUENCY_HZ;

static const smollm_config_t *cfg;   /* set once the model header is parsed */

/* Measured core clock and the model's byte count. Together they turn the per-answer cycle counts
 * into the two quantities that are comparable across bitstreams: cycles per MODEL BYTE (invariant
 * to prompt length and answer length, because decode streams the whole model once per token) and
 * wall-clock seconds. The clock is MEASURED rather than assumed because a bitstream that retimes
 * the memory bus may move the core PLL too, and then cyc/byte and tok/s tell different stories. */
static uint64_t g_core_hz;
static size_t g_model_bytes;

static uint64_t mtime_now(void) { return clint_get_time(CLINT); }

#if SMOLLM_DEBUG_GOLDEN
/* Print the residual-stream sum after every stage of the first forward pass next to the host
 * reference's, so a divergence names a layer instead of just showing up as bad text. */
static void golden_stage(int stage, const char *name, float sum) {
  if (stage >= SMOLLM_GOLDEN_STAGES) return;
  const float want = g_smollm_golden_sums[stage];
  const float rel = (fabsf(want) > 1e-6f) ? fabsf(sum - want) / fabsf(want) : fabsf(sum - want);
  printf("[smollm] GOLDEN %2d %-6s got %.6e want %.6e rel %.3e%s\r\n",
         stage, name, (double)sum, (double)want, (double)rel,
         rel > 0.02f ? "   <-- DIVERGES" : "");
}
#endif

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
}

/* ------------------------------------------------------------------------------------------------
 * Stack guard (lifted from c2c-demos/dsp-citrinet, where its absence cost a week).
 *
 * A stack overflow on this platform does not trap: it runs off the bottom of the range into address
 * space the SoC may not back, so stores silently fail to read back. Only AUTOMATICS corrupt —
 * register-held values stay correct — and under Spike the whole region is real zeroed memory, so
 * the failure is invisible in simulation. Paint at boot, scan after the first answer.
 * ---------------------------------------------------------------------------------------------- */
#if SMOLLM_STACK_PAINT
#define STACK_PAT 0x5A5A5A5Au
#define STACK_SLACK 512u   /* bytes below sp left unpainted: the live frame + call overhead */

static uintptr_t g_stack_lo, g_stack_hi;

static inline uintptr_t read_sp(void) {
  uintptr_t s;
  __asm__ volatile("mv %0, sp" : "=r"(s));
  return s;
}

static void stack_paint(void) {
  extern char __stack_start[], __stack_size[];  /* __stack_size is ABSOLUTE: its value IS the size */
  uint64_t hid;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hid));
  const uintptr_t size = (uintptr_t)__stack_size;
  const uintptr_t lo = (uintptr_t)__stack_start + (uintptr_t)hid * size;
  const uintptr_t hi = lo + size, sp = read_sp();
  g_stack_lo = lo;
  g_stack_hi = hi;
  SMOLLM_LOG("[smollm] stack: hart%lu [0x%lx, 0x%lx) = %lu KB, sp=0x%lx\r\n",
             (unsigned long)hid, (unsigned long)lo, (unsigned long)hi,
             (unsigned long)(size >> 10), (unsigned long)sp);
  if (sp <= lo || sp > hi) {
    SMOLLM_LOG("[smollm] STACK: sp is OUTSIDE the linker's range — check __stack_start/_sp\r\n");
    g_stack_lo = g_stack_hi = 0;
    return;
  }
  for (uintptr_t a = lo; a + 4u <= sp - STACK_SLACK; a += 4u) *(volatile uint32_t *)a = STACK_PAT;
}

static void stack_report(void) {
  if (!g_stack_hi) return;
  const uintptr_t lo = g_stack_lo, hi = g_stack_hi;
  uintptr_t deepest = hi;
  unsigned long touched = 0;
  for (uintptr_t a = lo; a + 4u <= hi; a += 4u)
    if (*(volatile uint32_t *)a != STACK_PAT) { if (deepest == hi) deepest = a; touched++; }
  const unsigned long used = (unsigned long)(hi - deepest), size = (unsigned long)(hi - lo);
  const unsigned long span = used / 4u;
  (void)size; (void)touched;
  /* Real depth disturbs nearly every word between the deepest mark and the top; a sparse scattering
   * means something wrote INTO the stack from outside and `used` overstates the depth. */
  const int sparse = (span > 64u) && (touched * 4u < span);
  (void)sparse; (void)span;
  SMOLLM_LOG("[smollm] stack: high-water %lu of %lu bytes (%lu%%), %lu/%lu words disturbed%s%s\r\n",
             used, size, size ? (used * 100u / size) : 0u, touched, span,
             sparse ? "  <-- SPARSE: a stray write, not real depth" : "",
             (deepest <= lo) ? "  <-- OVERFLOW: locals are unreliable on silicon" : "");
}
#else
static void stack_paint(void) {}
static void stack_report(void) {}
#endif

/* Model + forward pass: src/model.c, kept free of chip dependencies so the same code can be
 * compiled for the host (test/host_forward_test.c) and checked against the numpy reference.
 * ---------------------------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------------------------------
 * Tokenizer (byte-level BPE) — implementation in src/tokenizer.c, which has no chip dependencies
 * so scripts/check_c_tokenizer.py can compile that exact file for the host and diff it against
 * HuggingFace over a corpus.
 * ---------------------------------------------------------------------------------------------- */

/* Print a token's bytes, turning '\n' into CRLF for the serial console. */
static void emit_piece(int token) {
  int len = 0;
  const uint8_t *p = smollm_tok_piece(token, &len);
  char buf[2 * 96 + 1];
#if SMOLLM_C2C
  /* Capture the RAW piece for the link before the console's CRLF expansion — the DSP prints it
   * through its own console, so line endings are that console's business, not ours. */
  stt_bml_capture(p, len);
#endif
  int o = 0;
  for (int i = 0; i < len && o < (int)sizeof(buf) - 3; i++) {
    if (p[i] == '\n') { buf[o++] = '\r'; buf[o++] = '\n'; }
    else buf[o++] = (char)p[i];
  }
  buf[o] = '\0';
  printf("%s", buf);
  fflush(stdout);
}

#if SMOLLM_STOP_AT_SENTENCE
/* Does this token's text end a sentence? Trailing quotes and brackets are skipped so `end."` and
 * `end.)` still count. Decimals and abbreviations inside a sentence would also match, which is why
 * the caller additionally requires SMOLLM_MIN_NEW_TOKENS — for a spoken-answer demo that is the
 * right trade against paying ~10 s per extra token. */
static int piece_ends_sentence(int token) {
  int len = 0;
  const uint8_t *p = smollm_tok_piece(token, &len);
  for (int i = len - 1; i >= 0; i--) {
    const uint8_t c = p[i];
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '"' || c == '\'' ||
        c == ')' || c == ']') {
      continue;
    }
    return (c == '.' || c == '!' || c == '?');
  }
  return 0;   /* whitespace-only piece: not an ending */
}
#endif

/* ------------------------------------------------------------------------------------------------
 * Sampling
 * ---------------------------------------------------------------------------------------------- */

static uint64_t g_rng = 0x853c49e6748fea9bULL;

static float rand_f32(void) {
  g_rng ^= g_rng >> 12; g_rng ^= g_rng << 25; g_rng ^= g_rng >> 27;
  return (float)((g_rng * 0x2545F4914F6CDD1DULL) >> 40) / 16777216.0f;
}

/* Greedy at temperature 0 (deterministic, and what scripts/ref_runq.py reproduces); otherwise
 * top-k multinomial. Top-k rather than top-p because a 49152-entry nucleus needs the vocabulary
 * SORTED, which would cost more than a transformer layer, while the k largest can be selected in
 * one pass with a k-element insertion list. */
static int sample_token(float *logits) {
  const int n = cfg->vocab_size;
  if (SMOLLM_TEMPERATURE == 0.0f) {
    int best = 0;
    for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  for (int i = 0; i < n; i++) logits[i] /= (float)SMOLLM_TEMPERATURE;
  smollm_softmax(logits, n);

  const int k = (SMOLLM_TOPK > 0 && SMOLLM_TOPK < n) ? SMOLLM_TOPK : n;
  static int top_i[SMOLLM_TOPK > 0 ? SMOLLM_TOPK : 1];
  static float top_p[SMOLLM_TOPK > 0 ? SMOLLM_TOPK : 1];
  float sum = 0.0f;
  if (k < n) {
    int have = 0;
    for (int i = 0; i < n; i++) {
      const float p = logits[i];
      if (have == k && p <= top_p[k - 1]) continue;
      int j = (have < k) ? have++ : k - 1;
      while (j > 0 && top_p[j - 1] < p) { top_p[j] = top_p[j - 1]; top_i[j] = top_i[j - 1]; j--; }
      top_p[j] = p;
      top_i[j] = i;
    }
    for (int i = 0; i < have; i++) sum += top_p[i];
    const float coin = rand_f32() * sum;
    float cum = 0.0f;
    for (int i = 0; i < have; i++) { cum += top_p[i]; if (cum > coin) return top_i[i]; }
    return top_i[have - 1];
  }
  const float coin = rand_f32();
  float cum = 0.0f;
  for (int i = 0; i < n; i++) { cum += logits[i]; if (cum > coin) return i; }
  return n - 1;
}

/* ------------------------------------------------------------------------------------------------
 * Chat
 * ---------------------------------------------------------------------------------------------- */

static int g_pos;              /* current KV-cache position = conversation length in tokens */
static int g_turn_open;        /* an assistant turn still needs its <|im_end|>\n */
static int g_prompt_toks[2048];
static int g_last_generated;   /* tokens produced by the most recent answer() (link telemetry) */

/* Neither the autorun build nor the C2C build ever reads the console for a prompt. */
#if !defined(SMOLLM_PROMPT) && !SMOLLM_C2C
/* Non-blocking UART read. `uart_receive()` takes a timeout argument and IGNORES it — its RX-empty
 * spin has `return TIMEOUT` commented out (driver/rocket-chip-blocks/uart/uart.c) — so it can only
 * block forever. Poll the register directly instead. */
static int uart_getc_nb(unsigned char *c) {
  const uint32_t rx = UART0->RXDATA;
  if (READ_BITS(rx, UART_RXDATA_EMPTY_MSK)) return 0;
  *c = (unsigned char)READ_BITS(rx, UART_RXDATA_DATA_MSK);
  return 1;
}

/* Read one line, submitting on CR/LF — or after SMOLLM_LINE_IDLE_MS of silence with something
 * typed. The idle fallback exists because GUI serial senders commonly transmit a message with no
 * line terminator at all, and against a console that only submits on CR/LF that is indistinguishable
 * from a hung chip: the text echoes back and then nothing ever happens. */
static void read_line(const char *guide, char *buf, int cap) {
  printf("%s", guide);
  fflush(stdout);
  const uint64_t idle_cycles = (uint64_t)SMOLLM_LINE_IDLE_MS * (target_frequency / 1000ull);
  uint64_t last = rdcycle64();
  int n = 0;
  for (;;) {
    unsigned char c = 0;
    if (!uart_getc_nb(&c)) {
      if (n > 0 && (rdcycle64() - last) > idle_cycles) {
        SMOLLM_LOG("\r\n[smollm] (no newline received; taking the line after %d ms idle)",
                   (int)SMOLLM_LINE_IDLE_MS);
        printf("\r\n");
        break;
      }
      continue;
    }
    last = rdcycle64();
    if (c == '\r' || c == '\n') { printf("\r\n"); break; }
    if ((c == '\b' || c == 0x7f) && n > 0) { n--; printf("\b \b"); }
    else if (c >= 0x20 && c < 0x7f && n < cap - 1) { buf[n++] = (char)c; printf("%c", c); }
  }
  buf[n] = '\0';
}
#endif

/* Build the next turn's tokens per SmolLM2's chat template:
 *   <|im_start|>system\n<SYS><|im_end|>\n<|im_start|>user\n<MSG><|im_end|>\n<|im_start|>assistant\n
 * The system block is emitted once per conversation; a still-open assistant turn is closed first.
 * scripts/tokenizer_ref.py --check verifies this token stream against apply_chat_template. */
static int build_turn(const char *user, int *out, int cap) {
  static char scratch[SMOLLM_PROMPT_MAX + 16];
  int n = 0;
  if (g_turn_open) {
    if (n < cap) out[n++] = smollm_tok_eos();
    smollm_tok_encode("\n", out, &n, cap);
    g_turn_open = 0;
  }
  if (g_pos == 0) {
    if (n < cap) out[n++] = smollm_tok_bos();
    snprintf(scratch, sizeof(scratch), "system\n%s", SMOLLM_SYSTEM_PROMPT);
    smollm_tok_encode(scratch, out, &n, cap);
    if (n < cap) out[n++] = smollm_tok_eos();
    smollm_tok_encode("\n", out, &n, cap);
#if SMOLLM_FEWSHOT
    /* Two worked examples, in the same <|im_start|>role ... <|im_end|> form the model generates in.
     * At 135M this is what actually stops the prompt being echoed back — see smollm_config.h. */
    {
      static const char *const shots[][2] = {
        {SMOLLM_FEWSHOT_1_Q, SMOLLM_FEWSHOT_1_A},
        {SMOLLM_FEWSHOT_2_Q, SMOLLM_FEWSHOT_2_A},
      };
      for (unsigned s = 0; s < (sizeof(shots) / sizeof(shots[0])); ++s) {
        if (n < cap) out[n++] = smollm_tok_bos();
        snprintf(scratch, sizeof(scratch), "user\n%s", shots[s][0]);
        smollm_tok_encode(scratch, out, &n, cap);
        if (n < cap) out[n++] = smollm_tok_eos();
        smollm_tok_encode("\n", out, &n, cap);

        if (n < cap) out[n++] = smollm_tok_bos();
        snprintf(scratch, sizeof(scratch), "assistant\n%s", shots[s][1]);
        smollm_tok_encode(scratch, out, &n, cap);
        if (n < cap) out[n++] = smollm_tok_eos();
        smollm_tok_encode("\n", out, &n, cap);
      }
    }
#endif
  }
  if (n < cap) out[n++] = smollm_tok_bos();
  snprintf(scratch, sizeof(scratch), "user\n%s", user);
  smollm_tok_encode(scratch, out, &n, cap);
  if (n < cap) out[n++] = smollm_tok_eos();
  smollm_tok_encode("\n", out, &n, cap);
  if (n < cap) out[n++] = smollm_tok_bos();
  smollm_tok_encode("assistant\n", out, &n, cap);
  return n;
}

#if SMOLLM_MEM_PROBE
/* ------------------------------------------------------------------------------------------------
 * Boot-time memory-system probe.
 *
 * Decode is entirely weight-bandwidth bound, so this demo IS a memory benchmark — just a very slow
 * one that takes ten seconds per sample and mixes in prompt length, sampling and the core clock.
 * These few lines isolate the memory system itself, in the same units and with the same code as the
 * standalone `membw` sweep, so a run before and after a bitstream change can be diffed directly.
 * Costs ~2 s of boot; set SMOLLM_MEM_PROBE=0 to skip it.
 *
 * Runs BEFORE smollm_model_alloc() so the whole heap is free for the 16 MB chase buffer.
 * ---------------------------------------------------------------------------------------------- */
static void mem_probe_boot(const void *model, size_t model_bytes) {
  SMOLLM_LOG("[mem] --- memory system (c2c-demos/membw runs the full sweep) ------------------\r\n");

  void *buf = malloc(SMOLLM_MEM_PROBE_BYTES);
  if (buf != NULL) {
    mem_probe_summary(buf, SMOLLM_MEM_PROBE_BYTES, g_core_hz);
    free(buf);
  } else {
    SMOLLM_LOG("[mem] %lu MB scratch malloc failed — latency curve skipped\r\n",
               (unsigned long)(SMOLLM_MEM_PROBE_BYTES >> 20));
  }

  /* And the blob itself. The heap and .rodata measured identical on the old bitstream (it is the
   * same DRAM either way), but this is literally the memory decode streams, so quote it too rather
   * than argue about whether the region matters. */
  const uint8_t *p = (const uint8_t *)model;
  const size_t adj = (8u - ((uintptr_t)p & 7u)) & 7u;   /* mem_probe_read_u64 wants 8-alignment */
  p += adj;
  size_t span = model_bytes - adj;
  if (span > (32u << 20)) span = 32u << 20;
  uint64_t sink = 0;
  mem_probe_print_bw("model .rodata u64", mem_probe_read_u64(p, span, &sink), span, g_core_hz);
#if MEM_PROBE_RVV
  mem_probe_print_bw("model .rodata rvv m8", mem_probe_read_rvv(p, span, 8, &sink), span, g_core_hz);
#endif
  SMOLLM_LOG("[mem] ---------------------------------------------------------------------------\r\n");
}
#endif /* SMOLLM_MEM_PROBE */

/* Returns 1 if the answer ran to completion, 0 if it was abandoned part-way because a newer
 * question arrived over the C2C link (SMOLLM_C2C only; always 1 otherwise). */
static int answer(const char *user) {
  const int cap = (int)(sizeof(g_prompt_toks) / sizeof(g_prompt_toks[0]));
  int npt = build_turn(user, g_prompt_toks, cap);

  if (g_pos + npt + 8 >= cfg->seq_len) {   /* context exhausted: start the conversation over */
    SMOLLM_LOG("\r\n[smollm] context full (%d tokens) — resetting the conversation\r\n", g_pos);
    g_pos = 0;
    g_turn_open = 0;
    npt = build_turn(user, g_prompt_toks, cap);
  }

  /* Prefill produces no text, and every token of it streams all 143 MB of weights, so without a
   * heartbeat the console looks identical to a hung chip for the whole prompt. Print a dot per
   * token (flushed), and the per-token cost of the first one so the wait is predictable. */
  const uint64_t t0 = rdcycle64();
  float *logits = NULL;
  const int batch = SMOLLM_MAX_BATCH;
#if SMOLLM_QUIET
  /* Still print SOMETHING per pass: prefill takes ~14 s and a dead console reads as a hung chip
   * (which cost real debugging time earlier in this port). One dot is enough. */
  printf("\r\n");
#else
  printf("\r\n[smollm] reading %d prompt tokens in %d pass%s ", npt,
         (npt + batch - 1) / batch, ((npt + batch - 1) / batch == 1) ? "" : "es");
#endif
  fflush(stdout);
  for (int i = 0; i < npt; ) {
    int chunk = npt - i;
    if (chunk > batch) chunk = batch;
    logits = smollm_forward_batch(&g_prompt_toks[i], chunk, g_pos);
    g_pos += chunk;
    i += chunk;
    printf(".");
    fflush(stdout);
#if SMOLLM_C2C
    /* A long prompt is several passes of ~10 s each, so prefill has to be interruptible too —
     * otherwise a replacement question waits out the whole read of the one it replaced. */
    if (stt_bml_preempted()) {
      printf("\r\n[preempted during prefill — newer question waiting]\r\n");
      g_turn_open = 1;   /* the assistant turn was opened by build_turn; close it next time */
      return 0;
    }
#endif
  }
  const uint64_t t_prefill = rdcycle64() - t0;

  printf("\r\nSmolLM: ");
  fflush(stdout);
  int generated = 0;
  int next = sample_token(logits);
  const uint64_t t1 = rdcycle64();
  while (generated < SMOLLM_MAX_NEW_TOKENS && g_pos < cfg->seq_len) {
    if (next == smollm_tok_eos() || next == smollm_tok_bos()) break;
    emit_piece(next);
    generated++;
#if SMOLLM_STOP_AT_SENTENCE
    /* Break BEFORE the forward pass, not after: that pass would produce logits we are about to
     * throw away, and it costs a full sweep of the model (~10 s). The price is that this token is
     * absent from the KV cache — see the note in smollm_config.h. */
    if (generated >= SMOLLM_MIN_NEW_TOKENS && piece_ends_sentence(next)) break;
#endif
#if SMOLLM_C2C
    /* Checked before the forward, so a preempt saves the whole ~10 s pass rather than paying for a
     * token nobody will hear. The partial answer stays in the KV cache and build_turn closes it. */
    if (stt_bml_preempted()) {
      printf("\r\n[preempted — newer question waiting]\r\n");
      g_turn_open = 1;
      g_last_generated = generated;
      return 0;
    }
#endif
    logits = smollm_forward(next, g_pos++);
    next = sample_token(logits);
  }
  const uint64_t t_decode = rdcycle64() - t1;
  g_turn_open = 1;
  g_last_generated = generated;
  printf("\r\n");

#if SMOLLM_LOG_TIMING
  /* Integer milli-tokens/s: newlib's %f pulls in _dtoa_r, which alone can want a kilobyte of
   * stack, and this platform's stack overflows silently. */
  const uint64_t hz = g_core_hz ? g_core_hz : target_frequency;   /* measured beats configured */
  const unsigned long mtps = (t_decode && generated)
      ? (unsigned long)((uint64_t)generated * hz * 1000ull / t_decode) : 0ul;
  SMOLLM_LOG("[smollm] prompt %d tok in %lu Mcyc, generated %d tok in %lu Mcyc (%lu.%03lu tok/s)\r\n",
             npt, (unsigned long)(t_prefill / 1000000ull), generated,
             (unsigned long)(t_decode / 1000000ull), mtps / 1000ul, mtps % 1000ul);

  /* The bitstream-comparable line. Decode reads the whole model once per token and prefill once per
   * batch pass, so cycles / (passes * model bytes) is the memory cost per weight byte — the same
   * quantity `membw` reports as "seq read cyc/byte", measured on the real workload. Unlike tok/s it
   * does not move with the prompt, the answer length or the sampling temperature, so two runs of
   * different prompts on two bitstreams are still directly comparable. */
  {
    const int passes = (npt + batch - 1) / batch;
    const uint64_t pf_cpb = (passes > 0 && g_model_bytes)
        ? (t_prefill * 100ull) / ((uint64_t)passes * (uint64_t)g_model_bytes) : 0ull;
    const uint64_t dc_cpb = (generated > 0 && g_model_bytes)
        ? (t_decode * 100ull) / ((uint64_t)generated * (uint64_t)g_model_bytes) : 0ull;
    const uint64_t ms_tok = (generated > 0 && hz)
        ? (t_decode * 1000ull) / (uint64_t)generated / hz : 0ull;
    SMOLLM_LOG("[smollm] memory: prefill %lu.%02lu cyc/model-byte (%d pass%s), decode %lu.%02lu "
               "cyc/model-byte, %lu ms/token  [%lu MB model @ %lu.%02lu MHz measured]\r\n",
               (unsigned long)(pf_cpb / 100ull), (unsigned long)(pf_cpb % 100ull),
               passes, (passes == 1) ? "" : "es",
               (unsigned long)(dc_cpb / 100ull), (unsigned long)(dc_cpb % 100ull),
               (unsigned long)ms_tok, (unsigned long)((uint64_t)g_model_bytes >> 20),
               (unsigned long)(hz / 1000000ull), (unsigned long)((hz % 1000000ull) / 10000ull));
  }
  (void)t_prefill; (void)mtps;   /* both are only consumed by the log lines above */
  SMOLLM_LOG("[smollm] hart1 completed %lu row-ranges this answer%s\r\n",
             g_smollm_jobs_done,
             g_smollm_jobs_done ? "" : "   <-- the second hart did NO work");
  g_smollm_jobs_done = 0;
#endif
  return 1;
}

/* ------------------------------------------------------------------------------------------------
 * Entry
 * ---------------------------------------------------------------------------------------------- */

void app_init(void) {
  /* Sample cycles-per-mtime-tick BEFORE the PLL is touched, while the core is still on the reset
   * clock. Nothing can print yet; stash it. See clock_probe.h — on this silicon mtime is derived
   * from the core clock, so counting rdcycle against it reports 50 MHz whatever the PLL is doing,
   * and this before/after pair plus the PLL register readback is what disambiguates that from a
   * PLL that genuinely failed to engage. */
  const uint64_t cpt_ref = clock_probe_sample(mtime_now, 500u);

  /* PLL + UART0 bring-up, but ONLY on real hardware: under PLATFORM=SIMS the console is HTIF and
   * Spike models neither the PLL nor the UART, so programming them faults before a single character
   * is printed — which looks exactly like a hang. */
#if defined(TERMINAL_DEVICE_UART0)
  init_test(target_frequency);
#endif
  /* Unbuffered console, before any I/O touches the stream (setvbuf requires that). stdout is
   * block-buffered by default, so anything without a trailing flush sits in a 1 KB buffer until
   * some later flush comes along — during a minutes-long prefill that is indistinguishable from a
   * hung chip. Console traffic is a few hundred bytes per answer, so per-character writes cost
   * nothing that matters here. */
  setvbuf(stdout, NULL, _IONBF, 0);

  printf("\r\n");
  SMOLLM_LOG("[smollm] SmolLM2-135M-Instruct on bearly25, int8 Q8_0\r\n");

  /* Resolve the core clock before anything is timed: every seconds-per-token figure below scales
   * with it, and the naive rdcycle-vs-mtime measurement is wrong by the PLL ratio on this chip. */
  {
    static clock_probe_t clk;
    clock_probe_resolve(&clk, cpt_ref, clock_probe_sample(mtime_now, 500u), target_frequency);
    g_core_hz = clk.core_hz;
#if !SMOLLM_QUIET
    clock_probe_print(&clk, "[smollm]");
#endif
  }

  stack_paint();
  smollm_probe_uncore();   /* no-op unless SMOLLM_PROBE_UNCORE */
#if SMOLLM_DEBUG_GOLDEN
  smollm_set_stage_hook(golden_stage);
#endif

  const size_t model_bytes = (size_t)(g_smollm_model_end - g_smollm_model);
  const size_t tok_bytes = (size_t)(g_smollm_tokenizer_end - g_smollm_tokenizer);
  SMOLLM_LOG("[smollm] blobs: model %lu KB @ %p, tokenizer %lu KB @ %p\r\n",
             (unsigned long)(model_bytes >> 10), (const void *)g_smollm_model,
             (unsigned long)(tok_bytes >> 10), (const void *)g_smollm_tokenizer);
  g_model_bytes = model_bytes;   /* denominator of the per-answer cyc/model-byte figure */

#if SMOLLM_MEM_PROBE
  /* Before the KV cache takes the heap, and before anything else is timed. */
  mem_probe_boot(g_smollm_model, model_bytes);
#endif

  if (smollm_model_load(g_smollm_model, model_bytes) != 0) for (;;) __asm__ volatile("wfi");
  cfg = smollm_model_config();
  SMOLLM_LOG("[smollm] config dim=%d hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d "
             "vocab=%d seq_len=%d GS=%d rope_theta=%ld eps=1e-5\r\n",
             cfg->dim, cfg->hidden_dim, cfg->n_layers, cfg->n_heads, cfg->n_kv_heads,
             cfg->head_dim, cfg->vocab_size, cfg->seq_len, cfg->group_size, (long)cfg->rope_theta);
#if SMOLLM_CHECK_BLOB
  /* Prove the 143 MB actually arrived over the UART and is stable in DRAM. Without this,
   * "the weights are corrupt" and "the arithmetic is wrong" produce the same garbage on the
   * console. Timing it also measures the pure sequential read cost of the whole model. */
  {
    const uint64_t t = rdcycle64();
    const uint64_t got = smollm_fnv1a64(g_smollm_model, model_bytes);
    const uint64_t c = rdcycle64() - t;
    const unsigned long cpb = (unsigned long)((c * 100ull) / (uint64_t)model_bytes);
    SMOLLM_LOG("[smollm] blob fnv1a64 0x%016llx %s expected 0x%016llx  (%lu Mcyc to read %lu MB "
               "= %lu.%02lu cyc/byte)\r\n",
               (unsigned long long)got,
               (got == SMOLLM_MODEL_FNV1A64) ? "MATCHES" : "!!! DIFFERS FROM !!!",
               (unsigned long long)SMOLLM_MODEL_FNV1A64,
               (unsigned long)(c / 1000000ull), (unsigned long)(model_bytes >> 20),
               cpb / 100, cpb % 100);
    if (model_bytes != (size_t)SMOLLM_MODEL_BYTES)
      SMOLLM_LOG("[smollm] blob is %lu bytes, exporter wrote %lu — the ELF is stale\r\n",
                 (unsigned long)model_bytes, (unsigned long)SMOLLM_MODEL_BYTES);
  }
#endif
  if (smollm_tok_load(g_smollm_tokenizer, tok_bytes) != 0) for (;;) __asm__ volatile("wfi");
  SMOLLM_LOG("[smollm] tokenizer: vocab=%d merges=%d bos=%d eos=%d\r\n",
             smollm_tok_vocab_size(), smollm_tok_n_merges(), smollm_tok_bos(), smollm_tok_eos());

  if (smollm_model_alloc() != 0) for (;;) __asm__ volatile("wfi");
  smollm_dualcore_init();   /* no-op unless SMOLLM_DUALCORE, and self-disabling if the probe fails */
  SMOLLM_LOG("[smollm] KV cache %lu KB, %s matmul, temperature %ld/100\r\n",
             (unsigned long)((size_t)cfg->n_layers * (size_t)cfg->seq_len * (size_t)cfg->kv_dim
                             * 2u * sizeof(float) >> 10),
             smollm_uses_rvv() ? "RVV" : "scalar", (long)(SMOLLM_TEMPERATURE * 100.0f));
  g_rng ^= rdcycle64();
#if SMOLLM_BENCH_KERNELS
  smollm_bench(rdcycle64);
#endif
}

#if SMOLLM_C2C && SMOLLM_POLISH_PROMPT
/* Turn a raw ASR transcript into something that looks like a typed message.
 *
 * Citrinet emits lowercase with no punctuation — "hello how are you". To a language model that is
 * not a question, it is an UNFINISHED FRAGMENT, and the most natural continuation of a fragment is
 * more of the same fragment. That is a large part of why the answer came back as an echo. A capital
 * and a terminal mark cost nothing and put the text in the form the model saw during training.
 *
 * The question mark is chosen by looking at the first word, because guessing wrong is worse than not
 * guessing: a statement ending in '?' reads as confusion. Anything not clearly interrogative just
 * gets a full stop. */
static void polish_prompt(char *s) {
  /* Words that make an utterance a question when they LEAD it. Nothing is inferred from
   * punctuation — the ASR never produces any — so the first content word is the only signal there
   * is. Getting it wrong is asymmetric: a statement ending in '?' reads as confusion, while a
   * question ending in '.' is merely flat, so anything not clearly interrogative gets a full stop. */
  static const char *const q_words[] = {
    "how", "what", "why", "who", "when", "where", "which", "whose",
    "is", "are", "am", "was", "were", "do", "does", "did",
    "can", "could", "should", "would", "will", "have", "has", "may", "might",
  };
  /* Openers that carry no grammatical weight and would otherwise hide the real first word.
   * "hello how are you" is a question, but it does not START with an interrogative — and that
   * phrasing is far more natural out loud than in typing, which is exactly the input this gets. */
  static const char *const skip_words[] = {
    "hello", "hi", "hey", "ok", "okay", "so", "well", "um", "uh", "please", "marvin",
  };
  size_t n = strlen(s);
  size_t at = 0;
  int interrogative = 0;

  if (n == 0) return;

  /* Step over at most two leading fillers, then test the word that follows. Two covers "ok so ..."
   * and "hey marvin ..."; more than that and the guess is not worth trusting anyway. */
  for (int hop = 0; hop < 2; ++hop) {
    size_t w = 0;
    while (((at + w) < n) && (s[at + w] != ' ')) w++;
    int is_filler = 0;
    for (unsigned i = 0; i < (sizeof(skip_words) / sizeof(skip_words[0])); ++i) {
      if ((strlen(skip_words[i]) == w) && (strncmp(s + at, skip_words[i], w) == 0)) {
        is_filler = 1;
        break;
      }
    }
    if (!is_filler) break;
    at += w;
    while ((at < n) && (s[at] == ' ')) at++;   /* to the start of the next word */
    if (at >= n) { at = 0; break; }            /* filler was the whole utterance: judge it as-is */
  }

  /* First content word, lowercase already (Citrinet emits lowercase). */
  {
    size_t w = 0;
    while (((at + w) < n) && (s[at + w] != ' ')) w++;
    for (unsigned i = 0; i < (sizeof(q_words) / sizeof(q_words[0])); ++i) {
      if ((strlen(q_words[i]) == w) && (strncmp(s + at, q_words[i], w) == 0)) {
        interrogative = 1;
        break;
      }
    }
  }

  if ((s[0] >= 'a') && (s[0] <= 'z')) s[0] = (char)(s[0] - 'a' + 'A');

  /* Only add a mark if there is not one already, and only if there is room. */
  if ((s[n - 1] != '.') && (s[n - 1] != '?') && (s[n - 1] != '!') &&
      ((n + 2) < (size_t)SMOLLM_PROMPT_MAX)) {
    s[n] = interrogative ? '?' : '.';
    s[n + 1] = '\0';
  }
}

#endif /* SMOLLM_C2C */

void app_main(void) {
#if SMOLLM_C2C
  /* Voice assistant: the prompt arrives from DSP 25 over the C2C link (mic -> Citrinet transcript),
   * the answer goes back the same way. The console still shows the whole exchange.
   *
   * The link is brought up HERE, not in app_init, for two reasons that are both hardware rules:
   * cross-link writes must not happen during boot, and the readiness announcement is precisely what
   * unblocks the DSP — so it must not be made until the model and KV cache are actually up. */
  static char line[SMOLLM_PROMPT_MAX];
  int first = 1;
  stt_bml_link_init();
  for (;;) {
    const uint32_t idx = stt_bml_wait_prompt(line, sizeof(line));
    if (line[0] == '\0') {
      /* An empty transcript should have been filtered on the DSP side; ack it rather than spend
       * minutes of decode on nothing, and keep the turn-taking sequence intact. */
      stt_bml_capture_reset();
      stt_bml_answer_done(idx, 0u);
      continue;
    }
#if SMOLLM_MULTI_TURN
    /* A restarted DSP is a new session. Without this the model keeps answering into the context of a
     * conversation whose other half no longer exists, which looks like the model misbehaving. */
    if (stt_bml_take_restart()) {
      printf("[smollm] DSP restarted — starting a fresh conversation\r\n");
      g_pos = 0;
      g_turn_open = 0;
    }
#else
    g_pos = 0;
    g_turn_open = 0;
#endif
#if SMOLLM_POLISH_PROMPT
    polish_prompt(line);   /* ASR text -> something that reads as a finished message */
#endif
    printf("\r\nYou (spoken): %s\r\n", line);
    stt_bml_capture_reset();
    if (answer(line)) {
      stt_bml_answer_done(idx, (uint32_t)g_last_generated);
    }
    /* else: abandoned for a newer question. Deliberately NOT acked — the DSP stopped waiting on
     * this one the moment it published the replacement, and wait_prompt() will pick that up on the
     * next pass. Acking here would tell the DSP an answer it never asked for had arrived. */
    if (first) { stack_report(); first = 0; }   /* the link module's buffers are static, but check */
  }
#elif defined(SMOLLM_PROMPT)
  /* Autorun (Spike, or an unattended silicon run): one prompt, then halt. */
  SMOLLM_LOG("[smollm] prompt: \"%s\"\r\n", SMOLLM_PROMPT);
  answer(SMOLLM_PROMPT);
  stack_report();
  SMOLLM_LOG("[smollm] done.\r\n");
  for (;;) __asm__ volatile("wfi");
#else
  static char line[SMOLLM_PROMPT_MAX];
  int first = 1;
  SMOLLM_LOG("[smollm] ready — type a question and press Enter.\r\n");
  for (;;) {
    read_line("\r\nYou: ", line, sizeof(line));
    if (line[0] == '\0') continue;
#if !SMOLLM_MULTI_TURN
    g_pos = 0;
    g_turn_open = 0;
#endif
    answer(line);   /* prints its own "SmolLM: " once the prompt has been read in */
    if (first) { stack_report(); first = 0; }   /* no-op output under SMOLLM_QUIET */
  }
#endif
}

int main(void) {
  app_init();
  app_main();
  return 0;
}

/* hart 1..N park here unless src/dualcore.c's STRONG __main takes over (SMOLLM_DUALCORE=1). */
void __attribute__((weak, noreturn)) __main(void) {
  for (;;) __asm__ volatile("wfi");
}
