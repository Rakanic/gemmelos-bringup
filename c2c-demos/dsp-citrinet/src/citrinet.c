/*
 * citrinet.c — Citrinet-256 CTC speech-to-text engine for DSP 25. SCALAR reference implementation.
 *
 * This is the executable twin of dsp25-tests/citrinet-test/scripts/dump_citrinet_reference.py:
 * that script parses the same "cn01" blob, runs the same maths in numpy, and emits the golden in
 * citrinet_reference.h. Anything that diverges here shows up as a per-stage sum mismatch, which
 * localizes the bug to a block rather than to "the transcript is wrong".
 *
 * Layout convention: every activation is CHANNEL-MAJOR, x[c * n_frames + t]. The depthwise conv is
 * then contiguous along t, and the pointwise becomes, for each (out,in) pair, a scalar-times-vector
 * accumulate over t — which is exactly the shape an RVV kernel wants when we vectorize later.
 *
 * See include/citrinet.h for the architecture summary and why this model was chosen over Moonshine.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "citrinet.h"
#include "citrinet_vocab.h"
#include "citrinet_reference.h"

/* ---- build-time knobs (all overridable from CMake) --------------------------------------------- */
#ifndef CN_PROFILE
#define CN_PROFILE 0
#endif
/* Only used to turn measured cycles into ms in the validate report. CMake keeps this in step with
 * DSP_CITRINET_TARGET_FREQ_HZ so the two can't drift. */
#ifndef CN_TARGET_FREQ_HZ
#define CN_TARGET_FREQ_HZ 750000000ULL
#endif

#ifndef CN_DEBUG_GEOM
#define CN_DEBUG_GEOM 0
#endif
/* Print a fingerprint after every op inside one block (dw / pw / relu / se / res). -1 = off.
 * Used to localize WHERE a block diverges when silicon disagrees with the golden. */
#ifndef CN_TRACE_BLOCK
#define CN_TRACE_BLOCK -1
#endif
/* Run the encoder twice on identical input and diff the per-block fingerprints. Splits "the
 * hardware is flaky" from "the hardware is deterministically different". */
#ifndef CN_USE_RVV
#define CN_USE_RVV 0
#endif
#if CN_USE_RVV
#include <riscv_vector.h>   /* also included at the kernels below; the header is guarded */
#endif

/* Fill every scratch buffer before use: 0 = leave as malloc returned it, 1 = zero, 2 = NaN poison.
 * Spike hands out ZEROED fresh memory while real DRAM holds the previous run's leftovers, so an
 * uninitialized read passes on Spike forever and fails intermittently on silicon — including
 * "passing" whenever the leftover happens to be the correct value from a prior identical run.
 * Poisoning with NaN makes any such read propagate a NaN that the stage fingerprints cannot miss. */
/* After each block, fingerprint its output, then stream enough real weight data to evict it from
 * the 256 KB cache, then fingerprint the SAME buffer again. If the two disagree, activations are
 * not surviving eviction to DRAM — which the sequential MEMTEST cannot detect, because the model
 * interleaves a large read stream with its writes and MEMTEST does not. */
#ifndef CN_EVICT_CHECK
#define CN_EVICT_CHECK 0
#endif

#ifndef CN_POISON
#define CN_POISON 0
#endif

#ifndef CN_DETERMINISM_CHECK
#define CN_DETERMINISM_CHECK 0
#endif

/* Upper bound on the Q8_0 group size. Sizes the small dequant staging buffer in the RVV kernels and
 * is enforced by the header sanity check in cn_model_load. This model uses gs=32. */
#define CN_GS_MAX 128

/* Max input channels for a pointwise, sizing the dequantized-weight staging buffer. */
#define CN_CI_MAX 640

#define CN_FFT_MAX 1024      /* upper bound on n_fft; sizes the FFT scratch + twiddles */
#define CN_MAGIC 0x636E3031u    /* "cn01" — must match MAGIC in export_citrinet.py */
#define CN_HEADER_BYTES 256
#define CN_DESC_I32 12

/* ---- profiling --------------------------------------------------------------------------------- */
static inline uint64_t cn_cy(void) { uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }

#if CN_PROFILE
static uint64_t g_cyc_fe, g_cyc_dw, g_cyc_pw, g_cyc_se, g_cyc_res, g_cyc_dec;
#define prof_cy() cn_cy()
#define PROF_T0() uint64_t _t0 = prof_cy()
#define PROF_ADD(acc) do { (acc) += prof_cy() - _t0; } while (0)
#else
#define PROF_T0() do { } while (0)
#define PROF_ADD(acc) do { } while (0)
#endif

/* Report the vector unit's ACTUAL geometry. The kernels are designed around "VLEN=256, so e32m4
 * gives vl=32 lanes and 20 of 32 vector registers are live" — but the build uses -march=rv64gcv_zfh,
 * which implies only zvl128b, and nothing has ever checked the hardware. vsetvlmax is the ground
 * truth and costs two instructions, so measure it instead of assuming: if vlmax_e32m1 is 4 rather
 * than 8, VLEN is 128 and every lane-count assumption in the RVV kernels (and the cycles/MAC target)
 * is off by 2x. */
void cn_report_vlen(void) {
#if CN_USE_RVV
  const unsigned long m1 = (unsigned long)__riscv_vsetvlmax_e32m1();
  const unsigned long m4 = (unsigned long)__riscv_vsetvlmax_e32m4();
  printf("[citrinet] rvv: vlmax e32m1=%lu e32m4=%lu -> VLEN=%lu bits (built -march=rv64gcv_zfh = zvl128b)\n",
         m1, m4, m1 * 32u);
#else
  printf("[citrinet] rvv: disabled (scalar kernels)\n");
#endif
}

void cn_profile_reset(void) {
#if CN_PROFILE
  g_cyc_fe = g_cyc_dw = g_cyc_pw = g_cyc_se = g_cyc_res = g_cyc_dec = 0;
#endif
}

void cn_profile_report(void) {
#if CN_PROFILE
  uint64_t tot = g_cyc_fe + g_cyc_dw + g_cyc_pw + g_cyc_se + g_cyc_res + g_cyc_dec;
  printf("[citrinet] PROFILE cycles: frontend=%llu dw=%llu pw=%llu se=%llu res=%llu dec=%llu total=%llu\n",
         (unsigned long long)g_cyc_fe, (unsigned long long)g_cyc_dw, (unsigned long long)g_cyc_pw,
         (unsigned long long)g_cyc_se, (unsigned long long)g_cyc_res, (unsigned long long)g_cyc_dec,
         (unsigned long long)tot);
#endif
}

/* ---- trap handler + secondary-hart park -------------------------------------------------------- *
 * glossy ships a WEAK no-op trap_handler that returns to (and so re-executes) the faulting
 * instruction forever, which makes every fault look like a hang. Overriding it to print and halt
 * was one of the two keepers from the moonshine dual-core work. __main parks harts != 0 so they
 * don't fall through crt0's `tail exit` into newlib exit() and fight hart 0 for the UART. */
uintptr_t trap_handler(uintptr_t m_epc, uintptr_t m_cause, uintptr_t m_tval, uintptr_t regs[32]) {
  (void)regs;
  uint64_t hid; __asm__ volatile("csrr %0, mhartid" : "=r"(hid));
  static const char *const why[] = { "instr-misalign", "instr-access", "illegal-instr", "breakpoint",
                                     "load-misalign", "load-access", "store-misalign", "store-access" };
  printf("\n[citrinet] TRAP hart=%lu mcause=%lu (%s) mepc=0x%lx mtval=0x%lx\n",
         (unsigned long)hid, (unsigned long)m_cause,
         (m_cause < 8u) ? why[m_cause] : "other/interrupt",
         (unsigned long)m_epc, (unsigned long)m_tval);
  for (;;) __asm__ volatile("wfi");
  return m_epc;
}

#if !CN_DUALCORE
void __main(void) { for (;;) __asm__ volatile("wfi"); }
#else
/* hthread.c supplies __main for secondary harts (its scheduler loop). Defining our own here would
 * park hart 1 in wfi forever and every hthread_join would hang. */
#endif

/* Tagged so an OOM says WHICH allocation died. A wrapped-negative size here means a geometry
 * computation went wrong upstream, not that the heap is small — print it as such. */
#if CN_POISON
static void cn_fill(void *p, size_t bytes) {
#if CN_POISON == 2
  const uint32_t pat = 0x7FC00000u;   /* quiet NaN */
#else
  const uint32_t pat = 0u;
#endif
  uint32_t *w = (uint32_t *)p;
  for (size_t i = 0; i < bytes / 4u; i++) w[i] = pat;
}
#define CN_FILL(p, bytes) cn_fill((p), (bytes))
#else
#define CN_FILL(p, bytes) do { } while (0)
#endif

static void *xmalloc_tag(size_t n, const char *tag) {
  if (n > (256u << 20)) {
    printf("[citrinet] BUG: %s asked for %lu bytes (0x%lx) — geometry overflow, not a heap problem\n",
           tag, (unsigned long)n, (unsigned long)n);
    exit(1);
  }
  void *p = malloc(n);
  if (!p) {
    printf("[citrinet] OOM: %s malloc(%lu) failed\n", tag, (unsigned long)n);
    exit(1);
  }
  return p;
}
#define xmalloc(n, tag) xmalloc_tag((n), (tag))


/* ---- DMA strided-gather probe -------------------------------------------------------------------
 * Proves the ONE transfer the DMA optimisation would rely on, before any of it is built:
 * gather a strided activation tile out of DRAM into the 64 KB scratchpad at 0x08000000 in a single
 * transaction, and verify it byte-for-byte against a scalar gather.
 *
 * Geometry is citrinet's, not a toy: Ci packets of CN_TILE_T*4 bytes, source stride T*4, destination
 * contiguous. The risky part is that the source stride (600 B at T=150) is NOT a multiple of the
 * 128 B packet size, so if the engine requires packet-aligned reads this fails here rather than
 * silently corrupting activations later.
 *
 * Why this is worth doing at all: software packing of this same tile LOST (4,380 -> 4,641 ms) for two
 * reasons — it burned core cycles on the strided copy, and its 80 KB buffer evicted the cache badly
 * enough to slow the untouched depthwise by 25%. A DMA gather has neither problem: no core cycles,
 * and the SRAM is separate memory that does not compete for cache. That is the first idea in a while
 * aimed at what actually limits this machine.
 *
 * History: whisper (2026-08-04) and moonshine (2026-08-06) both recorded "DMA to this SRAM hangs —
 * the SRAM is cached and the DMA is not coherent". dsp25-bmarks/dma-bmarks now PASSES the
 * DRAM<->scratchpad cases with verification, so that conclusion no longer holds (both were measured
 * with the stack in the partially-backed high region, and with an unbounded DMA wait). */
#if CN_DMA_PROBE
#include "hal_dma.h"

#define CN_SPAD_BASE 0x08000000UL

void cn_dma_probe(void) {
  /* Settles ONE question: after a DMA lands new data in the scratchpad, does the core read it, or a
   * stale cached copy? dma-bmarks cannot answer this — its COLD/WARM cases run a 256 KB eviction
   * walk before reading, and HOT_REPEAT re-sends identical data, so a stale cache passes too. Every
   * one of its PASSes is consistent with a non-coherent cached scratchpad.
   *
   * So: contiguous transfer in the PROVEN configuration (logw=6 / inc_r=inc_w=64, as used by
   * dma-bmarks and dsp-whisper), then read straight back with NO eviction. Runs twice with different
   * payloads — pass 2 is the one that matters, because by then the scratchpad lines are cached from
   * pass 1, which is exactly the situation weight staging creates on every chunk. */
  const size_t bytes = 16u << 10;
  const size_t nw = bytes / 4u;
  uint32_t *src = (uint32_t *)xmalloc(bytes, "dma.src");

  for (int pass = 0; pass < 2; pass++) {
    for (size_t i = 0; i < nw; i++) src[i] = (uint32_t)(i * 2654435761u) ^ (pass ? 0xFFFFFFFFu : 0u);
    __asm__ volatile("fence rw, rw" ::: "memory");

    dma_transaction_t tx;
    tx.core = 0; tx.transaction_id = (uint16_t)(pass + 1); tx.transaction_priority = 1;
    tx.peripheral_id = 0;
    tx.addr_r = (uint64_t)(uintptr_t)src;
    tx.addr_w = (uint64_t)CN_SPAD_BASE;
    tx.inc_r = 64; tx.inc_w = 64;
    tx.len = (uint16_t)(bytes / 64u);
    tx.logw = 6;
    tx.do_interrupt = false; tx.do_address_gate = false;

    const uint64_t t0 = cn_cy();
    if (!set_DMA_C(0, tx, true)) { printf("[citrinet] DMA probe: set_DMA_C rejected\n"); free(src); return; }
    start_DMA(0, tx.transaction_id, (void *)0);
    dma_wait_till_inactive(CN_DMA_WAIT_SPIN);
    __asm__ volatile("fence rw, rw" ::: "memory");
    const uint64_t t1 = cn_cy();
    dma_reset();

    const volatile uint32_t *dst = (const volatile uint32_t *)CN_SPAD_BASE;
    unsigned long bad = 0; long first = -1;
    for (size_t i = 0; i < nw; i++)
      if (dst[i] != src[i]) { if (first < 0) first = (long)i; bad++; }
    printf("[citrinet] DMA probe pass%d (contiguous, logw=6, NO evict): %lu B in %lu cycles, "
           "%lu/%lu words bad%s\n", pass, (unsigned long)bytes, (unsigned long)(t1 - t0),
           bad, (unsigned long)nw,
           bad ? (pass ? "  <-- STALE CACHE: scratchpad is not coherent with DMA" : "  <-- transfer broken")
               : "  (verified)");
    if (bad && first >= 0)
      printf("[citrinet] DMA probe pass%d: first bad word %ld: got 0x%08lx want 0x%08lx\n",
             pass, first, (unsigned long)dst[first], (unsigned long)src[first]);
  }
  free(src);
}
#else
void cn_dma_probe(void) {}
#endif /* CN_DMA_PROBE */

/* ---- blob cursor + model load ------------------------------------------------------------------ */
typedef struct { const unsigned char *p; } cur_t;

static const float *take_f32(cur_t *c, size_t n) {
  const float *r = (const float *)c->p;
  c->p += n * sizeof(float);
  return r;
}

/* Q8_0 when (in % gs) == 0, else fp32 — the same predicate export_citrinet.py used. */
static cn_mat_t take_mat(cur_t *c, int out, int in, int gs) {
  cn_mat_t w;
  w.out = out; w.in = in; w.q = 0; w.s = 0; w.f = 0;
  if ((in % gs) == 0) {
    w.q = (const int8_t *)c->p; c->p += (size_t)out * in;
    w.s = (const float *)c->p;  c->p += ((size_t)out * in / gs) * sizeof(float);
  } else {
    w.f = take_f32(c, (size_t)out * in);
  }
  return w;
}

/* FNV-1a over 64-bit words, matching fnv1a64_words() in export_citrinet.py. Covers the ENTIRE blob
 * including the header, skipping only words 9 and 10 (blob_bytes and the checksum itself). The blob
 * is padded to a multiple of 8 by the exporter, so there is no tail to handle. Word-wise (not
 * byte-wise) because this runs over ~11.8 MB at boot.
 *
 * v2 hashed only [256, end). A silicon run then came back with a ZEROED header — n_fft/win/hop/
 * bins/blank/max_k/max_c all 0, i.e. blob bytes 32..71 gone — while still printing `blob ok`,
 * because the header sat outside the hash. The most destructive 256 bytes were the unprotected
 * ones. Hence v3. */
#define CN_CKSUM_WORD0 9u   /* bytes 72..79 = blob_bytes */
#define CN_CKSUM_WORD1 10u  /* bytes 80..87 = checksum   */

static uint64_t cn_fnv1a64(const void *p, size_t n_words) {
  const uint64_t *w = (const uint64_t *)p;
  uint64_t h = 0xCBF29CE484222325ull;
  const size_t head = (n_words < CN_CKSUM_WORD0) ? n_words : CN_CKSUM_WORD0;
  for (size_t i = 0; i < head; i++) h = (h ^ w[i]) * 0x100000001B3ull;
  for (size_t i = CN_CKSUM_WORD1 + 1u; i < n_words; i++) h = (h ^ w[i]) * 0x100000001B3ull;
  return h;
}

/* Byte-wise FNV-1a over the .text image. The blob checksum proves the WEIGHTS arrived intact; this
 * proves the CODE did. A silicon run once produced deterministic-but-wrong arithmetic from blk1
 * onward with a passing blob checksum, and the next flash of the same source was exact — which is
 * corrupted .text, the only sizeable region still unverified. .text is ~55 KB, so this is free.
 * There is no baked-in expected value (it would have to live inside the region it hashes); compare
 * the printed value against the host-computed one, or simply across two flashes. */
uint64_t cn_fnv1a64_bytes(const void *p, size_t n) {
  const unsigned char *b = (const unsigned char *)p;
  uint64_t h = 0xCBF29CE484222325ull;
  for (size_t i = 0; i < n; i++) h = (h ^ (uint64_t)b[i]) * 0x100000001B3ull;
  return h;
}

static uint64_t cn_text_checksum(void) {
  extern char __text_start[], __text_end[];
  return cn_fnv1a64_bytes(__text_start, (size_t)((uintptr_t)__text_end - (uintptr_t)__text_start));
}

/* Deterministic, position-dependent pattern: any mismatch tells us the address AND lets us see
 * whether the damage is single-bit, a whole word, or a shifted neighbour. */
static inline uint32_t cn_pat(size_t i) { return (uint32_t)(i * 2654435761u) ^ 0xA5A5A5A5u; }

unsigned long cn_memtest(size_t bytes) {
  const size_t n = bytes / 4u;
  uint32_t *p = (uint32_t *)xmalloc(bytes, "memtest");
  /* Writing the whole span pushes the early part out of the 256 KB cache and into DRAM, so the
   * read-back below genuinely round-trips through memory rather than hitting cache. */
  for (size_t i = 0; i < n; i++) p[i] = cn_pat(i);
  unsigned long bad = 0;
  for (size_t i = 0; i < n; i++) {
    const uint32_t want = cn_pat(i), got = p[i];
    if (got != want) {
      if (bad < 8u)
        printf("[citrinet] MEMTEST mismatch @%p (word %lu): want 0x%08lx got 0x%08lx xor 0x%08lx\n",
               (void *)&p[i], (unsigned long)i, (unsigned long)want, (unsigned long)got,
               (unsigned long)(want ^ got));
      bad++;
    }
  }
  printf("[citrinet] MEMTEST %lu MB: %lu of %lu words bad%s\n",
         (unsigned long)(bytes >> 20), bad, (unsigned long)n,
         bad ? "  <-- DRAM WRITE/READ-BACK IS UNRELIABLE" : "  (memory ok)");
  free(p);
  return bad;
}

/* ---- stack guard --------------------------------------------------------------------------------
 * crt0 gives hart h the range [__stack_start + h*__stack_size, +__stack_size). A stack overflow on
 * this platform is SILENT and INTERMITTENT rather than a fault: the overflow runs off the bottom of
 * the range into address space that is outside the DRAM the SoC actually backs, so stores may not
 * read back — which corrupts only the automatic variables, while anything the compiler kept in a
 * register stays correct. That asymmetry is what made the original bug so hard to see (a header
 * parsed from stack locals came back as garbage in the SAME function whose register-held checksum
 * over those same bytes matched the host).
 *
 * Under Spike the whole region is real zeroed memory, so an overflow there is invisible. That is
 * exactly the "passes in simulation, flaky on silicon" shape, so the check has to be on-chip and
 * cheap enough to leave on: paint at boot, scan once at the end. */
#define CN_STACK_PAT 0x5A5A5A5Au
#define CN_STACK_SLACK 512u     /* bytes below sp left unpainted: the live frame + call overhead */

static inline uintptr_t cn_sp(void) { uintptr_t s; __asm__ volatile("mv %0, sp" : "=r"(s)); return s; }

static uintptr_t g_stack_lo, g_stack_hi;

void cn_stack_paint(void) {
  extern char __stack_start[], __stack_size[];   /* __stack_size is an ABSOLUTE symbol: value = size */
  uint64_t hid; __asm__ volatile("csrr %0, mhartid" : "=r"(hid));
  const uintptr_t size = (uintptr_t)__stack_size;
  const uintptr_t lo = (uintptr_t)__stack_start + (uintptr_t)hid * size;
  const uintptr_t hi = lo + size, sp = cn_sp();
  g_stack_lo = lo; g_stack_hi = hi;
  printf("[citrinet] stack: hart%lu [0x%lx, 0x%lx) = %lu KB, sp=0x%lx\n",
         (unsigned long)hid, (unsigned long)lo, (unsigned long)hi,
         (unsigned long)(size >> 10), (unsigned long)sp);
  if (sp <= lo || sp > hi) {           /* sp already outside its own range — nothing safe to paint */
    printf("[citrinet] STACK: sp is OUTSIDE the linker's stack range — check __stack_start/_sp\n");
    g_stack_lo = g_stack_hi = 0;
    return;
  }
  for (uintptr_t a = lo; a + 4u <= sp - CN_STACK_SLACK; a += 4u) *(volatile uint32_t *)a = CN_STACK_PAT;
}

void cn_stack_report(void) {
  if (!g_stack_hi) return;
  const uintptr_t lo = g_stack_lo, hi = g_stack_hi;
  uintptr_t deepest = hi;
  unsigned long touched = 0;
  for (uintptr_t a = lo; a + 4u <= hi; a += 4u)
    if (*(volatile uint32_t *)a != CN_STACK_PAT) { if (deepest == hi) deepest = a; touched++; }
  const unsigned long used = (unsigned long)(hi - deepest), size = (unsigned long)(hi - lo);
  const unsigned long span = used / 4u;
  /* Real stack usage disturbs nearly every word between the deepest mark and the top. If only a
   * scattering of words in that span changed, something wrote INTO the stack region from outside
   * (a stray pointer) and `used` overstates the true depth. The two need different responses —
   * "raise the stack" vs "find the bad write" — so say which one this is rather than reporting a
   * high-water mark that might not be one. */
  const int sparse = (span > 64u) && (touched * 4u < span);
  /* The bottom word being consumed means the mark reached the very first slot, i.e. the stack was
   * (at least) exactly full — treat it as an overflow, because anything past it wrote outside the
   * range and would not have been recorded here at all. */
  const int overflow = (deepest <= lo);
  printf("[citrinet] stack: high-water %lu of %lu bytes (%lu%%), %lu/%lu words disturbed%s%s\n",
         used, size, size ? (used * 100u / size) : 0u, touched, span,
         sparse ? "  <-- SPARSE: a stray write into the stack region, not real depth" : "",
         overflow ? "  <-- STACK OVERFLOW: locals below the stack are unreliable on silicon" : "");
}

int cn_blob_recheck(const void *blob, size_t blob_bytes) {
  const unsigned char *b = (const unsigned char *)blob;
  uint64_t hb[2];
  memcpy(hb, b + 8 + 11 * 4 + 12 + 8, sizeof(hb));
  const uint64_t got = cn_fnv1a64(b, blob_bytes / 8u);
  if (got == hb[1]) { printf("[citrinet] blob RE-CHECK ok (still 0x%016lx)\n", (unsigned long)got); return 1; }
  printf("[citrinet] blob RE-CHECK FAILED: now 0x%016lx, was 0x%016lx\n",
         (unsigned long)got, (unsigned long)hb[1]);
  printf("[citrinet] the weights CHANGED while the program ran — the model was intact at boot.\n"
         "[citrinet] That is memory instability, not a software bug. Try a lower PLL frequency.\n");
  return 0;
}

int cn_model_load(const void *blob, size_t blob_bytes, cn_model_t *m) {
  const unsigned char *b = (const unsigned char *)blob;

  /* ---- INTEGRITY FIRST, PARSE SECOND. This order is load-bearing. -------------------------------
   * uart_tsi writes DRAM through a back door that does not invalidate the core's cache, so the
   * FIRST read of a freshly-loaded line can return whatever was there before. Observed on silicon:
   * a header parse returned n_fft=1216 / blank=0x80B7A400 / max_c=0 while the whole-blob checksum
   * over the SAME bytes, run moments later, matched the host exactly — same addresses, two reads,
   * two answers, reproducible. The 11.8 MB streaming hash below is itself the eviction pass (the
   * cache is 256 KB), so nothing is parsed until after it has run. Same family as the documented
   * "the cache-controller flush register does not evict" quirk in CLAUDE.md.
   *
   * blob_bytes comes from the LINKER, not from the blob: the value that bounds the integrity check
   * must not itself be read from the region whose integrity is in question. */
  if (blob_bytes <= CN_HEADER_BYTES || (blob_bytes & 7u)) {
    printf("[citrinet] bad blob_bytes=%lu (from linker symbols)\n", (unsigned long)blob_bytes);
    return -5;
  }
  const uint64_t got = cn_fnv1a64(b, blob_bytes / 8u);

  uint64_t hb[2];
  memcpy(hb, b + 8 + 11 * 4 + 12 + 8, sizeof(hb));   /* blob_bytes, checksum — read AFTER the hash */
  const uint64_t hdr_bytes = hb[0], want = hb[1];
  if (hdr_bytes != (uint64_t)blob_bytes) {
    printf("[citrinet] blob_bytes disagree: header says %lu, linker says %lu\n",
           (unsigned long)hdr_bytes, (unsigned long)blob_bytes);
    return -5;
  }
  if (got != want) {
    printf("[citrinet] BLOB CHECKSUM MISMATCH over %lu bytes: got 0x%016lx want 0x%016lx\n",
           (unsigned long)blob_bytes, (unsigned long)got, (unsigned long)want);
    printf("[citrinet] the model did not load intact — RE-FLASH the ELF (this is a transfer fault,\n"
           "[citrinet] not an engine bug). Everything downstream would be garbage.\n");
    return -6;
  }
  printf("[citrinet] blob ok: %lu bytes, fnv1a64=0x%016lx (header included)\n",
         (unsigned long)blob_bytes, (unsigned long)got);

  /* ---- now it is safe to read fields ----------------------------------------------------------- */
  uint32_t magic; memcpy(&magic, b, 4);
  if (magic != CN_MAGIC) return -1;
  int version; memcpy(&version, b + 4, 4);
  if (version < 3) { printf("[citrinet] blob v%d is too old — re-run export_citrinet.py\n", version); return -4; }

  int hdr[13];
  memcpy(hdr, b + 8, sizeof(hdr));           /* after magic(4) + version(4) */
  m->n_blocks = hdr[0]; m->n_mels = hdr[1]; m->feat_out = hdr[2]; m->n_classes = hdr[3];
  m->gs = hdr[4]; m->sample_rate = hdr[5]; m->n_fft = hdr[6]; m->win_length = hdr[7];
  m->hop_length = hdr[8]; m->n_bins = hdr[9]; m->blank = hdr[10];
  float fh[3];
  memcpy(fh, b + 8 + 11 * 4, sizeof(fh));
  m->log_guard = fh[0]; m->norm_eps = fh[1]; m->preemph = fh[2];
  int th[2];
  memcpy(th, b + 8 + 11 * 4 + 12, sizeof(th));
  m->max_k = th[0]; m->max_c = th[1];
  {
    extern char __text_start[], __text_end[];
    printf("[citrinet] text: %lu bytes, fnv1a64=0x%016lx  <- must match across flashes\n",
           (unsigned long)((uintptr_t)__text_end - (uintptr_t)__text_start),
           (unsigned long)cn_text_checksum());
  }

  /* Header sanity, independent of the checksum. A zeroed header previously slipped through and
   * surfaced as "audio too short" (n_fft=0 made cn_num_frames divide by zero, which on RISC-V
   * quietly returns -1 rather than trapping). Fail loudly and say what is wrong instead. */
  if (m->n_blocks <= 0 || m->n_blocks > CN_MAX_BLOCKS ||
      m->n_mels <= 0 || m->n_mels > 1024 || m->feat_out <= 0 || m->feat_out > 4096 ||
      m->n_classes <= 1 || m->n_classes > 65536 || m->gs <= 0 || m->gs > CN_GS_MAX || (m->gs & (m->gs - 1)) != 0 ||
      m->sample_rate <= 0 || m->n_fft <= 0 || (m->n_fft & (m->n_fft - 1)) != 0 ||
      m->n_fft > CN_FFT_MAX || m->win_length <= 0 || m->win_length > m->n_fft ||
      m->hop_length <= 0 || m->hop_length > m->n_fft ||
      m->n_bins != m->n_fft / 2 + 1 || m->blank != m->n_classes - 1 ||
      m->max_k <= 0 || m->max_k > 256 || m->max_c <= 0 || m->max_c > 4096) {
    printf("[citrinet] BAD HEADER: blocks=%d n_mels=%d feat_out=%d classes=%d gs=%d sr=%d "
           "n_fft=%d win=%d hop=%d bins=%d blank=%d max_k=%d max_c=%d\n",
           m->n_blocks, m->n_mels, m->feat_out, m->n_classes, m->gs, m->sample_rate,
           m->n_fft, m->win_length, m->hop_length, m->n_bins, m->blank, m->max_k, m->max_c);
    return -7;
  }

#if CN_DEBUG_GEOM
  /* Between the `text:` line and the `model:` line there is no loop and no allocation — only
   * integer loads from the parsed header — so a stop in that window means the core stalled at the
   * bus (this chip's documented response to a bad access is a silent stall, not a trap) rather than
   * spinning in this code. These markers say which side of the window it died on. */
  printf("[citrinet]   hdr ok: blocks=%d n_mels=%d feat=%d classes=%d gs=%d sr=%d n_fft=%d win=%d "
         "hop=%d bins=%d blank=%d max_k=%d max_c=%d\n",
         m->n_blocks, m->n_mels, m->feat_out, m->n_classes, m->gs, m->sample_rate, m->n_fft,
         m->win_length, m->hop_length, m->n_bins, m->blank, m->max_k, m->max_c);
#endif

  /* descriptor table, then payload */
  const unsigned char *d = b + CN_HEADER_BYTES;
  cur_t c = { d + (size_t)m->n_blocks * CN_DESC_I32 * 4 };
  const int gs = m->gs;

  m->hann = take_f32(&c, m->win_length);
  m->fb = take_f32(&c, (size_t)m->n_mels * m->n_bins);

  for (int i = 0; i < m->n_blocks; i++) {
    int f[CN_DESC_I32];
    memcpy(f, d + (size_t)i * CN_DESC_I32 * 4, sizeof(f));
    cn_block_t *B = &m->blk[i];
    B->c_in = f[0]; B->c_out = f[1]; B->repeat = f[2]; B->k = f[3]; B->stride = f[4];
    B->stride_last = f[5]; B->dilation = f[6]; B->has_se = f[7]; B->se_hidden = f[8];
    B->has_res = f[9]; B->pad = f[10];
    /* Range-check every descriptor field. Without this a single bad word propagates into the
     * encoder's buffer-size pre-pass and surfaces as a nonsense malloc instead of a bad blob. */
    if (B->repeat <= 0 || B->repeat > CN_MAX_SUB) return -3;
    if (B->c_in <= 0 || B->c_in > 4096 || B->c_out <= 0 || B->c_out > 4096 ||
        B->k <= 0 || B->k > 256 || B->stride < 1 || B->stride > 8 ||
        B->dilation < 1 || B->dilation > 8 || B->pad < 0 || B->pad > 512 ||
        B->se_hidden < 0 || B->se_hidden > B->c_out) {
      printf("[citrinet] bad descriptor blk%d: c_in=%d c_out=%d rep=%d k=%d s=%d sl=%d dil=%d "
             "se=%d seh=%d res=%d pad=%d\n", i, B->c_in, B->c_out, B->repeat, B->k, B->stride,
             B->stride_last, B->dilation, B->has_se, B->se_hidden, B->has_res, B->pad);
      return -(10 + i);
    }

    for (int j = 0; j < B->repeat; j++) {
      int cin = (j == 0) ? B->c_in : B->c_out;
      B->sub[j].dw = take_f32(&c, (size_t)cin * B->k);
      B->sub[j].pw = take_mat(&c, B->c_out, cin, gs);
      B->sub[j].bias = take_f32(&c, B->c_out);
    }
    if (B->has_se) {
      B->se1 = take_mat(&c, B->se_hidden, B->c_out, gs);
      B->se2 = take_mat(&c, B->c_out, B->se_hidden, gs);
    }
    if (B->has_res) {
      B->res = take_mat(&c, B->c_out, B->c_in, gs);
      B->res_b = take_f32(&c, B->c_out);
    }
  }
  m->dec = take_mat(&c, m->n_classes, m->feat_out, gs);
  m->dec_b = take_f32(&c, m->n_classes);
#if CN_DEBUG_GEOM
  /* The cursor must land on the end of the blob, up to the exporter's tail padding: it pads to a
   * multiple of 8 for the word-wise hash (export_citrinet.py:224), so a remainder of 1..7 bytes is
   * EXPECTED, not a parse error. Anything larger means the descriptor table and the payload
   * disagree and every weight pointer past that point is off — which surfaces much later as
   * nonsense arithmetic rather than as a load failure, so it is worth stating explicitly. */
  {
    const unsigned long used = (unsigned long)((uintptr_t)c.p - (uintptr_t)b);
    const unsigned long tail = (unsigned long)blob_bytes - used;
    printf("[citrinet]   desc ok: cursor ended at +%lu of %lu bytes (%lu tail — %s)\n",
           used, (unsigned long)blob_bytes, tail,
           tail < 8u ? "exporter 8B padding, expected" : "MISALIGNED — bad blob/parser");
  }
#endif
  return 0;
}

/* ---- FFT ---------------------------------------------------------------------------------------
 * Iterative radix-2 Cooley-Tukey, complex-in/complex-out with a zero imaginary part. A real-only
 * transform would halve this, but the front-end is ~1% of the model's MACs (300 frames x 2304
 * butterflies vs 804 M MACs), so the simple version stays. Whisper's front-end in this repo uses a
 * direct DFT with a 1.3 MB cos/sin table; at n_fft=512 that would be 28x the work and 1 MB of
 * tables, hence the FFT here. */
static float g_re[CN_FFT_MAX], g_im[CN_FFT_MAX];
static float g_tw_re[CN_FFT_MAX / 2], g_tw_im[CN_FFT_MAX / 2];
static int g_fft_n;

static void fft_init(int n) {
  if (g_fft_n == n) return;
  for (int j = 0; j < n / 2; j++) {
    float a = -2.0f * (float)M_PI * (float)j / (float)n;
    g_tw_re[j] = cosf(a);
    g_tw_im[j] = sinf(a);
  }
  g_fft_n = n;
}

static void fft_run(int n) {
  for (int i = 1, j = 0; i < n; i++) {           /* bit-reversal permutation */
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float tr = g_re[i]; g_re[i] = g_re[j]; g_re[j] = tr;
      float ti = g_im[i]; g_im[i] = g_im[j]; g_im[j] = ti;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    int half = len >> 1, step = n / len;
    for (int i = 0; i < n; i += len) {
      for (int k = 0; k < half; k++) {
        float wr = g_tw_re[k * step], wi = g_tw_im[k * step];
        float xr = g_re[i + k + half], xi = g_im[i + k + half];
        float vr = xr * wr - xi * wi;
        float vi = xr * wi + xi * wr;
        float ur = g_re[i + k], ui = g_im[i + k];
        g_re[i + k] = ur + vr;       g_im[i + k] = ui + vi;
        g_re[i + k + half] = ur - vr; g_im[i + k + half] = ui - vi;
      }
    }
  }
}

/* ---- log-mel front-end ------------------------------------------------------------------------- */
int cn_num_frames(const cn_model_t *m, int n_samples) {
  /* NeMo get_seq_len: floor((N + (n_fft/2)*2 - n_fft) / hop) == floor(N / hop) */
  int v = (n_samples + (m->n_fft / 2) * 2 - m->n_fft) / m->hop_length;
  return v > 0 ? v : 0;
}

int cn_logmel(const cn_model_t *m, const float *audio, int n_samples, float **mel_p, int *n_frames_p) {
  PROF_T0();
  const int T = cn_num_frames(m, n_samples);
  const int N = m->n_fft, W = m->win_length, HOP = m->hop_length;
  const int BINS = m->n_bins, MEL = m->n_mels;
  if (T <= 1) return -1;
  if (N > CN_FFT_MAX) return -2;
  fft_init(N);

  /* preemphasis over the whole signal: y[0] = x[0], y[i] = x[i] - a*x[i-1] */
  float *pre = (float *)xmalloc((size_t)n_samples * sizeof(float), "logmel.pre");
  pre[0] = audio[0];
  for (int i = 1; i < n_samples; i++) pre[i] = audio[i] - m->preemph * audio[i - 1];

  float *mel = (float *)xmalloc((size_t)MEL * T * sizeof(float), "logmel.mel");
  CN_FILL(mel, (size_t)MEL * T * sizeof(float));
  float *power = (float *)xmalloc((size_t)BINS * sizeof(float), "logmel.power");
#if CN_DEBUG_GEOM
  /* First real heap traffic of the run: if the demo dies here it is the allocator / DRAM, not the
   * maths. Printing the addresses also shows where the heap actually landed. */
  printf("[citrinet]   logmel: T=%d pre=%p mel=%p power=%p\n", T, (void *)pre, (void *)mel, (void *)power);
#endif

  /* center=True with pad_mode="constant": the frame at t starts at t*hop - n_fft/2 in the original
   * signal. torch.stft centers a short window inside n_fft, so the window sits at woff. */
  const int woff = (N - W) / 2;
  for (int t = 0; t < T; t++) {
    const int base = t * HOP - N / 2;
    memset(g_re, 0, (size_t)N * sizeof(float));
    memset(g_im, 0, (size_t)N * sizeof(float));
    for (int j = 0; j < W; j++) {
      int idx = base + woff + j;
      if ((unsigned)idx < (unsigned)n_samples) g_re[woff + j] = pre[idx] * m->hann[j];
    }
    fft_run(N);
    /* mag_power=2: NeMo takes sqrt(re^2+im^2) then squares it, which is just re^2+im^2. */
    for (int b = 0; b < BINS; b++) power[b] = g_re[b] * g_re[b] + g_im[b] * g_im[b];
    for (int f = 0; f < MEL; f++) {
      const float *fbf = m->fb + (size_t)f * BINS;
      float acc = 0.0f;
      for (int b = 0; b < BINS; b++) acc += fbf[b] * power[b];
      mel[(size_t)f * T + t] = logf(acc + m->log_guard);
    }
  }
#if CN_DEBUG_GEOM
  printf("[citrinet]   logmel: %d FFT frames done\n", T);
#endif
  free(power);
  free(pre);

  /* per_feature normalization: per mel bin, mean and std over frames with ddof=1, std += 1e-5 */
  for (int f = 0; f < MEL; f++) {
    float *row = mel + (size_t)f * T;
    float sum = 0.0f;
    for (int t = 0; t < T; t++) sum += row[t];
    float mean = sum / (float)T;
    float acc = 0.0f;
    for (int t = 0; t < T; t++) { float d = row[t] - mean; acc += d * d; }
    float sd = sqrtf(acc / (float)(T - 1)) + m->norm_eps;
    float inv = 1.0f / sd;
    for (int t = 0; t < T; t++) row[t] = (row[t] - mean) * inv;
  }

  *mel_p = mel;
  *n_frames_p = T;
  PROF_ADD(g_cyc_fe);
  return 0;
}

/* ---- primitives --------------------------------------------------------------------------------- */
static inline int out_len(int T, int pad, int k, int dil, int stride) {
  return (T + 2 * pad - dil * (k - 1) - 1) / stride + 1;
}

/* One output sample of the depthwise conv, taps outside [0,T) skipped ('same'-style zero padding).
 * Shared by the scalar path and by the RVV path's edge handling so both accumulate over j in the
 * SAME order — which is what keeps the two bit-identical. */
static inline float dw_dot(const float *xc, const float *wc, int T, int k, int base, int dil) {
  float acc = 0.0f;
  for (int j = 0; j < k; j++) {
    const int idx = base + j * dil;
    if ((unsigned)idx < (unsigned)T) acc += wc[j] * xc[idx];
  }
  return acc;
}


/* ---- CONV1D accelerator path for the depthwise --------------------------------------------------
 * The depthwise is by far the worst kernel on this chip: ~470 M cycles for ~15 M MACs = 31
 * cycles/MAC, against 2.9 for the pointwise. The cause is read amplification — producing one output
 * vector needs k overlapping vle32s of the same row, so at k=9 we load 9x128 B to produce 128 B. On
 * a machine where bytes are the only currency that is fatal, and vectorising it bought only 17%.
 *
 * The accelerator streams each input element exactly ONCE (taps live in its internal shift
 * register), which structurally removes that amplification. It is not parallelism: CONV_INPUT_ADDR
 * is a fixed FIFO data port, not an address register, so the core hand-feeds every element and is
 * fully occupied. This is a straight swap of the dw kernel, with an Amdahl ceiling of ~15%.
 *
 * Semantics (from the hal_conv example, {1,2,3,4}*{-1,1,-1} -> {-2,-3,-1,-4}):
 *     out[t] = sum_j kernel[j] * in[t+j],  left-aligned, zero-extended to the right.
 * Ours is y[t] = sum_j w[j] * x[t - pad + j], so prepending `pad` zeros to the row makes the two
 * identical, and the engine's own right-zero-extension supplies the tail. The kernel is zero-padded
 * to the 8 or 16 taps the hardware requires (k = 3,5,7 -> 8; 9,11 -> 16); the extra taps contribute
 * nothing but do cost engine throughput.
 *
 * NOT bit-identical: the engine's FP32 accumulation order need not match our j = 0..k-1 loop, so the
 * stage fingerprints move slightly. The golden compare has a 2% relative tolerance, so STAGES PASS
 * remains the correctness gate — but this is the first change in this port that is not exact. */
#if CN_CONV1D
#include "hal_conv.h"

#define CN_C1D_MAX 512            /* max padded row: pad(<=5) + T(<=300), rounded even */
static float g_c1d_in[CN_C1D_MAX];
static float g_c1d_out[CN_C1D_MAX + 32];
static float g_c1d_k[16];

/* Returns 1 if the accelerator handled the whole depthwise, 0 to fall back to the RVV path. */
static int conv_dw_accel(const float *x, int C, int T, const float *w, int k,
                         int pad, int dil, float *y, int To) {
  if (dil != 1 || To != T || k > 16 || pad < 0) return 0;
  const uint8_t KL = (k <= 8) ? 8u : 16u;
  int L = pad + T;
  if (L & 1) L++;                                   /* FIFO packets carry 2 FP32 */
  if (L + (int)KL > CN_C1D_MAX) return 0;
  if (conv_begin_preconfigured_session((uint32_t)L, 1u, KL) != 0) return 0;

  for (int c = 0; c < C; c++) {
    memset(g_c1d_in, 0, (size_t)L * sizeof(float));
    memcpy(g_c1d_in + pad, x + (size_t)c * T, (size_t)T * sizeof(float));
    memset(g_c1d_k, 0, sizeof(g_c1d_k));
    memcpy(g_c1d_k, w + (size_t)c * k, (size_t)k * sizeof(float));

    const uint8_t st = perform_convolution_1D_preconfigured(
        (uint32_t *)g_c1d_in, (uint32_t)L, (uint32_t *)g_c1d_k, KL, (uint32_t *)g_c1d_out);
    if (st & (STATUS_ERROR | STATUS_INVALID)) {
      printf("[citrinet] CONV1D: engine status 0x%02x on channel %d (k=%d KL=%u L=%d) — "
             "falling back to the RVV kernel for this call\n", (unsigned)st, c, k, (unsigned)KL, L);
      return 0;
    }
    memcpy(y + (size_t)c * To, g_c1d_out, (size_t)To * sizeof(float));
  }
  return 1;
}
#endif /* CN_CONV1D */

/* Depthwise conv1d, channel-major, zero ('same'-style) padding. y is [C, To].
 *
 * RVV path (stride == 1 only): vectorized over t, exactly like the pointwise. For a fixed tap j the
 * input index is t - pad + j*dil, which is CONTIGUOUS in t, so each tap is one unit-stride vle32 and
 * one vfmacc — no strided or gathered loads.
 *
 * Only the INTERIOR is vectorized: the outputs whose whole kernel window lies inside [0,T), i.e.
 * t in [pad, T-1+pad-(k-1)*dil]. There no tap is skipped, so each output still accumulates
 * j = 0..k-1 in order and the result is BIT-IDENTICAL to the scalar path — the golden stays an
 * exact test. The k-1 edge outputs keep the scalar dw_dot, which is where the tap-clipping lives;
 * that is 8 of 38 outputs at worst (k=9, T=38) and 4 of 300 at best.
 *
 * stride > 1 stays scalar on purpose. Only the last sub-block of blocks 1, 7 and 14 strides, which
 * is ~0.1% of the depthwise MACs, and vectorizing it would need vlse32 — the strided load this
 * chip appears to handle badly (see the residual kernel's cost in the silicon profile). */
static void conv_dw(const float *x, int C, int T, const float *w, int k,
                    int stride, int pad, int dil, float *y, int To);

static void conv_dw_range(const float *x, int C, int T, const float *w, int k,
                          int stride, int pad, int dil, float *y, int To, int c0, int c1) {
  (void)C;
#if CN_USE_RVV
  if (stride == 1) {
    int t_lo = pad;
    if (t_lo > To) t_lo = To;
    int t_hi = T - 1 + pad - (k - 1) * dil;
    if (t_hi > To - 1) t_hi = To - 1;
    if (t_hi < t_lo - 1) t_hi = t_lo - 1;            /* empty interior; edges cover everything */
    for (int c = c0; c < c1; c++) {
      const float *xc = x + (size_t)c * T;
      const float *wc = w + (size_t)c * k;
      float *yc = y + (size_t)c * To;
      for (int t = 0; t < t_lo; t++) yc[t] = dw_dot(xc, wc, T, k, t - pad, dil);
      for (int t = t_lo; t <= t_hi; ) {
        const size_t vl = __riscv_vsetvl_e32m4((size_t)(t_hi - t + 1));
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        for (int j = 0; j < k; j++)
          acc = __riscv_vfmacc_vf_f32m4(acc, wc[j],
                                        __riscv_vle32_v_f32m4(xc + (t - pad) + j * dil, vl), vl);
        __riscv_vse32_v_f32m4(yc + t, acc, vl);
        t += (int)vl;
      }
      for (int t = t_hi + 1; t < To; t++) yc[t] = dw_dot(xc, wc, T, k, t - pad, dil);
    }
    return;
  }
#endif
  for (int c = c0; c < c1; c++) {
    const float *xc = x + (size_t)c * T;
    const float *wc = w + (size_t)c * k;
    float *yc = y + (size_t)c * To;
    for (int t = 0; t < To; t++) yc[t] = dw_dot(xc, wc, T, k, t * stride - pad, dil);
  }
}

#if CN_USE_RVV
#include <riscv_vector.h>

/* ---- RVV pointwise ------------------------------------------------------------------------------
 * Same maths as the scalar pointwise below, vectorized over the TIME axis. Structure borrowed from
 * vec-nn's xnn_f32_gemm_ukernel_7x4v__rvv (vfmacc_vf: acc += scalar * vector), but the numerics are
 * ours: fp32 activations with Q8_0 weights, NOT vec-nn's int8-in/int8-out requantizing pipeline.
 *
 * Why this is safe: vectorizing over t does not reorder any single output's accumulation over c —
 * element (o,t) still sums c = 0..Ci-1 in the same sequence, just with 32 values of t in flight. The
 * result is therefore BIT-IDENTICAL to the scalar path, so the golden stays an exact test.
 *
 * MR=4 output rows share each activation load (4 fmaccs per vle32). At VLEN=256, e32m4 gives
 * vl = 32 lanes and uses 4*4 + 4 = 20 vector registers — deliberately short of the 32 available.
 * vec-nn's 7-row variant would need 32 exactly; the moonshine port's experience with GCC 13 spilling
 * and mangling vsetvl under register pressure makes that a bad trade for one extra row of reuse.
 * One vtype (e32m4) is held across the whole inner loop, which is the other moonshine lesson. */
#define CN_MR 4


/* ---- activation tiling over t -------------------------------------------------------------------
 * MEASURED: this machine costs ~4.5 cycles per byte of activation loaded, and both the pointwise and
 * the residual sat exactly on that line. The kernels were structured output-rows-outermost, so each
 * one re-streamed the WHOLE x buffer once per output-row block — 64 times for the pointwise (Co=256,
 * MR=4) and 256 times for the residual (no MR blocking). At block 2, x is 256 x 150 x 4 = 153 KB and
 * y is another 153 KB, so the working set blows past the 256 KB cache and every one of those
 * re-reads misses.
 *
 * Fix: hoist the t loop OUTSIDE the output-row loop. Within one t-tile the activation slice is
 * Ci * TILE * 4 bytes and stays resident across all output rows; the weights stream past once per
 * tile instead. At TILE=32, Ci=Co=256: x tile 32 KB + weights ~73 KB + y tile 32 KB = ~137 KB,
 * comfortably inside 256 KB. Traffic for a T=150 sub-block goes from ~10 MB to ~0.5 MB.
 *
 * This is a pure loop interchange — it changes WHEN each output is computed, never the order in
 * which any single output accumulates over c. Results stay bit-identical, so the golden remains an
 * exact test.
 *
 * TILE is in ELEMENTS of t and should stay a multiple of the vector length (32 at VLEN=256). Raising
 * it cuts weight re-streaming but grows the resident slice: at Ci=Co=256, TILE=64 needs ~201 KB and
 * starts to crowd the cache. */
#ifndef CN_TILE_T
#define CN_TILE_T 32
#endif

static inline float cn_deq(const int8_t *q, float sc, int c) { return (float)q[c] * sc; }

/* ---- DMA weight staging into the 64 KB scratchpad ----------------------------------------------
 * Per tile the pointwise streams Co*Ci bytes of int8 weights THROUGH THE CACHE, evicting the 32 KB
 * activation tile that is re-read once per output-row block. On this machine that eviction is what
 * costs: an 80 KB software packing buffer slowed even the untouched depthwise by 25%. Moving the
 * weight stream into the scratchpad takes it out of the cache entirely — separate memory, no
 * competition — and the DMA runs while the core computes.
 *
 * Budget: ~4.9 M cycles of compute per tile (1,855 M / 379 tiles) against 73 KB of weights at the
 * measured 0.158 B/cycle = ~466 K cycles. The transfer is ~10% of the window it hides in.
 *
 * Contiguous only: logw=6 / inc_r=inc_w=64, the configuration dma-bmarks and dsp-whisper both use.
 * A STRIDED gather (independent inc_r/inc_w, to pack activations instead) was probed on silicon and
 * corrupts — 6464/8192 words wrong at logw=7 — so weights, which are already contiguous, are the
 * only thing worth staging here.
 *
 * Double-buffered: chunk k+1 is in flight while the core consumes chunk k. Only a global
 * dma_wait_till_inactive() exists, but that is enough — the wait lands after the compute, by which
 * point the transfer has normally finished. */
#if CN_DMA_STAGE
#include "hal_dma.h"
#define CN_SPAD_BASE   0x08000000UL
#define CN_STAGE_ROWS  64            /* output rows per chunk */
#define CN_STAGE_CI    256           /* only the Ci=256 pointwise is staged (the only int8 shape) */
#define CN_STAGE_BYTES ((size_t)CN_STAGE_ROWS * CN_STAGE_CI)      /* 16 KB per half */
#define CN_STAGE_LOGW  6
static uint16_t g_stage_tid = 1;

/* Start a contiguous DRAM->scratchpad transfer of one weight chunk. */
static void cn_stage_start(const int8_t *src, int half) {
  dma_transaction_t tx;
  tx.core = 0; tx.transaction_id = g_stage_tid; tx.transaction_priority = 1; tx.peripheral_id = 0;
  tx.addr_r = (uint64_t)(uintptr_t)src;
  tx.addr_w = (uint64_t)(CN_SPAD_BASE + (uintptr_t)half * CN_STAGE_BYTES);
  tx.inc_r = 1u << CN_STAGE_LOGW;
  tx.inc_w = 1u << CN_STAGE_LOGW;
  tx.len = (uint16_t)(CN_STAGE_BYTES >> CN_STAGE_LOGW);
  tx.logw = CN_STAGE_LOGW;
  tx.do_interrupt = false; tx.do_address_gate = false;
  if (set_DMA_C(0, tx, true)) start_DMA(0, tx.transaction_id, (void *)0);
  g_stage_tid++;
}
static inline void cn_stage_wait(void) {
  dma_wait_till_inactive(CN_DMA_WAIT_SPIN);
  __asm__ volatile("fence rw, rw" ::: "memory");
}
#endif /* CN_DMA_STAGE */

/* Fused ReLU on the way out of a kernel. relu_inplace was a SCALAR read-modify-write pass over the
 * whole activation buffer, run after almost every pointwise and after every residual — ~130 full
 * extra passes per utterance, and pure memory traffic on a machine whose limit is memory. Folding it
 * into the store that was happening anyway makes it free. The flag is loop-invariant, so the branch
 * hoists out of the t loop.
 * vfmax(x,0) differs from `x<0?0:x` only for -0.0 (yields +0.0) and NaN; neither changes a stage sum
 * or an absmax, and the golden still matches exactly. */
static inline vfloat32m4_t cn_relu_m4(vfloat32m4_t v, int relu, size_t vl) {
  return relu ? __riscv_vfmax_vf_f32m4(v, 0.0f, vl) : v;
}

/* ---- activation-tile packing -------------------------------------------------------------------
 * The inner loop reads x[c*T + t0] for c = 0..Ci-1: consecutive c are T*4 bytes apart (600 B at
 * T=150), so each pass uses 128 bytes out of every 600 and walks a 153 KB span — and it does that
 * once per output-row block, 64 times per tile. Packing copies the tile into [c][tw] order ONCE,
 * after which every pass over it is a single contiguous 32 KB stream: one strided walk instead of
 * 64. Same idea as pack_weight_matrix() in bearly25-bmarks/rvv-matmul.
 *
 * Cost is one extra read+write of the tile; the win is 63 fewer strided passes. Whether that trade
 * pays is exactly the kind of "restructured, not eliminated" change that MR=6 lost on, so it is a
 * build option and must be measured, not assumed. */
#if CN_PACK
#ifndef CN_PACK_MAX_C
#define CN_PACK_MAX_C 640
#endif
static float g_pack[(size_t)CN_PACK_MAX_C * CN_TILE_T];
#endif

static void pointwise_rvv(const cn_mat_t *W, const float *x, int T, const float *bias,
                          float *y, int gs, int orow0, int orow1, int relu) {
  const int Co = orow1, Ci = W->in, ng = Ci / gs;
  for (int tb = 0; tb < T; tb += CN_TILE_T) {
  const int tend = (T - tb < CN_TILE_T) ? T : tb + CN_TILE_T;
  /* Both paths index as xb[c*xstride + (t0 - tbase)] so the kernels below are shared verbatim. */
  const float *xb = x;
  int xstride = T, tbase = 0;
#if CN_PACK
  const int tw = tend - tb;
  if (Ci <= CN_PACK_MAX_C) {
    for (int c = 0; c < Ci; c++) {
      const float *src = x + (size_t)c * T + tb;
      float *dst = g_pack + (size_t)c * tw;
      for (int i = 0; i < tw; ) {
        const size_t vl = __riscv_vsetvl_e32m4((size_t)(tw - i));
        __riscv_vse32_v_f32m4(dst + i, __riscv_vle32_v_f32m4(src + i, vl), vl);
        i += (int)vl;
      }
    }
    xb = g_pack; xstride = tw; tbase = tb;
  }
#endif
  /* Output rows are walked in chunks so a chunk's weights can be staged in the scratchpad. With
   * staging off the chunk spans the whole range and this is exactly the previous loop. */
#if CN_DMA_STAGE
  const int staged = (W->q != 0) && (Ci == CN_STAGE_CI);
  int half = 0;
  if (staged) { cn_stage_start(W->q + (size_t)orow0 * Ci, 0); cn_stage_wait(); }
  const int CHUNK = staged ? CN_STAGE_ROWS : (Co - orow0);
#else
  const int CHUNK = Co - orow0;
#endif
  for (int cs = orow0; cs < Co; cs += CHUNK) {
  const int ce = (cs + CHUNK < Co) ? cs + CHUNK : Co;
  const int8_t *qbase = W->q;
#if CN_DMA_STAGE
  if (staged) {
    /* next chunk in flight while this one is consumed */
    if (ce < Co) cn_stage_start(W->q + (size_t)ce * Ci, half ^ 1);
    /* rebase so the kernels below can keep indexing by ABSOLUTE row */
    qbase = (const int8_t *)(uintptr_t)(CN_SPAD_BASE + (uintptr_t)half * CN_STAGE_BYTES)
            - (size_t)cs * Ci;
  }
#endif
  int o = cs;

  /* ---- 6-row tier (OFF by default: MEASURED SLOWER on silicon, see below) -------------------------------------------------------------------------------
   * x traffic in this kernel is (Co/MR) * Ci * T * 4 bytes — every activation load is re-read once
   * per output-row block. Tiling moved that traffic from DRAM into L2 (1.57x) but did NOT reduce it;
   * the dual-core result then showed the limit is the shared memory path, so the only lever left is
   * more reuse per load. MR=6 shares each vle32 across 6 fmaccs instead of 4: 1.5x fewer activation
   * loads for identical arithmetic.
   *
   * 6 is the register ceiling in practice: 6 m4 accumulators + 1 m4 load = 28 of 32 vector
   * registers. vec-nn's 7-row variant needs exactly 32, and the moonshine port found GCC 13 spills
   * and mangles vsetvl at that pressure — 4 spare registers is the margin that avoids it.
   *
   * Co is 256 or 640, both ≡ 4 (mod 6), so the 4-row tier below cleans up exactly one block and the
   * single-row tier never runs. Accumulation order over c is unchanged => bit-identical.
   *
   * RESULT: SLOWER ON SILICON (pw 1857 M -> 1937 M, total 4609 -> 4721 ms). 1.5x fewer activation
   * loads made it worse, which REFUTES "the pointwise is bound by activation load traffic" once
   * tiling is in place. Best explanation: MR=6 raises the number of concurrent memory streams from
   * 9 (4 weight rows + 1 activation + 4 output rows) to 13, and a low-associativity L1 thrashes on
   * stream count even while total bytes fall. MR=4 is a real local optimum. Kept behind
   * CITRINET_MR6 (default OFF) so the experiment is reproducible rather than re-derived. */
#if CN_MR6
  for (; o + 6 <= Co; o += 6) {
    float *y0 = y + (size_t)(o + 0) * T, *y1 = y + (size_t)(o + 1) * T;
    float *y2 = y + (size_t)(o + 2) * T, *y3 = y + (size_t)(o + 3) * T;
    float *y4 = y + (size_t)(o + 4) * T, *y5 = y + (size_t)(o + 5) * T;
    const float b0 = bias ? bias[o + 0] : 0.0f, b1 = bias ? bias[o + 1] : 0.0f;
    const float b2 = bias ? bias[o + 2] : 0.0f, b3 = bias ? bias[o + 3] : 0.0f;
    const float b4 = bias ? bias[o + 4] : 0.0f, b5 = bias ? bias[o + 5] : 0.0f;
    for (int t0 = tb; t0 < tend; ) {
      const size_t vl = __riscv_vsetvl_e32m4((size_t)(tend - t0));
      vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(b0, vl);
      vfloat32m4_t a1 = __riscv_vfmv_v_f_f32m4(b1, vl);
      vfloat32m4_t a2 = __riscv_vfmv_v_f_f32m4(b2, vl);
      vfloat32m4_t a3 = __riscv_vfmv_v_f_f32m4(b3, vl);
      vfloat32m4_t a4 = __riscv_vfmv_v_f_f32m4(b4, vl);
      vfloat32m4_t a5 = __riscv_vfmv_v_f_f32m4(b5, vl);
      if (W->f) {
        const float *w0 = W->f + (size_t)(o + 0) * Ci, *w1 = W->f + (size_t)(o + 1) * Ci;
        const float *w2 = W->f + (size_t)(o + 2) * Ci, *w3 = W->f + (size_t)(o + 3) * Ci;
        const float *w4 = W->f + (size_t)(o + 4) * Ci, *w5 = W->f + (size_t)(o + 5) * Ci;
        for (int c = 0; c < Ci; c++) {
          vfloat32m4_t vx = __riscv_vle32_v_f32m4(xb + (size_t)c * xstride + (t0 - tbase), vl);
          a0 = __riscv_vfmacc_vf_f32m4(a0, w0[c], vx, vl);
          a1 = __riscv_vfmacc_vf_f32m4(a1, w1[c], vx, vl);
          a2 = __riscv_vfmacc_vf_f32m4(a2, w2[c], vx, vl);
          a3 = __riscv_vfmacc_vf_f32m4(a3, w3[c], vx, vl);
          a4 = __riscv_vfmacc_vf_f32m4(a4, w4[c], vx, vl);
          a5 = __riscv_vfmacc_vf_f32m4(a5, w5[c], vx, vl);
        }
      } else {
        const int8_t *q0 = W->q + (size_t)(o + 0) * Ci, *q1 = W->q + (size_t)(o + 1) * Ci;
        const int8_t *q2 = W->q + (size_t)(o + 2) * Ci, *q3 = W->q + (size_t)(o + 3) * Ci;
        const int8_t *q4 = W->q + (size_t)(o + 4) * Ci, *q5 = W->q + (size_t)(o + 5) * Ci;
        const float *s0 = W->s + (size_t)(o + 0) * ng, *s1 = W->s + (size_t)(o + 1) * ng;
        const float *s2 = W->s + (size_t)(o + 2) * ng, *s3 = W->s + (size_t)(o + 3) * ng;
        const float *s4 = W->s + (size_t)(o + 4) * ng, *s5 = W->s + (size_t)(o + 5) * ng;
        for (int g = 0; g < ng; g++) {
          const float c0 = s0[g], c1 = s1[g], c2 = s2[g], c3 = s3[g], c4 = s4[g], c5 = s5[g];
          for (int u = 0; u < gs; u++) {
            const int c = g * gs + u;
            vfloat32m4_t vx = __riscv_vle32_v_f32m4(xb + (size_t)c * xstride + (t0 - tbase), vl);
            a0 = __riscv_vfmacc_vf_f32m4(a0, cn_deq(q0, c0, c), vx, vl);
            a1 = __riscv_vfmacc_vf_f32m4(a1, cn_deq(q1, c1, c), vx, vl);
            a2 = __riscv_vfmacc_vf_f32m4(a2, cn_deq(q2, c2, c), vx, vl);
            a3 = __riscv_vfmacc_vf_f32m4(a3, cn_deq(q3, c3, c), vx, vl);
            a4 = __riscv_vfmacc_vf_f32m4(a4, cn_deq(q4, c4, c), vx, vl);
            a5 = __riscv_vfmacc_vf_f32m4(a5, cn_deq(q5, c5, c), vx, vl);
          }
        }
      }
      __riscv_vse32_v_f32m4(y0 + t0, cn_relu_m4(a0, relu, vl), vl);
      __riscv_vse32_v_f32m4(y1 + t0, cn_relu_m4(a1, relu, vl), vl);
      __riscv_vse32_v_f32m4(y2 + t0, cn_relu_m4(a2, relu, vl), vl);
      __riscv_vse32_v_f32m4(y3 + t0, cn_relu_m4(a3, relu, vl), vl);
      __riscv_vse32_v_f32m4(y4 + t0, cn_relu_m4(a4, relu, vl), vl);
      __riscv_vse32_v_f32m4(y5 + t0, cn_relu_m4(a5, relu, vl), vl);
      t0 += (int)vl;
    }
  }
#endif /* CN_MR6 */
  for (; o + CN_MR <= ce; o += CN_MR) {
    float *yr[CN_MR];
    float bs[CN_MR];
    for (int r = 0; r < CN_MR; r++) {
      yr[r] = y + (size_t)(o + r) * T;
      bs[r] = bias ? bias[o + r] : 0.0f;
    }
    /* NOTE — two measured dead ends, so nobody re-derives them:
     * 1. The int8 inner loop below carries ~13 STACK SPILL/RELOAD instructions per 4 vfmaccs (GCC 13
     *    runs out of scalar registers holding 4 q + 4 s + x + 4 y pointers). Staging the
     *    dequantized weights in a buffer to relieve that removes every spill — inner loop 28 -> 15
     *    instructions — and is SLOWER on silicon (4,380 -> 4,762 ms): the buffer adds ~2.5 MB of
     *    traffic per sub-block, while the spills were hitting hot L1 stack slots and were cheap.
     *    Instruction count is not the currency here; memory traffic is.
     * 2. Do NOT "fix" it by putting the dequant loop between the vsetvl and the vfmaccs. That trips
     *    the GCC 13.2 vsetvl miscompile the moonshine port documented (it rewrites which register
     *    feeds vl): results come out wrong but plausible, drifting a little more each block, with
     *    the CTC tokens still matching. Only the stage fingerprints catch it. */
    for (int t0 = tb; t0 < tend; ) {
      const size_t vl = __riscv_vsetvl_e32m4((size_t)(tend - t0));
      vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(bs[0], vl);
      vfloat32m4_t a1 = __riscv_vfmv_v_f_f32m4(bs[1], vl);
      vfloat32m4_t a2 = __riscv_vfmv_v_f_f32m4(bs[2], vl);
      vfloat32m4_t a3 = __riscv_vfmv_v_f_f32m4(bs[3], vl);
      if (W->f) {
        const float *w0 = W->f + (size_t)(o + 0) * Ci, *w1 = W->f + (size_t)(o + 1) * Ci;
        const float *w2 = W->f + (size_t)(o + 2) * Ci, *w3 = W->f + (size_t)(o + 3) * Ci;
        for (int c = 0; c < Ci; c++) {
          vfloat32m4_t vx = __riscv_vle32_v_f32m4(xb + (size_t)c * xstride + (t0 - tbase), vl);
          a0 = __riscv_vfmacc_vf_f32m4(a0, w0[c], vx, vl);
          a1 = __riscv_vfmacc_vf_f32m4(a1, w1[c], vx, vl);
          a2 = __riscv_vfmacc_vf_f32m4(a2, w2[c], vx, vl);
          a3 = __riscv_vfmacc_vf_f32m4(a3, w3[c], vx, vl);
        }
      } else {
        const int8_t *q0 = qbase + (size_t)(o + 0) * Ci, *q1 = qbase + (size_t)(o + 1) * Ci;
        const int8_t *q2 = qbase + (size_t)(o + 2) * Ci, *q3 = qbase + (size_t)(o + 3) * Ci;
        const float *s0 = W->s + (size_t)(o + 0) * ng, *s1 = W->s + (size_t)(o + 1) * ng;
        const float *s2 = W->s + (size_t)(o + 2) * ng, *s3 = W->s + (size_t)(o + 3) * ng;
        for (int g = 0; g < ng; g++) {
          const float c0 = s0[g], c1 = s1[g], c2 = s2[g], c3 = s3[g];
          for (int u = 0; u < gs; u++) {
            const int c = g * gs + u;
            vfloat32m4_t vx = __riscv_vle32_v_f32m4(xb + (size_t)c * xstride + (t0 - tbase), vl);
            a0 = __riscv_vfmacc_vf_f32m4(a0, cn_deq(q0, c0, c), vx, vl);
            a1 = __riscv_vfmacc_vf_f32m4(a1, cn_deq(q1, c1, c), vx, vl);
            a2 = __riscv_vfmacc_vf_f32m4(a2, cn_deq(q2, c2, c), vx, vl);
            a3 = __riscv_vfmacc_vf_f32m4(a3, cn_deq(q3, c3, c), vx, vl);
          }
        }
      }
      __riscv_vse32_v_f32m4(yr[0] + t0, cn_relu_m4(a0, relu, vl), vl);
      __riscv_vse32_v_f32m4(yr[1] + t0, cn_relu_m4(a1, relu, vl), vl);
      __riscv_vse32_v_f32m4(yr[2] + t0, cn_relu_m4(a2, relu, vl), vl);
      __riscv_vse32_v_f32m4(yr[3] + t0, cn_relu_m4(a3, relu, vl), vl);
      t0 += (int)vl;
    }
  }
  /* leftover output rows (chunk % MR): one row at a time, same maths, same tile */
  for (; o < ce; o++) {
    float *yo = y + (size_t)o * T;
    const float b = bias ? bias[o] : 0.0f;
    for (int t0 = tb; t0 < tend; ) {
      const size_t vl = __riscv_vsetvl_e32m4((size_t)(tend - t0));
      vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(b, vl);
      if (W->f) {
        const float *wo = W->f + (size_t)o * Ci;
        for (int c = 0; c < Ci; c++)
          a0 = __riscv_vfmacc_vf_f32m4(a0, wo[c],
                                       __riscv_vle32_v_f32m4(xb + (size_t)c * xstride + (t0 - tbase), vl), vl);
      } else {
        const int8_t *q = qbase + (size_t)o * Ci;
        const float *s = W->s + (size_t)o * ng;
        for (int g = 0; g < ng; g++) {
          const float sc = s[g];
          for (int u = 0; u < gs; u++) {
            const int c = g * gs + u;
            a0 = __riscv_vfmacc_vf_f32m4(a0, cn_deq(q, sc, c),
                                         __riscv_vle32_v_f32m4(xb + (size_t)c * xstride + (t0 - tbase), vl), vl);
          }
        }
      }
      __riscv_vse32_v_f32m4(yo + t0, cn_relu_m4(a0, relu, vl), vl);
      t0 += (int)vl;
    }
  }
#if CN_DMA_STAGE
  if (staged) { cn_stage_wait(); half ^= 1; }
#endif
  }   /* output-row chunk */
  }   /* t-tile */
}

/* RVV residual: y += W * x_strided + bias. Same shape as the pointwise, except the residual branch
 * subsamples the block input by `stride`.
 *
 * THE STRIDE-1 CASE MUST NOT USE vlse32. This kernel originally issued a strided load
 * unconditionally, "because the residual subsamples by stride" — but 18 of the 21 residual blocks
 * have stride == 1 (only blocks 1, 7 and 14 stride), which is 85% of the residual's MACs. Those
 * were walking contiguous memory with a strided-load instruction, and this chip evidently
 * decomposes vlse32 into per-element accesses: the silicon profile showed the residual at 17.9
 * cycles/MAC against the pointwise's 4.6, for arithmetic that is otherwise identical. Unit-stride
 * loads where the data is unit-stride.
 *
 * The load is selected OUTSIDE the accumulation loops so there is no per-element branch, and the
 * accumulation itself is written once (via the macro) so the two paths cannot drift apart. Same
 * values in the same order either way -> bit-identical to both the old kernel and the scalar path. */
#define CN_RES_LOAD_UNIT(c)    __riscv_vle32_v_f32m4(x + (size_t)(c) * T_in + t0, vl)
#define CN_RES_LOAD_STRIDED(c) __riscv_vlse32_v_f32m4(x + (size_t)(c) * T_in + (size_t)t0 * stride, \
                                                      bstride, vl)

/* MR=4 accumulation: one activation load feeds four output rows, matching pointwise_rvv. Tiling
 * already made these loads hit cache; MR removes three quarters of the load INSTRUCTIONS, which is
 * the part tiling cannot reach (RVV-off runs 3.4x slower, so issue rate is a real cost here too).
 * 4 m4 accumulators + 1 m4 load = 20 of 32 vector registers, same budget as the pointwise. */
#define CN_RES_ACC4(LOADX)                                                        \
  do {                                                                            \
    if (W->f) {                                                                   \
      const float *w0 = W->f + (size_t)(o + 0) * Ci, *w1 = W->f + (size_t)(o + 1) * Ci; \
      const float *w2 = W->f + (size_t)(o + 2) * Ci, *w3 = W->f + (size_t)(o + 3) * Ci; \
      for (int c = 0; c < Ci; c++) {                                              \
        vfloat32m4_t vx = LOADX(c);                                               \
        a0 = __riscv_vfmacc_vf_f32m4(a0, w0[c], vx, vl);                          \
        a1 = __riscv_vfmacc_vf_f32m4(a1, w1[c], vx, vl);                          \
        a2 = __riscv_vfmacc_vf_f32m4(a2, w2[c], vx, vl);                          \
        a3 = __riscv_vfmacc_vf_f32m4(a3, w3[c], vx, vl);                          \
      }                                                                           \
    } else {                                                                      \
      const int8_t *q0 = W->q + (size_t)(o + 0) * Ci, *q1 = W->q + (size_t)(o + 1) * Ci; \
      const int8_t *q2 = W->q + (size_t)(o + 2) * Ci, *q3 = W->q + (size_t)(o + 3) * Ci; \
      const float *s0 = W->s + (size_t)(o + 0) * ng, *s1 = W->s + (size_t)(o + 1) * ng; \
      const float *s2 = W->s + (size_t)(o + 2) * ng, *s3 = W->s + (size_t)(o + 3) * ng; \
      for (int g = 0; g < ng; g++) {                                              \
        const float c0 = s0[g], c1 = s1[g], c2 = s2[g], c3 = s3[g];               \
        for (int u = 0; u < gs; u++) {                                            \
          const int c = g * gs + u;                                               \
          vfloat32m4_t vx = LOADX(c);                                             \
          a0 = __riscv_vfmacc_vf_f32m4(a0, cn_deq(q0, c0, c), vx, vl);            \
          a1 = __riscv_vfmacc_vf_f32m4(a1, cn_deq(q1, c1, c), vx, vl);            \
          a2 = __riscv_vfmacc_vf_f32m4(a2, cn_deq(q2, c2, c), vx, vl);            \
          a3 = __riscv_vfmacc_vf_f32m4(a3, cn_deq(q3, c3, c), vx, vl);            \
        }                                                                         \
      }                                                                           \
    }                                                                             \
  } while (0)

/* Single-row variant, for the Co % MR remainder. */
#define CN_RES_ACC(LOADX)                                                     \
  do {                                                                        \
    if (W->f) {                                                               \
      const float *wo = W->f + (size_t)o * Ci;                                \
      for (int c = 0; c < Ci; c++)                                            \
        a0 = __riscv_vfmacc_vf_f32m4(a0, wo[c], LOADX(c), vl);                \
    } else {                                                                  \
      const int8_t *q = W->q + (size_t)o * Ci;                                \
      const float *s = W->s + (size_t)o * ng;                                 \
      for (int g = 0; g < ng; g++) {                                          \
        const float sc = s[g];                                                \
        for (int u = 0; u < gs; u++) {                                        \
          const int c = g * gs + u;                                           \
          a0 = __riscv_vfmacc_vf_f32m4(a0, cn_deq(q, sc, c), LOADX(c), vl);   \
        }                                                                     \
      }                                                                       \
    }                                                                         \
  } while (0)

/* y[row] += acc  (read-modify-write; the residual adds onto the block output, it does not store) */
#define CN_RES_STORE(row, acc)                                                            \
  __riscv_vse32_v_f32m4((row) + t0, cn_relu_m4(                                           \
      __riscv_vfadd_vv_f32m4(__riscv_vle32_v_f32m4((row) + t0, vl), (acc), vl),           \
      relu, vl), vl)

static void residual_add_rvv(const cn_mat_t *W, const float *x, int T_in, int stride,
                             const float *bias, float *y, int T, int gs, int orow0, int orow1,
                             int relu) {
  const int Co = orow1, Ci = W->in, ng = Ci / gs;
  const ptrdiff_t bstride = (ptrdiff_t)stride * (ptrdiff_t)sizeof(float);
  const int unit = (stride == 1);
  /* Tiled over t for the same reason as the pointwise, and it matters MORE here: with no MR
   * blocking this kernel re-read the whole x buffer once per OUTPUT ROW (Co = 256 times, against the
   * pointwise's 64), which is exactly the 4x-worse cycles/MAC the silicon profile showed. Hoisting
   * the tile loop keeps the slice resident across all Co rows. */
  for (int tb = 0; tb < T; tb += CN_TILE_T) {
    const int tend = (T - tb < CN_TILE_T) ? T : tb + CN_TILE_T;
    int o = orow0;
    for (; o + CN_MR <= Co; o += CN_MR) {
      float *y0 = y + (size_t)(o + 0) * T, *y1 = y + (size_t)(o + 1) * T;
      float *y2 = y + (size_t)(o + 2) * T, *y3 = y + (size_t)(o + 3) * T;
      for (int t0 = tb; t0 < tend; ) {
        const size_t vl = __riscv_vsetvl_e32m4((size_t)(tend - t0));
        vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(bias[o + 0], vl);
        vfloat32m4_t a1 = __riscv_vfmv_v_f_f32m4(bias[o + 1], vl);
        vfloat32m4_t a2 = __riscv_vfmv_v_f_f32m4(bias[o + 2], vl);
        vfloat32m4_t a3 = __riscv_vfmv_v_f_f32m4(bias[o + 3], vl);
        if (unit) CN_RES_ACC4(CN_RES_LOAD_UNIT);
        else      CN_RES_ACC4(CN_RES_LOAD_STRIDED);
        CN_RES_STORE(y0, a0); CN_RES_STORE(y1, a1);
        CN_RES_STORE(y2, a2); CN_RES_STORE(y3, a3);
        t0 += (int)vl;
      }
    }
    for (; o < Co; o++) {                       /* Co % MR remainder */
      float *yo = y + (size_t)o * T;
      for (int t0 = tb; t0 < tend; ) {
        const size_t vl = __riscv_vsetvl_e32m4((size_t)(tend - t0));
        vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(bias[o], vl);
        if (unit) CN_RES_ACC(CN_RES_LOAD_UNIT);
        else      CN_RES_ACC(CN_RES_LOAD_STRIDED);
        CN_RES_STORE(yo, a0);
        t0 += (int)vl;
      }
    }
  }
}
#undef CN_RES_STORE
#undef CN_RES_ACC
#undef CN_RES_ACC4
#undef CN_RES_LOAD_UNIT
#undef CN_RES_LOAD_STRIDED
#endif /* CN_USE_RVV */

/* Pointwise (1x1) conv = [Co,Ci] x [Ci,T] + bias, channel-major. The inner loop is a scalar-times-
 * vector accumulate over t, with the weight dequantized once per (o,c). Computes output rows
 * [orow0, orow1) only, so the two harts can each take a slice. */
static void pointwise_kernel(const cn_mat_t *W, const float *x, int T, const float *bias,
                             float *y, int gs, int orow0, int orow1, int relu) {
#if CN_USE_RVV
  pointwise_rvv(W, x, T, bias, y, gs, orow0, orow1, relu);
  return;
#else
  const int Ci = W->in;
  for (int o = orow0; o < orow1; o++) {
    float *yo = y + (size_t)o * T;
    const float b = bias ? bias[o] : 0.0f;
    for (int t = 0; t < T; t++) yo[t] = b;
    if (W->f) {
      const float *wo = W->f + (size_t)o * Ci;
      for (int c = 0; c < Ci; c++) {
        const float wv = wo[c];
        const float *xc = x + (size_t)c * T;
        for (int t = 0; t < T; t++) yo[t] += wv * xc[t];
      }
    } else {
      const int8_t *q = W->q + (size_t)o * Ci;
      const float *s = W->s + (size_t)o * (Ci / gs);
      for (int g = 0; g < Ci / gs; g++) {
        const float sc = s[g];
        for (int u = 0; u < gs; u++) {
          const int c = g * gs + u;
          const float wv = (float)q[c] * sc;
          const float *xc = x + (size_t)c * T;
          for (int t = 0; t < T; t++) yo[t] += wv * xc[t];
        }
      }
    }
    if (relu) for (int t = 0; t < T; t++) if (yo[t] < 0.0f) yo[t] = 0.0f;
  }
#endif /* !CN_USE_RVV */
}

/* ---- dual-core dispatch -------------------------------------------------------------------------
 * Pattern taken from bearly25-bmarks/rvv-matmul + c2c-demos/dsp-whisper (proven on this silicon):
 *   fence; hthread_issue(1, worker, &arg); worker(own half) inline; hthread_join(1); fence.
 * hart 1 has its OWN vector unit, so both halves vectorize at full rate.
 *
 * Every split here is over OUTPUT rows (pointwise/residual) or CHANNELS (depthwise): the two harts
 * write disjoint rows of the destination and only ever READ shared inputs, so there is no write
 * sharing to reason about. The fence pair around the fork is what publishes hart 0's inputs and
 * collects hart 1's outputs.
 *
 * Descriptors live in BSS, NOT on hart 0's stack. dsp-whisper does the same, and the moonshine
 * dual-core work lost a lot of time to hart 1 dereferencing a hart-0 stack local. (That may have
 * been the old stack-outside-DRAM bug rather than a coherence limit, but BSS costs nothing.)
 *
 * Rows go to hart 1 from the TOP of the range so hart 0 starts at row 0, where the tiled activation
 * slice it touches first is already warm.
 */
#if CN_DUALCORE
#include "hthread.h"

typedef struct {                     /* pointwise / residual */
  const cn_mat_t *W; const float *x; int T; const float *bias; float *y; int gs;
  int o0, o1;                        /* row range */
  int T_in, stride;                  /* residual only */
  int relu;
} cn_mc_mat_t;

typedef struct {                     /* depthwise */
  const float *x; int C, T; const float *w; int k, stride, pad, dil; float *y; int To;
  int c0, c1;
} cn_mc_dw_t;

static cn_mc_mat_t g_mc_pw, g_mc_res;   /* BSS, single-issue: no kernel here is nested */
static cn_mc_dw_t  g_mc_dw;

static void residual_kernel(const cn_mat_t *W, const float *x, int T_in, int stride,
                            const float *bias, float *y, int T, int gs, int orow0, int orow1,
                            int relu);

/* How many of the rows hart 0 keeps, in percent. hart 0 also pays the issue/join overhead, so the
 * optimum is not necessarily 50 (rvv-matmul uses 55/45). Tunable without a source edit. */
#ifndef CN_MC_SPLIT_PCT
#define CN_MC_SPLIT_PCT 50
#endif

/* Boundary row for a [lo,hi) range: hart 0 takes [lo,cut), hart 1 takes [cut,hi). Returns hi when
 * the range is too small to be worth forking, which makes the caller run it single-core. */
static inline int cn_mc_cut(int lo, int hi, int quantum) {
  const int n = hi - lo;
  if (n < 2 * quantum) return hi;                      /* too small to split usefully */
  int k = (int)(((long)n * CN_MC_SPLIT_PCT + 50) / 100);
  k -= k % quantum;                                    /* keep both halves MR-aligned */
  if (k < quantum) k = quantum;
  if (k > n - quantum) k = n - quantum;
  return lo + k;
}

static void cn_mc_pw_worker(void *p) {
  const cn_mc_mat_t *a = (const cn_mc_mat_t *)p;
  pointwise_kernel(a->W, a->x, a->T, a->bias, a->y, a->gs, a->o0, a->o1, a->relu);
}
static void cn_mc_res_worker(void *p) {
  const cn_mc_mat_t *a = (const cn_mc_mat_t *)p;
  residual_kernel(a->W, a->x, a->T_in, a->stride, a->bias, a->y, a->T, a->gs, a->o0, a->o1, a->relu);
}
static void cn_mc_dw_worker(void *p) {
  const cn_mc_dw_t *a = (const cn_mc_dw_t *)p;
  conv_dw_range(a->x, a->C, a->T, a->w, a->k, a->stride, a->pad, a->dil, a->y, a->To, a->c0, a->c1);
}
static void cn_mc_nop(void *p) { (void)p; }

/* Start hart 1 once. Order matters: init_test (PLL/UART) -> hthread_init -> a nop dispatch that
 * lets hart 1 settle into its scheduler loop. hthread's __main ignores all work until hthread_init
 * publishes its cookie, so SKIPPING hthread_init looks exactly like "hart 1 never takes a task". */
void cn_dualcore_init(void) {
  hthread_init();
  hthread_issue(1, cn_mc_nop, NULL);
  hthread_join(1);
  printf("[citrinet] dual-core: hart 1 up (split %d/%d)\n", CN_MC_SPLIT_PCT, 100 - CN_MC_SPLIT_PCT);
}
#else
void cn_dualcore_init(void) {}
#endif /* CN_DUALCORE */

/* Depthwise is split over CHANNELS: each channel's output depends only on that channel's input row,
 * so the two harts touch entirely disjoint data. Quantum 1 — there is no MR blocking to preserve. */
static void conv_dw(const float *x, int C, int T, const float *w, int k,
                    int stride, int pad, int dil, float *y, int To) {
  PROF_T0();
#if CN_CONV1D
  /* Accelerator first; it silently declines (returns 0) for any geometry it cannot express. */
  if (stride == 1 && conv_dw_accel(x, C, T, w, k, pad, dil, y, To)) { PROF_ADD(g_cyc_dw); return; }
#endif
#if CN_DUALCORE
  const int cut = cn_mc_cut(0, C, 1);
  if (cut < C) {
    g_mc_dw.x = x; g_mc_dw.C = C; g_mc_dw.T = T; g_mc_dw.w = w; g_mc_dw.k = k;
    g_mc_dw.stride = stride; g_mc_dw.pad = pad; g_mc_dw.dil = dil; g_mc_dw.y = y; g_mc_dw.To = To;
    g_mc_dw.c0 = cut; g_mc_dw.c1 = C;
    __asm__ volatile("fence rw, rw" ::: "memory");
    hthread_issue(1, cn_mc_dw_worker, &g_mc_dw);
    conv_dw_range(x, C, T, w, k, stride, pad, dil, y, To, 0, cut);
    hthread_join(1);
    __asm__ volatile("fence rw, rw" ::: "memory");
    PROF_ADD(g_cyc_dw);
    return;
  }
#endif
  conv_dw_range(x, C, T, w, k, stride, pad, dil, y, To, 0, C);
  PROF_ADD(g_cyc_dw);
}


static void pointwise(const cn_mat_t *W, const float *x, int T, const float *bias,
                      float *y, int gs, int relu) {
  PROF_T0();
#if CN_DUALCORE
  const int cut = cn_mc_cut(0, W->out, CN_MR);
  if (cut < W->out) {
    g_mc_pw.W = W; g_mc_pw.x = x; g_mc_pw.T = T; g_mc_pw.bias = bias; g_mc_pw.y = y;
    g_mc_pw.gs = gs; g_mc_pw.o0 = cut; g_mc_pw.o1 = W->out; g_mc_pw.relu = relu;
    __asm__ volatile("fence rw, rw" ::: "memory");
    hthread_issue(1, cn_mc_pw_worker, &g_mc_pw);
    pointwise_kernel(W, x, T, bias, y, gs, 0, cut, relu);
    hthread_join(1);
    __asm__ volatile("fence rw, rw" ::: "memory");
    PROF_ADD(g_cyc_pw);
    return;
  }
#endif
  pointwise_kernel(W, x, T, bias, y, gs, 0, W->out, relu);
  PROF_ADD(g_cyc_pw);
}

/* y[o] = sum_i W[o][i] * x[i] */
static void matvec(const cn_mat_t *W, const float *x, float *y, int gs) {
  const int Co = W->out, Ci = W->in;
  for (int o = 0; o < Co; o++) {
    float acc = 0.0f;
    if (W->f) {
      const float *wo = W->f + (size_t)o * Ci;
      for (int i = 0; i < Ci; i++) acc += wo[i] * x[i];
    } else {
      const int8_t *q = W->q + (size_t)o * Ci;
      const float *s = W->s + (size_t)o * (Ci / gs);
      for (int g = 0; g < Ci / gs; g++) {
        float part = 0.0f;
        const int8_t *qq = q + g * gs;
        const float *xx = x + g * gs;
        for (int u = 0; u < gs; u++) part += (float)qq[u] * xx[u];
        acc += part * s[g];
      }
    }
    y[o] = acc;
  }
}

static void relu_inplace(float *x, size_t n) {
  for (size_t i = 0; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
}

/* Squeeze-excite: global mean over frames -> fc1 -> ReLU -> fc2 -> sigmoid -> per-channel gate. */
static void se_apply(const cn_block_t *B, float *x, int C, int T, int gs, float *scratch) {
  PROF_T0();
  float *pool = scratch;                 /* C     */
  float *hid = scratch + C;              /* se_hidden */
  float *gate = hid + B->se_hidden;      /* C     */
  const float invT = 1.0f / (float)T;
  for (int c = 0; c < C; c++) {
    const float *xc = x + (size_t)c * T;
    float sum = 0.0f;
    for (int t = 0; t < T; t++) sum += xc[t];
    pool[c] = sum * invT;
  }
  matvec(&B->se1, pool, hid, gs);
  for (int i = 0; i < B->se_hidden; i++) if (hid[i] < 0.0f) hid[i] = 0.0f;
  matvec(&B->se2, hid, gate, gs);
  for (int c = 0; c < C; c++) {
    const float g = 1.0f / (1.0f + expf(-gate[c]));
    float *xc = x + (size_t)c * T;
    for (int t = 0; t < T; t++) xc[t] *= g;
  }
  PROF_ADD(g_cyc_se);
}

/* y += W * x_strided + bias, where x is [Ci, T_in] sampled every `stride` frames (the 1x1 residual
 * branch, 'stride_add' mode). */
static void residual_kernel(const cn_mat_t *W, const float *x, int T_in, int stride,
                            const float *bias, float *y, int T, int gs, int orow0, int orow1,
                            int relu) {
#if CN_USE_RVV
  residual_add_rvv(W, x, T_in, stride, bias, y, T, gs, orow0, orow1, relu);
  return;
#else
  const int Ci = W->in;
  for (int o = orow0; o < orow1; o++) {
    float *yo = y + (size_t)o * T;
    const float b = bias[o];
    for (int t = 0; t < T; t++) yo[t] += b;
    if (W->f) {
      const float *wo = W->f + (size_t)o * Ci;
      for (int c = 0; c < Ci; c++) {
        const float wv = wo[c];
        const float *xc = x + (size_t)c * T_in;
        for (int t = 0; t < T; t++) yo[t] += wv * xc[t * stride];
      }
    } else {
      const int8_t *q = W->q + (size_t)o * Ci;
      const float *s = W->s + (size_t)o * (Ci / gs);
      for (int g = 0; g < Ci / gs; g++) {
        const float sc = s[g];
        for (int u = 0; u < gs; u++) {
          const int c = g * gs + u;
          const float wv = (float)q[c] * sc;
          const float *xc = x + (size_t)c * T_in;
          for (int t = 0; t < T; t++) yo[t] += wv * xc[t * stride];
        }
      }
    }
    if (relu) for (int t = 0; t < T; t++) if (yo[t] < 0.0f) yo[t] = 0.0f;
  }
#endif /* !CN_USE_RVV */
}

static void residual_add(const cn_mat_t *W, const float *x, int T_in, int stride,
                         const float *bias, float *y, int T, int gs, int relu) {
  PROF_T0();
#if CN_DUALCORE
  const int cut = cn_mc_cut(0, W->out, CN_MR);
  if (cut < W->out) {
    g_mc_res.W = W; g_mc_res.x = x; g_mc_res.T_in = T_in; g_mc_res.stride = stride;
    g_mc_res.bias = bias; g_mc_res.y = y; g_mc_res.T = T; g_mc_res.gs = gs;
    g_mc_res.o0 = cut; g_mc_res.o1 = W->out; g_mc_res.relu = relu;
    __asm__ volatile("fence rw, rw" ::: "memory");
    hthread_issue(1, cn_mc_res_worker, &g_mc_res);
    residual_kernel(W, x, T_in, stride, bias, y, T, gs, 0, cut, relu);
    hthread_join(1);
    __asm__ volatile("fence rw, rw" ::: "memory");
    PROF_ADD(g_cyc_res);
    return;
  }
#endif
  residual_kernel(W, x, T_in, stride, bias, y, T, gs, 0, W->out, relu);
  PROF_ADD(g_cyc_res);
}

#if CN_EVICT_CHECK
static volatile int32_t g_evict_sink;
/* Touch one byte per 64-byte line across the decoder matrix (~656 KB of real weight data, well past
 * the 256 KB cache) — the same kind of traffic the encoder itself generates. */
static void cn_evict_stream(const cn_model_t *m) {
  const int8_t *p = m->dec.q ? (const int8_t *)m->dec.q : (const int8_t *)m->dec.f;
  const size_t n = (size_t)m->dec.out * (size_t)m->dec.in;
  int32_t acc = 0;
  for (size_t i = 0; i < n; i += 64u) acc += p[i];
  g_evict_sink = acc;
}
#endif

static void stat_of(const float *x, size_t n, cn_stat_t *st) {
  double sum = 0.0, amax = 0.0;
  for (size_t i = 0; i < n; i++) {
    double v = (double)x[i];
    sum += v;
    double a = v < 0 ? -v : v;
    if (a > amax) amax = a;
  }
  st->sum = sum;
  st->absmax = amax;
  st->mean = n ? sum / (double)n : 0.0;
}

/* ---- encoder ------------------------------------------------------------------------------------ */
int cn_encode(const cn_model_t *m, const float *mel, int n_frames,
              float **enc_p, int *n_enc_p, cn_stat_t *stats) {
  const int gs = m->gs;

  /* Size the ping-pong buffers from the actual geometry rather than the worst case: T shrinks 8x
   * through the network while channels only grow at the very end, so the true peak (256 x T0) is
   * well under max_c * T0. */
  size_t maxel = (size_t)m->n_mels * n_frames;
  {
    int T = n_frames;
    for (int i = 0; i < m->n_blocks; i++) {
      const cn_block_t *B = &m->blk[i];
      size_t e = (size_t)B->c_in * T;
      if (e > maxel) maxel = e;
      for (int j = 0; j < B->repeat; j++) {
        int cin = (j == 0) ? B->c_in : B->c_out;
        int st = (B->stride > 1 && (!B->stride_last || j == B->repeat - 1)) ? B->stride : 1;
        int To = out_len(T, B->pad, B->k, B->dilation, st);
        if ((size_t)cin * To > maxel) maxel = (size_t)cin * To;
        if ((size_t)B->c_out * To > maxel) maxel = (size_t)B->c_out * To;
        T = To;
      }
    }
  }

  printf("[citrinet] encode: n_frames=%d maxel=%lu -> 4 buffers of %lu KB\n",
         n_frames, (unsigned long)maxel, (unsigned long)(maxel * sizeof(float) / 1024));
  float *buf = (float *)xmalloc(4 * maxel * sizeof(float), "encode.buf");
  CN_FILL(buf, 4 * maxel * sizeof(float));
  float *cur = buf, *nxt = buf + maxel, *tmp = buf + 2 * maxel, *res = buf + 3 * maxel;
  /* SE workspace: pool[C] + hid[se_hidden] + gate[C]. se_hidden <= c_out <= max_c, so 3*max_c is a
   * safe bound (2*max_c is NOT: block 22 is c_out=640, se_hidden=80 -> 1360 floats). */
  float *scr = (float *)xmalloc((size_t)(3 * m->max_c + 64) * sizeof(float), "encode.scr");
  CN_FILL(scr, (size_t)(3 * m->max_c + 64) * sizeof(float));

  memcpy(cur, mel, (size_t)m->n_mels * n_frames * sizeof(float));
  int T = n_frames, C = m->n_mels;

  for (int i = 0; i < m->n_blocks; i++) {
    const cn_block_t *B = &m->blk[i];
    const int T_in = T, C_in = C;
    if (B->has_res) memcpy(res, cur, (size_t)C_in * T_in * sizeof(float));

#if CN_TRACE_BLOCK >= 0
    const int tr = (i == CN_TRACE_BLOCK);
    cn_stat_t ts;
#define CN_TRACE(what, ptr, n) do { if (tr) { stat_of((ptr), (n), &ts); \
      printf("[citrinet]   TRACE blk%d %-10s sum=% .6e absmax=%.6e\n", i, (what), ts.sum, ts.absmax); } } while (0)
#else
#define CN_TRACE(what, ptr, n) do { } while (0)
#endif

    for (int j = 0; j < B->repeat; j++) {
      const int cin = (j == 0) ? B->c_in : B->c_out;
      const int st = (B->stride > 1 && (!B->stride_last || j == B->repeat - 1)) ? B->stride : 1;
      const int To = out_len(T, B->pad, B->k, B->dilation, st);
      conv_dw(cur, cin, T, B->sub[j].dw, B->k, st, B->pad, B->dilation, tmp, To);
      CN_TRACE("dw", tmp, (size_t)cin * To);
      /* ReLU on every sub-block but the last is FUSED into the pointwise store (see cn_relu_m4).
       * Note the "pw" trace below is therefore post-ReLU on those sub-blocks. */
      pointwise(&B->sub[j].pw, tmp, To, B->sub[j].bias, nxt, gs, j < B->repeat - 1);
      CN_TRACE("pw", nxt, (size_t)B->c_out * To);
      float *sw = cur; cur = nxt; nxt = sw;      /* ping-pong; tmp and res are untouched */
      T = To; C = B->c_out;
    }

    if (B->has_se) { se_apply(B, cur, C, T, gs, scr); CN_TRACE("se", cur, (size_t)C * T); }
    /* The block-final ReLU folds into the residual store when there is one. Blocks 0 and 22 have no
     * residual branch, so they still need the standalone pass. */
    if (B->has_res) { residual_add(&B->res, res, T_in, B->stride, B->res_b, cur, T, gs, 1);
                      CN_TRACE("res", cur, (size_t)C * T); }
    else            { relu_inplace(cur, (size_t)C * T); }
#undef CN_TRACE
    if (stats) stat_of(cur, (size_t)C * T, &stats[i]);
#if CN_EVICT_CHECK
    if (stats) {
      cn_stat_t after;
      cn_evict_stream(m);
      stat_of(cur, (size_t)C * T, &after);
      if (after.sum != stats[i].sum || after.absmax != stats[i].absmax)
        printf("[citrinet] EVICT-CHECK blk%d CHANGED: sum % .6e -> % .6e   absmax %.6e -> %.6e\n",
               i, stats[i].sum, after.sum, stats[i].absmax, after.absmax);
    }
#endif
#if CN_DEBUG_GEOM
    /* Live trace: without this the encoder is a silent black box for tens of seconds on silicon,
     * so a wedge is indistinguishable from "still working". */
    printf("[citrinet]   blk%-2d done C=%-4d T=%d\n", i, C, T);
#endif
  }

  float *out = (float *)xmalloc((size_t)C * T * sizeof(float), "encode.out");
  memcpy(out, cur, (size_t)C * T * sizeof(float));
  free(scr);
  free(buf);
  *enc_p = out;
  *n_enc_p = T;
  return 0;
}

/* ---- decoder + greedy CTC ----------------------------------------------------------------------- */
int cn_ctc_greedy(const cn_model_t *m, const float *enc, int n_enc,
                  int *out_tokens, int max_out, int *n_out) {
  PROF_T0();
  const int gs = m->gs, K = m->n_classes, F = m->feat_out;
#if CN_DEBUG_GEOM
  printf("[citrinet]   ctc: classes=%d feat_in=%d frames=%d dec=%s\n",
         K, F, n_enc, m->dec.f ? "fp32" : "q8");
#endif
  float *row = (float *)xmalloc((size_t)n_enc * sizeof(float), "ctc.row");
  float *best = (float *)xmalloc((size_t)n_enc * sizeof(float), "ctc.best");
  int *bestid = (int *)xmalloc((size_t)n_enc * sizeof(int), "ctc.bestid");
  for (int t = 0; t < n_enc; t++) { best[t] = -INFINITY; bestid[t] = 0; }

  /* One class row at a time: keeps a running per-frame argmax instead of materializing
   * [n_classes, n_enc] logits, and keeps the encoder access pattern contiguous in t. */
  for (int o = 0; o < K; o++) {
    const float b = m->dec_b[o];
#if CN_USE_RVV
    /* Same vectorize-over-t shape as the pointwise. Only one output row is live at a time here
     * (the argmax is folded in per row), so there is no MR blocking to do. */
    {
      const int ng = F / gs;
      const int8_t *q = m->dec.q ? m->dec.q + (size_t)o * F : 0;
      const float *s = m->dec.s ? m->dec.s + (size_t)o * ng : 0;
      const float *wo = m->dec.f ? m->dec.f + (size_t)o * F : 0;
      for (int t0 = 0; t0 < n_enc; ) {
        const size_t vl = __riscv_vsetvl_e32m4((size_t)(n_enc - t0));
        vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(b, vl);
        if (wo) {
          for (int c = 0; c < F; c++)
            a0 = __riscv_vfmacc_vf_f32m4(a0, wo[c],
                                         __riscv_vle32_v_f32m4(enc + (size_t)c * n_enc + t0, vl), vl);
        } else {
          for (int g = 0; g < ng; g++) {
            const float sc = s[g];
            for (int u = 0; u < gs; u++) {
              const int c = g * gs + u;
              a0 = __riscv_vfmacc_vf_f32m4(a0, cn_deq(q, sc, c),
                                           __riscv_vle32_v_f32m4(enc + (size_t)c * n_enc + t0, vl), vl);
            }
          }
        }
        __riscv_vse32_v_f32m4(row + t0, a0, vl);
        t0 += (int)vl;
      }
    }
    goto argmax;
#endif
    for (int t = 0; t < n_enc; t++) row[t] = b;
    if (m->dec.f) {
      const float *wo = m->dec.f + (size_t)o * F;
      for (int c = 0; c < F; c++) {
        const float wv = wo[c];
        const float *xc = enc + (size_t)c * n_enc;
        for (int t = 0; t < n_enc; t++) row[t] += wv * xc[t];
      }
    } else {
      const int8_t *q = m->dec.q + (size_t)o * F;
      const float *s = m->dec.s + (size_t)o * (F / gs);
      for (int g = 0; g < F / gs; g++) {
        const float sc = s[g];
        for (int u = 0; u < gs; u++) {
          const int c = g * gs + u;
          const float wv = (float)q[c] * sc;
          const float *xc = enc + (size_t)c * n_enc;
          for (int t = 0; t < n_enc; t++) row[t] += wv * xc[t];
        }
      }
    }
#if CN_USE_RVV
argmax:
#endif
    /* strict > keeps the FIRST maximum, matching numpy/torch argmax tie-breaking */
    for (int t = 0; t < n_enc; t++) if (row[t] > best[t]) { best[t] = row[t]; bestid[t] = o; }
  }

#if CN_DEBUG_GEOM
  printf("[citrinet]   ctc: logits done, collapsing\n");
#endif
  int n = 0, prev = -1;
  for (int t = 0; t < n_enc; t++) {
    const int id = bestid[t];
    if (id != prev && id != m->blank && n < max_out) out_tokens[n++] = id;
    prev = id;
  }
  *n_out = n;
  free(bestid); free(best); free(row);
  PROF_ADD(g_cyc_dec);
  return 0;
}

int cn_transcribe(const cn_model_t *m, const float *audio, int n_samples,
                  int *out_tokens, int max_out, int *n_out) {
  float *mel; int n_frames;
  if (cn_logmel(m, audio, n_samples, &mel, &n_frames) != 0) { *n_out = 0; return 0; }
  float *enc; int n_enc;
  cn_encode(m, mel, n_frames, &enc, &n_enc, NULL);
  free(mel);
  cn_ctc_greedy(m, enc, n_enc, out_tokens, max_out, n_out);
  free(enc);
  return n_enc;
}

/* ---- detokenize --------------------------------------------------------------------------------- */
void cn_print_tokens_text(const char *tag, const int *toks, int n) {
  printf("%s\"", tag);
  int emitted = 0, lead_stripped = 0;
  for (int i = 0; i < n; i++) {
    const int t = toks[i];
    if (t < 0 || t >= CV_N_VOCAB) continue;
    for (int b = cv_offset[t]; b < cv_offset[t + 1]; b++) {
      const unsigned char ch = cv_bytes[b];
      if (!lead_stripped && !emitted && ch == ' ') { lead_stripped = 1; continue; }
      putchar(ch); emitted = 1;
    }
  }
  printf("\"\n");
}

/* ---- validate ----------------------------------------------------------------------------------- */
#define CN_REL_TOL 0.02

static double rel_err(double got, double ref) {
  double d = got - ref; if (d < 0) d = -d;
  double r = ref < 0 ? -ref : ref;
  return (r > 1e-9) ? d / r : d;
}

int cn_run_validate(const void *blob, size_t blob_bytes, const float *audio, int n_samples) {
  /* STATIC, not automatic. cn_model_t is 17,272 bytes; as a local it made this frame 18,800 bytes
   * against a 4 KiB stack, and the resulting out-of-bounds locals were the reason the same ELF gave
   * different answers on different silicon runs (Spike has real memory there, so it never showed).
   * The CMake __stack_size=0x10000 now covers this too, but keeping the big objects off the stack
   * means the demo does not depend on a link flag to be correct. Same for the fingerprint arrays
   * and the token buffer below. cn_run_validate is called once, so static costs nothing. */
  static cn_model_t m;
  if (cn_model_load(blob, blob_bytes, &m) != 0) { printf("[citrinet] bad model magic/geometry\n"); return 1; }
  printf("[citrinet] model: blocks=%d n_mels=%d feat_out=%d classes=%d blank=%d gs=%d kernel=scalar\n",
         m.n_blocks, m.n_mels, m.feat_out, m.n_classes, m.blank, m.gs);
  printf("[citrinet] frontend: sr=%d n_fft=%d win=%d hop=%d bins=%d\n",
         m.sample_rate, m.n_fft, m.win_length, m.hop_length, m.n_bins);
  printf("[citrinet] audio: n_samples=%d -> mel frames=%d  (max_k=%d max_c=%d)\n",
         n_samples, cn_num_frames(&m, n_samples), m.max_k, m.max_c);
#if CN_DEBUG_GEOM
  {
    extern char __end[], __heap_end[];
    printf("[citrinet] heap: __end=0x%lx __heap_end=0x%lx (%lu MB of address space)\n",
           (unsigned long)(uintptr_t)__end, (unsigned long)(uintptr_t)__heap_end,
           (unsigned long)((uintptr_t)__heap_end - (uintptr_t)__end) >> 20);
  }
  for (int i = 0; i < m.n_blocks; i++) {
    const cn_block_t *B = &m.blk[i];
    printf("[citrinet] blk%-2d c_in=%-4d c_out=%-4d rep=%d k=%-3d s=%d sl=%d dil=%d se=%d seh=%-3d res=%d pad=%d\n",
           i, B->c_in, B->c_out, B->repeat, B->k, B->stride, B->stride_last,
           B->dilation, B->has_se, B->se_hidden, B->has_res, B->pad);
  }
#endif

  cn_profile_reset();
  /* Time the COMPUTE only. On silicon the stage-fingerprint printf block below costs far more than
   * the inference at 115200 baud, so a wall-clock measurement across the whole function would be
   * mostly UART. */
  uint64_t compute = 0, t0 = cn_cy();
  float *mel; int n_frames;
  if (cn_logmel(&m, audio, n_samples, &mel, &n_frames) != 0) {
    printf("[citrinet] front-end failed (audio too short)\n"); return 1;
  }
  cn_stat_t mel_st;
  stat_of(mel, (size_t)m.n_mels * n_frames, &mel_st);

  static cn_stat_t stats[CN_MAX_BLOCKS];
  float *enc; int n_enc;
  cn_encode(&m, mel, n_frames, &enc, &n_enc, stats);
  free(mel);
  compute += cn_cy() - t0;

  printf("[citrinet] frames: mel=%d (ref %d)  enc=%d (ref %d)\n",
         n_frames, CN_REF_MEL_FRAMES, n_enc, CN_REF_ENC_FRAMES);

#if CN_DETERMINISM_CHECK
  /* Re-run the identical encode on the identical input. Bit-identical results mean the hardware is
   * deterministic and any divergence from the golden is systematic (FPU semantics / a real bug);
   * differing results mean the hardware itself is unreliable and chasing the maths is pointless. */
  {
    float *mel2; int nf2;
    cn_logmel(&m, audio, n_samples, &mel2, &nf2);
    static cn_stat_t s2[CN_MAX_BLOCKS];
    float *enc2; int ne2;
    cn_encode(&m, mel2, nf2, &enc2, &ne2, s2);
    free(mel2);
    int first_diff = -1;
    for (int i = 0; i < m.n_blocks; i++)
      if (s2[i].sum != stats[i].sum || s2[i].absmax != stats[i].absmax) { first_diff = i; break; }
    if (first_diff < 0) {
      printf("[citrinet] DETERMINISM: two runs bit-identical -> hardware is deterministic;\n"
             "[citrinet]              a golden mismatch is therefore systematic, not flaky.\n");
    } else {
      printf("[citrinet] DETERMINISM: runs DIFFER first at blk%d (run1 sum=% .6e run2 sum=% .6e)\n"
             "[citrinet]              -> the hardware is not reproducing its own arithmetic.\n",
             first_diff, stats[first_diff].sum, s2[first_diff].sum);
    }
    free(enc2);
  }
#endif

  /* Stage compare. absmax is the primary metric: the per-feature-normalized log-mel has a sum of
   * ~0 by construction, so a relative test on `sum` there is meaningless. `sum` is still checked
   * whenever it is large enough to be informative. */
  int stage_pass = 1;
  printf("[citrinet] stage fingerprints (C vs ref):\n");
  for (int i = 0; i < CN_REF_NUM_STAGES; i++) {
    const cn_stat_t *got = (i == 0) ? &mel_st : &stats[i - 1];
    const cn_ref_stage_t *ref = &cn_ref_stages[i];
    double ra = rel_err(got->absmax, ref->absmax);
    double rs = rel_err(got->sum, ref->sum);
    int ok = (ra < CN_REL_TOL) && (fabs(ref->sum) <= 1.0 || rs < CN_REL_TOL);
    if (!ok) stage_pass = 0;
    printf("    %-7s sum=% .5e (ref % .5e) absmax=%.5e (ref %.5e) rel_amax=%.4f %s\n",
           ref->name, got->sum, ref->sum, got->absmax, ref->absmax, ra, ok ? "ok" : "BAD");
  }

  static int toks[256];
  int n_out = 0;
  t0 = cn_cy();
  cn_ctc_greedy(&m, enc, n_enc, toks, 256, &n_out);
  compute += cn_cy() - t0;
  free(enc);

  int cmp = (n_out < CN_REF_NUM_TOKENS) ? n_out : CN_REF_NUM_TOKENS, match = 0;
  for (int i = 0; i < cmp; i++) if (toks[i] == cn_ref_tokens[i]) match++;
  printf("[citrinet] tokens C  =");
  for (int i = 0; i < n_out; i++) printf(" %d", toks[i]);
  printf("\n[citrinet] tokens ref=");
  for (int i = 0; i < CN_REF_NUM_TOKENS; i++) printf(" %d", cn_ref_tokens[i]);
  printf("\n");
  cn_print_tokens_text("[citrinet] text (C)  : ", toks, n_out);
  printf("[citrinet] text (ref): \"%s\"\n", CN_REF_TEXT);

  const int pass = stage_pass && (match == cmp) && (n_out == CN_REF_NUM_TOKENS);
  printf("[citrinet] STAGES %s  |  TOKENS %d/%d match (n_out=%d ref=%d)\n",
         stage_pass ? "PASS" : "FAIL", match, cmp, n_out, CN_REF_NUM_TOKENS);
  {
    const uint64_t hz = (uint64_t)CN_TARGET_FREQ_HZ;
    printf("[citrinet] LATENCY audio->transcript: %lu ms  (%lu cycles @ %u MHz, %d samples = %d ms audio)\n",
           (unsigned long)(compute / (hz / 1000u)), (unsigned long)compute,
           (unsigned)(hz / 1000000u), n_samples, n_samples / (m.sample_rate / 1000));
  }
  cn_profile_report();
  cn_stack_report();
  /* Did the model survive the run? Verifying at boot only proves it ARRIVED intact. */
  cn_blob_recheck(blob, blob_bytes);
  printf("[citrinet] RESULT %s\n", pass ? "PASS" : "FAIL");
  fflush(stdout);
  return pass ? 0 : 1;
}
