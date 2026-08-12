/*
 * moonshine.c — Moonshine tiny float inference engine for DSP 25 (plan 006).
 *
 * Pure C (no HAL) so it compiles identically for host gcc and the RISC-V/Spike target. Weights are
 * Q8_0, dequantized per-group inside the matvec; activations are float. Raw-audio conv preprocessor
 * + encoder + decoder + greedy decode. Adapted from the Whisper engine (whisper.c); the deltas are
 * documented in moonshine.h.
 *
 * The matvec / attention loops are the RVV vectorization targets; the scalar forms are the
 * reference oracle and must produce identical stage sums / tokens (validated against the host golden).
 */
#include "moonshine.h"
#include "moonshine_reference.h"
#include "moonshine_vocab.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef MS_USE_RVV
#define MS_USE_RVV 0             /* 1 -> vectorized matvec (RISC-V V); 0 -> scalar reference */
#endif
#ifndef MS_PROFILE
#define MS_PROFILE 0
#endif

/* ---- bandwidth-bound classifier / decoder-fc1 optimizations (ports of the whisper.c originals,
 * proven on this silicon). matmul cost here is DRAM-weight-stream dominated; these three cut that
 * stream on the big single-row reduction matmuls (classifier out=vocab, decoder fc1 out=2*ff_dec).
 * The tiled encoder GEMM (many-row) is sequential-streaming / compute-bound and keeps int8. -------- */

/* int4 weights: re-quantize a Q8_0 matrix to int4 [-7,7] per group and stream half the bytes. Applied
 * to reduction matmuls with out >= MS_INT4_MIN_OUT (classifier 32768, decoder fc1 2304). Numerically
 * lossy — Spike-validate tokens vs the golden. Mutually exclusive with MS_DMA_CLASSIFIER. */
#ifndef MS_INT4
#define MS_INT4 0
#endif
#ifndef MS_INT4_GS
#define MS_INT4_GS 32            /* int4 group size; == the Q8_0 gs (32) so groups align 1:1 */
#endif
#ifndef MS_INT4_MIN_OUT
#define MS_INT4_MIN_OUT 384      /* only repack matmuls whose output dim >= this */
#endif

/* DMA weight double-buffering for the classifier (the biggest single DRAM weight stream, ~9.4 MB
 * int8/token). Prefetch weight-row chunks DRAM->SRAM via the DMA engine while the core computes the
 * dots from the previous chunk in SRAM (SRAM reads ~4x faster than core-DRAM; DMA never starves).
 * Lossless (stays int8). Single-core. Silicon-only (Spike has no DMA engine). Requires MS_INT4=0. */
#ifndef MS_DMA_CLASSIFIER
#define MS_DMA_CLASSIFIER 0
#endif
#ifndef MS_DMA_SRAM_BASE
#define MS_DMA_SRAM_BASE 0x08000000UL   /* on-chip SCRATCH (dsp25.ld) */
#endif
#ifndef MS_DMA_CHUNK_ROWS
#define MS_DMA_CHUNK_ROWS 64u           /* rows/chunk: 64*288 = 18 KiB; two buffers = 36 KiB < 64 KiB */
#endif

/* Vocab pruning: run the classifier over only the first MS_VOCAB_TOPK token ids (+ always eos). Cuts
 * the classifier weight stream ~vocab/TOPK x. SentencePiece ids are roughly score/frequency ordered so
 * [0,TOPK) is the common subset, but pruning CAN drop a needed id -> experimental, default OFF. */
#ifndef MS_VOCAB_TOPK
#define MS_VOCAB_TOPK 0
#endif

#if MS_INT4 && MS_DMA_CLASSIFIER
#error "MS_INT4 and MS_DMA_CLASSIFIER both target the classifier matmul — pick one"
#endif

/* Coarse phase markers to localize a dual-core wedge to conv-stem / a specific encoder layer /
 * cross-KV setup / a decode step / the classifier. hart 0 only (the engine runs on hart 0; forks are
 * transient), so no UART contention. No-op unless the dual-core debug build. */
#ifndef MS_DUALCORE_DEBUG
#define MS_DUALCORE_DEBUG 0
#endif
#if MS_DUALCORE_DEBUG
#define MS_DBG(...) do { printf(__VA_ARGS__); } while (0)
#else
#define MS_DBG(...) do {} while (0)
#endif

#define MS_MAX_ROW 4096          /* upper bound on a matvec `in` (dec fc1 out=2304, in=288) */
#define MS_MHA_MAX_KEYS 2048     /* attention stack scratch bound (>= typical encoder positions) */

#if MS_PROFILE
static uint64_t g_cyc_conv, g_cyc_mm, g_cyc_attn, g_cyc_gelu, g_cyc_ln, g_cyc_rope;
static inline uint64_t prof_cy(void) { uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }
#define PROF_T0()      uint64_t _pt0 = prof_cy()
#define PROF_ADD(acc)  do { (acc) += prof_cy() - _pt0; } while (0)
#else
#define PROF_T0()      do {} while (0)
#define PROF_ADD(acc)  do {} while (0)
#endif

/* Override glossy's WEAK trap_handler. The default is a no-op that returns mepc unchanged, so the
 * faulting instruction retries forever: on hart 0 any fault is a SILENT infinite loop, which looks
 * exactly like a hang and is why several dual-core runs simply stopped printing. hart 1 has its own
 * mtvec catcher (ms_h1_trap_entry), so only hart 0 arrives here — no UART contention. */
uintptr_t trap_handler(uintptr_t m_epc, uintptr_t m_cause, uintptr_t m_tval, uintptr_t regs[32]) {
  (void)regs;
  uint64_t hid; __asm__ volatile("csrr %0, mhartid" : "=r"(hid));
  static const char *const why[] = { "instr-misalign", "instr-access", "illegal-instr", "breakpoint",
                                     "load-misalign", "load-access", "store-misalign", "store-access" };
  printf("\n[moonshine] TRAP hart=%lu mcause=%lu (%s) mepc=0x%lx mtval=0x%lx\n",
         (unsigned long)hid, (unsigned long)m_cause,
         (m_cause < 8u) ? why[m_cause] : "other/interrupt",
         (unsigned long)m_epc, (unsigned long)m_tval);
  printf("[moonshine] TRAP: halted (the default handler would have spun here silently)\n");
  for (;;) __asm__ volatile("wfi");
  return m_epc;
}

/* Checked malloc: on this bare target a NULL deref just spins the trap handler, so fail loudly. */
static void *xmalloc(size_t n) {
  void *p = malloc(n);
  if (!p) { printf("[moonshine] OOM: malloc(%lu) failed\n", (unsigned long)n); exit(1); }
  return p;
}

/* ------------------------------------------------------------------ blob cursor + model load --- */
typedef struct { const unsigned char *p; } cur_t;

static const float *take_f32(cur_t *c, size_t n) {
  const float *r = (const float *)c->p;
  c->p += n * sizeof(float);
  return r;
}
static msq8_t take_q8(cur_t *c, size_t out, size_t in, int gs) {
  msq8_t w;
  w.q = (const int8_t *)c->p;   c->p += out * in;
  w.s = (const float *)c->p;    c->p += (out * in / gs) * sizeof(float);
  w.qT = 0; w.sT = 0;           /* K-major transpose built lazily for many-row matmuls */
  w.q4 = 0; w.s4 = 0;           /* K-major int4 repack built lazily for big reduction matmuls (MS_INT4) */
  return w;
}

int ms_model_load(const void *blob, ms_model_t *m) {
  const unsigned char *b = (const unsigned char *)blob;
  uint32_t magic; memcpy(&magic, b, 4);
  if (magic != 0x6D733031u) return -1;               /* "ms01" */
  const int *hdr = (const int *)(b + 8);             /* after magic(4) + version(4) */
  m->hidden = hdr[0]; m->n_enc = hdr[1]; m->n_dec = hdr[2]; m->n_head = hdr[3];
  m->head_dim = hdr[4]; m->rotary_dim = hdr[5]; m->ff_enc = hdr[6]; m->ff_dec = hdr[7];
  m->vocab = hdr[8]; m->bos = hdr[9]; m->eos = hdr[10]; m->max_pos = hdr[11];
  /* gs/theta live at odd offsets 57/61 (after the tied u8) — read via memcpy; a direct 32-bit
   * load would be misaligned and traps on this silicon (same reason whisper.c uses memcpy here). */
  memcpy(&m->gs, b + 8 + 12 * 4 + 1, 4);
  memcpy(&m->rope_theta, b + 8 + 12 * 4 + 1 + 4, 4);

  const int H = m->hidden, FE = m->ff_enc, FD = m->ff_dec, gs = m->gs;
  const int EL = m->n_enc, DL = m->n_dec, C2 = 2 * H;

  /* precompute rotary inverse frequencies: inv_freq[i] = theta^(-2i/rotary_dim), i=0..rd/2-1 */
  const int rh = m->rotary_dim / 2;
  for (int i = 0; i < rh; i++)
    m->inv_freq[i] = powf(m->rope_theta, -(2.0f * (float)i) / (float)m->rotary_dim);

  cur_t c = { b + 256 };

  /* FP32 conv stem + groupnorm */
  m->conv1_w = take_f32(&c, (size_t)H * 1 * MS_CONV1_K);
  m->gn_w = take_f32(&c, H); m->gn_b = take_f32(&c, H);
  m->conv2_w = take_f32(&c, (size_t)C2 * H * MS_CONV2_K); m->conv2_b = take_f32(&c, C2);
  m->conv3_w = take_f32(&c, (size_t)H * C2 * MS_CONV3_K); m->conv3_b = take_f32(&c, H);
  /* FP32 encoder norms + fc biases */
  for (int i = 0; i < EL; i++) {
    ms_enc_layer_t *L = &m->enc[i];
    L->in_ln_w = take_f32(&c, H); L->post_ln_w = take_f32(&c, H);
    L->fc1_b = take_f32(&c, FE);  L->fc2_b = take_f32(&c, H);
  }
  m->enc_ln_w = take_f32(&c, H);
  /* FP32 decoder norms + fc biases */
  for (int i = 0; i < DL; i++) {
    ms_dec_layer_t *L = &m->dec[i];
    L->in_ln_w = take_f32(&c, H); L->post_ln_w = take_f32(&c, H); L->final_ln_w = take_f32(&c, H);
    L->fc1_b = take_f32(&c, 2 * FD); L->fc2_b = take_f32(&c, H);
  }
  m->dec_ln_w = take_f32(&c, H);
  /* Q8_0 block */
  m->embed = take_q8(&c, m->vocab, H, gs);
  for (int i = 0; i < EL; i++) {
    ms_enc_layer_t *L = &m->enc[i];
    L->q = take_q8(&c, H, H, gs); L->k = take_q8(&c, H, H, gs);
    L->v = take_q8(&c, H, H, gs); L->o = take_q8(&c, H, H, gs);
    L->fc1 = take_q8(&c, FE, H, gs); L->fc2 = take_q8(&c, H, FE, gs);
  }
  for (int i = 0; i < DL; i++) {
    ms_dec_layer_t *L = &m->dec[i];
    L->q = take_q8(&c, H, H, gs);  L->k = take_q8(&c, H, H, gs);
    L->v = take_q8(&c, H, H, gs);  L->o = take_q8(&c, H, H, gs);
    L->cq = take_q8(&c, H, H, gs); L->ck = take_q8(&c, H, H, gs);
    L->cv = take_q8(&c, H, H, gs); L->co = take_q8(&c, H, H, gs);
    L->fc1 = take_q8(&c, 2 * FD, H, gs); L->fc2 = take_q8(&c, H, FD, gs);
  }
  return 0;
}

/* ------------------------------------------------------------------------------- primitive ops -- */
static void matvec_q8_scalar(float *y, const float *x, const msq8_t *W, const float *bias,
                             int out, int in, int gs) {
  const int gpr = in / gs;
  for (int o = 0; o < out; o++) {
    const int8_t *q = W->q + (size_t)o * in;
    const float *s = W->s + (size_t)o * gpr;
    float acc = bias ? bias[o] : 0.0f;
    for (int g = 0; g < gpr; g++) {
      const int8_t *qq = q + g * gs;
      const float *xx = x + g * gs;
      float part = 0.0f;
      for (int k = 0; k < gs; k++) part += (float)qq[k] * xx[k];
      acc += part * s[g];
    }
    y[o] = acc;
  }
}

#if MS_USE_RVV
#include <riscv_vector.h>
/* Hand-written RVV inner kernel (moonshine_rvv.S): sum_i (q[i]*s[i/gs])*x[i] for one weight row. */
extern float ms_matrow_q8(const int8_t *q, const float *x, const float *s, int in, int gs);
static void matvec_q8_rvv(float *y, const float *x, const msq8_t *W, const float *bias,
                          int out, int in, int gs) {
  const int gpr = in / gs;
  for (int o = 0; o < out; o++)
    y[o] = (bias ? bias[o] : 0.0f) + ms_matrow_q8(W->q + (size_t)o * in, x, W->s + (size_t)o * gpr, in, gs);
}
#endif

/* One hart's matmul block: output rows [r0,r1) x output columns [o0,o1). */
typedef struct {
  float *out_mat; const float *in_mat; const msq8_t *W; const float *bias;
  int out, in, gs, r0, r1, o0, o1;
} mm_blk_t;

#if MS_USE_RVV
/* ---- register-tiled outer-product GEMM (whisper WHISPER_TILED, silicon-proven) ------------------
 * For MANY-ROW matmuls (encoder / cross-KV) the reduction kernel re-reads weights per row and does a
 * vfredusum per output. Instead: transpose weights to K-major once, then stream them and reuse each
 * weight vector across MR=4 output rows held in vector-register accumulators (no per-output reduction,
 * ~MR-fewer weight streams). int8->f32 via the vwcvt chain proven on this silicon. */
#ifndef MS_TILE_MR
#define MS_TILE_MR 4
#endif
static void wq8_build_tiled(msq8_t *W, int out, int in, int gs) {
  if (W->qT) return;                       /* built once, on hart 0 before any fork */
  const int gpr = in / gs;
  int8_t *qT = (int8_t *)xmalloc((size_t)in * out);
  float  *sT = (float  *)xmalloc((size_t)gpr * out * sizeof(float));
  for (int o = 0; o < out; o++) {
    const int8_t *qo = W->q + (size_t)o * in;
    for (int k = 0; k < in; k++) qT[(size_t)k * out + o] = qo[k];
    const float *so = W->s + (size_t)o * gpr;
    for (int g = 0; g < gpr; g++) sT[(size_t)g * out + o] = so[g];
  }
  __sync_synchronize();
  W->qT = qT; W->sT = sT;
}
static void gemm_tiled_mr4(float *o0p, float *o1p, float *o2p, float *o3p,
                           const float *x0, const float *x1, const float *x2, const float *x3,
                           const int8_t *qT, const float *sT, const float *bias,
                           int N, int K, int gs, int oc0, int oc1) {
  for (int o = oc0; o < oc1;) {
    size_t vl = __riscv_vsetvl_e32m4((size_t)(oc1 - o));
    vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(0.0f, vl), a1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t a2 = __riscv_vfmv_v_f_f32m4(0.0f, vl), a3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t sv = a0; int cg = -1;
    for (int k = 0; k < K; k++) {
      int g = k / gs;
      if (g != cg) { sv = __riscv_vle32_v_f32m4(sT + (size_t)g * N + o, vl); cg = g; }
      vint8m1_t w8 = __riscv_vle8_v_i8m1(qT + (size_t)k * N + o, vl);
      vint32m4_t w32 = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwcvt_x_x_v_i16m2(w8, vl), vl);
      vfloat32m4_t wf = __riscv_vfmul_vv_f32m4(__riscv_vfcvt_f_x_v_f32m4(w32, vl), sv, vl);
      a0 = __riscv_vfmacc_vf_f32m4(a0, x0[k], wf, vl);
      a1 = __riscv_vfmacc_vf_f32m4(a1, x1[k], wf, vl);
      a2 = __riscv_vfmacc_vf_f32m4(a2, x2[k], wf, vl);
      a3 = __riscv_vfmacc_vf_f32m4(a3, x3[k], wf, vl);
    }
    if (bias) {
      vfloat32m4_t bv = __riscv_vle32_v_f32m4(bias + o, vl);
      a0 = __riscv_vfadd_vv_f32m4(a0, bv, vl); a1 = __riscv_vfadd_vv_f32m4(a1, bv, vl);
      a2 = __riscv_vfadd_vv_f32m4(a2, bv, vl); a3 = __riscv_vfadd_vv_f32m4(a3, bv, vl);
    }
    __riscv_vse32_v_f32m4(o0p + o, a0, vl); __riscv_vse32_v_f32m4(o1p + o, a1, vl);
    __riscv_vse32_v_f32m4(o2p + o, a2, vl); __riscv_vse32_v_f32m4(o3p + o, a3, vl);
    o += (int)vl;
  }
}
static void gemm_tiled_mr1(float *o0p, const float *x0, const int8_t *qT, const float *sT,
                           const float *bias, int N, int K, int gs, int oc0, int oc1) {
  for (int o = oc0; o < oc1;) {
    size_t vl = __riscv_vsetvl_e32m4((size_t)(oc1 - o));
    vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t sv = a0; int cg = -1;
    for (int k = 0; k < K; k++) {
      int g = k / gs;
      if (g != cg) { sv = __riscv_vle32_v_f32m4(sT + (size_t)g * N + o, vl); cg = g; }
      vint8m1_t w8 = __riscv_vle8_v_i8m1(qT + (size_t)k * N + o, vl);
      vint32m4_t w32 = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwcvt_x_x_v_i16m2(w8, vl), vl);
      vfloat32m4_t wf = __riscv_vfmul_vv_f32m4(__riscv_vfcvt_f_x_v_f32m4(w32, vl), sv, vl);
      a0 = __riscv_vfmacc_vf_f32m4(a0, x0[k], wf, vl);
    }
    if (bias) a0 = __riscv_vfadd_vv_f32m4(a0, __riscv_vle32_v_f32m4(bias + o, vl), vl);
    __riscv_vse32_v_f32m4(o0p + o, a0, vl);
    o += (int)vl;
  }
}
/* Cache-block the OUTPUT COLUMNS so the weight slice stays resident across every row tile.
 *
 * Register tiling (MR=4) only reuses a weight vector across 4 rows, so the full weight matrix is
 * re-streamed ceil(nrows/4) times — 6 times for a 23-position encoder, i.e. 35.8 MB of the ~170 MB
 * an inference moves. Blocking the o-range so that one K-major slice (K * width bytes) fits in the
 * 256 KB cache lets all row tiles hit it, dropping that to ONE pass over the weights.
 *
 * Width is a multiple of 32 outputs so each 64-byte line of qT (which holds consecutive o for a
 * given k) is fully consumed rather than half-wasted. */
#ifndef MS_CACHE_BYTES
#define MS_CACHE_BYTES (256u * 1024u)
#endif
#ifndef MS_CACHE_FRAC_NUM        /* fraction of the cache to spend on the weight slice; the rest is */
#define MS_CACHE_FRAC_NUM 1u     /* activations (MR*K floats), the output tile and the sT scales.   */
#endif
#ifndef MS_CACHE_FRAC_DEN
#define MS_CACHE_FRAC_DEN 2u
#endif
/* noinline is load-bearing: with an output-column loop around these intrinsics, GCC 13.2's
 * vsetvl-insertion pass rewrites which register supplies vl (vsetvli zero,s9 instead of a5,a5) and
 * the encoder silently computes wrong values. Same pass that forced ms_matrow_q8 into asm. Keeping
 * this function's body exactly as-is, and blocking in the CALLER, reproduces the known-good codegen. */
__attribute__((noinline)) static void gemm_tiled_block(const mm_blk_t *b) {
  const int N = b->out, K = b->in, gs = b->gs;
  const int8_t *qT = b->W->qT; const float *sT = b->W->sT;
  float *out = b->out_mat; const float *x = b->in_mat;
  int r = b->r0;
  for (; r + 4 <= b->r1; r += 4)
    gemm_tiled_mr4(out + (size_t)(r+0)*N, out + (size_t)(r+1)*N, out + (size_t)(r+2)*N, out + (size_t)(r+3)*N,
                   x + (size_t)(r+0)*K, x + (size_t)(r+1)*K, x + (size_t)(r+2)*K, x + (size_t)(r+3)*K,
                   qT, sT, b->bias, N, K, gs, b->o0, b->o1);
  for (; r < b->r1; r++)
    gemm_tiled_mr1(out + (size_t)r*N, x + (size_t)r*K, qT, sT, b->bias, N, K, gs, b->o0, b->o1);
}
#endif /* MS_USE_RVV */

#if MS_INT4
/* Re-quantize a Q8_0 weight matrix to int4 [-7,7] (per MS_INT4_GS K-group) in K-MAJOR SPLIT-HALF
 * layout for the outer-product kernel. Per tap k the N outputs are packed nblk*16 bytes; within each
 * 32-output block, byte i (i=0..15) holds output (b*32+i) in its low nibble and (b*32+16+i) in its
 * high nibble — so one byte-vector load + two shifts yields two CONTIGUOUS output vectors (no
 * deinterleave). s4[g*N + o] = per (K-group, output) scale. One-time, single-threaded (before fork). */
static void wq8_build_int4(msq8_t *W, int out, int in, int gs) {
  if (W->q4) return;
#if MS_USE_RVV && !MS_INT4_SCALAR
  /* gemm_int4_row drives a 16-output half-block with ONE e32m2 group, which needs VLEN >= 256. On a
   * narrower VLEN vsetvl would quietly return 8 and half of every block would go uncomputed — fail
   * loudly instead. (The older LMUL=4 form tolerated VLEN=128; this one does not.) */
  if (__riscv_vsetvl_e32m2(16) < 16) {
    printf("[moonshine] FATAL: MS_INT4 RVV path needs VLEN>=256 (e32m2 vl=%u)\n",
           (unsigned)__riscv_vsetvl_e32m2(16));
    exit(1);
  }
#endif
  const int gpr8 = in / gs;
  const int g4n = MS_INT4_GS, gpr4 = in / g4n;
  const int nblk = (out + 31) / 32;
  const size_t rowstride = (size_t)nblk * 16;               /* packed bytes per tap k */
  int8_t *q4 = (int8_t *)xmalloc((size_t)in * rowstride);
  float  *s4 = (float  *)xmalloc((size_t)gpr4 * out * sizeof(float));
  memset(q4, 0, (size_t)in * rowstride);                     /* nibbles OR'd in; init 0 */
  for (int o = 0; o < out; o++) {
    const int8_t *qo = W->q + (size_t)o * in;
    const float  *so = W->s + (size_t)o * gpr8;
    const int blk = o / 32, local = o % 32;
    const int byte_idx = (local < 16) ? local : (local - 16);
    const int hi_nib = (local >= 16);
    for (int g = 0; g < gpr4; g++) {
      const int base = g * g4n;
      float vmax = 0.0f;
      for (int i = 0; i < g4n; i++) {
        float v = (float)qo[base + i] * so[(base + i) / gs];
        float a = v < 0 ? -v : v; if (a > vmax) vmax = a;
      }
      const float inv = (vmax > 0.0f) ? (7.0f / vmax) : 0.0f;
      s4[(size_t)g * out + o] = vmax / 7.0f;
      for (int i = 0; i < g4n; i++) {
        const int k = base + i;
        float v = (float)qo[k] * so[k / gs];
        int nib = (int)lrintf(v * inv); if (nib > 7) nib = 7; else if (nib < -7) nib = -7;
        int8_t *byte = &q4[(size_t)k * rowstride + (size_t)blk * 16 + byte_idx];
        *byte |= (int8_t)(hi_nib ? ((nib & 0xF) << 4) : (nib & 0xF));
      }
    }
  }
  __sync_synchronize();
  W->q4 = q4; W->s4 = s4;
}

#ifndef MS_INT4_SCALAR
#define MS_INT4_SCALAR 0   /* 1 = scalar int4 reference (packing check); 0 = RVV */
#endif
/* int4 outer-product row: out_row[o] = bias[o] + sum_k x[k] * dequant(w[k][o]). No reduction, m4,
 * contiguous loads/stores. Processes output cols [o0,o1) (32-aligned) for one x row. */
static void gemm_int4_row(float *out_row, const float *x, const int8_t *q4, const float *s4,
                          const float *bias, int N, int K, int gs4, int o0, int o1) {
  const int nblk = (N + 31) / 32;
  const size_t rowstride = (size_t)nblk * 16;
#if MS_INT4_SCALAR || !MS_USE_RVV
  for (int blk = o0; blk < o1; blk += 32) {
    int lo_n = N - blk;      if (lo_n > 16) lo_n = 16; else if (lo_n < 0) lo_n = 0;
    int hi_n = N - blk - 16; if (hi_n > 16) hi_n = 16; else if (hi_n < 0) hi_n = 0;
    for (int j = 0; j < lo_n; j++) out_row[blk + j]      = bias ? bias[blk + j] : 0.0f;
    for (int j = 0; j < hi_n; j++) out_row[blk + 16 + j] = bias ? bias[blk + 16 + j] : 0.0f;
    for (int k = 0; k < K; k++) {
      const int g = k / gs4;
      const int8_t *wb = q4 + (size_t)k * rowstride + (size_t)(blk / 32) * 16;
      for (int j = 0; j < lo_n; j++) {
        int lo = (int)(int8_t)(wb[j] << 4) >> 4;
        out_row[blk + j] += x[k] * (float)lo * s4[(size_t)g * N + blk + j];
      }
      for (int j = 0; j < hi_n; j++) {
        int hi = (int)wb[j] >> 4;
        out_row[blk + 16 + j] += x[k] * (float)hi * s4[(size_t)g * N + blk + 16 + j];
      }
    }
  }
#else
  /* Two f32m2 accumulators (lo/hi halves of each 32-output block). i8mf2->i16m1->i32m2->f32m2 chain
   * (the vwcvt form proven on this silicon; sign-extend nibbles via vsll+vsra).
   *
   * LMUL=2, NOT 4. A half-block is at most 16 outputs, and at VLEN=256 e32m2 covers exactly those 16
   * lanes, so LMUL=4 was reserving a 4-register group per variable to use half its lanes.
   * NOT a spill fix — that was the original hypothesis and it is wrong: measured at -O3, both forms
   * emit 88 vector instructions in this function with ZERO stack spills, so GCC handled the m4
   * pressure fine (alo/ahi/slo/shi live across the k loop = 16 of 32 registers, plus the widening
   * chain). The reason to prefer m2 is execution cost: an LMUL=4 op with vl=16 still names a 4-register
   * group, and on implementations that sequence over the group rather than over vl it costs twice
   * what the same work needs. Neutral if the hardware scales with vl. Verify on silicon before
   * assuming a win; correctness is unaffected (Spike golden: 7/7 tokens). */
  for (int blk = o0; blk < o1; blk += 32) {
    int lo_n = N - blk;      if (lo_n > 16) lo_n = 16; else if (lo_n < 0) lo_n = 0;
    int hi_n = N - blk - 16; if (hi_n > 16) hi_n = 16; else if (hi_n < 0) hi_n = 0;
    const size_t vlo = __riscv_vsetvl_e32m2((size_t)lo_n);
    const size_t vhi = hi_n > 0 ? __riscv_vsetvl_e32m2((size_t)hi_n) : 1;
    vfloat32m2_t alo = __riscv_vfmv_v_f_f32m2(0.0f, vlo);
    vfloat32m2_t ahi = __riscv_vfmv_v_f_f32m2(0.0f, vhi);
    vfloat32m2_t slo = alo, shi = alo; int cg = -1;
    const int bblk = blk / 32;
    for (int k = 0; k < K; k++) {
      const int g = k / gs4;
      if (g != cg) {
        slo = __riscv_vle32_v_f32m2(s4 + (size_t)g * N + blk, vlo);
        if (hi_n > 0) shi = __riscv_vle32_v_f32m2(s4 + (size_t)g * N + blk + 16, vhi);
        cg = g;
      }
      const int8_t *wb = q4 + (size_t)k * rowstride + (size_t)bblk * 16;
      vint8mf2_t v8 = __riscv_vle8_v_i8mf2(wb, vlo);
      vint8mf2_t lo8 = __riscv_vsra_vx_i8mf2(__riscv_vsll_vx_i8mf2(v8, 4, vlo), 4, vlo); /* low nibble */
      vfloat32m2_t lof = __riscv_vfcvt_f_x_v_f32m2(
          __riscv_vwcvt_x_x_v_i32m2(__riscv_vwcvt_x_x_v_i16m1(lo8, vlo), vlo), vlo);
      alo = __riscv_vfmacc_vf_f32m2(alo, x[k], __riscv_vfmul_vv_f32m2(lof, slo, vlo), vlo);
      if (hi_n > 0) {
        vint8mf2_t hi8 = __riscv_vsra_vx_i8mf2(v8, 4, vhi);                              /* high nibble */
        vfloat32m2_t hif = __riscv_vfcvt_f_x_v_f32m2(
            __riscv_vwcvt_x_x_v_i32m2(__riscv_vwcvt_x_x_v_i16m1(hi8, vhi), vhi), vhi);
        ahi = __riscv_vfmacc_vf_f32m2(ahi, x[k], __riscv_vfmul_vv_f32m2(hif, shi, vhi), vhi);
      }
    }
    if (bias) {
      alo = __riscv_vfadd_vv_f32m2(alo, __riscv_vle32_v_f32m2(bias + blk, vlo), vlo);
      if (hi_n > 0) ahi = __riscv_vfadd_vv_f32m2(ahi, __riscv_vle32_v_f32m2(bias + blk + 16, vhi), vhi);
    }
    __riscv_vse32_v_f32m2(out_row + blk, alo, vlo);
    if (hi_n > 0) __riscv_vse32_v_f32m2(out_row + blk + 16, ahi, vhi);
  }
#endif
}
#endif /* MS_INT4 */

/* Pure-scalar reduction block: no RVV, no tiling, no int4. Always compiled, so the dual-core ladder
 * can validate the hart-1 mechanism with the vector unit entirely out of the picture. */
static void matmul_block_scalar(const mm_blk_t *b) {
  const int gpr = b->in / b->gs;
  for (int r = b->r0; r < b->r1; r++) {
    const float *xr = b->in_mat + (size_t)r * b->in;
    float *y = b->out_mat + (size_t)r * b->out;
    for (int o = b->o0; o < b->o1; o++) {
      const int8_t *qq = b->W->q + (size_t)o * b->in;
      const float *ss = b->W->s + (size_t)o * gpr;
      float acc = b->bias ? b->bias[o] : 0.0f;
      for (int g = 0; g < gpr; g++) {
        float part = 0.0f;
        for (int k = 0; k < b->gs; k++) part += (float)qq[g * b->gs + k] * xr[g * b->gs + k];
        acc += part * ss[g];
      }
      y[o] = acc;
    }
  }
}

/* Compute one hart's block: tiled (K-major transpose built) or the per-row reduction kernel. */
static void matmul_block(const mm_blk_t *b) {
#if MS_INT4
  if (b->W->q4) {   /* int4 outer-product path (classifier / fc1), one x row at a time */
    for (int r = b->r0; r < b->r1; r++)
      gemm_int4_row(b->out_mat + (size_t)r * b->out, b->in_mat + (size_t)r * b->in,
                    b->W->q4, b->W->s4, b->bias, b->out, b->in, MS_INT4_GS, b->o0, b->o1);
    return;
  }
#endif
#if MS_USE_RVV
  if (b->W->qT) { gemm_tiled_block(b); return; }
  const int gpr = b->in / b->gs;
  for (int r = b->r0; r < b->r1; r++) {
    float *y = b->out_mat + (size_t)r * b->out;
    const float *x = b->in_mat + (size_t)r * b->in;
    for (int o = b->o0; o < b->o1; o++)
      y[o] = (b->bias ? b->bias[o] : 0.0f)
           + ms_matrow_q8(b->W->q + (size_t)o * b->in, x, b->W->s + (size_t)o * gpr, b->in, b->gs);
  }
#else
  const int gpr = b->in / b->gs;   /* scalar reduction over [r0,r1) x [o0,o1) (unused: no dual-core without RVV) */
  for (int r = b->r0; r < b->r1; r++) {
    const int8_t *qq; const float *ss;
    for (int o = b->o0; o < b->o1; o++) {
      qq = b->W->q + (size_t)o * b->in; ss = b->W->s + (size_t)o * gpr;
      float acc = b->bias ? b->bias[o] : 0.0f;
      for (int g = 0; g < gpr; g++) {
        float part = 0.0f;
        for (int k = 0; k < b->gs; k++) part += (float)qq[g*b->gs+k] * b->in_mat[(size_t)r*b->in + g*b->gs+k];
        acc += part * ss[g];
      }
      b->out_mat[(size_t)r * b->out + o] = acc;
    }
  }
#endif
}

/* ---- 2-hart split of the matmul + conv stem (MS_DUALCORE) --------------------------------------
 * Dispatcher: single producer (hart 0), single consumer (hart 1), ONE job slot. No locks, no deques,
 * no CLINT. crt0 sends every non-boot hart into `__main`; ours parks hart 1 in a spin on a monotonic
 * `go` counter, and it answers with a monotonic `done` counter.
 *
 * Why not thread-lib/hthread.c, which this used to call: hart 1 never picked its task up there — and
 * it reproduces on Spike -p2, not just on silicon, so it is a runtime bug and not a chip quirk. Its
 * hthread_join() also spins re-taking deque_locks[1] and re-poking hart 1's MSIP, i.e. hart 0
 * hammering the exact lock and interconnect hart 1 needs to get started. hthread.c is no longer
 * linked into this target.
 *
 * Why the EARLIER hand-rolled handshakes wedged, and why this one is different: they published the
 * job with PLAIN volatile stores — fn, then arg, then the flag — and no release fence. `volatile`
 * orders the COMPILER, not the machine; RVWMO lets hart 1 observe the flag before fn/arg, which is
 * exactly the recorded failure (`mcause=1 mepc=0` = jumped to a NULL fn; then, after publishing fn
 * by amoswap, `mcause=5 mtval=0x022c8ffc` = dereferenced the still-plain arg). That is a MISSING
 * RELEASE FENCE, not the "plain cross-hart stores don't propagate" it was read as. Here fn/arg are
 * written first and published by a SEQ_CST increment of `go` that hart 1 reads with ACQUIRE — a real
 * release/acquire pair, so everything written before the bump is visible once the bump is seen. The
 * same pairing (`done` bump / acquire in ms_h1_wait) publishes hart 1's results back.
 *
 * Boot ordering: hart 1 reaches __main while hart 0 is still copying .data and zeroing .bss, so it
 * touches NO global until hart 0 publishes g_dc_boot from ms_dualcore_init() — otherwise hart 0's
 * bss_init would race every flag this depends on.
 *
 * On top of the dispatcher:
 *  - BOUNDED wait + a hart-1 trap catcher. A wedge prints one `DC-WEDGE` line naming the fork, how
 *    far hart 1 got (stage beacon) and any trap it took (mcause/mepc/mtval); hart 0 then computes
 *    hart 1's block itself and drops to single-core, so the run finishes with a correct result
 *    instead of hanging the chip.
 *  - RAMPABLE SPLIT (MS_H1_PCT). hart 1 gets MS_H1_PCT percent of each offloaded matmul — rows when
 *    n>=2, else output columns — and only matmuls with n*out >= MS_MM_MIN_WORK fork at all. Raise
 *    MS_H1_PCT as it proves stable. MS_H1_PCT=0 forks with an EMPTY hart-1 block: full-rate fork/join
 *    with zero hart-1 compute, which separates a control-plane wedge from a hart-1-compute wedge. */
#ifndef MS_DUALCORE
#define MS_DUALCORE 0
#endif
#ifndef MS_DUALCORE_CONV
#define MS_DUALCORE_CONV 1     /* also split the conv stem (MOONSHINE_DUALCORE_CONV=OFF -> 0) */
#endif
#if MS_DUALCORE
#if !MS_USE_RVV
#error "MS_DUALCORE requires MS_USE_RVV"
#endif

#ifndef MS_MM_MIN_WORK
#define MS_MM_MIN_WORK 16384   /* only offload matmuls at least this big (n*out); small ones stay solo */
#endif
#ifndef MS_H1_PCT
#define MS_H1_PCT 25           /* percent of each offloaded matmul/conv computed by hart 1 (0..100) */
#endif
#if (MS_H1_PCT) < 0 || (MS_H1_PCT) > 100
#error "MS_H1_PCT must be 0..100"
#endif
#ifndef MS_CONV_MIN_COUT
#define MS_CONV_MIN_COUT 8     /* don't fork a conv with fewer output channels than this */
#endif
/* Bounded wait for one fork. The largest single kernel here (conv1 over ~3 s of audio) is ~3.3 B
 * cycles across BOTH halves, so 8 B cycles (~10 s @ 750 MHz) is far above any healthy fork and far
 * below "the chip is hung". The self-test jobs are all <100 M cycles, so they get a much shorter
 * bound — a probe wedge should report in under a second, not after ten. */
#ifndef MS_DC_WATCHDOG_CYCLES
#define MS_DC_WATCHDOG_CYCLES 8000000000ULL
#endif
#ifndef MS_DC_SELFTEST_WATCHDOG_CYCLES
#define MS_DC_SELFTEST_WATCHDOG_CYCLES 750000000ULL
#endif
/* Cycles hart 0 backs off between polls of `done` while hart 1 works. Jobs run for millions of
 * cycles, so polling every few thousand costs nothing in join latency and keeps hart 0 off the
 * interconnect while hart 1 streams weights from DRAM. */
#ifndef MS_DC_POLL_BACKOFF_NOPS
#define MS_DC_POLL_BACKOFF_NOPS 2048
#endif
/* Probe hart 1 with escalating real RVV matmuls at startup and verify them against hart 0. */
#ifndef MS_DC_SELFTEST
#define MS_DC_SELFTEST 1
#endif
/* Run the real workload's dual-core forks with the SCALAR block on both harts (no RVV/tiled/int4).
 * The bottom rung of the ladder, applied in situ: slow, but it isolates the dispatch mechanism. */
#ifndef MS_DC_SCALAR
#define MS_DC_SCALAR 0
#endif

#define MS_DC_MM    1u
#define MS_DC_CONV  2u
#define MS_DC_PROBE 3u

#define MS_DC_BOOT_TOKEN 0x4D534443u    /* "MSDC": hart 0 says the globals below are initialized */
#ifndef MS_DC_BOOT_TIMEOUT_CYCLES
#define MS_DC_BOOT_TIMEOUT_CYCLES 500000000ULL
#endif

/* Beacon + trap sink. GLOBAL (not static) because the naked trap stub reaches them with `la`. */
volatile uint32_t ms_h1_stage;                    /* 0 issued, 1 entered, 2 finished, 9 TRAPPED */
volatile uint64_t ms_h1_mcause, ms_h1_mepc, ms_h1_mtval;

/* Hart 0 writes `go`, hart 1 writes `done`/`hb`; keep the two directions on separate 64 B lines so
 * the producer's publish does not invalidate the line the consumer is answering on. */
static volatile int      g_dc_active;             /* 1 = hart 1 is up and trusted */
static volatile uint32_t g_dc_boot;               /* AMO: .bss/.data are initialized, globals usable */
static void (* volatile g_h1_fn)(void *);         /* job slot; published by the g_h1_go bump */
static void * volatile  g_h1_arg;
static volatile uint32_t g_h1_go   __attribute__((aligned(64)));   /* AMO: jobs published (hart 0) */
static volatile uint32_t g_h1_alive __attribute__((aligned(64)));  /* AMO: hart 1 reached dispatch */
static volatile uint32_t g_h1_done;               /* AMO: jobs completed (hart 1) */
static volatile uint32_t g_h1_hb;                 /* AMO: hart-1 heartbeat (throttled, see below) */
static uint32_t g_dc_kind, g_dc_taskno;           /* hart 0 only: what the outstanding fork is */
static mm_blk_t g_mm_h1 __attribute__((aligned(64)));   /* descriptor in BSS, own cache line */

static inline void ms_relax(void) { for (int i = 0; i < 16; i++) __asm__ volatile("nop"); }

/* ---- cross-hart cache maintenance (MS_DC_FLUSH) -------------------------------------------------
 * A `fence` ORDERS accesses; it does not PROPAGATE them. If the two harts' L1s are not coherent,
 * hart 1's output stays dirty in its L1 while hart 0 keeps reading its own stale copy — which is
 * what silicon shows: the AMO handshake (go/done/alive) works, yet hart 1's matmul output reads back
 * as hart 0's pre-write value. AMOs get through because they are performed at a coherent point; plain
 * lines do not. This is the same family as the C2C shared-region quirk in the bug log, and the same
 * remedy: force-eviction by walking a larger-than-cache buffer. The cache-controller flush register
 * (0x02010200) does NOT evict on this silicon, and cbo.inval does not invalidate (see the DMA
 * finding), so the walk is the only tool available.
 *
 * hart 1 flushes after writing its block (push it out of L1); hart 0 flushes before reading it
 * (drop its own stale lines). Expensive — measure before shipping it on. */
#ifndef MS_DC_FLUSH
#define MS_DC_FLUSH 0
#endif
#if MS_DC_FLUSH
/* Sized to exceed the cache being evicted. Silicon cost is brutal — the smallest probe went from
 * 0.58 M to 42 M cycles with 256 KB x 2 passes on BOTH harts, i.e. ~40 M cycles of pure flush per
 * fork. That rules flushing out at per-matmul granularity (thousands of forks per inference) and is
 * the argument for forking once per layer / per inference instead. Tune down to find the smallest
 * walk that still restores correctness — if only the L1 needs evicting, this can be far smaller. */
#ifndef MS_DC_EVICT_BYTES
#define MS_DC_EVICT_BYTES (256u * 1024u)
#endif
#ifndef MS_DC_EVICT_LINE
#define MS_DC_EVICT_LINE 64u
#endif
#ifndef MS_DC_EVICT_PASSES
#define MS_DC_EVICT_PASSES 2u
#endif
_Static_assert((MS_DC_EVICT_LINE & (MS_DC_EVICT_LINE - 1u)) == 0u, "line size must be a power of two");
_Static_assert((MS_DC_EVICT_BYTES % MS_DC_EVICT_LINE) == 0u, "evict bytes must be a line multiple");
/* Per-hart buffers: if both harts walked the same one they would evict each other's lines and, worse,
 * ping-pong ownership of every line they touch. */
static uint8_t g_dc_evict_buf[2][MS_DC_EVICT_BYTES] __attribute__((aligned(0x8000)));
static volatile uint8_t g_dc_evict_sink[2];
static void ms_dc_cache_flush(uint32_t hart) {
  volatile uint8_t *buf = (volatile uint8_t *)g_dc_evict_buf[hart & 1u];
  volatile uint8_t sink = g_dc_evict_sink[hart & 1u];
  __asm__ volatile("fence rw, rw" ::: "memory");
  for (uint32_t pass = 0; pass < MS_DC_EVICT_PASSES; ++pass) {
    for (uint32_t i = 0; i < (uint32_t)MS_DC_EVICT_BYTES; i += MS_DC_EVICT_LINE) {
      sink ^= buf[i];
      buf[i] = (uint8_t)(sink + (uint8_t)i + (uint8_t)pass);
    }
    __asm__ volatile("fence rw, rw" ::: "memory");
  }
  g_dc_evict_sink[hart & 1u] = sink;
  __asm__ volatile("fence rw, rw" ::: "memory");
}
#else
static inline void ms_dc_cache_flush(uint32_t hart) {
  (void)hart; __asm__ volatile("fence rw, rw" ::: "memory");
}
#endif

/* Heartbeat, bumped once every MS_DC_HB_EVERY spins rather than every spin: it only has to prove
 * hart 1 is executing, and an AMO on a hart-0-visible line every iteration would be hart 1 spamming
 * invalidations across the interconnect for the whole time hart 0 is computing its own half. */
#ifndef MS_DC_HB_EVERY
#define MS_DC_HB_EVERY 4096u
#endif
static inline void ms_h1_idle(uint32_t *spins) {
  if (++*spins >= MS_DC_HB_EVERY) { *spins = 0u; __atomic_add_fetch(&g_h1_hb, 1u, __ATOMIC_SEQ_CST); }
  ms_relax();
}

/* hart-1 trap catcher: record the trap and park in stage 9 so hart 0's watchdog can report it,
 * instead of the default handler spinning silently. Naked + 4-byte aligned (mtvec direct mode). */
__attribute__((naked, aligned(64))) static void ms_h1_trap_entry(void) {
  __asm__ volatile(
      "csrr t0, mcause\n\t"  "la t1, ms_h1_mcause\n\t"  "sd t0, 0(t1)\n\t"
      "csrr t0, mepc\n\t"    "la t1, ms_h1_mepc\n\t"    "sd t0, 0(t1)\n\t"
      "csrr t0, mtval\n\t"   "la t1, ms_h1_mtval\n\t"   "sd t0, 0(t1)\n\t"
      "li   t0, 9\n\t"       "la t1, ms_h1_stage\n\t"   "amoswap.w zero, t0, (t1)\n\t"
      "1: j 1b\n\t");
}
static inline void ms_h1_arm_trap(void) {
  __asm__ volatile("csrw mtvec, %0" :: "r"((uintptr_t)&ms_h1_trap_entry) : "memory");
}

/* hart-1 entry: crt0 branches every non-boot hart here. Only hart 1 is used. */
void __main(void) {
  uint32_t id; __asm__ volatile("csrr %0, mhartid" : "=r"(id));
  if (id != 1u) { for (;;) __asm__ volatile("wfi"); }
  ms_h1_arm_trap();                                        /* touches only a CSR and a .text address */
  /* hart 0 is still running crt0's data/bss init — do not read any global until it says otherwise.
   * The heartbeat proves hart 1 is executing even if the token/job handshake never completes, which
   * is what separates "hart 1 is dead" from "hart 1 is alive but not seeing hart 0's writes". */
  uint32_t spins = 0u;
  while (__atomic_load_n(&g_dc_boot, __ATOMIC_ACQUIRE) != MS_DC_BOOT_TOKEN) ms_h1_idle(&spins);
  __atomic_store_n(&g_h1_alive, 1u, __ATOMIC_SEQ_CST);
  uint32_t seen = 0u;
  for (;;) {
    uint32_t go;
    while ((go = __atomic_load_n(&g_h1_go, __ATOMIC_ACQUIRE)) == seen) ms_h1_idle(&spins);
    seen = go;
    /* Drop hart 1's stale lines BEFORE reading anything hart 0 wrote. Without this, hart 1 reads a
     * stale all-zero g_mm_h1 out of .bss and faults on b->W->q4 (mcause=5 mtval=0x20, observed on
     * silicon). The acquire above orders these reads; it cannot make them coherent. */
    ms_dc_cache_flush(1u);
    void (*fn)(void *) = g_h1_fn;
    void *arg = g_h1_arg;
    __atomic_store_n(&ms_h1_stage, 1u, __ATOMIC_SEQ_CST);
    if (fn) fn(arg);
    ms_dc_cache_flush(1u);                                 /* push this block's output out of L1 */
    __atomic_store_n(&ms_h1_stage, 2u, __ATOMIC_SEQ_CST);
    __atomic_store_n(&g_h1_done, go, __ATOMIC_SEQ_CST);    /* release: publishes hart 1's output */
  }
}

static void mm_worker(void *a)  { matmul_block((const mm_blk_t *)a); }
static void mm_worker_scalar(void *a) { matmul_block_scalar((const mm_blk_t *)a); }
static void nop_worker(void *a) { (void)a; }

/* Publish the job. The seq-cst bump of g_h1_go is the release that makes fn/arg — and the descriptor
 * and activation/weight buffers they point at — visible to hart 1's acquire. */
static void ms_h1_fork(void (*fn)(void *), void *arg, uint32_t kind) {
  g_dc_kind = kind; g_dc_taskno++;
  g_h1_fn = fn; g_h1_arg = arg;
  __atomic_store_n(&ms_h1_stage, 0u, __ATOMIC_SEQ_CST);
  ms_dc_cache_flush(0u);   /* push the descriptor + this op's inputs out before publishing `go` */
  __atomic_add_fetch(&g_h1_go, 1u, __ATOMIC_SEQ_CST);
}

/* Bounded join. 1 = hart 1 finished; 0 = wedged (dual-core disabled, caller must redo hart 1's block). */
static int ms_h1_wait_bounded(uint64_t budget) {
  const uint32_t want = __atomic_load_n(&g_h1_go, __ATOMIC_ACQUIRE);
  uint64_t t0; __asm__ volatile("rdcycle %0" : "=r"(t0));
  for (;;) {
    if (__atomic_load_n(&g_h1_done, __ATOMIC_ACQUIRE) == want) {
      ms_dc_cache_flush(0u);        /* drop hart 0's stale lines for the region hart 1 just wrote */
      return 1;
    }
    uint64_t t; __asm__ volatile("rdcycle %0" : "=r"(t));
    if (t - t0 > budget) break;
    for (int i = 0; i < MS_DC_POLL_BACKOFF_NOPS; i++) __asm__ volatile("nop");
  }
  /* Sample the heartbeat across a gap. It advances only while hart 1 spins in the dispatcher, so it
   * separates "stuck inside the kernel" from "finished, but its `done` never reached hart 0". */
  const uint32_t stage = __atomic_load_n(&ms_h1_stage, __ATOMIC_ACQUIRE);
  const uint32_t hbA = __atomic_load_n(&g_h1_hb, __ATOMIC_ACQUIRE);
  for (int i = 0; i < 4000000; i++) __asm__ volatile("nop");
  const uint32_t hbB = __atomic_load_n(&g_h1_hb, __ATOMIC_ACQUIRE);
  printf("[moonshine] DC-WEDGE taskno=%u kind=%s stage=%u (%s) go=%u done=%u hb=%u->%u (%s)\n",
         (unsigned)g_dc_taskno,
         g_dc_kind == MS_DC_MM ? "matmul" : (g_dc_kind == MS_DC_CONV ? "conv" : "probe"),
         (unsigned)stage,
         stage == 0u ? "never picked the job up" :
         stage == 1u ? "entered the kernel, never returned" :
         stage == 2u ? "finished the kernel" :
         stage == 9u ? "TRAPPED" : "?",
         (unsigned)want, (unsigned)__atomic_load_n(&g_h1_done, __ATOMIC_ACQUIRE),
         (unsigned)hbA, (unsigned)hbB,
         hbB != hbA ? "SPINNING in the dispatcher" : "NOT spinning: in-kernel or dead");
  if (stage == 9u)
    printf("[moonshine] DC-WEDGE hart1 trap: mcause=%lu mepc=0x%lx mtval=0x%lx\n",
           (unsigned long)ms_h1_mcause, (unsigned long)ms_h1_mepc, (unsigned long)ms_h1_mtval);
  printf("[moonshine] DC-WEDGE -> hart 0 finishes the block; dual-core DISABLED for the rest of the run\n");
  g_dc_active = 0;
  return 0;
}
static int ms_h1_wait(void) { return ms_h1_wait_bounded((uint64_t)MS_DC_WATCHDOG_CYCLES); }

/* Split point: hart 0 takes [0,mid), hart 1 takes [mid,total). `align` keeps the boundary on a tile
 * / packing / cache-line multiple. MS_H1_PCT=0 -> mid==total -> hart 1's block is empty. */
static inline int ms_split(int total, int align) {
  long mid = (long)total - (((long)total * (long)MS_H1_PCT + 50) / 100);
  if (align > 1) mid = (mid / align) * align;
  if (mid < 0) mid = 0;
  if (mid > total) mid = total;
  return (int)mid;
}

#if MS_DC_SELFTEST
/* ---- dual-core ladder --------------------------------------------------------------------------
 * One rung at a time, each adding exactly ONE capability, so a failure names its own cause:
 *
 *   scalar  -> hart 1 runs a plain C reduction. No RVV at all. If this fails, the problem is the
 *              hart-1 dispatch/memory mechanism and nothing to do with the vector unit.
 *   rvv     -> same split, same data, but the RVV reduction kernel. Fails only here => vector unit.
 *   tiled   -> the register-tiled GEMM (needs a hart-0-built K-major transpose, so it also tests
 *              publishing a freshly written weight buffer to hart 1).
 *
 * Each rung is the PRODUCTION shape: hart 0 computes [0,mid), hart 1 computes [mid,out), boundary on
 * a 64 B line, result compared against a single-core reference over the whole output.
 *
 * The descriptor and the weight handle live in BSS, never on hart 0's stack. An earlier version of
 * this probe put `msq8_t W` on the stack and handed hart 1 `&W`; the whisper bring-up had already
 * learned to move worker descriptors off the hart-0 stack and I reintroduced it. */
static msq8_t  g_probe_W;                                  /* BSS, not hart 0's stack */
static mm_blk_t g_probe_ref;

typedef struct { int n, out, in; } probe_sz_t;

/* Run one rung: k0/k1 select each hart's kernel independently (0 = scalar C, 1 = RVV), so RVV can be
 * introduced on ONE hart at a time. Returns 1 ok, 0 failed (and prints why).
 *
 * Mixed rungs need a TOLERANCE compare: the scalar and RVV kernels sum in different orders, so their
 * results differ in the last bits even when both are correct. A rounding difference shows up as a
 * tiny max-rel; a real failure (stale/garbage data) shows up as ~1.0. The number is always printed so
 * the two are never confused. */
#ifndef MS_DC_TOL
#define MS_DC_TOL 1.0e-3f
#endif
static int ms_dc_rung(const char *name, int k0, int k1, int tiled,
                      probe_sz_t z, int gs, int8_t *q, float *s, float *x, float *yr, float *y1) {
  const int n = z.n, out = z.out, in = z.in;
  printf("[moonshine] dc-ladder: run %-9s n=%-3d out=%-5d in=%-4d (%luKB weights)\n",
         name, n, out, in, (unsigned long)(((size_t)out * in) >> 10));

  g_probe_W.q = q; g_probe_W.s = s;
  g_probe_W.qT = 0; g_probe_W.sT = 0; g_probe_W.q4 = 0; g_probe_W.s4 = 0;
#if MS_USE_RVV
  if (tiled) wq8_build_tiled(&g_probe_W, out, in, gs);     /* hart-0-written buffer hart 1 must see */
#else
  (void)tiled;
#endif

  uint64_t rt0; __asm__ volatile("rdcycle %0" : "=r"(rt0));
  /* Single-core reference into yr, using HART 0's kernel. Run once UNTIMED first: it is the first
   * read of these weights, so it would eat every cold miss while the dual run re-reads them warm —
   * which is how an earlier version reported a 3.64x "speedup" two cores cannot produce. */
  g_probe_ref = (mm_blk_t){ yr, x, &g_probe_W, 0, out, in, gs, 0, n, 0, out };
  if (k0) matmul_block(&g_probe_ref); else matmul_block_scalar(&g_probe_ref);
  __asm__ volatile("rdcycle %0" : "=r"(rt0));                    /* restart the clock, caches warm */
  if (k0) matmul_block(&g_probe_ref); else matmul_block_scalar(&g_probe_ref);
  uint64_t rt1; __asm__ volatile("rdcycle %0" : "=r"(rt1));
  printf("[moonshine] dc-ladder:   .. 1-core reference done (%lu cycles)\n", (unsigned long)(rt1 - rt0));

  /* Dual-core into y1, production split. Nobody pre-writes the other hart's columns. */
  const int mid = ms_split(out, 16);
  mm_blk_t h0 = { y1, x, &g_probe_W, 0, out, in, gs, 0, n, 0, mid };
  g_mm_h1      = (mm_blk_t){ y1, x, &g_probe_W, 0, out, in, gs, 0, n, mid, out };
  __asm__ volatile("fence rw, rw" ::: "memory");
  uint64_t t0; __asm__ volatile("rdcycle %0" : "=r"(t0));
  ms_h1_fork(k1 ? mm_worker : mm_worker_scalar, &g_mm_h1, MS_DC_PROBE);
  if (k0) matmul_block(&h0); else matmul_block_scalar(&h0);
  uint64_t rt2; __asm__ volatile("rdcycle %0" : "=r"(rt2));
  const uint64_t h0_half = rt2 - t0;
  /* Must print BEFORE the join so a hart-1 wedge is localized, but UART costs ~4 M cycles a line, so
   * it must not sit inside the measured window: time the join separately. */
  printf("[moonshine] dc-ladder:   .. hart0 half done (%lu cycles), joining\n", (unsigned long)h0_half);
  uint64_t tw0; __asm__ volatile("rdcycle %0" : "=r"(tw0));
  if (!ms_h1_wait_bounded((uint64_t)MS_DC_SELFTEST_WATCHDOG_CYCLES)) {
    printf("[moonshine] dc-ladder: WEDGED %s n=%d out=%d in=%d\n", name, n, out, in);
    return 0;
  }
  uint64_t tw1; __asm__ volatile("rdcycle %0" : "=r"(tw1));
  const uint64_t dual = h0_half + (tw1 - tw0);

  int bad = -1, nbad = 0; float worst = 0.0f;
  for (int r = 0; r < n; r++)
    for (int o = 0; o < out; o++) {
      const int j = r * out + o;
      const float a = y1[j], e = yr[j];
      float d = a - e; if (d < 0) d = -d;
      float m = e < 0 ? -e : e; if (m < 1.0f) m = 1.0f;
      const float rel = d / m;
      if (rel > worst) { worst = rel; }
      if (rel > MS_DC_TOL) { nbad++; if (bad < 0) bad = j; }
    }
  if (bad >= 0) {
    printf("[moonshine] dc-ladder: MISMATCH %s n=%d out=%d in=%d bad=%d/%d first=%d (%s half)"
           " got=%d/1e3 want=%d/1e3 maxrel=%d/1e6\n",
           name, n, out, in, nbad, n * out, bad, (bad % out) < mid ? "hart0" : "hart1",
           (int)lrintf(y1[bad] * 1000.0f), (int)lrintf(yr[bad] * 1000.0f),
           (int)lrintf(worst * 1.0e6f));
    return 0;
  }
  const unsigned long spd = dual ? (unsigned long)(((rt1 - rt0) * 100ull) / dual) : 0ul;
  printf("[moonshine] dc-ladder: ok  %-9s n=%-3d out=%-5d in=%-4d  1core=%lu dual=%lu  speedup=%lu.%02lux  maxrel=%d/1e6\n",
         name, n, out, in, (unsigned long)(rt1 - rt0), (unsigned long)dual,
         spd / 100ul, spd % 100ul, (int)lrintf(worst * 1.0e6f));
  return 1;
}

static int ms_dc_selftest(void) {
  static const probe_sz_t small = { 1,  256, 256 };   /*  64 KB weights */
  static const probe_sz_t mid_  = { 1, 1024, 288 };   /* 288 KB — where RVV probes started failing */
  static const probe_sz_t big   = { 1, 4096, 288 };   /* 1.15 MB — classifier-shaped */
  static const probe_sz_t enc   = { 96, 288, 288 };   /*  81 KB — encoder-shaped, many-row */
  const int gs = 32;
  const int max_q = 4096 * 288, max_x = 96 * 288, max_y = 96 * 288;

  int8_t *q  = (int8_t *)xmalloc((size_t)max_q);
  float  *s  = (float *)xmalloc((size_t)(max_q / gs) * sizeof(float));
  float  *x  = (float *)xmalloc((size_t)max_x * sizeof(float));
  float  *yr = (float *)xmalloc((size_t)max_y * sizeof(float));
  float  *y1 = (float *)xmalloc((size_t)max_y * sizeof(float));
  uint32_t r = 12345u;
  for (int i = 0; i < max_q; i++) { r = r * 1664525u + 1013904223u; q[i] = (int8_t)((r >> 16) & 0xFF); }
  for (int i = 0; i < max_q / gs; i++) { r = r * 1664525u + 1013904223u; s[i] = (float)((r >> 20) & 0xFF) * (1.0f / 4096.0f); }
  for (int i = 0; i < max_x; i++) { r = r * 1664525u + 1013904223u; x[i] = (float)((int)((r >> 18) & 0x3FF) - 512) * (1.0f / 512.0f); }

  /* Rungs in strict order of INCREASING RVV EXPOSURE, stopping at the first failure — the rung that
   * fails is the answer. RVV arrives one hart at a time, which is what separates "hart 1 cannot run
   * vectors" from "the two harts cannot both run vectors at once":
   *    sc/sc  both scalar            — the mechanism alone (already proven on silicon)
   *    rv/sc  hart 0 vector only     — hart 1 scalar alongside a vectorizing hart 0
   *    sc/rv  hart 1 vector only     — THE key rung: hart 1's vector unit, hart 0 out of the way
   *    rv/rv  both vector            — contention between the two vector units
   *    tiled  both, tiled GEMM       — adds a hart-0-built transpose hart 1 must read */
  int ok = 1;
  static const struct { const char *name; int k0, k1; } combo[] = {
    { "sc/sc", 0, 0 },
#if MS_USE_RVV
    { "rv/sc", 1, 0 },
    { "sc/rv", 0, 1 },
    { "rv/rv", 1, 1 },
#endif
  };
  const probe_sz_t szs[3] = { small, mid_, big };
  for (unsigned c = 0; c < sizeof(combo) / sizeof(combo[0]) && ok; c++)
    for (unsigned i = 0; i < 3 && ok; i++)
      ok = ms_dc_rung(combo[c].name, combo[c].k0, combo[c].k1, 0, szs[i], gs, q, s, x, yr, y1);
#if MS_USE_RVV
  if (ok) ok = ms_dc_rung("tiled", 1, 1, 1, enc, gs, q, s, x, yr, y1);
#else
  (void)enc;
#endif
  if (ok) { free(q); free(s); free(x); free(yr); free(y1); }   /* leak after a wedge: hart 1 may still write */
  return ok;
}
#endif /* MS_DC_SELFTEST */

void ms_dualcore_init(void) {
  /* Release hart 1: .data/.bss are initialized, so the dispatcher globals are safe to read now. */
  const uint32_t hb0 = __atomic_load_n(&g_h1_hb, __ATOMIC_ACQUIRE);
  __atomic_store_n(&g_dc_boot, MS_DC_BOOT_TOKEN, __ATOMIC_SEQ_CST);
  uint64_t t0; __asm__ volatile("rdcycle %0" : "=r"(t0));
  while (__atomic_load_n(&g_h1_alive, __ATOMIC_ACQUIRE) == 0u) {
    uint64_t t; __asm__ volatile("rdcycle %0" : "=r"(t));
    if (t - t0 > (uint64_t)MS_DC_BOOT_TIMEOUT_CYCLES) {
      const uint32_t hb1 = __atomic_load_n(&g_h1_hb, __ATOMIC_ACQUIRE);
      printf("[moonshine] dual-core: hart 1 never reached the dispatcher -> SINGLE-CORE"
             " (heartbeat %u->%u: hart 1 is %s)\n", (unsigned)hb0, (unsigned)hb1,
             hb1 != hb0 ? "RUNNING but not seeing hart 0's boot token" : "NOT EXECUTING at all");
      return;
    }
    ms_relax();
  }
  g_dc_active = 1;                      /* ms_h1_wait() needs it set; it clears it on a wedge */
  ms_h1_fork(nop_worker, 0, MS_DC_PROBE);
  if (!ms_h1_wait()) {
    printf("[moonshine] dual-core: hart 1 alive but did not run the warmup job -> SINGLE-CORE\n");
    return;
  }
  printf("[moonshine] dual-core: hart 1 up (warmup ok)\n");
#if MS_DC_SELFTEST
  if (!ms_dc_selftest()) {
    g_dc_active = 0;
    printf("[moonshine] dual-core: self-test FAILED -> SINGLE-CORE\n");
    return;
  }
#endif
  printf("[moonshine] dual-core: ACTIVE  h1_pct=%d mm_min_work=%d conv=%d\n",
         MS_H1_PCT, MS_MM_MIN_WORK, MS_DUALCORE_CONV);
}
#else
/* Single-core builds still need a __main: crt0 sends every non-boot hart there, and if it returns,
 * crt0 tails into exit() — newlib atexit handlers + stdio flush running on hart 1, fighting hart 0
 * for the UART. thread-lib/hthread.c used to supply this; that file is no longer linked, so park
 * secondary harts here instead. */
void __main(void) {
  for (;;) __asm__ volatile("wfi");
}
void ms_dualcore_init(void) {}
#endif /* MS_DUALCORE */

#if MS_DMA_CLASSIFIER
#include "hal_dma.h"
extern float ms_matrow_q8(const int8_t *q, const float *x, const float *s, int in, int gs);
static uint16_t g_dma_tid = 1;
/* Start an async DMA of `bytes` (a multiple of 64) from DRAM `src` to SRAM `dst` on channel 0. */
static void dma_chunk_start(void *dst, const void *src, uint32_t bytes) {
  dma_transaction_t tx;
  tx.core = 0; tx.transaction_id = g_dma_tid; tx.transaction_priority = 1; tx.peripheral_id = 0;
  tx.addr_r = (uint64_t)(uintptr_t)src; tx.addr_w = (uint64_t)(uintptr_t)dst;
  tx.inc_r = 64; tx.inc_w = 64; tx.len = (uint16_t)(bytes / 64u); tx.logw = 6;
  tx.do_interrupt = false; tx.do_address_gate = false;
  set_DMA_C(0, tx, true);
  start_DMA(0, g_dma_tid, (void *)0);
}
static void dma_chunk_finish(void) {
  dma_wait_till_inactive(30);
  dma_reset();
  __asm__ volatile("fence rw, rw" ::: "memory");   /* order DMA completion before the core reads SRAM */
  g_dma_tid++;
}
/* Invalidate the cache line containing `p` (cbo.inval — discard, no writeback). Raw insn since the
 * toolchain march lacks zicbom. The SRAM buffers are only ever DMA-written, never core-written, so the
 * cached copies are clean; invalidating forces the core to refetch the freshly-DMA'd data. */
static inline void cbo_inval(const void *p) {
  __asm__ volatile(".insn i 0x0f, 2, x0, 0(%0)" :: "r"(p) : "memory");
}
static void sram_inval(const int8_t *buf, uint32_t bytes) {
  for (uint32_t i = 0; i < bytes; i += 64u) cbo_inval(buf + i);
  __asm__ volatile("fence rw, rw" ::: "memory");
}
/* Classifier via DMA weight double-buffering: stream embedding rows through two SRAM ping-pong buffers
 * while the core computes each row's dot from SRAM. Single-core, int8, lossless. */
static void classifier_dma(float *logits, const float *h, const msq8_t *W, int out, int in, int gs) {
  const int gpr = in / gs;
  const uint32_t R = MS_DMA_CHUNK_ROWS;
  const uint32_t CHUNK = R * (uint32_t)in;                 /* bytes/chunk (in=288 -> 288*R mult of 64) */
  int8_t *buf[2];
  buf[0] = (int8_t *)MS_DMA_SRAM_BASE;
  buf[1] = buf[0] + CHUNK;
  const int8_t *q = W->q;
  const int nchunks = (out + (int)R - 1) / (int)R;
  const uint32_t rows0 = (uint32_t)((out < (int)R) ? out : (int)R);
  dma_chunk_start(buf[0], q, rows0 * (uint32_t)in);        /* prime chunk 0 */
  dma_chunk_finish();
  for (int c = 0; c < nchunks; c++) {
    int8_t *cur = buf[c & 1], *nxt = buf[(c + 1) & 1];
    const int r0 = c * (int)R, rc = (out - r0 < (int)R) ? (out - r0) : (int)R;
    const int has_next = (c + 1 < nchunks);
    if (has_next) {                                        /* prefetch chunk c+1 while computing chunk c */
      const int rn0 = (c + 1) * (int)R;
      const uint32_t rn = (uint32_t)((out - rn0 < (int)R) ? (out - rn0) : (int)R);
      dma_chunk_start(nxt, q + (size_t)rn0 * in, rn * (uint32_t)in);
    }
    sram_inval(cur, (uint32_t)rc * (uint32_t)in);          /* discard stale cache; read the fresh chunk */
    for (int r = 0; r < rc; r++) {
      const int o = r0 + r;
      logits[o] = ms_matrow_q8(cur + (size_t)r * in, h, W->s + (size_t)o * gpr, in, gs);
    }
    if (has_next) dma_chunk_finish();
  }
}
#endif /* MS_DMA_CLASSIFIER */

/* matmul: n rows (n,in) x weight (out,in) -> (n,out). Many-row matmuls use the register-tiled GEMM
 * (K-major transpose built lazily on hart 0); large matmuls also split across 2 harts under
 * MS_DUALCORE. n==1 (decoder per-token, classifier) stays on the reduction kernel — unless a
 * bandwidth optimization (int4 repack / DMA double-buffer) claims the big single-row matmuls. */
static void matmul_q8(float *out_mat, const float *in_mat, const msq8_t *W, const float *bias,
                      int n, int out, int in, int gs) {
  PROF_T0();
#if MS_DMA_CLASSIFIER
  /* The classifier is the only n==1, huge-out matmul; stream its weights through SRAM (double-buffered)
   * instead of the DRAM reduction. bias is NULL for the tied classifier. */
  if (n == 1 && out >= 4096) {
    classifier_dma(out_mat, in_mat, W, out, in, gs);
    PROF_ADD(g_cyc_mm);
    return;
  }
#endif
#if MS_USE_RVV
  if (n >= MS_TILE_MR) wq8_build_tiled((msq8_t *)W, out, in, gs);   /* many-row -> tiled (hart 0, pre-fork) */
#endif
#if MS_INT4
  /* Big single-row reduction matmuls (classifier out=vocab, decoder fc1 out=2*ff_dec) are
   * bandwidth-bound: repack to int4 to halve the weight DRAM stream. Built once, hart 0, pre-fork. */
  if (n < MS_TILE_MR && out >= MS_INT4_MIN_OUT) wq8_build_int4((msq8_t *)W, out, in, gs);
#endif
#if MS_DUALCORE
  if (g_dc_active && (long)n * out >= MS_MM_MIN_WORK) {
    mm_blk_t h0 = { out_mat, in_mat, W, bias, out, in, gs, 0, n, 0, out };
    g_mm_h1 = h0;
    if (n >= 2) {                     /* split ROWS, on a tile boundary so each half keeps MR=4 */
      const int mid = ms_split(n, (n >= MS_TILE_MR) ? MS_TILE_MR : 1);
      h0.r1 = mid; g_mm_h1.r0 = mid;
    } else {                          /* split output COLUMNS (n == 1) */
      int align = 16;                 /* 64 B of float: keep the two harts off a shared line */
#if MS_INT4
      if (W->q4) align = 32;          /* int4 packs 32 outputs per block */
#endif
      const int mid = ms_split(out, align);
      h0.o1 = mid; g_mm_h1.o0 = mid;
    }
    __asm__ volatile("fence rw, rw" ::: "memory");
#if MS_DC_SCALAR
    /* Scalar rung of the ladder, applied to the real workload: same fork/join, same split, but no
     * RVV/tiled/int4 in either hart's block. Slow by design — it proves the dual-core MECHANISM in
     * situ before the vector kernels are layered back on. */
    ms_h1_fork(mm_worker_scalar, &g_mm_h1, MS_DC_MM);
    matmul_block_scalar(&h0);
    if (!ms_h1_wait()) matmul_block_scalar(&g_mm_h1);
#else
    ms_h1_fork(mm_worker, &g_mm_h1, MS_DC_MM);   /* hart 1: [mid,end) */
    matmul_block(&h0);                            /* hart 0: [0,mid), concurrently */
    if (!ms_h1_wait()) matmul_block(&g_mm_h1);    /* wedged: finish hart 1's block ourselves */
#endif
    PROF_ADD(g_cyc_mm);
    return;
  }
#endif
#if MS_USE_RVV
  if (W->qT) {   /* single-core tiled (many-row matmul below the dual-core threshold) */
    /* Block the output columns so one K-major weight slice (K*width bytes) stays resident in the
     * data cache across EVERY row tile. Register tiling alone reuses a weight vector across only
     * MR=4 rows, so the matrix is re-streamed ceil(n/4) times — 6 passes for a 23-position encoder,
     * ~36 MB of the ~170 MB an inference moves. Blocked, it streams once.
     * The loop lives HERE, not inside gemm_tiled_block: see the noinline note there. */
    mm_blk_t b = { out_mat, in_mat, W, bias, out, in, gs, 0, n, 0, out };
    int ob = (int)((MS_CACHE_BYTES * MS_CACHE_FRAC_NUM / MS_CACHE_FRAC_DEN) / (unsigned)in);
    ob &= ~31;                                  /* whole 64 B lines of qT */
    if (ob < 32) ob = 32;
    for (int o0 = 0; o0 < out; o0 += ob) {
      b.o0 = o0;
      b.o1 = (o0 + ob < out) ? (o0 + ob) : out;
      gemm_tiled_block(&b);
    }
    PROF_ADD(g_cyc_mm);
    return;
  }
#endif
#if MS_INT4
  if (W->q4) {   /* int4 built (big reduction matmul below the dual-core threshold) */
    for (int r = 0; r < n; r++)
      gemm_int4_row(out_mat + (size_t)r * out, in_mat + (size_t)r * in, W->q4, W->s4,
                    bias, out, in, MS_INT4_GS, 0, out);
    PROF_ADD(g_cyc_mm);
    return;
  }
#endif
  for (int r = 0; r < n; r++) {
    float *y = out_mat + (size_t)r * out;
    const float *x = in_mat + (size_t)r * in;
#if MS_USE_RVV
    matvec_q8_rvv(y, x, W, bias, out, in, gs);
#else
    matvec_q8_scalar(y, x, W, bias, out, in, gs);
#endif
  }
  PROF_ADD(g_cyc_mm);
}

/* dequantize one Q8_0 weight row (used for the token embedding lookup). */
static void deq_row(const msq8_t *W, int row, int in, int gs, float *out) {
  const int8_t *q = W->q + (size_t)row * in;
  const float *s = W->s + (size_t)row * (in / gs);
  for (int i = 0; i < in; i++) out[i] = (float)q[i] * s[i / gs];
}

/* torch-default (exact-erf) GELU and SiLU. */
static inline float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }
static inline float silu_f(float x)   { return x / (1.0f + expf(-x)); }

static void gelu_inplace(float *x, size_t n) {
  PROF_T0();
  for (size_t i = 0; i < n; i++) x[i] = gelu_erf(x[i]);
  PROF_ADD(g_cyc_gelu);
}

/* Bias-free LayerNorm over the last dim for each of n rows (eps 1e-5). b may be NULL. */
static void layernorm(const float *x, const float *w, const float *b, float *out, int n, int dim) {
  PROF_T0();
  for (int r = 0; r < n; r++) {
    const float *xr = x + (size_t)r * dim;
    float *orow = out + (size_t)r * dim;
    float mean = 0.0f; for (int i = 0; i < dim; i++) mean += xr[i]; mean /= dim;
    float var = 0.0f; for (int i = 0; i < dim; i++) { float d = xr[i] - mean; var += d * d; } var /= dim;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < dim; i++) orow[i] = (xr[i] - mean) * inv * w[i] + (b ? b[i] : 0.0f);
  }
  PROF_ADD(g_cyc_ln);
}

/* GroupNorm(num_groups=1, C) over a (C,L) tensor: mean/var over ALL C*L elements, per-channel affine. */
static void groupnorm1(float *x, int C, int L, const float *w, const float *b) {
  PROF_T0();
  const long N = (long)C * L;
  double mean = 0.0; for (long i = 0; i < N; i++) mean += x[i]; mean /= (double)N;
  double var = 0.0; for (long i = 0; i < N; i++) { double d = x[i] - mean; var += d * d; } var /= (double)N;
  float mf = (float)mean, inv = 1.0f / sqrtf((float)var + 1e-5f);
  for (int cc = 0; cc < C; cc++) {
    float wc = w[cc], bc = b[cc]; float *xc = x + (size_t)cc * L;
    for (int t = 0; t < L; t++) xc[t] = (xc[t] - mf) * inv * wc + bc;
  }
  PROF_ADD(g_cyc_ln);
}

/* Interleaved partial RoPE, in place, on one row of length hidden = n_head*head_dim, at absolute
 * position `pos`. Rotates each head's first rotary_dim dims as pairs (2i,2i+1); the rest pass through.
 * q_embed[2i] = q[2i]*cos - q[2i+1]*sin ; q_embed[2i+1] = q[2i+1]*cos + q[2i]*sin  (GPT-J style). */
static void rope_row(const ms_model_t *m, float *x, int pos) {
  const int nh = m->n_head, hd = m->head_dim, rh = m->rotary_dim / 2;
  for (int i = 0; i < rh; i++) {
    float ang = (float)pos * m->inv_freq[i];
    float c = cosf(ang), s = sinf(ang);
    for (int h = 0; h < nh; h++) {
      float *xh = x + (size_t)h * hd + 2 * i;
      float a = xh[0], b = xh[1];
      xh[0] = a * c - b * s;
      xh[1] = b * c + a * s;
    }
  }
}
/* Apply RoPE to every row of an (nq, hidden) matrix, row t at position t. */
static void rope_rows(const ms_model_t *m, float *x, int nq) {
  PROF_T0();
  for (int t = 0; t < nq; t++) rope_row(m, x + (size_t)t * m->hidden, t);
  PROF_ADD(g_cyc_rope);
}

/* Multi-head attention (scalar). q(nq,S),k(nk,S),v(nk,S) already projected (and RoPE'd for self).
 * causal: query i attends keys 0..i. Writes out(nq,S). Scale 1/sqrt(head_dim). */
static void mha_run(const float *q, const float *k, const float *v, int nq, int nk,
                    int S, int n_head, int causal, float *out, float *sc) {
  const int hd = S / n_head;
  const float rscale = 1.0f / sqrtf((float)hd);
  for (int h = 0; h < n_head; h++) {
    const int off = h * hd;
    for (int i = 0; i < nq; i++) {
      const float *qi = q + (size_t)i * S + off;
      const int lim = causal ? (i + 1) : nk;
      float mx = -1e30f;
      for (int j = 0; j < lim; j++) {
        const float *kj = k + (size_t)j * S + off;
        float dot = 0.0f; for (int d = 0; d < hd; d++) dot += qi[d] * kj[d];
        dot *= rscale; sc[j] = dot; if (dot > mx) mx = dot;
      }
      float sum = 0.0f;
      for (int j = 0; j < lim; j++) { float e = expf(sc[j] - mx); sc[j] = e; sum += e; }
      float invs = 1.0f / sum;
      float *oi = out + (size_t)i * S + off;
      for (int d = 0; d < hd; d++) oi[d] = 0.0f;
      for (int j = 0; j < lim; j++) {
        float wj = sc[j] * invs;
        const float *vj = v + (size_t)j * S + off;
        for (int d = 0; d < hd; d++) oi[d] += wj * vj[d];
      }
    }
  }
}
static void mha_core(const float *q, const float *k, const float *v, int nq, int nk,
                     int S, int n_head, int causal, float *out) {
  PROF_T0();
  if (nk <= MS_MHA_MAX_KEYS) {
    float sc[MS_MHA_MAX_KEYS];
    mha_run(q, k, v, nq, nk, S, n_head, causal, out, sc);
  } else {
    float *big = (float *)xmalloc((size_t)nk * sizeof(float));
    mha_run(q, k, v, nq, nk, S, n_head, causal, out, big);
    free(big);
  }
  PROF_ADD(g_cyc_attn);
}

static void add_inplace(float *a, const float *b, size_t n) { for (size_t i = 0; i < n; i++) a[i] += b[i]; }

static void stat_of(const float *x, size_t n, ms_stat_t *st) {
  double sum = 0.0, amax = 0.0;
  for (size_t i = 0; i < n; i++) { double v = x[i]; sum += v; double a = v < 0 ? -v : v; if (a > amax) amax = a; }
  st->sum = sum; st->absmax = amax; st->mean = sum / (double)n;
}

/* Valid (no-pad) conv1d: in(Cin,L), weight(Cout,Cin,K), bias(Cout) -> out(Cout,Lout),
 * Lout = (L-K)/stride + 1. out[oc,t] = bias[oc] + sum_ic sum_kk w[oc,ic,kk]*in[ic, t*stride+kk]. */
static void conv1d_valid_scalar(int oc0, int oc1, const float *in, int Cin, int L,
                                const float *w, const float *b, int K, int stride,
                                float *out, int Lout) {
  for (int oc = oc0; oc < oc1; oc++) {
    const float *wo = w + (size_t)oc * Cin * K;
    float bias = b ? b[oc] : 0.0f;
    float *orow = out + (size_t)oc * Lout;
    for (int t = 0; t < Lout; t++) {
      float acc = bias; const int base = t * stride;
      for (int ic = 0; ic < Cin; ic++) {
        const float *wic = wo + ic * K;
        const float *inic = in + (size_t)ic * L + base;
        for (int kk = 0; kk < K; kk++) acc += wic[kk] * inic[kk];
      }
      orow[t] = acc;
    }
  }
}

#if MS_USE_RVV
/* RVV VALID conv1d — register-tiled over OUTPUT CHANNELS. Vectorize over output positions t (idiom
 * from dsp25-bmarks/simple-conv-bmark); the stride (64/3/2) forces strided input loads (vlse32), which
 * are the bottleneck on this core (~1 elem/cycle). The win is REUSE: the input working set is small
 * (in[Cin,L], stays cached), and we tile 4 output channels together so each strided load feeds FOUR
 * vfmacc.vf into four independent accumulators — the slow load is amortized 4x and the 4 FMA chains
 * pipeline. (im2col was tried and was WORSE: its col matrix is bigger than the input and got
 * re-streamed from DRAM once per output channel, blowing the cache.) VALID conv has no boundary
 * prolog/epilog; the only tail is the VL remainder (vsetvl). Registers: a0..a3 (m4=16) + vin (4) =
 * 20/32, no spill. Cout for moonshine (288/576/288) is a multiple of 4; a 1-channel tail covers the
 * general case. */
static void conv1d_valid_rvv(int oc0, int oc1, const float *in, int Cin, int L,
                             const float *w, const float *b, int K, int stride,
                             float *out, int Lout) {
  const ptrdiff_t sb = (ptrdiff_t)stride * (ptrdiff_t)sizeof(float);
  const int P = Cin * K;
  int oc = oc0;
  for (; oc + 4 <= oc1; oc += 4) {                      /* 4 output channels per strided load */
    const float *w0 = w + (size_t)(oc + 0) * P, *w1 = w + (size_t)(oc + 1) * P;
    const float *w2 = w + (size_t)(oc + 2) * P, *w3 = w + (size_t)(oc + 3) * P;
    const float b0 = b ? b[oc] : 0.0f, b1 = b ? b[oc + 1] : 0.0f;
    const float b2 = b ? b[oc + 2] : 0.0f, b3 = b ? b[oc + 3] : 0.0f;
    float *o0 = out + (size_t)(oc + 0) * Lout, *o1 = out + (size_t)(oc + 1) * Lout;
    float *o2 = out + (size_t)(oc + 2) * Lout, *o3 = out + (size_t)(oc + 3) * Lout;
    int t = 0;
    while (t < Lout) {
      size_t vl = __riscv_vsetvl_e32m4((size_t)(Lout - t));
      vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(b0, vl), a1 = __riscv_vfmv_v_f_f32m4(b1, vl);
      vfloat32m4_t a2 = __riscv_vfmv_v_f_f32m4(b2, vl), a3 = __riscv_vfmv_v_f_f32m4(b3, vl);
      const size_t tS = (size_t)t * stride;
      int p = 0;
      for (int ic = 0; ic < Cin; ic++) {
        const float *ib = in + (size_t)ic * L + tS;
        for (int kk = 0; kk < K; kk++, p++) {
          vfloat32m4_t vin = (stride == 1) ? __riscv_vle32_v_f32m4(ib + kk, vl)
                                           : __riscv_vlse32_v_f32m4(ib + kk, sb, vl);
          a0 = __riscv_vfmacc_vf_f32m4(a0, w0[p], vin, vl);
          a1 = __riscv_vfmacc_vf_f32m4(a1, w1[p], vin, vl);
          a2 = __riscv_vfmacc_vf_f32m4(a2, w2[p], vin, vl);
          a3 = __riscv_vfmacc_vf_f32m4(a3, w3[p], vin, vl);
        }
      }
      __riscv_vse32_v_f32m4(o0 + t, a0, vl); __riscv_vse32_v_f32m4(o1 + t, a1, vl);
      __riscv_vse32_v_f32m4(o2 + t, a2, vl); __riscv_vse32_v_f32m4(o3 + t, a3, vl);
      t += (int)vl;
    }
  }
  for (; oc < oc1; oc++) {                              /* 1-channel tail (Cout % 4 != 0) */
    const float *wr = w + (size_t)oc * P;
    const float bias = b ? b[oc] : 0.0f;
    float *orow = out + (size_t)oc * Lout;
    int t = 0;
    while (t < Lout) {
      size_t vl = __riscv_vsetvl_e32m4((size_t)(Lout - t));
      vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(bias, vl);
      const size_t tS = (size_t)t * stride;
      int p = 0;
      for (int ic = 0; ic < Cin; ic++) {
        const float *ib = in + (size_t)ic * L + tS;
        for (int kk = 0; kk < K; kk++, p++) {
          vfloat32m4_t vin = (stride == 1) ? __riscv_vle32_v_f32m4(ib + kk, vl)
                                           : __riscv_vlse32_v_f32m4(ib + kk, sb, vl);
          a0 = __riscv_vfmacc_vf_f32m4(a0, wr[p], vin, vl);
        }
      }
      __riscv_vse32_v_f32m4(orow + t, a0, vl);
      t += (int)vl;
    }
  }
}
#endif /* MS_USE_RVV */

#if MS_DUALCORE && MS_DUALCORE_CONV
/* conv output channels are independent -> split [0,Cout) across the two harts (the boundary is kept
 * a multiple of 4 to preserve the MR=4 tiling on each half). Same fork/watchdog path as matmul. */
typedef struct { const float *in; int Cin, L; const float *w, *b; int K, stride; float *out; int Lout; int oc0, oc1; } conv_blk_t;
static conv_blk_t g_conv_h1 __attribute__((aligned(64)));   /* own cache line (avoid false sharing) */
static void conv_worker(void *a) {
  const conv_blk_t *c = (const conv_blk_t *)a;
  conv1d_valid_rvv(c->oc0, c->oc1, c->in, c->Cin, c->L, c->w, c->b, c->K, c->stride, c->out, c->Lout);
}
#endif

static void conv1d_valid(const float *in, int Cin, int L, const float *w, const float *b,
                         int Cout, int K, int stride, float *out, int Lout) {
  PROF_T0();
#if MS_DUALCORE && MS_DUALCORE_CONV
  if (g_dc_active && Cout >= MS_CONV_MIN_COUT) {
    const int mid = ms_split(Cout, 4);
    g_conv_h1 = (conv_blk_t){ in, Cin, L, w, b, K, stride, out, Lout, mid, Cout };
    __asm__ volatile("fence rw, rw" ::: "memory");
    ms_h1_fork(conv_worker, &g_conv_h1, MS_DC_CONV);                   /* hart 1: [mid,Cout) */
    conv1d_valid_rvv(0, mid, in, Cin, L, w, b, K, stride, out, Lout);  /* hart 0: [0,mid) */
    if (!ms_h1_wait())                                                 /* wedged: do hart 1's share */
      conv1d_valid_rvv(mid, Cout, in, Cin, L, w, b, K, stride, out, Lout);
  } else {
    conv1d_valid_rvv(0, Cout, in, Cin, L, w, b, K, stride, out, Lout);
  }
#elif MS_USE_RVV
  conv1d_valid_rvv(0, Cout, in, Cin, L, w, b, K, stride, out, Lout);
#else
  conv1d_valid_scalar(0, Cout, in, Cin, L, w, b, K, stride, out, Lout);
#endif
  PROF_ADD(g_cyc_conv);
}

/* Fast unit test: RVV conv must equal scalar conv bit-for-(nearly)-bit on small synthetic inputs,
 * exercising every code path (stride 64/3/2 strided loads + stride-1 unit loads; even-Cin pair loop
 * + odd-Cin tail; large K=127; VL remainder tail). Runs in << 1 s on Spike (tiny sizes) and exits,
 * so it validates the kernel without simulating the whole model. Returns 0 = all PASS. */
int ms_conv_selftest(void) {
  static const struct { int Cin, K, stride, Cout, L; } cfg[] = {
    { 1, 127, 64, 8, 4096 },   /* conv1 geometry: Cin=1 -> odd tail, stride 64, big kernel */
    { 16,  7,  3, 8,  300 },   /* conv2 geometry: even Cin -> pair loop, stride 3 */
    { 17,  7,  3, 4,  300 },   /* odd Cin: pair loop + odd tail */
    { 16,  3,  2, 8,  200 },   /* conv3 geometry: stride 2 */
    {  4,  5,  1, 4,  103 },   /* stride==1 unit-load path, VL-tail (odd Lout) */
  };
  const int NC = (int)(sizeof(cfg) / sizeof(cfg[0]));
  int all_ok = 1;
  for (int c = 0; c < NC; c++) {
    const int Cin = cfg[c].Cin, K = cfg[c].K, S = cfg[c].stride, Cout = cfg[c].Cout, L = cfg[c].L;
    const int Lout = (L - K) / S + 1;
    float *in = (float *)xmalloc((size_t)Cin * L * sizeof(float));
    float *w  = (float *)xmalloc((size_t)Cout * Cin * K * sizeof(float));
    float *b  = (float *)xmalloc((size_t)Cout * sizeof(float));
    float *o0 = (float *)xmalloc((size_t)Cout * Lout * sizeof(float));   /* scalar */
    float *o1 = (float *)xmalloc((size_t)Cout * Lout * sizeof(float));   /* rvv */
    for (int i = 0; i < Cin * L; i++)  in[i] = (float)((i * 37 + 11) % 97 - 48) * 0.031f;
    for (int i = 0; i < Cout*Cin*K; i++) w[i] = (float)((i * 13 + 5) % 61 - 30) * 0.017f;
    for (int i = 0; i < Cout; i++) b[i] = (float)(i % 7 - 3) * 0.25f;
    conv1d_valid_scalar(0, Cout, in, Cin, L, w, b, K, S, o0, Lout);
#if MS_USE_RVV
    conv1d_valid_rvv(0, Cout, in, Cin, L, w, b, K, S, o1, Lout);
#else
    conv1d_valid_scalar(0, Cout, in, Cin, L, w, b, K, S, o1, Lout);
#endif
    double maxd = 0.0; int worst = -1;
    for (int i = 0; i < Cout * Lout; i++) {
      double d = (double)o1[i] - (double)o0[i]; if (d < 0) d = -d;
      if (d > maxd) { maxd = d; worst = i; }
    }
    int ok = (maxd < 1e-3);
    if (!ok) all_ok = 0;
    printf("[moonshine] CONV-CHECK Cin=%d K=%d s=%d Cout=%d L=%d Lout=%d  max_abs_diff=%.6g @%d %s\n",
           Cin, K, S, Cout, L, Lout, maxd, worst, ok ? "PASS" : "FAIL");
    free(in); free(w); free(b); free(o0); free(o1);
  }
  printf("[moonshine] CONV-CHECK RESULT %s\n", all_ok ? "PASS" : "FAIL");
  fflush(stdout);
  return all_ok ? 0 : 1;
}

/* --------------------------------------------------------------------------------- encoder ------ */
int ms_encode(const ms_model_t *m, const float *audio, int n_samples,
              float **enc_out_p, int *n_pos_p, ms_stat_t *stats) {
  const int H = m->hidden, C2 = 2 * H, FE = m->ff_enc, nh = m->n_head, gs = m->gs;
  const int L1 = (n_samples - MS_CONV1_K) / MS_CONV1_S + 1;
  const int L2 = (L1 - MS_CONV2_K) / MS_CONV2_S + 1;
  const int L3 = (L2 - MS_CONV3_K) / MS_CONV3_S + 1;
  if (L3 < 1) { *enc_out_p = NULL; *n_pos_p = 0; return -1; }
  int si = 0;

  /* raw-audio conv stem: tanh(conv1) -> groupnorm -> gelu(conv2) -> gelu(conv3) */
  float *c1 = (float *)xmalloc((size_t)H * L1 * sizeof(float));
  conv1d_valid(audio, 1, n_samples, m->conv1_w, NULL, H, MS_CONV1_K, MS_CONV1_S, c1, L1);
  for (size_t i = 0; i < (size_t)H * L1; i++) c1[i] = tanhf(c1[i]);
  groupnorm1(c1, H, L1, m->gn_w, m->gn_b);

  float *c2 = (float *)xmalloc((size_t)C2 * L2 * sizeof(float));
  conv1d_valid(c1, H, L1, m->conv2_w, m->conv2_b, C2, MS_CONV2_K, MS_CONV2_S, c2, L2);
  gelu_inplace(c2, (size_t)C2 * L2);
  free(c1);

  float *c3 = (float *)xmalloc((size_t)H * L3 * sizeof(float));
  conv1d_valid(c2, C2, L2, m->conv3_w, m->conv3_b, H, MS_CONV3_K, MS_CONV3_S, c3, L3);
  gelu_inplace(c3, (size_t)H * L3);
  free(c2);

  /* transpose (H,L3) -> x (T,H); NO positional embedding (RoPE handles position) */
  const int T = L3, S = H;
  float *x = (float *)xmalloc((size_t)T * S * sizeof(float));
  for (int t = 0; t < T; t++)
    for (int d = 0; d < S; d++)
      x[(size_t)t * S + d] = c3[(size_t)d * L3 + t];
  free(c3);
  MS_DBG("[ms] enc: conv-stem done T=%d\n", T);
  if (stats) stat_of(x, (size_t)T * S, &stats[si]); si++;

  float *h  = (float *)xmalloc((size_t)T * S * sizeof(float));
  float *qb = (float *)xmalloc((size_t)T * S * sizeof(float));
  float *kb = (float *)xmalloc((size_t)T * S * sizeof(float));
  float *vb = (float *)xmalloc((size_t)T * S * sizeof(float));
  float *ab = (float *)xmalloc((size_t)T * S * sizeof(float));
  float *pb = (float *)xmalloc((size_t)T * S * sizeof(float));
  float *hid = (float *)xmalloc((size_t)T * FE * sizeof(float));

  for (int l = 0; l < m->n_enc; l++) {
    const ms_enc_layer_t *Ly = &m->enc[l];
    /* self-attention (RoPE, bidirectional) */
    layernorm(x, Ly->in_ln_w, NULL, h, T, S);
    matmul_q8(qb, h, &Ly->q, NULL, T, S, S, gs);
    matmul_q8(kb, h, &Ly->k, NULL, T, S, S, gs);
    matmul_q8(vb, h, &Ly->v, NULL, T, S, S, gs);
    rope_rows(m, qb, T);
    rope_rows(m, kb, T);
    mha_core(qb, kb, vb, T, T, S, nh, 0, ab);
    matmul_q8(pb, ab, &Ly->o, NULL, T, S, S, gs);
    add_inplace(x, pb, (size_t)T * S);
    /* MLP (plain GELU) */
    layernorm(x, Ly->post_ln_w, NULL, h, T, S);
    matmul_q8(hid, h, &Ly->fc1, Ly->fc1_b, T, FE, S, gs);
    gelu_inplace(hid, (size_t)T * FE);
    matmul_q8(pb, hid, &Ly->fc2, Ly->fc2_b, T, S, FE, gs);
    add_inplace(x, pb, (size_t)T * S);
    MS_DBG("[ms] enc layer %d done\n", l);
    if (stats) stat_of(x, (size_t)T * S, &stats[si]); si++;
  }

  layernorm(x, m->enc_ln_w, NULL, h, T, S);
  memcpy(x, h, (size_t)T * S * sizeof(float));
  if (stats) stat_of(x, (size_t)T * S, &stats[si]); si++;
  MS_DBG("[ms] enc: final norm+copy done, freeing buffers\n");

  free(h); free(qb); free(kb); free(vb); free(ab); free(pb); free(hid);
  MS_DBG("[ms] enc: buffers freed; enc_out_p=%p n_pos_p=%p x=%p T=%d\n",
         (void *)enc_out_p, (void *)n_pos_p, (void *)x, T);
  *enc_out_p = x;
  MS_DBG("[ms] enc: wrote enc_out_p\n");
  *n_pos_p = T;
  MS_DBG("[ms] enc: wrote n_pos_p, executing return 0\n");
  return 0;
}

/* --------------------------------------------------------------------------------- decoder ------ */
int ms_decode_greedy(const ms_model_t *m, const float *enc_out, int n_enc,
                     int *out_tokens, int max_new, int guards, int *n_out) {
  const int S = m->hidden, FD = m->ff_dec, nh = m->n_head, DL = m->n_dec, gs = m->gs;
  const int eot = m->eos, maxL = max_new + 1;
  MS_DBG("[ms] dec: entered greedy (S=%d FD=%d DL=%d maxL=%d)\n", S, FD, DL, maxL);

  float **ck = (float **)xmalloc(DL * sizeof(float *));   /* cross K,V per layer (from encoder) */
  float **cv = (float **)xmalloc(DL * sizeof(float *));
  float **kc = (float **)xmalloc(DL * sizeof(float *));   /* self-attn K,V caches per layer */
  float **vc = (float **)xmalloc(DL * sizeof(float *));
  MS_DBG("[ms] dec: cross-KV arrays alloced, entering loop (DL=%d n_enc=%d)\n", DL, n_enc);
  for (int l = 0; l < DL; l++) {
    ck[l] = (float *)xmalloc((size_t)n_enc * S * sizeof(float));
    cv[l] = (float *)xmalloc((size_t)n_enc * S * sizeof(float));
    MS_DBG("[ms] dec: crossKV layer %d ck-matmul...\n", l);
    matmul_q8(ck[l], enc_out, &m->dec[l].ck, NULL, n_enc, S, S, gs);
    MS_DBG("[ms] dec: crossKV layer %d cv-matmul...\n", l);
    matmul_q8(cv[l], enc_out, &m->dec[l].cv, NULL, n_enc, S, S, gs);
    kc[l] = (float *)xmalloc((size_t)maxL * S * sizeof(float));
    vc[l] = (float *)xmalloc((size_t)maxL * S * sizeof(float));
  }
  MS_DBG("[ms] dec: cross-KV setup done\n");

  int *tokens = (int *)xmalloc(maxL * sizeof(int));
  tokens[0] = m->bos;

  float *x   = (float *)xmalloc((size_t)S * sizeof(float));
  float *h   = (float *)xmalloc((size_t)S * sizeof(float));
  float *q   = (float *)xmalloc((size_t)S * sizeof(float));
  float *ao  = (float *)xmalloc((size_t)S * sizeof(float));
  float *pb  = (float *)xmalloc((size_t)S * sizeof(float));
  float *ffu = (float *)xmalloc((size_t)2 * FD * sizeof(float));   /* SwiGLU up|gate */
  float *ffa = (float *)xmalloc((size_t)FD * sizeof(float));
  float *logits = (float *)xmalloc((size_t)m->vocab * sizeof(float));
  int produced = 0;

  for (int pos = 0; pos < maxL; pos++) {
    MS_DBG("[ms] dec pos %d (self+cross+mlp x%d layers)\n", pos, DL);
    deq_row(&m->embed, tokens[pos], S, gs, x);           /* token embedding; NO positional add */

    for (int l = 0; l < DL; l++) {
      const ms_dec_layer_t *Ly = &m->dec[l];
      /* causal self-attention with RoPE + KV cache */
      layernorm(x, Ly->in_ln_w, NULL, h, 1, S);
      matmul_q8(q, h, &Ly->q, NULL, 1, S, S, gs);
      matmul_q8(kc[l] + (size_t)pos * S, h, &Ly->k, NULL, 1, S, S, gs);
      matmul_q8(vc[l] + (size_t)pos * S, h, &Ly->v, NULL, 1, S, S, gs);
      rope_row(m, q, pos);
      rope_row(m, kc[l] + (size_t)pos * S, pos);          /* rotate the new key at its position */
      mha_core(q, kc[l], vc[l], 1, pos + 1, S, nh, 0, ao);
      matmul_q8(pb, ao, &Ly->o, NULL, 1, S, S, gs);
      for (int d = 0; d < S; d++) x[d] += pb[d];
      /* cross-attention into encoder output (no RoPE) */
      layernorm(x, Ly->post_ln_w, NULL, h, 1, S);
      matmul_q8(q, h, &Ly->cq, NULL, 1, S, S, gs);
      mha_core(q, ck[l], cv[l], 1, n_enc, S, nh, 0, ao);
      matmul_q8(pb, ao, &Ly->co, NULL, 1, S, S, gs);
      for (int d = 0; d < S; d++) x[d] += pb[d];
      /* SwiGLU MLP: fc1 -> (up|gate), silu(gate)*up, fc2 */
      layernorm(x, Ly->final_ln_w, NULL, h, 1, S);
      matmul_q8(ffu, h, &Ly->fc1, Ly->fc1_b, 1, 2 * FD, S, gs);
      for (int j = 0; j < FD; j++) ffa[j] = silu_f(ffu[FD + j]) * ffu[j];
      matmul_q8(pb, ffa, &Ly->fc2, Ly->fc2_b, 1, S, FD, gs);
      for (int d = 0; d < S; d++) x[d] += pb[d];
    }

    /* tied classifier -> argmax. Vocab pruning (MS_VOCAB_TOPK): score only the first TOPK ids — the
     * SentencePiece common subset (eos=2 is always inside it). Cuts the classifier weight stream
     * ~vocab/TOPK x. Rounded to a multiple of 32 so the int4 (32-block) / DMA paths stay aligned. */
    int cls_out = m->vocab;
#if MS_VOCAB_TOPK
    if (MS_VOCAB_TOPK > 0 && MS_VOCAB_TOPK < m->vocab) cls_out = (MS_VOCAB_TOPK / 32) * 32;
#endif
    layernorm(x, m->dec_ln_w, NULL, h, 1, S);
    matmul_q8(logits, h, &m->embed, NULL, 1, cls_out, S, gs);

    if (guards && produced >= 2 && out_tokens[produced - 1] == out_tokens[produced - 2]
        && out_tokens[produced - 1] < cls_out)
      logits[out_tokens[produced - 1]] = -1e30f;          /* ban a 3rd-consecutive identical token */
    int best = 0; float bv = logits[0];
    for (int v = 1; v < cls_out; v++) if (logits[v] > bv) { bv = logits[v]; best = v; }
    out_tokens[produced++] = best;
    if (best == eot || produced >= max_new) break;
    if (pos + 1 < maxL) tokens[pos + 1] = best;

    if (guards) {
      /* repetition cycle guard: if the last C tokens duplicate the preceding C, drop + stop. */
      int looped = 0;
      for (int C = 2; C <= 8 && 2 * C <= produced; C++) {
        int match = 1;
        for (int j = 0; j < C; j++)
          if (out_tokens[produced - 1 - j] != out_tokens[produced - 1 - C - j]) { match = 0; break; }
        if (match) { produced -= C; looped = 1; break; }
      }
      if (looped) break;
    }
  }

  *n_out = produced;
  for (int l = 0; l < DL; l++) { free(ck[l]); free(cv[l]); free(kc[l]); free(vc[l]); }
  free(ck); free(cv); free(kc); free(vc); free(tokens);
  free(x); free(h); free(q); free(ao); free(pb); free(ffu); free(ffa); free(logits);
  return 0;
}

int ms_transcribe(const ms_model_t *m, const float *audio, int n_samples,
                  int *out_tokens, int max_new, int guards, int *n_out) {
  float *enc_out; int n_pos;
  if (ms_encode(m, audio, n_samples, &enc_out, &n_pos, NULL) != 0) { *n_out = 0; return 0; }
  MS_DBG("[ms] transcribe: encode returned n_pos=%d enc_out=%p, calling decode\n", n_pos, (void *)enc_out);
  ms_decode_greedy(m, enc_out, n_pos, out_tokens, max_new, guards, n_out);
  free(enc_out);
  return n_pos;
}

/* --------------------------------------------------------------------------------- text/decode -- */
/* SentencePiece detok: concatenate each token's baked output bytes (specials -> empty), then strip
 * ONE leading space (matches the moonshine tokenizer's Replace(▁,' ')+Strip decoder). */
void ms_print_tokens_text(const char *tag, const int *toks, int n) {
  /* gather into a small buffer so we can strip a single leading space */
  printf("%s\"", tag);
  int emitted = 0, lead_stripped = 0;
  for (int i = 0; i < n; i++) {
    int t = toks[i];
    if (t < 0 || t >= MV_N_VOCAB) continue;
    for (int b = mv_offset[t]; b < mv_offset[t + 1]; b++) {
      unsigned char ch = mv_bytes[b];
      if (!lead_stripped && !emitted && ch == ' ') { lead_stripped = 1; continue; }
      putchar(ch); emitted = 1;
    }
  }
  printf("\"\n");
}

/* ---- hardware roofline probe (MS_ROOFLINE) ------------------------------------------------------
 * Every software lever tried so far — RVV vs scalar, tiled vs untiled, int4 vs int8, -O1 vs -O3 —
 * lands the matmul between 20 and 60 cycles per MAC. When vectorization, tiling and instruction count
 * all fail to move a number, the kernel is not what is setting it. This measures the two rooflines
 * separately using the REAL kernel, so the next decision is made against the hardware limit instead
 * of another guess:
 *   COMPUTE  one weight row reused REP times, so it stays in L1: cycles/MAC with no memory pressure.
 *   MEMORY   a multi-MB weight matrix streamed once, so every line is a cold DRAM fetch.
 * If COMPUTE is fast and MEMORY is slow, the wall is bytes -> the only wins left reduce weight
 * traffic. If COMPUTE is also ~30 cycles/MAC, the vector unit itself is the wall and no amount of
 * kernel work will help; the answer becomes a smaller model or shorter audio. */
#if MS_ROOFLINE
#if !MS_USE_RVV
#error "MS_ROOFLINE measures the RVV kernel; build with MOONSHINE_USE_RVV=ON"
#endif
extern float ms_matrow_q8(const int8_t *q, const float *x, const float *s, int in, int gs);
static inline uint64_t ms_rdc(void) { uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }

void ms_roofline_report(void) {
  const int gs = 32, in = 288, gpr = in / gs;
  float *x = (float *)xmalloc((size_t)in * sizeof(float));
  for (int i = 0; i < in; i++) x[i] = 0.5f;
  volatile float sink = 0.0f;

  /* COMPUTE: one row, reused — after the first call it is L1-resident. */
  int8_t *q1 = (int8_t *)xmalloc((size_t)in);
  float  *s1 = (float *)xmalloc((size_t)gpr * sizeof(float));
  for (int i = 0; i < in; i++) q1[i] = (int8_t)(i * 7);
  for (int i = 0; i < gpr; i++) s1[i] = 0.01f;
  const int REP = 100000;
  sink += ms_matrow_q8(q1, x, s1, in, gs);                 /* warm */
  uint64_t t0 = ms_rdc();
  for (int r = 0; r < REP; r++) sink += ms_matrow_q8(q1, x, s1, in, gs);
  uint64_t t1 = ms_rdc();
  const double cmacs = (double)REP * (double)in;
  printf("[moonshine] ROOFLINE compute (L1-resident row): %lu cycles / %d MACs = %d/1000 cycles per MAC\n",
         (unsigned long)(t1 - t0), (int)cmacs, (int)(1000.0 * (double)(t1 - t0) / cmacs));

  /* MEMORY: stream a matrix far larger than any cache, each row touched once. */
  const int OUT = 16384;                                    /* 16384 x 288 = 4.5 MB of weights */
  int8_t *qb = (int8_t *)xmalloc((size_t)OUT * in);
  float  *sb = (float *)xmalloc((size_t)OUT * gpr * sizeof(float));
  memset(qb, 3, (size_t)OUT * in);
  for (size_t i = 0; i < (size_t)OUT * gpr; i++) sb[i] = 0.01f;
  t0 = ms_rdc();
  for (int o = 0; o < OUT; o++) sink += ms_matrow_q8(qb + (size_t)o * in, x, sb + (size_t)o * gpr, in, gs);
  t1 = ms_rdc();
  const double bytes = (double)OUT * (double)in;
  printf("[moonshine] ROOFLINE memory (%d MB cold stream): %lu cycles / %d MACs = %d/1000 cycles per MAC"
         "  (%d/1000 bytes per cycle)\n",
         (int)(bytes / 1048576.0), (unsigned long)(t1 - t0), (int)bytes,
         (int)(1000.0 * (double)(t1 - t0) / bytes), (int)(1000.0 * bytes / (double)(t1 - t0)));
  free(x); free(q1); free(s1); free(qb); free(sb);
}
#else
void ms_roofline_report(void) {}
#endif

void ms_profile_report(void) {
#if MS_PROFILE
  printf("[moonshine] PROFILE cycles: matmul=%lu attn=%lu conv=%lu gelu=%lu norm=%lu rope=%lu\n",
         (unsigned long)g_cyc_mm, (unsigned long)g_cyc_attn, (unsigned long)g_cyc_conv,
         (unsigned long)g_cyc_gelu, (unsigned long)g_cyc_ln, (unsigned long)g_cyc_rope);
#endif
}
void ms_profile_reset(void) {
#if MS_PROFILE
  g_cyc_conv = g_cyc_mm = g_cyc_attn = g_cyc_gelu = g_cyc_ln = g_cyc_rope = 0;
#endif
}

/* --------------------------------------------------------------------------------- validate ----- */
int ms_run_validate(const void *model_blob, const float *audio, int n_samples) {
  ms_model_t m;
  if (ms_model_load(model_blob, &m) != 0) { printf("[moonshine] bad model magic\n"); return 1; }
  printf("[moonshine] model: hidden=%d enc=%d dec=%d heads=%d hd=%d rot=%d ff=%d vocab=%d gs=%d matvec=%s\n",
         m.hidden, m.n_enc, m.n_dec, m.n_head, m.head_dim, m.rotary_dim, m.ff_enc, m.vocab, m.gs,
         MS_USE_RVV ? "RVV" : "scalar");

  ms_stat_t stats[MS_MAX_LAYERS + 2];
  float *enc_out; int n_pos;
  ms_profile_reset();
  ms_encode(&m, audio, n_samples, &enc_out, &n_pos, stats);

  int enc_pass = 1;
  printf("[moonshine] ENC stage sums (name: C_sum vs ref_sum  rel):\n");
  for (int i = 0; i < MS_REF_NUM_STAGES && i < MS_MAX_LAYERS + 2; i++) {
    double refs = ms_ref_stages[i].sum;
    double rel = (refs != 0.0) ? fabs(stats[i].sum - refs) / fabs(refs) : fabs(stats[i].sum);
    int ok = rel < 0.03;
    if (!ok) enc_pass = 0;
    printf("    %-9s C=% .5e ref=% .5e  rel=%.4f %s\n",
           ms_ref_stages[i].name, stats[i].sum, refs, rel, ok ? "ok" : "BAD");
  }

  int toks[64], n_out;
  ms_decode_greedy(&m, enc_out, n_pos, toks, MS_REF_GREEDY_LEN, 0, &n_out);
  free(enc_out);

  int tok_cmp = (n_out < MS_REF_GREEDY_LEN) ? n_out : MS_REF_GREEDY_LEN, tok_match = 0;
  printf("[moonshine] greedy tokens (C vs ref):\n    C  =");
  for (int i = 0; i < n_out; i++) printf(" %d", toks[i]);
  printf("\n    ref=");
  for (int i = 0; i < MS_REF_GREEDY_LEN; i++) printf(" %d", ms_ref_greedy_tokens[i]);
  printf("\n");
  for (int i = 0; i < tok_cmp; i++) if (toks[i] == ms_ref_greedy_tokens[i]) tok_match++;
  ms_print_tokens_text("[moonshine] text (C)  : ", toks, n_out);
  ms_print_tokens_text("[moonshine] text (ref): ", ms_ref_greedy_tokens, MS_REF_GREEDY_LEN);

  int pass = enc_pass && (tok_match == tok_cmp) && (n_out == MS_REF_GREEDY_LEN);
  printf("[moonshine] ENC %s  |  TOKENS %d/%d match (n_out=%d ref=%d)\n",
         enc_pass ? "PASS" : "FAIL", tok_match, tok_cmp, n_out, MS_REF_GREEDY_LEN);
  ms_profile_report();   /* no-op unless MS_PROFILE=1 */
  printf("[moonshine] RESULT %s\n", pass ? "PASS" : "FAIL");
  fflush(stdout);   /* validate mode is Spike-only; flush so the pipe/file capture survives _exit */
  return pass ? 0 : 1;
}
