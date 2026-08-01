/*
 * whisper.c — Whisper tiny.en float inference engine for DSP 25 (plan 005).
 *
 * Pure C (no HAL) so it compiles identically for host gcc and the RISC-V/Spike target. Weights are
 * Q8_0, dequantized per-group inside matmul(); activations are float. Encoder + decoder + greedy.
 *
 * The matmul()/attention loops here are the RVV vectorization targets for the int8 phase; keep them
 * simple and correct now, optimize later behind the same stage-sum/token validation.
 */
#include "whisper.h"
#include "whisper_reference.h"
#include "whisper_vocab.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef WHISPER_USE_RVV
#define WHISPER_USE_RVV 0        /* 1 -> vectorized matvec (RISC-V V); 0 -> scalar reference */
#endif
#if WHISPER_USE_RVV
#include <riscv_vector.h>        /* f32 intrinsics for conv1d / attention (no int widening -> GCC-safe) */
#endif
#ifndef WHISPER_DUALCORE
#define WHISPER_DUALCORE 0       /* 1 -> split matmul across 2 harts (requires RVV) */
#endif
#ifndef WHISPER_DUALCORE_MIN_WORK
#define WHISPER_DUALCORE_MIN_WORK 16384  /* attention: parallelize only nq*nk*n_head >= this */
#endif
/* matmul_q8 dual-core threshold. Kept at the attention value: dual-coring the decoder's tiny per-token
 * projections (n=1, out=384/1536) MEASURED SLOWER on silicon — 256 fork/joins/utterance cost more than
 * the split saved (matmul 11.3B -> 13.1B). Only large matmuls (encoder/cross-KV n=100, classifier) fork. */
#ifndef WHISPER_MM_MIN_WORK
#define WHISPER_MM_MIN_WORK 16384
#endif
#ifndef WHISPER_DUALCORE_DEBUG
#define WHISPER_DUALCORE_DEBUG 0         /* 1 -> print around the first dual-core fork/join */
#endif
#if WHISPER_DUALCORE_DEBUG
#define WDBG(...) do { printf(__VA_ARGS__); } while (0)
#else
#define WDBG(...) do {} while (0)
#endif
#if WHISPER_DUALCORE
#if !WHISPER_USE_RVV
#error "WHISPER_DUALCORE requires WHISPER_USE_RVV"
#endif
#include "hthread.h"
#endif
#ifndef WHISPER_INT8_ACT
#define WHISPER_INT8_ACT 0       /* 1 -> quantize activations to int8 (int8xint8 matvec) */
#endif
/* The RVV attention kernel (strided loads + exp_approx_m8 softmax) is gated SEPARATELY: it hangs on
 * DSP-25 silicon (works on Spike) — same class as the RVV DFT front-end. Attention is a small slice
 * of runtime, so it stays scalar on hardware by default. Flip to 1 to re-test the vector attention. */
#ifndef WHISPER_MHA_RVV
#define WHISPER_MHA_RVV 0
#endif
/* Register-tiled outer-product GEMM (XNNPACK/vec-nn style): K-major weights streamed sequentially,
 * one weight vector reused across WHISPER_TILE_MR output positions held in vector-register accumulators
 * (no per-output reduction). Cuts encoder weight DRAM traffic ~MR x versus the reduction kernel while
 * keeping the sequential access this memory system rewards. Weights transposed to K-major lazily at
 * first use. Requires WHISPER_USE_RVV. */
#ifndef WHISPER_TILED
#define WHISPER_TILED 0
#endif
#ifndef WHISPER_TILE_MR
#define WHISPER_TILE_MR 4        /* output positions per weight-vector load (register-blocked) */
#endif
#define WHISPER_MAX_ROW 4096     /* upper bound on `in` for a matvec (mlp hidden = 1536) */

/* Coarse cycle profiling of the hot kernels (rdcycle; RV only). WHISPER_PROFILE=1 to enable. */
#ifndef WHISPER_PROFILE
#define WHISPER_PROFILE 0
#endif
#if WHISPER_PROFILE
static uint64_t g_cyc_conv, g_cyc_mm, g_cyc_attn, g_cyc_gelu, g_cyc_ln, g_cyc_quant;
static inline uint64_t prof_cy(void) { uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }
#define PROF_T0()      uint64_t _pt0 = prof_cy()
#define PROF_ADD(acc)  do { (acc) += prof_cy() - _pt0; } while (0)
#else
#define PROF_T0()      do {} while (0)
#define PROF_ADD(acc)  do {} while (0)
#endif

/* Checked malloc: on this bare target a NULL deref just spins the trap handler, so fail loudly. */
static void *xmalloc(size_t n) {
  void *p = malloc(n);
  if (!p) { printf("[whisper] OOM: malloc(%lu) failed\n", (unsigned long)n); exit(1); }
  return p;
}

/* ------------------------------------------------------------------ blob cursor + model load --- */
typedef struct { const unsigned char *p; } cur_t;

static const float *take_f32(cur_t *c, size_t n) {
  const float *r = (const float *)c->p;
  c->p += n * sizeof(float);
  return r;
}
static wq8_t take_q8(cur_t *c, size_t out, size_t in, int gs) {
  wq8_t w;
  w.q = (const int8_t *)c->p;         c->p += out * in;                 /* int8 values */
  w.s = (const float *)c->p;          c->p += (out * in / gs) * sizeof(float); /* group scales */
  w.qT = 0; w.sT = 0;                 /* K-major transpose built lazily (WHISPER_TILED) */
  return w;
}

int whisper_model_load(const void *blob, whisper_model_t *m) {
  const unsigned char *b = (const unsigned char *)blob;
  uint32_t magic; memcpy(&magic, b, 4);
  if (magic != 0x77683031u) return -1;               /* "wh01" */
  const int *hdr = (const int *)(b + 8);             /* after magic(4) + version(4) */
  m->n_mels = hdr[0];  m->n_audio_ctx = hdr[1];  m->n_audio_state = hdr[2];
  m->n_audio_head = hdr[3];  m->n_audio_layer = hdr[4];
  m->n_text_ctx = hdr[5];  m->n_text_state = hdr[6];  m->n_text_head = hdr[7];
  m->n_text_layer = hdr[8]; m->n_vocab = hdr[9]; m->mlp_hidden = hdr[10]; m->conv_k = hdr[11];
  m->shared_classifier = *(const unsigned char *)(b + 8 + 12 * 4);
  memcpy(&m->gs, b + 8 + 12 * 4 + 1, 4);

  const int S = m->n_audio_state, H = m->mlp_hidden, K = m->conv_k, gs = m->gs;
  const int EL = m->n_audio_layer, DL = m->n_text_layer;

  cur_t c = { b + 256 };                              /* payload starts after the 256-byte header */

  /* FP32 conv stem */
  m->conv1_w = take_f32(&c, (size_t)S * m->n_mels * K); m->conv1_b = take_f32(&c, S);
  m->conv2_w = take_f32(&c, (size_t)S * S * K);         m->conv2_b = take_f32(&c, S);
  /* FP32 positional */
  m->enc_pos = take_f32(&c, (size_t)m->n_audio_ctx * S);
  m->dec_pos = take_f32(&c, (size_t)m->n_text_ctx * S);
  /* FP32 encoder norms/biases */
  for (int i = 0; i < EL; i++) {
    whisper_enc_layer_t *L = &m->enc[i];
    L->attn_ln_w = take_f32(&c, S); L->attn_ln_b = take_f32(&c, S);
    L->attn_q_b = take_f32(&c, S);  L->attn_v_b = take_f32(&c, S);  L->attn_o_b = take_f32(&c, S);
    L->mlp_ln_w = take_f32(&c, S);  L->mlp_ln_b = take_f32(&c, S);
    L->mlp0_b = take_f32(&c, H);    L->mlp2_b = take_f32(&c, S);
  }
  m->enc_ln_post_w = take_f32(&c, S); m->enc_ln_post_b = take_f32(&c, S);
  /* FP32 decoder norms/biases */
  for (int i = 0; i < DL; i++) {
    whisper_dec_layer_t *L = &m->dec[i];
    L->attn_ln_w = take_f32(&c, S); L->attn_ln_b = take_f32(&c, S);
    L->attn_q_b = take_f32(&c, S);  L->attn_v_b = take_f32(&c, S);  L->attn_o_b = take_f32(&c, S);
    L->cross_ln_w = take_f32(&c, S); L->cross_ln_b = take_f32(&c, S);
    L->cross_q_b = take_f32(&c, S); L->cross_v_b = take_f32(&c, S); L->cross_o_b = take_f32(&c, S);
    L->mlp_ln_w = take_f32(&c, S);  L->mlp_ln_b = take_f32(&c, S);
    L->mlp0_b = take_f32(&c, H);    L->mlp2_b = take_f32(&c, S);
  }
  m->dec_ln_w = take_f32(&c, S); m->dec_ln_b = take_f32(&c, S);
  /* Q8_0 block */
  m->token_embedding = take_q8(&c, m->n_vocab, S, gs);
  for (int i = 0; i < EL; i++) {
    whisper_enc_layer_t *L = &m->enc[i];
    L->attn_q = take_q8(&c, S, S, gs); L->attn_k = take_q8(&c, S, S, gs);
    L->attn_v = take_q8(&c, S, S, gs); L->attn_o = take_q8(&c, S, S, gs);
    L->mlp0 = take_q8(&c, H, S, gs);   L->mlp2 = take_q8(&c, S, H, gs);
  }
  for (int i = 0; i < DL; i++) {
    whisper_dec_layer_t *L = &m->dec[i];
    L->attn_q = take_q8(&c, S, S, gs); L->attn_k = take_q8(&c, S, S, gs);
    L->attn_v = take_q8(&c, S, S, gs); L->attn_o = take_q8(&c, S, S, gs);
    L->cross_q = take_q8(&c, S, S, gs); L->cross_k = take_q8(&c, S, S, gs);
    L->cross_v = take_q8(&c, S, S, gs); L->cross_o = take_q8(&c, S, S, gs);
    L->mlp0 = take_q8(&c, H, S, gs);   L->mlp2 = take_q8(&c, S, H, gs);
  }
  return 0;
}

/* ------------------------------------------------------------------------------- primitive ops -- */

/* y[o] = bias[o] + sum_i W(o,i)*x[i], with W(o,i) = q[o*in+i] * s[group]. Q8_0 weight, float
 * activation. This is ~99% of inference cycles — the primary RVV target. The scalar version is the
 * reference oracle; the RVV version must produce identical stage sums / tokens (validated on Spike). */
static void matvec_q8_scalar(float *y, const float *x, const wq8_t *W, const float *bias,
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

#if WHISPER_USE_RVV
/* Hand-written RVV inner kernel (whisper_rvv.S): sum_i (q[i]*s[i/gs])*x[i] for one weight row.
 * In asm because GCC miscompiles the intrinsic widening (emits vsext.vf4 under an e8 vtype). */
extern float whisper_matrow_q8(const int8_t *q, const float *x, const float *s, int in, int gs);

static void matvec_q8_rvv(float *y, const float *x, const wq8_t *W, const float *bias,
                          int out, int in, int gs) {
  const int gpr = in / gs;
  for (int o = 0; o < out; o++) {
    float part = whisper_matrow_q8(W->q + (size_t)o * in, x, W->s + (size_t)o * gpr, in, gs);
    y[o] = (bias ? bias[o] : 0.0f) + part;
  }
}
#endif /* WHISPER_USE_RVV */

#if WHISPER_INT8_ACT
/* Quantize one activation vector to int8 with a single per-vector scale (sx = max|x|/127). Inputs
 * to every matvec are post-LayerNorm / post-GELU (bounded), so a per-vector scale is adequate; the
 * enc_block3 residual outlier never enters a matvec un-normalized. Returns sx; fills xq[in]. */
static float quantize_row_i8(const float *x, int in, int8_t *xq) {
  PROF_T0();
  float amax = 0.0f;
  for (int i = 0; i < in; i++) { float a = x[i] < 0 ? -x[i] : x[i]; if (a > amax) amax = a; }
  float sx = amax / 127.0f;
  float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
  for (int i = 0; i < in; i++) {
    int v = (int)lrintf(x[i] * inv);
    if (v > 127) v = 127; else if (v < -127) v = -127;
    xq[i] = (int8_t)v;
  }
  PROF_ADD(g_cyc_quant);
  return sx;
}

#if WHISPER_USE_RVV
/* int8xint8->int32 inner kernel (whisper_rvv.S): sum_g s[g] * sum_{i in g} q[i]*xq[i]. */
extern float whisper_matrow_i8(const int8_t *q, const int8_t *xq, const float *s, int in, int gs);
static inline float i8_row(const int8_t *q, const int8_t *xq, const float *s, int in, int gs) {
  return whisper_matrow_i8(q, xq, s, in, gs);
}
#else
/* Scalar int8 reference (host): validates the quantization accuracy independently of the RVV asm. */
static float i8_row(const int8_t *q, const int8_t *xq, const float *s, int in, int gs) {
  const int gpr = in / gs;
  float acc = 0.0f;
  for (int g = 0; g < gpr; g++) {
    long dot = 0;
    for (int k = 0; k < gs; k++) dot += (int)q[g * gs + k] * (int)xq[g * gs + k];
    acc += (float)dot * s[g];
  }
  return acc;
}
#endif

static void matvec_q8_i8(float *y, const float *x, const wq8_t *W, const float *bias,
                         int out, int in, int gs) {
  int8_t xq[WHISPER_MAX_ROW];
  float sx = quantize_row_i8(x, in, xq);            /* quantize the activation once per matvec */
  const int gpr = in / gs;
  for (int o = 0; o < out; o++) {
    float part = i8_row(W->q + (size_t)o * in, xq, W->s + (size_t)o * gpr, in, gs);
    y[o] = (bias ? bias[o] : 0.0f) + sx * part;
  }
}
#endif /* WHISPER_INT8_ACT */

static inline void matvec_q8(float *y, const float *x, const wq8_t *W, const float *bias,
                             int out, int in, int gs) {
#if WHISPER_INT8_ACT
  matvec_q8_i8(y, x, W, bias, out, in, gs);
#elif WHISPER_USE_RVV
  matvec_q8_rvv(y, x, W, bias, out, in, gs);
#else
  matvec_q8_scalar(y, x, W, bias, out, in, gs);
#endif
}
#if WHISPER_DUALCORE || WHISPER_TILED
/* Block descriptor + kernels shared by the dual-core split and the register-tiled GEMM. Each block is
 * a disjoint rectangle (output rows [r0,r1) x columns [o0,o1)); fences make halves visible across harts
 * after join (intra-die is coherent — no cache flush, per rvv-matmul-threadlib). */
extern float whisper_matrow_q8(const int8_t *q, const float *x, const float *s, int in, int gs);
#if WHISPER_INT8_ACT
static float quantize_row_i8(const float *x, int in, int8_t *xq);   /* defined below */
extern float whisper_matrow_i8(const int8_t *q, const int8_t *xq, const float *s, int in, int gs);
#endif
typedef struct {
  float *out_mat; const float *in_mat; const wq8_t *W; const float *bias;
  int out, in, gs, r0, r1, o0, o1;
} mm_blk_t;

#if WHISPER_TILED
#include <stdlib.h>
/* Build the K-major transpose of a Q8_0 matrix on first use (single-threaded, before any fork).
 * qT[k*out+o] = q[o*in+k]; sT[g*out+o] = s[o*gpr+g]. Buffers persist for the run (~model-size total). */
static void wq8_build_tiled(wq8_t *W, int out, int in, int gs) {
  if (W->qT) return;
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

/* 4-row register-tiled microkernel: out rows r0..r0+3 (each stride N), output cols [o0,o1).
 * Each weight vector qT[k][o-block] is loaded once and fed into 4 vfmacc (one per row) — accumulators
 * stay in vector registers across the whole K loop. Q8_0 group scale sT[g][o-block] is folded per lane
 * (reloaded each 64-wide group; weight vector scaled once per k). */
static void gemm_tiled_mr4(float *o0p, float *o1p, float *o2p, float *o3p,
                           const float *x0, const float *x1, const float *x2, const float *x3,
                           const int8_t *qT, const float *sT, const float *bias,
                           int N, int K, int gs, int oc0, int oc1) {
  for (int o = oc0; o < oc1;) {
    size_t vl = __riscv_vsetvl_e32m4((size_t)(oc1 - o));
    vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t a1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t a2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t a3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t sv = a0; int cg = -1;
    for (int k = 0; k < K; k++) {
      int g = k / gs;
      if (g != cg) { sv = __riscv_vle32_v_f32m4(sT + (size_t)g * N + o, vl); cg = g; }
      vint8m1_t w8 = __riscv_vle8_v_i8m1(qT + (size_t)k * N + o, vl);
      /* int8 -> f32 via the vwcvt/vfcvt chain proven on dsp25 silicon (vec-nn/borai); avoid the
       * widening float-convert (vfwcvt.f.x) and the 4x vsext that GCC miscompiles on this toolchain. */
      vint32m4_t w32 = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwcvt_x_x_v_i16m2(w8, vl), vl);
      vfloat32m4_t wf = __riscv_vfcvt_f_x_v_f32m4(w32, vl);
      wf = __riscv_vfmul_vv_f32m4(wf, sv, vl);
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

/* 1-row tail microkernel (same dataflow, single accumulator). */
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
      /* int8 -> f32 via the vwcvt/vfcvt chain proven on dsp25 silicon (vec-nn/borai); avoid the
       * widening float-convert (vfwcvt.f.x) and the 4x vsext that GCC miscompiles on this toolchain. */
      vint32m4_t w32 = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwcvt_x_x_v_i16m2(w8, vl), vl);
      vfloat32m4_t wf = __riscv_vfcvt_f_x_v_f32m4(w32, vl);
      wf = __riscv_vfmul_vv_f32m4(wf, sv, vl);
      a0 = __riscv_vfmacc_vf_f32m4(a0, x0[k], wf, vl);
    }
    if (bias) a0 = __riscv_vfadd_vv_f32m4(a0, __riscv_vle32_v_f32m4(bias + o, vl), vl);
    __riscv_vse32_v_f32m4(o0p + o, a0, vl);
    o += (int)vl;
  }
}

/* Tiled GEMM over one hart's block (rows [r0,r1) x cols [o0,o1)): 4 rows at a time, tail 1-at-a-time.
 * (An mr=8 m2 variant was tried and measured SLOWER on silicon — the narrower m2 vectors doubled the
 * loop count and the tiled encoder matmul is closer to instruction-bound than weight-DRAM-bound here,
 * so fewer weight streams didn't pay for the extra iterations. mr=4 m4 stays.) */
static void gemm_tiled_block(const mm_blk_t *b) {
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
#endif /* WHISPER_TILED */

static void matmul_block(const mm_blk_t *b) {
  const int gpr = b->in / b->gs;
#if WHISPER_TILED
  if (b->W->qT) { gemm_tiled_block(b); return; }
#endif
#if WHISPER_INT8_ACT
  int8_t xq[WHISPER_MAX_ROW];
  for (int r = b->r0; r < b->r1; r++) {
    const float *xrow = b->in_mat + (size_t)r * b->in;
    const float sx = quantize_row_i8(xrow, b->in, xq);   /* quantize this row's activation once */
    float *orow = b->out_mat + (size_t)r * b->out;
    for (int o = b->o0; o < b->o1; o++)
      orow[o] = (b->bias ? b->bias[o] : 0.0f)
              + sx * whisper_matrow_i8(b->W->q + (size_t)o * b->in, xq, b->W->s + (size_t)o * gpr, b->in, b->gs);
  }
#else
  for (int r = b->r0; r < b->r1; r++) {
    float *orow = b->out_mat + (size_t)r * b->out;
    const float *xrow = b->in_mat + (size_t)r * b->in;
    for (int o = b->o0; o < b->o1; o++)
      orow[o] = (b->bias ? b->bias[o] : 0.0f)
              + whisper_matrow_q8(b->W->q + (size_t)o * b->in, xrow, b->W->s + (size_t)o * gpr, b->in, b->gs);
  }
#endif
}
#if WHISPER_DUALCORE
volatile int g_h1_stage = 0;   /* hart-1 progress beacon: 0=idle 1=entered 2=done (dual-core debug) */
static void mm_worker(void *a) {
  g_h1_stage = 1; __sync_synchronize();
  matmul_block((const mm_blk_t *)a);
  __sync_synchronize(); g_h1_stage = 2;
}
static void mm_nop(void *a) { (void)a; }
/* hart-1's block descriptor lives in BSS (not hart 0's stack) so it's visible to the other hart the
 * same way the hthread deque is. Only hart 0 issues work (hart 1 is the worker), and matmul is not
 * nested, so a single global is safe. */
static mm_blk_t g_mm_h1;

/* Start the second hart (once). Mirrors rvv-matmul-threadlib: init + a nop warmup dispatch. */
void whisper_dualcore_init(void) {
  hthread_init();
  hthread_issue(1, mm_nop, NULL);
  hthread_join(1);
}
#endif /* WHISPER_DUALCORE */
#endif /* WHISPER_DUALCORE || WHISPER_TILED */
#if !WHISPER_DUALCORE
void whisper_dualcore_init(void) {}   /* stub when there is no second hart */
#endif

/* matmul: n input rows x out outputs. Single-core (dispatcher per row), or — under WHISPER_DUALCORE
 * — split across 2 harts for large matmuls (by rows when n>=2, else output columns). */
static void matmul_q8(float *out_mat, const float *in_mat, const wq8_t *W, const float *bias,
                      int n, int out, int in, int gs) {
  PROF_T0();
#if WHISPER_TILED
  /* Tile only many-row matmuls (encoder / cross-KV, n=100): the register tiling reuses each weight
   * vector across MR rows, so a single-row matmul (classifier n=1, decoder per-token) gains nothing
   * and its K-major transpose would just waste ~20 MB (the 51864-wide token embedding). Those stay on
   * the reduction kernel. Build the K-major transpose here on hart 0, before any fork. */
  if (n >= WHISPER_TILE_MR) wq8_build_tiled((wq8_t *)W, out, in, gs);
#endif
#if WHISPER_DUALCORE
  if ((long)n * out >= WHISPER_MM_MIN_WORK) {
    mm_blk_t h0 = { out_mat, in_mat, W, bias, out, in, gs, 0, n, 0, out };
    g_mm_h1 = h0;
    if (n >= 2) { int mid = n / 2;   h0.r1 = mid; g_mm_h1.r0 = mid; }  /* split rows */
    else        { int mid = out / 2; h0.o1 = mid; g_mm_h1.o0 = mid; }  /* split columns (n==1) */
#if WHISPER_DUALCORE_DEBUG
    static int dbg_first = 1;
    if (dbg_first) printf("[whisper] dc matmul: n=%d out=%d in=%d -> issue hart1\n", n, out, in);
#endif
    g_h1_stage = 0;
    __asm__ volatile("fence rw, rw" ::: "memory");
    hthread_issue(1, mm_worker, &g_mm_h1);  /* hart 1 does its block (descriptor in BSS) */
#if WHISPER_DUALCORE_DEBUG
    if (dbg_first) printf("[whisper] dc matmul: issue returned (h1_stage=%d) -> hart0 block\n", g_h1_stage);
#endif
    matmul_block(&h0);                       /* hart 0 does its block */
#if WHISPER_DUALCORE_DEBUG
    if (dbg_first) printf("[whisper] dc matmul: hart0 half done (h1_stage=%d) -> join\n", g_h1_stage);
#endif
    hthread_join(1);
    __asm__ volatile("fence rw, rw" ::: "memory");
#if WHISPER_DUALCORE_DEBUG
    if (dbg_first) { printf("[whisper] dc matmul: join done (both harts ok, h1_stage=%d)\n", g_h1_stage); dbg_first = 0; }
#endif
    PROF_ADD(g_cyc_mm);
    return;
  }
#endif
#if WHISPER_TILED
  if (W->qT) {   /* transpose was built (many-row matmul) -> single-core tiled path */
    mm_blk_t b = { out_mat, in_mat, W, bias, out, in, gs, 0, n, 0, out };
    gemm_tiled_block(&b);
    PROF_ADD(g_cyc_mm);
    return;
  }
#endif
  for (int r = 0; r < n; r++)
    matvec_q8(out_mat + (size_t)r * out, in_mat + (size_t)r * in, W, bias, out, in, gs);
  PROF_ADD(g_cyc_mm);
}

/* tanh via a Padé[7/6] rational approximation (accurate to ~1e-9 for |x|<4.9, clamp beyond) — no
 * exp, so it vectorizes with plain f32 ops + one divide. Used by the tanh-approximation GELU. */
static inline float tanh_pade(float x) {
  if (x > 4.9f) return 1.0f;
  if (x < -4.9f) return -1.0f;
  float x2 = x * x;
  float p = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
  float q = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
  return p / q;
}
/* tanh-approximation GELU (matches torch F.gelu(approximate='tanh'); ~1e-3 from the exact-erf form
 * torch uses by default, but validated to preserve tokens). */
static inline float gelu_f(float x) {
  float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + tanh_pade(u));
}

#if WHISPER_USE_RVV
/* Vectorized tanh-approx GELU: the same Padé tanh evaluated over f32m8 lanes (clamp via vfmin/vfmax,
 * Horner via vfmul/vfadd, one vfdiv). */
static void gelu_inplace(float *x, size_t n) {
  PROF_T0();
  for (size_t i = 0; i < n;) {
    size_t vl = __riscv_vsetvl_e32m8(n - i);
    vfloat32m8_t xv = __riscv_vle32_v_f32m8(x + i, vl);
    vfloat32m8_t x2 = __riscv_vfmul_vv_f32m8(xv, xv, vl);
    vfloat32m8_t x3 = __riscv_vfmul_vv_f32m8(x2, xv, vl);
    /* u = 0.79788456 * (x + 0.044715*x^3), clamped to [-4.9, 4.9] */
    vfloat32m8_t u = __riscv_vfmul_vf_f32m8(__riscv_vfmacc_vf_f32m8(xv, 0.044715f, x3, vl), 0.7978845608028654f, vl);
    u = __riscv_vfmin_vf_f32m8(__riscv_vfmax_vf_f32m8(u, -4.9f, vl), 4.9f, vl);
    vfloat32m8_t u2 = __riscv_vfmul_vv_f32m8(u, u, vl);
    /* p = u * (135135 + u2*(17325 + u2*(378 + u2))) */
    vfloat32m8_t p = __riscv_vfadd_vf_f32m8(u2, 378.0f, vl);
    p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, u2, vl), 17325.0f, vl);
    p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, u2, vl), 135135.0f, vl);
    p = __riscv_vfmul_vv_f32m8(p, u, vl);
    /* q = 135135 + u2*(62370 + u2*(3150 + u2*28)) */
    vfloat32m8_t q = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vf_f32m8(u2, 28.0f, vl), 3150.0f, vl);
    q = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(q, u2, vl), 62370.0f, vl);
    q = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(q, u2, vl), 135135.0f, vl);
    vfloat32m8_t th = __riscv_vfdiv_vv_f32m8(p, q, vl);   /* tanh(u) */
    /* gelu = 0.5 * x * (1 + tanh) */
    vfloat32m8_t g = __riscv_vfmul_vf_f32m8(
        __riscv_vfmul_vv_f32m8(xv, __riscv_vfadd_vf_f32m8(th, 1.0f, vl), vl), 0.5f, vl);
    __riscv_vse32_v_f32m8(x + i, g, vl);
    i += vl;
  }
  PROF_ADD(g_cyc_gelu);
}
#else
static void gelu_inplace(float *x, size_t n) {
  PROF_T0();
  for (size_t i = 0; i < n; i++) x[i] = gelu_f(x[i]);
  PROF_ADD(g_cyc_gelu);
}
#endif

/* LayerNorm over last dim (dim) for each of n rows, eps 1e-5, affine (w,b). */
static void layernorm(const float *x, const float *w, const float *b, float *out, int n, int dim) {
  PROF_T0();
  for (int r = 0; r < n; r++) {
    const float *xr = x + (size_t)r * dim;
    float *orow = out + (size_t)r * dim;
    float mean = 0.0f; for (int i = 0; i < dim; i++) mean += xr[i]; mean /= dim;
    float var = 0.0f; for (int i = 0; i < dim; i++) { float d = xr[i] - mean; var += d * d; } var /= dim;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < dim; i++) orow[i] = (xr[i] - mean) * inv * w[i] + b[i];
  }
  PROF_ADD(g_cyc_ln);
}

/* Multi-head attention core. q(nq,S), k(nk,S), v(nk,S) already projected. causal: query i attends
 * keys 0..i. Writes out(nq,S). Scale = 1/sqrt(head_dim) applied to the q.k dot. */
/* Multi-head attention over a range of heads [h0,h1). q(nq,S), k(nk,S), v(nk,S) already projected.
 * causal: query i attends keys 0..i. Writes the corresponding head columns of out(nq,S). sc is a
 * caller-provided per-hart scratch of >= nk floats (no malloc -> safe to run concurrently on 2 harts).
 * Heads are independent (disjoint output columns), so a head range can run on its own hart. */
#define WHISPER_MHA_MAX_KEYS 2048   /* stack scratch bound (>= max encoder positions) */

#if WHISPER_USE_RVV
/* Vectorized expf approximation (2^(x*log2e) via round + degree-4 poly of 2^frac + bit-built 2^int).
 * Accurate to ~1e-6; softmax is robust to that. Inputs are <= 0 (x = score - max), clamped to avoid
 * denormals/overflow. */
static inline vfloat32m8_t exp_approx_m8(vfloat32m8_t x, size_t vl) {
  x = __riscv_vfmax_vf_f32m8(x, -87.0f, vl);
  vfloat32m8_t y = __riscv_vfmul_vf_f32m8(x, 1.4426950408889634f, vl);   /* x * log2(e) */
  vint32m8_t k = __riscv_vfcvt_x_f_v_i32m8(y, vl);                        /* round to nearest */
  vfloat32m8_t f = __riscv_vfsub_vv_f32m8(y, __riscv_vfcvt_f_x_v_f32m8(k, vl), vl);  /* frac in [-.5,.5] */
  /* p(f) ~= 2^f (Taylor of 2^f) */
  vfloat32m8_t p = __riscv_vfmv_v_f_f32m8(0.0096181291f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, f, vl), 0.0555041087f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, f, vl), 0.2402265069f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, f, vl), 0.6931471805f, vl);
  p = __riscv_vfadd_vf_f32m8(__riscv_vfmul_vv_f32m8(p, f, vl), 1.0f, vl);
  /* 2^k = reinterpret((k+127) << 23) */
  vfloat32m8_t pow2k = __riscv_vreinterpret_v_i32m8_f32m8(
      __riscv_vsll_vx_i32m8(__riscv_vadd_vx_i32m8(k, 127, vl), 23, vl));
  return __riscv_vfmul_vv_f32m8(p, pow2k, vl);
}
#endif

static void mha_run_heads(const float *q, const float *k, const float *v, int nq, int nk,
                          int S, int n_head, int causal, float *out, int h0, int h1, float *sc) {
  const int hd = S / n_head;
  const float rscale = 1.0f / sqrtf((float)hd);
#if WHISPER_USE_RVV && WHISPER_MHA_RVV
  const int use_rvv = ((size_t)hd <= __riscv_vsetvlmax_e32m8());
  const size_t vl = use_rvv ? __riscv_vsetvl_e32m8((size_t)hd) : 0;
#endif
  for (int h = h0; h < h1; h++) {
    const int off = h * hd;
    for (int i = 0; i < nq; i++) {
      const float *qi = q + (size_t)i * S + off;
      const int lim = causal ? (i + 1) : nk;
      float mx = -1e30f;
#if WHISPER_USE_RVV && WHISPER_MHA_RVV
      if (use_rvv) {
        /* transpose formulation: vectorize the scores over KEYS and accumulate across head_dim with
         * vfmacc.vf (strided load of each head-dim column) — no per-key reduction. */
        for (int j0 = 0; j0 < lim;) {
          size_t vlk = __riscv_vsetvl_e32m8((size_t)(lim - j0));
          vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vlk);
          const float *kb = k + (size_t)j0 * S + off;         /* k[j0][off + 0] */
          for (int d = 0; d < hd; d++)                        /* k[:,d] strided by S across keys */
            acc = __riscv_vfmacc_vf_f32m8(acc, qi[d],
                    __riscv_vlse32_v_f32m8(kb + d, (ptrdiff_t)S * sizeof(float), vlk), vlk);
          __riscv_vse32_v_f32m8(sc + j0, __riscv_vfmul_vf_f32m8(acc, rscale, vlk), vlk);
          j0 += (int)vlk;
        }
        for (int j = 0; j < lim; j++) if (sc[j] > mx) mx = sc[j];
      } else
#endif
      {
        for (int j = 0; j < lim; j++) {
          const float *kj = k + (size_t)j * S + off;
          float dot = 0.0f; for (int d = 0; d < hd; d++) dot += qi[d] * kj[d];
          dot *= rscale; sc[j] = dot; if (dot > mx) mx = dot;
        }
      }
      float sum = 0.0f;
#if WHISPER_USE_RVV && WHISPER_MHA_RVV
      {
        const size_t vmax = __riscv_vsetvlmax_e32m8();
        vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, vmax);
        for (int j = 0; j < lim;) {
          size_t vlx = __riscv_vsetvl_e32m8((size_t)(lim - j));
          vfloat32m8_t e = exp_approx_m8(__riscv_vfsub_vf_f32m8(__riscv_vle32_v_f32m8(sc + j, vlx), mx, vlx), vlx);
          __riscv_vse32_v_f32m8(sc + j, e, vlx);
          vsum = __riscv_vfadd_vv_f32m8(vsum, e, vlx);
          j += (int)vlx;
        }
        sum = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m8_f32m1(vsum, __riscv_vfmv_v_f_f32m1(0.0f, 1), vmax));
      }
#else
      for (int j = 0; j < lim; j++) { float e = expf(sc[j] - mx); sc[j] = e; sum += e; }
#endif
      float invs = 1.0f / sum;
      float *oi = out + (size_t)i * S + off;
#if WHISPER_USE_RVV && WHISPER_MHA_RVV
      if (use_rvv) {
        vfloat32m8_t ov = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        for (int j = 0; j < lim; j++)
          ov = __riscv_vfmacc_vf_f32m8(ov, sc[j] * invs, __riscv_vle32_v_f32m8(v + (size_t)j * S + off, vl), vl);
        __riscv_vse32_v_f32m8(oi, ov, vl);
      } else
#endif
      {
        for (int d = 0; d < hd; d++) oi[d] = 0.0f;
        for (int j = 0; j < lim; j++) {
          float wj = sc[j] * invs;
          const float *vj = v + (size_t)j * S + off;
          for (int d = 0; d < hd; d++) oi[d] += wj * vj[d];
        }
      }
    }
  }
}

#if WHISPER_DUALCORE
typedef struct { const float *q, *k, *v; int nq, nk, S, n_head, causal; float *out; int h0, h1; } mha_arg_t;
static mha_arg_t g_mha_h1;   /* BSS, not hart 0's stack — visible to hart 1 (see g_mm_h1) */
static void mha_worker(void *a_) {
  mha_arg_t *a = (mha_arg_t *)a_;
  float sc[WHISPER_MHA_MAX_KEYS];
  mha_run_heads(a->q, a->k, a->v, a->nq, a->nk, a->S, a->n_head, a->causal, a->out, a->h0, a->h1, sc);
}
#endif

static void mha_core(const float *q, const float *k, const float *v, int nq, int nk,
                     int S, int n_head, int causal, float *out) {
  PROF_T0();
  float sc[WHISPER_MHA_MAX_KEYS];
#if WHISPER_DUALCORE
  if (nk <= WHISPER_MHA_MAX_KEYS && n_head >= 2 && (long)nq * nk * n_head >= WHISPER_DUALCORE_MIN_WORK) {
    int mid = n_head / 2;
    g_mha_h1 = (mha_arg_t){ q, k, v, nq, nk, S, n_head, causal, out, mid, n_head };
    __asm__ volatile("fence rw, rw" ::: "memory");
    hthread_issue(1, mha_worker, &g_mha_h1);                            /* hart 1: heads [mid,n_head) */
    mha_run_heads(q, k, v, nq, nk, S, n_head, causal, out, 0, mid, sc); /* hart 0: heads [0,mid) */
    hthread_join(1);
    __asm__ volatile("fence rw, rw" ::: "memory");
    PROF_ADD(g_cyc_attn);
    return;
  }
#endif
  if (nk <= WHISPER_MHA_MAX_KEYS) {
    mha_run_heads(q, k, v, nq, nk, S, n_head, causal, out, 0, n_head, sc);
  } else {
    float *big = (float *)xmalloc((size_t)nk * sizeof(float));       /* rare: nk > stack bound */
    mha_run_heads(q, k, v, nq, nk, S, n_head, causal, out, 0, n_head, big);
    free(big);
  }
  PROF_ADD(g_cyc_attn);
}

static void add_inplace(float *a, const float *b, size_t n) { for (size_t i = 0; i < n; i++) a[i] += b[i]; }

static void stat_of(const float *x, size_t n, whisper_stat_t *st) {
  double sum = 0.0, amax = 0.0;
  for (size_t i = 0; i < n; i++) { double v = x[i]; sum += v; double a = v < 0 ? -v : v; if (a > amax) amax = a; }
  st->sum = sum; st->absmax = amax; st->mean = sum / (double)n;
}

/* conv1d: in(Cin,L), weight(Cout,Cin,3), bias(Cout), pad=1, given stride -> out(Cout,Lout).
 * out[oc,t] = bias[oc] + sum_ic sum_{kk=0..2} w[oc,ic,kk] * in[ic, t*stride + kk - 1]. */
static void conv1d_scalar(const float *in, int Cin, int L, const float *w, const float *b,
                          int Cout, int stride, float *out, int Lout) {
  for (int oc = 0; oc < Cout; oc++) {
    const float *wo = w + (size_t)oc * Cin * 3;
    for (int t = 0; t < Lout; t++) {
      float acc = b[oc];
      const int base = t * stride - 1;
      for (int ic = 0; ic < Cin; ic++) {
        const float *wic = wo + ic * 3;
        const float *inic = in + (size_t)ic * L;
        for (int kk = 0; kk < 3; kk++) {
          int idx = base + kk;
          if (idx >= 0 && idx < L) acc += wic[kk] * inic[idx];
        }
      }
      out[(size_t)oc * Lout + t] = acc;
    }
  }
}

#if WHISPER_USE_RVV
#include <riscv_vector.h>
/* One output point (oc,t) with bounds checks — used for the few boundary positions. */
static float conv1d_point(const float *in, int Cin, int L, const float *w, float bias,
                          int oc, int stride, int t) {
  float acc = bias;
  const int base = t * stride - 1;
  for (int ic = 0; ic < Cin; ic++) {
    const float *wv = w + ((size_t)oc * Cin + ic) * 3;
    const float *inrow = in + (size_t)ic * L;
    for (int kk = 0; kk < 3; kk++) { int idx = base + kk; if (idx >= 0 && idx < L) acc += wv[kk] * inrow[idx]; }
  }
  return acc;
}

/* RVV conv1d: vectorize over output positions t (f32m8), accumulate over input channels and the 3
 * taps with vfmacc.vf (weight tap = scalar). Interior t (all 3 taps in-bounds) is vectorized with
 * the accumulator held in registers; the handful of boundary positions fall back to scalar. Pure
 * f32 (no int widening) so GCC intrinsics compile correctly. Idiom from dsp25-bmarks/simple-conv. */
static void conv1d_rvv(int oc0, int oc1, const float *in, int Cin, int L, const float *w, const float *b,
                       int stride, float *out, int Lout) {
  int t_beg = 1;                                   /* t=0 has tap kk=0 -> idx<0 */
  int t_end = (L - 2) / stride + 1;                /* first t whose kk=2 tap goes >= L (exclusive) */
  if (t_end > Lout) t_end = Lout;
  if (t_beg > t_end) t_beg = t_end;
  const ptrdiff_t bstride = (ptrdiff_t)stride * (ptrdiff_t)sizeof(float);

  for (int oc = oc0; oc < oc1; oc++) {
    float *orow = out + (size_t)oc * Lout;
    const float bias = b[oc];
    for (int t = 0; t < t_beg; t++)        orow[t] = conv1d_point(in, Cin, L, w, bias, oc, stride, t);
    for (int t = t_end; t < Lout; t++)     orow[t] = conv1d_point(in, Cin, L, w, bias, oc, stride, t);

    int t = t_beg;
    while (t < t_end) {
      size_t vl = __riscv_vsetvl_e32m8((size_t)(t_end - t));
      vfloat32m8_t vacc = __riscv_vfmv_v_f_f32m8(bias, vl);
      for (int ic = 0; ic < Cin; ic++) {
        const float *wv = w + ((size_t)oc * Cin + ic) * 3;
        const float *base = in + (size_t)ic * L + (size_t)(stride * t - 1);
        if (stride == 1) {
          vacc = __riscv_vfmacc_vf_f32m8(vacc, wv[0], __riscv_vle32_v_f32m8(base + 0, vl), vl);
          vacc = __riscv_vfmacc_vf_f32m8(vacc, wv[1], __riscv_vle32_v_f32m8(base + 1, vl), vl);
          vacc = __riscv_vfmacc_vf_f32m8(vacc, wv[2], __riscv_vle32_v_f32m8(base + 2, vl), vl);
        } else {
          vacc = __riscv_vfmacc_vf_f32m8(vacc, wv[0], __riscv_vlse32_v_f32m8(base + 0, bstride, vl), vl);
          vacc = __riscv_vfmacc_vf_f32m8(vacc, wv[1], __riscv_vlse32_v_f32m8(base + 1, bstride, vl), vl);
          vacc = __riscv_vfmacc_vf_f32m8(vacc, wv[2], __riscv_vlse32_v_f32m8(base + 2, bstride, vl), vl);
        }
      }
      __riscv_vse32_v_f32m8(orow + t, vacc, vl);
      t += (int)vl;
    }
  }
}
#endif /* WHISPER_USE_RVV */

#if WHISPER_DUALCORE
/* Split conv1d output channels [0,Cout) across 2 harts (channels are independent). */
typedef struct { const float *in; int Cin, L; const float *w, *b; int stride; float *out; int Lout; int oc0, oc1; } conv_blk_t;
static conv_blk_t g_conv_h1;
static void conv_worker(void *a) {
  const conv_blk_t *c = (const conv_blk_t *)a;
  conv1d_rvv(c->oc0, c->oc1, c->in, c->Cin, c->L, c->w, c->b, c->stride, c->out, c->Lout);
}
#endif

static void conv1d(const float *in, int Cin, int L, const float *w, const float *b,
                   int Cout, int stride, float *out, int Lout) {
  PROF_T0();
#if WHISPER_USE_RVV
#if WHISPER_DUALCORE
  if (Cout >= 2) {
    int mid = Cout / 2;
    g_conv_h1 = (conv_blk_t){ in, Cin, L, w, b, stride, out, Lout, mid, Cout };  /* hart 1: [mid,Cout) */
    __asm__ volatile("fence rw, rw" ::: "memory");
    hthread_issue(1, conv_worker, &g_conv_h1);
    conv1d_rvv(0, mid, in, Cin, L, w, b, stride, out, Lout);                     /* hart 0: [0,mid) */
    hthread_join(1);
    __asm__ volatile("fence rw, rw" ::: "memory");
  } else
#endif
    conv1d_rvv(0, Cout, in, Cin, L, w, b, stride, out, Lout);
#else
  conv1d_scalar(in, Cin, L, w, b, Cout, stride, out, Lout);
#endif
  PROF_ADD(g_cyc_conv);
}

/* --------------------------------------------------------------------------------- encoder ------ */
int whisper_encode(const whisper_model_t *m, const float *mel, int n_frames,
                   float **enc_out_p, int *n_pos_p, whisper_stat_t *stats) {
  const int S = m->n_audio_state, H = m->mlp_hidden, nh = m->n_audio_head;
  const int L1 = n_frames;
  const int L2 = (n_frames - 1) / 2 + 1;             /* conv2 stride2 pad1 k3 */
  int si = 0;

  WDBG("[whisper] encode: n_frames=%d L1=%d L2=%d S=%d\n", n_frames, L1, L2, S);
  float *c1 = (float *)xmalloc((size_t)S * L1 * sizeof(float));
  conv1d(mel, m->n_mels, L1, m->conv1_w, m->conv1_b, S, 1, c1, L1);
  gelu_inplace(c1, (size_t)S * L1);
  if (stats) stat_of(c1, (size_t)S * L1, &stats[si]); si++;

  float *c2 = (float *)xmalloc((size_t)S * L2 * sizeof(float));
  conv1d(c1, S, L1, m->conv2_w, m->conv2_b, S, 2, c2, L2);
  gelu_inplace(c2, (size_t)S * L2);
  free(c1);
  if (stats) stat_of(c2, (size_t)S * L2, &stats[si]); si++;

  /* transpose (S, L2) -> x (L2, S), add positional */
  const int T = L2;
  float *x = (float *)xmalloc((size_t)T * S * sizeof(float));
  for (int t = 0; t < T; t++)
    for (int d = 0; d < S; d++)
      x[(size_t)t * S + d] = c2[(size_t)d * L2 + t] + m->enc_pos[(size_t)t * S + d];
  free(c2);
  if (stats) stat_of(x, (size_t)T * S, &stats[si]); si++;

  float *h  = (float *)xmalloc((size_t)T * S * sizeof(float));
  float *qb = (float *)xmalloc((size_t)T * S * sizeof(float));
  float *kb = (float *)xmalloc((size_t)T * S * sizeof(float));
  float *vb = (float *)xmalloc((size_t)T * S * sizeof(float));
  float *ab = (float *)xmalloc((size_t)T * S * sizeof(float));
  float *pb = (float *)xmalloc((size_t)T * S * sizeof(float));
  float *hid = (float *)xmalloc((size_t)T * H * sizeof(float));

  for (int l = 0; l < m->n_audio_layer; l++) {
    WDBG("[whisper] encode layer %d/%d\n", l, m->n_audio_layer);
    const whisper_enc_layer_t *Ly = &m->enc[l];
    /* self-attention branch */
    layernorm(x, Ly->attn_ln_w, Ly->attn_ln_b, h, T, S);
    matmul_q8(qb, h, &Ly->attn_q, Ly->attn_q_b, T, S, S, m->gs);
    matmul_q8(kb, h, &Ly->attn_k, NULL,          T, S, S, m->gs);
    matmul_q8(vb, h, &Ly->attn_v, Ly->attn_v_b, T, S, S, m->gs);
    mha_core(qb, kb, vb, T, T, S, nh, 0, ab);
    matmul_q8(pb, ab, &Ly->attn_o, Ly->attn_o_b, T, S, S, m->gs);
    add_inplace(x, pb, (size_t)T * S);
    /* mlp branch */
    layernorm(x, Ly->mlp_ln_w, Ly->mlp_ln_b, h, T, S);
    matmul_q8(hid, h, &Ly->mlp0, Ly->mlp0_b, T, H, S, m->gs);
    gelu_inplace(hid, (size_t)T * H);
    matmul_q8(pb, hid, &Ly->mlp2, Ly->mlp2_b, T, S, H, m->gs);
    add_inplace(x, pb, (size_t)T * S);
    if (stats) stat_of(x, (size_t)T * S, &stats[si]); si++;
  }

  /* ln_post */
  layernorm(x, m->enc_ln_post_w, m->enc_ln_post_b, h, T, S);
  memcpy(x, h, (size_t)T * S * sizeof(float));
  if (stats) stat_of(x, (size_t)T * S, &stats[si]); si++;

  free(h); free(qb); free(kb); free(vb); free(ab); free(pb); free(hid);
  *enc_out_p = x; *n_pos_p = T;
  return 0;
}

/* --------------------------------------------------------------------------------- decoder ------ */
static void deq_row(const wq8_t *W, int row, int in, int gs, float *out) {
  const int8_t *q = W->q + (size_t)row * in;
  const float *s = W->s + (size_t)row * (in / gs);
  for (int i = 0; i < in; i++) out[i] = (float)q[i] * s[i / gs];
}

/* Greedy decode with a KV-cache: each step processes ONLY the new token (1 position) through the
 * layers, appending its self-attention K/V to a per-layer cache and attending to all cached
 * positions. Cost is O(N) positions x model + O(N^2) attention, versus the O(N^2 x model) of
 * recomputing the whole sequence each step — identical outputs (the cache is exact). */
int whisper_decode_greedy(const whisper_model_t *m, const float *enc_out, int n_enc,
                          const int *sot, int sot_len, int eot,
                          int *out_tokens, int max_new, int *n_out) {
  const int S = m->n_text_state, H = m->mlp_hidden, nh = m->n_text_head, DL = m->n_text_layer;
  const int maxL = sot_len + max_new;

  /* cross K,V per layer from the (fixed) encoder output */
  float **ck = (float **)xmalloc(DL * sizeof(float *));
  float **cv = (float **)xmalloc(DL * sizeof(float *));
  /* self-attention K,V caches: one row per processed position, per layer */
  float **kc = (float **)xmalloc(DL * sizeof(float *));
  float **vc = (float **)xmalloc(DL * sizeof(float *));
  WDBG("[whisper] decode: n_enc=%d maxL=%d DL=%d vocab=%d\n", n_enc, maxL, DL, m->n_vocab);
  for (int l = 0; l < DL; l++) {
    ck[l] = (float *)xmalloc((size_t)n_enc * S * sizeof(float));
    cv[l] = (float *)xmalloc((size_t)n_enc * S * sizeof(float));
    matmul_q8(ck[l], enc_out, &m->dec[l].cross_k, NULL,               n_enc, S, S, m->gs);
    matmul_q8(cv[l], enc_out, &m->dec[l].cross_v, m->dec[l].cross_v_b, n_enc, S, S, m->gs);
    kc[l] = (float *)xmalloc((size_t)maxL * S * sizeof(float));
    vc[l] = (float *)xmalloc((size_t)maxL * S * sizeof(float));
  }

  int *tokens = (int *)xmalloc(maxL * sizeof(int));
  for (int i = 0; i < sot_len; i++) tokens[i] = sot[i];

  float *x   = (float *)xmalloc((size_t)S * sizeof(float));    /* single-position buffers */
  float *h   = (float *)xmalloc((size_t)S * sizeof(float));
  float *q   = (float *)xmalloc((size_t)S * sizeof(float));
  float *ao  = (float *)xmalloc((size_t)S * sizeof(float));
  float *pb  = (float *)xmalloc((size_t)S * sizeof(float));
  float *hid = (float *)xmalloc((size_t)H * sizeof(float));
  float *logits = (float *)xmalloc((size_t)m->n_vocab * sizeof(float));
  int produced = 0;
  WDBG("[whisper] decode: cross-KV done, starting %d positions\n", maxL);

  for (int pos = 0; pos < maxL; pos++) {
    WDBG("[whisper] decode pos=%d produced=%d\n", pos, produced);
    /* embed tokens[pos] + positional[pos] */
    deq_row(&m->token_embedding, tokens[pos], S, m->gs, x);
    const float *pe = m->dec_pos + (size_t)pos * S;
    for (int d = 0; d < S; d++) x[d] += pe[d];

    for (int l = 0; l < DL; l++) {
      const whisper_dec_layer_t *Ly = &m->dec[l];
      WDBG("[whisper]   pos=%d L%d selfattn (kv=%d)\n", pos, l, pos + 1);
      /* self-attention: project q,k,v for this position; cache k,v; attend to 0..pos (all cached) */
      layernorm(x, Ly->attn_ln_w, Ly->attn_ln_b, h, 1, S);
      WDBG("[whisper]     ln done\n");
      matmul_q8(q, h, &Ly->attn_q, Ly->attn_q_b, 1, S, S, m->gs);
      WDBG("[whisper]     q done\n");
      matmul_q8(kc[l] + (size_t)pos * S, h, &Ly->attn_k, NULL, 1, S, S, m->gs);
      WDBG("[whisper]     k done\n");
      matmul_q8(vc[l] + (size_t)pos * S, h, &Ly->attn_v, Ly->attn_v_b, 1, S, S, m->gs);
      WDBG("[whisper]     v done, calling mha (nk=%d)\n", pos + 1);
      mha_core(q, kc[l], vc[l], 1, pos + 1, S, nh, 0, ao);
      WDBG("[whisper]     mha done\n");
      matmul_q8(pb, ao, &Ly->attn_o, Ly->attn_o_b, 1, S, S, m->gs);
      WDBG("[whisper]     o done\n");
      for (int d = 0; d < S; d++) x[d] += pb[d];
      /* cross-attention into encoder output */
      WDBG("[whisper]   pos=%d L%d crossattn (nk=%d)\n", pos, l, n_enc);
      layernorm(x, Ly->cross_ln_w, Ly->cross_ln_b, h, 1, S);
      matmul_q8(q, h, &Ly->cross_q, Ly->cross_q_b, 1, S, S, m->gs);
      mha_core(q, ck[l], cv[l], 1, n_enc, S, nh, 0, ao);
      matmul_q8(pb, ao, &Ly->cross_o, Ly->cross_o_b, 1, S, S, m->gs);
      for (int d = 0; d < S; d++) x[d] += pb[d];
      /* mlp */
      WDBG("[whisper]   pos=%d L%d mlp\n", pos, l);
      layernorm(x, Ly->mlp_ln_w, Ly->mlp_ln_b, h, 1, S);
      matmul_q8(hid, h, &Ly->mlp0, Ly->mlp0_b, 1, H, S, m->gs);
      gelu_inplace(hid, (size_t)H);
      matmul_q8(pb, hid, &Ly->mlp2, Ly->mlp2_b, 1, S, H, m->gs);
      for (int d = 0; d < S; d++) x[d] += pb[d];
    }
    WDBG("[whisper]   pos=%d layers done -> classifier\n", pos);

    /* emit once the last SOT position (and each generated token thereafter) has been processed */
    if (pos >= sot_len - 1) {
      layernorm(x, m->dec_ln_w, m->dec_ln_b, h, 1, S);
      /* tied classifier — route through matmul_q8 so it's dual-cored (n=1 -> column split) + profiled */
      matmul_q8(logits, h, &m->token_embedding, NULL, 1, m->n_vocab, S, m->gs);
      int best = 0; float bv = logits[0];
      for (int v = 1; v < m->n_vocab; v++) if (logits[v] > bv) { bv = logits[v]; best = v; }
      out_tokens[produced++] = best;
      WDBG("[whisper] decode pos=%d -> tok=%d (logit=%d)\n", pos, best, (int)bv);
      if (best == eot || produced >= max_new) break;
      if (pos + 1 < maxL) tokens[pos + 1] = best;
      /* Repetition guard: truncated / low-SNR audio makes greedy decode loop (it never emits eot).
       * If the last C tokens duplicate the preceding C, we're in a cycle — drop the repeat and stop. */
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
  free(x); free(h); free(q); free(ao); free(pb); free(hid); free(logits);
  return 0;
}

/* Decode a token sequence to text via the byte-level BPE vocab table (whisper_vocab.h). Specials
 * and eot (id >= WV_EOT) emit nothing. Concatenated bytes reproduce PyTorch tok.decode(). */
static void whisper_print_text(const char *tag, const int *toks, int n) {
  printf("%s\"", tag);
  for (int i = 0; i < n; i++) {
    int t = toks[i];
    if (t < 0 || t >= WV_N_VOCAB || t >= WV_EOT) continue;
    for (int b = wv_offset[t]; b < wv_offset[t + 1]; b++) putchar(wv_bytes[b]);
  }
  printf("\"\n");
}

void whisper_print_tokens_text(const char *tag, const int *toks, int n) {
  whisper_print_text(tag, toks, n);
}

void whisper_profile_report(void) {
#if WHISPER_PROFILE
  printf("[whisper] PROFILE cycles: matmul=%lu attn=%lu conv=%lu gelu=%lu layernorm=%lu quant=%lu\n",
         (unsigned long)g_cyc_mm, (unsigned long)g_cyc_attn, (unsigned long)g_cyc_conv,
         (unsigned long)g_cyc_gelu, (unsigned long)g_cyc_ln, (unsigned long)g_cyc_quant);
#endif
}

/* Zero the kernel cycle counters — call at the start of each utterance so the PROFILE line is
 * per-utterance, not a running sum across every word spoken this session. */
void whisper_profile_reset(void) {
#if WHISPER_PROFILE
  g_cyc_conv = g_cyc_mm = g_cyc_attn = g_cyc_gelu = g_cyc_ln = g_cyc_quant = 0;
#endif
}

int whisper_transcribe(const whisper_model_t *m, const float *mel, int n_frames,
                       int *out_tokens, int max_new, int *n_out) {
  float *enc_out; int n_pos;
  whisper_encode(m, mel, n_frames, &enc_out, &n_pos, NULL);
  whisper_decode_greedy(m, enc_out, n_pos, whisper_ref_sot_tokens, WHISPER_REF_SOT_LEN,
                        WHISPER_REF_EOT, out_tokens, max_new, n_out);
  free(enc_out);
  return n_pos;
}

/* --------------------------------------------------------------------------------- validate ----- */
int whisper_run_validate(const void *model_blob, const float *mel, int n_frames) {
  whisper_model_t m;
  if (whisper_model_load(model_blob, &m) != 0) { printf("[whisper] bad model magic\n"); return 1; }
  printf("[whisper] model: state=%d enc_layers=%d dec_layers=%d vocab=%d gs=%d  matvec=%s\n",
         m.n_audio_state, m.n_audio_layer, m.n_text_layer, m.n_vocab, m.gs,
         WHISPER_INT8_ACT ? (WHISPER_USE_RVV ? "int8-RVV" : "int8-scalar")
                          : (WHISPER_USE_RVV ? "RVV" : "scalar"));

  whisper_stat_t stats[8];
  float *enc_out; int n_pos;
  whisper_encode(&m, mel, n_frames, &enc_out, &n_pos, stats);

  int enc_pass = 1;
  printf("[whisper] ENC stage sums (name: C_sum vs ref_sum  rel):\n");
  for (int i = 0; i < WHISPER_REF_NUM_STAGES && i < 8; i++) {
    double refs = whisper_ref_stages[i].sum;
    double rel = (refs != 0.0) ? fabs(stats[i].sum - refs) / fabs(refs) : fabs(stats[i].sum);
    int ok = rel < 0.03;
    if (!ok) enc_pass = 0;
    printf("    %-11s C=% .5e ref=% .5e  rel=%.4f %s\n",
           whisper_ref_stages[i].name, stats[i].sum, refs, rel, ok ? "ok" : "BAD");
  }

  int toks[64], n_out;
  whisper_decode_greedy(&m, enc_out, n_pos, whisper_ref_sot_tokens, WHISPER_REF_SOT_LEN,
                        WHISPER_REF_EOT, toks, WHISPER_REF_GREEDY_LEN, &n_out);
  free(enc_out);

  int tok_match = 0, tok_cmp = (n_out < WHISPER_REF_GREEDY_LEN) ? n_out : WHISPER_REF_GREEDY_LEN;
  printf("[whisper] greedy tokens (C vs ref):\n    C  =");
  for (int i = 0; i < n_out; i++) printf(" %d", toks[i]);
  printf("\n    ref=");
  for (int i = 0; i < WHISPER_REF_GREEDY_LEN; i++) printf(" %d", whisper_ref_greedy_tokens[i]);
  printf("\n");
  for (int i = 0; i < tok_cmp; i++) if (toks[i] == whisper_ref_greedy_tokens[i]) tok_match++;
  whisper_print_text("[whisper] text (C)  : ", toks, n_out);
  whisper_print_text("[whisper] text (ref): ", whisper_ref_greedy_tokens, WHISPER_REF_GREEDY_LEN);

#if WHISPER_PROFILE
  printf("[whisper] PROFILE cycles: matmul=%lu attn=%lu conv=%lu gelu=%lu layernorm=%lu quant=%lu\n",
         (unsigned long)g_cyc_mm, (unsigned long)g_cyc_attn, (unsigned long)g_cyc_conv,
         (unsigned long)g_cyc_gelu, (unsigned long)g_cyc_ln, (unsigned long)g_cyc_quant);
#endif

  int pass = enc_pass && (tok_match == tok_cmp);
  printf("[whisper] ENC %s  |  TOKENS %d/%d match\n", enc_pass ? "PASS" : "FAIL", tok_match, tok_cmp);
  printf("[whisper] RESULT %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
