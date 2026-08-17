/* SmolLM grouped-Q8_0 forward pass — see include/smollm_model.h.
 *
 * Derived from llama2.c's runq.c (Andrej Karpathy, MIT) via dsp25-demos/tinyllama. No chip
 * dependencies: scripts/check_c_forward.py compiles this exact file for the host and checks it
 * against the numpy reference, so the code that runs on silicon is the code that was validated.
 *
 * The one structural change from runq.c is embed_token(): runq.c dequantizes the WHOLE token
 * embedding table at load, which for SmolLM's 49152 x 576 vocab is a 113 MB float array. Here a
 * single row is dequantized per step. */

#include "smollm_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Two separate questions, deliberately not the same switch:
 *   SMOLLM_RVV       — are the vector kernels COMPILED (the toolchain has the V extension)?
 *   SMOLLM_USE_RVV   — does the FORWARD PASS use them?
 * Keeping them apart lets one image compute answers with the trusted scalar kernel while
 * smollm_bench() still times the vector variants and cross-checks them against it — which is how
 * you find out that a vector unit computes the wrong thing without flashing twice. */
#if defined(__riscv_vector)
#include <riscv_vector.h>
#define SMOLLM_RVV 1
#else
#define SMOLLM_RVV 0
#endif

#if SMOLLM_RVV && SMOLLM_USE_RVV
#define SMOLLM_FWD_RVV 1
#else
#define SMOLLM_FWD_RVV 0
#endif

int smollm_uses_rvv(void) { return SMOLLM_FWD_RVV; }

/* Tokens per prefill pass over the weights. Bigger = fewer passes over 143 MB, at ~19 KB of scratch
 * per token; 32 already turns a 31-token prompt into a single pass. */
#ifndef SMOLLM_MAX_BATCH
#define SMOLLM_MAX_BATCH 32
#endif

static smollm_stage_fn g_stage_hook;
void smollm_set_stage_hook(smollm_stage_fn fn) { g_stage_hook = fn; }


#define MODEL_MAGIC 0x616b3432u  /* 'ak42' — llama2.c export.py --version 2 */
#define MODEL_EXT_MAGIC 0x534d4c4du  /* 'SMLM' — the SmolLM extension block in the header pad */

/* A weight tensor as it sits in the blob: quantized values followed by one fp32 scale per group.
 * `q4` selects the packing:
 *   0 — one int8 per weight.
 *   1 — two int4 per byte. Within a group of GS, byte j holds weight j in the LOW nibble and
 *       weight j+GS/2 in the HIGH nibble, so unpacking gives two CONTIGUOUS halves and needs no
 *       shuffle: exactly llama.cpp's Q4_0 layout, chosen for that property. */
typedef struct {
  const int8_t *q;
  const float *s;      /* per group: int8/Q4_0 scale, or Q4_1's `d` */
  const float *m;      /* Q4_1 only: per-group minimum; w = q*d + m */
  int fmt;             /* QFMT_* */
} QTensor;

enum { QFMT_I8 = 0, QFMT_Q4_0 = 1, QFMT_Q4_1 = 2 };

typedef struct {
  int8_t *q;
  float *s;
  float *gsum;   /* per group: sum of the quantized activations, for Q4_1's m * sum(x) term.
                  * Depends only on the activation, so it is computed once per quantize() and
                  * reused across every output row — which is what keeps Q4_1 nearly as cheap
                  * as Q4_0 despite the affine dequantization. */
} QBuf;   /* quantized activations, which we do write */

typedef struct {
  QTensor tokens;              /* (vocab_size, dim), also the tied classifier */
  const float *rms_att;        /* (n_layers, dim) */
  const float *rms_ffn;        /* (n_layers, dim) */
  const float *rms_final;      /* (dim) */
  QTensor *wq, *wk, *wv, *wo;  /* per layer */
  QTensor *w1, *w2, *w3;
} Weights;

typedef struct {
  float *x, *xb, *xb2;   /* residual stream and scratch (dim) */
  float *hb, *hb2;       /* ffn scratch (hidden_dim) */
  QBuf xq, hq;           /* quantized activations */
  float *q, *k, *v;      /* attention projections */
  float *att;            /* (n_heads, seq_len) */
  float *logits;         /* (vocab_size) */
  float *key_cache;      /* (n_layers, seq_len, kv_dim) */
  float *value_cache;
} RunState;

static smollm_config_t g_cfg;
static int g_alloc_failed;
static Weights g_w;
static RunState g_s;
#define GS (g_cfg.group_size)
#define g_kv_dim (g_cfg.kv_dim)

static void *xcalloc(size_t n, size_t sz, const char *what) {
  void *p = calloc(n, sz);
  if (!p) {
    printf("[smollm] FATAL: out of heap allocating %s (%lu bytes)\r\n",
           what, (unsigned long)(n * sz));
    g_alloc_failed = 1;
  }
  return p;
}

/* Carve `n` consecutive (rows x cols) quantized tensors out of the blob, advancing *p. */
static QTensor *map_tensors(const uint8_t **p, int n, size_t elems, int fmt) {
  QTensor *t = (QTensor *)xcalloc((size_t)n, sizeof(QTensor), "tensor table");
  const size_t ngroups = elems / (size_t)GS;
  for (int i = 0; i < n; i++) {
    t[i].q = (const int8_t *)*p;
    t[i].fmt = fmt;
    *p += (fmt == QFMT_I8) ? elems : (elems / 2u);
    t[i].s = (const float *)*p;
    *p += ngroups * sizeof(float);
    if (fmt == QFMT_Q4_1) {
      t[i].m = (const float *)*p;
      *p += ngroups * sizeof(float);
    }
  }
  return t;
}

int smollm_model_load(const uint8_t *blob, size_t blob_bytes) {
  uint32_t magic;
  int version;
  memcpy(&magic, blob + 0, 4);
  memcpy(&version, blob + 4, 4);
  if (magic != MODEL_MAGIC || version != 2) {
    printf("[smollm] BAD MODEL HEADER magic=0x%08lx version=%d (want 'ak42' v2)\r\n",
               (unsigned long)magic, version);
    return -1;
  }
  int hdr[7];
  memcpy(hdr, blob + 8, sizeof(hdr));
  g_cfg.dim = hdr[0]; g_cfg.hidden_dim = hdr[1]; g_cfg.n_layers = hdr[2];
  g_cfg.n_heads = hdr[3]; g_cfg.n_kv_heads = hdr[4]; g_cfg.vocab_size = hdr[5];
  g_cfg.seq_len = hdr[6];
  memcpy(&g_cfg.group_size, blob + 37, 4);   /* deliberately unaligned in the v2 layout */

  uint32_t ext;
  memcpy(&ext, blob + 41, 4);
  if (ext == MODEL_EXT_MAGIC) {
    memcpy(&g_cfg.rope_theta, blob + 45, 4);
    memcpy(&g_cfg.norm_eps, blob + 49, 4);
    memcpy(&g_cfg.head_dim, blob + 53, 4);
    memcpy(&g_cfg.quant_mode, blob + 57, 4);
  } else {
    /* A stock llama2.c v2 blob: fall back to llama2's constants rather than reading garbage. */
    g_cfg.rope_theta = 10000.0f;
    g_cfg.norm_eps = 1e-5f;
    g_cfg.head_dim = g_cfg.dim / g_cfg.n_heads;
    g_cfg.quant_mode = 0;
    printf("[smollm] no SMLM extension block; assuming rope_theta=10000 eps=1e-5\r\n");
  }
  g_cfg.kv_dim = g_cfg.head_dim * g_cfg.n_kv_heads;

  const uint8_t *p = blob + 256;
  const size_t L = (size_t)g_cfg.n_layers, dim = (size_t)g_cfg.dim, hd = (size_t)g_cfg.hidden_dim;
  g_w.rms_att = (const float *)p; p += L * dim * sizeof(float);
  g_w.rms_ffn = (const float *)p; p += L * dim * sizeof(float);
  g_w.rms_final = (const float *)p; p += dim * sizeof(float);

  /* quant_mode: 0 = all int8, 1 = layers int4 with the tied embedding/classifier left int8,
   * 2 = everything int4. The classifier is the most quantization-sensitive tensor in the model,
   * which is why mode 1 exists. */
  /* quant_mode: 0 = all int8; 1/2 = Q4_0 layers/all; 3/4 = Q4_1 layers/all. The "layers" modes
   * leave the tied embedding/classifier at int8 because it is the most quantization-sensitive
   * tensor in the model — measured, not assumed: Q4_0 on it doubled perplexity (44.5 -> 91.2). */
  const int packed_fmt = (g_cfg.quant_mode >= 3) ? QFMT_Q4_1
                       : (g_cfg.quant_mode >= 1) ? QFMT_Q4_0 : QFMT_I8;
  const int all = (g_cfg.quant_mode == 2 || g_cfg.quant_mode == 4);
  const int q4_emb = all ? packed_fmt : QFMT_I8;
  const int q4_lay = packed_fmt;
  QTensor *tok = map_tensors(&p, 1, (size_t)g_cfg.vocab_size * dim, q4_emb);
  g_w.tokens = tok[0];
  free(tok);
  g_w.wq = map_tensors(&p, g_cfg.n_layers, dim * dim, q4_lay);
  g_w.wk = map_tensors(&p, g_cfg.n_layers, (size_t)g_kv_dim * dim, q4_lay);
  g_w.wv = map_tensors(&p, g_cfg.n_layers, (size_t)g_kv_dim * dim, q4_lay);
  g_w.wo = map_tensors(&p, g_cfg.n_layers, dim * dim, q4_lay);
  g_w.w1 = map_tensors(&p, g_cfg.n_layers, hd * dim, q4_lay);
  g_w.w2 = map_tensors(&p, g_cfg.n_layers, dim * hd, q4_lay);
  g_w.w3 = map_tensors(&p, g_cfg.n_layers, hd * dim, q4_lay);

  /* The blob's own length is the cheapest end-to-end check that the ELF holds what we think it
   * holds: a truncated .incbin or a stale model shows up here instead of as gibberish output. */
  const size_t consumed = (size_t)(p - blob);
  if (consumed != blob_bytes) {
    printf("[smollm] MODEL SIZE MISMATCH: header describes %lu bytes, blob is %lu\r\n",
               (unsigned long)consumed, (unsigned long)blob_bytes);
    return -1;
  }
  return 0;
}

/* Scratch for a batch of P tokens. Allocated once at SMOLLM_MAX_BATCH (~600 KB at P=32), which is
 * nothing next to the 23 MB KV cache. */
static struct {
  int cap;
  float *x, *xb, *hb, *hb2;
  int8_t *xq_q, *hq_q;
  float *xq_s, *hq_s, *xq_g, *hq_g;
} g_b;

static void alloc_batch_state(int cap) {
  const size_t dim = (size_t)g_cfg.dim, hd = (size_t)g_cfg.hidden_dim, P = (size_t)cap;
  g_b.cap = cap;
  g_b.x = xcalloc(P * dim, sizeof(float), "batch x");
  g_b.xb = xcalloc(P * dim, sizeof(float), "batch xb");
  g_b.hb = xcalloc(P * hd, sizeof(float), "batch hb");
  g_b.hb2 = xcalloc(P * hd, sizeof(float), "batch hb2");
  g_b.xq_q = xcalloc(P * dim, 1, "batch xq");
  g_b.xq_s = xcalloc(P * (dim / (size_t)GS), sizeof(float), "batch xq scales");
  g_b.hq_q = xcalloc(P * hd, 1, "batch hq");
  g_b.hq_s = xcalloc(P * (hd / (size_t)GS), sizeof(float), "batch hq scales");
  g_b.xq_g = xcalloc(P * (dim / (size_t)GS), sizeof(float), "batch xq group sums");
  g_b.hq_g = xcalloc(P * (hd / (size_t)GS), sizeof(float), "batch hq group sums");
}

static void alloc_run_state(void) {
  const size_t dim = (size_t)g_cfg.dim, hd = (size_t)g_cfg.hidden_dim;
  const size_t kvc = (size_t)g_cfg.n_layers * (size_t)g_cfg.seq_len * (size_t)g_kv_dim;
  g_s.x = xcalloc(dim, sizeof(float), "x");
  g_s.xb = xcalloc(dim, sizeof(float), "xb");
  g_s.xb2 = xcalloc(dim, sizeof(float), "xb2");
  g_s.hb = xcalloc(hd, sizeof(float), "hb");
  g_s.hb2 = xcalloc(hd, sizeof(float), "hb2");
  g_s.xq.q = xcalloc(dim, 1, "xq.q");
  g_s.xq.s = xcalloc(dim / (size_t)GS, sizeof(float), "xq.s");
  g_s.xq.gsum = xcalloc(dim / (size_t)GS, sizeof(float), "xq.gsum");
  g_s.hq.q = xcalloc(hd, 1, "hq.q");
  g_s.hq.s = xcalloc(hd / (size_t)GS, sizeof(float), "hq.s");
  g_s.hq.gsum = xcalloc(hd / (size_t)GS, sizeof(float), "hq.gsum");
  g_s.q = xcalloc(dim, sizeof(float), "q");
  g_s.k = xcalloc((size_t)g_kv_dim, sizeof(float), "k");
  g_s.v = xcalloc((size_t)g_kv_dim, sizeof(float), "v");
  g_s.att = xcalloc((size_t)g_cfg.n_heads * (size_t)g_cfg.seq_len, sizeof(float), "att");
  g_s.logits = xcalloc((size_t)g_cfg.vocab_size, sizeof(float), "logits");
  g_s.key_cache = xcalloc(kvc, sizeof(float), "key cache");
  g_s.value_cache = xcalloc(kvc, sizeof(float), "value cache");
}

/* ------------------------------------------------------------------------------------------------
 * Kernels
 * ---------------------------------------------------------------------------------------------- */

/* Dot product of GS int8 values, accumulated in int32. Products are at most 127*127 and a group of
 * 64 sums to under 2^21, so the int16 widening multiply feeding a widening reduction cannot
 * overflow — and the result is bit-identical to the scalar loop, which keeps the host reference an
 * exact oracle rather than an approximate one. */
static inline int32_t dot_i8_scalar(const int8_t *a, const int8_t *b, int n) {
  int32_t sum = 0;
  for (int i = 0; i < n; i++) sum += (int32_t)a[i] * (int32_t)b[i];
  return sum;
}

#if SMOLLM_RVV
static inline int32_t dot_i8_rvv(const int8_t *a, const int8_t *b, int n) {
  vint32m1_t acc = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
  int i = 0;
  while (i < n) {
    size_t vl = __riscv_vsetvl_e8m1((size_t)(n - i));
    vint8m1_t va = __riscv_vle8_v_i8m1(a + i, vl);
    vint8m1_t vb = __riscv_vle8_v_i8m1(b + i, vl);
    acc = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(va, vb, vl), acc, vl);
    i += (int)vl;
  }
  return __riscv_vmv_x_s_i32m1_i32(acc);
}
#endif

/* W(d, n) @ x(n) -> xout(d), both operands quantized per group of GS.
 *
 * "ROW-DOT" — one dot product per output row, reduced per group. Simple, and the layout the blob
 * already has, but it pays a vector reduction AND a vector->scalar move for every group of 64
 * weights: 2.1M vector-pipeline drains per token. On a decoupled vector unit that is the dominant
 * cost, which is what smollm_bench() exists to measure. */
/* One group of GS int4 weights against GS activations. Byte j carries weight j (low nibble) and
 * weight j+GS/2 (high nibble), so the two halves of the activation vector pair with the two
 * nibble planes — no de-interleaving anywhere. */
static inline int32_t dot_q4_group_scalar(const int8_t *x, const uint8_t *b, int half) {
  int32_t s = 0;
  for (int j = 0; j < half; j++) {
    const int32_t lo = (int8_t)(b[j] << 4) >> 4;
    const int32_t hi = (int8_t)b[j] >> 4;
    s += lo * (int32_t)x[j] + hi * (int32_t)x[half + j];
  }
  return s;
}

/* Defined below, once all the per-format kernels exist: the single place packing is decided. */
static inline float dot_row_t(const int8_t *xq, const float *xs, const float *xg,
                              const QTensor *w, int row, int n);

/* Q4_1 group: nibbles are UNSIGNED 0..15 and dequantize affinely as q*d + m, so
 *     sum_j w_j x_j = d * sum_j(q_j x_j) + m * sum_j(x_j)
 * and the caller supplies sum_j(x_j) precomputed for the whole matmul. */
static inline int32_t dot_q41_group_scalar(const int8_t *x, const uint8_t *b, int half) {
  int32_t s = 0;
  for (int j = 0; j < half; j++) {
    const int32_t lo = b[j] & 0x0F;
    const int32_t hi = (b[j] >> 4) & 0x0F;
    s += lo * (int32_t)x[j] + hi * (int32_t)x[half + j];
  }
  return s;
}

static float dot_row_scalar_t(const int8_t *xq, const float *xs, const float *xg,
                              const QTensor *w, int row, int n) {
  const int ng = n / GS, half = GS / 2;
  const float *ws = w->s + (size_t)row * (size_t)ng;
  float val = 0.0f;
  if (w->fmt == QFMT_Q4_1) {
    const uint8_t *wq = (const uint8_t *)w->q + (size_t)row * ((size_t)n / 2u);
    const float *wm = w->m + (size_t)row * (size_t)ng;
    for (int g = 0; g < ng; g++) {
      const float acc = (float)dot_q41_group_scalar(xq + g * GS, wq + (size_t)g * half, half);
      val += (acc * ws[g] + wm[g] * xg[g]) * xs[g];
    }
  } else if (w->fmt == QFMT_Q4_0) {
    const uint8_t *wq = (const uint8_t *)w->q + (size_t)row * ((size_t)n / 2u);
    for (int g = 0; g < ng; g++)
      val += (float)dot_q4_group_scalar(xq + g * GS, wq + (size_t)g * half, half) * ws[g] * xs[g];
  } else {
    const int8_t *wq = w->q + (size_t)row * (size_t)n;
    for (int g = 0; g < ng; g++)
      val += (float)dot_i8_scalar(xq + g * GS, wq + g * GS, GS) * ws[g] * xs[g];
  }
  return val;
}

#define MATMUL_ROWDOT_BODY(DOT)                                            \
  const int ng = n / GS;                                                    \
  for (int i = 0; i < d; i++) {                                             \
    const int8_t *wq = w->q + (size_t)i * (size_t)n;                        \
    const float *ws = w->s + (size_t)i * (size_t)ng;                        \
    float val = 0.0f;                                                       \
    for (int g = 0; g < ng; g++)                                            \
      val += (float)DOT(x->q + g * GS, wq + g * GS, GS) * ws[g] * x->s[g];  \
    xout[i] = val;                                                          \
  }

static void matmul_rowdot_scalar(float *xout, const QBuf *x, const QTensor *w, int n, int d) {
  for (int i = 0; i < d; i++) xout[i] = dot_row_scalar_t(x->q, x->s, x->gsum, w, i, n);
}

#if SMOLLM_RVV
static void matmul_rowdot_rvv(float *xout, const QBuf *x, const QTensor *w, int n, int d) {
  MATMUL_ROWDOT_BODY(dot_i8_rvv)
}
#endif

#if SMOLLM_RVV
/* "LANE-FLOAT" — same weight layout, but the products stay IN LANES and are accumulated into a
 * float vector, so the only reduction is one per output row instead of one per group (9x fewer
 * here, and no vmv.x.s in the inner loop at all). Valid because
 *   sum_g scale_g * sum_lane p[g][lane]  ==  sum_lane sum_g scale_g * p[g][lane]
 * Requires GS to be a multiple of the vector length, which holds for GS=64 at VLEN 128/256/512. */
/* int4 rows. A group is only 32 packed bytes, so groups are processed in PAIRS: one e8m2 load
 * covers two groups and asks for exactly one 64-byte cache line, which is the only request size
 * this memory system rewards (see the LMUL sweep in smollm_bench). Sign-extending a nibble is
 * shift-left-4 then arithmetic-shift-right-4 for the low plane, and a bare arithmetic shift for
 * the high plane. */
static inline float dot_row_lane_q4(const int8_t *xq, const float *xs,
                                    const uint8_t *wq, const float *ws, int n) {
  const int ng = n / GS, half = GS / 2;
  const size_t vl = (size_t)half;                  /* 32 lanes: one group per half-register */
  const size_t vlmax = __riscv_vsetvlmax_e32m4();
  vfloat32m4_t facc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

  int g = 0;
  for (; g + 2 <= ng; g += 2) {
    const vint8m2_t pair = __riscv_vle8_v_i8m2((const int8_t *)(wq + (size_t)g * half), vl * 2);
    for (int u = 0; u < 2; u++) {
      const vint8m1_t b = (u == 0) ? __riscv_vget_v_i8m2_i8m1(pair, 0)
                                   : __riscv_vget_v_i8m2_i8m1(pair, 1);
      const vint8m1_t lo = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(b, 4, vl), 4, vl);
      const vint8m1_t hi = __riscv_vsra_vx_i8m1(b, 4, vl);
      const int8_t *x = xq + (size_t)(g + u) * GS;
      vint16m2_t p = __riscv_vwmul_vv_i16m2(lo, __riscv_vle8_v_i8m1(x, vl), vl);
      p = __riscv_vadd_vv_i16m2(p, __riscv_vwmul_vv_i16m2(hi, __riscv_vle8_v_i8m1(x + half, vl), vl), vl);
      facc = __riscv_vfmacc_vf_f32m4(facc, ws[g + u] * xs[g + u],
                                     __riscv_vfcvt_f_x_v_f32m4(__riscv_vwcvt_x_x_v_i32m4(p, vl), vl), vl);
    }
  }
  for (; g < ng; g++) {                            /* odd group count: one 32-byte load */
    const vint8m1_t b = __riscv_vle8_v_i8m1((const int8_t *)(wq + (size_t)g * half), vl);
    const vint8m1_t lo = __riscv_vsra_vx_i8m1(__riscv_vsll_vx_i8m1(b, 4, vl), 4, vl);
    const vint8m1_t hi = __riscv_vsra_vx_i8m1(b, 4, vl);
    const int8_t *x = xq + (size_t)g * GS;
    vint16m2_t p = __riscv_vwmul_vv_i16m2(lo, __riscv_vle8_v_i8m1(x, vl), vl);
    p = __riscv_vadd_vv_i16m2(p, __riscv_vwmul_vv_i16m2(hi, __riscv_vle8_v_i8m1(x + half, vl), vl), vl);
    facc = __riscv_vfmacc_vf_f32m4(facc, ws[g] * xs[g],
                                   __riscv_vfcvt_f_x_v_f32m4(__riscv_vwcvt_x_x_v_i32m4(p, vl), vl), vl);
  }
  vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(facc, z, vlmax));
}

/* Q4_1 rows. Identical memory behaviour to Q4_0 — two groups per 64-byte load — with unsigned
 * nibbles and one extra SCALAR term per group (m * sum(x)), where sum(x) came free from
 * quantize(). */
static inline float dot_row_lane_q41(const int8_t *xq, const float *xs, const float *xg,
                                     const uint8_t *wq, const float *ws, const float *wm, int n) {
  const int ng = n / GS, half = GS / 2;
  const size_t vl = (size_t)half;                  /* 32 lanes = one group */
  const size_t vlmax = __riscv_vsetvlmax_e32m4();
  vfloat32m4_t facc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
  float affine = 0.0f;

  /* Live vector state is deliberately small: one f32m4 accumulator (4 regs), one u8m2 weight pair
   * (2), and short-lived i8m1/i16m2/i32m4 temporaries (~7). That is ~13 of 32 registers, so this
   * does not spill — checked in the disassembly, not assumed. */
  int g = 0;
  for (; g + 2 <= ng; g += 2) {
    const vuint8m2_t pair = __riscv_vle8_v_u8m2(wq + (size_t)g * half, vl * 2);  /* one 64 B line */
    for (int u = 0; u < 2; u++) {
      const vuint8m1_t b = (u == 0) ? __riscv_vget_v_u8m2_u8m1(pair, 0)
                                    : __riscv_vget_v_u8m2_u8m1(pair, 1);
      const int8_t *x = xq + (size_t)(g + u) * GS;
      const vint8m1_t lo = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(b, 0x0F, vl));
      const vint8m1_t hi = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(b, 4, vl));
      vint16m2_t p = __riscv_vwmul_vv_i16m2(lo, __riscv_vle8_v_i8m1(x, vl), vl);
      p = __riscv_vadd_vv_i16m2(p, __riscv_vwmul_vv_i16m2(hi, __riscv_vle8_v_i8m1(x + half, vl), vl), vl);
      facc = __riscv_vfmacc_vf_f32m4(facc, ws[g + u] * xs[g + u],
                                     __riscv_vfcvt_f_x_v_f32m4(__riscv_vwcvt_x_x_v_i32m4(p, vl), vl), vl);
      affine += wm[g + u] * xg[g + u] * xs[g + u];
    }
  }
  for (; g < ng; g++) {                            /* odd group count: one 32-byte load */
    const vuint8m1_t b = __riscv_vle8_v_u8m1(wq + (size_t)g * half, vl);
    const int8_t *x = xq + (size_t)g * GS;
    const vint8m1_t lo = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(b, 0x0F, vl));
    const vint8m1_t hi = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(b, 4, vl));
    vint16m2_t p = __riscv_vwmul_vv_i16m2(lo, __riscv_vle8_v_i8m1(x, vl), vl);
    p = __riscv_vadd_vv_i16m2(p, __riscv_vwmul_vv_i16m2(hi, __riscv_vle8_v_i8m1(x + half, vl), vl), vl);
    facc = __riscv_vfmacc_vf_f32m4(facc, ws[g] * xs[g],
                                   __riscv_vfcvt_f_x_v_f32m4(__riscv_vwcvt_x_x_v_i32m4(p, vl), vl), vl);
    affine += wm[g] * xg[g] * xs[g];
  }
  vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(facc, z, vlmax)) + affine;
}

/* Decode reads every weight byte exactly once from DRAM, so the ONLY thing that matters in this
 * loop is the width of each memory request. Measured on bearly25: the memory system returns one
 * 64-byte line per request and cannot overlap requests, so
 *     32-byte loads (e8m1 @ VLEN=256)  49.45 cyc/byte
 *     64-byte loads (e8m2)             31.64 cyc/byte     <- 1.56x, and the floor
 *     128/256-byte loads               31.64 / 31.95      <- no further gain
 * Hence the wide path below: one e8m2 load per 64-weight group asks for exactly one cache line.
 * It accumulates at LMUL=8 (i16m4 -> i32m8 -> f32m8), which is only viable because a single token
 * needs one accumulator; the batched prefill kernel keeps LMUL=4 so four of them fit in registers.
 * Falls back to the narrow path when the group does not match the vector length. */
static inline float dot_row_lane(const int8_t *xq, const float *xs,
                                 const int8_t *wq, const float *ws, int n) {
  const int ng = n / GS;

  if (__riscv_vsetvlmax_e8m2() >= (size_t)GS) {
    const size_t vl = (size_t)GS;                     /* one whole group == one cache line */
    const size_t vlmax8 = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t facc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax8);
    for (int g = 0; g < ng; g++) {
      vint8m2_t vb = __riscv_vle8_v_i8m2(wq + g * GS, vl);   /* the only DRAM traffic here */
      vint8m2_t va = __riscv_vle8_v_i8m2(xq + g * GS, vl);
      vint32m8_t p32 = __riscv_vwcvt_x_x_v_i32m8(__riscv_vwmul_vv_i16m4(va, vb, vl), vl);
      facc = __riscv_vfmacc_vf_f32m8(facc, ws[g] * xs[g],
                                     __riscv_vfcvt_f_x_v_f32m8(p32, vl), vl);
    }
    vfloat32m1_t z8 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m8_f32m1(facc, z8, vl));
  }

  const size_t vlmax = __riscv_vsetvlmax_e32m4();
  vfloat32m4_t facc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
  for (int g = 0; g < ng; g++) {
    const float scale = ws[g] * xs[g];
    int k = 0;
    while (k < GS) {
      const size_t vl = __riscv_vsetvl_e8m1((size_t)(GS - k));
      vint8m1_t va = __riscv_vle8_v_i8m1(xq + g * GS + k, vl);
      vint8m1_t vb = __riscv_vle8_v_i8m1(wq + g * GS + k, vl);
      vint32m4_t p32 = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwmul_vv_i16m2(va, vb, vl), vl);
      facc = __riscv_vfmacc_vf_f32m4(facc, scale, __riscv_vfcvt_f_x_v_f32m4(p32, vl), vl);
      k += (int)vl;
    }
  }
  vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(facc, z, vlmax));
}

static void matmul_lanefloat(float *xout, const QBuf *x, const QTensor *w, int n, int d) {
  for (int i = 0; i < d; i++) xout[i] = dot_row_t(x->q, x->s, x->gsum, w, i, n);
}

/* "TRANSPOSED" — output rows live in the vector lanes, so there is NO reduction anywhere: each
 * input element broadcasts against a contiguous column of weights. Needs the weights (and their
 * scales) stored column-major, which the blob is not — smollm_bench() transposes one tensor into
 * the heap to measure what the layout would be worth before committing the exporter to it. */
static void matmul_transposed(float *xout, const QBuf *x, const int8_t *wT, const float *sT,
                              int n, int d) {
  const int ng = n / GS;
  for (int row = 0; row < d; ) {
    const size_t vl = __riscv_vsetvl_e32m4((size_t)(d - row));
    vfloat32m4_t facc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    for (int g = 0; g < ng; g++) {
      vint32m4_t iacc = __riscv_vmv_v_x_i32m4(0, vl);
      for (int k = 0; k < GS; k++) {
        const int16_t xv = (int16_t)x->q[g * GS + k];
        vint8m1_t wv = __riscv_vle8_v_i8m1(wT + (size_t)(g * GS + k) * (size_t)d + row, vl);
        iacc = __riscv_vwmacc_vx_i32m4(iacc, xv, __riscv_vwcvt_x_x_v_i16m2(wv, vl), vl);
      }
      vfloat32m4_t sc = __riscv_vle32_v_f32m4(sT + (size_t)g * (size_t)d + row, vl);
      facc = __riscv_vfmacc_vv_f32m4(facc, __riscv_vfmul_vf_f32m4(sc, x->s[g], vl),
                                     __riscv_vfcvt_f_x_v_f32m4(iacc, vl), vl);
    }
    __riscv_vse32_v_f32m4(xout + row, facc, vl);
    row += (int)vl;
  }
}
#endif /* SMOLLM_RVV */

/* The same matmul, but over P activation vectors at once. This is the whole point of batched
 * prefill: the weight row is loaded once and reused P times, so streaming 143 MB of weights costs
 * ONE pass for the entire prompt instead of one pass per token. Everything else about the
 * arithmetic is unchanged, so batched and sequential results are bit-identical. */
/* Tokens processed per pass over a weight row. The weight vector is loaded ONCE and multiplied
 * against four activations, so vector loads drop from 2 per (token, chunk) to 5 per 4 tokens.
 * Four f32m4 accumulators cost 16 of the 32 vector registers, which leaves room for the operands;
 * a wider tile spills. (RVV types are sizeless, so these have to be named variables — they cannot
 * live in an array.) */
#if SMOLLM_FWD_RVV
/* Two tokens per pass over a weight row: the weight vector is loaded once and multiplied against
 * both activations, so vector loads drop from 4 to 3 per 2 tokens.
 *
 * TWO, not four, because of register pressure: an f32m4 accumulator occupies one of only EIGHT
 * m4-aligned register groups, and the i32m4/f32m4 temporaries want two more. A 4-token tile
 * measured 4 whole-group spills (`vs4r.v`/`vl4re32.v`) in the disassembly; 2 measures zero. Prefill
 * is within ~1.5x of its memory floor anyway, so the extra loads a narrower tile costs are worth
 * far less than spilling 128 bytes per group to the stack. */
static void dot_rows2_lane(float *o0, float *o1,
                           const int8_t *x0, const float *s0,
                           const int8_t *x1, const float *s1,
                           const int8_t *wq, const float *ws, int n) {
  const int ng = n / GS;
  const size_t vlmax = __riscv_vsetvlmax_e32m4();
  vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
  vfloat32m4_t a1 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

  for (int g = 0; g < ng; g++) {
    const float w_s = ws[g];
    int k = 0;
    while (k < GS) {
      const size_t vl = __riscv_vsetvl_e8m1((size_t)(GS - k));
      const int off = g * GS + k;
      const vint8m1_t vw = __riscv_vle8_v_i8m1(wq + off, vl);            /* loaded once ... */
      vint32m4_t p;
      p = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwmul_vv_i16m2(vw, __riscv_vle8_v_i8m1(x0 + off, vl), vl), vl);
      a0 = __riscv_vfmacc_vf_f32m4(a0, w_s * s0[g], __riscv_vfcvt_f_x_v_f32m4(p, vl), vl);
      p = __riscv_vwcvt_x_x_v_i32m4(__riscv_vwmul_vv_i16m2(vw, __riscv_vle8_v_i8m1(x1 + off, vl), vl), vl);
      a1 = __riscv_vfmacc_vf_f32m4(a1, w_s * s1[g], __riscv_vfcvt_f_x_v_f32m4(p, vl), vl);
      k += (int)vl;
    }
  }
  const vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  *o0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(a0, z, vlmax));
  *o1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(a1, z, vlmax));
}
#endif

/* The single place packing is decided. Everything above is a kernel; everything below calls this. */
static inline float dot_row_t(const int8_t *xq, const float *xs, const float *xg,
                              const QTensor *w, int row, int n) {
#if SMOLLM_FWD_RVV
  const int ng = n / GS;
  const size_t off4 = (size_t)row * ((size_t)n / 2u);
  if (w->fmt == QFMT_Q4_1)
    return dot_row_lane_q41(xq, xs, xg, (const uint8_t *)w->q + off4,
                            w->s + (size_t)row * (size_t)ng, w->m + (size_t)row * (size_t)ng, n);
  if (w->fmt == QFMT_Q4_0)
    return dot_row_lane_q4(xq, xs, (const uint8_t *)w->q + off4,
                           w->s + (size_t)row * (size_t)ng, n);
  return dot_row_lane(xq, xs, w->q + (size_t)row * (size_t)n, w->s + (size_t)row * (size_t)ng, n);
#else
  return dot_row_scalar_t(xq, xs, xg, w, row, n);
#endif
}

/* Rows [row_lo, row_hi) of W(d, n) @ x(n) for P tokens at once. Splitting by output row is what
 * makes a second hart trivial to add: the ranges are disjoint and nothing is shared but read-only
 * weights. */
static void matmul_batch_rows(float *xout, const int8_t *xq, const float *xs, const float *xg,
                              int P, const QTensor *w, int n, int d, int row_lo, int row_hi) {
  const int ng = n / GS;
  for (int i = row_lo; i < row_hi; i++) {
    const int8_t *wq = w->q + (size_t)i * (size_t)n;
    const float *ws = w->s + (size_t)i * (size_t)ng;
    (void)wq; (void)ws;
#if SMOLLM_FWD_RVV
    int b = 0;
    /* The tile is int8-only; a packed tensor falls through to the per-token dispatcher, which
     * still issues full-cache-line weight loads. */
    if (w->fmt == QFMT_I8)
    for (; b + 2 <= P; b += 2) {
      dot_rows2_lane(&xout[(size_t)(b + 0) * d + i], &xout[(size_t)(b + 1) * d + i],
                     xq + (size_t)(b + 0) * n, xs + (size_t)(b + 0) * ng,
                     xq + (size_t)(b + 1) * n, xs + (size_t)(b + 1) * ng,
                     wq, ws, n);
    }
    for (; b < P; b++)
      xout[(size_t)b * d + i] = dot_row_t(xq + (size_t)b * n, xs + (size_t)b * ng,
                                          xg ? xg + (size_t)b * ng : NULL, w, i, n);
#else
    for (int b = 0; b < P; b++)
      xout[(size_t)b * (size_t)d + i] =
          dot_row_scalar_t(xq + (size_t)b * (size_t)n, xs + (size_t)b * (size_t)ng,
                           xg ? xg + (size_t)b * (size_t)ng : NULL, w, i, n);
#endif
  }
}

/* ---- optional second hart -----------------------------------------------------------------------
 * The output rows of a batched matmul are independent, so a second hart just takes the top half of
 * the range. model.c stays chip-free: the chip layer (main.c) installs a start/wait pair and calls
 * smollm_run_job() from its worker. Both harts only READ the weights, and they write DISJOINT rows
 * of the output, so the only thing that has to cross harts is "the other half is finished" — which
 * is exactly the property main.c probes at boot before enabling any of this. */
void (*smollm_cowork_start)(const smollm_job_t *job);
void (*smollm_cowork_wait)(void);

/* How many row-ranges hart 1 has actually completed. Without this, "the second hart bought us
 * nothing" and "the second hart never ran" look identical from the console — and they call for
 * opposite conclusions. */
volatile unsigned long g_smollm_jobs_done;

void smollm_run_job(const smollm_job_t *j) {
  g_smollm_jobs_done++;
  matmul_batch_rows((float *)j->xout, (const int8_t *)j->xq, (const float *)j->xs,
                    (const float *)j->xg, j->P, (const QTensor *)j->w, j->n, j->d,
                    j->row_lo, j->row_hi);
}

static void matmul_batch(float *xout, const int8_t *xq, const float *xs, const float *xg, int P,
                         const QTensor *w, int n, int d) {
  /* Only worth splitting when there is enough work to cover the handoff. */
  if (smollm_cowork_start && smollm_cowork_wait && d >= 64) {
    const int mid = (d / 2) & ~3;      /* keep both halves a multiple of 4 rows */
    smollm_job_t j = { xout, xq, xs, xg, w, P, n, d, mid, d };
    smollm_cowork_start(&j);
    matmul_batch_rows(xout, xq, xs, xg, P, w, n, d, 0, mid);
    smollm_cowork_wait();
    return;
  }
  matmul_batch_rows(xout, xq, xs, xg, P, w, n, d, 0, d);
}

/* Which kernel the forward pass uses; smollm_bench() measures the alternatives on real weights. */
static void matmul(float *xout, const QBuf *x, const QTensor *w, int n, int d) {
  /* Decode is one token, but its output rows are just as independent as prefill's, so the same
   * row split applies — P=1 through the batched path. Whether it BUYS anything is a different
   * question: prefill saw ~0% from the second hart, which is what you would expect if the memory
   * system is a single serialized resource rather than something two harts can each pull from.
   * Gated so it can be measured rather than assumed. */
#if SMOLLM_DUALCORE_DECODE
  if (smollm_cowork_start && smollm_cowork_wait && d >= 128) {
    matmul_batch(xout, x->q, x->s, x->gsum, 1, w, n, d);
    return;
  }
#endif
#if SMOLLM_FWD_RVV && SMOLLM_MATMUL_LANEFLOAT
  matmul_lanefloat(xout, x, w, n, d);
#elif SMOLLM_FWD_RVV
  matmul_rowdot_rvv(xout, x, w, n, d);
#else
  matmul_rowdot_scalar(xout, x, w, n, d);
#endif
}

static void quantize_to_g(int8_t *q, float *sc, float *gsum, const float *x, int n) {
  for (int g = 0; g < n / GS; g++) {
    float wmax = 0.0f;
    for (int i = 0; i < GS; i++) {
      float v = fabsf(x[g * GS + i]);
      if (v > wmax) wmax = v;
    }
    const float scale = wmax / 127.0f;
    sc[g] = scale;
    /* DIVIDE, do not multiply by a precomputed reciprocal. x*(1/s) differs from x/s by an ulp,
     * which is enough to flip a value sitting near a .5 rounding boundary to the next int8 — and a
     * single activation off by one grows into a visibly different answer a dozen layers later
     * (measured: identical to the host reference through layer 12, then diverging to 5% by layer
     * 20 and a different token at the end). llama2.c divides here too. */
    for (int i = 0; i < GS; i++)
      q[g * GS + i] = (scale == 0.0f) ? 0 : (int8_t)roundf(x[g * GS + i] / scale);
    if (gsum) {
      int32_t sum = 0;
      for (int i = 0; i < GS; i++) sum += q[g * GS + i];
      gsum[g] = (float)sum;
    }
  }
}

static void quantize(QBuf *qx, const float *x, int n) {
  quantize_to_g(qx->q, qx->s, qx->gsum, x, n);
}

static void rmsnorm(float *o, const float *x, const float *weight, int size) {
  float ss = 0.0f;
  for (int j = 0; j < size; j++) ss += x[j] * x[j];
  ss = 1.0f / sqrtf(ss / (float)size + g_cfg.norm_eps);
  for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

void smollm_softmax(float *x, int size) {
  float m = x[0];
  for (int i = 1; i < size; i++) if (x[i] > m) m = x[i];
  float sum = 0.0f;
  for (int i = 0; i < size; i++) { x[i] = expf(x[i] - m); sum += x[i]; }
  for (int i = 0; i < size; i++) x[i] /= sum;
}

/* One row of the embedding table, dequantized on demand (the whole table would be 113 MB). */
static void embed_token(float *x, int token) {
  const size_t dim = (size_t)g_cfg.dim, half = (size_t)GS / 2u;
  const float *s = g_w.tokens.s + (size_t)token * (dim / (size_t)GS);
  if (g_w.tokens.fmt == QFMT_I8) {
    const int8_t *q = g_w.tokens.q + (size_t)token * dim;
    for (size_t i = 0; i < dim; i++) x[i] = (float)q[i] * s[i / (size_t)GS];
    return;
  }
  const uint8_t *b = (const uint8_t *)g_w.tokens.q + (size_t)token * (dim / 2u);
  const float *m = (g_w.tokens.fmt == QFMT_Q4_1)
                 ? g_w.tokens.m + (size_t)token * (dim / (size_t)GS) : NULL;
  for (size_t g = 0; g < dim / (size_t)GS; g++) {
    const uint8_t *bg = b + g * half;
    for (size_t j = 0; j < half; j++) {
      int lo, hi;
      if (m) { lo = bg[j] & 0x0F; hi = (bg[j] >> 4) & 0x0F; }
      else   { lo = (int8_t)(bg[j] << 4) >> 4; hi = (int8_t)bg[j] >> 4; }
      x[g * GS + j]        = (float)lo * s[g] + (m ? m[g] : 0.0f);
      x[g * GS + half + j] = (float)hi * s[g] + (m ? m[g] : 0.0f);
    }
  }
}

static int g_stage_idx;
static int g_stage_pos;   /* which position the stage hook reports on */
static int g_trace_layer = -1;   /* report every op inside this layer (-1 = off) */

void smollm_set_trace_layer(int layer) { g_trace_layer = layer; }

void smollm_set_stage_pos(int pos) { g_stage_pos = pos; }

static void stage_report(const float *x, int n, const char *name) {
  if (!g_stage_hook) return;
  float sum = 0.0f;
  for (int i = 0; i < n; i++) sum += x[i];
  g_stage_hook(g_stage_idx++, name, sum);
}

/* RoPE frequencies: one per dimension PAIR within a head, so head_dim/2 of them. The angle for a
 * token is pos*freq and is shared by every layer and every head, so the whole forward pass needs
 * head_dim/2 cos/sin evaluations instead of n_layers*dim/2 of them (30x fewer here). */
static float *g_rope_freq, *g_rope_cos, *g_rope_sin;

static void rope_init(void) {
  const int half = g_cfg.head_dim / 2;
  g_rope_freq = xcalloc((size_t)half, sizeof(float), "rope freq");
  g_rope_cos = xcalloc((size_t)half, sizeof(float), "rope cos");
  g_rope_sin = xcalloc((size_t)half, sizeof(float), "rope sin");
  for (int j = 0; j < half; j++)
    g_rope_freq[j] = 1.0f / powf(g_cfg.rope_theta, (float)(2 * j) / (float)g_cfg.head_dim);
}

float *smollm_forward(int token, int pos) {
  const smollm_config_t *p = &g_cfg;
  RunState *s = &g_s;
  const int dim = p->dim, hs = p->head_dim, kv_dim = g_kv_dim;
  const int kv_mul = p->n_heads / p->n_kv_heads;
  float *x = s->x;

  for (int j = 0; j < hs / 2; j++) {
    const float a = (float)pos * g_rope_freq[j];
    g_rope_cos[j] = cosf(a);
    g_rope_sin[j] = sinf(a);
  }

  embed_token(x, token);
  if (pos == g_stage_pos) { g_stage_idx = 0; stage_report(x, dim, "embed"); }

  for (int l = 0; l < p->n_layers; l++) {
    const int tr = (pos == g_stage_pos && l == g_trace_layer);
    rmsnorm(s->xb, x, g_w.rms_att + (size_t)l * dim, dim);
    if (tr) stage_report(s->xb, dim, "att_norm");
    quantize(&s->xq, s->xb, dim);
    matmul(s->q, &s->xq, &g_w.wq[l], dim, dim);
    matmul(s->k, &s->xq, &g_w.wk[l], dim, kv_dim);
    matmul(s->v, &s->xq, &g_w.wv[l], dim, kv_dim);
    if (tr) { stage_report(s->q, dim, "q"); stage_report(s->k, kv_dim, "k"); stage_report(s->v, kv_dim, "v"); }

    /* RoPE on adjacent pairs; the exporter de-interleaved q/k out of HF's split-half layout.
     * cos/sin depend only on (pos, i % head_dim), so they were computed once for this token
     * before the layer loop — recomputing them here would be 30x the transcendentals. */
    for (int i = 0; i < dim; i += 2) {
      const float fcr = g_rope_cos[(i % hs) >> 1], fci = g_rope_sin[(i % hs) >> 1];
      const int rotn = (i < kv_dim) ? 2 : 1;   /* 2 = rotate q and k, 1 = q only */
      for (int v = 0; v < rotn; v++) {
        float *vec = (v == 0) ? s->q : s->k;
        const float v0 = vec[i], v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }

    const size_t loff = (size_t)l * (size_t)p->seq_len * (size_t)kv_dim;
    memcpy(s->key_cache + loff + (size_t)pos * kv_dim, s->k, (size_t)kv_dim * sizeof(float));
    memcpy(s->value_cache + loff + (size_t)pos * kv_dim, s->v, (size_t)kv_dim * sizeof(float));

    for (int h = 0; h < p->n_heads; h++) {
      const float *q = s->q + h * hs;
      float *att = s->att + h * p->seq_len;
      const int kvh = h / kv_mul;   /* grouped-query attention: 3 query heads share one KV head */
      for (int t = 0; t <= pos; t++) {
        const float *k = s->key_cache + loff + (size_t)t * kv_dim + kvh * hs;
        float score = 0.0f;
        for (int i = 0; i < hs; i++) score += q[i] * k[i];
        att[t] = score / sqrtf((float)hs);
      }
      smollm_softmax(att, pos + 1);
      float *xb = s->xb + h * hs;
      memset(xb, 0, (size_t)hs * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        const float *v = s->value_cache + loff + (size_t)t * kv_dim + kvh * hs;
        const float a = att[t];
        for (int i = 0; i < hs; i++) xb[i] += a * v[i];
      }
    }

    if (tr) { stage_report(s->q, dim, "q_rope"); stage_report(s->xb, dim, "attn_out");
              stage_report(s->att, pos + 1, "att0"); }
    quantize(&s->xq, s->xb, dim);
    if (tr) { float t = 0.0f; for (int i = 0; i < dim; i++) t += (float)s->xq.q[i];
              stage_report(&t, 1, "xq_wo"); }
    matmul(s->xb2, &s->xq, &g_w.wo[l], dim, dim);
    if (tr) stage_report(s->xb2, dim, "wo");
    for (int i = 0; i < dim; i++) x[i] += s->xb2[i];

    rmsnorm(s->xb, x, g_w.rms_ffn + (size_t)l * dim, dim);
    quantize(&s->xq, s->xb, dim);
    matmul(s->hb, &s->xq, &g_w.w1[l], dim, p->hidden_dim);
    matmul(s->hb2, &s->xq, &g_w.w3[l], dim, p->hidden_dim);
    for (int i = 0; i < p->hidden_dim; i++) {     /* SwiGLU: w2(silu(w1(x)) * w3(x)) */
      float v = s->hb[i];
      v *= 1.0f / (1.0f + expf(-v));
      s->hb[i] = v * s->hb2[i];
    }
    if (tr) { stage_report(s->hb, p->hidden_dim, "swiglu"); }
    quantize(&s->hq, s->hb, p->hidden_dim);
    matmul(s->xb, &s->hq, &g_w.w2[l], p->hidden_dim, dim);
    if (tr) stage_report(s->xb, dim, "w2");
    for (int i = 0; i < dim; i++) x[i] += s->xb[i];

    if (pos == g_stage_pos) stage_report(x, dim, "layer");
  }

  rmsnorm(x, x, g_w.rms_final, dim);
  quantize(&s->xq, x, dim);
  matmul(s->logits, &s->xq, &g_w.tokens, dim, p->vocab_size);   /* classifier is tied to the embedding */
  if (pos == g_stage_pos) stage_report(s->logits, p->vocab_size, "logits");
  return s->logits;
}


/* Allocate everything the forward pass writes: RunState, KV cache, RoPE tables. */
int smollm_model_alloc(void) {
  g_alloc_failed = 0;
  alloc_run_state();
  alloc_batch_state(SMOLLM_MAX_BATCH);
  rope_init();
  return g_alloc_failed ? -1 : 0;
}

const smollm_config_t *smollm_model_config(void) { return &g_cfg; }

/* ------------------------------------------------------------------------------------------------
 * Kernel benchmark
 *
 * One flash of this answers "is it the memory system or the kernel?", which is otherwise a guess
 * that costs 26 minutes per attempt. It runs every variant over a REAL weight tensor (w1 of layer
 * 0, ~865 KB) and reports cycles per weight byte, plus two memory floors that do no arithmetic at
 * all — if the floors are as slow as the kernels, no kernel change can help and the answer is a
 * smaller model (int4) instead. It also cross-checks the variants against each other, so a fast
 * kernel that is wrong cannot be mistaken for a win.
 * ---------------------------------------------------------------------------------------------- */

uint64_t smollm_fnv1a64(const uint8_t *p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; i++) { h ^= (uint64_t)p[i]; h *= 1099511628211ull; }
  return h;
}

static float bench_maxdiff(const float *a, const float *b, int n) {
  float m = 0.0f;
  for (int i = 0; i < n; i++) {
    const float d = fabsf(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

void smollm_bench(uint64_t (*now)(void)) {
  const int n = g_cfg.dim, d = g_cfg.hidden_dim;      /* w1: (hidden_dim, dim) */
  const QTensor *w = &g_w.w1[0];
  const int ng = n / GS;
  const size_t bytes = (size_t)n * (size_t)d;
  float *out_a = xcalloc((size_t)d, sizeof(float), "bench out");
  float *out_b = xcalloc((size_t)d, sizeof(float), "bench out");
  if (g_alloc_failed) return;

  /* a plausible activation: quantize the first embedding row */
  float *xf = xcalloc((size_t)n, sizeof(float), "bench x");
  QBuf xq = { xcalloc((size_t)n, 1, "bench xq"), xcalloc((size_t)ng, sizeof(float), "bench xs"),
              xcalloc((size_t)ng, sizeof(float), "bench xg") };
  if (g_alloc_failed) return;
  embed_token(xf, 1);
  quantize(&xq, xf, n);

#if SMOLLM_RVV
  printf("[bench] rvv vlmax e32m1=%lu e32m4=%lu e8m1=%lu -> VLEN=%lu bits\r\n",
         (unsigned long)__riscv_vsetvlmax_e32m1(), (unsigned long)__riscv_vsetvlmax_e32m4(),
         (unsigned long)__riscv_vsetvlmax_e8m1(),
         (unsigned long)__riscv_vsetvlmax_e32m1() * 32ul);
#else
  printf("[bench] rvv disabled (scalar build)\r\n");
#endif
  printf("[bench] tensor w1[0] %d x %d = %lu KB, GS=%d\r\n",
         d, n, (unsigned long)(bytes >> 10), GS);

  uint64_t t;
  /* Floor 1: touch every weight byte, scalar, no arithmetic beyond an add. */
  t = now();
  int32_t sink = 0;
  for (size_t i = 0; i < bytes; i++) sink += w->q[i];
  const uint64_t c_scalar_read = now() - t;

  /* Floor 2: same, vector loads, no reduction. */
#if SMOLLM_RVV
  uint64_t c_vec_read = 0;
  t = now();
  {
    const size_t vlmax = __riscv_vsetvlmax_e8m1();
    vint8m1_t acc = __riscv_vmv_v_x_i8m1(0, vlmax);
    for (size_t i = 0; i + vlmax <= bytes; i += vlmax)
      acc = __riscv_vadd_vv_i8m1(acc, __riscv_vle8_v_i8m1(w->q + i, vlmax), vlmax);
    sink += __riscv_vmv_x_s_i8m1_i8(acc);
  }
  c_vec_read = now() - t;
#endif

  t = now();
  matmul_rowdot_scalar(out_a, &xq, w, n, d);   /* the reference every variant is checked against */
  const uint64_t c_scalar_mm = now() - t;

#if SMOLLM_RVV
  uint64_t c_rowdot = 0, c_lane = 0, c_trans = 0;
  float diff_rowdot = 0.0f, diff_lane = 0.0f, diff_trans = 0.0f;
#endif
#if SMOLLM_RVV
  t = now();
  matmul_rowdot_rvv(out_b, &xq, w, n, d);
  c_rowdot = now() - t;
  diff_rowdot = bench_maxdiff(out_a, out_b, d);

  t = now();
  matmul_lanefloat(out_b, &xq, w, n, d);
  c_lane = now() - t;
  diff_lane = bench_maxdiff(out_a, out_b, d);

  /* Transpose into the heap so the column-major kernel can be measured without changing the blob
   * format. The transpose itself is NOT timed — it would be done once at export.
   * int8 ONLY: it reads w->q as one byte per weight, so on a Q4_0/Q4_1 blob both the transpose and
   * the kernel are meaningless and max_diff is garbage. Skipped rather than printed as a scary
   * number that looks like a hardware fault. */
  int8_t *wT = xcalloc(bytes, 1, "bench wT");
  float *sT = xcalloc((size_t)ng * (size_t)d, sizeof(float), "bench sT");
  if (!g_alloc_failed && w->fmt == QFMT_I8) {
    for (int i = 0; i < d; i++) {
      for (int k = 0; k < n; k++) wT[(size_t)k * (size_t)d + i] = w->q[(size_t)i * (size_t)n + k];
      for (int g = 0; g < ng; g++) sT[(size_t)g * (size_t)d + i] = w->s[(size_t)i * (size_t)ng + g];
    }
    t = now();
    matmul_transposed(out_b, &xq, wT, sT, n, d);
    c_trans = now() - t;
    diff_trans = bench_maxdiff(out_a, out_b, d);
  }
  free(wT);
  free(sT);
#endif

  /* Is the model region ITSELF slow, or was the transposed kernel just hitting cache? Read a heap
   * buffer far larger than any cache, and read the blob with several independent streams — the
   * second one answers whether this core can keep more than one miss in flight, which decides
   * whether decode can be made faster at all or only smaller. */
  uint64_t c_heap_read = 0, c_stream[4] = {0, 0, 0, 0};
  const size_t heap_bytes = 8u << 20;
  int8_t *heap_buf = xcalloc(heap_bytes, 1, "bench heap");
  if (!g_alloc_failed) {
    for (size_t i = 0; i < heap_bytes; i++) heap_buf[i] = (int8_t)(i & 0x7f);
    t = now();
    for (size_t i = 0; i < heap_bytes; i++) sink += heap_buf[i];
    c_heap_read = now() - t;
  }
  free(heap_buf);

  /* N interleaved read streams over the model blob, N = 1, 2, 4, 8. If cyc/byte falls as N rises,
   * the core is LATENCY-bound and more streams (or prefetch) is the fix; if it stays flat, the
   * bus is saturated and only reading fewer bytes helps. */
  {
    const size_t span = 8u << 20;                       /* well past any cache */
    const uint8_t *base = (const uint8_t *)g_w.tokens.q;
    static const int nstream[4] = { 1, 2, 4, 8 };
    for (int si = 0; si < 4; si++) {
      const int ns = nstream[si];
      const size_t chunk = span / (size_t)ns;
      t = now();
      for (size_t i = 0; i < chunk; i++)
        for (int k = 0; k < ns; k++) sink += (int8_t)base[(size_t)k * chunk + i];
      c_stream[si] = now() - t;
    }
  }

  /* Does a READ populate the cache at all? Read a small buffer (well inside any cache) TWICE. If
   * the second pass is no faster than the first, read misses are not allocating — which would
   * explain everything: DRAM reads at 55 cyc/byte, freshly WRITTEN data at 1.5, and concurrency
   * making things worse rather than better. That is a cache/uncore configuration question, not a
   * software one, and it is worth up to 36x. */
  uint64_t c_reread[4] = {0, 0, 0, 0};
  {
    const size_t small = 64u << 10;              /* 64 KB: inside any plausible L2 */
    const uint8_t *b = (const uint8_t *)g_w.tokens.q;
    for (int pass = 0; pass < 2; pass++) {
      t = now();
      for (size_t i = 0; i < small; i++) sink += (int8_t)b[i];
      c_reread[pass] = now() - t;
    }
    int8_t *wbuf = xcalloc(small, 1, "bench reread");   /* calloc => written => write-allocated */
    if (!g_alloc_failed) {
      for (int pass = 0; pass < 2; pass++) {
        t = now();
        for (size_t i = 0; i < small; i++) sink += wbuf[i];
        c_reread[2 + pass] = now() - t;
      }
    }
    free(wbuf);
  }

  /* Are memory requests pipelined at all? Same sequential read, but one vector load of 32 / 64 /
   * 128 / 256 bytes at a time. If a bigger LMUL is much faster, the memory system CAN overlap and
   * the matmul should be restructured to issue wider loads; if it is flat, every request is
   * serialized and only reading fewer bytes helps. */
#if SMOLLM_RVV
  uint64_t c_lmul[4] = {0, 0, 0, 0};
  {
    const size_t span = 2u << 20;
    const int8_t *b = (const int8_t *)g_w.tokens.q;
    size_t vl;
    t = now(); vl = __riscv_vsetvlmax_e8m1();
    for (size_t i = 0; i + vl <= span; i += vl) sink += __riscv_vmv_x_s_i8m1_i8(__riscv_vle8_v_i8m1(b + i, vl));
    c_lmul[0] = now() - t;
    t = now(); vl = __riscv_vsetvlmax_e8m2();
    for (size_t i = 0; i + vl <= span; i += vl) sink += __riscv_vmv_x_s_i8m2_i8(__riscv_vle8_v_i8m2(b + i, vl));
    c_lmul[1] = now() - t;
    t = now(); vl = __riscv_vsetvlmax_e8m4();
    for (size_t i = 0; i + vl <= span; i += vl) sink += __riscv_vmv_x_s_i8m4_i8(__riscv_vle8_v_i8m4(b + i, vl));
    c_lmul[2] = now() - t;
    t = now(); vl = __riscv_vsetvlmax_e8m8();
    for (size_t i = 0; i + vl <= span; i += vl) sink += __riscv_vmv_x_s_i8m8_i8(__riscv_vle8_v_i8m8(b + i, vl));
    c_lmul[3] = now() - t;
  }
#endif

  /* ---- int8 vs Q4_1 decode kernel, measured head to head in ONE boot ----------------------
   * Requantize this tensor to Q4_1 on-chip so both kernels run on the SAME weights, then evict
   * the cache before each timing run. Both matter: comparing against a heap-resident copy is what
   * made `matmul transp` look 36x faster than it is, and a warm cache flatters whichever kernel
   * runs second. The comparable metric is cycles per WEIGHT (the byte counts differ by design). */
  uint64_t c_i8_dec = 0, c_q41_dec = 0;
  {
    const size_t nweights = (size_t)n * (size_t)d;
    uint8_t *q4 = (uint8_t *)xcalloc(nweights / 2u, 1, "bench q4");
    float *d4 = (float *)xcalloc((size_t)ng * (size_t)d, sizeof(float), "bench d4");
    float *m4 = (float *)xcalloc((size_t)ng * (size_t)d, sizeof(float), "bench m4");
    int8_t *evict = (int8_t *)xcalloc(512u << 10, 1, "bench evict");
    if (!g_alloc_failed) {
      const int half = GS / 2;
      for (int i = 0; i < d; i++) {                       /* int8 -> Q4_1, group by group */
        for (int g = 0; g < ng; g++) {
          const int8_t *src = w->q + (size_t)i * n + g * GS;
          const float sc = w->s[(size_t)i * ng + g];
          float lo = src[0] * sc, hi = lo;
          for (int j = 1; j < GS; j++) {
            const float v = src[j] * sc;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
          }
          const float dd = (hi - lo) / 15.0f;
          const float inv = (dd == 0.0f) ? 0.0f : 1.0f / dd;
          d4[(size_t)i * ng + g] = dd;
          m4[(size_t)i * ng + g] = lo;
          uint8_t *dst = q4 + (size_t)i * ((size_t)n / 2u) + (size_t)g * half;
          for (int j = 0; j < half; j++) {
            int a = (int)((src[j] * sc - lo) * inv + 0.5f);
            int b2 = (int)((src[j + half] * sc - lo) * inv + 0.5f);
            if (a < 0) { a = 0; } else if (a > 15) { a = 15; }
            if (b2 < 0) { b2 = 0; } else if (b2 > 15) { b2 = 15; }
            dst[j] = (uint8_t)(a | (b2 << 4));
          }
        }
      }
      #define EVICT() do { for (size_t e = 0; e < (512u << 10); e += 64) evict[e] ^= 1; } while (0)
      EVICT();
      t = now();
      for (int i = 0; i < d; i++)
        out_b[i] = dot_row_lane(xq.q, xq.s, w->q + (size_t)i * n, w->s + (size_t)i * ng, n);
      c_i8_dec = now() - t;
      EVICT();
      t = now();
      for (int i = 0; i < d; i++)
        out_b[i] = dot_row_lane_q41(xq.q, xq.s, xq.gsum, q4 + (size_t)i * ((size_t)n / 2u),
                                    d4 + (size_t)i * ng, m4 + (size_t)i * ng, n);
      c_q41_dec = now() - t;
      #undef EVICT
      sink += (int32_t)out_b[0];
    }
    free(q4); free(d4); free(m4); free(evict);
  }

  /* cycles per weight byte, in hundredths, without pulling in float formatting */
  #define CPB(c) (unsigned long)(((c) * 100ull) / (uint64_t)bytes)
  printf("[bench] scalar read   %8lu Kcyc  %3lu.%02lu cyc/byte  (memory floor)\r\n",
         (unsigned long)(c_scalar_read / 1000ull), CPB(c_scalar_read) / 100, CPB(c_scalar_read) % 100);
#if SMOLLM_RVV
  printf("[bench] vector read   %8lu Kcyc  %3lu.%02lu cyc/byte  (memory floor)\r\n",
         (unsigned long)(c_vec_read / 1000ull), CPB(c_vec_read) / 100, CPB(c_vec_read) % 100);
#endif
  #define CPB_N(c, n) (unsigned long)(((c) * 100ull) / (uint64_t)(n))
  {
    static const char *const what[4] = { "blob  1st", "blob  2nd", "heap  1st", "heap  2nd" };
    for (int i = 0; i < 4; i++) {
      const unsigned long v = CPB_N(c_reread[i], (size_t)(64u << 10));
      printf("[bench] reread %s  %8lu Kcyc  %3lu.%02lu cyc/byte%s\r\n", what[i],
             (unsigned long)(c_reread[i] / 1000ull), v / 100, v % 100,
             (i == 1 || i == 3) ? "   <-- 2nd pass: cached?" : "");
    }
  }
#if SMOLLM_RVV
  for (int i = 0; i < 4; i++) {
    const unsigned long v = CPB_N(c_lmul[i], (size_t)(2u << 20));
    printf("[bench] vread LMUL=%d    %8lu Kcyc  %3lu.%02lu cyc/byte\r\n", 1 << i,
           (unsigned long)(c_lmul[i] / 1000ull), v / 100, v % 100);
  }
#endif
  if (c_heap_read) {
    const unsigned long h = CPB_N(c_heap_read, heap_bytes);
    printf("[bench] heap read 8MB %8lu Kcyc  %3lu.%02lu cyc/byte  (same DRAM, not .rodata)\r\n",
           (unsigned long)(c_heap_read / 1000ull), h / 100, h % 100);
  }
  for (int si = 0; si < 4; si++) {
    const unsigned long v = CPB_N(c_stream[si], (size_t)(8u << 20));
    printf("[bench] blob %d-stream   %8lu Kcyc  %3lu.%02lu cyc/byte\r\n",
           (int[]){1,2,4,8}[si], (unsigned long)(c_stream[si] / 1000ull), v / 100, v % 100);
  }
  #undef CPB_N
  printf("[bench] matmul scalar %8lu Kcyc  %3lu.%02lu cyc/byte  (reference)\r\n",
         (unsigned long)(c_scalar_mm / 1000ull), CPB(c_scalar_mm) / 100, CPB(c_scalar_mm) % 100);
#if SMOLLM_RVV
  printf("[bench] matmul rowdot %8lu Kcyc  %3lu.%02lu cyc/byte  max_diff %d/1e6\r\n",
         (unsigned long)(c_rowdot / 1000ull), CPB(c_rowdot) / 100, CPB(c_rowdot) % 100,
         (int)(diff_rowdot * 1e6f));
  printf("[bench] matmul lane   %8lu Kcyc  %3lu.%02lu cyc/byte  max_diff %d/1e6\r\n",
         (unsigned long)(c_lane / 1000ull), CPB(c_lane) / 100, CPB(c_lane) % 100,
         (int)(diff_lane * 1e6f));
  {
    /* cycles per weight, in thousandths, and the whole-model per-token cost each implies */
    const size_t nw = (size_t)n * (size_t)d;
    const unsigned long a = (unsigned long)((c_i8_dec * 1000ull) / nw);
    const unsigned long b2 = (unsigned long)((c_q41_dec * 1000ull) / nw);
    printf("[bench] DECODE int8  %8lu Kcyc  %lu.%03lu cyc/weight -> %lu Mcyc/token\r\n",
           (unsigned long)(c_i8_dec / 1000ull), a / 1000, a % 1000,
           (unsigned long)((c_i8_dec / nw) * 134479872ull / 1000000ull));
    printf("[bench] DECODE q4_1  %8lu Kcyc  %lu.%03lu cyc/weight -> %lu Mcyc/token  (%lu.%02lux)\r\n",
           (unsigned long)(c_q41_dec / 1000ull), b2 / 1000, b2 % 1000,
           (unsigned long)((c_q41_dec / nw) * 134479872ull / 1000000ull),
           c_q41_dec ? (unsigned long)((c_i8_dec * 100ull / c_q41_dec) / 100) : 0ul,
           c_q41_dec ? (unsigned long)((c_i8_dec * 100ull / c_q41_dec) % 100) : 0ul);
  }
  if (c_trans)
    printf("[bench] matmul transp %8lu Kcyc  %3lu.%02lu cyc/byte  max_diff %d/1e6\r\n",
           (unsigned long)(c_trans / 1000ull), CPB(c_trans) / 100, CPB(c_trans) % 100,
           (int)(diff_trans * 1e6f));
  else
    printf("[bench] matmul transp  skipped (int8-only kernel, blob is packed)\r\n");
#endif
  /* Scaled to the whole 143 MB model, i.e. what one generated token costs. Done in cyc/byte to
   * avoid the integer-division trap that made this print 1.8e16 the first time: bytes/1000000 is
   * ZERO for an 864 KB tensor. */
  #define PERTOK(c) (unsigned long)((((c) * 100ull) / (uint64_t)bytes) * 143000000ull / 100000000ull)
  printf("[bench] per-token estimate: scalar %lu Mcyc", PERTOK(c_scalar_mm));
#if SMOLLM_RVV
  printf(", rowdot %lu Mcyc, lane %lu Mcyc, transposed %lu Mcyc",
         PERTOK(c_rowdot), PERTOK(c_lane), PERTOK(c_trans));
#endif
  #undef PERTOK
  printf("  (sink %ld)\r\n", (long)sink);
  #undef CPB

  free(out_a); free(out_b); free(xf); free(xq.q); free(xq.s);
}

/* ------------------------------------------------------------------------------------------------
 * Batched prefill
 *
 * Reading the prompt is not a decode problem: token p's activations do not depend on token p+1's,
 * so the whole prompt can go through one pass over the weights instead of one pass per token. On
 * this board that is the difference between minutes and seconds, because the forward pass is
 * entirely weight-bandwidth bound (measured: 55 cyc/byte to read the model, and a matmul over it
 * costs 61 — the arithmetic is nearly free).
 *
 * Only the LAST token's logits are ever used, so the classifier — 28 MB, a fifth of the model — is
 * skipped for every other token in the batch.
 *
 * Per-token arithmetic and its order are unchanged, so this is bit-identical to calling
 * smollm_forward() in a loop. scripts/check_c_forward.py --batch verifies exactly that.
 * ---------------------------------------------------------------------------------------------- */
float *smollm_forward_batch(const int *tokens, int ntok, int start_pos) {
  const smollm_config_t *p = &g_cfg;
  RunState *s = &g_s;
  const int dim = p->dim, hs = p->head_dim, kv_dim = g_kv_dim, hd = p->hidden_dim;
  const int kv_mul = p->n_heads / p->n_kv_heads;
  const int ng = dim / GS, ngh = hd / GS;

  if (ntok <= 0) return s->logits;
  if (ntok == 1) return smollm_forward(tokens[0], start_pos);   /* decode: no batching to do */
  if (ntok > g_b.cap) ntok = g_b.cap;                           /* caller chunks; guard anyway */

  for (int b = 0; b < ntok; b++) embed_token(g_b.x + (size_t)b * dim, tokens[b]);

  for (int l = 0; l < p->n_layers; l++) {
    for (int b = 0; b < ntok; b++) {
      rmsnorm(g_b.xb + (size_t)b * dim, g_b.x + (size_t)b * dim,
              g_w.rms_att + (size_t)l * dim, dim);
      quantize_to_g(g_b.xq_q + (size_t)b * dim, g_b.xq_s + (size_t)b * ng,
                    g_b.xq_g + (size_t)b * ng, g_b.xb + (size_t)b * dim, dim);
    }
    /* q/k/v for the whole batch, one pass over each weight matrix */
    matmul_batch(g_b.hb, g_b.xq_q, g_b.xq_s, g_b.xq_g, ntok, &g_w.wq[l], dim, dim);      /* hb: q  [P][dim] */
    matmul_batch(g_b.hb2, g_b.xq_q, g_b.xq_s, g_b.xq_g, ntok, &g_w.wk[l], dim, kv_dim);  /* hb2: k [P][kv]  */

    /* RoPE + KV-cache write per token; k first, then v reuses hb2 */
    const size_t loff = (size_t)l * (size_t)p->seq_len * (size_t)kv_dim;
    for (int b = 0; b < ntok; b++) {
      const int pos = start_pos + b;
      for (int j = 0; j < hs / 2; j++) {
        const float a = (float)pos * g_rope_freq[j];
        g_rope_cos[j] = cosf(a);
        g_rope_sin[j] = sinf(a);
      }
      float *qv = g_b.hb + (size_t)b * dim;
      float *kv = g_b.hb2 + (size_t)b * kv_dim;
      for (int i = 0; i < dim; i += 2) {
        const float fcr = g_rope_cos[(i % hs) >> 1], fci = g_rope_sin[(i % hs) >> 1];
        const int rotn = (i < kv_dim) ? 2 : 1;
        for (int v = 0; v < rotn; v++) {
          float *vec = (v == 0) ? qv : kv;
          const float v0 = vec[i], v1 = vec[i + 1];
          vec[i] = v0 * fcr - v1 * fci;
          vec[i + 1] = v0 * fci + v1 * fcr;
        }
      }
      memcpy(s->key_cache + loff + (size_t)pos * kv_dim, kv, (size_t)kv_dim * sizeof(float));
    }
    matmul_batch(g_b.hb2, g_b.xq_q, g_b.xq_s, g_b.xq_g, ntok, &g_w.wv[l], dim, kv_dim);  /* hb2: v [P][kv] */
    for (int b = 0; b < ntok; b++)
      memcpy(s->value_cache + loff + (size_t)(start_pos + b) * kv_dim,
             g_b.hb2 + (size_t)b * kv_dim, (size_t)kv_dim * sizeof(float));

    /* attention per token — causal, and every key/value it needs is already in the cache */
    for (int b = 0; b < ntok; b++) {
      const int pos = start_pos + b;
      const float *qv = g_b.hb + (size_t)b * dim;
      float *xb = g_b.xb + (size_t)b * dim;
      for (int h = 0; h < p->n_heads; h++) {
        const float *q = qv + h * hs;
        float *att = s->att + h * p->seq_len;
        const int kvh = h / kv_mul;
        for (int t = 0; t <= pos; t++) {
          const float *k = s->key_cache + loff + (size_t)t * kv_dim + kvh * hs;
          float score = 0.0f;
          for (int i = 0; i < hs; i++) score += q[i] * k[i];
          att[t] = score / sqrtf((float)hs);
        }
        smollm_softmax(att, pos + 1);
        float *xbh = xb + h * hs;
        memset(xbh, 0, (size_t)hs * sizeof(float));
        for (int t = 0; t <= pos; t++) {
          const float *v = s->value_cache + loff + (size_t)t * kv_dim + kvh * hs;
          const float a = att[t];
          for (int i = 0; i < hs; i++) xbh[i] += a * v[i];
        }
      }
    }

    for (int b = 0; b < ntok; b++)
      quantize_to_g(g_b.xq_q + (size_t)b * dim, g_b.xq_s + (size_t)b * ng,
                    g_b.xq_g + (size_t)b * ng, g_b.xb + (size_t)b * dim, dim);
    matmul_batch(g_b.hb, g_b.xq_q, g_b.xq_s, g_b.xq_g, ntok, &g_w.wo[l], dim, dim);
    for (int b = 0; b < ntok; b++)
      for (int i = 0; i < dim; i++) g_b.x[(size_t)b * dim + i] += g_b.hb[(size_t)b * dim + i];

    for (int b = 0; b < ntok; b++) {
      rmsnorm(g_b.xb + (size_t)b * dim, g_b.x + (size_t)b * dim,
              g_w.rms_ffn + (size_t)l * dim, dim);
      quantize_to_g(g_b.xq_q + (size_t)b * dim, g_b.xq_s + (size_t)b * ng,
                    g_b.xq_g + (size_t)b * ng, g_b.xb + (size_t)b * dim, dim);
    }
    matmul_batch(g_b.hb, g_b.xq_q, g_b.xq_s, g_b.xq_g, ntok, &g_w.w1[l], dim, hd);
    matmul_batch(g_b.hb2, g_b.xq_q, g_b.xq_s, g_b.xq_g, ntok, &g_w.w3[l], dim, hd);
    for (int b = 0; b < ntok; b++) {
      float *h1 = g_b.hb + (size_t)b * hd, *h3 = g_b.hb2 + (size_t)b * hd;
      for (int i = 0; i < hd; i++) {
        float v = h1[i];
        v *= 1.0f / (1.0f + expf(-v));
        h1[i] = v * h3[i];
      }
      quantize_to_g(g_b.hq_q + (size_t)b * hd, g_b.hq_s + (size_t)b * ngh,
                    g_b.hq_g + (size_t)b * ngh, h1, hd);
    }
    matmul_batch(g_b.hb2, g_b.hq_q, g_b.hq_s, g_b.hq_g, ntok, &g_w.w2[l], hd, dim);
    for (int b = 0; b < ntok; b++)
      for (int i = 0; i < dim; i++) g_b.x[(size_t)b * dim + i] += g_b.hb2[(size_t)b * dim + i];
  }

  /* Only the last token's logits matter, so the 28 MB classifier runs once, not ntok times. */
  float *xl = g_b.x + (size_t)(ntok - 1) * dim;
  rmsnorm(s->x, xl, g_w.rms_final, dim);
  quantize(&s->xq, s->x, dim);
  matmul(s->logits, &s->xq, &g_w.tokens, dim, p->vocab_size);
  return s->logits;
}
