#ifndef SMOLLM_MODEL_H
#define SMOLLM_MODEL_H

#include <stdint.h>
#include <stddef.h>

/* The grouped-Q8_0 SmolLM forward pass. Deliberately free of chip dependencies: the same file is
 * compiled into the bearly25 demo and into test/host_forward_test.c, which runs it on the host
 * against the numpy reference (scripts/ref_runq.py). See src/model.c. */

typedef struct {
  int dim;
  int hidden_dim;
  int n_layers;
  int n_heads;
  int n_kv_heads;
  int vocab_size;      /* also the classifier width: SmolLM ties them */
  int seq_len;         /* KV-cache depth, baked in at export time */
  int head_dim;
  int kv_dim;          /* head_dim * n_kv_heads */
  int group_size;      /* quantization group (GS) */
  int quant_mode;      /* 0 = all int8, 1 = layers int4 + int8 classifier, 2 = all int4 */
  float rope_theta;    /* 100000 for SmolLM, 10000 for llama2 */
  float norm_eps;
} smollm_config_t;

/* Parses the blob header and points the weights AT THE BLOB — nothing is copied. Returns 0 on
 * success. `blob_bytes` is checked against the size the header implies. */
int smollm_model_load(const uint8_t *blob, size_t blob_bytes);

/* Allocates the RunState + KV cache (about 24 MB at seq_len 512) and the RoPE tables. */
int smollm_model_alloc(void);

const smollm_config_t *smollm_model_config(void);

/* One decode step. Returns the logits (vocab_size floats), owned by the model. */
float *smollm_forward(int token, int pos);

/* Prefill: run `ntok` consecutive tokens (positions start_pos ...) through the model in ONE pass
 * over the weights, and return the LAST token's logits. Bit-identical to calling smollm_forward()
 * in a loop, but reads the 143 MB of weights once instead of ntok times — the difference between
 * minutes and seconds on hardware that is entirely weight-bandwidth bound. Caller must keep
 * ntok <= SMOLLM_MAX_BATCH. */
float *smollm_forward_batch(const int *tokens, int ntok, int start_pos);

/* FNV-1a over a blob. The device computes this over the .incbin'd model and compares it against
 * the value the exporter recorded, which is the only way to tell "the weights did not survive the
 * 143 MB UART load / are not stable in DRAM" apart from "the arithmetic is wrong". Timing it also
 * measures the pure sequential-read cost of the whole model. */
uint64_t smollm_fnv1a64(const uint8_t *p, size_t n);

/* Measure the memory floor and every matmul variant on a real weight tensor, cross-checking them
 * against each other. Takes the cycle counter as a callback so this file stays chip-free. */
void smollm_bench(uint64_t (*now)(void));

/* In-place softmax over `size` floats. Exposed because the sampler needs exactly the same one the
 * attention uses. */
void smollm_softmax(float *x, int size);

/* ---- optional second hart ------------------------------------------------------------------
 * A batched matmul's output rows are independent, so hart 1 can take a range of them. The chip
 * layer owns hart bring-up and the handshake; this file only needs "start this range" and "wait".
 * Leave the pointers NULL (the default) for single-core, which is what the host tests use. */
typedef struct {
  void *xout;
  const void *xq;
  const void *xs;
  const void *xg;     /* per-group activation sums (Q4_1's m * sum(x) term); NULL otherwise */
  const void *w;
  int P, n, d, row_lo, row_hi;
} smollm_job_t;

extern void (*smollm_cowork_start)(const smollm_job_t *job);
extern void (*smollm_cowork_wait)(void);

/* Run one job's row range. Called by the chip layer's hart-1 worker. */
void smollm_run_job(const smollm_job_t *job);

/* Row-ranges hart 1 has completed. Zero after a run means the split never engaged. */
extern volatile unsigned long g_smollm_jobs_done;

/* 1 when the vector int8 dot product is compiled in (bit-identical to the scalar path). */
int smollm_uses_rvv(void);

/* Optional per-stage hook: called after the embedding, after every layer, and after the
 * classifier, with the sum of the vector at that point. Used by SMOLLM_DEBUG_GOLDEN and by the
 * host test to localize a divergence to a layer. Pass NULL to disable. */
typedef void (*smollm_stage_fn)(int stage, const char *name, float sum);
void smollm_set_stage_hook(smollm_stage_fn fn);

/* Which decode position the hook reports on (default 0). RoPE is the identity at position 0, so
 * a position-dependent bug is invisible there — set this to 1 or more to see it. */
void smollm_set_stage_pos(int pos);

/* Report every op inside one layer (norm, q/k/v, RoPE, attention, wo, SwiGLU, w2) instead of just
 * the layer's output — the same idea as dsp-citrinet's CN_TRACE_BLOCK. -1 = off. */
void smollm_set_trace_layer(int layer);

#endif /* SMOLLM_MODEL_H */
