#ifndef C2C_DSP_MOONSHINE_H
#define C2C_DSP_MOONSHINE_H

/* Moonshine tiny speech-to-text engine for DSP 25 (plan 006). FLOAT pipeline: activations in
 * float, weights Q8_0 (dequantized per-group inside the matvec — same scheme as whisper.c /
 * llama2.c runq). Raw-audio conv preprocessor + encoder + decoder + greedy decode. Validated
 * against the HF transformers golden via encoder stage sums + greedy token ids.
 *
 * Architecture deltas vs the Whisper engine (whisper.c), all handled here:
 *   - front-end is the model's own raw-audio conv stem (tanh(conv1)->groupnorm->gelu(conv2)->
 *     gelu(conv3)); NO log-mel, NO positional embedding.
 *   - interleaved partial RoPE (rotary_dim of head_dim) on encoder AND decoder self-attention.
 *   - all LayerNorm are bias-free; decoder MLP is SwiGLU (silu-gated); encoder MLP is plain GELU.
 *   - tied classifier (logits = h @ embed_tokens.weight.T).
 * See .claude/plans/006-moonshine-dsp.md. */
#include <stdint.h>
#include <stddef.h>

#define MS_MAX_LAYERS 8

/* Fixed conv-stem geometry (same for tiny/base moonshine): kernel sizes + strides, all VALID
 * (no padding). conv2 out channels = 2*hidden. */
#define MS_CONV1_K 127
#define MS_CONV1_S 64
#define MS_CONV2_K 7
#define MS_CONV2_S 3
#define MS_CONV3_K 3
#define MS_CONV3_S 2

/* A Q8_0 weight matrix (out,in) row-major: q has out*in int8; s has out*in/gs fp32 group scales.
 * value(o,i) = q[o*in+i] * s[(o*in+i)/gs]. `in` is a multiple of gs so groups never cross a row.
 * qT/sT (built lazily for many-row matmuls under the register-tiled GEMM) hold the K-major transpose:
 * qT[k*out+o] = q[o*in+k]; sT[g*out+o] = s[o*(in/gs)+g]. NULL until built.
 * q4/s4 (built lazily for the big bandwidth-bound reduction matmuls under MS_INT4) hold a K-major
 * int4 [-7,7] repack: halves the weight DRAM stream of the classifier + decoder fc1. NULL until built. */
typedef struct { const int8_t *q; const float *s; int8_t *qT; float *sT; int8_t *q4; float *s4; } msq8_t;

typedef struct {
  const float *in_ln_w, *post_ln_w;      /* bias-free LayerNorm weights */
  const float *fc1_b, *fc2_b;            /* encoder MLP biases (attention has NO bias) */
  msq8_t q, k, v, o, fc1, fc2;
} ms_enc_layer_t;

typedef struct {
  const float *in_ln_w, *post_ln_w, *final_ln_w;
  const float *fc1_b, *fc2_b;            /* fc1 out = 2*ff (SwiGLU gate+up); fc2 in = ff */
  msq8_t q, k, v, o;                     /* causal self-attention (RoPE) */
  msq8_t cq, ck, cv, co;                 /* cross-attention into encoder output (no RoPE) */
  msq8_t fc1, fc2;
} ms_dec_layer_t;

typedef struct {
  int hidden, n_enc, n_dec, n_head, head_dim, rotary_dim, ff_enc, ff_dec, vocab;
  int bos, eos, max_pos, gs;
  float rope_theta;
  float inv_freq[64];                    /* rotary_dim/2 precomputed inverse frequencies */

  const float *conv1_w;                  /* [hidden,1,127] (no bias) */
  const float *gn_w, *gn_b;              /* GroupNorm(1,hidden) */
  const float *conv2_w, *conv2_b;        /* [2*hidden,hidden,7] */
  const float *conv3_w, *conv3_b;        /* [hidden,2*hidden,3] */
  const float *enc_ln_w, *dec_ln_w;      /* final bias-free norms */
  msq8_t embed;                          /* [vocab,hidden] — tied classifier */

  ms_enc_layer_t enc[MS_MAX_LAYERS];
  ms_dec_layer_t dec[MS_MAX_LAYERS];
} ms_model_t;

/* Parse an ms01 blob (from export_moonshine.py) into pointers. Returns 0 on success, <0 on bad magic. */
int ms_model_load(const void *blob, ms_model_t *m);

/* One (sum, absmax, mean) fingerprint of a stage's activation, matching the host golden. */
typedef struct { double sum, absmax, mean; } ms_stat_t;

/* Encode raw 16 kHz audio (n_samples float). Writes enc_out (n_pos, hidden) to a malloc'd buffer
 * returned via *enc_out_p (caller frees); n_pos via *n_pos_p. If stats != NULL it must hold
 * (n_enc+2) slots and receives per-stage fingerprints [pre, enc0..encN-1, enc_out]. */
int ms_encode(const ms_model_t *m, const float *audio, int n_samples,
              float **enc_out_p, int *n_pos_p, ms_stat_t *stats);

/* Greedy decode, seeded with BOS. Generates up to max_new tokens, stops at eot. guards!=0 enables
 * the mic-mode anti-loop / sentence-stop heuristics (leave 0 for exact golden reproduction). */
int ms_decode_greedy(const ms_model_t *m, const float *enc_out, int n_enc,
                     int *out_tokens, int max_new, int guards, int *n_out);

/* Full pipeline: encode audio then greedy-decode. Returns encoder positions; tokens via out/n_out. */
int ms_transcribe(const ms_model_t *m, const float *audio, int n_samples,
                  int *out_tokens, int max_new, int guards, int *n_out);

/* Validate: load model + run encoder(stage taps)+decoder on `audio`, diff against moonshine_reference.h.
 * Prints a report; returns 0 = PASS, 1 = FAIL. */
int ms_run_validate(const void *model_blob, const float *audio, int n_samples);

/* Print token ids as decoded text after `tag` (SentencePiece detok via the moonshine vocab table). */
void ms_print_tokens_text(const char *tag, const int *toks, int n);

/* Fast conv unit test: RVV vs scalar conv on small synthetic inputs (all stride/kernel geometries).
 * Prints per-config CONV-CHECK lines; returns 0 = all PASS. Runs in << 1 s (validates the kernel
 * without simulating the whole model — ideal for a quick Spike check). */
int ms_conv_selftest(void);

/* Per-kernel cycle profile (no-op unless built MS_PROFILE=1). */
void ms_profile_report(void);
void ms_profile_reset(void);

/* Start the second hart for the dual-core matmul/conv split (no-op unless built MS_DUALCORE=1).
 * Call once at startup, after init_test (PLL/UART up) and before any inference. */
void ms_dualcore_init(void);

/* Measure the compute vs memory roofline with the real matvec kernel (no-op unless
 * MOONSHINE_ROOFLINE=ON). Prints two lines; call once at startup. */
void ms_roofline_report(void);

#endif /* C2C_DSP_MOONSHINE_H */
