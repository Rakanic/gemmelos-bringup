#ifndef C2C_DSP_WHISPER_H
#define C2C_DSP_WHISPER_H

/* Whisper tiny.en inference engine for DSP 25 (plan 005). FLOAT pipeline: activations in float,
 * weights Q8_0 (dequantized per-group in the matmul — same scheme as llama2.c runq / the borai int8
 * port). Encoder + decoder + greedy decode. Validated against the PyTorch golden via stage sums
 * (encoder) and greedy token ids (decoder). See .claude/plans/005-whisper-dsp.md. */
#include <stdint.h>
#include <stddef.h>

#define WHISPER_MAX_LAYERS 8

/* A Q8_0 weight matrix (out, in), row-major. q has out*in int8 values; s has out*in/gs fp32 group
 * scales. value(o,i) = q[o*in+i] * s[(o*in+i)/gs]. in is a multiple of gs so groups never cross a row.
 * qT/sT (built lazily under WHISPER_TILED) hold the K-major transpose used by the register-tiled GEMM:
 * qT[k*out + o] = q[o*in + k]; sT[g*out + o] = s[o*(in/gs) + g]. NULL until built. */
/* q4/s4 (built lazily under WHISPER_INT4) hold an int4 re-quantization of the weights, N-major, for the
 * bandwidth-bound reduction path (classifier / decoder). q4 packs two signed nibbles [-7,7] per byte
 * (row o = in/2 bytes); s4 has one fp32 scale per WHISPER_INT4_GS-element group. Halves weight DRAM. */
typedef struct { const int8_t *q; const float *s; int8_t *qT; float *sT; int8_t *q4; float *s4; } wq8_t;

typedef struct {
  const float *attn_ln_w, *attn_ln_b;
  const float *attn_q_b, *attn_v_b, *attn_o_b;      /* key projection has no bias in Whisper */
  const float *mlp_ln_w, *mlp_ln_b, *mlp0_b, *mlp2_b;
  wq8_t attn_q, attn_k, attn_v, attn_o, mlp0, mlp2;
} whisper_enc_layer_t;

typedef struct {
  const float *attn_ln_w, *attn_ln_b, *attn_q_b, *attn_v_b, *attn_o_b;
  const float *cross_ln_w, *cross_ln_b, *cross_q_b, *cross_v_b, *cross_o_b;
  const float *mlp_ln_w, *mlp_ln_b, *mlp0_b, *mlp2_b;
  wq8_t attn_q, attn_k, attn_v, attn_o;
  wq8_t cross_q, cross_k, cross_v, cross_o;
  wq8_t mlp0, mlp2;
} whisper_dec_layer_t;

typedef struct {
  int n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer;
  int n_text_ctx, n_text_state, n_text_head, n_text_layer, n_vocab;
  int mlp_hidden, conv_k, gs, shared_classifier;

  const float *conv1_w, *conv1_b, *conv2_w, *conv2_b;
  const float *enc_pos, *dec_pos;
  const float *enc_ln_post_w, *enc_ln_post_b, *dec_ln_w, *dec_ln_b;
  wq8_t token_embedding;                              /* (n_vocab, n_state); tied classifier */

  whisper_enc_layer_t enc[WHISPER_MAX_LAYERS];
  whisper_dec_layer_t dec[WHISPER_MAX_LAYERS];
} whisper_model_t;

/* Parse a wh01 blob (from export_whisper.py) into pointers. Returns 0 on success, <0 on bad magic. */
int whisper_model_load(const void *blob, whisper_model_t *m);

/* One (sum, absmax, mean) fingerprint of a stage's activation, matching the host golden. */
typedef struct { double sum, absmax, mean; } whisper_stat_t;

/* Encode: mel is (n_mels, n_frames) row-major. Writes enc_out (n_pos, n_state) into a malloc'd
 * buffer returned via *enc_out_p (caller frees); n_pos via *n_pos_p. If stats != NULL it must hold
 * 8 slots and receives per-stage fingerprints [conv1, conv2, pos_add, block0..3, enc_out]. */
int whisper_encode(const whisper_model_t *m, const float *mel, int n_frames,
                   float **enc_out_p, int *n_pos_p, whisper_stat_t *stats);

/* Greedy decode. Seeds with sot[sot_len], generates up to max_new tokens (stop at eot). Writes ids
 * to out_tokens, count to *n_out. */
int whisper_decode_greedy(const whisper_model_t *m, const float *enc_out, int n_enc,
                          const int *sot, int sot_len, int eot,
                          int *out_tokens, int max_new, int *n_out);

/* Full validate: load model + testcase (golden mel blob), run encoder+decoder, diff against the
 * goldens compiled in from whisper_reference.h. Prints a report; returns 0 = PASS, 1 = FAIL. */
int whisper_run_validate(const void *model_blob, const float *mel, int n_frames);

/* Transcribe a normalized log-mel (n_mels, n_frames): encode then greedy-decode, seeded with the
 * tiny.en SOT sequence, up to max_new tokens. Writes ids to out_tokens, count to *n_out. Returns
 * the number of encoder positions. */
int whisper_transcribe(const whisper_model_t *m, const float *mel, int n_frames,
                       int *out_tokens, int max_new, int *n_out);

/* Print token ids as decoded UTF-8 text after `tag` (uses the BPE vocab table). */
void whisper_print_tokens_text(const char *tag, const int *toks, int n);

/* Print the accumulated per-kernel cycle profile (no-op unless built with WHISPER_PROFILE=1). */
void whisper_profile_report(void);
void whisper_profile_reset(void);

/* Start the second hart for the dual-core matmul split (no-op unless built with WHISPER_DUALCORE=1).
 * Call once at startup, before any inference. */
void whisper_dualcore_init(void);

#endif /* C2C_DSP_WHISPER_H */
