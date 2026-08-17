#ifndef C2C_DSP_CITRINET_H
#define C2C_DSP_CITRINET_H

/* Citrinet-256 (NeMo `stt_en_citrinet_256_gamma_0_25`) speech-to-text for DSP 25.
 * SCALAR reference pipeline: activations fp32, pointwise weights Q8_0 (dequantized per group
 * inside the kernel — same scheme as moonshine.c / whisper.c / llama2.c runq).
 *
 * Why this model instead of Moonshine (measured, see the port notes): Citrinet is a CTC conv
 * encoder — NON-autoregressive. Every weight is read exactly ONCE per utterance, and each layer's
 * working set (<= 160 KB) fits the 256 KB cache and is reused across 38..301 time frames. Moonshine
 * moved ~170 MB per inference for a 27 MB model because its decoder + classifier (76% of the byte
 * budget) are re-read per output token; Citrinet moves ~10 MB, and that does not grow with the
 * length of the transcript.
 *
 * Architecture (23 blocks, all separable + squeeze-excite, 8x total stride):
 *   front-end : 80-bin log-mel, preemph 0.97, Hann-400 in a 512 FFT, hop 160, power spectrum,
 *               slaney mel, log(x + 2^-24), then PER-FEATURE mean/var normalization (ddof=1).
 *   block     : [depthwise conv -> pointwise matmul + bias (BatchNorm folded in) -> ReLU] x R,
 *               with the ReLU omitted on the last sub-block; then squeeze-excite; then a strided
 *               1x1 residual add; then ReLU. Stride lives on the DEPTHWISE conv of the last
 *               sub-block (stride_last).
 *   decoder   : one 1x1 conv to 1025 classes (1024 SentencePiece pieces + CTC blank), greedy CTC.
 *
 * BatchNorm never appears at runtime: export_citrinet.py folds it (with NeMo's eps=1e-3, NOT
 * torch's 1e-5 default) into the preceding pointwise as a row rescale + bias.
 *
 * Activations are CHANNEL-MAJOR throughout: x[c * n_frames + t]. That makes the depthwise conv
 * contiguous in t and turns the pointwise into an AXPY over t per (out,in) pair — the layout a
 * later RVV kernel wants. */
#include <stdint.h>
#include <stddef.h>

#define CN_MAX_BLOCKS 32
#define CN_MAX_SUB 8

/* A weight matrix [out, in]. Stored Q8_0 when (in % gs) == 0, else plain fp32 — the SAME predicate
 * export_citrinet.py applies, so no per-tensor flag is needed in the blob. Exactly one of
 * {q,s} / {f} is valid: f != NULL means fp32.
 * Q8_0: value(o,i) = q[o*in + i] * s[(o*in + i)/gs]; `in` is a multiple of gs so groups never
 * straddle a row. */
typedef struct {
  const int8_t *q;
  const float *s;
  const float *f;
  int out, in;
} cn_mat_t;

typedef struct {
  const float *dw;      /* depthwise kernel [c_in_j, k] */
  cn_mat_t pw;          /* pointwise [c_out, c_in_j], BatchNorm folded in */
  const float *bias;    /* [c_out], from the folded BatchNorm */
} cn_sub_t;

typedef struct {
  int c_in, c_out, repeat, k, stride, stride_last, dilation, has_se, se_hidden, has_res, pad;
  cn_sub_t sub[CN_MAX_SUB];
  cn_mat_t se1, se2;    /* [se_hidden, c_out] and [c_out, se_hidden] */
  cn_mat_t res;         /* [c_out, c_in] 1x1, applied to the block input with the block stride */
  const float *res_b;
} cn_block_t;

typedef struct {
  int n_blocks, n_mels, feat_out, n_classes, gs;
  int sample_rate, n_fft, win_length, hop_length, n_bins, blank;
  float log_guard, norm_eps, preemph;
  int max_k, max_c;
  const float *hann;    /* [win_length] symmetric Hann, from the blob */
  const float *fb;      /* [n_mels, n_bins] slaney-normalized mel filterbank, from the blob */
  cn_block_t blk[CN_MAX_BLOCKS];
  cn_mat_t dec;         /* [n_classes, feat_out] */
  const float *dec_b;
} cn_model_t;

/* Parse a "cn01" blob (from export_citrinet.py). `blob_bytes` must come from the LINKER symbols
 * (citrinet_model_blob_end - citrinet_model_blob), not from the blob's own header: on this silicon
 * the first read of a freshly-TSI-loaded region can return stale data, so the size that bounds the
 * integrity check must not itself be read from that region.
 * Returns 0 on success, <0 on bad magic / checksum / geometry. */
int cn_model_load(const void *blob, size_t blob_bytes, cn_model_t *m);

/* Re-run the whole-blob hash and report whether it STILL matches. Called after inference: if the
 * weights verify at boot but not afterwards, the DRAM contents changed while the program ran —
 * a hardware-stability problem, not a software one. Returns 1 if still intact, 0 if not. */
int cn_blob_recheck(const void *blob, size_t blob_bytes);

/* Standalone DRAM write/read-back test — no model involved. Writes a position-dependent pattern
 * over `bytes` (choose >> cache so early words are forced out to DRAM), then reads it back and
 * reports mismatches. Everything verified intact so far (the blob, .text) is READ-ONLY data that is
 * never written after load; the activations are written constantly. If this reports errors, the
 * store path is the root cause and no amount of kernel work will fix it. Returns mismatch count. */
unsigned long cn_memtest(size_t bytes);

/* Stack-overflow detector. Call cn_stack_paint() once at boot (it prints the range crt0 actually
 * gave this hart) and cn_stack_report() after the work is done to get the high-water mark. An
 * overflow here is silent on this platform — it runs off the bottom of the stack into address space
 * the SoC does not back, so stores stop reading back and only the AUTOMATIC variables go wrong,
 * while register-held values stay correct. Spike backs that whole region with real zeroed memory,
 * so the same overflow is invisible in simulation. Leave both calls in. */
void cn_stack_paint(void);
void cn_stack_report(void);

/* Byte-wise FNV-1a-64. Exposed so the demo can fingerprint regions that carry no checksum of their
 * own — notably the embedded AUDIO blob, which has no header to hold one yet fully determines the
 * transcript. dsp25-tests/citrinet-test/scripts/text_checksum.py prints the host-side value to
 * compare against; a value that changes between two flashes of the same ELF means the loader
 * corrupted the payload, which is otherwise indistinguishable from a numerical bug. */
uint64_t cn_fnv1a64_bytes(const void *p, size_t n);


/* One (sum, absmax, mean) fingerprint of a stage, matching the host golden. */
typedef struct { double sum, absmax, mean; } cn_stat_t;

/* Number of mel frames a given sample count produces (NeMo's get_seq_len). */
int cn_num_frames(const cn_model_t *m, int n_samples);

/* Log-mel front-end. Writes CHANNEL-MAJOR mel[c * n_frames + t] into a malloc'd buffer returned
 * via *mel_p (caller frees); frame count via *n_frames_p. Returns 0 on success. */
int cn_logmel(const cn_model_t *m, const float *audio, int n_samples, float **mel_p, int *n_frames_p);

/* Run the 23-block conv encoder. mel is [n_mels, n_frames] channel-major; enc_out is
 * [feat_out, n_enc] channel-major, malloc'd (caller frees). If stats != NULL it must hold
 * n_blocks slots and receives the per-block fingerprints. */
int cn_encode(const cn_model_t *m, const float *mel, int n_frames,
              float **enc_p, int *n_enc_p, cn_stat_t *stats);

/* Decoder 1x1 + greedy CTC (argmax per frame, collapse repeats, drop blanks). */
int cn_ctc_greedy(const cn_model_t *m, const float *enc, int n_enc,
                  int *out_tokens, int max_out, int *n_out);

/* Full pipeline: audio -> log-mel -> encoder -> greedy CTC. Returns encoder frames. */
int cn_transcribe(const cn_model_t *m, const float *audio, int n_samples,
                  int *out_tokens, int max_out, int *n_out);

/* Validate: run the full pipeline on `audio` and diff against citrinet_reference.h (per-stage
 * sums + CTC token ids). Prints a report; returns 0 = PASS, 1 = FAIL. */
int cn_run_validate(const void *blob, size_t blob_bytes, const float *audio, int n_samples);

/* Print token ids as text (SentencePiece detok via citrinet_vocab.h). */
void cn_print_tokens_text(const char *tag, const int *toks, int n);

/* Same detokenization into a caller-supplied buffer instead of the console — what a non-console
 * consumer needs (the C2C demo forwards the transcript to Bearly ML 25 as an LLM prompt). Always
 * NUL-terminates; returns the length written, excluding the NUL. */
int cn_tokens_to_text(char *buf, int cap, const int *toks, int n);

/* Per-kernel cycle profile (no-op unless built CN_PROFILE=1). */
void cn_profile_report(void);
void cn_profile_reset(void);

/* Print the vector unit's measured geometry (vsetvlmax -> VLEN). The RVV kernels are written around
 * VLEN=256 while the build only implies zvl128b, so this checks the assumption rather than trusting
 * it. Cheap enough to leave in. */
void cn_report_vlen(void);

/* One-shot probe of the DMA strided gather this port's scratchpad staging would depend on. Verifies
 * a citrinet-shaped tile gather into the 64 KB SRAM and reports cycles. No-op unless CN_DMA_PROBE=1. */
void cn_dma_probe(void);

/* Bring up hart 1 (no-op unless built CN_DUALCORE=1). MUST be called after init_test() — the PLL and
 * UART have to be live first — and before any kernel runs. Skipping hthread_init() makes hart 1
 * ignore every task, which presents as a hang in hthread_join, not as an error. */
void cn_dualcore_init(void);

#endif /* C2C_DSP_CITRINET_H */
