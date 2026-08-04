/*
 * whisper_frontend.c — Whisper log-mel front-end for DSP 25 (plan 005, P0).
 *
 * Mirrors whisper.audio.log_mel_spectrogram per frame:
 *   windowed = frame * hann(400)
 *   X[k]     = sum_n windowed[n] * exp(-j 2*pi*k*n/400),  k = 0..200   (201 one-sided bins)
 *   power[k] = Re(X[k])^2 + Im(X[k])^2
 *   mel[m]   = sum_k mel_filter[m][k] * power[k]
 *   out[m]   = log10(max(mel[m], 1e-10))
 * The trailing (log_spec+4)/4 after a GLOBAL-max clamp is applied by the caller over all frames.
 *
 * n_fft=400 is not a power of two -> we use a direct DFT (precomputed cos/sin), not the radix-2
 * NMSIS rFFT. The two per-bin dot-products vectorize cleanly under RVV later.
 */
#include "whisper_frontend.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef WHISPER_DUALCORE_DEBUG
#define WHISPER_DUALCORE_DEBUG 0
#endif
#if WHISPER_DUALCORE_DEBUG
#define FE_DBG(...) do { printf(__VA_ARGS__); } while (0)
#else
#define FE_DBG(...) do {} while (0)
#endif

#ifndef WHISPER_USE_RVV
#define WHISPER_USE_RVV 0
#endif
/* Front-end DFT RVV is gated SEPARATELY from the engine RVV. The f32m8 DFT reduction hangs on
 * DSP-25 silicon (works on Spike) — same class as the vsext.vf4 GCC miscompile. The DFT is ~5% of
 * runtime, so we keep it scalar on hardware by default and leave the engine matvec RVV untouched.
 * Flip to 1 only to re-test the vector DFT on silicon. */
#ifndef WHISPER_FE_RVV
#define WHISPER_FE_RVV 0
#endif
#if WHISPER_FE_RVV
#include <riscv_vector.h>
#endif

#ifndef WHISPER_DUALCORE
#define WHISPER_DUALCORE 0
#endif
#if WHISPER_DUALCORE
#include "hthread.h"
#endif

/* Direct-DFT twiddles, built once at init. ~640 KB each in .bss (DRAM is 256 MiB — trivial).
 * Bin-major (g_dft_cos[k*NFFT+n]) feeds the scalar per-bin reduction. Tap-major (g_dft_cosT[n*NBINS+k])
 * feeds the OUTER-PRODUCT RVV path: for each tap n, one contiguous vector load of the k-th bins, no
 * reduction (reductions hang on this silicon — same class as the old RVV DFT). */
static float g_dft_cos[WF_N_BINS * WF_N_FFT];
static float g_dft_sin[WF_N_BINS * WF_N_FFT];
#if WHISPER_FE_RVV
static float g_dft_cosT[WF_N_FFT * WF_N_BINS];
static float g_dft_sinT[WF_N_FFT * WF_N_BINS];
#endif
static int   g_inited = 0;

void whisper_frontend_init(void) {
  const double two_pi = 6.283185307179586476925286766559;
  for (int k = 0; k < WF_N_BINS; k++) {
    for (int n = 0; n < WF_N_FFT; n++) {
      double ang = two_pi * (double)k * (double)n / (double)WF_N_FFT;
      float cv = (float)cos(ang), sv = (float)sin(ang);
      g_dft_cos[k * WF_N_FFT + n] = cv;
      g_dft_sin[k * WF_N_FFT + n] = sv;
#if WHISPER_FE_RVV
      g_dft_cosT[n * WF_N_BINS + k] = cv;
      g_dft_sinT[n * WF_N_BINS + k] = sv;
#endif
    }
  }
  g_inited = 1;
}

/* Mel filterbank + log10 for one frame's power spectrum. */
static void mel_from_power(const float *power, float *out_mel) {
  for (int m = 0; m < WF_N_MELS; m++) {
    const float *mf = &wf_mel_filter[m * WF_N_BINS];
    float acc = 0.0f;
    for (int k = 0; k < WF_N_BINS; k++) acc += mf[k] * power[k];
    if (acc < 1e-10f) acc = 1e-10f;
    out_mel[m] = log10f(acc);
  }
}

void whisper_logmel_frame(const float *frame, float *out_mel) {
  float win[WF_N_FFT];
  for (int n = 0; n < WF_N_FFT; n++) win[n] = frame[n] * wf_hann[n];

  float power[WF_N_BINS];
#if WHISPER_FE_RVV
  /* Outer-product DFT (silicon-safe: no reduction, m4, contiguous loads — mirrors the tiled matmul).
   * For each block of bins, accumulate re/im across all taps: re[k] += win[n]*cosT[n][k] (vfmacc, one
   * scalar win[n] broadcast over the bin vector), im[k] -= win[n]*sinT[n][k] (vfnmsac). Then power = re^2+im^2. */
  for (int k0 = 0; k0 < WF_N_BINS;) {
    size_t vl = __riscv_vsetvl_e32m4((size_t)(WF_N_BINS - k0));
    vfloat32m4_t vre = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t vim = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    for (int n = 0; n < WF_N_FFT; n++) {
      float wn = win[n];
      vre = __riscv_vfmacc_vf_f32m4(vre, wn, __riscv_vle32_v_f32m4(&g_dft_cosT[(size_t)n * WF_N_BINS + k0], vl), vl);
      vim = __riscv_vfnmsac_vf_f32m4(vim, wn, __riscv_vle32_v_f32m4(&g_dft_sinT[(size_t)n * WF_N_BINS + k0], vl), vl);
    }
    vfloat32m4_t vpow = __riscv_vfmul_vv_f32m4(vre, vre, vl);
    vpow = __riscv_vfmacc_vv_f32m4(vpow, vim, vim, vl);
    __riscv_vse32_v_f32m4(&power[k0], vpow, vl);
    k0 += (int)vl;
  }
#else
  for (int k = 0; k < WF_N_BINS; k++) {
    const float *c = &g_dft_cos[k * WF_N_FFT];
    const float *s = &g_dft_sin[k * WF_N_FFT];
    float re = 0.0f, im = 0.0f;
    for (int n = 0; n < WF_N_FFT; n++) {
      re += win[n] * c[n];
      im -= win[n] * s[n];   /* exp(-j...) -> negative sine */
    }
    power[k] = re * re + im * im;
  }
#endif
  mel_from_power(power, out_mel);
}

#if WHISPER_FE_RVV
/* Batched 2-frame DFT: each twiddle vector is loaded ONCE and reused across both frames — halves the
 * 640 KB-table DRAM traffic (the front-end's memory bottleneck) and uses the idle half of the vector
 * register file (4 accumulators instead of 2). Identical math to whisper_logmel_frame. */
static void whisper_logmel_frame2(const float *frame0, const float *frame1,
                                  float *out_mel0, float *out_mel1) {
  float win0[WF_N_FFT], win1[WF_N_FFT];
  for (int n = 0; n < WF_N_FFT; n++) { win0[n] = frame0[n] * wf_hann[n]; win1[n] = frame1[n] * wf_hann[n]; }

  float power0[WF_N_BINS], power1[WF_N_BINS];
  for (int k0 = 0; k0 < WF_N_BINS;) {
    size_t vl = __riscv_vsetvl_e32m4((size_t)(WF_N_BINS - k0));
    vfloat32m4_t re0 = __riscv_vfmv_v_f_f32m4(0.0f, vl), im0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vfloat32m4_t re1 = __riscv_vfmv_v_f_f32m4(0.0f, vl), im1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    for (int n = 0; n < WF_N_FFT; n++) {
      vfloat32m4_t cv = __riscv_vle32_v_f32m4(&g_dft_cosT[(size_t)n * WF_N_BINS + k0], vl);
      vfloat32m4_t sv = __riscv_vle32_v_f32m4(&g_dft_sinT[(size_t)n * WF_N_BINS + k0], vl);
      re0 = __riscv_vfmacc_vf_f32m4(re0, win0[n], cv, vl); im0 = __riscv_vfnmsac_vf_f32m4(im0, win0[n], sv, vl);
      re1 = __riscv_vfmacc_vf_f32m4(re1, win1[n], cv, vl); im1 = __riscv_vfnmsac_vf_f32m4(im1, win1[n], sv, vl);
    }
    vfloat32m4_t p0 = __riscv_vfmacc_vv_f32m4(__riscv_vfmul_vv_f32m4(re0, re0, vl), im0, im0, vl);
    vfloat32m4_t p1 = __riscv_vfmacc_vv_f32m4(__riscv_vfmul_vv_f32m4(re1, re1, vl), im1, im1, vl);
    __riscv_vse32_v_f32m4(&power0[k0], p0, vl);
    __riscv_vse32_v_f32m4(&power1[k0], p1, vl);
    k0 += (int)vl;
  }
  mel_from_power(power0, out_mel0);
  mel_from_power(power1, out_mel1);
}
#endif

/* Compute log-mel for a frame range [f0,f1) into out_mel (coeff-major) and return the local max.
 * Frames are independent (each writes its own column), so two harts can split the range with no
 * shared writes — only the gmax reduction is combined by the caller. */
static float logmel_frame_range(const float *pad, float *out_mel, int n_frames, int f0, int f1) {
  float tmp0[WF_N_MELS], tmp1[WF_N_MELS];
  float gmax = -1e30f;
  int f = f0;
#if WHISPER_FE_RVV
  for (; f + 2 <= f1; f += 2) {   /* two frames per twiddle-table pass */
    whisper_logmel_frame2(&pad[(size_t)f * WF_HOP], &pad[(size_t)(f + 1) * WF_HOP], tmp0, tmp1);
    for (int m = 0; m < WF_N_MELS; m++) {
      out_mel[(size_t)m * n_frames + f]     = tmp0[m]; if (tmp0[m] > gmax) gmax = tmp0[m];
      out_mel[(size_t)m * n_frames + f + 1] = tmp1[m]; if (tmp1[m] > gmax) gmax = tmp1[m];
    }
  }
#endif
  for (; f < f1; f++) {
    whisper_logmel_frame(&pad[(size_t)f * WF_HOP], tmp0);
    for (int m = 0; m < WF_N_MELS; m++) {
      out_mel[(size_t)m * n_frames + f] = tmp0[m];
      if (tmp0[m] > gmax) gmax = tmp0[m];
    }
  }
  return gmax;
}

#if WHISPER_DUALCORE
/* hart-1 frame-range descriptor + result, in BSS (visible across harts, like the matmul path). */
typedef struct { const float *pad; float *out_mel; int n_frames, f0, f1; float gmax; } fe_blk_t;
static fe_blk_t g_fe_h1;
static void fe_worker(void *a) {
  fe_blk_t *b = (fe_blk_t *)a;
  b->gmax = logmel_frame_range(b->pad, b->out_mel, b->n_frames, b->f0, b->f1);
}
#endif

/* Minimum mel frames fed to the encoder. Whisper is trained on 30 s (3000 frames) zero-padded, so an
 * ultra-short clip (e.g. 34 frames) is badly out-of-distribution and the decoder hallucinates repeats.
 * Zero-padding the mel up to a moderate floor stabilizes short utterances while keeping the encoder
 * bounded (~64 positions, not 1500). Real frames are computed from audio; the pad frames are silence. */
#ifndef WHISPER_FE_MIN_FRAMES
#define WHISPER_FE_MIN_FRAMES 160   /* ~80 enc positions: enough context to stop short-clip hallucination */
#endif

int whisper_logmel_full(const float *audio, int n_samples, float *out_mel, int max_frames) {
  const int P = WF_N_FFT / 2;                       /* center pad = 200 */
  int n_real = n_samples / WF_HOP;                  /* whisper: floor(n/hop), after dropping last */
  if (n_real > max_frames) n_real = max_frames;
  if (n_real <= 0) return 0;
  int n_frames = n_real;                            /* stride of the (possibly padded) output */
  if (n_frames < WHISPER_FE_MIN_FRAMES) n_frames = WHISPER_FE_MIN_FRAMES;
  if (n_frames > max_frames) n_frames = max_frames;

  FE_DBG("[frontend] logmel_full: n_samples=%d n_frames=%d P=%d\n", n_samples, n_frames, P);
  /* Build the reflect-padded signal (torch.stft center=True, pad_mode='reflect'). */
  float *pad = (float *)malloc((size_t)(n_samples + 2 * P) * sizeof(float));
  FE_DBG("[frontend] pad malloc=%p\n", (void *)pad);
  for (int i = 0; i < P; i++) pad[i] = audio[(P - i < n_samples) ? (P - i) : 0];
  for (int i = 0; i < n_samples; i++) pad[P + i] = audio[i];
  for (int i = 0; i < P; i++) {
    int src = n_samples - 2 - i;
    pad[P + n_samples + i] = audio[(src >= 0) ? src : 0];
  }

  /* Per-frame raw log-mel (coeff-major, stride n_frames) over the REAL frames [0,n_real); track gmax
   * for whisper's normalization (real frames only — pad frames must not lift the max). */
  float gmax = -1e30f;
  FE_DBG("[frontend] pad filled, starting %d DFT frames (out stride %d)...\n", n_real, n_frames);
#if WHISPER_DUALCORE
  if (n_real >= 4) {
    const int mid = n_real / 2;
    g_fe_h1 = (fe_blk_t){ pad, out_mel, n_frames, mid, n_real, -1e30f };
    __asm__ volatile("fence rw, rw" ::: "memory");
    hthread_issue(1, fe_worker, &g_fe_h1);                              /* hart 1: frames [mid,n_real) */
    float gmax0 = logmel_frame_range(pad, out_mel, n_frames, 0, mid);  /* hart 0: frames [0,mid) */
    hthread_join(1);
    __asm__ volatile("fence rw, rw" ::: "memory");
    gmax = gmax0 > g_fe_h1.gmax ? gmax0 : g_fe_h1.gmax;
  } else
#endif
  {
    gmax = logmel_frame_range(pad, out_mel, n_frames, 0, n_real);
  }
  free(pad);

  /* Zero-pad (silence) the frames [n_real, n_frames): set a very-negative raw value so the normalization
   * below clamps them to the floor — exactly what a run of silent mel frames produces in whisper. */
  for (int f = n_real; f < n_frames; f++)
    for (int m = 0; m < WF_N_MELS; m++) out_mel[(size_t)m * n_frames + f] = -1e30f;
  FE_DBG("[frontend] all frames done (real=%d pad_to=%d), normalizing\n", n_real, n_frames);

  /* Normalize: log_spec = max(log_spec, gmax-8); (log_spec + 4) / 4. */
  const float floorv = gmax - 8.0f;
  const size_t total = (size_t)WF_N_MELS * n_frames;
  for (size_t i = 0; i < total; i++) {
    float v = out_mel[i];
    if (v < floorv) v = floorv;
    out_mel[i] = (v + 4.0f) / 4.0f;
  }
  return n_frames;
}
