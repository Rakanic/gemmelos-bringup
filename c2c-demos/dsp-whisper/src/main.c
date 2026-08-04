/*
 * dsp-whisper — Whisper tiny.en on DSP 25 (plan 005). FLOAT+RVV pipeline. Three run modes:
 *
 *   default            : validate against embedded golden mel (stage sums + tokens). Spike-testable.
 *   DSP_WHISPER_FROM_AUDIO : transcribe the embedded audio blob through the full on-device path
 *                            (audio -> log-mel -> encode -> decode). Spike-testable, no I2S.
 *   DSP_WHISPER_USE_MIC    : capture live audio from the I2S mic (VAD-gated) and transcribe in a
 *                            loop. Real silicon only (Spike has no I2S). Build PLATFORM=CHIP.
 *
 * Build (Spike):  make build CHIP=dsp25 PLATFORM=SIMS TARGET=dsp-whisper EXTRA_CMAKE_ARGS="-DWHISPER_USE_RVV=ON"
 * Build (mic):    make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-whisper EXTRA_CMAKE_ARGS="-DWHISPER_USE_RVV=ON -DDSP_WHISPER_USE_MIC=ON"
 * See .claude/plans/005-whisper-dsp.md.
 */
#include <stdint.h>
#include <math.h>

#include "main.h"
#include "whisper.h"
#include "whisper_frontend_testvec.h"

#if DSP_WHISPER_USE_MIC
#include "rocketcore.h"
#include "hal_i2s.h"
#endif

/* Embedded blobs from model_data.S */
extern const unsigned char whisper_model_blob[];
extern const unsigned char whisper_testcase_blob[];
extern const unsigned char whisper_testcase_blob_end[];
extern const unsigned char whisper_audio_blob[];
extern const unsigned char whisper_audio_blob_end[];

static inline uint64_t wf_rdcycle(void) {
  uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c;
}

/* One-shot DMA feasibility probe: time the core streaming a 32 KB weight-sized chunk from DRAM vs from
 * the 64 KB on-chip SRAM (0x08000000) after DMA-ing it there. The DRAM/SRAM read-cycle RATIO is the gate
 * for weight double-buffering — if SRAM reads aren't meaningfully faster, prefetching won't hide the
 * DRAM-latency stalls and the (large) integration isn't worth it. Runs once at boot, then the normal demo. */
#if DSP_WHISPER_DMA_PROBE
#include "hal_dma.h"
static void dma_feasibility_probe(void) {
  enum { BYTES = 32u * 1024u, WORDS = BYTES / 4u, PKT = 64u };  /* PKT = 2^LOGW(6) */
  const volatile uint32_t *dram = (const volatile uint32_t *)(whisper_model_blob + 4096);
  volatile uint32_t *sram = (volatile uint32_t *)0x08000000u;
  volatile uint32_t acc = 0;
  uint64_t t0 = wf_rdcycle();
  for (uint32_t i = 0; i < WORDS; i++) acc += dram[i];      /* streaming DRAM read */
  uint64_t t1 = wf_rdcycle();
  dma_transaction_t tx;
  tx.core = 0; tx.transaction_id = 1; tx.transaction_priority = 1; tx.peripheral_id = 0;
  tx.addr_r = (uint64_t)(uintptr_t)dram; tx.addr_w = (uint64_t)(uintptr_t)sram;
  tx.inc_r = PKT; tx.inc_w = PKT; tx.len = (uint16_t)(BYTES / PKT); tx.logw = 6;
  tx.do_interrupt = false; tx.do_address_gate = false;
  uint64_t d0 = wf_rdcycle();
  set_DMA_C(0, tx, true); start_DMA(0, tx.transaction_id, (void *)0); dma_wait_till_inactive(30); dma_reset();
  uint64_t d1 = wf_rdcycle();
  uint64_t t2 = wf_rdcycle();
  for (uint32_t i = 0; i < WORDS; i++) acc += sram[i];      /* streaming SRAM read */
  uint64_t t3 = wf_rdcycle();
  DSP_WHISPER_LOG("[dma-probe] 32KB: dram_read=%lu sram_read=%lu dma_xfer=%lu  (sram is %lux faster; acc=%u)\n",
                  (unsigned long)(t1 - t0), (unsigned long)(t3 - t2), (unsigned long)(d1 - d0),
                  (unsigned long)((t1 - t0) / ((t3 - t2) ? (t3 - t2) : 1)), (unsigned)acc);
}
#endif

static whisper_model_t g_model;
static float g_mel[WF_N_MELS * DSP_WHISPER_MAX_FRAMES];
static int   g_toks[DSP_WHISPER_MAX_TOKENS];

/* P0 front-end self-check (self-contained golden testvec). Returns 1 = PASS. */
static int frontend_check(void) {
  whisper_frontend_init();
  float mel[WF_N_MELS];
  double max_abs = 0.0;
  for (int f = 0; f < WF_TEST_FRAMES; f++) {
    whisper_logmel_frame(&wf_test_audio[f * WF_HOP], mel);
    for (int m = 0; m < WF_N_MELS; m++) {
      double d = (double)mel[m] - (double)wf_golden_logmel[f * WF_N_MELS + m];
      if (d < 0.0) d = -d;
      if (d > max_abs) max_abs = d;
    }
  }
  int pass = (max_abs <= (double)DSP_WHISPER_LOGMEL_TOL);
  DSP_WHISPER_LOG("[dsp-whisper] FRONTEND-CMP frames=%d max_abs_diff=%.6f -> %s\n",
                  WF_TEST_FRAMES, max_abs, pass ? "PASS" : "FAIL");
  return pass;
}

/* audio (n_samples @16 kHz) -> log-mel -> encode -> greedy decode -> print transcript. */
static void transcribe_audio(const float *audio, int n_samples) {
  whisper_profile_reset();   /* per-utterance counters (else the PROFILE line sums across all words) */
  uint64_t t0 = wf_rdcycle();
  int nf = whisper_logmel_full(audio, n_samples, g_mel, DSP_WHISPER_MAX_FRAMES);
  DSP_WHISPER_LOG("[dsp-whisper] log-mel done (%d frames); running encoder+decoder...\n", nf);
  uint64_t tm = wf_rdcycle();
  int n_out = 0;
  int n_pos = whisper_transcribe(&g_model, g_mel, nf, g_toks, DSP_WHISPER_MAX_TOKENS, &n_out);
  uint64_t t1 = wf_rdcycle();
  DSP_WHISPER_LOG("[dsp-whisper] frames=%d enc_pos=%d tokens=%d frontend=%lu engine=%lu total=%lu cycles\n",
                  nf, n_pos, n_out, (unsigned long)(tm - t0), (unsigned long)(t1 - tm),
                  (unsigned long)(t1 - t0));
  whisper_profile_report();
  whisper_print_tokens_text("[dsp-whisper] transcript: ", g_toks, n_out);
  /* Wall-clock latency from audio-in-hand (word received) to prediction ready. Uses the configured
   * PLL target; if the PLL locked elsewhere, scale by (actual_freq / DSP_WHISPER_TARGET_FREQ_HZ). */
  uint64_t cyc = t1 - t0;
  uint64_t ms  = cyc / (uint64_t)(DSP_WHISPER_TARGET_FREQ_HZ / 1000u);
  DSP_WHISPER_LOG("[dsp-whisper] LATENCY word->prediction: %lu ms  (%lu cycles @ %u MHz)\n",
                  (unsigned long)ms, (unsigned long)cyc,
                  (unsigned)(DSP_WHISPER_TARGET_FREQ_HZ / 1000000u));
}

/* ------------------------------------------------------------------------- I2S mic capture ------ */
#if DSP_WHISPER_USE_MIC
uint64_t target_frequency = DSP_WHISPER_TARGET_FREQ_HZ;  /* PLL target; mic clkdiv derived from it */

static i2s_params_t g_i2s_params_mic = {
    .tx_en = 1, .rx_en = 1, .bitdepth_tx = I2S_BITDEPTH_32, .bitdepth_rx = I2S_BITDEPTH_32,
    .clkgen = 1, .dacen = 0, .ws_len = 3, .clkdiv = 8,
    .tx_fp = 0, .rx_fp = 0, .tx_force_left = 0, .rx_force_left = 0,
};

static float g_mic_audio[DSP_WHISPER_MIC_NUM_SAMPLES];
#if DSP_WHISPER_VAD_ENABLE
static float g_vad_ring[DSP_WHISPER_VAD_PREROLL_SAMPLES];
#endif

static inline int32_t mic_extract(uint32_t slot) { return ((int32_t)slot) >> DSP_WHISPER_MIC_SAMPLE_SHIFT; }
static inline void mic_read_pair(float *s0, float *s1) {
  uint64_t v = read_I2S_rx(DSP_WHISPER_MIC_CHANNEL, I2S_LEFT);
  *s0 = (float)mic_extract((uint32_t)(v & 0xFFFFFFFFu)) * (1.0f / DSP_WHISPER_MIC_FULLSCALE);
  *s1 = (float)mic_extract((uint32_t)(v >> 32)) * (1.0f / DSP_WHISPER_MIC_FULLSCALE);
}

/* Capture one VAD-gated utterance into g_mic_audio (scaled float, DC removed) and return the sample
 * count. Waits for a speech onset (energy >= VAD_THRESHOLD, with pre-roll), then captures until
 * end-of-speech (energy < VAD_END_THRESHOLD for HANGOVER frames) or the window fills — so the
 * encoder runs on just the utterance, not a fixed 2 s window. */
static uint32_t mic_capture(void) {
  const uint32_t N = DSP_WHISPER_MIC_NUM_SAMPLES;
  const uint32_t FR = DSP_WHISPER_VAD_FRAME_SAMPLES;
  uint32_t w = 0u;
#if DSP_WHISPER_VAD_ENABLE
  const uint32_t PRE = DSP_WHISPER_VAD_PREROLL_SAMPLES;
  uint32_t ring_pos = 0u, ring_filled = 0u, frame_ctr = 0u;
  DSP_WHISPER_LOG("[dsp-whisper] mic: listening (VAD thresh=%d/1e6; speak now)...\n",
                  (int)lrintf((float)DSP_WHISPER_VAD_THRESHOLD * 1.0e6f));
  for (;;) {
    float sum = 0.0f, sumsq = 0.0f;
    for (uint32_t i = 0; i < FR; i += 2u) {
      float a, b; mic_read_pair(&a, &b);
      g_vad_ring[ring_pos] = a; ring_pos = (ring_pos + 1u) % PRE;
      g_vad_ring[ring_pos] = b; ring_pos = (ring_pos + 1u) % PRE;
      sum += a + b; sumsq += a * a + b * b;
    }
    if (ring_filled < PRE) { ring_filled += FR; if (ring_filled > PRE) ring_filled = PRE; }
    float mean = sum / (float)FR;
    float energy = (sumsq / (float)FR) - mean * mean;
    if (energy < 0.0f) energy = 0.0f;
#if DSP_WHISPER_VAD_LOG_EVERY
    if ((frame_ctr % DSP_WHISPER_VAD_LOG_EVERY) == 0u)
      DSP_WHISPER_LOG("[dsp-whisper] mic: vad frame=%u energy=%d/1e6\n",
                      (unsigned)frame_ctr, (int)lrintf(energy * 1.0e6f));
#endif
    frame_ctr++;
    if (energy >= (float)DSP_WHISPER_VAD_THRESHOLD) {
      DSP_WHISPER_LOG("[dsp-whisper] mic: onset energy=%d/1e6 -> capturing\n",
                      (int)lrintf(energy * 1.0e6f));
      break;
    }
  }
  uint32_t start = (ring_filled < PRE) ? 0u : ring_pos;
  for (uint32_t k = 0; (k < ring_filled) && (w < N); ++k) g_mic_audio[w++] = g_vad_ring[(start + k) % PRE];
#endif /* VAD */

  /* Capture frames until end-of-speech (or the window fills). */
  uint32_t silence = 0u;
  while (w + FR <= N) {
    float sum = 0.0f, sumsq = 0.0f;
    for (uint32_t i = 0; i < FR; i += 2u) {
      float a, b; mic_read_pair(&a, &b);
      g_mic_audio[w++] = a; g_mic_audio[w++] = b;
      sum += a + b; sumsq += a * a + b * b;
    }
    float mean = sum / (float)FR;
    float energy = (sumsq / (float)FR) - mean * mean;
    if (energy < 0.0f) energy = 0.0f;
#if DSP_WHISPER_VAD_END_ENABLE
    if (energy < (float)DSP_WHISPER_VAD_END_THRESHOLD) {
      if (++silence >= DSP_WHISPER_VAD_HANGOVER_FRAMES && w >= DSP_WHISPER_VAD_MIN_SAMPLES) break;
    } else {
      silence = 0u;
    }
#endif
  }
#if DSP_WHISPER_VAD_END_ENABLE
  /* Drop most of the trailing silence (keep ~1 frame of tail). */
  if (silence > 1u) { uint32_t trim = (silence - 1u) * FR; if (trim < w) w -= trim; }
#endif

  /* DC removal over the captured span. */
  float dsum = 0.0f; for (uint32_t i = 0; i < w; ++i) dsum += g_mic_audio[i];
  float dc = (w ? dsum / (float)w : 0.0f), absmean = 0.0f;
  for (uint32_t i = 0; i < w; ++i) { float x = g_mic_audio[i] - dc; g_mic_audio[i] = x; absmean += x < 0 ? -x : x; }
  absmean = w ? absmean / (float)w : 0.0f;
  DSP_WHISPER_LOG("[dsp-whisper] mic: captured %u samples (%u ms) absmean=%d/1e6\n",
                  (unsigned)w, (unsigned)(w / 16u), (int)lrintf(absmean * 1.0e6f));
  return w;
}
#endif /* DSP_WHISPER_USE_MIC */

/* ------------------------------------------------------------------------------- entry --------- */
void app_init(void) {
#if DSP_WHISPER_USE_MIC
  /* init_test sets up the PLL (-> target_frequency) AND the UART divisor for that clock. It MUST run
   * before any printf on silicon — the console UART is not usable until then (a printf to it would
   * hang the core). Mirrors dsp-kws-rolling app_init ordering. */
  init_test(target_frequency);
#endif
  DSP_WHISPER_LOG("[dsp-whisper] Whisper tiny.en on DSP 25\n");

  /* Start the second hart HERE — right after PLL/UART are up, before any peripheral (I2S) setup —
   * matching the proven rvv-matmul-threadlib ordering (init_test -> hthread_init). A print bookends
   * it so a hang in secondary-hart bringup is visible. No-op unless built WHISPER_DUALCORE=1. */
  whisper_dualcore_init();
  DSP_WHISPER_LOG("[dsp-whisper] runtime up (hart 1 dispatched)\n");

#if DSP_WHISPER_DMA_PROBE
  dma_feasibility_probe();
#endif

#if DSP_WHISPER_USE_MIC
  config_I2S(DSP_WHISPER_MIC_CHANNEL, &g_i2s_params_mic);
  set_I2S_sample_freq(DSP_WHISPER_MIC_CHANNEL, (uint64_t)target_frequency,
                      (uint64_t)DSP_WHISPER_MIC_SAMPLE_RATE_HZ, (uint8_t)DSP_WHISPER_MIC_BITDEPTH);
  DSP_WHISPER_LOG("[dsp-whisper] I2S mic configured ch=%d rate=%uHz\n",
                  DSP_WHISPER_MIC_CHANNEL, (unsigned)DSP_WHISPER_MIC_SAMPLE_RATE_HZ);
#endif
}

int app_main(void) {
#if DSP_WHISPER_USE_MIC
  if (whisper_model_load(whisper_model_blob, &g_model) != 0) { DSP_WHISPER_LOG("[dsp-whisper] bad model\n"); return 1; }
  whisper_frontend_init();
  DSP_WHISPER_LOG("[dsp-whisper] ready — speak a short phrase at the mic.\n");
  for (;;) {
    uint32_t w = mic_capture();
    transcribe_audio(g_mic_audio, (int)w);
  }
  /* not reached */
#elif DSP_WHISPER_FROM_AUDIO
  int fe = frontend_check();
  if (whisper_model_load(whisper_model_blob, &g_model) != 0) { DSP_WHISPER_LOG("[dsp-whisper] bad model\n"); return 1; }
  whisper_frontend_init();
  int n = (int)((whisper_audio_blob_end - whisper_audio_blob) / sizeof(float));
  transcribe_audio((const float *)whisper_audio_blob, n);
  return fe ? 0 : 1;
#else
  int fe = frontend_check();
  int n_frames = (int)((whisper_testcase_blob_end - whisper_testcase_blob) / (WF_N_MELS * sizeof(float)));
  int wr = whisper_run_validate(whisper_model_blob, (const float *)whisper_testcase_blob, n_frames);
  return (fe && wr == 0) ? 0 : 1;
#endif
}

int main(void) {
  app_init();
  return app_main();
}
