/*
 * dsp-moonshine — Moonshine tiny speech-to-text on DSP 25 (plan 006). FLOAT+RVV pipeline.
 * Three run modes (see moonshine_config.h):
 *
 *   default                  : transcribe the embedded audio blob and VALIDATE against the golden
 *                              (encoder stage sums + greedy token ids). Spike-testable, no I2S.
 *   DSP_MOONSHINE_FROM_AUDIO  : transcribe the embedded audio blob (no compare). Spike-testable.
 *   DSP_MOONSHINE_USE_MIC     : capture live I2S mic audio (VAD-gated) and transcribe in a loop.
 *                              Real silicon only (Spike has no I2S). Build PLATFORM=CHIP.
 *
 * Build (Spike):  make build CHIP=dsp25 PLATFORM=SIMS TARGET=dsp-moonshine LINKER=llm EXTRA_CMAKE_ARGS="-DMOONSHINE_USE_RVV=ON"
 * Build (mic):    make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-moonshine LINKER=llm EXTRA_CMAKE_ARGS="-DMOONSHINE_USE_RVV=ON -DDSP_MOONSHINE_USE_MIC=ON"
 */
#include <stdint.h>
#include <math.h>

#include "main.h"
#include "moonshine.h"

#if DSP_MOONSHINE_USE_MIC
#include "rocketcore.h"
#include "hal_i2s.h"
#endif

/* Embedded blobs from model_data.S */
extern const unsigned char moonshine_model_blob[];
extern const unsigned char moonshine_audio_blob[];
extern const unsigned char moonshine_audio_blob_end[];

static inline uint64_t ms_rdcycle(void) { uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }

static ms_model_t g_model;
static int g_toks[DSP_MOONSHINE_MAX_TOKENS];

/* audio (n_samples @16 kHz float) -> transcript. guards!=0 enables the mic anti-loop heuristics. */
static void transcribe_audio(const float *audio, int n_samples, int guards) {
  ms_profile_reset();
  uint64_t t0 = ms_rdcycle();
  int n_out = 0;
  int n_pos = ms_transcribe(&g_model, audio, n_samples, g_toks, DSP_MOONSHINE_MAX_TOKENS, guards, &n_out);
  uint64_t t1 = ms_rdcycle();
  DSP_MOONSHINE_LOG("[dsp-moonshine] samples=%d enc_pos=%d tokens=%d engine=%lu cycles\n",
                    n_samples, n_pos, n_out, (unsigned long)(t1 - t0));
  ms_profile_report();
  ms_print_tokens_text("[dsp-moonshine] transcript: ", g_toks, n_out);
  uint64_t cyc = t1 - t0;
  uint64_t ms = cyc / (uint64_t)(DSP_MOONSHINE_TARGET_FREQ_HZ / 1000u);
  DSP_MOONSHINE_LOG("[dsp-moonshine] LATENCY audio->transcript: %lu ms  (%lu cycles @ %u MHz)\n",
                    (unsigned long)ms, (unsigned long)cyc,
                    (unsigned)(DSP_MOONSHINE_TARGET_FREQ_HZ / 1000000u));
}

/* ------------------------------------------------------------------------- I2S mic capture ------ */
#if DSP_MOONSHINE_USE_MIC
uint64_t target_frequency = DSP_MOONSHINE_TARGET_FREQ_HZ;

static i2s_params_t g_i2s_params_mic = {
    .tx_en = 1, .rx_en = 1, .bitdepth_tx = I2S_BITDEPTH_32, .bitdepth_rx = I2S_BITDEPTH_32,
    .clkgen = 1, .dacen = 0, .ws_len = 3, .clkdiv = 8,
    .tx_fp = 0, .rx_fp = 0, .tx_force_left = 0, .rx_force_left = 0,
};

static float g_mic_audio[DSP_MOONSHINE_MIC_NUM_SAMPLES];
#if DSP_MOONSHINE_VAD_ENABLE
static float g_vad_ring[DSP_MOONSHINE_VAD_PREROLL_SAMPLES];
#endif

static inline int32_t mic_extract(uint32_t slot) { return ((int32_t)slot) >> DSP_MOONSHINE_MIC_SAMPLE_SHIFT; }
static inline void mic_read_pair(float *s0, float *s1) {
  uint64_t v = read_I2S_rx(DSP_MOONSHINE_MIC_CHANNEL, I2S_LEFT);
  *s0 = (float)mic_extract((uint32_t)(v & 0xFFFFFFFFu)) * (1.0f / DSP_MOONSHINE_MIC_FULLSCALE);
  *s1 = (float)mic_extract((uint32_t)(v >> 32)) * (1.0f / DSP_MOONSHINE_MIC_FULLSCALE);
}

/* Capture one VAD-gated utterance into g_mic_audio (scaled float, DC removed); return sample count.
 * Mirrors dsp-whisper mic_capture exactly (proven on silicon). */
static uint32_t mic_capture(void) {
  const uint32_t N = DSP_MOONSHINE_MIC_NUM_SAMPLES;
  const uint32_t FR = DSP_MOONSHINE_VAD_FRAME_SAMPLES;
  uint32_t w = 0u;
#if DSP_MOONSHINE_VAD_ENABLE
  const uint32_t PRE = DSP_MOONSHINE_VAD_PREROLL_SAMPLES;
  uint32_t ring_pos = 0u, ring_filled = 0u, frame_ctr = 0u;
  DSP_MOONSHINE_LOG("[dsp-moonshine] mic: listening (VAD thresh=%d/1e6; speak now)...\n",
                    (int)lrintf((float)DSP_MOONSHINE_VAD_THRESHOLD * 1.0e6f));
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
#if DSP_MOONSHINE_VAD_LOG_EVERY
    if ((frame_ctr % DSP_MOONSHINE_VAD_LOG_EVERY) == 0u)
      DSP_MOONSHINE_LOG("[dsp-moonshine] mic: vad frame=%u energy=%d/1e6\n",
                        (unsigned)frame_ctr, (int)lrintf(energy * 1.0e6f));
#endif
    frame_ctr++;
    if (energy >= (float)DSP_MOONSHINE_VAD_THRESHOLD) {
      DSP_MOONSHINE_LOG("[dsp-moonshine] mic: onset energy=%d/1e6 -> capturing\n",
                        (int)lrintf(energy * 1.0e6f));
      break;
    }
  }
  uint32_t start = (ring_filled < PRE) ? 0u : ring_pos;
  for (uint32_t k = 0; (k < ring_filled) && (w < N); ++k) g_mic_audio[w++] = g_vad_ring[(start + k) % PRE];
#endif /* VAD */

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
#if DSP_MOONSHINE_VAD_END_ENABLE
    if (energy < (float)DSP_MOONSHINE_VAD_END_THRESHOLD) {
      if (++silence >= DSP_MOONSHINE_VAD_HANGOVER_FRAMES && w >= DSP_MOONSHINE_VAD_MIN_SAMPLES) break;
    } else {
      silence = 0u;
    }
#endif
  }
#if DSP_MOONSHINE_VAD_END_ENABLE
  if (silence > 1u) { uint32_t trim = (silence - 1u) * FR; if (trim < w) w -= trim; }
#endif

  float dsum = 0.0f; for (uint32_t i = 0; i < w; ++i) dsum += g_mic_audio[i];
  float dc = (w ? dsum / (float)w : 0.0f), absmean = 0.0f;
  for (uint32_t i = 0; i < w; ++i) { float x = g_mic_audio[i] - dc; g_mic_audio[i] = x; absmean += x < 0 ? -x : x; }
  absmean = w ? absmean / (float)w : 0.0f;
  DSP_MOONSHINE_LOG("[dsp-moonshine] mic: captured %u samples (%u ms) absmean=%d/1e6\n",
                    (unsigned)w, (unsigned)(w / 16u), (int)lrintf(absmean * 1.0e6f));
  return w;
}
#endif /* DSP_MOONSHINE_USE_MIC */

/* ------------------------------------------------------------------------------- entry --------- */
void app_init(void) {
#if DSP_MOONSHINE_USE_MIC || DSP_MOONSHINE_FROM_AUDIO
  /* init_test brings up the PLL AND the UART divisor; it MUST run before any printf on real silicon
   * (a printf to an unconfigured UART hangs the core). Only the silicon (CHIP) run modes call it —
   * the default validate mode is Spike-only (HTIF console, no PLL; calling init_test there faults). */
  init_test((uint64_t)DSP_MOONSHINE_TARGET_FREQ_HZ);
#endif
  DSP_MOONSHINE_LOG("[dsp-moonshine] Moonshine tiny on DSP 25\n");

  /* Start hart 1 for the dual-core matmul/conv split (no-op unless built MS_DUALCORE=1). Must run
   * after the PLL/UART are up, before any inference — mirrors the proven rvv-matmul-threadlib order. */
  ms_dualcore_init();

  ms_roofline_report();   /* no-op unless MOONSHINE_ROOFLINE=ON */

#if DSP_MOONSHINE_USE_MIC
  config_I2S(DSP_MOONSHINE_MIC_CHANNEL, &g_i2s_params_mic);
  set_I2S_sample_freq(DSP_MOONSHINE_MIC_CHANNEL, (uint64_t)target_frequency,
                      (uint64_t)DSP_MOONSHINE_MIC_SAMPLE_RATE_HZ, (uint8_t)DSP_MOONSHINE_MIC_BITDEPTH);
  DSP_MOONSHINE_LOG("[dsp-moonshine] I2S mic configured ch=%d rate=%uHz\n",
                    DSP_MOONSHINE_MIC_CHANNEL, (unsigned)DSP_MOONSHINE_MIC_SAMPLE_RATE_HZ);
#endif
}

int app_main(void) {
#if DSP_MOONSHINE_USE_MIC
  if (ms_model_load(moonshine_model_blob, &g_model) != 0) { DSP_MOONSHINE_LOG("[dsp-moonshine] bad model\n"); return 1; }
  DSP_MOONSHINE_LOG("[dsp-moonshine] ready — speak a short phrase at the mic.\n");
  for (;;) {
    uint32_t w = mic_capture();
    transcribe_audio(g_mic_audio, (int)w, 1);
  }
  /* not reached */
#elif DSP_MOONSHINE_FROM_AUDIO
  if (ms_model_load(moonshine_model_blob, &g_model) != 0) { DSP_MOONSHINE_LOG("[dsp-moonshine] bad model\n"); return 1; }
  int n = (int)((moonshine_audio_blob_end - moonshine_audio_blob) / sizeof(float));
#if defined(MS_DUALCORE_DEBUG) && MS_DUALCORE_DEBUG
  /* Deterministic wedge-repro: re-run the IDENTICAL embedded-audio inference forever so an
   * intermittent dual-core wedge reliably surfaces. On a wedge the DC-WEDGE line + this iter number
   * localize it. */
  for (int it = 0; ; it++) {
    DSP_MOONSHINE_LOG("[dsp-moonshine] ===== iter %d =====\n", it);
    transcribe_audio((const float *)moonshine_audio_blob, n, 0);
  }
#else
  transcribe_audio((const float *)moonshine_audio_blob, n, 0);
  return 0;
#endif
#else
  int n = (int)((moonshine_audio_blob_end - moonshine_audio_blob) / sizeof(float));
  return ms_run_validate(moonshine_model_blob, (const float *)moonshine_audio_blob, n);
#endif
}

int main(void) {
  app_init();
  return app_main();
}
