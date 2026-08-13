/*
 * dsp-citrinet — NeMo Citrinet-256 CTC speech-to-text on DSP 25. Scalar pipeline.
 * Three run modes (see citrinet_config.h):
 *
 *   default                 : transcribe the embedded audio blob and VALIDATE against the golden
 *                             (per-stage fingerprints + CTC token ids). Spike-testable, no I2S.
 *   DSP_CITRINET_FROM_AUDIO : transcribe the embedded audio blob (no compare). Spike-testable.
 *   DSP_CITRINET_USE_MIC    : capture live I2S mic audio (VAD-gated) and transcribe in a loop.
 *                             Real silicon only (Spike has no I2S). Build PLATFORM=CHIP.
 *
 * Build (Spike):  make build CHIP=dsp25 PLATFORM=SIMS TARGET=dsp-citrinet EXTRA_CMAKE_ARGS="-DLINKER=llm"
 * Build (mic):    make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-citrinet EXTRA_CMAKE_ARGS="-DLINKER=llm -DDSP_CITRINET_USE_MIC=ON"
 */
#include <stdint.h>
#include <math.h>

#include "main.h"
#include "citrinet.h"
#if CN_CONV1D
#include "hal_conv.h"
#endif

#if DSP_CITRINET_USE_MIC
#include "rocketcore.h"
#include "hal_i2s.h"
#endif

/* Embedded blobs from model_data.S */
extern const unsigned char citrinet_model_blob[];
extern const unsigned char citrinet_model_blob_end[];
extern const unsigned char citrinet_audio_blob[];
extern const unsigned char citrinet_audio_blob_end[];

/* Size from the LINKER, never from the blob header: it bounds the integrity check, so it must
 * not itself be read from the region being checked (a stale first read of freshly-TSI-loaded
 * DRAM is exactly the failure this guards against). */
#define CITRINET_BLOB_BYTES ((size_t)(citrinet_model_blob_end - citrinet_model_blob))
#define CITRINET_AUDIO_BYTES ((size_t)(citrinet_audio_blob_end - citrinet_audio_blob))

/* The MODEL blob carries its own checksum in its header and the .text image is hashed at load, but
 * the embedded AUDIO blob had neither — 191 KB that fully determines the transcript, with nothing
 * to say whether it arrived intact. A load that corrupts it produces a plausible-but-different
 * transcript and no complaint anywhere, which reads exactly like a flaky chip. Print its
 * fingerprint; text_checksum.py prints the host-side value to compare against, and it must also be
 * identical across two flashes of the same ELF. */
#if !DSP_CITRINET_USE_MIC
static void report_audio_checksum(void) {
  DSP_CITRINET_LOG("[dsp-citrinet] audio: %lu bytes, fnv1a64=0x%016lx  <- must match across flashes\n",
                   (unsigned long)CITRINET_AUDIO_BYTES,
                   (unsigned long)cn_fnv1a64_bytes(citrinet_audio_blob, CITRINET_AUDIO_BYTES));
}
#endif

static inline uint64_t cn_rdcycle(void) { uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }

static cn_model_t g_model;
static int g_toks[DSP_CITRINET_MAX_TOKENS];

static void transcribe_audio(const float *audio, int n_samples) {
  cn_profile_reset();
  uint64_t t0 = cn_rdcycle();
  int n_out = 0;
  int n_enc = cn_transcribe(&g_model, audio, n_samples, g_toks, DSP_CITRINET_MAX_TOKENS, &n_out);
  uint64_t t1 = cn_rdcycle();
  DSP_CITRINET_LOG("[dsp-citrinet] samples=%d enc_frames=%d tokens=%d engine=%lu cycles\n",
                   n_samples, n_enc, n_out, (unsigned long)(t1 - t0));
  cn_profile_report();
  cn_print_tokens_text("[dsp-citrinet] transcript: ", g_toks, n_out);
  uint64_t cyc = t1 - t0;
  uint64_t ms = cyc / (uint64_t)(DSP_CITRINET_TARGET_FREQ_HZ / 1000u);
  DSP_CITRINET_LOG("[dsp-citrinet] LATENCY audio->transcript: %lu ms  (%lu cycles @ %u MHz)\n",
                   (unsigned long)ms, (unsigned long)cyc,
                   (unsigned)(DSP_CITRINET_TARGET_FREQ_HZ / 1000000u));
  cn_stack_report();
}

/* ------------------------------------------------------------------------- I2S mic capture ------ */
#if DSP_CITRINET_USE_MIC
uint64_t target_frequency = DSP_CITRINET_TARGET_FREQ_HZ;

static i2s_params_t g_i2s_params_mic = {
    .tx_en = 1, .rx_en = 1, .bitdepth_tx = I2S_BITDEPTH_32, .bitdepth_rx = I2S_BITDEPTH_32,
    .clkgen = 1, .dacen = 0, .ws_len = 3, .clkdiv = 8,
    .tx_fp = 0, .rx_fp = 0, .tx_force_left = 0, .rx_force_left = 0,
};

static float g_mic_audio[DSP_CITRINET_MIC_NUM_SAMPLES];
#if DSP_CITRINET_VAD_ENABLE
static float g_vad_ring[DSP_CITRINET_VAD_PREROLL_SAMPLES];
#endif

static inline int32_t mic_extract(uint32_t slot) { return ((int32_t)slot) >> DSP_CITRINET_MIC_SAMPLE_SHIFT; }
static inline void mic_read_pair(float *s0, float *s1) {
  uint64_t v = read_I2S_rx(DSP_CITRINET_MIC_CHANNEL, I2S_LEFT);
  *s0 = (float)mic_extract((uint32_t)(v & 0xFFFFFFFFu)) * (1.0f / DSP_CITRINET_MIC_FULLSCALE);
  *s1 = (float)mic_extract((uint32_t)(v >> 32)) * (1.0f / DSP_CITRINET_MIC_FULLSCALE);
}

/* Capture one VAD-gated utterance into g_mic_audio (scaled float, DC removed); return sample count.
 * Mirrors dsp-moonshine mic_capture exactly (proven on silicon). */
static uint32_t mic_capture(void) {
  const uint32_t N = DSP_CITRINET_MIC_NUM_SAMPLES;
  const uint32_t FR = DSP_CITRINET_VAD_FRAME_SAMPLES;
  uint32_t w = 0u;
#if DSP_CITRINET_VAD_ENABLE
  const uint32_t PRE = DSP_CITRINET_VAD_PREROLL_SAMPLES;
  uint32_t ring_pos = 0u, ring_filled = 0u, frame_ctr = 0u;
  DSP_CITRINET_LOG("[dsp-citrinet] mic: listening (VAD thresh=%d/1e6; speak now)...\n",
                   (int)lrintf((float)DSP_CITRINET_VAD_THRESHOLD * 1.0e6f));
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
#if DSP_CITRINET_VAD_LOG_EVERY
    if ((frame_ctr % DSP_CITRINET_VAD_LOG_EVERY) == 0u)
      DSP_CITRINET_LOG("[dsp-citrinet] mic: vad frame=%u energy=%d/1e6\n",
                       (unsigned)frame_ctr, (int)lrintf(energy * 1.0e6f));
#endif
    frame_ctr++;
    if (energy >= (float)DSP_CITRINET_VAD_THRESHOLD) {
      DSP_CITRINET_LOG("[dsp-citrinet] mic: onset energy=%d/1e6 -> capturing\n",
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
#if DSP_CITRINET_VAD_END_ENABLE
    if (energy < (float)DSP_CITRINET_VAD_END_THRESHOLD) {
      if (++silence >= DSP_CITRINET_VAD_HANGOVER_FRAMES && w >= DSP_CITRINET_VAD_MIN_SAMPLES) break;
    } else {
      silence = 0u;
    }
#endif
  }
#if DSP_CITRINET_VAD_END_ENABLE
  if (silence > 1u) { uint32_t trim = (silence - 1u) * FR; if (trim < w) w -= trim; }
#endif

  float dsum = 0.0f; for (uint32_t i = 0; i < w; ++i) dsum += g_mic_audio[i];
  float dc = (w ? dsum / (float)w : 0.0f), absmean = 0.0f;
  for (uint32_t i = 0; i < w; ++i) { float x = g_mic_audio[i] - dc; g_mic_audio[i] = x; absmean += x < 0 ? -x : x; }
  absmean = w ? absmean / (float)w : 0.0f;
  DSP_CITRINET_LOG("[dsp-citrinet] mic: captured %u samples (%u ms) absmean=%d/1e6\n",
                   (unsigned)w, (unsigned)(w / 16u), (int)lrintf(absmean * 1.0e6f));
  return w;
}
#endif /* DSP_CITRINET_USE_MIC */

/* ------------------------------------------------------------------------------- entry --------- */
void app_init(void) {
#if defined(TERMINAL_DEVICE_UART0)
  /* init_test brings up the PLL AND the UART divisor; it MUST run before any printf on real silicon
   * (a printf to an unconfigured UART hangs the core). Gate it on the CONSOLE, not on the run mode:
   * every PLATFORM=CHIP build needs it, including the validate mode, while under Spike
   * (TERMINAL_DEVICE_HTIF) there is no PLL and calling it faults.
   * PLL ratio = target/SYS_CLK_FREQ = 750/50 = 15; UART DIV = 750e6/115200 - 1 = 6509. */
  init_test((uint64_t)DSP_CITRINET_TARGET_FREQ_HZ);
#endif
  DSP_CITRINET_LOG("[dsp-citrinet] Citrinet-256 CTC on DSP 25\n");
  /* Before anything else: prove the stack is where we think it is and big enough. A 4 KiB default
   * stack against an 18 KB frame was what made this demo non-reproducible on silicon while passing
   * on Spike every time; the paint/report pair makes a recurrence a printed line, not a mystery. */
  cn_stack_paint();
  cn_report_vlen();
  /* Start hart 1 right after the PLL/UART are up, before any peripheral setup — the ordering
   * dsp-whisper documents as proven. No-op in single-core builds. */
  cn_dualcore_init();
  cn_dma_probe();
#if CN_CONV1D
  conv_init();          /* stop / clear / reset the 1D conv engine before first use */
  DSP_CITRINET_LOG("[dsp-citrinet] CONV1D engine initialised (depthwise runs on the accelerator)\n");
#endif
#if CN_MEMTEST_MB
  /* Before trusting any inference result, prove memory round-trips. */
  cn_memtest((size_t)CN_MEMTEST_MB << 20);
#endif
#if defined(TERMINAL_DEVICE_UART0)
  DSP_CITRINET_LOG("[dsp-citrinet] PLL @ %u MHz (ratio %u x %u MHz), UART %u baud\n",
                   (unsigned)(DSP_CITRINET_TARGET_FREQ_HZ / 1000000u),
                   (unsigned)(DSP_CITRINET_TARGET_FREQ_HZ / SYS_CLK_FREQ),
                   (unsigned)(SYS_CLK_FREQ / 1000000u), 115200u);
#endif

#if DSP_CITRINET_USE_MIC
  config_I2S(DSP_CITRINET_MIC_CHANNEL, &g_i2s_params_mic);
  set_I2S_sample_freq(DSP_CITRINET_MIC_CHANNEL, (uint64_t)target_frequency,
                      (uint64_t)DSP_CITRINET_MIC_SAMPLE_RATE_HZ, (uint8_t)DSP_CITRINET_MIC_BITDEPTH);
  DSP_CITRINET_LOG("[dsp-citrinet] I2S mic configured ch=%d rate=%uHz\n",
                   DSP_CITRINET_MIC_CHANNEL, (unsigned)DSP_CITRINET_MIC_SAMPLE_RATE_HZ);
#endif
}

int app_main(void) {
#if DSP_CITRINET_USE_MIC
  if (cn_model_load(citrinet_model_blob, CITRINET_BLOB_BYTES, &g_model) != 0) { DSP_CITRINET_LOG("[dsp-citrinet] bad model\n"); return 1; }
  DSP_CITRINET_LOG("[dsp-citrinet] ready — speak a short phrase at the mic.\n");
  for (;;) {
    uint32_t w = mic_capture();
    transcribe_audio(g_mic_audio, (int)w);
  }
  /* not reached */
#elif DSP_CITRINET_FROM_AUDIO
  if (cn_model_load(citrinet_model_blob, CITRINET_BLOB_BYTES, &g_model) != 0) { DSP_CITRINET_LOG("[dsp-citrinet] bad model\n"); return 1; }
  report_audio_checksum();
  int n = (int)(CITRINET_AUDIO_BYTES / sizeof(float));
  transcribe_audio((const float *)citrinet_audio_blob, n);
  return 0;
#else
  report_audio_checksum();
  int n = (int)(CITRINET_AUDIO_BYTES / sizeof(float));
  return cn_run_validate(citrinet_model_blob, CITRINET_BLOB_BYTES, (const float *)citrinet_audio_blob, n);
#endif
}

int main(void) {
  app_init();
  return app_main();
}
