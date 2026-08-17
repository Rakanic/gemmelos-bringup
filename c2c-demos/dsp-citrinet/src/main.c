/*
 * dsp-citrinet — NeMo Citrinet-256 CTC speech-to-text on DSP 25. Scalar pipeline.
 * Three run modes (see citrinet_config.h):
 *
 *   default                 : transcribe the embedded audio blob and VALIDATE against the golden
 *                             (per-stage fingerprints + CTC token ids). Spike-testable, no I2S.
 *   DSP_CITRINET_FROM_AUDIO : transcribe the embedded audio blob (no compare). Spike-testable.
 *   DSP_CITRINET_USE_MIC    : capture live I2S mic audio (VAD-gated) and transcribe in a loop.
 *                             Real silicon only (Spike has no I2S). Build PLATFORM=CHIP.
 *   DSP_CITRINET_C2C        : mic mode, plus forward each transcript over the C2C link to Bearly
 *                             ML 25 and print the answer SmolLM sends back. Built as the separate
 *                             target `dsp-citrinet-c2c`; see that directory's CMakeLists.
 *
 * Build (Spike):  make build CHIP=dsp25 PLATFORM=SIMS TARGET=dsp-citrinet EXTRA_CMAKE_ARGS="-DLINKER=llm"
 * Build (mic):    make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-citrinet EXTRA_CMAKE_ARGS="-DLINKER=llm -DDSP_CITRINET_USE_MIC=ON"
 */
#include <stdint.h>
#include <math.h>

#include "main.h"
#include "citrinet.h"
#if DSP_CITRINET_C2C
#include "stt_link_dsp.h"
#include "wake_gate.h"
#endif
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

/* The transcript as text. Filled on every transcription (not just in C2C mode) so there is exactly
 * one detokenization path, and static because this platform's stack overflows silently. */
static char g_transcript[DSP_CITRINET_TRANSCRIPT_MAX];
static int  g_last_enc_frames;

/* Returns the transcript text (also printed). Callers that only want the console ignore it. */
static const char *transcribe_audio(const float *audio, int n_samples) {
  cn_profile_reset();
  uint64_t t0 = cn_rdcycle();
  int n_out = 0;
  int n_enc = cn_transcribe(&g_model, audio, n_samples, g_toks, DSP_CITRINET_MAX_TOKENS, &n_out);
  uint64_t t1 = cn_rdcycle();
  g_last_enc_frames = n_enc;
  cn_tokens_to_text(g_transcript, (int)sizeof(g_transcript), g_toks, n_out);
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
  return g_transcript;
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
#if DSP_CITRINET_MIC_STEREO
/* Second mic kept in its own buffer rather than combined on the fly: the combine needs the whole
 * utterance anyway (per-mic DC removal, and a cross-correlation lag estimate in mode 3), and
 * keeping them separate is what makes a dead second mic visible in the per-mic stats instead of
 * silently halving the level. */
static float g_mic_aux[DSP_CITRINET_MIC_NUM_SAMPLES];
#if DSP_CITRINET_VAD_ENABLE
static float g_vad_ring_aux[DSP_CITRINET_VAD_PREROLL_SAMPLES];
#endif
/* Set by mic_probe_right() at boot; 0 = right slot never filled, so run mono. */
static int g_stereo_ok = 0;
#endif

static inline int32_t mic_extract(uint32_t slot) { return ((int32_t)slot) >> DSP_CITRINET_MIC_SAMPLE_SHIFT; }

/* One 64-bit FIFO block = two CONSECUTIVE samples of one slot (two 32-bit samples packed per the
 * HAL's "samples are packed into a 64 bit read"), so one left block plus one right block yields two
 * time-aligned stereo pairs. */
typedef struct { float m0, m1, a0, a1; } mic_pair_t;

static inline mic_pair_t mic_read_pair(void) {
  mic_pair_t p;
  uint64_t v = read_I2S_rx(DSP_CITRINET_MIC_CHANNEL, I2S_LEFT);
  p.m0 = (float)mic_extract((uint32_t)(v & 0xFFFFFFFFu)) * (1.0f / DSP_CITRINET_MIC_FULLSCALE);
  p.m1 = (float)mic_extract((uint32_t)(v >> 32)) * (1.0f / DSP_CITRINET_MIC_FULLSCALE);
  p.a0 = p.a1 = 0.0f;
#if DSP_CITRINET_MIC_STEREO
  /* Exactly ONE right read per left read, always in this order. The two slots have independent
   * FIFOs and read_I2S_rx blocks on empty, so a skipped or doubled read on either side offsets them
   * against each other PERMANENTLY — and a fixed inter-mic sample offset is precisely what destroys
   * the phase relationship every combine mode depends on. Never make either read conditional on
   * anything that can differ between the two sides. */
  if (g_stereo_ok) {
    uint64_t u = read_I2S_rx(DSP_CITRINET_MIC_CHANNEL, I2S_RIGHT);
    p.a0 = (float)mic_extract((uint32_t)(u & 0xFFFFFFFFu)) * (1.0f / DSP_CITRINET_MIC_FULLSCALE);
    p.a1 = (float)mic_extract((uint32_t)(u >> 32)) * (1.0f / DSP_CITRINET_MIC_FULLSCALE);
  }
#endif
  return p;
}

#if DSP_CITRINET_MIC_STEREO
/* Decide whether the right slot is actually being filled, WITHOUT calling read_I2S_rx — that spins
 * forever on an empty FIFO, so probing with it would turn "second mic not wired" into a silent hang
 * with no console output, which on this platform is indistinguishable from a dead chip. */
static int mic_probe_right(void) {
  for (uint32_t i = 0; i < DSP_CITRINET_MIC_PROBE_SPINS; i++) {
    if (!get_I2S_rx_empty(DSP_CITRINET_MIC_CHANNEL, I2S_RIGHT)) return 1;
  }
  return 0;
}

/* Remove each mic's OWN DC offset before combining. The MEMS mics sit on a large, per-part DC
 * offset (~-98e6 counts was measured on this rig), so combining first would either add two
 * unrelated offsets (sum) or leave their difference as a constant pedestal (difference) — in the
 * difference case a mismatch of a few percent swamps the speech entirely. */
static void mic_remove_dc(float *x, uint32_t n, float *absmean_out) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < n; ++i) sum += x[i];
  const float dc = n ? sum / (float)n : 0.0f;
  float am = 0.0f;
  for (uint32_t i = 0; i < n; ++i) { float v = x[i] - dc; x[i] = v; am += v < 0.0f ? -v : v; }
  *absmean_out = n ? am / (float)n : 0.0f;
}

#if DSP_CITRINET_MIC_COMBINE == 3
/* Integer-sample inter-mic delay by peak cross-correlation over a window from the middle of the
 * utterance (the loudest part, and it avoids the pre-roll and the trailing tail). Returns the lag d
 * that maximizes sum(m[i] * a[i+d]). */
static int mic_estimate_lag(const float *m, const float *a, uint32_t n) {
  const int L = DSP_CITRINET_MIC_MAX_LAG;
  uint32_t win = 8192u;
  if (win > n) win = n;
  if (win <= (uint32_t)(2 * L + 64)) return 0;          /* too short to estimate anything */
  const uint32_t off = (n - win) / 2u;
  const float *x = m + off, *y = a + off;
  int best = 0; float bestv = -1e30f;
  for (int d = -L; d <= L; d++) {
    float acc = 0.0f;
    for (uint32_t i = (uint32_t)L; i + (uint32_t)L < win; i++) acc += x[i] * y[(int)i + d];
    if (acc > bestv) { bestv = acc; best = d; }
  }
  return best;
}
#endif

/* Fold the two mics into g_mic_audio in place. Returns the lag used (0 unless mode 3). */
static int mic_combine(uint32_t w, float *am_l, float *am_r) {
  mic_remove_dc(g_mic_audio, w, am_l);
  mic_remove_dc(g_mic_aux, w, am_r);
  int lag = 0;
  if (!g_stereo_ok) return 0;                            /* mono fallback: primary already clean */
#if DSP_CITRINET_MIC_COMBINE == 1
  for (uint32_t i = 0; i < w; ++i) g_mic_audio[i] = 0.5f * (g_mic_audio[i] + g_mic_aux[i]);
#elif DSP_CITRINET_MIC_COMBINE == 2
  for (uint32_t i = 0; i < w; ++i) g_mic_audio[i] = g_mic_audio[i] - g_mic_aux[i];
#elif DSP_CITRINET_MIC_COMBINE == 3
  lag = mic_estimate_lag(g_mic_audio, g_mic_aux, w);
  for (uint32_t i = 0; i < w; ++i) {
    const int j = (int)i + lag;
    const float av = ((j >= 0) && ((uint32_t)j < w)) ? g_mic_aux[j] : g_mic_aux[i];
    g_mic_audio[i] = 0.5f * (g_mic_audio[i] + av);
  }
#endif
  /* mode 0 leaves g_mic_audio as the primary mic — the A/B control. */
  return lag;
}
#endif /* DSP_CITRINET_MIC_STEREO */

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
      mic_pair_t p = mic_read_pair();
      const float a = p.m0, b = p.m1;
#if DSP_CITRINET_MIC_STEREO
      /* The aux pre-roll ring must track the primary one sample for sample — otherwise the first
       * PREROLL samples of the utterance would combine a real mic against a stale/zero buffer. */
      g_vad_ring_aux[ring_pos] = p.a0;
#endif
      g_vad_ring[ring_pos] = a; ring_pos = (ring_pos + 1u) % PRE;
#if DSP_CITRINET_MIC_STEREO
      g_vad_ring_aux[ring_pos] = p.a1;
#endif
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
  for (uint32_t k = 0; (k < ring_filled) && (w < N); ++k) {
#if DSP_CITRINET_MIC_STEREO
    g_mic_aux[w] = g_vad_ring_aux[(start + k) % PRE];
#endif
    g_mic_audio[w++] = g_vad_ring[(start + k) % PRE];
  }
#endif /* VAD */

  uint32_t silence = 0u;
  while (w + FR <= N) {
    float sum = 0.0f, sumsq = 0.0f;
    for (uint32_t i = 0; i < FR; i += 2u) {
      mic_pair_t p = mic_read_pair();
      const float a = p.m0, b = p.m1;
#if DSP_CITRINET_MIC_STEREO
      g_mic_aux[w] = p.a0; g_mic_aux[w + 1u] = p.a1;
#endif
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
  /* The loop only exits after HANGOVER_FRAMES of silence, so `w` currently ends ~800 ms past the
   * last speech frame. Give back all of that except TAIL_FRAMES, so the recording keeps a short
   * run-out instead of stopping the instant energy dropped. Keeping a tail is what lets
   * END_THRESHOLD be raised without clipping quiet word endings, since a trailing unvoiced
   * consonant lands inside the kept run rather than after it.
   * `silence` can be < TAIL_FRAMES when the capture buffer filled first — then there is nothing to
   * trim and the whole thing is kept, which is the right answer. */
  _Static_assert(DSP_CITRINET_VAD_TAIL_FRAMES <= DSP_CITRINET_VAD_HANGOVER_FRAMES,
                 "VAD tail longer than the hangover: the capture never records that far past "
                 "speech, so the tail would be silently truncated to the hangover length");
  const uint32_t TAIL_FR = (uint32_t)DSP_CITRINET_VAD_TAIL_FRAMES;
  if (silence > TAIL_FR) { uint32_t trim = (silence - TAIL_FR) * FR; if (trim < w) w -= trim; }
#endif

#if DSP_CITRINET_MIC_STEREO
  /* Per-mic DC removal, then fold the array down to one signal. The per-mic absmeans are printed
   * below: a second mic that is dead, unwired or SEL-strapped to the wrong slot shows up there as a
   * near-zero or wildly mismatched level, which is the failure you would otherwise only notice as
   * "the transcript got worse". */
  float am_l = 0.0f, am_r = 0.0f;
  const int lag = mic_combine(w, &am_l, &am_r);
  float absmean = 0.0f;
  for (uint32_t i = 0; i < w; ++i) { float v = g_mic_audio[i]; absmean += v < 0.0f ? -v : v; }
  absmean = w ? absmean / (float)w : 0.0f;
  DSP_CITRINET_LOG("[dsp-citrinet] mic: array mode=%d L=%d/1e6 R=%d/1e6 lag=%d%s\n",
                   (int)DSP_CITRINET_MIC_COMBINE, (int)lrintf(am_l * 1.0e6f),
                   (int)lrintf(am_r * 1.0e6f), lag, g_stereo_ok ? "" : " (MONO fallback)");
#else
  float dsum = 0.0f; for (uint32_t i = 0; i < w; ++i) dsum += g_mic_audio[i];
  float dc = (w ? dsum / (float)w : 0.0f), absmean = 0.0f;
  for (uint32_t i = 0; i < w; ++i) { float x = g_mic_audio[i] - dc; g_mic_audio[i] = x; absmean += x < 0 ? -x : x; }
  absmean = w ? absmean / (float)w : 0.0f;
#endif
  DSP_CITRINET_LOG("[dsp-citrinet] mic: captured %u samples (%u ms) absmean=%d/1e6 tail=%u ms%s\n",
                   (unsigned)w, (unsigned)(w / 16u), (int)lrintf(absmean * 1.0e6f),
#if DSP_CITRINET_VAD_END_ENABLE
                   /* What was actually KEPT after speech ended — < TAIL_MS means the buffer filled
                    * before the hangover elapsed, i.e. you were still talking when it ran out. */
                   (unsigned)(((silence < TAIL_FR ? silence : TAIL_FR) * FR) / 16u),
                   (silence < TAIL_FR) ? " (buffer full — speech may be cut)" : "");
#else
                   0u, " (end-detect off)");
#endif
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
#if DSP_CITRINET_MIC_STEREO
  /* Both mics live on THIS channel's two slots — no second config_I2S call, and no second clkdiv:
   * set_I2S_sample_freq already framed the link as stereo (mclk = rate * bits * 2). */
  g_stereo_ok = mic_probe_right();
  DSP_CITRINET_LOG("[dsp-citrinet] mic array: right slot %s (combine mode=%d)\n",
                   g_stereo_ok ? "ALIVE" : "SILENT -> falling back to a single mic",
                   (int)DSP_CITRINET_MIC_COMBINE);
#endif
#endif

#if DSP_CITRINET_C2C
  /* MFCC front-end + TinySpeech runtime for the wake gate. Local memory and a local peripheral
   * only — no cross-link traffic, so app_init is the right place (see the boot rule in CLAUDE.md). */
  if (wake_gate_init() != 0) {
    DSP_CITRINET_LOG("[dsp-citrinet] wake gate init FAILED — cannot listen\n");
    while (1) { __asm__ volatile("wfi"); }
  }
#endif
}

int app_main(void) {
#if DSP_CITRINET_C2C
  /* Voice assistant: mic -> Citrinet -> C2C link -> SmolLM on Bearly ML 25 -> answer back here.
   * The model is loaded BEFORE the link is brought up so the boot barrier is the only thing the
   * DSP waits on afterwards, and so a bad blob is reported before the other chip is involved. */
  if (cn_model_load(citrinet_model_blob, CITRINET_BLOB_BYTES, &g_model) != 0) { DSP_CITRINET_LOG("[dsp-citrinet] bad model\n"); return 1; }
  stt_dsp_link_init();   /* blocks until Bearly ML 25 has finished booting SmolLM */
  DSP_CITRINET_LOG("[dsp-citrinet] ready — say the wake word, then ask a question.\n");
  for (;;) {
    /* Always-on wake gate. stt_dsp_poll is handed in so the link keeps making progress while we
     * listen: an answer to an earlier question lands on this console mid-listen, and a question
     * asked now REPLACES one Bearly is still working on. That is the whole reason the gate runs
     * here rather than the DSP blocking on the answer. */
    wake_gate_listen(stt_dsp_poll);

    {
      uint32_t w = mic_capture();
      const char *text = transcribe_audio(g_mic_audio, (int)w);
      if ((text == NULL) || (text[0] == '\0')) {
        /* Citrinet heard no words after the wake word (a cough, a door, a VAD false trigger).
         * Sending an empty prompt would cost minutes of SmolLM decode on nothing. */
        DSP_CITRINET_LOG("[dsp-citrinet] empty transcript — back to listening\n");
        continue;
      }
      stt_dsp_publish_prompt(text, (uint32_t)g_last_enc_frames, w);
    }
  }
  /* not reached */
#elif DSP_CITRINET_USE_MIC
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
