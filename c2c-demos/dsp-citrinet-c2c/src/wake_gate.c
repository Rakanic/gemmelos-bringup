/*
 * "Hey Marvin" wake-word gate — DSP 25.
 *
 * SHAPE: this is dsp-kws-rolling's capture path with a classifier bolted on the end, and that is
 * deliberate. Monitor cheap short frames for an energy onset, keeping a pre-roll ring; when
 * something is said, capture one ~1 s window, compute its 94 MFCC frames, run the CNN once, and go
 * back to monitoring. Only if the CNN says `marvin` does the demo spend anything further (Citrinet
 * transcription, then the C2C link to Bearly).
 *
 * WHY NOT A CONTINUOUSLY SLIDING WINDOW. The first version of this file slid a 1 s window forward
 * every 160 ms and ran the CNN on every step. That created a real-time budget the KWS demo never
 * had — it never runs a CNN on this chip at all, it streams features to Bearly — and on silicon the
 * scalar inference took 151% of the hop, so the microphone kept running while the core was busy and
 * the "1 second window" quietly became a second of audio with holes punched through it. A gated
 * one-shot capture has no such budget: nothing is being tracked while the CNN runs, so there is
 * nothing to fall behind.
 *
 * It is also more ACCURATE, not just cheaper. The pre-roll makes the window start ~200 ms before
 * the onset, which is roughly where the word sits in a Speech Commands clip — so the one look this
 * takes is a well-aligned look, whereas a blind sliding window mostly produces badly-centred ones
 * and the model was only trained with +-100 ms of position jitter.
 *
 * TWO THINGS ABOUT THE INFERENCE, both learned on silicon:
 *   1. The kernels must be built SCALAR. The RVV TinySpeech kernels fail the golden self-test on
 *      this chip — every case picked the same class with the same ranking and 10-100x magnitudes.
 *      Scalar reproduces the host bit-exactly (max_diff 0). See the CMakeLists.
 *   2. The FLOAT pipeline is used, not int8, because the int8 conv2 kernel also returns garbage
 *      here (the RVV integer-reduction defect — see /CLAUDE.md).
 */

#include <stdint.h>
#include <math.h>
#include <string.h>

#include "main.h"              /* citrinet_config.h: DSP_CITRINET_LOG, mic constants */
#include "wake_config.h"
#include "wake_gate.h"

#include "mfcc_driver.h"
#include "tensor.h"
#include "tinyspeech_model.h"

#include "rocketcore.h"
#include "hal_i2s.h"

_Static_assert(WAKE_NUM_CLASSES == TINYSPEECH_NUM_CLASSES,
               "the wake model's class count must match the TinySpeech runtime's");
_Static_assert(WAKE_MFCC_DIM <= MFCC_DRIVER_NUM_DCT,
               "the model wants more MFCC coefficients than the driver produces");
_Static_assert((DSP_WAKE_VAD_FRAME_SAMPLES % 2u) == 0u,
               "the VAD frame must be even — the I2S FIFO is read in sample PAIRS");
_Static_assert((DSP_WAKE_PREROLL_SAMPLES % DSP_WAKE_VAD_FRAME_SAMPLES) == 0u,
               "the pre-roll must be a whole number of VAD frames");
_Static_assert(DSP_WAKE_PREROLL_SAMPLES < WAKE_WINDOW_SAMPLES,
               "the pre-roll must be shorter than the capture window");

/* All static: multi-KB objects on the stack are how this platform produces silent, irreproducible
 * corruption (see the LINKER=llm entry in /CLAUDE.md). */
static float32_t g_audio[WAKE_WINDOW_SAMPLES];
static float32_t g_preroll[DSP_WAKE_PREROLL_SAMPLES];
static float32_t g_frames[WAKE_FRAMES * WAKE_MFCC_DIM];   /* frame-major: [frame][coeff] */
static int8_t    g_case[WAKE_FRAMES * WAKE_MFCC_DIM];     /* coeff-major, the model's layout */
static float32_t g_fft_in[MFCC_DRIVER_FFT_LEN];
static mfcc_driver_t g_mfcc;

static uint64_t g_last_cycles;
static float    g_last_margin;
static float32_t g_last_amax;
static uint32_t g_events;      /* onsets that reached the classifier */

/* Running DC estimate of the microphone.
 *
 * The MEMS mics sit on a large per-part DC offset, and dsp-citrinet's capture path removes it before
 * its own MFCC — training clips are WAV files with no DC at all. Estimated during monitoring (which
 * is mostly silence, so the mean IS the offset) and subtracted from the captured window.
 *
 * Honest scope note: this was first SUSPECTED as the cause of the -85..-119 margins seen on silicon
 * and it was NOT — injecting up to 20x full-scale DC on the host moves the margin only within
 * [-6, +5]. The real cause was the RVV kernels. Kept because it is correct, not because it fixed
 * that. */
static float32_t g_dc;

static inline uint64_t rdcycle64(void) {
  uint64_t x;
  __asm__ volatile("rdcycle %0" : "=r"(x));
  return x;
}

uint64_t wake_gate_last_cycles(void) { return g_last_cycles; }
float    wake_gate_last_margin(void) { return g_last_margin; }

/* One 64-bit RX block = two consecutive samples of the primary mic, scaled to ~[-1,1]. Identical
 * arithmetic to dsp-citrinet's capture path (same config macros), so the wake window and the
 * command utterance that follows it are on the same scale. */
static inline void mic_read_pair(float32_t *a, float32_t *b) {
  const uint64_t v = read_I2S_rx(DSP_WAKE_MIC_CHANNEL, I2S_LEFT);
  *a = (float32_t)(((int32_t)(uint32_t)(v & 0xFFFFFFFFu)) >> DSP_CITRINET_MIC_SAMPLE_SHIFT) *
       (1.0f / DSP_CITRINET_MIC_FULLSCALE);
  *b = (float32_t)(((int32_t)(uint32_t)(v >> 32)) >> DSP_CITRINET_MIC_SAMPLE_SHIFT) *
       (1.0f / DSP_CITRINET_MIC_FULLSCALE);
}

/* MFCC for one frame of the captured window into g_frames[frame_idx]. */
static void compute_frame(uint32_t frame_idx) {
  float32_t out[MFCC_DRIVER_NUM_DCT];
  uint64_t cycles = 0;
  const uint32_t start = frame_idx * WAKE_FRAME_HOP;
  mfcc_driver_status_t st;

  for (uint32_t n = 0; n < MFCC_DRIVER_FFT_LEN; ++n) {
    const uint32_t idx = start + n;
    g_fft_in[n] = (idx < WAKE_WINDOW_SAMPLES) ? g_audio[idx] : 0.0f;
  }

  st = mfcc_driver_run_sp1024x23x12_f32(&g_mfcc, g_fft_in, out, &cycles);
  if (st != MFCC_DRIVER_OK) {
    st = mfcc_driver_run_f32(&g_mfcc, g_fft_in, out, &cycles);
  }
  for (uint32_t k = 0; k < WAKE_MFCC_DIM; ++k) {
    g_frames[(frame_idx * WAKE_MFCC_DIM) + k] = (st == MFCC_DRIVER_OK) ? out[k] : 0.0f;
  }
}

/* Quantize the frame array into the model's input layout.
 *
 * Two things must match training exactly or the CNN is being shown something else entirely:
 * COEFFICIENT-MAJOR order (the input is {1,1,H=12,W=94}, so element (coeff k, frame f) lives at
 * k*94 + f — writing it frame-major transposes the features), and PER-CASE PEAK quantization
 * (q = round(x * 127/max|x|) over the whole map), which is what _quantize_like_runtime does in the
 * training script. */
static void quantize_case(void) {
  float32_t amax = 0.0f;
  for (uint32_t i = 0; i < (WAKE_FRAMES * WAKE_MFCC_DIM); ++i) {
    const float32_t a = (g_frames[i] < 0.0f) ? -g_frames[i] : g_frames[i];
    if (a > amax) {
      amax = a;
    }
  }
  g_last_amax = amax;
  const float32_t scale = (amax < 1e-12f) ? 0.0f : (127.0f / amax);

  for (uint32_t f = 0; f < WAKE_FRAMES; ++f) {
    for (uint32_t k = 0; k < WAKE_MFCC_DIM; ++k) {
      int32_t q = (int32_t)lrintf(g_frames[(f * WAKE_MFCC_DIM) + k] * scale);
      if (q > 127) { q = 127; }
      if (q < -127) { q = -127; }
      g_case[(k * WAKE_FRAMES) + f] = (int8_t)q;
    }
  }
}

/* Run the CNN and return the wake margin: logit[marvin] - max(other logits).
 *
 * A margin, not a probability and not the top logit alone. Softmax is off in this runtime, and a
 * bare logit moves with the overall loudness of the window; the DIFFERENCE against the best
 * competing class is what the training report swept and what the thresholds in wake_config.h mean. */
static float wake_margin(void) {
  uint8_t shape[4] = {1, 1, (uint8_t)WAKE_MFCC_DIM, (uint8_t)WAKE_FRAMES};
  Tensor input = create_tensor(shape, 4);
  for (uint32_t i = 0; i < (WAKE_FRAMES * WAKE_MFCC_DIM); ++i) {
    input.data[i] = g_case[i];
  }

  Tensor logits = tinyspeech_run_inference(&input);

  float wake = logits.f_data[WAKE_CLASS];
  float best_other = -1e30f;
  for (int c = 0; c < WAKE_NUM_CLASSES; ++c) {
    if ((c != WAKE_CLASS) && (logits.f_data[c] > best_other)) {
      best_other = logits.f_data[c];
    }
  }

#if DSP_WAKE_DEBUG_FEATURES
  if ((g_events % DSP_WAKE_DEBUG_FEATURES) == 0u) {
    DSP_CITRINET_LOG("[wake] LOGITS");
    for (int c = 0; c < WAKE_NUM_CLASSES; ++c) {
      DSP_CITRINET_LOG(" %d/100", (int)lrintf(logits.f_data[c] * 100.0f));
    }
    DSP_CITRINET_LOG("  amax=%d/1e3  (marvin digits commands shortwords longwords background)\n",
                     (int)lrintf(g_last_amax * 1000.0f));
  }
#endif

  free_tensor(&logits);
  free_tensor(&input);
  return wake - best_other;
}

#if DSP_WAKE_GOLDEN_CHECK
#include "wake_golden.h"

/* Run the model on fixed inputs whose answers were computed on the host from THESE weights.
 *
 * This is the bisect a wake margin cannot give you: the microphone, the MFCC front-end and the
 * quantization are all bypassed, so a mismatch means the kernels on this chip and a match means the
 * front-end. It is how the RVV-vs-scalar problem was found and it stays in as a permanent guard —
 * if someone re-enables the vector kernels, this says so at boot instead of the demo just becoming
 * mysteriously deaf. Same idea as bearly-kws-rolling's KWS_BEARLY_ROLLING_USE_GOLDEN_INPUT. */
static void golden_check(void) {
  uint8_t shape[4] = {1, 1, (uint8_t)WAKE_MFCC_DIM, (uint8_t)WAKE_FRAMES};
  uint32_t bad = 0u;

  _Static_assert(WAKE_GOLDEN_CASE_BYTES == (int)(WAKE_FRAMES * WAKE_MFCC_DIM),
                 "golden case size does not match the model input");
  _Static_assert(WAKE_GOLDEN_NUM_CLASSES == WAKE_NUM_CLASSES,
                 "golden class count does not match the model");

  DSP_CITRINET_LOG("[wake] GOLDEN self-test (%d cases, host logits from the deployed weights)\n",
                   (int)WAKE_GOLDEN_NUM_CASES);
  for (int i = 0; i < WAKE_GOLDEN_NUM_CASES; ++i) {
    const wake_golden_case_t *g = &g_wake_golden[i];
    Tensor in = create_tensor(shape, 4);
    for (uint32_t k = 0; k < (WAKE_FRAMES * WAKE_MFCC_DIM); ++k) {
      in.data[k] = g->data[k];
    }
    Tensor out = tinyspeech_run_inference(&in);

    float worst = 0.0f;
    int argmax = 0;
    for (int c = 0; c < WAKE_NUM_CLASSES; ++c) {
      const float d = out.f_data[c] - g->logits[c];
      const float ad = (d < 0.0f) ? -d : d;
      if (ad > worst) { worst = ad; }
      if (out.f_data[c] > out.f_data[argmax]) { argmax = c; }
    }
    const int ok = (worst < 0.05f) && (argmax == (int)g->expected_class);
    if (!ok) {
      bad++;
      DSP_CITRINET_LOG("[wake]  %-12s max_diff=%d/100 argmax=%d/%ld  <-- FAIL\n",
                       g->name, (int)lrintf(worst * 100.0f), argmax, (long)g->expected_class);
    }
    free_tensor(&out);
    free_tensor(&in);
  }
  DSP_CITRINET_LOG("[wake] GOLDEN %s (%u/%d failed)%s\n",
                   bad ? "FAIL" : "PASS", (unsigned)bad, (int)WAKE_GOLDEN_NUM_CASES,
                   bad ? "  <-- the kernels disagree with the host; the front-end is NOT the problem"
                       : "  <-- inference matches the host exactly");
}
#endif /* DSP_WAKE_GOLDEN_CHECK */

int wake_gate_init(void) {
  if (mfcc_driver_init(&g_mfcc) != MFCC_DRIVER_OK) {
    DSP_CITRINET_LOG("[wake] MFCC init FAILED\n");
    return -1;
  }
  tinyspeech_prepare_runtime();
#if DSP_WAKE_GOLDEN_CHECK
  golden_check();
#endif

  memset(g_audio, 0, sizeof(g_audio));
  memset(g_frames, 0, sizeof(g_frames));
  g_dc = 0.0f;
  g_events = 0u;

  DSP_CITRINET_LOG("[wake] gate ready: onset>%d/1e6, window %u ms (pre-roll %u ms), margin > %d/100\n",
                   (int)lrintf((float)DSP_WAKE_VAD_THRESHOLD * 1.0e6f),
                   (unsigned)(WAKE_WINDOW_SAMPLES / (WAKE_SAMPLE_RATE_HZ / 1000u)),
                   (unsigned)(DSP_WAKE_PREROLL_SAMPLES / (WAKE_SAMPLE_RATE_HZ / 1000u)),
                   (int)lrintf((float)DSP_WAKE_MARGIN * 100.0f));
  return 0;
}

void wake_gate_listen(int (*poll)(void)) {
  const uint32_t FR = DSP_WAKE_VAD_FRAME_SAMPLES;
  const uint32_t PRE = DSP_WAKE_PREROLL_SAMPLES;

  DSP_CITRINET_LOG("[wake] listening — say the wake word.\n");

  for (;;) {
    uint32_t ring_pos = 0u, ring_filled = 0u, frames_seen = 0u;
    float32_t onset_energy = 0.0f;
    uint32_t w;

    /* ---- monitor ------------------------------------------------------------------------------
     * Cheap: read a 20 ms frame, take its AC energy, keep it in the pre-roll ring. No MFCC, no CNN.
     * This is where the gate spends essentially all of its time, and it is real-time by
     * construction because the blocking FIFO reads are the only thing pacing it. */
    for (;;) {
      float32_t sum = 0.0f, sumsq = 0.0f;

      for (uint32_t i = 0; i < FR; i += 2u) {
        float32_t a, b;
        mic_read_pair(&a, &b);
        g_preroll[ring_pos] = a; ring_pos = (ring_pos + 1u) % PRE;
        g_preroll[ring_pos] = b; ring_pos = (ring_pos + 1u) % PRE;
        sum += a + b;
        sumsq += (a * a) + (b * b);
      }
      if (ring_filled < PRE) {
        ring_filled += FR;
        if (ring_filled > PRE) { ring_filled = PRE; }
      }

      const float32_t mean = sum / (float32_t)FR;
      float32_t energy = (sumsq / (float32_t)FR) - (mean * mean);
      if (energy < 0.0f) { energy = 0.0f; }

      /* The mean of a quiet frame IS the microphone's DC offset; track it here rather than in the
       * captured window, where speech would bias it. */
      g_dc += 0.05f * (mean - g_dc);

      /* Service the C2C link while we wait. This is why the DSP can collect an answer to an earlier
       * question, or replace one still being generated, without ever stopping listening. */
      if ((poll != NULL) && ((frames_seen % DSP_WAKE_POLL_EVERY_FRAMES) == 0u)) {
        (void)poll();
      }
#if DSP_WAKE_LOG_EVERY
      if ((frames_seen % DSP_WAKE_LOG_EVERY) == 0u) {
        DSP_CITRINET_LOG("[wake] idle energy=%d/1e6 (onset at %d/1e6)\n",
                         (int)lrintf(energy * 1.0e6f),
                         (int)lrintf((float)DSP_WAKE_VAD_THRESHOLD * 1.0e6f));
      }
#endif
      frames_seen++;

      if (energy >= (float32_t)DSP_WAKE_VAD_THRESHOLD) {
        onset_energy = energy;
        break;
      }
    }

    /* ---- capture ------------------------------------------------------------------------------
     * Pre-roll first, so the window starts BEFORE the onset and the word's attack is inside it —
     * which also puts the word roughly where it sits in a training clip. Then fill the remainder
     * with fresh audio. Blocking reads only; nothing else runs, so the window is contiguous. */
    w = 0u;
    {
      const uint32_t start = (ring_filled < PRE) ? 0u : ring_pos;
      for (uint32_t k = 0; (k < ring_filled) && (w < WAKE_WINDOW_SAMPLES); ++k) {
        g_audio[w++] = g_preroll[(start + k) % PRE];
      }
    }
    while (w < WAKE_WINDOW_SAMPLES) {
      float32_t a, b;
      mic_read_pair(&a, &b);
      g_audio[w++] = a;
      if (w < WAKE_WINDOW_SAMPLES) { g_audio[w++] = b; }
    }
    for (uint32_t i = 0; i < WAKE_WINDOW_SAMPLES; ++i) {
      g_audio[i] -= g_dc;
    }

    /* ---- classify ------------------------------------------------------------------------------
     * One inference per onset. Nothing is being tracked while this runs, so however long it takes
     * costs latency and not corrupted audio — the reason this shape has no real-time budget. */
    {
      const uint64_t t0 = rdcycle64();
      for (uint32_t f = 0; f < WAKE_FRAMES; ++f) {
        compute_frame(f);
      }
      quantize_case();
      g_events++;
      g_last_margin = wake_margin();
      g_last_cycles = rdcycle64() - t0;
    }

    DSP_CITRINET_LOG("[wake] onset energy=%d/1e6 -> margin=%d/100 (need >%d/100) in %lu ms\n",
                     (int)lrintf(onset_energy * 1.0e6f),
                     (int)lrintf(g_last_margin * 100.0f),
                     (int)lrintf((float)DSP_WAKE_MARGIN * 100.0f),
                     (unsigned long)(g_last_cycles /
                                     (uint64_t)(DSP_CITRINET_TARGET_FREQ_HZ / 1000u)));

    if (g_last_margin > (float)DSP_WAKE_MARGIN) {
      DSP_CITRINET_LOG("[wake] *** WAKE *** (event %u)\n", (unsigned)g_events);
      return;
    }
  }
}
