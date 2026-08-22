/*
 * 3-element endfire microphone array for the wake gate — DSP 25.
 *
 * See include/mic_array.h for the design rationale. The short version: take out the INSTRUMENT part
 * of each element's delay (the cross-channel stream offset and the per-capture FIFO slip, both
 * integers) and leave the ACOUSTIC part alone, then let fixed endfire delays decide which direction
 * sums coherently. Re-measured every capture because the FIFOs slip whenever the read loop stops.
 *
 * The correlation, DC removal and fractional-shift primitives are dsp-mic-bench's `mic_quality.c`,
 * reused verbatim rather than reimplemented: that file is chip-free (math.h only) and covered by
 * dsp-mic-bench/test/host_mic_quality_test.c, and both of its failure modes — a sign error in the
 * shift direction, and a lag returned from a saturated search — are invisible on silicon.
 */

#include <stdint.h>
#include <math.h>
#include <string.h>

#include "main.h"              /* citrinet_config.h: DSP_CITRINET_LOG, mic scale macros */
#include "wake_config.h"
#include "mic_array.h"
#include "mic_quality.h"

#include "rocketcore.h"
#include "hal_i2s.h"

_Static_assert(MIC_ARRAY_ELEMS == 3, "the slot map and the gain table below assume three elements");
_Static_assert(DSP_WAKE_ARRAY_MAX_LAG_CROSS <= MQ_LAG_LIMIT,
               "the cross-channel lag search exceeds mic_quality's correlation buffer");

/* ---- slot map ------------------------------------------------------------------------------------
 * Element 0 and 1 share ONE I2S channel (its L and R slots), so they share BCLK and WS and can only
 * ever slip by whole samples. Element 2 is the second channel's left slot, which additionally
 * carries the large per-boot stream offset. Order is FRONT FIRST: element 0 is the mic the beam
 * looks out past.
 *
 * The physical board is M0 = ch1 left, M1 = ch1 right, M2 = ch0 left (M3, ch0 right, is dead), which
 * is why the channel defaults are 1 and 0 rather than the other way round. */
typedef struct { int ch; i2s_channel_side_t side; const char *name; } arr_slot_t;

static const arr_slot_t g_slot[MIC_ARRAY_ELEMS] = {
  { DSP_WAKE_ARRAY_CHANNEL,   I2S_LEFT,  "E0" },
  { DSP_WAKE_ARRAY_CHANNEL,   I2S_RIGHT, "E1" },
  { DSP_WAKE_ARRAY_CHANNEL_B, I2S_LEFT,  "E2" },
};

/* Per-element gain, dB, relative to element 0. Measured BROADSIDE at range on the soldered board —
 * the only geometry where all elements are equidistant, so the only one where a level ratio is a
 * sensitivity difference and not a distance. A close source corrupts these (see the near-field note
 * in wake_config.h), which is why they are compile-time constants and not re-estimated at runtime. */
static const float g_gain_db[MIC_ARRAY_ELEMS] = {
  0.0f, (float)DSP_WAKE_ARRAY_GAIN_DB_E1, (float)DSP_WAKE_ARRAY_GAIN_DB_E2
};
static float g_gain[MIC_ARRAY_ELEMS];

/* All static: multi-KB automatics are how this platform produces silent corruption (/CLAUDE.md). */
static float32_t g_el[MIC_ARRAY_ELEMS][WAKE_WINDOW_SAMPLES];
static float32_t g_ring[MIC_ARRAY_ELEMS][DSP_WAKE_PREROLL_SAMPLES];
static uint32_t  g_ring_pos;
static uint32_t  g_ring_filled;
static uint32_t  g_live;                 /* bitmask of slots that carry signal */
static uint32_t  g_nlive;
static uint32_t  g_w;                    /* samples currently in the capture windows */

static inline float32_t arr_scale(uint32_t slot) {
  return (float32_t)(((int32_t)slot) >> DSP_CITRINET_MIC_SAMPLE_SHIFT) *
         (1.0f / DSP_CITRINET_MIC_FULLSCALE);
}

/* Per-gap acoustic lag the geometry permits, in samples. At 2.15 cm and 16 kHz this is 1.00 — the
 * spacing was chosen to make it exactly one sample — and it is a HARD CEILING set by the board, so
 * any measured lag beyond it is instrument, not sound. */
static inline float arr_bound(void) {
  return (float)DSP_WAKE_ARRAY_SPACING_CM /
         (34400.0f / (float)WAKE_SAMPLE_RATE_HZ);
}

/* Is this slot actually being filled? Checked WITHOUT read_I2S_rx, which spins forever on an empty
 * FIFO — a dead slot must be discovered here or the monitor loop hangs the core on the first read.
 * Same guard as dsp-citrinet's mic_probe_right(). */
static int arr_probe(int ch, i2s_channel_side_t side) {
  for (uint32_t i = 0; i < DSP_CITRINET_MIC_PROBE_SPINS; ++i) {
    if (!get_I2S_rx_empty(ch, side)) { return 1; }
  }
  return 0;
}

int mic_array_init(void) {
  static i2s_params_t params = {
      .tx_en = 1, .rx_en = 1, .bitdepth_tx = I2S_BITDEPTH_32, .bitdepth_rx = I2S_BITDEPTH_32,
      .clkgen = 1, .dacen = 0, .ws_len = 3, .clkdiv = 8,
      .tx_fp = 0, .rx_fp = 0, .tx_force_left = 0, .rx_force_left = 0,
  };

  /* Both channels configured ADJACENTLY with the IDENTICAL divider. Both BCLKs are sys_clk/(2*(N+1)),
   * two integer dividers counting the same core clock, so identical clkdiv means identical frequency
   * and no drift — measured < 5 ppm on silicon. The remaining fixed phase offset is re-measured every
   * capture, so the (small) skew between these two calls does not need to be small.
   *
   * One of these two channels was already configured by dsp-citrinet's app_init for its own capture.
   * Re-configuring with identical parameters is idempotent, and doing both here keeps the array's
   * setup in one place instead of split across two files. */
  config_I2S(DSP_WAKE_ARRAY_CHANNEL, &params);
  set_I2S_sample_freq(DSP_WAKE_ARRAY_CHANNEL, (uint64_t)DSP_CITRINET_TARGET_FREQ_HZ,
                      (uint64_t)WAKE_SAMPLE_RATE_HZ, (uint8_t)DSP_CITRINET_MIC_BITDEPTH);
  config_I2S(DSP_WAKE_ARRAY_CHANNEL_B, &params);
  set_I2S_sample_freq(DSP_WAKE_ARRAY_CHANNEL_B, (uint64_t)DSP_CITRINET_TARGET_FREQ_HZ,
                      (uint64_t)WAKE_SAMPLE_RATE_HZ, (uint8_t)DSP_CITRINET_MIC_BITDEPTH);

  g_live = 0u;
  g_nlive = 0u;
  for (uint32_t e = 0; e < MIC_ARRAY_ELEMS; ++e) {
    g_gain[e] = powf(10.0f, g_gain_db[e] / 20.0f);
    if (arr_probe(g_slot[e].ch, g_slot[e].side)) {
      g_live |= (1u << e);
      g_nlive++;
    }
    DSP_CITRINET_LOG("[wake] array %s (ch%d %s): %s\n", g_slot[e].name, g_slot[e].ch,
                     (g_slot[e].side == I2S_LEFT) ? "L" : "R",
                     (g_live & (1u << e)) ? "ALIVE" : "SILENT");
  }

  memset(g_el, 0, sizeof(g_el));
  memset(g_ring, 0, sizeof(g_ring));
  g_ring_pos = 0u;
  g_ring_filled = 0u;
  g_w = 0u;

  if (!(g_live & 1u)) {
    /* Element 0 is the reference for every lag, level and gain, and the beam's front. Without it
     * there is no array — and no way to even say which direction "front" is. */
    DSP_CITRINET_LOG("[wake] array FAILED: element 0 (ch%d L) is silent — it is the reference\n",
                     DSP_WAKE_ARRAY_CHANNEL);
    return -1;
  }
  if (g_nlive < 2u) {
    DSP_CITRINET_LOG("[wake] array FAILED: only 1 of %d slots carries signal\n", MIC_ARRAY_ELEMS);
    return -1;
  }

  DSP_CITRINET_LOG("[wake] array ready: %u of %d elements, %s cm apart, endfire delay %d smp, "
                   "beam looks out past E0 (point E0 at the user)\n",
                   (unsigned)g_nlive, MIC_ARRAY_ELEMS,
                   (DSP_WAKE_ARRAY_SPACING_CM == 2.15f) ? "2.15" : "custom",
                   (int)DSP_WAKE_ARRAY_ENDFIRE_DELAY);
  DSP_CITRINET_LOG("[wake] array gains E1=%d/100dB E2=%d/100dB; per-gap acoustic bound %d/100 smp\n",
                   (int)lrintf(g_gain_db[1] * 100.0f), (int)lrintf(g_gain_db[2] * 100.0f),
                   (int)lrintf(arr_bound() * 100.0f));
  return 0;
}

/* Read one 64-bit block from every live element. Dead elements are left at zero and excluded
 * downstream — reading them would spin forever. */
static inline void arr_read_pair(float32_t out[MIC_ARRAY_ELEMS][2]) {
  for (uint32_t e = 0; e < MIC_ARRAY_ELEMS; ++e) {
    if (!(g_live & (1u << e))) { out[e][0] = 0.0f; out[e][1] = 0.0f; continue; }
    const uint64_t v = read_I2S_rx(g_slot[e].ch, g_slot[e].side);
    out[e][0] = arr_scale((uint32_t)(v & 0xFFFFFFFFu));
    out[e][1] = arr_scale((uint32_t)(v >> 32));
  }
}

void mic_array_monitor_pair(float *a, float *b) {
  float32_t s[MIC_ARRAY_ELEMS][2];
  arr_read_pair(s);
  for (uint32_t e = 0; e < MIC_ARRAY_ELEMS; ++e) {
    g_ring[e][g_ring_pos] = s[e][0];
    g_ring[e][(g_ring_pos + 1u) % DSP_WAKE_PREROLL_SAMPLES] = s[e][1];
  }
  g_ring_pos = (g_ring_pos + 2u) % DSP_WAKE_PREROLL_SAMPLES;
  if (g_ring_filled < DSP_WAKE_PREROLL_SAMPLES) {
    g_ring_filled += 2u;
    if (g_ring_filled > DSP_WAKE_PREROLL_SAMPLES) { g_ring_filled = DSP_WAKE_PREROLL_SAMPLES; }
  }
  *a = s[0][0];
  *b = s[0][1];
}

uint32_t mic_array_capture(uint32_t n) {
  const uint32_t PRE = DSP_WAKE_PREROLL_SAMPLES;
  uint32_t w = 0u;

  if (n > WAKE_WINDOW_SAMPLES) { n = WAKE_WINDOW_SAMPLES; }

  /* Pre-roll first, oldest sample first, so the window starts BEFORE the onset and the word's attack
   * is inside it. Identical policy to the single-mic path it replaces. */
  {
    const uint32_t start = (g_ring_filled < PRE) ? 0u : g_ring_pos;
    for (uint32_t k = 0; (k < g_ring_filled) && (w < n); ++k) {
      const uint32_t src = (start + k) % PRE;
      for (uint32_t e = 0; e < MIC_ARRAY_ELEMS; ++e) { g_el[e][w] = g_ring[e][src]; }
      w++;
    }
  }
  while (w < n) {
    float32_t s[MIC_ARRAY_ELEMS][2];
    arr_read_pair(s);
    for (uint32_t e = 0; e < MIC_ARRAY_ELEMS; ++e) { g_el[e][w] = s[e][0]; }
    w++;
    if (w < n) {
      for (uint32_t e = 0; e < MIC_ARRAY_ELEMS; ++e) { g_el[e][w] = s[e][1]; }
      w++;
    }
  }

  /* The ring is stale the moment capture ends — it holds audio from before the window. Restarting it
   * empty stops the NEXT onset inheriting a pre-roll that is a whole window old. */
  g_ring_filled = 0u;
  g_ring_pos = 0u;
  g_w = w;
  return w;
}

/* Solve the source direction from element 1 ALONE.
 *
 * Element 1 shares a channel with element 0, so its lag carries no stream offset — only the acoustic
 * lag and a whole-sample slip. That makes it the one measurement that needs no calibration constant,
 * which is why the direction is solved here and then PROPAGATED to element 2 rather than measured
 * from it.
 *
 * The ambiguity this has to resolve: at 2.15 cm one gap is exactly 1.000 sample, so "on axis in
 * front" and "on axis behind" differ by a whole sample — indistinguishable from the slip by timing
 * alone. LEVEL breaks the tie, because the nearer mic is louder, and that is the only thing that can:
 * a run that got this wrong had its rear wavefront silently re-timed into a front one and reported a
 * working beam. Inside the level deadband the source is taken to be near broadside, which is the
 * reading that assumes the least. */
static float arr_solve_tau(const mic_array_info_t *info, int *decided) {
  const float bound = arr_bound();
  const float lvl = info->lvl_db[1];          /* > 0 = E1 louder = source is BEHIND */
  const float mag = (lvl < 0.0f) ? -lvl : lvl;
  const int off_broadside = (mag > (float)DSP_WAKE_ARRAY_LEVEL_DEADBAND);
  const int want_neg = off_broadside && (lvl > 0.0f);
  const int want_pos = off_broadside && (lvl < 0.0f);
  float best = 0.0f, best_score = -1.0e30f;
  int found = 0;

  *decided = off_broadside;

  for (int sl = -DSP_WAKE_ARRAY_MAX_LAG_SAME; sl <= DSP_WAKE_ARRAY_MAX_LAG_SAME; ++sl) {
    const float tau = info->lag[1] - (float)sl;
    const float at = (tau < 0.0f) ? -tau : tau;
    if (at > bound + 0.05f) { continue; }             /* the board does not permit it */
    if (want_neg && (tau > 0.05f)) { continue; }      /* level says behind; reject a front solution */
    if (want_pos && (tau < -0.05f)) { continue; }
    /* Off broadside, prefer the LARGEST |tau| the level cue allows; inside the deadband prefer the
     * smallest, i.e. actually broadside. */
    const float score = off_broadside ? at : -at;
    if (score > best_score) { best_score = score; best = tau; found = 1; }
  }
  return found ? best : 0.0f;
}

void mic_array_beamform(float *dst, uint32_t n, mic_array_info_t *info) {
  const int D = (int)DSP_WAKE_ARRAY_ENDFIRE_DELAY;
  const float bound = arr_bound();
  float rms[MIC_ARRAY_ELEMS];
  uint32_t win;

  memset(info, 0, sizeof(*info));
  if (n > g_w) { n = g_w; }
  if (n == 0u) { return; }

  win = (n < (uint32_t)DSP_WAKE_ARRAY_CORR_SAMPLES) ? n : (uint32_t)DSP_WAKE_ARRAY_CORR_SAMPLES;

  /* Per-element DC removal. Not one shared estimate: the MEMS parts sit on individually different
   * offsets (4-5% of full scale measured), and a few percent of DC dominates a cross-correlation —
   * it pins the lag at 0 with gamma near 1, which looks exactly like a flawless array. */
  for (uint32_t e = 0; e < MIC_ARRAY_ELEMS; ++e) {
    if (!(g_live & (1u << e))) { rms[e] = 0.0f; continue; }
    (void)mq_remove_dc(g_el[e], n);
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; ++i) { acc += g_el[e][i] * g_el[e][i]; }
    rms[e] = sqrtf(acc / (float)n);
  }

  /* Level of each element against element 0, with its calibrated gain applied. This is what the
   * direction solve uses to break the front/back tie. */
  for (uint32_t e = 0; e < MIC_ARRAY_ELEMS; ++e) {
    info->lvl_db[e] = (g_live & (1u << e)) ? (mq_db_ratio(rms[e], rms[0]) + g_gain_db[e]) : MQ_DB_FLOOR;
  }

  /* Measure each element's delay against element 0.
   *
   * The search half-range is PER ELEMENT and deliberately tight for element 1: same channel means the
   * only instrument term is a +-2 sample slip, so a wide search there can only find an aliased peak
   * in periodic speech and call it a lag. Element 2 needs a wide window because the cross-channel
   * stream offset is 7-13 samples and moves every boot. */
  info->lag[0] = 0.0f;
  info->gamma[0] = 1.0f;
  for (uint32_t e = 1; e < MIC_ARRAY_ELEMS; ++e) {
    mq_pair_t p;
    const int ml = (g_slot[e].ch == g_slot[0].ch) ? (int)DSP_WAKE_ARRAY_MAX_LAG_SAME
                                                 : (int)DSP_WAKE_ARRAY_MAX_LAG_CROSS;
    if (!(g_live & (1u << e))) { info->gamma[e] = 0.0f; continue; }
    mq_pair_analyze(g_el[0], g_el[e], n, ml, win, &p);
    info->lag[e] = p.lag_frac;
    info->gamma[e] = p.gamma;
    /* A peak sitting exactly at the search limit is a saturated search, not a measurement — reading
     * one as a lag is what once turned a too-small window into a day of believing the two channels
     * had no common time base. Treat it as unmeasurable. */
    if (p.lag_frac <= (float)(-ml) + 0.01f || p.lag_frac >= (float)ml - 0.01f) {
      info->gamma[e] = 0.0f;
    }
  }

  /* Direction, then the instrument correction it implies for every element. */
  info->tau = (info->gamma[1] > (float)DSP_WAKE_ARRAY_MIN_GAMMA)
                  ? arr_solve_tau(info, &info->level_decided)
                  : 0.0f;
  {
    float t = info->tau / ((bound > 1e-6f) ? bound : 1.0f);
    if (t > 1.0f) { t = 1.0f; }
    if (t < -1.0f) { t = -1.0f; }
    info->angle_deg = acosf(t) * (180.0f / 3.14159265f);
  }

  info->used = 1u;              /* element 0 always contributes: it is the reference, shift 0 */
  info->nused = 1u;
  for (uint32_t e = 1; e < MIC_ARRAY_ELEMS; ++e) {
    if (!(g_live & (1u << e)) || !(info->gamma[e] > (float)DSP_WAKE_ARRAY_MIN_GAMMA)) {
      /* Unalignable. EXCLUDE it rather than summing it raw: an unaligned element adds an incoherent
       * copy of the same sound, which raises the output level without any directional meaning and so
       * flatters the array gain exactly when the array is working least well. */
      continue;
    }
    /* Acoustic lag for a uniform linear array in a plane wave is exactly e * tau. Whatever is left
     * over is instrument, and only its INTEGER part is removed — the measured fraction of the stream
     * offset is -0.045 samples (4 degrees at 4 kHz), so interpolating it out would cost more in
     * treble roll-off than it recovers. */
    const float acoustic = (float)e * info->tau;
    const float excess = info->lag[e] - acoustic;
    info->shift[e] = (int)((excess >= 0.0f) ? (excess + 0.5f) : (excess - 0.5f));
    if (info->shift[e] != 0) {
      mq_shift_advance(g_el[e], n, (float)info->shift[e]);
    }
    info->used |= (1u << e);
    info->nused++;
  }

  /* Endfire delay-and-sum. The delay is set by the element's PHYSICAL index, not by its rank among
   * the survivors: if a middle element drops out, the remaining two are still two gaps apart and
   * need two gaps of delay between them. Front element delayed MOST, so an on-axis wavefront — which
   * reaches the elements in order — lands in phase. */
  {
    const float scale = 1.0f / (float)info->nused;
    for (uint32_t i = 0; i < n; ++i) {
      float acc = 0.0f;
      for (uint32_t e = 0; e < MIC_ARRAY_ELEMS; ++e) {
        if (!(info->used & (1u << e))) { continue; }
        const int k = (int)(MIC_ARRAY_ELEMS - 1u - e);      /* gaps of delay for this element */
        const int j = (int)i - (k * D);
        acc += g_gain[e] * g_el[e][(j > 0) ? (uint32_t)j : 0u];
      }
      dst[i] = acc * scale;
    }
  }

  /* The array's own gain: output level against the reference element. On axis the elements add
   * coherently and this is high; off axis they smear and it falls. That difference is the energy
   * gate in wake_config.h — and it is only about 2 dB, which is why the direction is reported too. */
  {
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; ++i) { acc += dst[i] * dst[i]; }
    info->gain_db = mq_db_ratio(sqrtf(acc / (float)n), rms[0]);
  }
}
