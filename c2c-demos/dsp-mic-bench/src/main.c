/*
 * dsp-mic-bench — audio-quality benchmark for the DSP 25 I2S microphone path.
 *
 * MICB_MICS mics (1, 2 or 4). Every mic gets the SAME metric set from the SAME capture, plus the
 * combined signal C and the pairwise geometry, so "did the array help?" is a difference measured
 * inside one window rather than a comparison between two flashes.
 *
 *   1 -> M0                       channel A left
 *   2 -> M0, M1                   channel A left + right: one BCLK/WS, sample-locked by construction
 *   4 -> M0, M1, M2, M3           adds channel B's two slots; the TRACK line is what proves whether
 *                                 the two channels really share a time base
 *
 * Bring these up IN THAT ORDER. The 2-mic step is where you learn what a good gamma and a stable lag
 * look like on this rig with coherence guaranteed; only then does a cross-channel number mean anything.
 *
 *   make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-mic-bench EXTRA_CMAKE_ARGS="-DMICB_MICS=2"
 *   make tsi-run TTY=<tty> BINARY=build/c2c-demos/dsp-mic-bench/dsp-mic-bench.elf
 *
 * Real silicon only: Spike models no I2S.
 */

#include <stdint.h>
#include <math.h>

#include "main.h"
#include "mic_quality.h"

#include "rocketcore.h"
#include "hal_i2s.h"
#include "hal_mmio.h"

/* Consumed by the HAL/startup path, same as every other mic demo here. */
uint64_t target_frequency = MICB_TARGET_FREQ_HZ;

_Static_assert((MICB_CAPTURE_SAMPLES % 2u) == 0u,
               "capture length must be even: one FIFO read yields TWO time samples");
_Static_assert((MICB_DISCARD_SAMPLES % 2u) == 0u, "discard length must be even (same reason)");
_Static_assert(MICB_CAPTURE_SAMPLES >= (4u * MICB_FRAME_SAMPLES),
               "capture shorter than 4 analysis frames: the percentiles would be meaningless");
_Static_assert(MICB_NUM_FRAMES >= 10u,
               "fewer than 10 frames in the window: a 10th percentile of that is just the minimum");
#if MICB_TONE_HZ > 0
_Static_assert(MICB_TONE_HARMONICS >= 3 && MICB_TONE_HARMONICS <= MQ_MAX_HARMONICS,
               "the tone record prints h2 and h3, so at least 3 harmonics must be fitted");
_Static_assert((2 * MICB_TONE_HZ) < (int)MICB_SAMPLE_RATE_HZ,
               "tone frequency at or above Nyquist: what would be measured is an alias");
#endif

/* Mic parameters, identical to the capture path proven on silicon by dsp-citrinet / dsp-moonshine:
 * RX on, 32-bit slots, internal clock generator, DAC off, ws_len 3, clkdiv 8. rx_force_left MUST
 * stay 0 or both I2S slots collapse into the left FIFO and the second mic becomes invisible. */
static i2s_params_t g_i2s_params_mic = {
    .tx_en = 1, .rx_en = 1, .bitdepth_tx = I2S_BITDEPTH_32, .bitdepth_rx = I2S_BITDEPTH_32,
    .clkgen = 1, .dacen = 0, .ws_len = 3, .clkdiv = MICB_CLKDIV,
    .tx_fp = 0, .rx_fp = 0, .tx_force_left = 0, .rx_force_left = 0,
};

/* The slot map. Fixed and deliberate: M0/M1 are one channel's two slots, so they share BCLK and WS
 * and are sample-locked by construction — that pair is the reference. M2/M3 are the second channel,
 * whose coherence with the first is the thing under test (see mic_bench_config.h). */
typedef struct { int ch; i2s_channel_side_t side; const char *name; } mic_slot_t;

static const mic_slot_t g_slot[MICB_MICS] = {
  { MICB_MIC_CHANNEL,   I2S_LEFT,  "M0" },
#if MICB_MICS >= 2
  { MICB_MIC_CHANNEL,   I2S_RIGHT, "M1" },
#endif
#if MICB_MICS >= 3
  { MICB_MIC_CHANNEL_B, I2S_LEFT,  "M2" },   /* channel B, SEL -> GND */
#endif
#if MICB_MICS >= 4
  { MICB_MIC_CHANNEL_B, I2S_RIGHT, "M3" },   /* channel B, SEL -> VDD */
#endif
};

/* Capture buffers are static: multi-KB automatics are how this platform's silent stack overflow gets
 * reached (/CLAUDE.md), and 128 KB of them would reach it immediately. */
static float g_mic[MICB_MICS][MICB_CAPTURE_SAMPLES];
static float g_frames[MICB_NUM_FRAMES];
/* Which slots actually deliver data. Decided ONCE at boot by mic_probe(); a dead slot is dropped from
 * the read set and never probed again, because a read set that can change between samples would
 * silently offset the live mics against each other. */
static int g_live[MICB_MICS];
static int g_nlive = 1;

/* Which slots carry an actual SIGNAL, re-decided every run from the measured level. This is a
 * different question from g_live and the distinction is not academic: an undriven slot still FILLS
 * (the peripheral clocks in whatever is on SDIN during that slot), so g_live says "yes" for a slot
 * with no microphone on it. Averaging M0 with a constant is exactly a halving — d_rms = d_nf =
 * -6.02 dB with d_snr = 0.0 — which the ARRAY line would otherwise report as an array result that
 * beat the theoretical limit. So the combine and the verdict use THIS, while the read set stays
 * g_live: what may not change per run is how many FIFO reads happen, because that is what holds the
 * mics in sample alignment. */
static int g_sig[MICB_MICS];
static int g_nsig = 1;

/* Per-mic gain corrections (linear), from the MICB_GAIN_DB_Mx knobs. Applied only when building C —
 * see the knobs' comment for why they exist and how to measure them. */
static float g_gain[MICB_MICS];
static float g_gain_db[MICB_MICS];

#if MICB_MICS >= 2 && !MICB_PROBE_ONLY
/* Reference inter-mic lag, and how many runs have slipped a whole sample away from it. See the SLIP
 * block in app_main. */
static float g_lag_ref = 0.0f;
static int g_lag_ref_valid = 0;
static unsigned g_slip_runs __attribute__((unused)) = 0;
#endif

#if MICB_MICS >= 2 && !MICB_PROBE_ONLY
/* The combined signal gets its OWN buffer rather than being folded over M0, because the whole point is
 * to report every mic AND the combination from the SAME capture. Folding in place would turn the array
 * comparison into a comparison between two different 2-second windows of a changing room. */
static float g_comb[MICB_CAPTURE_SAMPLES];
#endif

static inline uint64_t micb_rdcycle(void) { uint64_t c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }
static inline int32_t micb_extract(uint32_t slot) { return ((int32_t)slot) >> MICB_SAMPLE_SHIFT; }

/* Read one 64-bit block from every live slot, in a FIXED order, and write the two time samples it
 * carries into that mic's buffer at [w] and [w+1].
 *
 * Exactly one read per live slot per call, always in the same order. Every slot has its own FIFO and
 * read_I2S_rx blocks on empty, so a skipped or doubled read offsets that mic against the others
 * PERMANENTLY — and a fixed inter-mic sample offset is precisely what destroys the phase relationship
 * that every combine mode and every lag measurement depends on. Nothing in here may become
 * conditional on anything that can differ between slots at run time; `g_live` is fixed at boot for
 * exactly that reason. */
static inline void mic_read_block(uint32_t w) {
  for (int m = 0; m < MICB_MICS; ++m) {
    if (!g_live[m]) { g_mic[m][w] = 0.0f; g_mic[m][w + 1u] = 0.0f; continue; }
    const uint64_t v = read_I2S_rx(g_slot[m].ch, g_slot[m].side);
    g_mic[m][w]      = (float)micb_extract((uint32_t)(v & 0xFFFFFFFFu)) * (1.0f / MICB_FULLSCALE);
    g_mic[m][w + 1u] = (float)micb_extract((uint32_t)(v >> 32)) * (1.0f / MICB_FULLSCALE);
  }
}

/* Same reads, results discarded — for the warm-up and the per-run FIFO drain. Keeping the read
 * PATTERN identical to the capture is the point: draining only one slot would leave the others
 * offset. */
static inline void mic_drain_block(void) {
  for (int m = 0; m < MICB_MICS; ++m) {
    if (!g_live[m]) continue;
    (void)read_I2S_rx(g_slot[m].ch, g_slot[m].side);
  }
}

#if MICB_DUMP_REGS
/* Read the watermark registers as BYTES. The HAL getters get_I2S_rx_watermark/get_I2S_tx_watermark
 * read them as a 32-bit word at byte-offset addresses, which is a misaligned MMIO access that HANGS
 * this core (found by dsp-i2s-test). */
static void dump_i2s_regs(const char *when) {
  const int ch = MICB_MIC_CHANNEL;
  MICB_LOG("[mic-bench] i2s %s: ch=%d config=0x%04x status=0x%02x empty(L/R)=%d/%d "
           "rx_wm(L/R)=%u/%u\n",
           when, ch, (unsigned)reg_read16(I2S_CONFIG(ch)), (unsigned)reg_read8(I2S_STATUS(ch)),
           get_I2S_rx_empty(ch, I2S_LEFT), get_I2S_rx_empty(ch, I2S_RIGHT),
           (unsigned)reg_read8(I2S_WATERMARK_RX_L(ch)), (unsigned)reg_read8(I2S_WATERMARK_RX_R(ch)));
}
#endif

/* Decide which slots are actually being filled WITHOUT calling read_I2S_rx — that spins forever on an
 * empty FIFO, so probing with a read would turn "mic not wired" or "channel not clocked" into a
 * silent hang with no console output, which on this platform is indistinguishable from a dead chip.
 * With a second channel this matters more, not less: a channel whose clock generator never started
 * has four empty FIFOs and would hang the first read. */
static void mic_probe(void) {
  g_nlive = 0;
  for (int m = 0; m < MICB_MICS; ++m) {
    g_live[m] = 0;
    for (uint32_t i = 0; i < MICB_PROBE_SPINS; ++i) {
      if (!get_I2S_rx_empty(g_slot[m].ch, g_slot[m].side)) { g_live[m] = 1; break; }
    }
    if (g_live[m]) g_nlive++;
    MICB_LOG("[mic-bench] probe %s (ch%d %s): %s\n", g_slot[m].name, g_slot[m].ch,
             (g_slot[m].side == I2S_LEFT) ? "L" : "R",
             g_live[m] ? "ALIVE" : "SILENT -> dropped from the read set");
  }
  if (!g_live[0]) {
    /* M0 is the reference every delta is measured against; without it there is nothing to report. */
    MICB_LOG("[mic-bench] M0 is silent — no primary microphone. Check the mic, the level shifter and "
             "that ch%d is clocked (this I2S master may need TX traffic to keep BCLK/WS running).\n",
             MICB_MIC_CHANNEL);
  }
}

/* ------------------------------------------------------------------------------- capture --------- */

#if MICB_VAD_ENABLE
/* Optional: hold off the window until something is said. AC energy per frame, same quantity and the
 * same default threshold as dsp-citrinet's VAD, so a threshold tuned there transfers here. */
static void wait_for_onset(void) {
  MICB_LOG("[mic-bench] waiting for onset (thresh=%d/1e6)...\n",
           (int)(MICB_VAD_THRESHOLD * 1.0e6f));
  for (;;) {
    float sum = 0.0f, sumsq = 0.0f;
    /* Gate on M0 only, but keep reading every live slot so the mics stay aligned. */
    for (uint32_t i = 0; i < MICB_FRAME_SAMPLES; i += 2u) {
      mic_read_block(0u);
      const float a = g_mic[0][0], b = g_mic[0][1];
      sum += a + b;
      sumsq += a * a + b * b;
    }
    const float mean = sum / (float)MICB_FRAME_SAMPLES;
    float energy = (sumsq / (float)MICB_FRAME_SAMPLES) - mean * mean;
    if (energy < 0.0f) energy = 0.0f;
    if (energy >= (float)MICB_VAD_THRESHOLD) return;
  }
}
#endif

/* Re-align the per-slot FIFOs before a capture, and why that is necessary.
 *
 * Each slot has its OWN FIFO. Between runs the analysis and the printing take far longer than the
 * capture, so both FIFOs overflow and drop samples — and the two sides do not necessarily drop the
 * SAME NUMBER. Whatever integer offset that leaves between the two mic streams persists for the whole
 * next capture and is re-randomised at the next gap. Measured on silicon: the reported inter-mic lag
 * was bimodal run to run, two clusters exactly 1.0 sample apart (+0.10..+0.25 and -0.70..-0.92) from
 * a stationary source. A beamformer with a fixed steering delay cannot survive that, and neither can
 * a lag measurement.
 *
 * The fixed-count drain that used to be here cannot fix it: it removes the same number of blocks from
 * each side and so PRESERVES any offset. Draining to empty is the closest available thing to a reset —
 * with one caveat that has to be stated because it bounds what this can achieve: the empty flag is
 * per 64-BIT BLOCK (two samples), so "empty" still permits one leftover sample per side, i.e. a
 * residual +-1 sample. Cycling rx_en is what is meant to clear that; whether the FIFOs actually flush
 * on this silicon is not documented, so the SLIP detector below reports the answer instead of
 * assuming it. */
static void mic_resync(void) {
#if MICB_RESYNC
  set_I2S_en(MICB_MIC_CHANNEL, g_i2s_params_mic.tx_en, 0);
#if MICB_MICS >= 3
  set_I2S_en(MICB_MIC_CHANNEL_B, g_i2s_params_mic.tx_en, 0);
#endif
  /* Drain each live slot INDEPENDENTLY to empty — not in lockstep, which is what would preserve an
   * offset. Bounded so a slot that never reports empty cannot hang the run. */
  for (uint32_t pass = 0; pass < MICB_RESYNC_MAX_BLOCKS; ++pass) {
    int any = 0;
    for (int m = 0; m < MICB_MICS; ++m) {
      if (!g_live[m]) continue;
      if (!get_I2S_rx_empty(g_slot[m].ch, g_slot[m].side)) {
        (void)read_I2S_rx(g_slot[m].ch, g_slot[m].side);
        any = 1;
      }
    }
    if (!any) break;                     /* every live slot empty in one pass */
  }
  set_I2S_en(MICB_MIC_CHANNEL, g_i2s_params_mic.tx_en, 1);
#if MICB_MICS >= 3
  set_I2S_en(MICB_MIC_CHANNEL_B, g_i2s_params_mic.tx_en, 1);
#endif
#endif
  /* A short lockstep drain after the restart, to skip whatever the receiver emits while it settles. */
  for (uint32_t i = 0; i < MICB_DISCARD_SAMPLES; i += 2u) mic_drain_block();
}

/* The DC that was removed from each buffer before the alignment correlation, so print_stats can still
 * report the real figure — see mic_align_all. Zero unless the alignment path ran. */
static float g_dc_pre[MICB_MICS];

/* Does element m sit on the OTHER I2S channel from M0? The distinction runs through everything here:
 * a same-channel element shares BCLK and WS with M0 so its only misalignment is the integer FIFO slip,
 * while a cross-channel element additionally carries the constant, non-integer stream offset. */
static inline int mic_cross_channel(int m) { return g_slot[m].ch != g_slot[0].ch; }

/* The largest lag element m can have against M0 from ACOUSTICS ALONE, in samples.
 *
 * A uniform linear array puts element m at m * spacing from M0, and no wavefront can arrive with a
 * path difference larger than that separation — so this is a hard ceiling set by the board, not an
 * expectation set by where the source happens to be. It is the strongest tool available here, because
 * it converts "the lag is 11 samples" from an observation into a CONCLUSION: 2.15 cm spacing puts M2
 * 4.30 cm from M0, permitting +-2.00 samples, so a measured 11.2 has at least 8 samples in it that
 * cannot be sound. */
static float mic_acoustic_bound(int m) {
  const float cm_per_sample = 34400.0f / (float)MICB_SAMPLE_RATE_HZ;
  return (float)m * (float)MICB_SPACING_CM / cm_per_sample;
}

#if MICB_MICS >= 2 && MICB_SLIP_FIX && !MICB_PROBE_ONLY
/* Per-element alignment state, kept so the report can print what was decided and why.
 *
 * The decomposition is the whole design, so it is worth stating exactly. Element m's buffer holds the
 * same sound as M0's, delayed by
 *      L(m) = acoustic(m) + channel_offset(m) + slip(m)
 * of which only the first term is wanted. `acoustic` is known (the array geometry, via
 * MICB_EXPECTED_LAG); the other two are the instrument and must go. So:
 *      off  = measured L(m) - acoustic(m)     <- everything to be removed, in one number
 *      slip = round(off)                      <- integer: the FIFO slip plus whole samples of offset,
 *                                                inseparable and, usefully, not needing to be separated
 *      frac = off - slip                      <- fraction: a property of the CHANNEL PAIR, immune to a
 *                                                slip of exactly 1.000, so measurable in one capture
 * Note what is absent: MICB_CHAN_OFFSET. It cancels out of `off` identically, which is why it is
 * documentation rather than a tuning knob — see its comment in the config header. */
typedef struct {
  float meas;      /* measured lag against M0, samples */
  float gamma;
  float acoustic;  /* the part of the lag the geometry says is real, and which must SURVIVE */
  float expect;    /* acoustic + the configured channel offset: the full predicted lag, for the log */
  float off;       /* meas - acoustic: the instrument error to be removed */
  int   slip;      /* round(off) — the integer part */
  float frac;      /* off - slip — the fractional part */
  float applied;   /* what was actually shifted out */
  int   valid;     /* 0 = not measurable this run (low coherence, or a saturated search) */
  int   pinned;    /* the correlation peak sat at +-MICB_MAX_LAG: the number is not a measurement */
  int   dead;      /* the slot carries no signal at all — a constant, not a quiet microphone */
  float lvl_db;    /* gain-corrected level of this element against M0: the direction cue */
} mic_align_t;

static mic_align_t g_al[MICB_MICS];

/* The per-gap acoustic lag solved from the capture itself (MICB_ALIGN_SWEEP), and whether it is
 * trustworthy this run. Printed as an angle so a polar sweep can be read straight off the log. */
static float g_sweep_tau __attribute__((unused)) = 0.0f;
static int g_sweep_valid __attribute__((unused)) = 0;

/* Running estimate of the FRACTIONAL stream offset of channel B against channel A, in samples.
 *
 * A channel property, so it is averaged over every cross-channel element and every run rather than
 * taken per-run: applying a per-run fraction would make the array track the talker, and an endfire
 * beamformer whose look direction follows the source is not a beamformer. The integer part is NOT
 * accumulated — it is removed per run, because it is inseparable from the per-capture FIFO slip (see
 * MICB_CHAN_OFFSET) and removing the sum of the two is both possible and sufficient. */
/* Accumulated as a VECTOR, not a scalar sum, because a fractional offset is a CIRCULAR quantity: it
 * is only defined modulo one sample, so +0.47 and -0.47 are neighbours (0.06 apart), not opposites.
 * A linear mean of values straddling the +-0.5 rounding boundary averages them toward zero, which is
 * exactly wrong. Measured on silicon: five estimates of +0.47/+0.42/-0.40/-0.33/-0.47 — all within
 * 0.14 of each other once wrapped — gave a linear mean of -0.06 with an alarming 0.94 spread, and a
 * circular mean of -0.46 with a concentration of 0.85. The data was consistent; the estimator was not.
 * `R` (resultant length, 1 = perfect agreement, 0 = uniformly scattered) is the honest quality metric
 * for a wrapped quantity, where a min/max spread is meaningless. */
static float g_chan_cos_sum = 0.0f;
static float g_chan_sin_sum = 0.0f;
static unsigned g_chan_frac_n = 0u;

/* Circular mean of the accumulated fractions, in samples, wrapped to [-0.5, +0.5). Writes the
 * concentration to *R when non-NULL. Returns 0 if nothing has been accumulated. */
static int chan_frac_mean(float *out, float *R) {
  if (g_chan_frac_n == 0u) return 0;
  const float c = g_chan_cos_sum, sn = g_chan_sin_sum;
  if (R) *R = (float)sqrt((double)c * c + (double)sn * sn) / (float)g_chan_frac_n;
  if (out) *out = (float)(atan2((double)sn, (double)c) / (2.0 * 3.14159265358979323846));
  return 1;
}

/* The ACOUSTIC lag element m should show against M0 — the part that must survive alignment.
 *
 * Uniform linear array in a plane wave: element m's path difference is m times element 1's, exactly.
 * So one operator-supplied number (MICB_EXPECTED_LAG, the per-gap lag) describes the whole array's
 * expected geometry. */
static float mic_acoustic_lag(int m) { return (float)m * (float)MICB_EXPECTED_LAG; }

/* Align EVERY element against M0, in place, before anything is measured.
 *
 * Two passes on purpose: every element is measured against the UNSHIFTED M0 first, so the estimates
 * are independent of each other and of the order they are applied in, and the fractional channel
 * offset can be folded into its running mean before any of it is used.
 *
 * Everything downstream — the per-mic records, the pair lines, TRACK, the combine and the ARRAY
 * verdict — then sees streams that differ only by acoustics. Without this the 4-mic combine sums four
 * differently-shifted copies of the same sound, which is what made averaging RAISE the measured noise
 * floor by 3.9 dB in the first 4-mic log. */
static void mic_align_all(unsigned run) {
  /* DC must go before the correlation, not after. mq_analyze removes it, but that runs later, and the
   * channel-B parts carry 4-5% of full scale of it — a DC offset that size dominates a
   * cross-correlation and pins the lag at 0 with a gamma near 1, i.e. it looks like a flawless array.
   * The removed value is kept so the printed dc= figure is still the real one. */
  for (int m = 0; m < MICB_MICS; ++m) {
    g_dc_pre[m] = g_live[m] ? mq_remove_dc(g_mic[m], MICB_CAPTURE_SAMPLES) : 0.0f;
  }

  /* Pass 1 — measure. */
  for (int m = 1; m < MICB_MICS; ++m) {
    mic_align_t *a = &g_al[m];
    a->meas = a->gamma = a->off = a->frac = a->applied = 0.0f;
    a->slip = 0; a->valid = 0; a->pinned = 0; a->dead = 0; a->lvl_db = 0.0f;
    a->acoustic = mic_acoustic_lag(m);
    a->expect = a->acoustic + (mic_cross_channel(m) ? (float)MICB_CHAN_OFFSET : 0.0f);
    if (!g_live[m] || !g_live[0]) continue;

    mq_pair_t p;
    mq_pair_analyze(g_mic[0], g_mic[m], MICB_CAPTURE_SAMPLES, MICB_MAX_LAG, MICB_XCORR_WINDOW, &p);
    a->meas = p.lag_frac;
    a->gamma = p.gamma;
    /* Level, gain-corrected, over the SAME window the lag came from. This is the direction cue that
     * timing cannot supply at 1-sample-per-gap spacing — see MICB_ALIGN_SWEEP. */
    if (p.rms_a > 0.0f && p.rms_b > 0.0f) {
      a->lvl_db = mq_db_ratio(p.rms_b * g_gain[m], p.rms_a * g_gain[0]);
    }
    /* A peak sitting exactly at the search limit is a SATURATED SEARCH, not a lag. This is the single
     * most expensive mistake available here: with MICB_MAX_LAG at 4 every cross-channel pair pinned at
     * +-4 and the low coherence that goes with a wrong alignment was read as "the two channels do not
     * share a time base". Refuse to shift on it, and say so. */
    a->pinned = (p.lag_int <= -MICB_MAX_LAG) || (p.lag_int >= MICB_MAX_LAG);
    /* ... but distinguish a saturated search from a slot carrying NOTHING. An undriven slot is a
     * constant, so it is exactly zero once DC is removed, every correlation is zero, and the peak
     * lands wherever the scan started — which looks identical to saturation and would send the
     * operator off to widen a search window when the fault is a missing microphone. */
    a->dead = !(p.rms_b > 0.0f);
    if (a->dead || a->pinned || !(p.gamma > MICB_SLIP_MIN_GAMMA)) continue;

    /* Everything that is not acoustics, in one number. Deliberately NOT measured against `expect`:
     * subtracting the configured channel offset and then adding it back to decide the shift is the
     * same arithmetic with an extra chance to get a sign wrong. */
    a->off   = a->meas - a->acoustic;
    /* Unbiased per-run estimate of the fraction, for the running mean below: off - nearest integer. */
    a->slip  = (int)((a->off >= 0.0f) ? (a->off + 0.5f) : (a->off - 0.5f));
    a->frac  = a->off - (float)a->slip;
    a->valid = 1;

    if (mic_cross_channel(m)) {
      const double th = 2.0 * 3.14159265358979323846 * (double)a->frac;
      g_chan_cos_sum += (float)cos(th);
      g_chan_sin_sum += (float)sin(th);
      g_chan_frac_n++;
    }
  }

#if MICB_ALIGN_SWEEP
  /* ---- SOLVE FOR THE SOURCE DIRECTION, rather than being told it -----------------------------------
   *
   * This exists so a beam PATTERN can be swept: move the source, capture, repeat, no rebuild.
   *
   * The unknown is ONE number — tau, the per-gap acoustic lag, equal to `bound * cos(angle)`, spanning
   * [-bound, +bound] as the source travels from behind the array to in front of it.
   *
   * It is solved from M1 ALONE, and that is deliberate: M1 shares a channel with M0, so its lag
   * contains no cross-channel stream offset. Nothing here needs MICB_CHAN_OFFSET, which matters
   * because the offset is re-randomised by every boot (observed 7.8 to 15.1 samples) and flashing a
   * build IS a boot — so a measured offset can never be carried into the binary that would use it.
   * Element 2's correction then falls out as `meas[2] - 2*tau` without the offset ever being named.
   *
   * Three facts pin tau, and all three are needed:
   *   MAGNITUDE, from timing: candidates are meas - s for integer s, and the physical bound
   *     (|tau| <= one gap) discards all but one or two.
   *   SIGN, from level: at 2.15 cm a gap is EXACTLY 1.000 sample, so tau and tau-1 both sit inside the
   *     bound and describe OPPOSITE directions — timing is blind to front-vs-back at this spacing.
   *     The nearer mic is louder, which is not blind (a clean 3 dB swing between the two geometries).
   *   BROADSIDE-vs-ON-AXIS, from the level MAGNITUDE: with the bound exactly 1.000, an interval like
   *     [0, +1] still holds two candidates a whole sample apart — 0.00 (broadside) and 1.00 (on axis).
   *     A close source separates them: on-axis at 25 cm gives 20*log10(27.15/25) = 0.7 dB, broadside
   *     gives 0. So a level inside the deadband means broadside and the SMALLEST |tau| is right; a
   *     level outside it means off-broadside and the LARGEST |tau| is right. Checked against three
   *     independent silicon datasets (front: level -0.67 -> tau +1.00; back: +1.7 -> tau -0.78;
   *     broadside: -0.1 -> tau +0.24) and it picks correctly in all of them. */
  if (g_al[1].valid) {
    const float bound = mic_acoustic_bound(1);
    const float lvl = g_al[1].lvl_db;         /* > 0 means M1 is louder, so the source is BEHIND */
    const float mag = (lvl < 0.0f) ? -lvl : lvl;
    const int off_broadside = (mag > (float)MICB_SWEEP_LEVEL_DEADBAND);
    const int want_neg = off_broadside && (lvl > 0.0f);
    const int want_pos = off_broadside && (lvl < 0.0f);

    float best_tau = 0.0f, best_score = -1.0e30f;
    int found = 0;
    for (int sl = -3; sl <= 3; ++sl) {
      const float tau = g_al[1].meas - (float)sl;
      const float at = (tau < 0.0f) ? -tau : tau;
      if (at > bound + 0.05f) continue;                 /* physically impossible for this spacing */
      if (want_neg && tau > 0.05f) continue;            /* level says behind */
      if (want_pos && tau < -0.05f) continue;           /* level says ahead */
      /* Off broadside -> the bigger |tau| is the real one; at broadside -> the smaller one is. */
      const float score = off_broadside ? at : -at;
      if (score > best_score) { best_score = score; best_tau = tau; found = 1; }
    }

    if (found) {
      g_sweep_tau = best_tau;
      g_sweep_valid = 1;
      /* Re-derive every element against the MEASURED geometry. Pass 2 is then unchanged: it removes
       * `off`, which is now genuinely nothing but instrument error. */
      for (int m = 1; m < MICB_MICS; ++m) {
        mic_align_t *a = &g_al[m];
        if (!a->valid) continue;
        a->acoustic = (float)m * best_tau;
        a->expect   = a->acoustic + (mic_cross_channel(m) ? (float)MICB_CHAN_OFFSET : 0.0f);
        a->off      = a->meas - a->acoustic;
        a->slip     = (int)((a->off >= 0.0f) ? (a->off + 0.5f) : (a->off - 0.5f));
        a->frac     = a->off - (float)a->slip;
      }
      {
        char t[MQ_FMT_BYTES], l[MQ_FMT_BYTES], b[MQ_FMT_BYTES], d[MQ_FMT_BYTES];
        const float cosang = (bound > 0.0f) ? (best_tau / bound) : 0.0f;
        /* Print the ANGLE, so a polar sweep reads straight off the log without arithmetic. acos is
         * only valid on [-1,1]; the bound check above guarantees that to within the 0.05 slack. */
        float ca = cosang;
        if (ca > 1.0f) ca = 1.0f;
        if (ca < -1.0f) ca = -1.0f;
        const float deg = (float)(acos((double)ca) * 180.0 / 3.14159265358979323846);
        MICB_LOG("[mic-bench] run=%u SWEEP direction MEASURED: per-gap lag %ssmp of a possible %s "
                 "-> source is %sdeg off the M0 axis (%s); M1 level %sdB (%s)\n",
                 run, mq_fmt(t, sizeof(t), best_tau, 2), mq_fmt(b, sizeof(b), bound, 2),
                 mq_fmt(d, sizeof(d), deg, 0),
                 (ca > 0.7f) ? "roughly ON AXIS, front" : (ca < -0.7f) ? "roughly ON AXIS, behind"
                   : (ca > 0.2f) ? "forward of broadside" : (ca < -0.2f) ? "behind broadside"
                   : "near BROADSIDE",
                 mq_fmt(l, sizeof(l), lvl, 2),
                 want_neg ? "M1 louder -> behind" : want_pos ? "M0 louder -> ahead"
                          : "inside the deadband -> treated as broadside");
      }
      /* Element 2's leftover IS the cross-channel offset, measured with the acoustics taken out. It
       * must not move within a boot; if it does, either the solve is wrong or the source moved
       * mid-capture. Reported rather than used, so the solve stays independent of it. */
      if (MICB_MICS >= 3 && g_al[2].valid && mic_cross_channel(2)) {
        char o[MQ_FMT_BYTES];
        MICB_LOG("[mic-bench] run=%u SWEEP implied channel offset %ssmp (should be constant within "
                 "this boot)\n", run, mq_fmt(o, sizeof(o), g_al[2].off, 2));
      }
    } else {
      g_sweep_valid = 0;
      MICB_LOG("[mic-bench] run=%u SWEEP could not solve the direction: no candidate per-gap lag fits "
               "the bound with the sign the level implies. Nothing aligned this run.\n", run);
    }
  }
#endif

  float chan_frac = 0.0f;
  (void)chan_frac_mean(&chan_frac, (float *)0);
  (void)chan_frac;   /* unused with MICB_ALIGN_FRAC=0 */

  /* Pass 2 — apply, and report every decision. */
  for (int m = 1; m < MICB_MICS; ++m) {
    mic_align_t *a = &g_al[m];
    /* One buffer per conversion in the same printf: mq_fmt returns the buffer it was handed, so a
     * reused buffer makes every %s in that call print the LAST value formatted into it. That produced
     * the self-contradicting log line "lag -0.31 is -1.00 samples from the expected -1.00". */
    char l[MQ_FMT_BYTES], e[MQ_FMT_BYTES], r[MQ_FMT_BYTES], f[MQ_FMT_BYTES], g[MQ_FMT_BYTES];
    char o[MQ_FMT_BYTES];
    if (!g_live[m]) continue;
    if (a->dead) {
      MICB_LOG("[mic-bench] run=%u ALIGN %s NO SIGNAL: the slot is clocked but constant, so there is "
               "nothing to align. Check that a mic is on it and that its SEL strap selects this slot "
               "(its per-mic record below will read rms=-240).\n", run, g_slot[m].name);
      continue;
    }
    if (a->pinned) {
      MICB_LOG("[mic-bench] run=%u ALIGN %s NOT MEASURABLE: the correlation peak sat at the +-%d "
               "search limit, so the lag is saturated rather than measured — raise MICB_MAX_LAG "
               "above the cross-channel offset (~12) before believing any gamma\n",
               run, g_slot[m].name, (int)MICB_MAX_LAG);
      continue;
    }
    if (!a->valid) {
      MICB_LOG("[mic-bench] run=%u ALIGN %s skipped: gamma %s below %d/100, the lag is not measurable "
               "in this window\n", run, g_slot[m].name, mq_fmt(g, sizeof(g), a->gamma, 3),
               (int)(MICB_SLIP_MIN_GAMMA * 100.0f));
      continue;
    }

    /* What gets removed: the integer part always (an instrument artifact — a FIFO slip, plus whatever
     * whole samples of channel offset came with it), and the fractional part only for cross-channel
     * elements and only from the SESSION MEAN. A same-channel element shares BCLK and WS with M0, so
     * it has no stream offset to remove and its fractional residual is acoustics or estimator noise —
     * shifting by that would be steering the array at the talker. */
    /* WHAT TO SHIFT. With no fractional term this is just the integer slip. With one, the integer has
     * to be re-chosen AFTER the known fraction is taken out — `round(off) + frac` is NOT the same
     * thing and can be worse than not correcting at all.
     *
     * Measured on silicon: off = 11.46 with a session fraction of -0.31 gave round(11.46) = 11, hence
     * an applied 10.69 against a true 11.46 — under-removing by 0.77 samples, where applying the bare
     * integer would have erred by 0.46. The cause is that a per-run fraction wandering across the .5
     * rounding boundary moves the integer by a whole sample while the session mean stays put, so the
     * two terms stop being consistent. Choosing the integer to minimise |off - (k + frac)| keeps the
     * total within half a sample of what was actually measured, which is the most a session-constant
     * fraction can promise. Same value as before whenever the fraction is stable. */
    float d = (float)a->slip;
#if MICB_ALIGN_FRAC
    if (mic_cross_channel(m)) {
      const float r = a->off - chan_frac;
      const int k = (int)((r >= 0.0f) ? (r + 0.5f) : (r - 0.5f));
      d = (float)k + chan_frac;
    }
#endif
#if !MICB_ALIGN_MEASURE_ONLY
    /* With no fractional correction available, a residual far from a whole sample means the source is
     * not where MICB_EXPECTED_LAG says — treat it as the room, not as a slip, exactly as the 2-mic
     * path did. */
    const int trust = (a->frac > -MICB_SLIP_TOLERANCE && a->frac < MICB_SLIP_TOLERANCE);
    if (!trust && !mic_cross_channel(m)) {
      MICB_LOG("[mic-bench] run=%u ALIGN %s declined: lag %s vs acoustic %s leaves %s, which is %s "
               "from the nearest whole sample (tolerance %d/100) — treated as the source moving, not "
               "a FIFO slip\n", run, g_slot[m].name,
               mq_fmt(l, sizeof(l), a->meas, 2), mq_fmt(e, sizeof(e), a->acoustic, 2),
               mq_fmt(r, sizeof(r), a->off, 2), mq_fmt(f, sizeof(f), a->frac, 2),
               (int)(MICB_SLIP_TOLERANCE * 100.0f));
      continue;
    }
    if (d != 0.0f) {
      mq_shift_advance(g_mic[m], MICB_CAPTURE_SAMPLES, d);
      a->applied = d;
      g_slip_runs++;
    }
#endif
    {
      char ap[MQ_FMT_BYTES];
      /* Flag a lag that the board cannot produce acoustically. Without this the operator has to do
       * the spacing arithmetic by hand to notice, and the number looks like a plausible geometry. */
      const float bnd = mic_acoustic_bound(m);
      const float amag = (a->meas < 0.0f) ? -a->meas : a->meas;
      if (amag > bnd + 1.0f) {
        char bb[MQ_FMT_BYTES], mm[MQ_FMT_BYTES], xx[MQ_FMT_BYTES];
        MICB_LOG("[mic-bench] run=%u ALIGN %s NOT ACOUSTIC: lag %ssmp exceeds the %ssmp that %s cm of "
                 "spacing permits (+1 for slip), so at least %ssmp of it is a STREAM OFFSET, not "
                 "sound%s\n", run, g_slot[m].name, mq_fmt(mm, sizeof(mm), a->meas, 2),
                 mq_fmt(bb, sizeof(bb), bnd, 2), mq_fmt(xx, sizeof(xx), (float)MICB_SPACING_CM, 2),
                 mq_fmt(mm, sizeof(mm), amag - bnd - 1.0f, 2),
                 mic_cross_channel(m) ? "  [CROSS-CHANNEL]" : "  <- and it is SAME-CHANNEL, which "
                 "should be impossible: check MICB_SPACING_CM against the board");
      }
      MICB_LOG("[mic-bench] run=%u ALIGN %s gamma=%s lag=%s (predicted %s) acoustic=%s off=%s "
               "(slip %+d, frac %s) -> "
#if MICB_ALIGN_MEASURE_ONLY
               "MEASURE-ONLY, nothing shifted (%ssmp would have been)%s\n", run, g_slot[m].name,
#else
               "shifted %ssmp%s\n", run, g_slot[m].name,
#endif
               mq_fmt(g, sizeof(g), a->gamma, 3), mq_fmt(l, sizeof(l), a->meas, 2),
               mq_fmt(e, sizeof(e), a->expect, 2), mq_fmt(r, sizeof(r), a->acoustic, 2),
               mq_fmt(o, sizeof(o), a->off, 2),
               a->slip, mq_fmt(f, sizeof(f), a->frac, 2),
#if MICB_ALIGN_MEASURE_ONLY
               mq_fmt(ap, sizeof(ap), d, 2),
#else
               mq_fmt(ap, sizeof(ap), a->applied, 2),
#endif
               mic_cross_channel(m) ? "  [CROSS-CHANNEL]" : "");
    }
  }
}
#endif

/* Fill the capture buffers and return the measured sample rate in Hz.
 *
 * The rate is not decoration. Every level, percentile and tone frequency in this demo is interpreted
 * against MICB_SAMPLE_RATE_HZ; if the mic is actually clocked at some other rate, all of it is
 * quietly wrong in a way no single number would reveal. Since the FIFO read blocks until a sample
 * arrives, the elapsed cycles across a known number of samples measure the mic's real clock.
 * (rdcycle, not CLINT mtime: mtime is derived from the core clock and reads ~50 MHz whatever the PLL
 * is set to — see /CLAUDE.md.) */
static uint32_t capture_window(void) {
#if MICB_VAD_ENABLE
  wait_for_onset();
#endif
  /* Drain whatever piled up in the FIFOs while the previous window was being analysed and printed,
   * and re-align the slots against each other — see mic_resync for why the second part matters. */
  mic_resync();

  const uint64_t t0 = micb_rdcycle();
  for (uint32_t w = 0; w < MICB_CAPTURE_SAMPLES; w += 2u) mic_read_block(w);
  const uint64_t cyc = micb_rdcycle() - t0;
  if (cyc == 0ull) return 0u;
  return (uint32_t)(((uint64_t)MICB_CAPTURE_SAMPLES * (uint64_t)MICB_TARGET_FREQ_HZ) / cyc);
}

/* ------------------------------------------------------------------------------- reporting ------- */

/* A slot nobody drives still FILLS — the peripheral clocks in whatever is on SDIN during that slot, so
 * the boot probe reports it ALIVE either way (with one mic connected the right FIFO already reads
 * non-empty: status=0x00, rx_wm_R=1). What an undriven slot actually looks like is a CONSTANT: a
 * floating line reads 0xFFFFFFFF every frame, which extracts to -1.19e-7 and, once the DC is removed,
 * to exactly zero. So the level is the presence test the probe cannot be.
 *
 * -120 dBFS is far below anything real: this mic's own noise floor is about -91 dBFS and the measured
 * room floor is -70. */
static const char *slot_verdict(const mq_stats_t *s) {
  if (!(s->rms > 0.0f)) return "  <- NOT DRIVEN (constant slot: no mic on this SEL, or SEL strapped wrong)";
  if (mq_dbfs(s->rms) < -120.0f) return "  <- essentially constant: is a mic really on this slot?";
  return "";
}

static void print_stats(unsigned run, const char *sig, const mq_stats_t *s) {
  char dc[MQ_FMT_BYTES], dr[MQ_FMT_BYTES], rms[MQ_FMT_BYTES], pk[MQ_FMT_BYTES], cr[MQ_FMT_BYTES];
  char nf[MQ_FMT_BYTES], ac[MQ_FMT_BYTES], sn[MQ_FMT_BYTES], hf[MQ_FMT_BYTES], af[MQ_FMT_BYTES];
  MICB_LOG("[mic-bench] run=%u sig=%s n=%u dc=%sfs drift=%sfs rms=%s peak=%s crest=%s "
           "nf=%s act=%s snr=%s hf=%s clip=%u active=%s%s\n",
           run, sig, (unsigned)s->n,
           mq_fmt(dc, sizeof(dc), s->dc, 4), mq_fmt(dr, sizeof(dr), s->drift, 4),
           mq_fmt(rms, sizeof(rms), mq_dbfs(s->rms), 1),
           mq_fmt(pk, sizeof(pk), mq_dbfs(s->peak), 1),
           mq_fmt(cr, sizeof(cr), mq_crest_db(s), 1),
           mq_fmt(nf, sizeof(nf), mq_dbfs(s->nf), 1),
           mq_fmt(ac, sizeof(ac), mq_dbfs(s->act), 1),
           mq_fmt(sn, sizeof(sn), mq_snr_db(s), 1),
           mq_fmt(hf, sizeof(hf), mq_hf_db(s), 1),
           (unsigned)s->clip, mq_fmt(af, sizeof(af), s->active_frac, 2), slot_verdict(s));
}

#if MICB_TONE_HZ > 0
/* Fit the tone in the middle of the capture, at the frequency it is ACTUALLY at (see
 * mq_tone_refine_f0 — a fit at the nominal frequency reports a clean mic as a distorting one). The
 * measured f0 is printed: if it is not close to MICB_TONE_HZ, either the generator or the mic clock
 * is off, and that is worth knowing before reading the distortion figures. */
static void analyze_tone(unsigned run, const char *sig, const float *x) {
  const uint32_t W = (MICB_TONE_WINDOW < MICB_CAPTURE_SAMPLES) ? MICB_TONE_WINDOW
                                                              : MICB_CAPTURE_SAMPLES;
  const float *w = x + (MICB_CAPTURE_SAMPLES - W) / 2u;
  const float fs = (float)MICB_SAMPLE_RATE_HZ;
  const float f0 = mq_tone_refine_f0(w, W, fs, (float)MICB_TONE_HZ,
                                     (float)MICB_TONE_SEARCH_PCT * 0.01f, MICB_TONE_SEARCH_STEPS);
  mq_tone_t t;
  mq_tone_analyze(w, W, fs, f0, MICB_TONE_HARMONICS, &t);

  char f[MQ_FMT_BYTES], lv[MQ_FMT_BYTES], sd[MQ_FMT_BYTES], th[MQ_FMT_BYTES];
  char h2[MQ_FMT_BYTES], h3[MQ_FMT_BYTES];
  MICB_LOG("[mic-bench] run=%u tone sig=%s f0=%sHz level=%s sinad=%s thd=%s h2=%s h3=%s n=%u\n",
           run, sig,
           mq_fmt(f, sizeof(f), t.f0, 1),
           mq_fmt(lv, sizeof(lv), mq_dbfs(t.amp[0] * 0.70710678f), 1),
           mq_fmt(sd, sizeof(sd), mq_sinad_db(&t), 1),
           mq_fmt(th, sizeof(th), mq_thd_db(&t), 1),
           mq_fmt(h2, sizeof(h2), mq_db_ratio(t.amp[1], t.amp[0]), 1),
           mq_fmt(h3, sizeof(h3), mq_db_ratio(t.amp[2], t.amp[0]), 1), (unsigned)W);
}
#endif

/* Rolling min/mean/max. A single 2-second capture of room noise wanders by a decibel or so, so the
 * summary — not one record — is what should be compared between two builds. */
#if MICB_SUMMARY_EVERY > 0u && !MICB_PROBE_ONLY
typedef struct { float v[MICB_SUMMARY_EVERY]; uint32_t n; } hist_t;

static void hist_push(hist_t *h, float x) {
  if (h->n < MICB_SUMMARY_EVERY) h->v[h->n++] = x;
  else {
    for (uint32_t i = 1; i < MICB_SUMMARY_EVERY; ++i) h->v[i - 1u] = h->v[i];
    h->v[MICB_SUMMARY_EVERY - 1u] = x;
  }
}

static void hist_print(const char *label, const hist_t *h) {
  if (h->n == 0u) return;
  float lo = h->v[0], hi = h->v[0];
  double sum = 0.0;
  for (uint32_t i = 0; i < h->n; ++i) {
    if (h->v[i] < lo) lo = h->v[i];
    if (h->v[i] > hi) hi = h->v[i];
    sum += (double)h->v[i];
  }
  char a[MQ_FMT_BYTES], b[MQ_FMT_BYTES], c[MQ_FMT_BYTES];
  MICB_LOG("[mic-bench] SUMMARY n=%u %-14s min=%s mean=%s max=%s\n", (unsigned)h->n, label,
           mq_fmt(a, sizeof(a), lo, 1), mq_fmt(b, sizeof(b), (float)(sum / (double)h->n), 1),
           mq_fmt(c, sizeof(c), hi, 1));
}

static hist_t g_h_rms_l, g_h_nf_l, g_h_snr_l;   /* M0: the reference every delta is against */
#if MICB_MICS >= 2
static hist_t g_h_rms_c, g_h_nf_c, g_h_snr_c, g_h_dnf, g_h_dsnr, g_h_gamma;
#endif

/* Same reduction as hist_print, but returned rather than printed — the calibration block needs the
 * numbers themselves, because a constant is only worth pasting into a build alongside the spread that
 * says how much to trust it. */
/* MEDIAN, and it is the number the calibration should be pasted from rather than the mean.
 *
 * A mic array is calibrated in a room, and a room contains one-off events: a bump on a mic body, a
 * cable shifting, a chair. On silicon, two of eight captures had M1 arrive 9-13 dB hot with a strongly
 * negative hf (i.e. dominated by rumble — a mechanical thump, not sound). Those two runs dragged the
 * MEAN gain from -0.3 dB to -2.8 dB, a fabricated 2.5 dB error that would then be applied to every
 * capture forever. The median of the same eight runs is -0.32 dB. A statistic that survives outliers
 * is not a refinement here, it is the difference between a calibration and a corrupted one.
 *
 * Copies into a static buffer (a stack overflow on this platform is silent — /CLAUDE.md) and insertion
 * sorts; n is MICB_SUMMARY_EVERY, so this is trivially small. */
static int __attribute__((unused)) hist_median(const hist_t *h, float *med) {
  static float t[MICB_SUMMARY_EVERY];
  if (h->n == 0u) return 0;
  for (uint32_t i = 0; i < h->n; ++i) t[i] = h->v[i];
  for (uint32_t i = 1u; i < h->n; ++i) {
    const float k = t[i];
    uint32_t j = i;
    while (j > 0u && t[j - 1u] > k) { t[j] = t[j - 1u]; --j; }
    t[j] = k;
  }
  *med = (h->n & 1u) ? t[h->n / 2u] : (0.5f * (t[h->n / 2u - 1u] + t[h->n / 2u]));
  return 1;
}

/* CIRCULAR mean of each sample's fractional part, in samples, wrapped to [-0.5, +0.5), plus the
 * concentration R (1 = perfect agreement, 0 = uniform scatter).
 *
 * This is the right estimator for any lag on this hardware, and the median is NOT — which cost a
 * silicon run to learn. The +-1 FIFO slip splits the measurements into two clusters exactly 1.000
 * apart, so the median simply reports whichever cluster held the majority: eight captures of a true
 * 0.27-sample lag, five of them slipped, gave a median of 1.255. The FRACTIONAL part, by contrast, is
 * untouched by a slip of exactly 1.000, so it is the same in both clusters — those same eight captures
 * agree on it to R = 0.997. Recover the fraction here, then resolve the integer from the physical
 * bound on the geometry, which is what the operator actually knows. */
static int __attribute__((unused)) hist_circmean_frac(const hist_t *h, float *frac, float *R) {
  if (h->n == 0u) return 0;
  double cs = 0.0, sn = 0.0;
  for (uint32_t i = 0; i < h->n; ++i) {
    const float v = h->v[i];
    const float nearest = (v >= 0.0f) ? (float)(int)(v + 0.5f) : (float)(int)(v - 0.5f);
    const double th = 2.0 * 3.14159265358979323846 * (double)(v - nearest);
    cs += cos(th); sn += sin(th);
  }
  if (R) *R = (float)(sqrt(cs * cs + sn * sn) / (double)h->n);
  if (frac) *frac = (float)(atan2(sn, cs) / (2.0 * 3.14159265358979323846));
  return 1;
}

static int __attribute__((unused)) hist_stat(const hist_t *h, float *lo, float *mean, float *hi) {
  if (h->n == 0u) return 0;
  float a = h->v[0], b = h->v[0];
  double sum = 0.0;
  for (uint32_t i = 0; i < h->n; ++i) {
    if (h->v[i] < a) a = h->v[i];
    if (h->v[i] > b) b = h->v[i];
    sum += (double)h->v[i];
  }
  *lo = a; *hi = b; *mean = (float)(sum / (double)h->n);
  return 1;
}

#if MICB_MICS >= 2 && MICB_CAL_REPORT
/* Per-element calibration histories. These are what the SETUP block is built from, and they are kept
 * per element rather than as one aggregate because each element needs its own gain and its own
 * expected lag — an array is calibrated element by element or not at all. */
static hist_t g_h_cal_gain[MICB_MICS];   /* level of M0 over element m, dB: the gain correction */
static hist_t g_h_cal_lag[MICB_MICS];    /* measured lag of element m against M0, samples */
static hist_t g_h_cal_off[MICB_MICS];    /* non-acoustic part of that lag, samples */
static hist_t g_h_cal_frac;              /* fractional channel offset estimates, samples */

/* THE CALIBRATION, as flags to paste into the next build.
 *
 * Everything here is an average over the last MICB_SUMMARY_EVERY captures with its run-to-run spread
 * printed beside it, because a constant measured once is a guess and the spread is the only thing that
 * says whether the number deserves to be baked into a build. Two of the three constants also require a
 * particular stimulus to be MEANINGFUL, and getting that wrong silently produces plausible numbers —
 * so each line states its own precondition rather than relying on the operator remembering. */
static void print_setup_block(unsigned run) {
  float lo, mean, hi;
  char a[MQ_FMT_BYTES], b[MQ_FMT_BYTES], c[MQ_FMT_BYTES];

  MICB_LOG("[mic-bench] ===== SETUP after run %u (%u-run averages) =====\n", run,
           (unsigned)MICB_SUMMARY_EVERY);

  /* 1. GAINS. Precondition: every mic the same distance from one steady broadband source. For a
   * colinear endfire array that means BROADSIDE — on the perpendicular bisector — which is the one
   * geometry where all four are equidistant. It is also the geometry where an integer slip cannot be
   * resolved, and that does not matter here: an RMS ratio is invariant under a time shift. */
  MICB_LOG("[mic-bench] SETUP gain (needs a BROADSIDE source, equidistant from every mic):\n");
  MICB_LOG("[mic-bench] SETUP  ");
  for (int m = 1; m < MICB_MICS; ++m) {
    float med;
    if (!hist_median(&g_h_cal_gain[m], &med)) continue;
    MICB_LOG(" -DMICB_GAIN_DB_M%d=%s", m, mq_fmt(a, sizeof(a), med, 2));
  }
  MICB_LOG("\n");
  for (int m = 1; m < MICB_MICS; ++m) {
    float med;
    if (!hist_stat(&g_h_cal_gain[m], &lo, &mean, &hi)) continue;
    (void)hist_median(&g_h_cal_gain[m], &med);
    /* Median is what gets pasted; mean and spread are printed beside it as the DIAGNOSTIC. When they
     * disagree, the run-to-run data contains outliers — a bumped mic, a moving source — and the
     * spread says how badly. A null is capped at 20*log10|1-g| by the RESIDUAL mismatch, so a gain
     * constant that wanders by more than the mismatch it corrects is not a calibration. */
    char d[MQ_FMT_BYTES];
    MICB_LOG("[mic-bench] SETUP    %s gain %sdB (mean %s, spread %s..%s over %u runs)%s\n",
             g_slot[m].name, mq_fmt(a, sizeof(a), med, 2), mq_fmt(d, sizeof(d), mean, 2),
             mq_fmt(b, sizeof(b), lo, 2), mq_fmt(c, sizeof(c), hi, 2),
             (unsigned)g_h_cal_gain[m].n,
             ((hi - lo) > 3.0f) ? "  <- OUTLIERS: >3dB spread. Something moved or was touched; the "
                                  "median is robust to it but check the per-run rms and hf." : "");
  }

  /* 2. PER-GAP ACOUSTIC LAG. Precondition: the source ON AXIS, in front of M0, so the lag is at the
   * edge of the physically permitted range and the integer ambiguity resolves — see
   * MICB_EXPECTED_LAG. Reported from M1, the pair that shares a clock with M0. */
  float lfrac, lR;
  if (hist_circmean_frac(&g_h_cal_lag[1], &lfrac, &lR)) {
    const float cm_per_sample = 34400.0f / (float)MICB_SAMPLE_RATE_HZ;
    const float bound = (float)MICB_SPACING_CM / cm_per_sample;
    char d[MQ_FMT_BYTES], e[MQ_FMT_BYTES], f[MQ_FMT_BYTES];
    MICB_LOG("[mic-bench] SETUP geometry:\n");
    /* Report the FRACTION and its concentration, never a mean or a median of the raw lag: the +-1
     * slip makes the raw distribution bimodal, so both of those statistics report the cluster that
     * happened to win rather than the geometry. */
    MICB_LOG("[mic-bench] SETUP    M1 fractional lag %ssmp, concentration R=%s over %u runs "
             "(slip-immune, so this is solid even when the raw lag is bimodal)\n",
             mq_fmt(a, sizeof(a), lfrac, 3), mq_fmt(b, sizeof(b), lR, 3),
             (unsigned)g_h_cal_lag[1].n);
    if (lR < 0.9f) {
      MICB_LOG("[mic-bench] SETUP     <- R below 0.9: the runs disagree about the fraction too, so "
               "the source is moving or the captures are not coherent. Fix that before reading "
               "anything below.\n");
    }
    /* Resolve the integer from the PHYSICAL BOUND, which is the operator's knowledge and the only
     * thing that can break the tie. `d` spacing permits |lag| <= d / (c/fs) samples, so only a couple
     * of candidates survive and the geometry picks between them. */
    MICB_LOG("[mic-bench] SETUP    %s cm spacing permits |lag| <= %ssmp, so the geometry is one of:",
             mq_fmt(d, sizeof(d), (float)MICB_SPACING_CM, 2), mq_fmt(e, sizeof(e), bound, 2));
    for (int k = -3; k <= 3; ++k) {
      const float cand = lfrac + (float)k;
      if (cand > bound + 1.0e-4f || cand < -bound - 1.0e-4f) continue;
      MICB_LOG(" %s", mq_fmt(f, sizeof(f), cand, 2));
    }
    MICB_LOG("\n");
    /* Say which one to take for each placement, rather than guessing at the operator's intent. */
    MICB_LOG("[mic-bench] SETUP    -> source ON AXIS in front of M0: take the candidate nearest "
             "+%ssmp.  BROADSIDE/equidistant: take the one nearest 0.  BEHIND: nearest -%ssmp. "
             "That value is -DMICB_EXPECTED_LAG.\n",
             mq_fmt(a, sizeof(a), bound, 2), mq_fmt(b, sizeof(b), bound, 2));
    MICB_LOG("[mic-bench] SETUP    (the fraction alone is %scm of path difference, so a nominally "
             "equidistant source that reads non-zero is simply off-centre by that much)\n",
             mq_fmt(a, sizeof(a), lfrac * cm_per_sample, 2));
  }

  /* 3. THE CHANNEL OFFSET. No stimulus precondition on the fractional part — it is immune to the
   * integer slip, so any coherent capture measures it. The integer part is reported as the RANGE of
   * whole samples observed, not as a constant, because it is genuinely not separable from the slip
   * (see MICB_CHAN_OFFSET) and printing a single number for it would be a fabrication. */
  float cfrac, cR;
  if (chan_frac_mean(&cfrac, &cR)) {
    int int_lo = 0, int_hi = 0, seen = 0;
    for (int m = 1; m < MICB_MICS; ++m) {
      float ilo, imean, ihi;
      if (!mic_cross_channel(m) || !hist_stat(&g_h_cal_off[m], &ilo, &imean, &ihi)) continue;
      const int a_lo = (int)((ilo >= 0.0f) ? (ilo + 0.5f) : (ilo - 0.5f));
      const int a_hi = (int)((ihi >= 0.0f) ? (ihi + 0.5f) : (ihi - 0.5f));
      if (!seen) { int_lo = a_lo; int_hi = a_hi; seen = 1; }
      else { if (a_lo < int_lo) int_lo = a_lo; if (a_hi > int_hi) int_hi = a_hi; }
    }
    /* State the conclusion the geometry forces, and how much of the measured lag it accounts for.
     * The integer range alone reads like an inconvenience; "8 of these 11 samples cannot be sound" is
     * the finding. */
    {
      float ilo, imean, ihi;
      for (int m = 1; m < MICB_MICS; ++m) {
        if (!mic_cross_channel(m) || !hist_stat(&g_h_cal_off[m], &ilo, &imean, &ihi)) continue;
        const float bnd = mic_acoustic_bound(m);
        const float mag = (imean < 0.0f) ? -imean : imean;
        char x[MQ_FMT_BYTES], y[MQ_FMT_BYTES], z[MQ_FMT_BYTES];
        if (mag > bnd + 1.0f) {
          MICB_LOG("[mic-bench] SETUP %s lag averages %ssmp but %s cm of spacing permits only "
                   "+-%ssmp: the excess is a STREAM OFFSET between the two I2S channels, not "
                   "acoustics. This is what the alignment exists to remove.\n", g_slot[m].name,
                   mq_fmt(x, sizeof(x), imean, 2), mq_fmt(y, sizeof(y), (float)MICB_SPACING_CM, 2),
                   mq_fmt(z, sizeof(z), bnd, 2));
        } else {
          MICB_LOG("[mic-bench] SETUP %s lag averages %ssmp, within the +-%ssmp that %s cm of spacing "
                   "permits — so it may be entirely ACOUSTIC and there may be no stream offset at "
                   "all. Move the source along the array axis to separate the two.\n",
                   g_slot[m].name, mq_fmt(x, sizeof(x), imean, 2), mq_fmt(z, sizeof(z), bnd, 2),
                   mq_fmt(y, sizeof(y), (float)MICB_SPACING_CM, 2));
        }
      }
    }
    MICB_LOG("[mic-bench] SETUP channel offset (any coherent capture; the fraction is slip-immune):\n");
    /* NOT printed as a -D flag, because it is not an input. It cancels out of the correction the
     * alignment applies (see MICB_CHAN_OFFSET), and it is re-randomised by every boot, so a measured
     * value can never reach the binary that would use it. It is reported because it DIAGNOSES: a
     * number this far past the physical bound is what proves the excess lag is an instrument artifact
     * rather than acoustics. Offering it as something to paste sent one session chasing a constant
     * that nothing reads. */
    MICB_LOG("[mic-bench] SETUP     channel offset ~%ssmp (integer %d..%d + fraction %s) — DIAGNOSTIC "
             "ONLY, nothing consumes this and it changes every boot\n",
             mq_fmt(a, sizeof(a), (float)int_lo + cfrac, 2), int_lo, int_hi,
             mq_fmt(b, sizeof(b), cfrac, 2));
    /* Concentration, not spread. The fraction wraps at +-0.5, so a set of estimates can be tightly
     * clustered and still show a near-1.0 min-to-max range; R is the length of the mean unit vector,
     * which is 1.0 for perfect agreement and 0 for uniform scatter and does not care where the
     * cluster sits relative to the wrap. */
    MICB_LOG("[mic-bench] SETUP     fraction %s, concentration R=%s over %u estimates — this is the "
             "part no integer shift can remove, and %s\n",
             mq_fmt(a, sizeof(a), cfrac, 2), mq_fmt(b, sizeof(b), cR, 3), g_chan_frac_n,
#if MICB_ALIGN_FRAC
             "MICB_ALIGN_FRAC is ON so it is being interpolated out");
#else
             "MICB_ALIGN_FRAC is OFF so it is still in every number above");
#endif
    if (cR < 0.7f) {
      MICB_LOG("[mic-bench] SETUP     <- R below 0.7: the estimates genuinely disagree (not a wrap "
               "artefact). Either the captures are not coherent enough (check the ALIGN gammas) or "
               "the source is moving. Do not bake this value in yet.\n");
    }
  } else if (MICB_MICS >= 3) {
    MICB_LOG("[mic-bench] SETUP channel offset: NO ESTIMATE YET — no capture had a cross-channel "
             "coherence above %d/100 with an unsaturated lag. Speak closer, or raise MICB_MAX_LAG "
             "above the offset (~12) so the peak is interior.\n", (int)(MICB_SLIP_MIN_GAMMA * 100.0f));
  }
}
#endif
#endif /* MICB_SUMMARY_EVERY */

/* ------------------------------------------------------------------------------- array ----------- */
#if MICB_MICS >= 2 && !MICB_PROBE_ONLY
/* Build the combined signal from the DC-free mics. Never writes g_mic, so every per-mic record stays
 * exactly what a single-mic build would have produced. Returns the lag applied (0 except in mode 3). */
static float combine_into(float *dst, uint32_t n, float lag_frac) {
  /* Fewer than two mics carrying signal: C == M0, so the deltas read 0 dB instead of a fabricated
   * gain. g_nsig, not g_nlive — see its declaration. */
  if (g_nsig < 2) {
    for (uint32_t i = 0; i < n; ++i) dst[i] = g_mic[0][i];
    return 0.0f;
  }
#if MICB_COMBINE == 0
  (void)lag_frac;
  for (uint32_t i = 0; i < n; ++i) dst[i] = g_mic[0][i];
  return 0.0f;
#elif MICB_COMBINE == 1
  /* Average over every LIVE mic, so four mics report the four-element figure (ideally 6 dB against
   * uncorrelated noise) instead of silently reporting a two-mic result. */
  (void)lag_frac;
  {
    const float scale = 1.0f / (float)g_nsig;
    for (uint32_t i = 0; i < n; ++i) {
      float acc = 0.0f;
      for (int m = 0; m < MICB_MICS; ++m) if (g_sig[m]) acc += g_gain[m] * g_mic[m][i];
      dst[i] = acc * scale;
    }
  }
  return 0.0f;
#elif MICB_COMBINE == 2
  /* Gradient across the M0/M1 pair only. A four-element "difference" is not a generalisation of a
   * gradient — it is a beamformer, and that belongs behind a measured beam pattern, not a combine knob. */
  (void)lag_frac;
  for (uint32_t i = 0; i < n; ++i) dst[i] = g_gain[0] * g_mic[0][i] - g_gain[1] * g_mic[1][i];
  return 0.0f;
#elif MICB_COMBINE == 3
  /* Integer-sample alignment of M1 onto M0 from the measured lag. Sub-sample alignment needs a
   * fractional-delay filter; the reported lag_frac says how much is being left on the table. */
  {
    int lag = (int)(lag_frac >= 0.0f ? (lag_frac + 0.5f) : (lag_frac - 0.5f));
    if (lag > MICB_MAX_LAG) lag = MICB_MAX_LAG;
    if (lag < -MICB_MAX_LAG) lag = -MICB_MAX_LAG;
    for (uint32_t i = 0; i < n; ++i) {
      const int j = (int)i + lag;
      const float rv = ((j >= 0) && ((uint32_t)j < n)) ? g_mic[1][j] : g_mic[1][i];
      dst[i] = 0.5f * (g_mic[0][i] + rv);
    }
    return (float)lag;
  }
#elif MICB_COMBINE == 4
  /* First-order ENDFIRE cardioid over the M0/M1 pair, null behind M1:
   *     C[n] = g0*M0[n] - g1*M1[n - D]
   * A source behind M1 reaches M1 first and M0 D samples later, so M0[n] = s[n-D] and M1[n-D] = s[n-D]
   * and the two cancel exactly — that is the null. A source in front of M0 gives s[n] - s[n-2D], which
   * survives. So M0 IS THE FRONT MIC: point it at the talker, with M1 behind it.
   *
   * Expect C to be much quieter and much brighter than M0 (`hf` rises): the on-axis response is
   * 2*|sin(2*pi*f*D/fs)|, i.e. -12.6 dB at 300 Hz and -2.3 dB at 1 kHz for D=1. That is the
   * differential high-pass, not a fault. Judge this mode ONLY by the front-to-back ratio of C between
   * two source positions, never by C's absolute level. */
  (void)lag_frac;
  {
    const int D = (int)MICB_ENDFIRE_DELAY;
    for (uint32_t i = 0; i < n; ++i) {
      const int j = (int)i - D;
      const float rear = g_mic[1][(j > 0) ? (uint32_t)j : 0u];
      dst[i] = g_gain[0] * g_mic[0][i] - g_gain[1] * rear;
    }
  }
  return (float)MICB_ENDFIRE_DELAY;
#elif MICB_COMBINE == 5
  /* ENDFIRE delay-and-sum over every contributing mic, same look direction as mode 4 (out past M0).
   * The front mic is delayed MOST — (N-1-m)*D for mic m — so an on-axis wavefront, which reaches the
   * mics in order, lands in phase at the summing point. */
  (void)lag_frac;
  {
    const int D = (int)MICB_ENDFIRE_DELAY;
    const int last = g_nsig - 1;
    const float scale = 1.0f / (float)g_nsig;
    for (uint32_t i = 0; i < n; ++i) {
      float acc = 0.0f;
      int k = last;                       /* delay index, counted over CONTRIBUTING mics only */
      for (int m = 0; m < MICB_MICS; ++m) {
        if (!g_sig[m]) continue;
        const int j = (int)i - k * D;
        acc += g_gain[m] * g_mic[m][(j > 0) ? (uint32_t)j : 0u];
        --k;
      }
      dst[i] = acc * scale;
    }
  }
  return (float)(MICB_ENDFIRE_DELAY * (g_nsig - 1));
#else
#error "MICB_COMBINE must be 0, 1, 2, 3, 4 or 5"
#endif
}

/* One-word reading of the noise-floor delta, so a log can be skimmed. The thresholds are physics:
 * averaging N mics gains 10*log10(N) against noise that is UNCORRELATED between them, and 0 dB
 * against correlated noise — which is what room tone and reverb are, and what a floor centred at a
 * few hundred hertz will be at any sane mic spacing (diffuse-field coherence sin(kd)/(kd) is 0.98 at
 * 340 Hz and 5 cm). */
/* The uncorrelated-noise prediction for d_nf, computed from the MEASURED per-mic floors rather than
 * assuming the mics are matched — because they are not, and the difference is not a detail.
 *
 * C = (1/N) * sum(m). If the floors are mutually uncorrelated their powers add, so
 *     P_C = (1/N^2) * sum(P_m)   and   d_nf = 10*log10(P_C / P_0).
 * Matched mics collapse this to the familiar -10*log10(N) (-3.01 dB for two). But if M0's floor
 * dominates — 10 dB above M1's, which is what this rig actually measures on some runs — the same
 * formula gives -5.7 dB, approaching -20*log10(N) = -6.02 in the limit. Judging that against -3.01
 * flags a correctly-behaving array as "suspicious: check gain match/polarity", i.e. sends you
 * looking for a wiring fault that is not there. The floors are already in hand; use them. */
static float array_ideal_nf_db(const mq_stats_t *sm) {
  double p = 0.0;
  for (int m = 0; m < MICB_MICS; ++m) {
    if (!g_sig[m]) continue;
    /* GAIN-CORRECTED floors, because C is built from gain-corrected mics. Using the raw floors here
     * predicts a mismatched array while the combine measures a matched one, and the two disagree by
     * exactly the correction: with M2's floor 8 dB hot and MICB_GAIN_DB_M2 = -8.2 bringing it into
     * line, the raw prediction is about +0.1 dB (averaging makes it worse) and the corrected one is
     * -4.8 dB (the full 10*log10(3)). A prediction that contradicts a working array is worse than no
     * prediction, since the natural reading is that the array is broken. */
    const double gnf = (double)g_gain[m] * (double)sm[m].nf;
    p += gnf * gnf;
  }
  const double p0 = (double)sm[0].nf * (double)sm[0].nf;
  if (!(p > 0.0) || !(p0 > 0.0)) return -10.0f * (float)log10((double)g_nsig);
  return (float)(10.0 * log10(p / ((double)g_nsig * (double)g_nsig) / p0));
}

/* What averaging N mics can do GIVEN the measured coherence of what they are hearing.
 *
 * `array_ideal_nf_db` above is the UNCORRELATED bound — the best case, when each mic's noise is its
 * own. Real room noise is not: two mics 2.15 cm apart see almost the same sound field, and averaging
 * two copies of one signal changes nothing. With pairwise coherence g the noise power in the average
 * is s^2 * (1 + (N-1)g) / N, so
 *      d_nf = 10*log10( (1 + (N-1)g) / N )
 * which is -10log10(N) at g = 0 and exactly 0 dB at g = 1.
 *
 * Printing this next to the uncorrelated bound is the difference between "the array is broken" and
 * "the array is working and the noise is coherent". Measured on silicon at g ~ 0.95 with 3 mics: this
 * predicts -0.15 dB and the array delivered -0.1 dB. The uncorrelated bound said -4.8 dB, and reading
 * THAT as the target would have sent someone hunting a fault that does not exist. */
static float array_coherent_nf_db(float gamma, int n) {
  if (n < 2) return 0.0f;
  float g = gamma;
  if (g < 0.0f) g = 0.0f;
  if (g > 1.0f) g = 1.0f;
  return 10.0f * (float)log10((1.0 + (double)(n - 1) * (double)g) / (double)n);
}

static const char *array_verdict(float d_nf_db, float ideal) {
  if (g_nsig < 2) {
    (void)d_nf_db; (void)ideal;
    return (g_nlive >= 2) ? "NO ARRAY: only 1 of the live slots carries signal (C forced to M0)"
                          : "single-mic(no-array)";
  }
#if MICB_COMBINE == 0
  (void)d_nf_db; (void)ideal;
  return "control(C==M0)";
#elif MICB_COMBINE == 1
  {
    /* nf is a percentile of a non-stationary signal, not a stationary RMS, so 3 dB of slack below the
     * prediction is measurement spread rather than a fault. Beyond that, something is genuinely
     * cancelling — which for an AVERAGE means a polarity or slot problem. */
    if (d_nf_db < ideal - 3.0f) return "beyond-ideal(suspicious: check gain match/polarity)";
    if (d_nf_db < ideal + 1.5f) return "uncorrelated(at the ideal for these mics' floors)";
    if (d_nf_db < -0.7f) return "partly-correlated";
    if (d_nf_db < 0.7f) return "correlated(room/reverb: averaging cannot help)";
    return "worse(check polarity/SEL/spacing)";
  }
#else
  (void)ideal;
  if (d_nf_db < -0.7f) return "gradient-lowered-the-floor";
  if (d_nf_db < 0.7f) return "no-change";
  return "raised-the-floor(normal for a gradient: it high-passes)";
#endif
}
#endif /* MICB_MICS >= 2 */

/* ------------------------------------------------------------------------------- entry ----------- */

void app_init(void) {
#if defined(TERMINAL_DEVICE_UART0)
  /* init_test brings up the PLL AND the UART divisor, and must run before any printf on real silicon
   * (a printf to an unconfigured UART hangs the core). */
  init_test((uint64_t)MICB_TARGET_FREQ_HZ);
#endif
  MICB_LOG("[mic-bench] DSP 25 microphone audio-quality benchmark\n");
  MICB_LOG("[mic-bench] build: mics=%d combine=%d capture=%ums (%u samples) frame=%u (%u frames) "
           "pct(nf/act)=%u/%u clip=%d/1e3 tone=%uHz warmup=%ums\n",
           (int)MICB_MICS, (int)MICB_COMBINE, (unsigned)MICB_CAPTURE_MS,
           (unsigned)MICB_CAPTURE_SAMPLES, (unsigned)MICB_FRAME_SAMPLES, (unsigned)MICB_NUM_FRAMES,
           (unsigned)MICB_NF_PCT, (unsigned)MICB_ACT_PCT, (int)(MICB_CLIP_LEVEL * 1000.0f),
           (unsigned)MICB_TONE_HZ, (unsigned)MICB_WARMUP_MS);
#if defined(TERMINAL_DEVICE_UART0)
  MICB_LOG("[mic-bench] PLL @ %u MHz (ratio %u x %u MHz)\n",
           (unsigned)(MICB_TARGET_FREQ_HZ / 1000000u),
           (unsigned)(MICB_TARGET_FREQ_HZ / SYS_CLK_FREQ), (unsigned)(SYS_CLK_FREQ / 1000000u));
#endif

  /* Channel A. M0/M1 are its left and right slots: one config_I2S and one clkdiv cover both, because
   * set_I2S_sample_freq already frames the link as stereo (mclk = rate * bits * 2), so the right slot
   * is already on the wire and being clocked whether or not anything reads it. */
  config_I2S(MICB_MIC_CHANNEL, &g_i2s_params_mic);
  set_I2S_sample_freq(MICB_MIC_CHANNEL, (uint64_t)target_frequency,
                      (uint64_t)MICB_SAMPLE_RATE_HZ, (uint8_t)MICB_BITDEPTH);
#if MICB_MICS >= 3
  /* Channel B, configured IMMEDIATELY after channel A and with the identical divider.
   *
   * Both channels' BCLK is sys_clk / (2*(N+1)) — two integer dividers counting the SAME core clock —
   * so an identical clkdiv means an identical frequency and no possible drift. What is left is a fixed
   * phase offset set by how far apart the two dividers started, which is why these two calls are
   * adjacent with nothing between them: the gap is printed below in core cycles, and at 750 MHz a few
   * cycles is single-digit nanoseconds against a 62.5 us sample period.
   *
   * Whether that reasoning survives contact with the silicon is exactly what the TRACK line measures. */
  const uint64_t t_a = micb_rdcycle();
  config_I2S(MICB_MIC_CHANNEL_B, &g_i2s_params_mic);
  set_I2S_sample_freq(MICB_MIC_CHANNEL_B, (uint64_t)target_frequency,
                      (uint64_t)MICB_SAMPLE_RATE_HZ, (uint8_t)MICB_BITDEPTH);
  const uint64_t t_b = micb_rdcycle();
  MICB_LOG("[mic-bench] I2S configured ch=%d and ch=%d rate=%uHz bits=%u clkdiv=%d "
           "(ch-start gap=%lu cycles = %lu ns; one sample = %lu ns)\n",
           MICB_MIC_CHANNEL, MICB_MIC_CHANNEL_B, (unsigned)MICB_SAMPLE_RATE_HZ,
           (unsigned)MICB_BITDEPTH, (int)MICB_CLKDIV, (unsigned long)(t_b - t_a),
           (unsigned long)((t_b - t_a) * 1000000000ull / (uint64_t)MICB_TARGET_FREQ_HZ),
           (unsigned long)(1000000000ull / (uint64_t)MICB_SAMPLE_RATE_HZ));
#else
  MICB_LOG("[mic-bench] I2S configured ch=%d rate=%uHz bits=%u clkdiv=%d\n",
           MICB_MIC_CHANNEL, (unsigned)MICB_SAMPLE_RATE_HZ, (unsigned)MICB_BITDEPTH,
           (int)MICB_CLKDIV);
#endif
#if MICB_DUMP_REGS
  dump_i2s_regs("post-config");
#endif

  /* Gain corrections, and a printed record of them: a level that has been scaled without saying so is
   * how a calibrated log becomes indistinguishable from an uncalibrated one three sessions later. */
  {
    static const float gdb[4] = { 0.0f, MICB_GAIN_DB_M1, MICB_GAIN_DB_M2, MICB_GAIN_DB_M3 };
    for (int m = 0; m < MICB_MICS; ++m) {
      g_gain_db[m] = gdb[m];
      g_gain[m] = (float)pow(10.0, (double)gdb[m] / 20.0);
    }
#if MICB_MICS >= 2
    {
      char b[MQ_FMT_BYTES];
      MICB_LOG("[mic-bench] gain correction (applied to C only):");
      for (int m = 0; m < MICB_MICS; ++m) {
        MICB_LOG(" %s=%sdB", g_slot[m].name, mq_fmt(b, sizeof(b), gdb[m], 2));
      }
      MICB_LOG("%s\n", (MICB_COMBINE == 2 || MICB_COMBINE == 4)
                       ? "  (this mode is a DIFFERENCE: calibrate before believing any null depth)"
                       : "");
    }
#endif
  }

  mic_probe();
  MICB_LOG("[mic-bench] %d of %d slots live\n", g_nlive, (int)MICB_MICS);
#if MICB_MICS >= 2 && (MICB_COMBINE == 4 || MICB_COMBINE == 5)
  MICB_LOG("[mic-bench] ENDFIRE mode=%d delay=%d smp/element (%s cm spacing assumed) — M0 IS THE "
           "FRONT MIC: point M0 at the source, the rest behind it along one axis\n",
           (int)MICB_COMBINE, (int)MICB_ENDFIRE_DELAY,
           (MICB_ENDFIRE_DELAY == 1) ? "2.15" : "2.15 x delay");
#endif
#if MICB_MICS == 1
  MICB_LOG("[mic-bench] single microphone. Next step: -DMICB_MICS=2 (both mics on ch%d's L/R slots, "
           "guaranteed sample-locked). Only after that: -DMICB_MICS=4.\n", MICB_MIC_CHANNEL);
#elif MICB_MICS == 2
  if (g_nlive < 2) {
    MICB_LOG("[mic-bench] check: mic B SEL strapped to VDD, BCLK/WS/SDIN shared with mic A, "
             "rx_force_left=0\n");
  }
#else
  if (g_nlive < 4) {
    MICB_LOG("[mic-bench] check: ch%d wiring, and that ch%d is CLOCKED at all — this I2S master may "
             "only generate BCLK/WS while its TX FIFO has data (see dsp-i2s-test KEEP_CLOCK)\n",
             MICB_MIC_CHANNEL_B, MICB_MIC_CHANNEL_B);
  }
#endif

  /* One-time warm-up. The first capture after power-up is the microphone's high-pass settling, not
   * audio: measured dc=0.0136fs, drift=-0.0361fs, hf=-51 dB (a ~7 Hz centroid). Throwing it away here
   * keeps run 1 usable and keeps the rolling summary from opening with a 40 dB spread. */
  if (g_live[0]) {
    MICB_LOG("[mic-bench] warm-up: discarding %u ms so run 1 is not the mic's HPF settling...\n",
             (unsigned)MICB_WARMUP_MS);
    for (uint32_t i = 0; i < MICB_WARMUP_SAMPLES; i += 2u) mic_drain_block();
  }

  MICB_LOG("[mic-bench] stimuli: (1) silence -> read nf/dc/drift/hf  (2) speech -> act/snr/crest/clip"
           "  (3) %uHz tone -> level/sinad/thd\n", (unsigned)MICB_TONE_HZ);
}

int app_main(void) {
  unsigned run = 0u;
  for (;;) {
    run++;
    const uint32_t rate = capture_window();

    /* One line per capture that says what was measured about the CAPTURE itself, before any audio
     * conclusion is drawn from it. A rate that is not the configured one invalidates every number
     * below, so it is printed first rather than buried. */
    const int rate_off = (rate == 0u) ||
                         (rate > (MICB_SAMPLE_RATE_HZ + MICB_SAMPLE_RATE_HZ / 20u)) ||
                         (rate < (MICB_SAMPLE_RATE_HZ - MICB_SAMPLE_RATE_HZ / 20u));
    MICB_LOG("[mic-bench] --- run=%u captured %u samples, measured rate=%uHz (configured %u)%s\n",
             run, (unsigned)MICB_CAPTURE_SAMPLES, (unsigned)rate, (unsigned)MICB_SAMPLE_RATE_HZ,
             rate_off ? "  <- OFF BY >5%: every level below is misinterpreted" : "");

#if MICB_MICS >= 2 && MICB_SLIP_FIX && !MICB_PROBE_ONLY
    /* Before ANY measurement: align every element against M0 — take out each stream's integer FIFO
     * slip and, for the elements on the second I2S channel, the constant fractional stream offset.
     * Must happen here: mq_analyze removes DC in place, and correcting alignment after the per-mic
     * records were computed would leave those records describing a different signal than the pair and
     * array records do. */
    mic_align_all(run);
#endif

    /* Per-mic records, every one from the SAME capture and the same code path, so a single-mic build's
     * M0 line and a four-mic build's M0 line are directly comparable. */
    mq_stats_t sm[MICB_MICS];
    for (int m = 0; m < MICB_MICS; ++m) {
      mq_analyze(g_mic[m], MICB_CAPTURE_SAMPLES, MICB_FRAME_SAMPLES, MICB_CLIP_LEVEL,
                 MICB_NF_PCT, MICB_ACT_PCT, MICB_ACTIVE_MULT,
                 g_frames, MICB_NUM_FRAMES, &sm[m]);
      /* The alignment pass already took the DC out, so mq_analyze saw an AC-coupled window and reports
       * dc ~ 0. Fold the earlier removal back into the REPORTED figure — the DC offset is a real
       * property of the part (channel B's mics carry 4-5% of full scale of it) and losing it from the
       * log to an implementation detail of the alignment would be a silent regression. */
      sm[m].dc += g_dc_pre[m];
      if (g_live[m]) print_stats(run, g_slot[m].name, &sm[m]);
    }
    const mq_stats_t sl = sm[0];

    /* Re-decide, from the levels just measured, which slots actually carry a microphone signal. Done
     * every run rather than once at boot so a mic that stops mid-session is caught in the run it
     * stops, and so the ARRAY line below can never average a live mic with a constant and call the
     * resulting -6.02 dB an array gain. */
    g_nsig = 0;
    for (int m = 0; m < MICB_MICS; ++m) {
      g_sig[m] = g_live[m] && (sm[m].rms > 0.0f) &&
                 (mq_dbfs(sm[m].rms) > (float)MICB_SIGNAL_FLOOR_DBFS);
      if (g_sig[m]) g_nsig++;
    }
    if (g_nsig != g_nlive) {
      MICB_LOG("[mic-bench] run=%u SLOTS %d of %d carry signal (>%d dBFS) —", run, g_nsig, g_nlive,
               (int)MICB_SIGNAL_FLOOR_DBFS);
      for (int m = 0; m < MICB_MICS; ++m) {
        if (g_live[m] && !g_sig[m]) MICB_LOG(" %s=constant", g_slot[m].name);
      }
      MICB_LOG("; excluded from C and from the ARRAY verdict\n");
    }

#if MICB_PROBE_ONLY
    /* LIVENESS ONLY. One line per slot saying whether a microphone is actually on it, and nothing
     * else: no pair, no combine, no array, no tone. This exists because "is the new mic wired and
     * clocked?" is a different question from "how good is the array?", and answering the second one
     * while the first is still open is how a wiring fault gets read as an acoustic result.
     *
     * Level is the test, not the FIFO. A slot nobody drives still FILLS — the peripheral clocks in
     * whatever is on SDIN during that slot — so the boot probe says ALIVE either way. What an undriven
     * slot looks like is a CONSTANT (a floating line reads 0xFFFFFFFF every frame, which is exactly
     * zero once DC is removed); a real microphone in a quiet room sits around -50 dBFS. The two are
     * 100 dB apart, so this needs no tuning.
     *
     * The remaining hazard is on the OTHER side of the probe: read_I2S_rx spins while its FIFO is
     * empty, so a channel whose clock generator never starts would hang the core. mic_probe() has
     * already used the non-blocking empty flag under a bounded spin to drop such a slot from the read
     * set before we ever get here. */
    for (int m = 0; m < MICB_MICS; ++m) {
      char rms[MQ_FMT_BYTES], pk[MQ_FMT_BYTES], nf[MQ_FMT_BYTES], dc[MQ_FMT_BYTES];
      const char *verdict;
      if (!g_live[m]) {
        verdict = "NO DATA — the FIFO never filled: channel not clocked, or nothing wired";
      } else if (!g_sig[m]) {
        verdict = "CONSTANT — the slot is clocked but no mic drives it: check SEL and SDIN";
      } else {
        verdict = "ALIVE — a microphone is on this slot";
      }
      MICB_LOG("[mic-bench] run=%u PROBE %s (ch%d %s) rms=%s peak=%s nf=%s dc=%sfs  %s\n",
               run, g_slot[m].name, g_slot[m].ch, (g_slot[m].side == I2S_LEFT) ? "L" : "R",
               mq_fmt(rms, sizeof(rms), mq_dbfs(sm[m].rms), 1),
               mq_fmt(pk, sizeof(pk), mq_dbfs(sm[m].peak), 1),
               mq_fmt(nf, sizeof(nf), mq_dbfs(sm[m].nf), 1),
               mq_fmt(dc, sizeof(dc), sm[m].dc, 4), verdict);
    }
    MICB_LOG("[mic-bench] run=%u PROBE-ONLY: %d of %d slots carry a microphone. Tap each mic in turn "
             "and watch which peak moves — that is the only check that a slot is mapped to the mic you "
             "think it is. Rebuild without -DMICB_PROBE_ONLY=ON to measure the array.\n",
             run, g_nsig, (int)MICB_MICS);
    (void)sl;
#else

#if MICB_MICS >= 2
    /* The gain calibration, as the numbers to paste into the build rather than a ratio to convert by
     * hand. Valid ONLY with every mic the same distance from one steady source — with a directional
     * source the level differences are the signal, and "calibrating" them away would null the thing
     * you are trying to hear. Printed every run because it costs nothing and because a calibration
     * taken from one 2-second window of a wandering room is worth less than the agreement between
     * several. */
    if (g_nsig >= 2 && (sm[0].rms > 0.0f)) {
      char b[MQ_FMT_BYTES];
      MICB_LOG("[mic-bench] run=%u CAL (only valid if all mics are equidistant from one source):", run);
      for (int m = 1; m < MICB_MICS; ++m) {
        if (!g_sig[m]) continue;
        MICB_LOG(" -DMICB_GAIN_DB_M%d=%s", m, mq_fmt(b, sizeof(b), mq_db_ratio(sm[0].rms, sm[m].rms), 2));
      }
      MICB_LOG("\n");
    }
#endif
#if MICB_TONE_HZ > 0
    analyze_tone(run, g_slot[0].name, g_mic[0]);
#endif

#if MICB_MICS >= 2
    /* Every pair's geometry. All signals are DC-free by now (mq_analyze removed it in place), which is
     * required: a per-part DC offset of a few percent of full scale would otherwise dominate the
     * correlation and pin every lag at 0 with a gamma near 1 — i.e. look like a perfect array.
     *
     * 1 sample = 2.14 cm of path difference at 16 kHz. Printing the distance is what turns a lag into
     * something checkable against where the mics physically are. */
    const float cm_per_sample = 34400.0f / (float)MICB_SAMPLE_RATE_HZ;
    mq_pair_t pr;                       /* M0-M1: the reference pair, used by the combine */
    mq_pair_analyze(g_mic[0], g_mic[1], MICB_CAPTURE_SAMPLES, MICB_MAX_LAG, MICB_XCORR_WINDOW, &pr);
    for (int i = 0; i < MICB_MICS; ++i) {
      for (int j = i + 1; j < MICB_MICS; ++j) {
        if (!g_live[i] || !g_live[j]) continue;
        /* A pair involving a slot that carries no SIGNAL has nothing to report: its lag is wherever
         * the scan started, its gamma is 0, and its level difference is -240 dB. Printing four such
         * records per run (pair, LEVEL, TRACK and a null-depth estimate) buries the mics that do
         * work — which is the state the board is in whenever one element is unpopulated or dead. The
         * SLOTS line above already names the slot, so nothing is being hidden. */
        if (!g_sig[i] || !g_sig[j]) continue;
        mq_pair_t p;
        mq_pair_analyze(g_mic[i], g_mic[j], MICB_CAPTURE_SAMPLES, MICB_MAX_LAG, MICB_XCORR_WINDOW, &p);
        /* Level difference from the FULL-capture RMS (sm), not from the correlation sub-window inside
         * `p`. They are not the same number: `p` covers MICB_XCORR_WINDOW samples chosen for a stable
         * lag estimate, and in a room that wanders the two disagreed by up to 4 dB — so this line and
         * the CAL line, both claiming to report "the level mismatch", printed different values and the
         * gain correction derived from one did not match the ceiling predicted by the other. gamma and
         * lag still come from the sub-window, which is what it is for. */
        const float dlev = mq_db_ratio(sm[j].rms, sm[i].rms);
        /* Two DIFFERENT mics a few centimetres apart never agree this exactly, even on a distant
         * coherent source — reverb and their own tolerances forbid it. What does produce a perfect
         * match is both FIFOs carrying the SAME slot, which is what rx_force_left=1 would do, and it
         * would otherwise read as a flawless array. */
        const int suspect_same = (p.gamma > 0.9995f) &&
                                 (p.lag_frac > -0.02f) && (p.lag_frac < 0.02f) &&
                                 (dlev > -0.05f) && (dlev < 0.05f);
        /* A level mismatch is the hard ceiling on every null this array will ever steer, and it is a
         * number, not a worry: subtracting two copies of the same sound that differ in gain by a
         * factor g leaves |1-g| of it behind, so the deepest possible null is 20*log10(|1-g|)
         * REGARDLESS of geometry, delay accuracy or how good the beamformer is. 1 dB of mismatch
         * caps it at -19 dB; 5 dB caps it at -7 dB, which is not a null. Printed here because this is
         * the pair line, i.e. where the mismatch is measured, and because it must be corrected (a
         * per-mic scale factor) before any directional work is attempted.
         *
         * The mismatch that governs the null is the one REMAINING after the gain correction, not the
         * raw one — otherwise this line keeps demanding a calibration that has already been applied,
         * and a warning that cannot be satisfied stops being read. Both are printed: the raw number
         * is what the CAL line is derived from, the residual is what caps the null. */
        const float dlev_res = dlev + g_gain_db[j] - g_gain_db[i];
        const float mag = (dlev_res < 0.0f) ? -dlev_res : dlev_res;
        const float g_ratio = (float)pow(10.0, -(double)mag / 20.0);
        /* Clamped: a residual that rounds to zero sent this to -97 dB, which reads as a measured null
         * depth and is not one — it is 20*log10 of a rounding error. Gain match stops being the
         * binding constraint long before -40 dB (phase match and the room take over), so reporting
         * "better than -40" is the honest ceiling. */
        float null_cap = 20.0f * (float)log10((double)(1.0f - g_ratio));
        if (!(null_cap > -40.0f)) null_cap = -40.0f;
        /* INTEGER SAMPLE SLIP between the two streams, relative to the first trustworthy run.
         *
         * A stationary source cannot move the lag. So once a reference lag is established, any
         * subsequent lag that differs from it by very nearly a WHOLE number of samples is the FIFO
         * offset described at mic_resync, not acoustics — and unlike a fractional wander it is
         * unambiguous, because 1.0 sample is 2.15 cm of apparent path difference appearing between
         * two captures. Reported rather than corrected: a silent correction would hide whether the
         * resync works, which is the thing that has to be established. */
        if (i == 0 && j == 1 && p.gamma > MICB_SLIP_MIN_GAMMA) {
          const float d_lag = p.lag_frac - (float)MICB_EXPECTED_LAG;
          const int resid = (int)((d_lag >= 0.0f) ? (d_lag + 0.5f) : (d_lag - 0.5f));
          const float frac = d_lag - (float)resid;
          if (resid != 0 && frac > -MICB_SLIP_TOLERANCE && frac < MICB_SLIP_TOLERANCE) {
            char sl[MQ_FMT_BYTES];
            MICB_LOG("[mic-bench] run=%u SLIP %+d sample(s) STILL PRESENT after correction — lag %s "
                     "vs expected %d.0; either MICB_EXPECTED_LAG does not match the geometry you "
                     "built, or the slip changed within the run\n", run, resid,
                     mq_fmt(sl, sizeof(sl), p.lag_frac, 2), (int)MICB_EXPECTED_LAG);
          }
          g_lag_ref = p.lag_frac;
          g_lag_ref_valid = 1;
        }

        /* DOES THE SOURCE DIRECTION MATCH WHAT THE BUILD WAS TOLD? Checked from LEVEL, not lag.
         *
         * This is the one guard that catches a wrong MICB_EXPECTED_LAG sign, and it is needed because
         * the LAG cannot catch it. At 2.15 cm spacing one element gap is exactly 1.000 sample, so a
         * source in front and a source behind differ by a whole number of samples — indistinguishable
         * from the +-1 FIFO slip. The alignment therefore accepts either, and if the sign is wrong it
         * quietly RE-TIMES a rear wavefront into a front one: measured on silicon, a back-facing
         * capture came out with M1 at +1.08 and M2 at +2.01 after alignment, identical to the
         * front-facing run, so the beamformer added the rear source coherently and the front/back
         * ratio read +0.8 dB (which was stale gain calibration, not directivity).
         *
         * Level has no such ambiguity: the nearer mic is louder, full stop. Front of M0 -> M0 louder
         * -> dlev < 0. Behind -> M1 louder -> dlev > 0. Measured swing was a clean 3 dB. The deadband
         * keeps a near-broadside source from tripping it. */
        if (i == 0 && j == 1 && ((float)MICB_EXPECTED_LAG > 0.1f || (float)MICB_EXPECTED_LAG < -0.1f)) {
          const float want_neg = ((float)MICB_EXPECTED_LAG > 0.0f);
          const int disagrees = want_neg ? (dlev_res > 0.5f) : (dlev_res < -0.5f);
          if (disagrees) {
            char q[MQ_FMT_BYTES], w[MQ_FMT_BYTES];
            MICB_LOG("[mic-bench] run=%u DIRECTION MISMATCH: MICB_EXPECTED_LAG=%s says the source is "
                     "%s of M0, but M1 is the LOUDER mic by %sdB, so it is %s. The alignment cannot "
                     "tell these apart (one gap is exactly 1.000 sample, same as a slip) and has "
                     "silently re-timed this capture into the other direction — every beam number "
                     "below is invalid. Rebuild with the opposite sign.\n", run,
                     mq_fmt(q, sizeof(q), (float)MICB_EXPECTED_LAG, 2),
                     want_neg ? "IN FRONT" : "BEHIND",
                     mq_fmt(w, sizeof(w), dlev_res, 2), want_neg ? "behind" : "in front");
          }
        }

        char dl[MQ_FMT_BYTES], gm[MQ_FMT_BYTES], lg[MQ_FMT_BYTES], cm[MQ_FMT_BYTES], nc[MQ_FMT_BYTES];
        char rs[MQ_FMT_BYTES];
        if (mag > 1.0f) {
          MICB_LOG("[mic-bench] run=%u LEVEL %s vs %s raw %sdB residual-after-gain %sdB -> deepest "
                   "possible null %sdB (calibrate: see the CAL line)\n", run,
                   g_slot[j].name, g_slot[i].name, mq_fmt(dl, sizeof(dl), dlev, 1),
                   mq_fmt(rs, sizeof(rs), dlev_res, 2), mq_fmt(nc, sizeof(nc), null_cap, 1));
        } else if (g_gain_db[j] != g_gain_db[i]) {
          MICB_LOG("[mic-bench] run=%u LEVEL %s vs %s raw %sdB residual-after-gain %sdB -> null "
                   "ceiling %sdB (calibrated)\n", run, g_slot[j].name, g_slot[i].name,
                   mq_fmt(dl, sizeof(dl), dlev, 1), mq_fmt(rs, sizeof(rs), dlev_res, 2),
                   mq_fmt(nc, sizeof(nc), null_cap, 1));
        }
        MICB_LOG("[mic-bench] run=%u pair %s-%s d=%sdB gamma=%s lag=%ssmp (%scm)%s%s\n", run,
                 g_slot[j].name, g_slot[i].name,
                 mq_fmt(dl, sizeof(dl), dlev, 1),
                 mq_fmt(gm, sizeof(gm), p.gamma, 3),
                 mq_fmt(lg, sizeof(lg), p.lag_frac, 2),
                 mq_fmt(cm, sizeof(cm), p.lag_frac * cm_per_sample, 1),
                 (g_slot[i].ch != g_slot[j].ch) ? "  [CROSS-CHANNEL]" : "",
                 suspect_same ? "  <- IDENTICAL: one source in both slots? (check rx_force_left=0, "
                                "and that the two SEL pins differ)" : "");

#if MICB_TRACK_BLOCKS > 0u
        /* Drift within one capture — the clocking test. Same-channel pairs share BCLK and WS, so their
         * slope is the instrument's own noise floor: measure it there FIRST, then judge a
         * cross-channel pair against that number rather than against zero. */
        mq_track_t tr;
        mq_pair_track(g_mic[i], g_mic[j], MICB_CAPTURE_SAMPLES, MICB_MAX_LAG, MICB_TRACK_BLOCKS,
                      (float)MICB_SAMPLE_RATE_HZ, &tr);
        /* Only when the WORST block correlated. A block that found no peak contributes a lag pinned
         * at the search limit, and one such block among eight produced `spread=205.94` on silicon —
         * a number with no meaning printed in a field the operator reads as drift. The drift verdict
         * is only as good as the least trustworthy block in it. */
        /* Two independent guards, because they catch different failures. Coherence catches a block
         * that found no peak. But it does NOT catch an ALIASED peak: on silicon, run 1 block 1
         * reported a lag of -241.05 samples at gamma_min 0.951 — a genuinely high correlation 241
         * samples away, which printed as spread=242.33 and slope=8653 ppm in fields the operator
         * reads as clock drift. So also require every measured lag to be PHYSICALLY POSSIBLE: the
         * array's own spacing bounds any real acoustic lag, and one sample of slip on top. */
        const float trk_bound = (float)MICB_SPACING_CM / cm_per_sample * (float)MICB_MICS + 1.5f;
        const int trk_sane = (tr.lag_min > -trk_bound) && (tr.lag_max < trk_bound);
        if (tr.blocks >= 2u && tr.gamma_min > MICB_SLIP_MIN_GAMMA && trk_sane) {
          char f[MQ_FMT_BYTES], l[MQ_FMT_BYTES], sp[MQ_FMT_BYTES], pp[MQ_FMT_BYTES];
          char gmn[MQ_FMT_BYTES], sprd[MQ_FMT_BYTES];
          MICB_LOG("[mic-bench] run=%u TRACK %s-%s blocks=%u lag %s->%s spread=%s slope=%ssmp/s "
                   "(%sppm) gamma_min=%s\n", run, g_slot[j].name, g_slot[i].name,
                   (unsigned)tr.blocks,
                   mq_fmt(f, sizeof(f), tr.lag_first, 2), mq_fmt(l, sizeof(l), tr.lag_last, 2),
                   mq_fmt(sprd, sizeof(sprd), tr.lag_max - tr.lag_min, 2),
                   mq_fmt(sp, sizeof(sp), tr.lag_slope, 3), mq_fmt(pp, sizeof(pp), tr.ppm, 1),
                   mq_fmt(gmn, sizeof(gmn), tr.gamma_min, 3));
        } else if (tr.blocks >= 2u) {
          char gmn[MQ_FMT_BYTES], lmn[MQ_FMT_BYTES], lmx[MQ_FMT_BYTES];
          MICB_LOG("[mic-bench] run=%u TRACK %s-%s not reported: %s (worst block gamma=%s, block lags "
                   "span %s..%s, physically possible is +-%dsmp) — the spread and slope would be "
                   "meaningless\n", run, g_slot[j].name, g_slot[i].name,
                   trk_sane ? "a block found no usable peak" : "a block's peak is an ALIAS, too far "
                              "away to be acoustic",
                   mq_fmt(gmn, sizeof(gmn), tr.gamma_min, 3),
                   mq_fmt(lmn, sizeof(lmn), tr.lag_min, 2), mq_fmt(lmx, sizeof(lmx), tr.lag_max, 2),
                   (int)trk_bound);
        }
#endif
      }
    }

    const float used_lag = combine_into(g_comb, MICB_CAPTURE_SAMPLES, pr.lag_frac);
    mq_stats_t sc;
    mq_analyze(g_comb, MICB_CAPTURE_SAMPLES, MICB_FRAME_SAMPLES, MICB_CLIP_LEVEL,
               MICB_NF_PCT, MICB_ACT_PCT, MICB_ACTIVE_MULT,
               g_frames, MICB_NUM_FRAMES, &sc);
    print_stats(run, "C ", &sc);
#if MICB_TONE_HZ > 0
    analyze_tone(run, "C ", g_comb);
#endif


    /* THE ARRAY LINE. Everything else describes a signal; this is the only line that answers "did
     * the second mic help?", and it is a difference taken inside one capture, so the room, the
     * talker and the gain settings are identical on both sides of it. */
    const float d_rms = mq_dbfs(sc.rms) - mq_dbfs(sl.rms);
    const float d_nf = mq_dbfs(sc.nf) - mq_dbfs(sl.nf);
    const float d_snr = mq_snr_db(&sc) - mq_snr_db(&sl);
    {
      /* The prediction is printed next to the measurement: a verdict word that cannot be checked
       * against the number it was derived from is not a measurement, it is an opinion. */
      const float ideal = array_ideal_nf_db(sm);
      const float coh = array_coherent_nf_db(pr.gamma, g_nsig);
      char a[MQ_FMT_BYTES], b[MQ_FMT_BYTES], c[MQ_FMT_BYTES], d[MQ_FMT_BYTES], e[MQ_FMT_BYTES];
      char f[MQ_FMT_BYTES], g[MQ_FMT_BYTES];
      /* TWO predictions, because one of them alone is misleading. `uncorr` is the best case (each
       * mic's noise its own); `at gamma` is what the coherence actually measured in this capture
       * permits. Judge d_nf against the SECOND. */
      MICB_LOG("[mic-bench] run=%u ARRAY mode=%d lag_used=%s d_rms=%sdB d_nf=%sdB "
               "(uncorr %sdB, at gamma=%s %sdB) d_snr=%sdB %s\n",
               run, (int)MICB_COMBINE, mq_fmt(d, sizeof(d), used_lag, 0),
               mq_fmt(a, sizeof(a), d_rms, 1), mq_fmt(b, sizeof(b), d_nf, 1),
               mq_fmt(e, sizeof(e), ideal, 1), mq_fmt(f, sizeof(f), pr.gamma, 2),
               mq_fmt(g, sizeof(g), coh, 1),
               mq_fmt(c, sizeof(c), d_snr, 1), array_verdict(d_nf, coh));
    }
#endif /* MICB_MICS >= 2 */

#if MICB_SUMMARY_EVERY > 0u && !MICB_PROBE_ONLY
    hist_push(&g_h_rms_l, mq_dbfs(sl.rms));
    hist_push(&g_h_nf_l, mq_dbfs(sl.nf));
    hist_push(&g_h_snr_l, mq_snr_db(&sl));
#if MICB_MICS >= 2
    hist_push(&g_h_rms_c, mq_dbfs(sc.rms));
    hist_push(&g_h_nf_c, mq_dbfs(sc.nf));
    hist_push(&g_h_snr_c, mq_snr_db(&sc));
    hist_push(&g_h_dnf, d_nf);
    hist_push(&g_h_dsnr, d_snr);
    hist_push(&g_h_gamma, pr.gamma);
#if MICB_CAL_REPORT
    /* Accumulate the calibration constants. Gains come from the full-capture RMS ratio (the same
     * quantity the CAL line prints); the lag and offset come from the alignment pass, and only from
     * runs it judged measurable — folding an uncorrelated window's lag into a calibration average is
     * how a random number acquires the authority of a mean. */
    for (int m = 1; m < MICB_MICS; ++m) {
      if (g_sig[m] && (sm[0].rms > 0.0f) && (sm[m].rms > 0.0f)) {
        hist_push(&g_h_cal_gain[m], mq_db_ratio(sm[0].rms, sm[m].rms));
      }
      if (!g_al[m].valid) continue;
      hist_push(&g_h_cal_lag[m], g_al[m].meas);
      hist_push(&g_h_cal_off[m], g_al[m].off);
      if (mic_cross_channel(m)) hist_push(&g_h_cal_frac, g_al[m].frac);
    }
#endif
#endif
    if ((run % MICB_SUMMARY_EVERY) == 0u) {
      MICB_LOG("[mic-bench] ===== summary over the last %u runs =====\n",
               (unsigned)MICB_SUMMARY_EVERY);
      hist_print("M0 rms dB", &g_h_rms_l);
      hist_print("M0 nf dB", &g_h_nf_l);
      hist_print("M0 snr dB", &g_h_snr_l);
#if MICB_MICS >= 2
      hist_print("C rms dB", &g_h_rms_c);
      hist_print("C nf dB", &g_h_nf_c);
      hist_print("C snr dB", &g_h_snr_c);
      hist_print("d_nf dB", &g_h_dnf);
      hist_print("d_snr dB", &g_h_dsnr);
      hist_print("gamma", &g_h_gamma);
#endif
#if MICB_MICS >= 2 && MICB_CAL_REPORT
      print_setup_block(run);
#endif
    }
#endif /* MICB_SUMMARY_EVERY */
#endif /* !MICB_PROBE_ONLY */

#if MICB_RUNS > 0u
    if (run >= MICB_RUNS) break;
#endif
  }

  MICB_LOG("[mic-bench] done after %u runs\n", run);
  while (1) { __asm__ volatile("wfi"); }
  return 0;
}

int main(void) {
  app_init();
  return app_main();
}

/* Park the other harts. Single-core demo: nothing here is SMP-safe, and an unparked hart running
 * through this code would issue its own I2S reads and steal samples out of the FIFO. */
void __attribute__((weak, noreturn)) __main(void) {
  while (1) { __asm__ volatile("wfi"); }
}
