/*
 * mic_quality.c — the audio-quality metrics for dsp-mic-bench.
 *
 * Chip-free (see the header for why). Built for the host by test/host_mic_quality_test.c and, on
 * silicon, compiled WITHOUT the vector extension: the RVV float kernels in this tree have already
 * produced confident garbage on dsp25 once (the TinySpeech entry in /CLAUDE.md), and these numbers
 * are the entire output of the demo — there is nothing on-chip to check them against. Scalar costs a
 * few milliseconds against a 2-second capture.
 */

#include <math.h>

#include "mic_quality.h"

/* newlib hides M_PI under __STRICT_ANSI__, and this file must compile the same way for the host
 * test and the chip. */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Guard for the correlation array. A stack overflow does not trap on this platform, so the lag range
 * is clamped rather than trusted.
 *
 * Raised from 64 to 256 because 64 was not enough for a CROSS-CHANNEL pair: on silicon every
 * cross-channel lag pinned at the search limit, which is what an estimator does when the true offset
 * lies OUTSIDE the window — it reports the edge, and a pinned edge is indistinguishable from a real
 * measurement unless you know to look for it. 256 samples is 5.5 m of path difference, i.e. far past
 * anything acoustic, so a lag found out there is a stream-alignment offset and is meant to be read as
 * one. The array is now static: at 256 lags it is 4 KB, which is the size of frame that reaches this
 * platform's silent stack overflow. Single-threaded by construction (see the parked-harts note in
 * main.c), and the host test is single-threaded too.
 *
 * MQ_LAG_LIMIT itself now lives in mic_quality.h so callers can assert their search range against
 * it — the clamp below is silent, and a lag reported from a saturated search reads like a real one. */

float mq_db20(float amplitude_ratio) {
  double a = (double)amplitude_ratio;
  if (a < 0.0) a = -a;
  if (!(a > 1e-12)) return MQ_DB_FLOOR;          /* also catches NaN, via the negated compare */
  return (float)(20.0 * log10(a));
}

float mq_db_ratio(float num, float den) {
  double d = (double)den;
  if (d < 0.0) d = -d;
  if (!(d > 1e-12)) return MQ_DB_FLOOR;
  return mq_db20((float)((double)num / d));
}

float mq_hf_db(const mq_stats_t *s) {
  if (s == NULL) return MQ_DB_FLOOR;
  double r = (double)s->hf_ratio;
  if (!(r > 1e-24)) return MQ_DB_FLOOR;
  return (float)(10.0 * log10(r));               /* a POWER ratio, hence 10 and not 20 */
}

float mq_sinad_db(const mq_tone_t *t) {
  if ((t == NULL) || (t->nharm <= 0)) return MQ_DB_FLOOR;
  return mq_db_ratio(t->amp[0] * 0.70710678f, t->sinad_rms);   /* fundamental RMS vs everything else */
}

float mq_thd_db(const mq_tone_t *t) {
  if ((t == NULL) || (t->nharm <= 0)) return MQ_DB_FLOOR;
  return mq_db_ratio(t->thd_rms, t->amp[0] * 0.70710678f);
}

void mq_shift_advance(float *x, uint32_t n, float d) {
  if (n < 4u) return;
  /* floor(), toward -inf, without libm: a (long) cast truncates toward zero, which is wrong for d < 0
   * and would make every retard land a sample short. */
  long k = (long)d;
  if ((float)k > d) k--;
  const float f = d - (float)k;                 /* 0 <= f < 1 */
  const long nn = (long)n;

  /* Split into a whole-sample move and a pure fractional one. Doing them separately is what keeps the
   * in-place aliasing tractable: the integer pass reads strictly ahead of (or strictly behind) the
   * write cursor, and the fractional pass then needs a history of exactly one sample. Fusing them
   * would require a 4-deep history whose depth depends on k, which is precisely the sort of
   * index-arithmetic corner that no on-silicon symptom would ever point at. */
  if (k > 0) {
    for (long i = 0; i < nn; ++i) { const long a = i + k; x[i] = (a >= nn) ? x[nn - 1] : x[a]; }
  } else if (k < 0) {
    for (long i = nn; i-- > 0; ) { const long a = i + k; x[i] = (a < 0) ? x[0] : x[a]; }
  }
  if (!(f > 0.0f)) return;

  /* 4-point Lagrange (cubic) interpolation, NOT linear.
   *
   * The difference is not academic at the fractions this is used for. Linear interpolation is a
   * one-zero low-pass whose response at frequency w is |(1-f) + f*e^-jw|: for the ~0.6-sample
   * cross-channel offset that is -3.0 dB at 4 kHz and -14 dB at 8 kHz (16 kHz sampling). Applied to
   * half a microphone array, that mismatch is itself a coherence loss — measured on a full-band test
   * signal it capped inter-element coherence at 0.92, which is a worse error than the 0.6-sample
   * misalignment it was correcting. The cubic is -1.1 dB at 4 kHz for the same shift.
   *
   * Both still have a zero at Nyquist: any even-length symmetric FIR does, and a half-sample delay at
   * f = 0.5 is exactly that. Nothing above ~7 kHz survives a fractional shift; the band that carries
   * speech does. Coefficients at f = 0.5 are the familiar -1/16, 9/16, 9/16, -1/16. */
  const float f2 = f * f, f3 = f2 * f;
  const float hm1 = (-f3 + 3.0f * f2 - 2.0f * f) / 6.0f;      /* -f(f-1)(f-2)/6      */
  const float h0  = ( f3 - 2.0f * f2 - f + 2.0f) * 0.5f;      /* (f+1)(f-1)(f-2)/2   */
  const float h1  = (-f3 + f2 + 2.0f * f) * 0.5f;             /* -(f+1)f(f-2)/2      */
  const float h2  = ( f3 - f) / 6.0f;                         /* (f+1)f(f-1)/6       */

  /* Forward, carrying the ORIGINAL x[i-1] in a register: the write at i destroys the value that step
   * i+1 needs. Out-of-range taps replicate the boundary sample, which touches at most two samples at
   * each end of 32000. */
  float prev = x[0];
  for (long i = 0; i < nn; ++i) {
    const float x0 = x[i];
    const float x1 = ((i + 1) < nn) ? x[i + 1] : x[nn - 1];
    const float x2 = ((i + 2) < nn) ? x[i + 2] : x[nn - 1];
    x[i] = hm1 * prev + h0 * x0 + h1 * x1 + h2 * x2;
    prev = x0;
  }
}

float mq_remove_dc(float *x, uint32_t n) {
  if ((x == NULL) || (n == 0u)) return 0.0f;
  double sum = 0.0;
  for (uint32_t i = 0; i < n; ++i) sum += (double)x[i];
  const float dc = (float)(sum / (double)n);
  for (uint32_t i = 0; i < n; ++i) x[i] -= dc;
  return dc;
}

/* Insertion sort. m is the frame count (100 for the default 2 s window), so this is ~5000 compares —
 * cheaper than the branch to a smarter sort, and it cannot be the thing that goes wrong. */
static void mq_sort(float *v, uint32_t m) {
  for (uint32_t i = 1u; i < m; ++i) {
    const float key = v[i];
    uint32_t j = i;
    while ((j > 0u) && (v[j - 1u] > key)) { v[j] = v[j - 1u]; --j; }
    v[j] = key;
  }
}

/* Percentile of an ASCENDING array by nearest index — no interpolation, so the value returned is
 * always one that was actually measured. */
static float mq_percentile(const float *sorted, uint32_t m, uint32_t pct) {
  if (m == 0u) return 0.0f;
  if (pct > 100u) pct = 100u;
  uint32_t idx = (pct * (m - 1u) + 50u) / 100u;
  if (idx >= m) idx = m - 1u;
  return sorted[idx];
}

void mq_analyze(float *x, uint32_t n, uint32_t frame_len, float clip_level,
                uint32_t nf_pct, uint32_t act_pct, float active_mult,
                float *fbuf, uint32_t fbuf_cap, mq_stats_t *out) {
  if (out == NULL) return;
  for (size_t i = 0; i < sizeof(*out); ++i) ((unsigned char *)out)[i] = 0u;
  if ((x == NULL) || (n == 0u)) return;

  /* --- raw-domain measurements: DC, drift and clipping all describe the signal BEFORE it is
   * AC-coupled, and clipping in particular is only meaningful against the converter's real full
   * scale. Halves are summed separately so the drift comes out of the same single pass. */
  const uint32_t half = n / 2u;
  double sum_lo = 0.0, sum_hi = 0.0;
  uint32_t clip = 0u;
  for (uint32_t i = 0; i < n; ++i) {
    const float v = x[i];
    if (i < half) sum_lo += (double)v; else sum_hi += (double)v;
    const float a = (v < 0.0f) ? -v : v;
    if (a >= clip_level) clip++;
  }
  const uint32_t n_hi = n - half;
  out->n = n;
  out->clip = clip;
  out->dc = (float)((sum_lo + sum_hi) / (double)n);
  out->drift = (half && n_hi) ? (float)((sum_hi / (double)n_hi) - (sum_lo / (double)half)) : 0.0f;

  /* --- AC domain. Everything below is on the DC-free signal, which is also what the caller wants
   * left in the buffer for the tone fit and the inter-mic correlation. */
  for (uint32_t i = 0; i < n; ++i) x[i] -= out->dc;

  double energy = 0.0, diff_energy = 0.0;
  float peak = 0.0f;
  for (uint32_t i = 0; i < n; ++i) {
    const float v = x[i];
    energy += (double)v * (double)v;
    const float a = (v < 0.0f) ? -v : v;
    if (a > peak) peak = a;
    if (i > 0u) { const double d = (double)v - (double)x[i - 1u]; diff_energy += d * d; }
  }
  out->rms = (float)sqrt(energy / (double)n);
  out->peak = peak;
  out->hf_ratio = (energy > 0.0) ? (float)((diff_energy / (double)(n > 1u ? n - 1u : 1u)) /
                                           (energy / (double)n))
                                 : 0.0f;

  /* --- per-frame RMS distribution. Only whole frames: a short tail frame's RMS is not comparable
   * with a full one's, and mixing them would bias exactly the percentiles that everything else is
   * measured against. */
  if ((fbuf == NULL) || (frame_len == 0u)) return;
  uint32_t m = n / frame_len;
  if (m > fbuf_cap) m = fbuf_cap;
  if (m == 0u) return;
  for (uint32_t f = 0; f < m; ++f) {
    const float *p = x + (size_t)f * frame_len;
    double fe = 0.0;
    for (uint32_t i = 0; i < frame_len; ++i) fe += (double)p[i] * (double)p[i];
    fbuf[f] = (float)sqrt(fe / (double)frame_len);
  }
  mq_sort(fbuf, m);                               /* order-independent stats are computed after */
  out->frames = m;
  out->nf = mq_percentile(fbuf, m, nf_pct);
  out->act = mq_percentile(fbuf, m, act_pct);
  const float thr = out->nf * active_mult;
  uint32_t active = 0u;
  for (uint32_t f = 0; f < m; ++f) if (fbuf[f] > thr) active++;
  out->active_frac = (float)active / (float)m;
}

void mq_pair_analyze(const float *a, const float *b, uint32_t n, int max_lag, uint32_t win,
                     mq_pair_t *out) {
  if (out == NULL) return;
  for (size_t i = 0; i < sizeof(*out); ++i) ((unsigned char *)out)[i] = 0u;
  if ((a == NULL) || (b == NULL) || (n == 0u)) return;
  int L = max_lag;
  if (L < 0) L = 0;
  if (L > MQ_LAG_LIMIT) L = MQ_LAG_LIMIT;
  if (win > n) win = n;
  /* Too short to say anything: the usable index range is win - 2L, and a correlation over a handful
   * of samples is noise. Reporting n = 0 is how the caller knows not to print a lag. */
  if (win < (uint32_t)(2 * L) + 64u) return;

  const uint32_t off = (n - win) / 2u;            /* centre of the capture: past any settling, and
                                                   * where a talker is most likely to be talking */
  const float *x = a + off;
  const float *y = b + off;
  const uint32_t i0 = (uint32_t)L, i1 = win - (uint32_t)L;   /* [i0, i1) keeps i+d in bounds */

  double ex = 0.0, ey = 0.0;
  for (uint32_t i = i0; i < i1; ++i) { ex += (double)x[i] * (double)x[i]; ey += (double)y[i] * (double)y[i]; }
  const uint32_t cnt = i1 - i0;
  out->n = cnt;
  out->rms_a = (float)sqrt(ex / (double)cnt);
  out->rms_b = (float)sqrt(ey / (double)cnt);

  static double r[2 * MQ_LAG_LIMIT + 1];   /* static: 4 KB is a stack-overflow-sized frame here */
  int best = -L;
  double bestv = -1e300;
  for (int d = -L; d <= L; ++d) {
    double acc = 0.0;
    for (uint32_t i = i0; i < i1; ++i) acc += (double)x[i] * (double)y[(uint32_t)((int)i + d)];
    r[d + L] = acc;
    if (acc > bestv) { bestv = acc; best = d; }
  }
  const double norm = sqrt(ex * ey);
  double gamma = (norm > 1e-30) ? (bestv / norm) : 0.0;
  if (gamma > 1.0) gamma = 1.0;
  if (gamma < -1.0) gamma = -1.0;
  out->gamma = (float)gamma;
  out->lag_int = best;

  /* Parabolic interpolation of the peak. One sample is 2.1 cm of path difference at 16 kHz, so
   * without this a 5-6 cm array resolves direction into about two buckets; with it, a fraction of a
   * sample. Standard, and it costs three values that were already computed. */
  float frac = 0.0f;
  if ((best > -L) && (best < L)) {
    const double ym = r[best + L - 1], y0 = r[best + L], yp = r[best + L + 1];
    const double den = ym - 2.0 * y0 + yp;
    if (den < -1e-30) {                            /* < 0 == a real maximum, not a flat or inverted
                                                    * three-point set (which interpolates to nonsense) */
      double dl = 0.5 * (ym - yp) / den;
      if (dl > 1.0) dl = 1.0;
      if (dl < -1.0) dl = -1.0;
      frac = (float)dl;
    }
  }
  out->lag_frac = (float)best + frac;
}

void mq_pair_track(const float *a, const float *b, uint32_t n, int max_lag, uint32_t nblocks,
                   float fs, mq_track_t *out) {
  if (out == NULL) return;
  for (size_t i = 0; i < sizeof(*out); ++i) ((unsigned char *)out)[i] = 0u;
  if ((a == NULL) || (b == NULL) || (n == 0u) || (nblocks == 0u) || !(fs > 0.0f)) return;

  const uint32_t blen = n / nblocks;
  int L = max_lag;
  if (L < 0) L = 0;
  /* Each block must still be long enough for mq_pair_analyze to accept it, or every block reports
   * nothing and the slope is computed over zeros — which would look like a perfectly shared clock. */
  if (blen < (uint32_t)(2 * L) + 64u) return;

  double gsum = 0.0;
  uint32_t got = 0u;
  float first = 0.0f, last = 0.0f, lo = 0.0f, hi = 0.0f, gmin = 1.0f;
  for (uint32_t k = 0; k < nblocks; ++k) {
    mq_pair_t p;
    mq_pair_analyze(a + (size_t)k * blen, b + (size_t)k * blen, blen, L, blen, &p);
    if (p.n == 0u) continue;
    if (got == 0u) { first = p.lag_frac; lo = hi = p.lag_frac; }
    last = p.lag_frac;
    if (p.lag_frac < lo) lo = p.lag_frac;
    if (p.lag_frac > hi) hi = p.lag_frac;
    if (p.gamma < gmin) gmin = p.gamma;
    gsum += (double)p.gamma;
    got++;
  }
  if (got == 0u) return;
  out->blocks = got;
  out->lag_first = first;
  out->lag_last = last;
  out->lag_min = lo;
  out->lag_max = hi;
  out->gamma_min = gmin;
  out->gamma_mean = (float)(gsum / (double)got);
  if (got >= 2u) {
    /* Block CENTRES are what the lags belong to, so the baseline is (got-1) block periods. */
    const double dt = (double)(got - 1u) * (double)blen / (double)fs;
    if (dt > 0.0) {
      const double slope = ((double)last - (double)first) / dt;   /* samples per second */
      out->lag_slope = (float)slope;
      out->ppm = (float)(slope / (double)fs * 1.0e6);             /* samples/s / (samples/s) */
    }
  }
}

/* Project x onto cos/sin at angular frequency w. Phase is advanced incrementally in double and
 * wrapped: over 32000 samples the accumulated error is ~1e-11 rad, where recomputing w*i in float
 * would be ~1e-3 rad by the end of the window. */
static void mq_project(const float *x, uint32_t n, double w, double *c_out, double *s_out) {
  double ph = 0.0, c = 0.0, s = 0.0;
  for (uint32_t i = 0; i < n; ++i) {
    c += (double)x[i] * cos(ph);
    s += (double)x[i] * sin(ph);
    ph += w;
    if (ph > 2.0 * M_PI) ph -= 2.0 * M_PI;
  }
  *c_out = c;
  *s_out = s;
}

/* Exact least-squares fit of a single sinusoid at angular frequency w:  x[i] ~ a*cos(w i) + b*sin(w i).
 *
 * The 2x2 normal equations are solved rather than assuming the basis is orthogonal. Over a window of
 * many cycles the cos/sin basis is orthogonal only to O(1/n), which sounds negligible and is not: it
 * biases the fitted amplitude by ~1e-4 relative, the ENERGY by ~2e-4, and an energy error of 2e-4
 * puts a -37 dB floor under a distortion measurement that needs to resolve -60 dB. Solving exactly
 * costs three more accumulators in a loop that is already dominated by the sin/cos calls.
 *
 * `explained` is the energy this component accounts for, from the least-squares identity
 * (a*Sxc + b*Sxs) — which is EXACT for the fitted coefficients, so the residual obtained by
 * subtracting it needs no second pass over the data and inherits no orthogonality assumption.
 * `phase` is defined by x ~ A*cos(w i + phase) with i measured from the start of the segment. */
static void mq_fit_sinusoid(const float *x, uint32_t n, double w,
                            double *amp, double *phase, double *explained) {
  double ph = 0.0;
  double sxc = 0.0, sxs = 0.0, scc = 0.0, scs = 0.0, sss = 0.0;
  for (uint32_t i = 0; i < n; ++i) {
    const double c = cos(ph), s = sin(ph), v = (double)x[i];
    sxc += v * c; sxs += v * s;
    scc += c * c; scs += c * s; sss += s * s;
    ph += w;
    if (ph > 2.0 * M_PI) ph -= 2.0 * M_PI;
  }
  const double det = scc * sss - scs * scs;
  double a = 0.0, b = 0.0;
  if ((det > 1e-12) || (det < -1e-12)) {
    a = (sxc * sss - sxs * scs) / det;
    b = (sxs * scc - sxc * scs) / det;
  }
  if (amp != NULL) *amp = sqrt(a * a + b * b);
  if (phase != NULL) *phase = atan2(-b, a);
  if (explained != NULL) {
    double e = a * sxc + b * sxs;
    if (e < 0.0) e = 0.0;
    *explained = e;
  }
}

#define MQ_SEARCH_LIMIT 129

float mq_tone_refine_f0(const float *x, uint32_t n, float fs, float f0_guess, float span_frac,
                        int steps) {
  if ((x == NULL) || (n == 0u) || !(fs > 0.0f) || !(f0_guess > 0.0f)) return 0.0f;
  if (!(span_frac > 0.0f)) return f0_guess;
  if (steps < 3) steps = 3;
  if (steps > MQ_SEARCH_LIMIT) steps = MQ_SEARCH_LIMIT;

  double lo = (double)f0_guess * (1.0 - (double)span_frac);
  double hi = (double)f0_guess * (1.0 + (double)span_frac);
  const double nyq = 0.5 * (double)fs;
  if (lo < 1.0) lo = 1.0;
  if (hi > 0.98 * nyq) hi = 0.98 * nyq;
  if (!(hi > lo)) return f0_guess;

  /* Grid of coherent projections. This exists because a fit at a merely NOMINAL frequency is
   * worthless: the basis has to stay phase-locked to the tone across the whole window, and 1/(2T) is
   * all the error that takes. Fitting 1000 Hz to a 1001 Hz tone across 0.5 s loses most of the
   * amplitude into the residual and reports the loss as distortion. */
  double p[MQ_SEARCH_LIMIT];
  int best = 0;
  double bestv = -1.0;
  for (int i = 0; i < steps; ++i) {
    const double f = lo + (hi - lo) * ((double)i / (double)(steps - 1));
    double c, s;
    mq_project(x, n, 2.0 * M_PI * f / (double)fs, &c, &s);
    p[i] = c * c + s * s;
    if (p[i] > bestv) { bestv = p[i]; best = i; }
  }
  double idx = (double)best;
  if ((best > 0) && (best < steps - 1)) {
    const double den = p[best - 1] - 2.0 * p[best] + p[best + 1];
    if (den < -1e-30) {
      double d = 0.5 * (p[best - 1] - p[best + 1]) / den;
      if (d > 1.0) d = 1.0;
      if (d < -1.0) d = -1.0;
      idx += d;
    }
  }
  double f = lo + (hi - lo) * (idx / (double)(steps - 1));

  /* The grid plus a parabola is only good to a few hundredths of a hertz, and that is not nearly
   * good enough: what SINAD measures is the energy NOT explained by the fundamental, so a fundamental
   * fitted at slightly the wrong frequency dumps its own mismatch into the residual and reports it as
   * noise. A 0.1 Hz error over half a second leaves ~-21 dB of it — swamping the -50 dB the analog
   * path actually deserves to be judged at.
   *
   * So the frequency is finished off by PHASE DIFFERENCE, which is both far more accurate and much
   * cheaper than continuing to search: fit the same frequency on the first and second halves of the
   * window, and the phase advance between them is exactly the frequency error times the half-window
   * length. Unambiguous as long as |df| < fs/n, which the grid spacing already guarantees. Two
   * iterations because the first one moves the fit onto the peak, where the phase estimate is best. */
  const uint32_t h = n / 2u;
  if (h >= 16u) {
    for (int it = 0; it < 2; ++it) {
      double w = 2.0 * M_PI * f / (double)fs;
      double ph1 = 0.0, ph2 = 0.0;
      mq_fit_sinusoid(x, h, w, NULL, &ph1, NULL);
      mq_fit_sinusoid(x + h, h, w, NULL, &ph2, NULL);
      /* Both halves are fitted with the basis restarting at zero, so the raw phase difference is the
       * signal's ENTIRE advance across h samples. Subtract the advance the trial frequency itself
       * predicts; what is left is the mismatch, and only that. */
      const double expected = fmod(w * (double)h, 2.0 * M_PI);
      double dphi = ph2 - ph1 - expected;
      while (dphi > M_PI) dphi -= 2.0 * M_PI;         /* the estimate lives on a circle */
      while (dphi < -M_PI) dphi += 2.0 * M_PI;
      const double df = dphi * (double)fs / (2.0 * M_PI * (double)h);
      const double cand = f + df;
      if ((cand <= 0.0) || (cand >= 0.98 * nyq)) break;
      f = cand;
    }
  }
  return (float)f;
}

void mq_tone_analyze(const float *x, uint32_t n, float fs, float f0, int nharm, mq_tone_t *out) {
  if (out == NULL) return;
  for (size_t i = 0; i < sizeof(*out); ++i) ((unsigned char *)out)[i] = 0u;
  if ((x == NULL) || (n == 0u) || !(fs > 0.0f) || !(f0 > 0.0f)) return;
  if (nharm < 1) nharm = 1;
  if (nharm > MQ_MAX_HARMONICS) nharm = MQ_MAX_HARMONICS;

  double energy = 0.0;
  for (uint32_t i = 0; i < n; ++i) energy += (double)x[i] * (double)x[i];
  out->f0 = f0;
  out->nharm = nharm;
  out->total_rms = (float)sqrt(energy / (double)n);

  const double nyq = 0.5 * (double)fs;
  double harm_energy = 0.0;                        /* explained energy of harmonics 2..nharm */
  double fund_energy = 0.0;
  for (int k = 1; k <= nharm; ++k) {
    const double fk = (double)k * (double)f0;
    if (fk >= nyq) continue;                       /* amp stays 0: an alias is not a harmonic */
    double amp = 0.0, explained = 0.0;
    mq_fit_sinusoid(x, n, 2.0 * M_PI * fk / (double)fs, &amp, NULL, &explained);
    out->amp[k - 1] = (float)amp;
    if (k == 1) fund_energy = explained; else harm_energy += explained;
  }
  /* Residual from the least-squares identity rather than by rebuilding and subtracting the fitted
   * waveform: it is exact for the fitted coefficients and needs no second buffer the size of the
   * capture. Everything the fundamental does not explain — noise, distortion, and any frequency
   * mismatch left by the fit — lands here, which is what makes it worth trusting. */
  double resid = energy - fund_energy;
  if (resid < 0.0) resid = 0.0;
  out->sinad_rms = (float)sqrt(resid / (double)n);
  out->thd_rms = (float)sqrt(harm_energy / (double)n);
}

const char *mq_fmt(char *buf, size_t cap, float v, unsigned decimals) {
  if ((buf == NULL) || (cap == 0u)) return buf;
  if (isnan(v) || isinf(v)) {
    /* Never emit "inf"/"nan": a record that cannot be parsed is worse than one that says nothing. */
    const char *na = "n/a";
    size_t i = 0;
    while ((na[i] != '\0') && (i + 1u < cap)) { buf[i] = na[i]; i++; }
    buf[i] = '\0';
    return buf;
  }
  if (decimals > 6u) decimals = 6u;
  unsigned long long scale = 1ull;
  for (unsigned d = 0; d < decimals; ++d) scale *= 10ull;

  const int neg = (v < 0.0f);
  double av = neg ? -(double)v : (double)v;
  double scaled = av * (double)scale + 0.5;
  if (scaled > 1.0e17) scaled = 1.0e17;            /* saturate rather than wrap the cast */
  unsigned long long t = (unsigned long long)scaled;
  unsigned long long ip = t / scale;
  unsigned long long fp = t % scale;

  /* Digits are emitted by hand so this stays integer-only and identical on host and chip. */
  char tmp[24];
  size_t nd = 0;
  do { tmp[nd++] = (char)('0' + (int)(ip % 10ull)); ip /= 10ull; } while ((ip != 0ull) && (nd < sizeof(tmp)));
  size_t o = 0;
  /* The sign is printed from the ORIGINAL value, so -0.04 at 1 decimal reads "-0.0" and not "0.0" —
   * a dropped minus on a DC offset or a level mismatch is a silently wrong conclusion. */
  if (neg && (o + 1u < cap)) buf[o++] = '-';
  while ((nd > 0u) && (o + 1u < cap)) buf[o++] = tmp[--nd];
  if (decimals > 0u) {
    if (o + 1u < cap) buf[o++] = '.';
    for (unsigned d = decimals; d > 0u; --d) {
      unsigned long long div = 1ull;
      for (unsigned e = 1; e < d; ++e) div *= 10ull;
      if (o + 1u < cap) buf[o++] = (char)('0' + (int)((fp / div) % 10ull));
    }
  }
  buf[o] = '\0';
  return buf;
}
