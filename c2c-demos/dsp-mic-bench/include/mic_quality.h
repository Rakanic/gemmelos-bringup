#ifndef C2C_DSP_MIC_QUALITY_H
#define C2C_DSP_MIC_QUALITY_H

#include <stdint.h>
#include <stddef.h>

/* ------------------------------------------------------------------------------------------------
 * Audio-quality metrics for a captured window. CHIP-FREE ON PURPOSE: no HAL, no MMIO, no printf, no
 * config macros — just float arrays in, numbers out. That is what lets test/host_mic_quality_test.c
 * compile THESE EXACT SOURCES for the host and check them against synthetic signals whose true SNR,
 * THD and inter-mic delay are known by construction.
 *
 * It matters here more than usual: the output of this file IS the deliverable. A capture path that is
 * subtly broken and a metric that is subtly broken produce the same thing — a plausible number — and
 * on silicon there is nothing to compare it against. (This is the same reason bearly-smollm keeps its
 * arithmetic in chip-free translation units; see /CLAUDE.md.)
 *
 * Conventions, applied everywhere below:
 *   - amplitudes are in FULL-SCALE units: 1.0 == digital full scale, so every dB is dBFS;
 *   - accumulators are double. A float sum of 32000 squares loses the small terms, and the small
 *     terms are exactly the noise floor this file exists to measure;
 *   - nothing allocates, and no function needs a scratch buffer larger than the frame count.
 * ---------------------------------------------------------------------------------------------- */

#define MQ_MAX_HARMONICS 8

/* Returned by every dB accessor for "immeasurably small" rather than -inf, so a printed record never
 * contains "inf"/"nan" and a log stays parseable. */
#define MQ_DB_FLOOR (-240.0f)

/* Largest lag half-range mq_pair_analyze / mq_pair_track can search, in samples.
 *
 * In the HEADER rather than private to the implementation because the clamp is SILENT: a caller
 * asking for more gets 256 back with no error, and a correlation peak that lands at the edge of the
 * searched window is a saturated search whose "lag" is not a measurement at all. Callers should
 * _Static_assert their own search range against this rather than discover the ceiling from a
 * plausible-looking number. */
#define MQ_LAG_LIMIT 256

typedef struct {
  uint32_t n;            /* samples analysed */
  uint32_t frames;       /* frames the percentiles were taken over */
  float dc;              /* mean of the raw window, full-scale units (this is REMOVED in place) */
  float drift;           /* mean(2nd half) - mean(1st half): DC wander / bad AC coupling */
  float rms;             /* RMS after DC removal */
  float peak;            /* max |x| after DC removal */
  uint32_t clip;         /* samples with |raw| >= clip_level, counted BEFORE DC removal */
  float nf;              /* quiet-percentile frame RMS  (noise floor) */
  float act;             /* loud-percentile frame RMS   (active/signal level) */
  float active_frac;     /* fraction of frames whose RMS exceeds nf * active_mult */
  float hf_ratio;        /* mean(diff^2) / mean(x^2): spectral tilt proxy, see mq_hf_db() */
} mq_stats_t;

typedef struct {
  float rms_a, rms_b;    /* per-signal RMS over the correlation window (DC assumed removed) */
  float gamma;           /* normalised cross-correlation peak in [-1, 1] */
  int lag_int;           /* integer peak lag in samples: b[i + lag] aligns with a[i] */
  float lag_frac;        /* parabolic-interpolated peak lag in samples */
  uint32_t n;            /* correlation window length actually used (0 = too short, nothing valid) */
} mq_pair_t;

/* Inter-mic lag measured in several sub-blocks of one capture, to tell a SHARED clock from two
 * separate ones. Two dividers fed by the same core clock cannot differ in frequency, so their lag is
 * constant; two genuinely independent time bases drift, and the drift shows up as a steady slope. */
typedef struct {
  uint32_t blocks;       /* sub-blocks actually measured (0 = capture too short to say anything) */
  float lag_first;       /* lag in the first block, samples */
  float lag_last;        /* lag in the last block, samples */
  float lag_min, lag_max;
  float lag_slope;       /* samples per SECOND, first block to last: the drift figure */
  float ppm;             /* lag_slope expressed as a clock offset in parts per million */
  float gamma_min;       /* worst per-block coherence: guards against reading a lag out of noise */
  float gamma_mean;
} mq_track_t;

typedef struct {
  float f0;                      /* fundamental that was fitted, Hz (0 = not attempted) */
  int   nharm;                   /* harmonics fitted, including the fundamental */
  float amp[MQ_MAX_HARMONICS];   /* amplitude of harmonic k+1; amp[0] is the fundamental */
  float total_rms;               /* RMS of the analysed window */
  float sinad_rms;               /* RMS of everything that is NOT the fundamental */
  float thd_rms;                 /* RMS of harmonics 2..nharm */
} mq_tone_t;

/* Shift x EARLIER by `d` samples, in place: x_new[i] = x_old[i + d]. `d` may be negative (a retard)
 * and need not be an integer — the fractional part is linearly interpolated between the two
 * neighbouring samples. Edges replicate the boundary sample.
 *
 * This is the inter-element alignment primitive for a mic array, and it is here rather than in the
 * demo because both of its ways of being wrong are invisible on silicon. (1) An in-place shift reads
 * at i+d and writes at i, so the iteration must run FORWARD for d > 0 and BACKWARD for d < 0; the
 * wrong direction fills the buffer with a smear of its own output, which downstream looks exactly like
 * an incoherent microphone. (2) The fractional interpolation has to move the signal the right way —
 * a sign error there is a small, plausible, and completely wrong alignment.
 *
 * The fractional capability is not a nicety: two I2S channels' streams are offset by a NON-INTEGER
 * number of samples (~11.6 measured on dsp25), and an integer shift leaves the ~0.6 remainder in
 * place — against a whole endfire steering delay of 1.0 sample at 2.15 cm spacing. Linear
 * interpolation costs one multiply-add per sample and a gentle treble roll-off on the shifted channel
 * (worst case -3.9 dB at Nyquist for a half-sample shift); leaving the error in costs partial
 * cancellation between the two halves of the array at every frequency. */
void mq_shift_advance(float *x, uint32_t n, float d);

/* Remove the mean from x in place; returns the mean that was removed. */
float mq_remove_dc(float *x, uint32_t n);

/* Full metric set for one signal.
 *
 * x is modified: the DC is removed in place, so the caller is left with an AC-coupled signal ready
 * for the tone fit and the pair correlation (both of which assume DC is gone). fbuf receives the
 * per-frame RMS values and is sorted in place; it needs n / frame_len entries.
 *
 * Frames are non-overlapping and any tail shorter than frame_len is ignored — a partial frame's RMS
 * is not comparable with a full one's, and silently mixing the two would bias the percentiles. */
void mq_analyze(float *x, uint32_t n, uint32_t frame_len, float clip_level,
                uint32_t nf_pct, uint32_t act_pct, float active_mult,
                float *fbuf, uint32_t fbuf_cap, mq_stats_t *out);

/* Inter-microphone delay and coherence, from a window of `win` samples centred in the capture.
 *
 * Both inputs must already be DC-free (mq_analyze does that). The peak is searched over lags
 * -max_lag..+max_lag and then parabolically interpolated, which recovers most of the sub-sample
 * resolution that a small mic spacing costs: at 16 kHz one whole sample is 2.1 cm of path
 * difference, so without interpolation a 5 cm array has only about +-2.3 samples of usable range. */
void mq_pair_analyze(const float *a, const float *b, uint32_t n, int max_lag, uint32_t win,
                     mq_pair_t *out);

/* Split [0, n) into `nblocks` contiguous sub-blocks and measure the lag and coherence in each.
 *
 * This is the test that decides whether two I2S channels can carry one microphone array. A frequency
 * offset between the two sample streams appears as a lag that grows linearly through the capture, and
 * the sensitivity is high: at 16 kHz a 1 ppm offset is 0.016 samples/s, against a lag measurable to
 * about 0.05 samples, so a 2 s capture resolves single-digit ppm. A flat lag means one clock source.
 *
 * Read `gamma_min` before believing any of it — a lag interpolated out of two uncorrelated signals is
 * a random number with a slope. */
void mq_pair_track(const float *a, const float *b, uint32_t n, int max_lag, uint32_t nblocks,
                   float fs, mq_track_t *out);

/* Measure the fundamental near f0_guess, searching +-span_frac (a fraction, e.g. 0.02 for +-2%) on a
 * grid of `steps` points, then parabolically interpolating the peak.
 *
 * This is not a refinement nicety, it is a precondition. A coherent fit needs the frequency to be
 * right to about 1/(2T): fitting a nominal 1000 Hz to a real 1001 Hz tone over half a second throws
 * most of the fundamental's energy into the residual, and mq_tone_analyze would then report a clean
 * mic as catastrophically distorted. Neither a phone's tone generator nor this chip's I2S clock is
 * accurate enough to skip this. Returns the measured f0 in Hz (f0_guess if the search is disabled). */
float mq_tone_refine_f0(const float *x, uint32_t n, float fs, float f0_guess, float span_frac,
                        int steps);

/* Least-squares fit of a fundamental and its harmonics, for the tone stimulus.
 *
 * x must be DC-free. Each harmonic is projected onto cos/sin at k*f0 (harmonics at or above Nyquist
 * are skipped, their amplitude reported as 0), and the residual energies are then obtained from
 * Parseval rather than by rebuilding and subtracting the fitted waveform — which is why this needs no
 * scratch buffer the size of the capture. Over a window of many cycles the basis is orthogonal to
 * within the O(1/n) leakage, far below the distortion levels being measured.
 *
 * The residuals are the useful part: sinad_rms is everything that is not the fundamental (noise AND
 * distortion), thd_rms is the harmonics alone. Their difference is how you tell a noisy mic from a
 * distorting one. */
void mq_tone_analyze(const float *x, uint32_t n, float fs, float f0, int nharm, mq_tone_t *out);

/* ---- derived values, all in dB ---------------------------------------------------------------- */
float mq_db20(float amplitude_ratio);              /* 20*log10, floored at MQ_DB_FLOOR */
float mq_db_ratio(float num, float den);           /* 20*log10(num/den), floored */
static inline float mq_dbfs(float amp)      { return mq_db20(amp); }
static inline float mq_crest_db(const mq_stats_t *s) { return mq_db_ratio(s->peak, s->rms); }
static inline float mq_snr_db(const mq_stats_t *s)   { return mq_db_ratio(s->act, s->nf); }
/* 10*log10(mean(diff^2)/mean(x^2)). For a pure tone at f this equals 20*log10(2*sin(pi*f/fs)), i.e.
 * -18.1 dB at 100 Hz, +6.0 dB at 8 kHz (16 kHz sampling). It is a one-number spectral tilt: strongly
 * negative means the window is dominated by rumble or DC wander, strongly positive means hiss. */
float mq_hf_db(const mq_stats_t *s);
/* Signal-to-noise-AND-distortion, and total harmonic distortion, both relative to the fundamental. */
float mq_sinad_db(const mq_tone_t *t);
float mq_thd_db(const mq_tone_t *t);

/* ---- formatting ------------------------------------------------------------------------------- */
/* Fixed-point formatting, because newlib's %f drags in about a kilobyte of stack and a stack
 * overflow on this platform does not trap — it silently corrupts automatics or hangs the core
 * (/CLAUDE.md). Integer-only, handles the sign correctly for values in (-1, 0), and prints "n/a" for
 * a NaN or an infinity rather than emitting an unparseable token. Returns buf. 24 bytes is enough
 * for any value these metrics produce. */
#define MQ_FMT_BYTES 24
const char *mq_fmt(char *buf, size_t cap, float v, unsigned decimals);

#endif /* C2C_DSP_MIC_QUALITY_H */
