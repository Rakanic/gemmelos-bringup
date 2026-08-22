/*
 * Host unit test for dsp-mic-bench's metrics.
 *
 *   cd c2c-demos/dsp-mic-bench/test
 *   gcc -O2 -Wall -Wextra -I../include -o host_mic_quality_test \
 *       host_mic_quality_test.c ../src/mic_quality.c -lm && ./host_mic_quality_test
 *
 * It compiles THE EXACT SOURCE that runs on silicon (src/mic_quality.c is chip-free for this reason)
 * and feeds it signals whose true SNR, spectral tilt, distortion and inter-mic delay are known by
 * construction.
 *
 * Why this test exists at all: on silicon there is nothing to compare these numbers against. A
 * capture path that is subtly broken and a metric that is subtly broken both produce a plausible
 * number, and the whole point of the demo is to tell one from the other — so at least one side has to
 * be verified somewhere it CAN be verified. Every check below would have been a flash-and-squint
 * guess otherwise (bearly-smollm's host reference is the precedent; see /CLAUDE.md).
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "mic_quality.h"

#define FS 16000.0f
#define N 8192u
#define FRAME 320u
#define NFRAMES (N / FRAME)

static int failures;

static void ok(const char *what, int cond, const char *detail) {
  if (cond) {
    printf("  ok   %-46s %s\n", what, detail ? detail : "");
  } else {
    printf("  FAIL %-46s %s\n", what, detail ? detail : "");
    failures++;
  }
}

static void near(const char *what, double got, double want, double tol) {
  char detail[128];
  snprintf(detail, sizeof(detail), "got %.4f want %.4f (tol %.4f)", got, want, tol);
  ok(what, fabs(got - want) <= tol, detail);
}

static void streq(const char *what, const char *got, const char *want) {
  char detail[128];
  snprintf(detail, sizeof(detail), "got \"%s\" want \"%s\"", got, want);
  ok(what, strcmp(got, want) == 0, detail);
}

/* Deterministic noise: the test must fail for a real reason, never because of the seed. */
static uint32_t rng_state = 0x12345678u;
static float noise(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return ((float)(rng_state >> 8) / 8388608.0f) - 1.0f;   /* ~uniform in [-1, 1) */
}

static float rms_of(const float *x, uint32_t n) {
  double e = 0.0;
  for (uint32_t i = 0; i < n; ++i) e += (double)x[i] * (double)x[i];
  return (float)sqrt(e / (double)n);
}

/* ---------------------------------------------------------------- formatting ---- */
static void test_fmt(void) {
  char b[MQ_FMT_BYTES];
  printf("\nmq_fmt (integer-only fixed point; newlib's %%f is not usable here)\n");
  streq("0 at 1dp", mq_fmt(b, sizeof(b), 0.0f, 1), "0.0");
  streq("-42.34 at 1dp", mq_fmt(b, sizeof(b), -42.34f, 1), "-42.3");
  /* -42.25 is exactly representable, so this really does test half-up rounding of the magnitude
   * (-42.35f is 42.3499... in float and would be testing the literal, not the code). */
  streq("-42.25 at 1dp rounds away from zero", mq_fmt(b, sizeof(b), -42.25f, 1), "-42.3");
  streq("12.345 at 2dp", mq_fmt(b, sizeof(b), 12.345f, 2), "12.35");
  streq("3.7 at 0dp", mq_fmt(b, sizeof(b), 3.7f, 0), "4");
  /* The sign of a value that rounds to zero is the whole reason this is hand-written: a dropped minus
   * on a DC offset or an inter-mic level mismatch is a silently inverted conclusion. */
  streq("-0.04 keeps its sign at 1dp", mq_fmt(b, sizeof(b), -0.04f, 1), "-0.0");
  streq("0.0457 at 4dp", mq_fmt(b, sizeof(b), 0.0457f, 4), "0.0457");
  streq("NaN is not printed as a number", mq_fmt(b, sizeof(b), (float)NAN, 1), "n/a");
  streq("inf is not printed as a number", mq_fmt(b, sizeof(b), (float)INFINITY, 1), "n/a");
  /* Truncation must still terminate: a runaway string here would corrupt a log line, and on chip it
   * would be writing past a stack buffer. */
  char small[4];
  mq_fmt(small, sizeof(small), -123.456f, 2);
  ok("respects a short buffer", strlen(small) < sizeof(small), small);
}

/* ---------------------------------------------------------------- dB helpers ---- */
static void test_db(void) {
  printf("\ndB helpers\n");
  near("db20(1)", mq_db20(1.0f), 0.0, 1e-4);
  near("db20(0.5)", mq_db20(0.5f), -6.0206, 1e-3);
  near("db_ratio(2,1)", mq_db_ratio(2.0f, 1.0f), 6.0206, 1e-3);
  ok("db20(0) is floored, not -inf", mq_db20(0.0f) == MQ_DB_FLOOR, "");
  ok("db_ratio(x,0) is floored", mq_db_ratio(1.0f, 0.0f) == MQ_DB_FLOOR, "");
}

/* ---------------------------------------------------------------- mq_analyze ---- */
static void test_analyze_sine(void) {
  printf("\nmq_analyze on a pure tone with a DC offset\n");
  static float x[N], fbuf[NFRAMES];
  const float A = 0.5f, dc = -0.0457f;       /* the DC these MEMS mics actually sit on, ~-4.6% FS */
  for (uint32_t i = 0; i < N; ++i) x[i] = dc + A * sinf(2.0f * (float)M_PI * 1000.0f * i / FS);

  mq_stats_t s;
  mq_analyze(x, N, FRAME, 0.98f, 10u, 90u, 4.0f, fbuf, NFRAMES, &s);
  near("dc measured", s.dc, dc, 1e-4);
  near("rms = A/sqrt(2) after DC removal", s.rms, A / sqrtf(2.0f), 1e-3);
  near("peak = A", s.peak, A, 2e-3);
  near("crest = 3.01 dB for a sinusoid", mq_crest_db(&s), 3.0103, 0.05);
  ok("no clipping at 0.5 FS", s.clip == 0u, "");
  ok("frames counted", s.frames == NFRAMES, "");
  /* Constant level, so the quiet and loud percentiles must coincide: an SNR of 0 dB is the CORRECT
   * answer for a signal with no dynamics, and a non-zero one would mean the percentiles are being
   * taken over something other than what they claim. */
  near("snr = 0 dB for a constant level", mq_snr_db(&s), 0.0, 0.05);
  near("active fraction = 0", s.active_frac, 0.0, 1e-6);
  /* A first-difference power ratio has a closed form for a sinusoid: (2 sin(pi f / fs))^2. */
  near("hf tilt matches 20log10(2 sin(pi f/fs))", mq_hf_db(&s),
       20.0 * log10(2.0 * sin(M_PI * 1000.0 / 16000.0)), 0.05);

  /* DC removal is in place, so the caller is left with an AC-coupled buffer — the tone fit and the
   * inter-mic correlation both depend on that having happened. */
  double mean = 0.0;
  for (uint32_t i = 0; i < N; ++i) mean += x[i];
  near("buffer is left DC-free", mean / N, 0.0, 1e-5);
}

static void test_analyze_dynamics(void) {
  printf("\nmq_analyze percentiles on a signal with two levels\n");
  static float x[N], fbuf[NFRAMES];
  /* Alternating quiet and loud frames, 20 dB apart: the classic "speech with gaps" shape. */
  const float quiet = 0.001f, loud = 0.01f;
  for (uint32_t f = 0; f < NFRAMES; ++f) {
    const float a = ((f % 2u) == 0u) ? quiet : loud;
    for (uint32_t i = 0; i < FRAME; ++i) x[f * FRAME + i] = a * noise();
  }
  mq_stats_t s;
  mq_analyze(x, N, FRAME, 0.98f, 10u, 90u, 4.0f, fbuf, NFRAMES, &s);
  near("snr recovers the 20 dB level split", mq_snr_db(&s), 20.0, 0.6);
  near("half the frames are active", s.active_frac, 0.5, 0.05);
  ok("noise floor is the quiet level, not the mean",
     mq_dbfs(s.nf) < mq_dbfs(s.rms) - 6.0f, "");
}

static void test_analyze_clip_and_drift(void) {
  printf("\nmq_analyze clipping and DC drift\n");
  static float x[N], fbuf[NFRAMES];
  for (uint32_t i = 0; i < N; ++i) x[i] = 0.1f;
  x[10] = 1.0f; x[20] = -0.99f; x[30] = 0.97f;      /* two clipped, one just under */
  mq_stats_t s;
  mq_analyze(x, N, FRAME, 0.98f, 10u, 90u, 4.0f, fbuf, NFRAMES, &s);
  ok("clip counts only samples at/over the level", s.clip == 2u, "");

  /* A ramp: mean of the second half minus mean of the first half is exactly half the total span. */
  for (uint32_t i = 0; i < N; ++i) x[i] = 0.2f * ((float)i / (float)N);
  mq_analyze(x, N, FRAME, 0.98f, 10u, 90u, 4.0f, fbuf, NFRAMES, &s);
  near("drift detects a slow DC ramp", s.drift, 0.1, 2e-3);
}

/* ---------------------------------------------------------------- pair ---- */
static void test_pair(void) {
  printf("\nmq_pair_analyze: inter-mic delay and coherence\n");
  static float a[N], b[N];
  /* Broadband, band-limited by a 2-tap average so a fractional delay is well defined. */
  float prev = 0.0f;
  for (uint32_t i = 0; i < N; ++i) { const float v = noise(); a[i] = 0.5f * (v + prev); prev = v; }

  /* b lags a by exactly 3 samples: b[i] = a[i-3], so the peak is at d = +3 (b[i+d] aligns with a[i]).
   * Sign convention matters — a sign error here would steer a beam the wrong way. */
  for (uint32_t i = 0; i < N; ++i) b[i] = (i >= 3u) ? a[i - 3u] : 0.0f;
  mq_pair_t p;
  mq_pair_analyze(a, b, N, 4, 4096u, &p);
  ok("integer lag sign and magnitude", p.lag_int == 3, "");
  near("interpolated lag", p.lag_frac, 3.0, 0.05);
  near("gamma ~ 1 for an identical delayed copy", p.gamma, 1.0, 0.02);
  near("level mismatch = 0 dB", mq_db_ratio(p.rms_b, p.rms_a), 0.0, 0.1);

  /* Fractional delay via linear interpolation: 2.5 samples. This is what parabolic interpolation of
   * the correlation peak is for — at 16 kHz one whole sample is 2.1 cm of path difference, so without
   * sub-sample resolution a 5 cm array resolves direction into about two buckets. */
  for (uint32_t i = 0; i < N; ++i) b[i] = (i >= 3u) ? 0.5f * (a[i - 2u] + a[i - 3u]) : 0.0f;
  mq_pair_analyze(a, b, N, 4, 4096u, &p);
  near("sub-sample lag", p.lag_frac, 2.5, 0.25);

  /* Independent noise in the two mics: gamma must be near zero. This is the case that decides
   * whether averaging can help at all, so it must not read as coherent. */
  for (uint32_t i = 0; i < N; ++i) { a[i] = noise(); b[i] = noise(); }
  mq_pair_analyze(a, b, N, 4, 4096u, &p);
  ok("gamma ~ 0 for independent noise", fabsf(p.gamma) < 0.1f, "");

  /* A window too short to mean anything reports n = 0 rather than a confident number. */
  mq_pair_analyze(a, b, N, 4, 16u, &p);
  ok("too-short window is refused", p.n == 0u, "");
}

/* ---------------------------------------------------------------- drift tracking ---- */
/* This is the metric that decides whether two I2S channels can carry one microphone array, so it is
 * worth more than a smoke test: a tracker that reported "flat" regardless of input would green-light
 * a 4-mic build on hardware that cannot support one. */
static void test_track(void) {
  printf("\nmq_pair_track: shared clock vs two time bases\n");
  static float a[N], b[N];
  float prev = 0.0f;
  for (uint32_t i = 0; i < N; ++i) { const float v = noise(); a[i] = 0.5f * (v + prev); prev = v; }

  /* Case 1: one clock. Fixed 2-sample delay, no drift — the slope must be zero, because a shared
   * divider cannot produce a frequency offset. */
  for (uint32_t i = 0; i < N; ++i) b[i] = (i >= 2u) ? a[i - 2u] : 0.0f;
  mq_track_t t;
  mq_pair_track(a, b, N, 8, 8u, FS, &t);
  ok("shared clock: blocks measured", t.blocks == 8u, "");
  near("shared clock: lag is the fixed offset", t.lag_first, 2.0, 0.1);
  near("shared clock: slope ~ 0 smp/s", t.lag_slope, 0.0, 0.05);
  ok("shared clock: coherent throughout", t.gamma_min > 0.9f, "");

  /* Case 2: two time bases 100 ppm apart. b is a resampled with a slowly growing delay, which is
   * exactly what a frequency offset looks like: 1e-4 samples per sample = 1.6 samples/s at 16 kHz. */
  const double rate = 1.0e-4;
  for (uint32_t i = 0; i < N; ++i) {
    const double src = (double)i - (2.0 + rate * (double)i);
    const int i0 = (int)floor(src);
    const double fr = src - (double)i0;
    b[i] = ((i0 >= 0) && (i0 + 1 < (int)N)) ? (float)((1.0 - fr) * a[i0] + fr * a[i0 + 1]) : 0.0f;
  }
  mq_pair_track(a, b, N, 8, 8u, FS, &t);
  /* Tolerances are wide on purpose. The job is to separate "no offset" from "an offset", which is an
   * order-of-magnitude call, not to be a calibrated ppm meter — and the linear interpolation used to
   * synthesise a fractional delay here is itself a mild low-pass whose strength varies with the
   * fractional part, which biases the correlation peak by some percent. */
  near("100 ppm offset: slope in samples/s", t.lag_slope, rate * FS, 0.35);
  near("100 ppm offset: reported as ppm", t.ppm, rate * 1.0e6, 40.0);
  /* Total drift across this window is rate * N = 0.82 samples — which is the point: sub-sample
   * interpolation is what makes a sub-sample drift visible at all. */
  near("100 ppm offset: lag moved by rate*N", (double)(t.lag_max - t.lag_min), rate * (double)N, 0.3);
  ok("drift is distinguishable from a shared clock", t.lag_slope > 1.0f, "");

  /* Case 3: uncorrelated inputs. The lag is then a random number with a random slope, so gamma_min is
   * the guard that stops it being read as a clocking result. */
  for (uint32_t i = 0; i < N; ++i) { a[i] = noise(); b[i] = noise(); }
  mq_pair_track(a, b, N, 8, 8u, FS, &t);
  ok("uncorrelated: gamma_min exposes a meaningless lag", t.gamma_min < 0.3f, "");

  /* Blocks too short to correlate must report nothing rather than a slope through zeros. */
  mq_pair_track(a, b, N, 8, 1024u, FS, &t);
  ok("over-fine blocking is refused", t.blocks == 0u, "");
}

/* ---------------------------------------------------------------- tone ---- */
static void test_tone(void) {
  printf("\nmq_tone_analyze / mq_tone_refine_f0\n");
  static float x[N], rest[N];
  const double f_true = 997.3;            /* deliberately NOT the nominal 1000 Hz */
  const float A = 0.5f;
  const float h2 = 0.005f;                /* -40 dB relative to the fundamental */
  for (uint32_t i = 0; i < N; ++i) {
    const double ph = 2.0 * M_PI * f_true * (double)i / (double)FS;
    rest[i] = h2 * (float)sin(2.0 * ph) + 0.0005f * noise();
    x[i] = A * (float)sin(ph) + rest[i];
  }

  /* The tolerance is the point: a grid-plus-parabola estimate lands within ~0.1 Hz, which is NOT
   * enough (see below), so the phase-difference step has to bring it to a few thousandths. */
  const float f0 = mq_tone_refine_f0(x, N, FS, 1000.0f, 0.02f, 33);
  near("f0 is measured to milli-hertz, not merely found", f0, f_true, 0.01);

  mq_tone_t t;
  mq_tone_analyze(x, N, FS, f0, 5, &t);
  near("fundamental amplitude", t.amp[0], A, 5e-3);
  near("2nd harmonic amplitude", t.amp[1], h2, 1e-3);
  near("thd", mq_thd_db(&t), 20.0 * log10(h2 / A), 0.5);
  /* The residual is compared against the RMS of the components that were actually added, not a
   * hand-computed ideal — so this checks the metric, not the test's own arithmetic. */
  near("sinad matches the injected noise+distortion", mq_sinad_db(&t),
       20.0 * log10((A / sqrt(2.0)) / rms_of(rest, N)), 0.5);

  /* The failure this whole refinement exists to prevent: fitting the nominal frequency to a tone
   * 2.7 Hz away throws the fundamental into the residual and reports a clean mic as a ruined one. */
  mq_tone_t bad;
  mq_tone_analyze(x, N, FS, 1000.0f, 5, &bad);
  ok("a nominal-frequency fit really is much worse (this is why refine exists)",
     mq_sinad_db(&bad) < mq_sinad_db(&t) - 10.0f, "");

  /* How deep the measurement can actually see. A MEMS mic is good for 60-65 dB SNR, so a metric that
   * bottomed out at 35 dB would be unable to judge the part it is pointed at. Noise only, at -60 dB
   * relative to the fundamental. */
  for (uint32_t i = 0; i < N; ++i) {
    const double ph = 2.0 * M_PI * f_true * (double)i / (double)FS;
    rest[i] = 0.000612f * noise();                   /* rms ~ A/sqrt(2)/1000 => 60 dB */
    x[i] = A * (float)sin(ph) + rest[i];
  }
  const float f0b = mq_tone_refine_f0(x, N, FS, 1000.0f, 0.02f, 33);
  mq_tone_analyze(x, N, FS, f0b, 5, &t);
  near("resolves a 60 dB sinad (the mic's own spec range)", mq_sinad_db(&t),
       20.0 * log10((A / sqrt(2.0)) / rms_of(rest, N)), 1.0);

  /* On a signal with no tone in it, sinad must not claim a strong fundamental. */
  for (uint32_t i = 0; i < N; ++i) x[i] = 0.01f * noise();
  mq_tone_analyze(x, N, FS, 1000.0f, 5, &t);
  ok("no tone -> sinad is not high", mq_sinad_db(&t) < 6.0f, "");

  /* A harmonic above Nyquist is an alias, not a harmonic: it must be reported as absent. */
  mq_tone_analyze(x, N, FS, 5000.0f, 5, &t);
  ok("harmonics above Nyquist are skipped", t.amp[1] == 0.0f && t.amp[4] == 0.0f, "");
}

/* ---------------------------------------------------------------- array claim ---- */
static void test_array_gain(void) {
  printf("\nthe array claim itself: averaging two mics\n");
  static float l[N], r[N], c[N], fbuf[NFRAMES];
  /* Case 1: UNCORRELATED noise in the two mics (mic self-noise, ADC noise). Averaging must gain
   * ~3 dB, because the signals add in power rather than amplitude. */
  for (uint32_t i = 0; i < N; ++i) { l[i] = 0.01f * noise(); r[i] = 0.01f * noise(); }
  for (uint32_t i = 0; i < N; ++i) c[i] = 0.5f * (l[i] + r[i]);
  mq_stats_t sl, sc;
  mq_analyze(l, N, FRAME, 0.98f, 10u, 90u, 4.0f, fbuf, NFRAMES, &sl);
  mq_analyze(c, N, FRAME, 0.98f, 10u, 90u, 4.0f, fbuf, NFRAMES, &sc);
  near("uncorrelated: d_rms = -3 dB", mq_dbfs(sc.rms) - mq_dbfs(sl.rms), -3.0103, 0.3);

  /* Case 2: the SAME noise in both mics (room tone, reverb, a distant talker). Averaging must gain
   * NOTHING. This is the case the demo's verdict string has to get right, because it is most
   * real-room noise — and reporting a gain here would be the demo telling a comfortable lie. */
  for (uint32_t i = 0; i < N; ++i) { l[i] = 0.01f * noise(); r[i] = l[i]; }
  for (uint32_t i = 0; i < N; ++i) c[i] = 0.5f * (l[i] + r[i]);
  mq_analyze(l, N, FRAME, 0.98f, 10u, 90u, 4.0f, fbuf, NFRAMES, &sl);
  mq_analyze(c, N, FRAME, 0.98f, 10u, 90u, 4.0f, fbuf, NFRAMES, &sc);
  near("correlated: d_rms = 0 dB", mq_dbfs(sc.rms) - mq_dbfs(sl.rms), 0.0, 0.05);
}

/* ---------------------------------------------------------------- alignment shifter ---- */
/* mq_shift_advance is the primitive the whole 4-mic array rests on, and both of its failure modes are
 * invisible on silicon: a wrong in-place iteration direction smears the buffer (which reads downstream
 * as an incoherent microphone, i.e. as a hardware fault), and a fractional sign error produces a
 * small, plausible, completely wrong alignment. So it is checked here against a delay measured by an
 * independent function — if the shifter and the correlator disagree about which way time runs, one of
 * them is wrong and the array would be built on it. */
static void test_shift(void) {
  printf("\nmq_shift_advance: inter-element alignment\n");
  static float a[N], b[N];
  float prev = 0.0f;
  for (uint32_t i = 0; i < N; ++i) { const float v = noise(); a[i] = 0.5f * (v + prev); prev = v; }

  /* An integer shift must UNDO a known delay, closing the loop with mq_pair_analyze: build b lagging
   * a by 3, advance b by 3, and the measured lag must come back to 0. */
  for (uint32_t i = 0; i < N; ++i) b[i] = (i >= 3u) ? a[i - 3u] : 0.0f;
  mq_pair_t p;
  mq_pair_analyze(a, b, N, 4, 4096u, &p);
  near("delay before the shift", p.lag_frac, 3.0, 0.05);
  mq_shift_advance(b, N, 3.0f);
  mq_pair_analyze(a, b, N, 4, 4096u, &p);
  near("integer advance removes the delay", p.lag_frac, 0.0, 0.05);
  near("and does not damage coherence", p.gamma, 1.0, 0.02);

  /* A NEGATIVE shift is the case an in-place implementation gets wrong: it reads BEHIND the write
   * cursor, so the loop has to run backward. Forward would overwrite each sample before reading it and
   * fill the buffer with the first value smeared across it. */
  for (uint32_t i = 0; i < N; ++i) b[i] = a[i];
  mq_shift_advance(b, N, -2.0f);
  mq_pair_analyze(a, b, N, 4, 4096u, &p);
  near("negative advance retards by exactly 2", p.lag_frac, 2.0, 0.05);
  near("backward iteration did not smear the buffer", p.gamma, 1.0, 0.02);

  /* The FRACTIONAL path — the reason this function exists, since the cross-channel stream offset is
   * ~11.6 samples and no integer shift can remove the 0.6. */
  for (uint32_t i = 0; i < N; ++i) b[i] = (i >= 12u) ? a[i - 12u] : 0.0f;
  mq_shift_advance(b, N, 11.6f);
  mq_pair_analyze(a, b, N, 16, 4096u, &p);
  /* Tolerance is wide because the estimator, not the shifter, is the loose part here: parabolic
   * interpolation of a correlation peak is biased toward the peak sample, and it reads this 0.4 as
   * about 0.35. That bias is why MICB_EXPECTED_LAG should be set from a MEASURED on-axis lag rather
   * than from the nominal spacing. */
  near("fractional advance leaves the expected 0.4", p.lag_frac, 0.4, 0.1);

  /* THE COST OF INTERPOLATING vs THE COST OF NOT. This is the measurement that justifies
   * MICB_ALIGN_FRAC, so it is asserted rather than asserted-about.
   *
   * An exact integer shift is lossless: it is a copy, so coherence must come back at 1.000 and any
   * drop would mean the shifter is damaging the signal. Two fractional shifts that sum to an integer
   * put the signal back in the same place but run it through the interpolator twice, and the residual
   * gap to 1.000 is what interpolation actually costs. On full-band noise that is ~2% for two passes
   * — against the ~8% that the 0.4-sample misalignment above costs by itself. Correcting the fraction
   * wins by roughly 4:1, and on speech (little energy near Nyquist) by more. */
  for (uint32_t i = 0; i < N; ++i) b[i] = (i >= 12u) ? a[i - 12u] : 0.0f;
  mq_shift_advance(b, N, 12.0f);
  mq_pair_analyze(a, b, N, 16, 4096u, &p);
  near("an integer shift is lossless", p.gamma, 1.0, 0.001);

  for (uint32_t i = 0; i < N; ++i) b[i] = (i >= 12u) ? a[i - 12u] : 0.0f;
  mq_shift_advance(b, N, 11.6f);
  mq_shift_advance(b, N, 0.4f);
  mq_pair_analyze(a, b, N, 16, 4096u, &p);
  ok("two interpolations cost little coherence", p.gamma > 0.96f, "");
  /* And they must not swallow the signal: the cubic is a mild low-pass, so a large RMS loss would
   * mean the coefficients are wrong (they must sum to 1 at every f). */
  near("interpolation preserves level", rms_of(b, N) / rms_of(a, N), 1.0, 0.15);

  /* Round trip: shift out and back must restore the signal, which no sign error survives. */
  static float c[N];
  for (uint32_t i = 0; i < N; ++i) { b[i] = a[i]; c[i] = a[i]; }
  mq_shift_advance(b, N, 3.25f);
  mq_shift_advance(b, N, -3.25f);
  double err = 0.0;
  /* Skip the edges: they replicate the boundary sample by design, so they cannot round-trip. */
  for (uint32_t i = 8u; i < N - 8u; ++i) err += fabs((double)b[i] - (double)c[i]);
  err /= (double)(N - 16u);
  /* Not exact: two interpolations low-pass twice and do not cancel. The bound is 15% of RMS on a
   * deliberately full-band signal (the cubic measures 12%, linear measured 22%) — loose enough not to
   * be brittle, tight enough that a sign error, which lands near 200%, cannot pass. */
  ok("round trip restores the signal", err < 0.15 * (double)rms_of(c, N), "");

  /* A zero shift must be a genuine no-op, not a pass through the interpolator. */
  for (uint32_t i = 0; i < N; ++i) b[i] = a[i];
  mq_shift_advance(b, N, 0.0f);
  int identical = 1;
  for (uint32_t i = 0; i < N; ++i) if (b[i] != a[i]) { identical = 0; break; }
  ok("zero shift is bit-exact", identical, "");
}

int main(void) {
  printf("host_mic_quality_test — dsp-mic-bench metrics against known-truth signals\n");
  test_fmt();
  test_db();
  test_analyze_sine();
  test_analyze_dynamics();
  test_analyze_clip_and_drift();
  test_pair();
  test_shift();
  test_track();
  test_tone();
  test_array_gain();
  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASS", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
