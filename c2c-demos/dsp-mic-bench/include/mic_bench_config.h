#ifndef C2C_DSP_MIC_BENCH_CONFIG_H
#define C2C_DSP_MIC_BENCH_CONFIG_H

#include <stdio.h>

#include "chip_config.h"
#include "hal_i2s.h"

/* ------------------------------------------------------------------------------------------------
 * dsp-mic-bench — objective audio-quality benchmark for the DSP 25 I2S microphone path.
 *
 * ONE microphone by default. The second mic is a build flag (MICB_STEREO), and when it is on the
 * SAME metric code runs on the primary mic (L), the second mic (R) and the combined signal (C), all
 * from ONE capture — so "did the array help?" is a printed difference, not a comparison between two
 * runs of two different binaries in two different rooms.
 *
 * That is the whole design constraint: every number must stay comparable across builds. Hence
 *   - a FIXED-LENGTH capture window (no VAD by default): the analysed duration cannot drift with how
 *     loudly you happened to speak, which is what would otherwise move the percentile levels;
 *   - the metrics live in src/mic_quality.c, which is chip-free and host-tested, so a metric can be
 *     wrong on the host in a second rather than after a flash;
 *   - the mono build's `L` record is produced by exactly the same call as the stereo build's `L`
 *     record. Enabling the second mic costs one extra FIFO read per sample pair and changes nothing
 *     about the primary mic's samples or their analysis.
 *
 * Every capture prints the full metric set, so ONE build covers all three stimuli — you just read
 * the column that matters:
 *   silence   -> nf (noise floor), dc, drift, hf     : self-noise / DC integrity of the mic path
 *   speech    -> act, snr, crest, clip, active       : usable dynamic range on real signals
 *   1 kHz tone-> level, sinad, thd                   : linearity / distortion of the analog path
 *
 * Standalone: no C2C link, no model, no accelerator. Stock linker script.
 *
 *   make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-mic-bench
 *   make tsi-run TTY=<tty> BINARY=build/c2c-demos/dsp-mic-bench/dsp-mic-bench.elf
 *
 * Add the second mic later, same binary layout, nothing else changed:
 *   make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-mic-bench \
 *       EXTRA_CMAKE_ARGS="-DMICB_STEREO=ON -DMICB_COMBINE=1"
 *
 * See README.md for the wiring, what each number means, and which one answers the array question.
 * ---------------------------------------------------------------------------------------------- */

#ifndef MICB_LOG_ENABLE
#define MICB_LOG_ENABLE 1
#endif
#if MICB_LOG_ENABLE
#define MICB_LOG(...) do { printf(__VA_ARGS__); } while (0)
#else
#define MICB_LOG(...) do { } while (0)
#endif

/* ---- clock ------------------------------------------------------------------------------------ */
/* PLL target. init_test() programs the PLL to (freq / SYS_CLK_FREQ) x 50 MHz and derives the UART
 * divisor from it, so this one value drives the clock AND the console. It also converts the
 * capture's rdcycle span into the measured sample rate, which is the check that the mic is actually
 * clocked at MICB_SAMPLE_RATE_HZ. (CLINT mtime is NOT used anywhere here: it is derived from the
 * core clock, so it reads ~50 MHz at every PLL setting — see the mtime entry in /CLAUDE.md.) */
#ifndef MICB_TARGET_FREQ_HZ
#define MICB_TARGET_FREQ_HZ 750000000ULL
#endif

/* ---- I2S mic (identical parameters to the proven dsp-citrinet / dsp-moonshine capture path) ---- */
#ifndef MICB_MIC_CHANNEL
#define MICB_MIC_CHANNEL 0
#endif
#ifndef MICB_SAMPLE_RATE_HZ
#define MICB_SAMPLE_RATE_HZ 16000u
#endif
#ifndef MICB_BITDEPTH
#define MICB_BITDEPTH 32u
#endif
#ifndef MICB_CLKDIV
#define MICB_CLKDIV 8
#endif
#ifndef MICB_SAMPLE_SHIFT
#define MICB_SAMPLE_SHIFT 8              /* 24-bit sample sits in the top of the 32-bit slot */
#endif
#ifndef MICB_FULLSCALE
#define MICB_FULLSCALE 8388608.0f        /* 2^23 — so 1.0 == digital full scale */
#endif

/* ---- capture window --------------------------------------------------------------------------- */
/* Length of the analysed window. Fixed, not VAD-trimmed, because the percentile levels below are
 * only comparable between runs if the analysed duration is the same. 2 s at 16 kHz = 32000 samples
 * = 100 frames of 20 ms, which is enough frames for a stable 10th/90th percentile. */
#ifndef MICB_CAPTURE_MS
#define MICB_CAPTURE_MS 2000u
#endif
#define MICB_CAPTURE_SAMPLES ((MICB_SAMPLE_RATE_HZ / 1000u) * MICB_CAPTURE_MS)

/* Samples read and thrown away before each window: drains whatever accumulated in the RX FIFO while
 * the previous window was being analysed and printed. Without it the first frames of every window are
 * stale audio from the printing gap, which shows up as a spuriously high noise floor. */
#ifndef MICB_DISCARD_SAMPLES
#define MICB_DISCARD_SAMPLES 3200u       /* 200 ms */
#endif

/* ONE-TIME warm-up at boot, before the first window. The per-run discard above is not enough for
 * this: measured on silicon, the first capture after power-up reads dc=0.0136fs with
 * drift=-0.0361fs and hf=-51 dB — that last number inverts to an energy centroid of about 7 Hz,
 * i.e. the run is the microphone's own high-pass filter settling, not audio. It made run 1 unusable
 * in every session so far and it polluted the rolling summary (a 40 dB spread in `nf`), which is the
 * number the whole array comparison depends on. One second of throw-away at boot removes it. */
#ifndef MICB_WARMUP_MS
#define MICB_WARMUP_MS 1000u
#endif
#define MICB_WARMUP_SAMPLES ((MICB_SAMPLE_RATE_HZ / 1000u) * MICB_WARMUP_MS)

/* Analysis frame for the percentile levels. 320 samples = 20 ms, matching the VAD frame used by
 * every other mic demo in this tree, so an energy number here is comparable to one there. */
#ifndef MICB_FRAME_SAMPLES
#define MICB_FRAME_SAMPLES 320u
#endif
#define MICB_NUM_FRAMES (MICB_CAPTURE_SAMPLES / MICB_FRAME_SAMPLES)

/* Percentiles of the per-frame RMS distribution.
 *   nf  = 10th : the quiet frames. In silence this IS the noise floor; in speech it is the floor
 *               between words, which is what an SNR should be measured against.
 *   act = 90th : the loud frames. Deliberately not the peak — one door slam should not define the
 *               signal level, and a percentile is reproducible where a max is not. */
#ifndef MICB_NF_PCT
#define MICB_NF_PCT 10u
#endif
#ifndef MICB_ACT_PCT
#define MICB_ACT_PCT 90u
#endif
/* A frame counts as "active" if its RMS exceeds nf * this. 4.0 = +12 dB over the floor. Only used
 * for the reported active fraction (a sanity check that you actually spoke into the window). */
#ifndef MICB_ACTIVE_MULT
#define MICB_ACTIVE_MULT 4.0f
#endif

/* |sample| at or above this (in full-scale units, BEFORE DC removal) counts as clipped. Any non-zero
 * clip count invalidates the distortion numbers for that window. */
#ifndef MICB_CLIP_LEVEL
#define MICB_CLIP_LEVEL 0.98f
#endif

/* A slot whose RMS is below this (dBFS) is treated as carrying no microphone signal: it is excluded
 * from the combined signal C and from the ARRAY verdict, however healthy the boot probe judged it.
 *
 * The probe cannot make this call. An undriven slot still FILLS at the sample rate, because the
 * peripheral clocks in whatever happens to be on SDIN during that slot; a floating line reads
 * 0xFFFFFFFF every frame, which is a constant, and a constant is a perfectly well-behaved FIFO
 * stream. Level is the only test that separates a microphone from an idle wire.
 *
 * -120 dBFS is far below anything physical: this part's own noise floor is about -91 dBFS (-26 dBFS
 * at 94 dB SPL, 65 dB SNR) and the measured room floor is -70 to -54. An undriven slot measures
 * -240 (exactly constant) or around -150 (constant with a bit of LSB dither), so the threshold has
 * 30+ dB of margin on both sides and needs no tuning. */
#ifndef MICB_SIGNAL_FLOOR_DBFS
#define MICB_SIGNAL_FLOOR_DBFS (-120)
#endif

/* ---- tone analysis --------------------------------------------------------------------------- */
/* Fundamental the tone fit looks for, in whole Hz (integer: this is used in #if, and the
 * preprocessor cannot compare floats). 0 disables the tone analysis entirely.
 *
 * It is computed on every capture — it costs a fraction of a second — and the numbers are only
 * MEANINGFUL while a tone is actually playing, which the reported sinad makes obvious: on speech or
 * silence there is no coherent fundamental and sinad collapses toward 0 dB.
 *
 * 1000 Hz is the convention for audio distortion measurements and sits well inside the mic's flat
 * band. Play it from a phone at a comfortable level: aim for level about -12 dBFS with clip = 0. */
#ifndef MICB_TONE_HZ
#define MICB_TONE_HZ 1000
#endif
/* Harmonics fitted above the fundamental (2f, 3f, ... up to this many INCLUDING f itself).
 * 5 covers 5 kHz at f0 = 1 kHz, comfortably below the 8 kHz Nyquist. */
#ifndef MICB_TONE_HARMONICS
#define MICB_TONE_HARMONICS 5
#endif
/* Window the tone is fitted over, taken from the middle of the capture.
 *
 * Shorter than the full capture ON PURPOSE. A coherent fit needs the frequency to be known to about
 * 1/(2T): over the full 2 s window that is 0.25 Hz, and neither a phone's tone generator nor this
 * chip's I2S clock is that accurate — a 1 Hz error slips two entire cycles across the window and the
 * fitted amplitude collapses, which would show up as catastrophic distortion that is not there. Over
 * 0.5 s the requirement is a much more realistic 1 Hz, and the search below covers the rest. 8192
 * samples is also ~500 cycles of a 1 kHz tone, far more than enough to measure its harmonics. */
#ifndef MICB_TONE_WINDOW
#define MICB_TONE_WINDOW 8192u
#endif
/* The frequency search around MICB_TONE_HZ, in percent, and how many grid points. +-2% of 1 kHz is
 * +-20 Hz, which covers any real generator/clock mismatch; 33 points is 1.25 Hz spacing, refined
 * further by parabolic interpolation. The refined value is PRINTED, so a tone generator that is not
 * where you think it is becomes visible rather than becoming "distortion". */
#ifndef MICB_TONE_SEARCH_PCT
#define MICB_TONE_SEARCH_PCT 2
#endif
#ifndef MICB_TONE_SEARCH_STEPS
#define MICB_TONE_SEARCH_STEPS 33
#endif

/* ---- microphone count and slot map ------------------------------------------------------------- */
/* MICB_MICS: 1, 2 or 4. The slot map is fixed and deliberate:
 *
 *   M0 = channel 0 left     M1 = channel 0 right     <- one channel, one clock: PROVABLY sample-locked
 *   M2 = channel 1 left     M3 = channel 1 right     <- second channel: coherence is what we measure
 *
 * Two mics per channel come from the L/R slots of the SAME frame, which is the strong case: they
 * share BCLK and WS, so they are sample-locked by construction. Nothing about the clock setup changes
 * to enable the second one — set_I2S_sample_freq already computes mclk = rate * bits * 2, and that *2
 * IS the stereo frame, so the right slot is already on the wire and being clocked; today it is read
 * and discarded. `rx_force_left` must stay 0 (it is) or both slots collapse into the left FIFO.
 *
 * Wiring per channel: mics share that channel's BCLK, LRCLK/WS and SDIN; SEL->GND selects the left
 * slot, SEL->VDD the right. I2S mics tri-state during the other's slot, so one shared data line is
 * correct rather than a bodge.
 *
 * GOING TO 4 MICS MEANS A SECOND CHANNEL, AND THAT IS THE ONE OPEN QUESTION.
 * Each channel has its own clkgen_en and its own I2S_CLKDIV, which is why the older notes in this
 * tree call two channels "two drifting time bases". Reading the HAL, that looks overstated: BCLK is
 * sys_clk / (2*(N+1)) on BOTH channels, i.e. two integer dividers counting the SAME PLL-derived core
 * clock. Identical clkdiv therefore means identical FREQUENCY with no drift possible; what is unknown
 * is the fixed PHASE offset, set by how far apart the two dividers were started. Configure the two
 * channels back-to-back and that is a few core cycles (~5-10 ns against a 62.5 us sample period).
 *
 * That is a hypothesis, not a measurement, and it is exactly what MICB_TRACK_BLOCKS tests: a shared
 * clock source implies a lag that does not move within a capture, while genuinely separate sources
 * show a steady slope. Validate the one-channel pair FIRST (M0/M1, where coherence is guaranteed) so
 * you have a calibrated reference for what a good gamma and a stable lag look like on this rig; then
 * any degradation across channels is attributable to the clocking rather than to the mics or the
 * mounting.
 *
 * An unwired or unclocked mic must never hang the core: read_I2S_rx spins forever on an empty FIFO,
 * so every slot is probed with the non-blocking empty flag under a bounded spin at boot, and a dead
 * slot is dropped from the read set (decided ONCE, never per sample). */
#ifndef MICB_MICS
#define MICB_MICS 1
#endif
#if (MICB_MICS < 1) || (MICB_MICS > 4)
#error "MICB_MICS must be 1..4 (1,2 = channel A's L/R; 3 adds channel B left; 4 adds channel B right)"
#endif
/* Second I2S channel. Used from MICB_MICS == 3 up: M2 is its LEFT slot (SEL -> GND), M3 its RIGHT
 * (SEL -> VDD), exactly as M0/M1 sit on channel A. */
#ifndef MICB_MIC_CHANNEL_B
#define MICB_MIC_CHANNEL_B 1
#endif

/* Split each capture into this many sub-blocks and measure the inter-mic lag in each one.
 *
 * This is the clocking discriminator, and it is the reason the 4-mic question is answerable in
 * software instead of by inspecting RTL. Two dividers fed from one core clock cannot have a frequency
 * offset, so the measured lag must be FLAT across the capture. Any steady slope means the two sample
 * streams really are on separate time bases: a 1 ppm offset is 0.016 samples/s at 16 kHz, and the lag
 * is measurable to ~0.05 samples, so a 2 s capture resolves a few ppm. 0 disables. */
#ifndef MICB_TRACK_BLOCKS
#define MICB_TRACK_BLOCKS 8u
#endif

/* How the mics are folded into the C (combined) signal that gets its own metric record.
 *
 * With MICB_MICS == 4, mode 1 averages ALL live mics; modes 2 and 3 stay pair operations on M0/M1,
 * because a 4-element difference is not a meaningful generalisation of a gradient — that is a
 * beamformer, and a beamformer belongs behind a measured beam pattern rather than in a combine knob.
 *
 *   0 = primary only          — the A/B control: captures all, combines nothing. C == M0, so any
 *                              difference between the C and L records is a bug in the harness, not
 *                              an effect of the array. Run this first.
 *   1 = average               — speech is coherent across closely spaced mics so it adds in
 *                              amplitude, while UNCORRELATED noise (mic self-noise, ADC noise) adds
 *                              in power: about 3 dB. It does nothing for correlated noise (room
 *                              tone, reverb, a competing talker), which is most real-room noise.
 *
 *                              It is an AVERAGE and not a SUM on purpose, and the reason is worth
 *                              recording because the change looks tempting. A sum is the average
 *                              times N — a constant — and a constant cannot change an SNR: it
 *                              scales signal and noise together. Summing 3 mics moves d_rms and
 *                              d_nf by +9.5 dB and leaves d_snr EXACTLY where it was. The array
 *                              gain lives in the coherence structure (who is delayed, who is
 *                              weighted, who is subtracted), never in the scale factor.
 *                              Dividing by N also keeps C's level comparable with M0's, which is
 *                              what makes d_rms and d_nf readable at all, and it makes clipping
 *                              impossible when no input clips — a sum of 3 mics peaking at -6 dBFS
 *                              lands at +3.5 dBFS and clips, which would corrupt the very numbers
 *                              the mode exists to produce.
 *   2 = difference            — a first-order gradient (cardioid-ish) response that rejects sound
 *                              arriving equally at both mics, i.e. distant and diffuse sources. It
 *                              also high-pass filters speech and is very sensitive to inter-mic gain
 *                              mismatch, so read the pair record's level mismatch before believing
 *                              it.
 *   3 = aligned sum           — mode 1 with the measured inter-mic delay compensated first, so
 *                              off-axis speech stays coherent. One sample = 2.1 cm of path
 *                              difference at 16 kHz, so this only does something with mics more than
 *                              about 2 cm apart; closer than that it degenerates into mode 1.
 *   4 = endfire cardioid     — the first-order ENDFIRE beamformer over the M0/M1 pair:
 *                              C[n] = M0[n] - M1[n - MICB_ENDFIRE_DELAY]. Steered along the mic axis,
 *                              looking out past M0, with the null behind M1. Unlike mode 2 (whose
 *                              null is broadside, to the SIDES) this is the mode that rejects a
 *                              source BEHIND the array, which is what an endfire array is for.
 *   5 = endfire delay-and-sum — the same look direction over ALL contributing mics:
 *                              C[n] = (1/N) * sum_m M_m[n - (N-1-m)*MICB_ENDFIRE_DELAY]. Delays the
 *                              front mic most so an on-axis wavefront lands in phase. Gains
 *                              10*log10(N) on uncorrelated noise like mode 1, and adds directivity at
 *                              the frequencies where the aperture is comparable to a wavelength (a
 *                              6.5 cm four-element aperture: roughly above 2 kHz). Well-conditioned —
 *                              no differencing, so no low-frequency noise penalty. */
#ifndef MICB_COMBINE
#define MICB_COMBINE 1
#endif

/* Inter-element delay for the endfire modes (4 and 5), in SAMPLES.
 *
 * This is why the mic spacing should be chosen to make it an integer. One sample at 16 kHz is
 * 344/16000 = 2.15 cm of travel, so mics spaced 2.15 cm apart along the look axis are exactly one
 * sample apart and every steering delay in an endfire array of any length is an integer sample count.
 * No fractional-delay filter, no interpolation error, nothing to tune. 2.15 cm is also exactly the
 * lambda/2 spatial-aliasing limit for the full 8 kHz band (d < c/(2*f_max)), so it is simultaneously
 * the largest spacing that produces no grating lobes anywhere below Nyquist.
 *
 * Double the spacing to 4.3 cm and this becomes 2 — still integer, but grating lobes appear above
 * 4 kHz and the beam stops being a beam in the fricative band. Raising the sample rate does NOT fix
 * that: the aliasing limit depends only on d and f_max. */
#ifndef MICB_ENDFIRE_DELAY
#define MICB_ENDFIRE_DELAY 1
#endif

/* Per-mic gain correction in dB, applied ONLY when building C. The per-mic records stay raw so they
 * remain directly comparable with every earlier log, and so the pair record keeps measuring the real,
 * uncorrected mismatch.
 *
 * These exist because inter-mic gain mismatch is the hard ceiling on every null: subtracting two
 * copies of one sound that differ in gain by a factor g leaves |1-g| behind, so 5 dB of mismatch caps
 * the deepest achievable null at -7 dB no matter how good the beamformer is, while 1 dB caps it at
 * -19 dB and 0.5 dB at -25 dB. Mode 1 (averaging) barely notices; modes 2 and 4 are governed by it.
 *
 * How to set them: place every mic the same distance from one steady broadband source, run mode 0,
 * and read the CAL line — it prints the offsets that would equalise the mics. Then rebuild with those
 * values. Do NOT calibrate from a directional source: the level differences are real signal then. */
#ifndef MICB_GAIN_DB_M1
#define MICB_GAIN_DB_M1 0.0f
#endif
#ifndef MICB_GAIN_DB_M2
#define MICB_GAIN_DB_M2 0.0f
#endif
#ifndef MICB_GAIN_DB_M3
#define MICB_GAIN_DB_M3 0.0f
#endif

/* ---- liveness-only mode ---------------------------------------------------------------------- */
/* 1 = capture, report per-slot LEVEL and an alive/constant/no-data verdict, and stop there. No pair,
 * combine, array or tone records.
 *
 * For bringing up a newly wired microphone, which is a different question from how good the array is —
 * and answering the second while the first is open is how a wiring fault gets read as an acoustic
 * result. Also the safe order: a slot that reports data can then be trusted by everything else. */
#ifndef MICB_PROBE_ONLY
#define MICB_PROBE_ONLY 0
#endif

/* ---- inter-slot FIFO re-alignment ------------------------------------------------------------ */
/* 1 = cycle rx_en and drain every slot's FIFO independently to empty before each capture, instead of
 * removing a fixed block count from each. See mic_resync() in src/main.c: the fixed-count drain
 * preserves any offset between the two FIFOs, and on silicon the two sides demonstrably lose
 * different sample counts during the inter-run gap (the lag came out bimodal, two clusters exactly
 * one sample apart, from a source that never moved). Set to 0 to reproduce the old behaviour. */
#ifndef MICB_RESYNC
#define MICB_RESYNC 1
#endif
/* Bound on the drain-to-empty loop, so a slot that never reports empty costs a delay, not a hang. */
#ifndef MICB_RESYNC_MAX_BLOCKS
#define MICB_RESYNC_MAX_BLOCKS 8192u
#endif
/* A run's lag must correlate at least this well before it is trusted — an uncorrelated window's lag
 * is a random number and would produce imaginary slips. Raised from 0.5 after run 1 of a silicon log
 * became the slip REFERENCE at gamma=0.575 / gamma_min=0.011, was itself in the slipped cluster, and
 * so inverted every subsequent verdict: the four correctly-aligned runs got flagged and the four
 * slipped ones did not. A reference has to be better-founded than the thing it judges. */
#ifndef MICB_SLIP_MIN_GAMMA
#define MICB_SLIP_MIN_GAMMA 0.7f
#endif

/* THE LAG YOUR GEOMETRY SHOULD PRODUCE, in samples — the operator's knowledge, which is what breaks
 * the tie the hardware cannot.
 *
 * Because `rx_en` cycling does not flush the FIFOs, the two streams still arrive with a random
 * integer offset. A measured lag is therefore `true + slip`, two unknowns from one number, and no
 * amount of signal processing resolves it. But YOU know where you put the source, and the physical
 * range is tiny: 2.2 cm of spacing permits only +-1.02 samples of true lag. So:
 *
 *   - Source ON AXIS (endfire), in front of M0: expected = +spacing in samples (+1.0 at 2.15 cm).
 *     The slipped alternatives are 0.0 and +2.0 — and +2.0 is physically impossible, while 0.0 means
 *     a broadside source, which contradicts where you put it. Unambiguous.
 *   - Source BEHIND the rear mic: expected = -1.0. Same argument.
 *   - Source EQUIDISTANT (broadside): expected = 0.0 — and this geometry is the one case that CANNOT
 *     be disambiguated, since 0.0 and +-1.0 are all physically allowed. Do not run a slip-sensitive
 *     measurement broadside.
 *
 * Set this to match the geometry you actually built, then MICB_SLIP_FIX removes the offset.
 *
 * WITH 4 MICS this is the PER-ELEMENT-GAP lag, and every other element is derived from it: for a
 * uniform linear array in a plane wave, element m's lag is exactly m times element 1's. So one number
 * describes the whole array's expected geometry, and MICB_ALIGN uses m * this (plus the channel offset
 * for cross-channel elements) as element m's expected lag. On axis at 2.15 cm spacing that is
 * 1.0 / 2.0 / 3.0 samples for M1 / M2 / M3. */
#ifndef MICB_EXPECTED_LAG
#define MICB_EXPECTED_LAG 0.0f
#endif
/* 1 = shift M1 by the detected integer slip, in place, right after capture and BEFORE any analysis,
 * so every downstream number is computed on aligned data and the pair line reports the true acoustic
 * lag. The correction is always logged; it is never silent. */
#ifndef MICB_SLIP_FIX
#define MICB_SLIP_FIX 1
#endif
/* How close to a whole number of samples the lag change must be to be called a slip rather than the
 * source or the talker moving.
 *
 * Was 0.3 and that was too tight: the back-facing silicon log missed by 0.01 — measured lag -0.31
 * against an expected -1.00 is 0.69 away, i.e. 0.31 from a whole sample — so six of nine runs went
 * uncorrected and the measured front/back ratio came out 2 dB smaller than it should have. Since the
 * ambiguity is exactly +-1 sample, anything below 0.5 is a valid decision boundary; 0.4 keeps a guard
 * band while tolerating a nominal MICB_EXPECTED_LAG that is off by a tenth of a sample. Better still,
 * set MICB_EXPECTED_LAG from the lag you MEASURED on axis rather than from the nominal spacing. */
#ifndef MICB_SLIP_TOLERANCE
#define MICB_SLIP_TOLERANCE 0.4f
#endif

/* Lag search half-range in samples for the inter-mic correlation (and for combine mode 3).
 *
 * 4 samples = 8.6 cm of path difference at 16 kHz, which covers any sane desk-mic SPACING — and that
 * reasoning is exactly why it was wrong for the 4-mic build. Two mics on DIFFERENT I2S channels are
 * additionally separated by a fixed stream offset that has nothing to do with acoustics (measured
 * ~11.6 samples, see MICB_CHAN_OFFSET), so every cross-channel pair pinned at the +-4 limit and read
 * as "the channels are incoherent". They were not; the window was too small.
 *
 * RULE: a lag that comes back exactly at +-max_lag is not a measurement, it is a saturated search.
 * Widen this until the peak is interior before concluding anything from gamma. 256 costs one static
 * 4 KB array in mic_quality.c and a longer correlation. */
#ifndef MICB_MAX_LAG
#define MICB_MAX_LAG 4
#endif

/* ---- array geometry and inter-element alignment ----------------------------------------------- */
/* Physical inter-element spacing in cm, along the array axis. Only used to print the PHYSICAL BOUND
 * on an acoustic lag, which is what makes an integer slip identifiable: at 2.15 cm one element gap is
 * 1.00 sample at 16 kHz, so any same-channel pair whose true lag exceeds ~1.02 samples is reporting
 * acoustics plus a slip, and the excess must be an integer. */
#ifndef MICB_SPACING_CM
#define MICB_SPACING_CM 2.15f
#endif

/* THE CONSTANT STREAM OFFSET BETWEEN THE TWO I2S CHANNELS, in samples.
 *
 * The two channels' RX streams start a fixed distance apart — their clock dividers count the same core
 * clock (so there is no drift, confirmed at < 3 ppm) but the dividers are STARTED at different times,
 * and the gap between the two configure calls is whatever the boot path takes. Measured on silicon:
 * ~11.6 samples (~725 us), consistent to 0.16 samples across three independent pair estimates and
 * stable across all runs of a boot.
 *
 * IT IS NOT AN INTEGER AND IT MAY DIFFER PER BOOT (the observed ch-start gap was 3208 core cycles on
 * one boot and 6422 on the next), so it is a starting point rather than a constant of the hardware.
 *
 * THIS KNOB IS DOCUMENTATION, NOT A TUNING PARAMETER — it cancels identically out of the correction
 * MICB_ALIGN applies, and understanding why is the crux of how the 4-mic alignment works.
 *
 * The offset's INTEGER part is not separable from the per-capture integer FIFO slip: both are whole
 * samples on the same lag, and no acoustic measurement can attribute one to the channel and the other
 * to the FIFO. That would be fatal if either had to be known on its own. Neither does. MICB_ALIGN
 * measures what it actually needs — the total non-acoustic part of the lag, offset and slip together —
 * and removes all of it every run. So channel B lands aligned whatever the split was, and it stays
 * aligned across a boot that changed the offset, with nothing configured.
 *
 * The FRACTIONAL part is the physically interesting half, because it is what no integer shift can
 * remove, and it IS separable: a slip moves a lag by exactly 1.000, so it cannot touch a fraction.
 * That is why one coherent capture measures it, and why MICB_ALIGN_FRAC can correct it from a session
 * mean rather than needing it configured.
 *
 * So set this only to record what a board measured (it is printed back in the ALIGN line as the
 * predicted lag, which makes an unexpected reading obvious). The value to record is what the SETUP
 * block reports.
 *
 * Sign: positive means channel B's stream LAGS channel A's, i.e. M2/M3 samples arrive later in their
 * buffers than the same acoustic event in M0/M1. */
#ifndef MICB_CHAN_OFFSET
#define MICB_CHAN_OFFSET 0.0f
#endif

/* MICB_SLIP_FIX (above) is the master switch for the alignment pass, and with MICB_MICS > 2 it aligns
 * EVERY element against M0 rather than only M1. The array cannot work without it: each element's
 * stream carries its own independent integer slip, so summing four raw streams sums four
 * differently-shifted copies of the same sound. That is what produced d_nf = +3.9 dB — averaging made
 * the noise floor WORSE — in the first 4-mic log.
 *
 * Model, per element m, all in samples:
 *      measured(m) = m * MICB_EXPECTED_LAG            <- acoustics: uniform linear array, plane wave,
 *                                                        so element m's lag is exactly m times M1's.
 *                                                        THIS TERM MUST SURVIVE; it is the beam.
 *                  + channel offset (cross-channel)   <- constant per boot, non-integer
 *                  + slip(m)                          <- an INTEGER, random per capture
 * The first term is known, so the rest is what is left, and it is what gets shifted out. Every
 * correction is logged; none is silent.
 *
 * 1 = measure and report all of that but DO NOT shift anything. For the first look at a new board,
 * where the question is what the offsets ARE rather than whether removing them helps. */
#ifndef MICB_ALIGN_MEASURE_ONLY
#define MICB_ALIGN_MEASURE_ONLY 0
#endif

/* 1 = also remove the FRACTIONAL part of the residual with a linear-interpolating delay.
 *
 * The integer shift cannot touch it, and it is not small compared with what the beamformer is doing:
 * the measured channel offset's fractional part was ~0.6 samples, while the entire endfire steering
 * delay at 2.15 cm spacing is 1.0 sample. Left in place it is a 0.6-sample phase error on half the
 * array — at 4 kHz that is 54 degrees, and those two elements then partly cancel the other two.
 *
 * Linear interpolation is a mild low-pass (worst case at Nyquist, -3.9 dB at 8 kHz for a half-sample
 * shift) and it is applied to ONE channel's pair only, so it slightly colours M2/M3 relative to M0/M1.
 * That trade is worth taking: a coherence error costs far more than a fraction of a dB of treble on
 * half the elements. Set to 0 to measure how much the fractional part is actually costing. */
#ifndef MICB_ALIGN_FRAC
#define MICB_ALIGN_FRAC 1
#endif

/* 1 = MEASURE the source direction every capture instead of being told it, so the source can be moved
 * anywhere along the array axis without rebuilding. This is the mode for sweeping a BEAM PATTERN.
 *
 * MICB_EXPECTED_LAG is ignored when this is on (it is the "operator states the geometry" path, which
 * is right for a single calibration and wrong for a sweep — ten source positions would be ten builds,
 * and a stale sign silently re-times one direction into the other).
 *
 * How it works, and why it needs BOTH timing and level. For a plane wave from any angle, a uniform
 * linear array's acoustic lags are 0, tau, 2*tau — ONE unknown, with |tau| <= one gap. Measuring two
 * lags therefore leaves a redundant equation, which pins tau and the per-element integer slips
 * together... except for one irreducible ambiguity: at 2.15 cm and 16 kHz a gap is EXACTLY 1.000
 * sample, so tau and tau-1 are BOTH inside the permitted range, and they describe opposite directions.
 * Timing alone cannot separate front from back at this spacing. (That is the hidden cost of choosing
 * the spacing that makes every steering delay an integer — worth knowing before picking it again.)
 *
 * LEVEL breaks the tie, unambiguously: the nearer microphone is louder. M0 louder => the source is
 * ahead => tau > 0; M1 louder => behind => tau < 0. Measured swing between the two geometries on
 * silicon was a clean 3 dB, far above the ~0.3 dB the level difference varies within one direction. So
 * the sign comes from level, the magnitude from timing, and the second element's residual settles any
 * remaining edge case.
 *
 * MICB_CHAN_OFFSET IS USED here, unlike in the operator-stated path: with the source position unknown
 * there is nothing else to separate the fixed cross-channel stream offset from the acoustics. Measure
 * it once per boot with MICB_CAL_REPORT and pass it in. It is fixed within a boot but NOT across boots
 * (observed 7.8 to 15.1 samples), so a value from yesterday is worthless. */
#ifndef MICB_ALIGN_SWEEP
#define MICB_ALIGN_SWEEP 0
#endif
/* Level difference (dB) below which the direction is treated as unknown and the smallest |tau| is
 * taken. A broadside source genuinely has no level gradient and no lag, so this is the right answer
 * there rather than a coin toss. */
#ifndef MICB_SWEEP_LEVEL_DEADBAND
#define MICB_SWEEP_LEVEL_DEADBAND 0.4f
#endif

/* 1 = print the per-element SETUP/OFFSET calibration report and the across-run aggregate.
 *
 * This is the "calibrate the array" mode. It prints, per element and per run, the measured lag, the
 * lag the geometry predicts, and the residual split into its integer (slip) and fractional (channel
 * offset) parts — then accumulates them so the SETUP block can emit paste-ready -D flags with the
 * run-to-run spread next to each one. A constant measured once is a guess; a constant measured eight
 * times with its spread is a calibration. */
#ifndef MICB_CAL_REPORT
#define MICB_CAL_REPORT 0
#endif

/* Window used for the inter-mic correlation, taken from the middle of the capture. Long enough to
 * average over several pitch periods, short enough to stay inside one talker position. */
#ifndef MICB_XCORR_WINDOW
#define MICB_XCORR_WINDOW 8192u
#endif

/* Bounded spin for the right-slot probe. At 750 MHz this is a few milliseconds — far longer than the
 * 62.5 us between 16 kHz samples, so a live slot is always seen. */
#ifndef MICB_PROBE_SPINS
#define MICB_PROBE_SPINS 2000000u
#endif

/* ---- run control ----------------------------------------------------------------------------- */
/* 0 = run forever. Otherwise stop after this many windows (useful for a scripted capture). */
#ifndef MICB_RUNS
#define MICB_RUNS 0u
#endif
/* Rolling summary (min / mean / max of the headline numbers) printed every N windows. A single
 * 2-second capture of room noise moves by a decibel or so run to run; the summary is what you
 * actually compare between builds. 0 disables it. */
#ifndef MICB_SUMMARY_EVERY
#define MICB_SUMMARY_EVERY 8u
#endif

/* Optional speech gate: wait for a frame above this energy before starting the window, instead of
 * capturing immediately. Convenience for the speech stimulus (it stops you from having to talk
 * continuously); OFF by default because an unconditional window is what keeps the noise-floor
 * numbers comparable. Threshold is AC energy per frame in full-scale units squared, the same
 * quantity and the same default as dsp-citrinet's VAD. */
#ifndef MICB_VAD_ENABLE
#define MICB_VAD_ENABLE 0
#endif
#ifndef MICB_VAD_THRESHOLD
#define MICB_VAD_THRESHOLD 1.5e-3f
#endif

/* Print the I2S registers (config/status/watermarks) at boot. The RX right-slot watermark is direct
 * evidence about whether the second mic's slot carries data, which is the one genuinely new hardware
 * claim in the stereo mode. */
#ifndef MICB_DUMP_REGS
#define MICB_DUMP_REGS 1
#endif

#endif /* C2C_DSP_MIC_BENCH_CONFIG_H */
