# dsp-mic-bench — audio-quality benchmark for the DSP 25 microphone path

One microphone, then two, then four, with numbers that stay comparable across each step.

```bash
# 1 mic (default)
make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-mic-bench
make tsi-run TTY=<tty> BINARY=build/c2c-demos/dsp-mic-bench/dsp-mic-bench.elf

# 2 mics: channel A's L and R slots — one BCLK/WS, sample-locked by construction
make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-mic-bench EXTRA_CMAKE_ARGS="-DMICB_MICS=2"

# 4 mics: adds channel B's two slots. Do this only after the 2-mic step passes.
make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-mic-bench EXTRA_CMAKE_ARGS="-DMICB_MICS=4"
```

`MICB_MICS` is 1, 2 or 4, with a fixed slot map: `M0`/`M1` = channel A left/right, `M2`/`M3` =
channel B left/right. **Bring them up in that order.** The 2-mic step runs on one channel, where the
two mics provably share BCLK and WS, so it tells you what a good `gamma` and a stable `lag` look like
*on this rig*. Without that reference number, a cross-channel measurement can't be judged — you
wouldn't know whether a wobbly lag means the clocks are separate or just that the room is quiet.

Real silicon only (Spike models no I2S). No C2C link, no model, no accelerator, stock linker script —
nothing in the failure surface but the microphone path being measured.

## What it does

Every run captures a **fixed 2-second window** and prints the full metric set. One build covers all
three stimuli; you read the column that matters:

| stimulus | look at | it tells you |
|---|---|---|
| silence | `nf`, `dc`, `drift`, `hf` | the mic's own noise floor and DC integrity |
| speech | `act`, `snr`, `crest`, `clip`, `active` | usable dynamic range on real signals |
| 1 kHz tone | `f0`, `level`, `sinad`, `thd` | linearity and distortion of the analog path |

The window is a fixed length on purpose — no VAD by default. Percentile levels are only comparable
between runs if the analysed duration is the same, and a VAD-trimmed window silently changes it based
on how loudly you happened to speak. (`-DMICB_VAD=ON` gates on speech if you want it; the noise-floor
numbers then stop being comparable with an ungated build.)

## Reading a record

```
[mic-bench] --- run=3 captured 32000 samples, measured rate=16001Hz (configured 16000)
[mic-bench] run=3 sig=M0 n=32000 dc=-0.0457fs drift=0.0002fs rms=-38.2 peak=-19.4 crest=18.8 \
            nf=-58.1 act=-36.7 snr=21.4 hf=-8.4 clip=0 active=0.43
[mic-bench] run=3 tone sig=M0 f0=997.3Hz level=-21.0 sinad=52.4 thd=-58.1 h2=-61.0 h3=-64.2 n=8192
```

- `rate` first, and deliberately so: every level below is interpreted against the configured sample
  rate, so if the mic is not actually clocked at 16 kHz then all of it is quietly wrong. It is
  measured from `rdcycle` across the capture (the FIFO read blocks, so elapsed cycles per sample *are*
  the mic clock). More than 5% off and the line says so. Not from CLINT `mtime` — that is derived from
  the core clock and reads ~50 MHz at every PLL setting (`/CLAUDE.md`).
- All dB values are **dBFS**: 0 dB is digital full scale.
- `dc` / `drift` are in full-scale units. These MEMS parts sit on a large per-part DC offset (~-0.046
  FS on this rig); `drift` is `mean(2nd half) - mean(1st half)`, so a wandering baseline shows up
  there rather than as mysterious low-frequency noise.
- `nf` / `act` are the 10th / 90th percentile of the per-frame RMS (20 ms frames). Percentiles, not
  min and max: one door slam should not define the signal level, and a percentile is reproducible.
- `snr` = `act - nf`. On silence it is near 0 dB, which is the correct answer for a signal with no
  dynamics — not a failure.
- `crest` = `peak - rms`: 3.0 dB for a sine, 10-18 dB for speech. Much lower than that with `clip > 0`
  means the input is being flattened.
- `hf` = `10*log10(mean(diff²)/mean(x²))`, a one-number spectral tilt. Strongly negative means rumble
  or DC wander dominates; strongly positive means hiss. For a pure tone it is exactly
  `20*log10(2*sin(pi*f/fs))` (-18.1 dB at 100 Hz, +6.0 dB at 8 kHz), which is also how the host test
  checks it.
- `active` is the fraction of frames more than 12 dB above the floor — a sanity check that you
  actually spoke into the window.
- The tone line prints the **measured** `f0`. If it is not close to `MICB_TONE_HZ`, the generator or
  the mic clock is off, and that matters before reading the distortion figures (see below).
- Every 8 runs a `SUMMARY` block prints min/mean/max. A single 2-second capture of room noise wanders
  by a decibel or so; **compare summaries, not single records.**

## Going to 4 mics: the clocking question, and how the TRACK line answers it

4 mics needs a **second I2S channel** — `SEL` only gives two slots per channel, and there is no TDM
here (`ws_len` maxes out at a 2-slot frame). Every older note in this tree says two channels are "two
independent, drifting time bases", which would rule the whole thing out.

Reading the HAL, that looks overstated. `set_I2S_sample_freq` gives `BCLK = sys_clk / (2*(N+1))` on
both channels: two integer dividers counting **the same PLL-derived core clock**. Identical `clkdiv`
therefore means identical frequency and *no possible drift* — there is no second oscillator to drift
against. What's genuinely unknown is the fixed **phase** offset, set by how far apart the two dividers
started. This demo configures the two channels back-to-back with nothing in between and prints the gap
in core cycles; at 750 MHz that's single-digit nanoseconds against a 62.5 µs sample period.

That is a hypothesis, and `MICB_TRACK_BLOCKS` tests it. Each capture is split into sub-blocks and the
inter-mic lag is measured in each one:

```
[mic-bench] run=3 TRACK M2-M0 blocks=8 lag 0.31->0.33 spread=0.04 slope=0.008smp/s (0.5ppm) gamma_min=0.94
```

A frequency offset between two sample streams shows up as a lag that grows linearly. At 16 kHz, 1 ppm
is 0.016 samples/s and the lag is measurable to ~0.05 samples, so a 2 s capture resolves single-digit
ppm.

| what the cross-channel TRACK shows | meaning | what to do |
|---|---|---|
| `slope` ≈ the same-channel pair's slope, `spread` small | one effective clock domain | **4 mics work now**; compensate the fixed integer lag |
| flat within a run, `lag_first` differs across boots | phase depends on start order | 4 mics work, add a **boot calibration** of the offset |
| steady non-zero `slope`, many ppm | genuinely separate time bases | needs shared clocks (below) or RTL work |
| low `gamma_min` | the lag is being read out of noise | fix coherence before believing any slope |

**Measure the same-channel pair's `slope` first — that is the instrument's own noise floor.** Judge the
cross-channel number against it, not against zero.

If it does turn out that the channels are independent, the fallback is to set channel B `clkgen=0` and
physically wire channel A's BCLK/WS to channel B's clock pins. Whether those pins are inputs in that
mode is an RTL/pinout question this tree can't answer — worth asking whoever owns the I2S block before
committing to a board layout.

One quirk to watch with a second channel: `dsp-i2s-test` found that this I2S master may only generate
BCLK/WS while its TX FIFO has data (`DSP_I2S_TEST_KEEP_CLOCK`). A channel whose clock never starts has
empty FIFOs, and `read_I2S_rx` blocks forever on empty — which is why every slot is probed with the
non-blocking empty flag at boot and a dead slot is dropped from the read set.

## Step 2: the second microphone (same channel)

The pair goes on the **L and R slots of one channel** — the strong case, and the reason it comes first:
they share that channel's BCLK and WS, so they are sample-locked by construction and any lag you
measure is acoustics or wiring, never clocking.

Wiring: share BCLK, LRCLK/WS and SDIN; `SEL->GND` = left (`M0`, the primary), `SEL->VDD` = right
(`M1`). I2S mics tri-state during the other's slot, so the single shared data line is correct rather
than a bodge. Nothing about the clock setup changes: `set_I2S_sample_freq` already computes
`mclk = rate * bits * 2`, and that `*2` **is** the stereo frame — the right slot is already on the wire
and being clocked; at `MICB_MICS=1` it is simply never read.

At boot every slot is probed with the non-blocking empty flag under a bounded spin (never with
`read_I2S_rx`, which spins forever on an empty FIFO — probing with a read would turn "mic not wired"
into a silent hang indistinguishable from a dead chip). A dead slot is dropped from the read set, once,
at boot. **No I2S right-slot read in this tree has ever carried real mic data**, so `probe M1 ... ALIVE`
is testing an unproven hardware claim, not just your soldering.

### Is the second mic actually alive? (the probe does not tell you)

**`probe M1 ... ALIVE` proves the FIFO is being filled, not that a microphone exists.** The peripheral
clocks in whatever is on SDIN during the right slot, so with a single mic connected the right FIFO
already reads non-empty — visible in a 1-mic log as `status=0x00 rx_wm(L/R)=1/1`. The probe exists to
stop `read_I2S_rx` hanging on an empty FIFO; it is not a presence test.

Four things that *are*:

| check | one mic | two mics |
|---|---|---|
| `M1 rms` | `-240.0` with `<- NOT DRIVEN` appended (a floating line reads the same value every frame, so it is a constant and vanishes when DC is removed) | within a few dB of `M0` |
| `gamma`, **with a loud common source** (talk, clap, tone) | stays near 0 | goes high (>0.7) |
| tap one mic at a time | only `M0` reacts, whatever you tap | tapping A moves `M0`, tapping B moves `M1` |
| `M0 nf` vs your saved 1-mic baseline | unchanged | unchanged — if it *changed*, the second mic's wiring is disturbing the primary (see below) |

The physical tap test is the one that cannot be argued with, and it needs no analysis at all.

**Two wiring faults that read as success rather than failure:**

- **Both `SEL` pins at the same level.** Then both mics drive SDIN during the *same* slot — bus
  contention that corrupts `M0` too, while the other slot stays undriven. The tell is `M0`'s own
  numbers moving away from your -69.7 dBFS baseline. This is the best use of that baseline.
- **`rx_force_left=1`** (or any path that routes one slot into both FIFOs). `M0` and `M1` then look
  perfect: `gamma=1.000`, `lag=0.00`, `d=0.0dB`. Two different mics centimetres apart never agree that
  exactly — reverb and their own tolerances forbid it — so the pair line flags it as
  `<- IDENTICAL: one source in both slots?`.

Silence is *not* a valid aliveness test: an uncorrelated floor and a dead mic both give a low `gamma`.
Make a loud sound.

### What to get out of this step before going any further

1. `probe M1 (ch0 R): ALIVE` — the right slot delivers audio at all.
2. `pair M1-M0 d=` — the **inter-mic gain match**. This caps every null you will ever steer: 1 dB
   mismatch limits rejection to ~-18 dB, 0.5 dB to ~-25 dB. Fix it here (same part, same reel, same
   acoustic loading) or the 4-mic build inherits the ceiling.
3. `gamma` on a silent room — **is your noise floor correlated?** With a floor centred near 340 Hz and
   5 cm spacing, diffuse-field coherence `sin(kd)/(kd)` is 0.98, so expect `gamma` high and
   `d_nf ≈ 0` for `COMBINE=1`. If instead `d_nf ≈ -3 dB`, the floor is per-mic (self-noise or board
   pickup) and averaging is worth keeping. Decide which you expect *before* you run it.
4. `lag` vs the physical spacing (2.14 cm per sample) — validates the geometry math end to end.
5. `TRACK M1-M0 slope=` — the **instrument's own noise floor** for the drift test, since these two
   cannot actually drift. This is the number the cross-channel measurement gets judged against later.

With `MICB_MICS=2` each capture prints a record per mic, one for the combined signal `C`, a `pair`
line per mic pair, and the `ARRAY` line:

```
[mic-bench] run=3 pair M1-M0 d=0.4dB gamma=0.921 lag=1.83smp (3.9cm)
[mic-bench] run=3 ARRAY mode=1 lag_used=0 d_rms=-2.9dB d_nf=-3.0dB d_snr=2.8dB uncorrelated(at the ideal for this mic count)
```

- `pair` is the inter-mic geometry: level mismatch, normalised correlation peak `gamma` (coherence),
  and the peak lag with parabolic sub-sample interpolation, printed in samples **and centimetres**
  (1 sample = 2.14 cm of path difference at 16 kHz) so you can check it against where the mics
  physically are. `gamma` near 1 means the two mics hear the same thing; near 0 means they do not.
- **`ARRAY` is the only line that answers "did the second mic help?"** — and it is a difference taken
  *inside one capture*, so the room, the talker and the gains are identical on both sides of it. That
  is the whole reason `C` is built into its own buffer instead of folded over `L`.

Combine modes (`-DMICB_COMBINE=`):

| mode | what | expect |
|---|---|---|
| 0 | primary only, `C == M0` | **the A/B control.** Run it first: any non-zero `d_*` is a harness bug, not an array effect |
| 1 | average of all live mics | ideally `10*log10(N)` on *uncorrelated* noise (mic self-noise, ADC noise); **0 dB on correlated noise** — room tone, reverb, a competing talker |
| 2 | difference | gradient/cardioid-ish: rejects distant and diffuse sources, but high-passes speech and is very sensitive to gain mismatch (read `pair R-L` first) |
| 3 | aligned sum | mode 1 with the measured delay compensated; only does something with mics >~2 cm apart |

The verdict string at the end of the `ARRAY` line reads `d_nf` against those expectations, so a log can
be skimmed. `correlated(room/reverb: averaging cannot help)` is a legitimate and common outcome — most
real-room noise is correlated between two closely spaced mics, and no amount of averaging touches it.
Honest suggested procedure: mode 0 first (prove the harness), then mode 1 in a quiet room (should show
~-3 dB on `d_nf`; that is mic self-noise), then mode 1 with speech (watch `d_snr`), then mode 2 only if
1 disappoints.

## Calibrating a 4-element endfire array

Four elements need three constants before the beamformer means anything: a **gain** per element, the
**per-gap acoustic lag** (the array geometry, in samples), and the **channel offset** between the two
I2S streams. `-DMICB_CAL_REPORT=ON` measures all three and prints them back as flags to paste, with the
run-to-run spread next to each — a constant measured once is a guess, and the spread is what says
whether it deserves to be baked into a build.

Two of the three need a *particular* source position to be meaningful, and getting that wrong produces
plausible numbers rather than an error. Hence two passes over the same build.

### Pass 1 — gains, BROADSIDE source

```bash
make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-mic-bench EXTRA_CMAKE_ARGS="\
  -DMICB_MICS=4 -DMICB_MIC_CHANNEL=1 -DMICB_MIC_CHANNEL_B=0 \
  -DMICB_COMBINE=0 -DMICB_CAL_REPORT=ON -DMICB_ALIGN_MEASURE_ONLY=ON \
  -DMICB_EXPECTED_LAG=0.0 -DMICB_MAX_LAG=256 -DMICB_TONE_HZ=0"
```

Put a steady broadband source (pink noise from a phone works; so does talking continuously) about
50 cm away **on the perpendicular bisector of the array** — broadside, equidistant from all four. For a
6.45 cm aperture at 50 cm that is equidistant to 0.04 samples and 0.02 dB, so it is a good calibration
position even though it looks approximate.

`COMBINE=0` because `C == M0` there: the A/B control, with no beamforming to confuse the reading.
`ALIGN_MEASURE_ONLY=ON` because this pass is about what the offsets *are*.

Read `SETUP gain`. Broadside is also the one geometry where an integer slip cannot be resolved, and
that does not matter here — an RMS ratio does not care how the signal is shifted in time.

### Pass 2 — geometry, ON-AXIS source

Same binary. Move the source to ~50 cm directly **in front of M0, on the array axis**. Read
`SETUP geometry`: `M1 lag` is the per-gap acoustic lag, and it should come back near `+1.00` samples for
2.15 cm spacing — the line prints the bound the spacing permits so an impossible reading is obvious.

That number, not the nominal spacing, is what to use for `MICB_EXPECTED_LAG`: the correlation peak's
parabolic interpolation is slightly biased toward the peak sample, so the measured value is what the
alignment will actually be compared against.

`SETUP channel offset` is readable in either pass — its fractional part is immune to the integer slip,
so any coherent capture measures it. It is reported as an integer *range* plus a fraction, because the
integer part is genuinely not separable from the per-capture FIFO slip (see `MICB_CHAN_OFFSET`) and
printing one number for it would be a fabrication. Nothing needs it to be separated.

### Calibrating with an element missing or dead

Keep `MICB_MICS=4`. Do **not** drop to 3 to exclude a dead element: the read set is what holds the mics
in sample alignment (one FIFO read per slot per call, fixed order, decided once at boot), so changing it
changes the thing being calibrated and stops draining that slot's FIFO. The bench already handles a dead
slot correctly — level, not the boot probe, is the test, so it is excluded from `C`, from the `ARRAY`
verdict and from the `(ideal ...)` prediction, its pair records are suppressed as meaningless, and its
`ALIGN` line reads `NO SIGNAL` until a microphone appears on it.

Everything except that element's own gain is still measurable. In particular the channel offset needs
only **one** working cross-channel element, so M0/M1/M2 is enough to calibrate it — you just get one
estimate per capture instead of two, so run more captures. A 3-element endfire array with
`MICB_COMBINE=5` is also a legitimate measurement (a 4.3 cm aperture instead of 6.45 cm).

### Pass 3 — apply, and read the array verdict

```bash
make build ... EXTRA_CMAKE_ARGS="\
  -DMICB_MICS=4 -DMICB_MIC_CHANNEL=1 -DMICB_MIC_CHANNEL_B=0 \
  -DMICB_COMBINE=1 -DMICB_EXPECTED_LAG=<pass 2> -DMICB_MAX_LAG=256 \
  -DMICB_GAIN_DB_M1=<pass 1> -DMICB_GAIN_DB_M2=<..> -DMICB_GAIN_DB_M3=<..> \
  -DMICB_CAL_REPORT=ON -DMICB_TONE_HZ=0"
```

`ALIGN_MEASURE_ONLY` is now off, so every element is aligned against M0 before anything is measured.
Back to broadside, and read `ARRAY`: `d_nf` should approach **-6 dB** (`10*log10(4)`) on uncorrelated
noise. This is the number that was **+3.9 dB** — averaging making the floor *worse* — before alignment
existed, and it is the check that the alignment is real.

If `d_nf` is much worse than the printed `(ideal ...)`, the gains are the first suspect: read the
`LEVEL` lines, which print the residual mismatch after correction and the null depth it permits.

### Pass 4 — the endfire beam

```bash
make build ... EXTRA_CMAKE_ARGS="... -DMICB_COMBINE=5 -DMICB_ENDFIRE_DELAY=1 -DMICB_EXPECTED_LAG=1.0"
```

Read `ARRAY d_rms` with the source in **front** of M0, then again with it **behind** M3. The difference
is the front/back ratio, i.e. the beam. **Set `-DMICB_EXPECTED_LAG=-1.0` for the back run** — the
acoustic lags reverse sign when the source moves behind the array, and an expected lag with the wrong
sign makes the alignment fight the geometry. The beamformer's own delays do *not* change; that is the
point of measuring the back.

### What the alignment does and does not remove

Element `m`'s buffer holds the same sound as M0's, delayed by

```
L(m) = acoustic(m) + channel_offset(m) + slip(m)
```

Only the first term is wanted — it *is* the beam. `acoustic(m) = m * MICB_EXPECTED_LAG` exactly, for a
uniform linear array in a plane wave, so one number describes the whole array. The other two are the
instrument:

- **`slip(m)`** — an integer, random per capture, from the I2S RX FIFOs (see the bug-log entry in
  `/CLAUDE.md`). Removed every run.
- **`channel_offset`** — constant per boot, **non-integer** (~11.6 samples measured), and only on
  elements sitting on the other I2S channel. Its integer part is removed per run along with the slip;
  its fraction is corrected from a session mean (`MICB_ALIGN_FRAC`), not per run, because a per-run
  fractional correction would make the array track the talker — and a beamformer whose look direction
  follows the source is not a beamformer.

The fraction is worth correcting, and the host test measures why: on a full-band signal, leaving a
0.4-sample misalignment in place costs ~8% of inter-element coherence, while the two interpolations
that remove it cost ~2%. Roughly 4:1 in favour of correcting, and better than that on speech, which has
little energy near Nyquist. `-DMICB_ALIGN_FRAC=OFF` measures what it is worth on your board.

**A lag that comes back exactly at `±MICB_MAX_LAG` is not a measurement, it is a saturated search.**
The `ALIGN` line refuses to shift on one and says so. This cost a day once: at the default `MAX_LAG=4`
every cross-channel pair pinned at ±4, and the low coherence that comes with a wrong alignment was read
as "the two I2S channels do not share a time base". They do. Keep `MAX_LAG` well above the channel
offset (256 is cheap — one static 4 KB array).

## Why the metrics are host-tested

`src/mic_quality.c` is chip-free (no HAL, no MMIO, no printf, no config macros), so
`test/host_mic_quality_test.c` compiles **the exact source that runs on silicon** and checks it against
synthetic signals whose true SNR, tilt, distortion and inter-mic delay are known by construction:

```bash
cd c2c-demos/dsp-mic-bench/test
gcc -O2 -Wall -Wextra -I../include -o host_mic_quality_test \
    host_mic_quality_test.c ../src/mic_quality.c -lm && ./host_mic_quality_test
```

This is not ceremony. On silicon there is nothing to compare these numbers against, and a broken
capture path and a broken metric both produce a plausible number — so at least one side has to be
verified where it *can* be. Two things it caught while being written, neither of which inspection
would have:

1. **A tone fit at the *nominal* frequency is worthless.** A coherent fit needs the frequency to be
   right to about `1/(2T)`; a 1 Hz error over half a second slips two whole cycles, dumps the
   fundamental into the residual, and reports a clean mic as catastrophically distorted. Neither a
   phone's tone generator nor this chip's I2S clock is that accurate. So `mq_tone_refine_f0` *measures*
   f0 — a coarse grid, then a **phase-difference** step between the two halves of the window, which is
   both far more accurate and much cheaper than continuing to search. The test asserts recovery of a
   997.3 Hz tone from a 1000 Hz guess to within 0.01 Hz, and that the naive fit really is >10 dB worse.
2. **Parseval subtraction cannot resolve a -40 dB residual.** With the near-orthogonal cos/sin basis,
   a 0.4% amplitude error is a 0.8% energy error, i.e. -21 dB of *fabricated* noise floor. The fit now
   solves the exact 2x2 normal equations and takes the residual from the least-squares identity, which
   measures a 60 dB SINAD correctly — the range a MEMS mic actually deserves to be judged over.

On chip that translation unit is compiled **without RVV** (`-march=rv64imafd`): the RVV float kernels
in this tree have already produced confident garbage on dsp25 once (the TinySpeech entry in
`/CLAUDE.md` — every input gave the same ranking at 10-100x the right magnitude), and these numbers are
the demo's entire output. Scalar costs a few milliseconds against a 2-second capture.

## Knobs

CMake cache variables, all prefixed `MICB_` (never reuse a sibling demo's option name — CMake cache
entries are global and the first declaration wins, which is how `dsp-citrinet-c2c` once shipped a
scalar encoder):

| variable | default | note |
|---|---|---|
| `MICB_MICS` | `1` | 1, 2 (one channel L/R) or 4 (two channels L/R) |
| `MICB_COMBINE` | `1` | 0=M0-only 1=average 2=difference(M0-M1) 3=aligned-sum 4=endfire cardioid 5=endfire delay-and-sum |
| `MICB_MIC_CHANNEL` / `_B` | `0` / `1` | which I2S channel carries M0/M1 and which carries M2/M3 |
| `MICB_SLIP_FIX` | `ON` | master switch for the inter-element alignment pass |
| `MICB_CAL_REPORT` | `OFF` | print the `SETUP` calibration block (needs `MICB_SLIP_FIX=ON`) |
| `MICB_ALIGN_MEASURE_ONLY` | `OFF` | measure and report the alignment but shift nothing |
| `MICB_ALIGN_FRAC` | `ON` | interpolate out the fractional cross-channel stream offset |
| `MICB_EXPECTED_LAG` | `0.0` | **per-gap** acoustic lag in samples; every element is `m` times this |
| `MICB_ENDFIRE_DELAY` | `1` | inter-element steering delay in samples (modes 4/5) |
| `MICB_GAIN_DB_M1..M3` | `0.0` | per-element gain correction, applied to `C` only |
| `MICB_MAX_LAG` | `4` | lag search half-range; **raise well above the channel offset for 4 mics** |
| `MICB_CHAN_OFFSET` | `0.0` | measured channel stream offset — informational, cancels out of the correction |
| `MICB_SPACING_CM` | `2.15` | element spacing, used to print the physical bound on a lag |
| `MICB_TRACK_BLOCKS` | `8` | sub-blocks for the lag-drift/clocking test; 0 = off |
| `MICB_WARMUP_MS` | `1000` | one-time boot discard, so run 1 isn't the mic's HPF settling |
| `MICB_CAPTURE_MS` | `2000` | fixed window per run |
| `MICB_TONE_HZ` | `1000` | whole Hz; 0 disables the tone analysis |
| `MICB_SUMMARY_EVERY` | `8` | rolling summary period; 0 = off |
| `MICB_VAD` | `OFF` | gate windows on speech |
| `MICB_TARGET_FREQ_HZ` | `750000000` | PLL target |

Everything else is an `#ifndef`-guarded `#define` in `include/mic_bench_config.h` (frame length,
percentiles, clip level, discard length, lag range, tone search span) and can be overridden per build
with `-DMICB_x=...`.

Sizes: ~60 KB text; `.bss` is 130 KB at 1 mic, 386 KB at 2, 642 KB at 4 (one capture buffer per mic
plus the combined one). `.bss` sits in the LOAD segment, so `uart_tsi` zero-fills it over the serial
link on every flash — roughly 1.5 s / 4.5 s / 7 s at 921600 baud.
