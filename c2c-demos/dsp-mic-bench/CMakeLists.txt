# dsp-mic-bench: objective audio-quality benchmark for the DSP 25 I2S microphone path.
#
#   make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-mic-bench
#   make tsi-run TTY=<tty> BINARY=build/c2c-demos/dsp-mic-bench/dsp-mic-bench.elf
#
# One mic by default. Add the second one later WITHOUT touching anything else:
#   make build CHIP=dsp25 PLATFORM=CHIP TARGET=dsp-mic-bench \
#       EXTRA_CMAKE_ARGS="-DMICB_STEREO=ON -DMICB_COMBINE=1"
#
# dsp25 only (I2S is a DSP peripheral) and real silicon only (Spike models no I2S). No C2C link, no
# model, no accelerator, stock linker script — so there is nothing in the failure surface except the
# microphone path this is meant to measure.
#
# EVERY CACHE VARIABLE HERE IS PREFIXED `MICB_`. CMake option()/set(CACHE) entries are GLOBAL and a
# second declaration of an existing name is a NO-OP, so reusing a sibling demo's name (say
# DSP_CITRINET_MIC_STEREO) would silently inherit THAT demo's default while this build's status line
# printed the value we intended. That mistake shipped dsp-citrinet-c2c with a scalar encoder; see
# /CLAUDE.md. The config header's #ifndef guards use the same MICB_ names, so a knob can also be
# overridden with -DMICB_x=... at the compiler level for a one-off experiment.

add_executable(dsp-mic-bench
  src/main.c
  src/mic_quality.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c   # init_test(): PLL + UART bring-up
)

target_include_directories(dsp-mic-bench PUBLIC include)
target_include_directories(dsp-mic-bench PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

# ---------------------------------------------------------------------------------------------------
# The second microphone. OFF by default: this demo's first job is to characterise ONE mic.
# ---------------------------------------------------------------------------------------------------
# MICB_MICS: 1 = M0 only; 2 = M0/M1 on channel A's L/R slots (one BCLK/WS, sample-locked by
# construction); 4 = adds channel B's two slots, whose coherence with channel A is measured, not
# assumed. Bring them up in that order — the 2-mic step calibrates what a good gamma and a stable lag
# look like on this rig, and without that reference a cross-channel number cannot be judged.
set(MICB_MICS 1 CACHE STRING "Microphones: 1-2 = channel A L/R; 3 = + channel B left; 4 = + channel B right")
if(NOT MICB_MICS MATCHES "^(1|2|3|4)$")
  message(FATAL_ERROR "dsp-mic-bench: MICB_MICS must be 1, 2, 3 or 4 (got '${MICB_MICS}')")
endif()
# MICB_STEREO was the old name for MICB_MICS=2. Mapped rather than ignored: CMake cache entries are
# global and silently doing nothing with a flag someone deliberately passed is how a build ends up
# measuring something other than what its log claims.
if(DEFINED MICB_STEREO AND MICB_STEREO AND MICB_MICS EQUAL 1)
  set(MICB_MICS 2 CACHE STRING "" FORCE)
  message(WARNING "dsp-mic-bench: MICB_STEREO is deprecated; using MICB_MICS=2 instead")
endif()

# 0 = primary only (the A/B control: C == M0), 1 = average of all live mics, 2 = M0-M1 difference,
# 3 = lag-aligned M0+M1 sum.
set(MICB_COMBINE 1 CACHE STRING
    "Combine: 0=M0-only 1=average 2=difference(M0-M1) 3=aligned-sum 4=endfire-cardioid 5=endfire-delay-and-sum")
if(NOT MICB_COMBINE MATCHES "^(0|1|2|3|4|5)$")
  message(FATAL_ERROR "dsp-mic-bench: MICB_COMBINE must be 0..5 (got '${MICB_COMBINE}')")
endif()
# Endfire steering delay per element, in samples. 1 sample = 2.15 cm at 16 kHz, which is also the
# lambda/2 aliasing limit for the full 8 kHz band — see the knob's comment in mic_bench_config.h.
# WHICH I2S CHANNEL IS "A". The slot map is M0=A-left, M1=A-right, M2=B-left, M3=B-right, and those
# labels are the ARRAY ORDER: M0 is the front element in the endfire modes, and every delta is measured
# against M0. So if the physical array runs channel 1 first, swap these rather than re-labelling
# anything downstream — otherwise the endfire delays are applied to the mics in the wrong order and the
# beam points somewhere the geometry does not.
set(MICB_MIC_CHANNEL   0 CACHE STRING "I2S channel carrying M0 (left slot) and M1 (right slot)")
set(MICB_MIC_CHANNEL_B 1 CACHE STRING "I2S channel carrying M2 (left slot) and M3 (right slot)")
if(MICB_MIC_CHANNEL STREQUAL MICB_MIC_CHANNEL_B)
  message(FATAL_ERROR "dsp-mic-bench: MICB_MIC_CHANNEL and MICB_MIC_CHANNEL_B must differ")
endif()
message(STATUS "dsp-mic-bench: M0/M1 on I2S ch${MICB_MIC_CHANNEL}, M2/M3 on ch${MICB_MIC_CHANNEL_B}")

# Integer FIFO slip correction. Only knows about the M0/M1 pair today, so an M1-only correction across
# a four-mic array is half a fix; leave it OFF until it is generalised per element.
option(MICB_SLIP_FIX "Correct the integer inter-mic FIFO slip from MICB_EXPECTED_LAG" ON)
set(_MICB_SLIP_FIX 0)
if(MICB_SLIP_FIX)
  set(_MICB_SLIP_FIX 1)
endif()

set(MICB_ENDFIRE_DELAY 1 CACHE STRING "Endfire inter-element delay in samples (modes 4/5)")
# The inter-mic lag your geometry should produce, in samples: +1 = source on axis in front of M0,
# -1 = source behind the rear mic, 0 = equidistant (the one geometry a slip CANNOT be resolved in).
# The FIFOs deliver a random integer offset that rx_en cycling does not clear, so this operator-
# supplied number is what disambiguates measured lag = true + slip. See mic_bench_config.h.
set(MICB_EXPECTED_LAG 0.0 CACHE STRING "Inter-mic lag the geometry should give, in samples")
# Per-mic gain correction in dB, applied to C only. Read them off the CAL line with every mic
# equidistant from one steady source. Mismatch is the hard ceiling on null depth: 5 dB caps it at -7.
set(MICB_GAIN_DB_M1 0.0 CACHE STRING "Gain correction for M1 in dB (C only)")
set(MICB_GAIN_DB_M2 0.0 CACHE STRING "Gain correction for M2 in dB (C only)")
set(MICB_GAIN_DB_M3 0.0 CACHE STRING "Gain correction for M3 in dB (C only)")
set(MICB_TRACK_BLOCKS 8 CACHE STRING "Sub-blocks per capture for the lag-drift (clocking) test; 0=off")
# Lag search half-range in samples. 4 covers any acoustic path difference this array can produce
# (1 sample = 2.15 cm), which is the right setting for measuring geometry — but it is NOT enough to
# FIND a stream-alignment offset: on silicon every cross-channel lag pinned at exactly the limit,
# which is what a correlator reports when the true offset is outside its window. Widen it (64, 256)
# to measure that offset, then put it back. Clamped to MQ_LAG_LIMIT in mic_quality.c.
set(MICB_MAX_LAG 4 CACHE STRING "Lag search half-range in samples (widen to hunt a stream offset)")

# --- 4-element array calibration -----------------------------------------------------------------
# MICB_CHAN_OFFSET is DOCUMENTATION, not a tuning knob: it cancels out of the correction that
# MICB_SLIP_FIX applies (the integer part of the channel offset is inseparable from the per-capture
# FIFO slip, and removing their sum is both possible and sufficient). Record what a board measured.
set(MICB_SPACING_CM 2.15 CACHE STRING "Inter-element spacing in cm, along the array axis")
set(MICB_CHAN_OFFSET 0.0 CACHE STRING "Measured ch-B stream offset in samples (informational)")
option(MICB_ALIGN_FRAC "Interpolate out the FRACTIONAL cross-channel stream offset" ON)
set(_MICB_ALIGN_FRAC 0)
if(MICB_ALIGN_FRAC)
  set(_MICB_ALIGN_FRAC 1)
endif()
option(MICB_ALIGN_MEASURE_ONLY "Measure and report the alignment but shift nothing" OFF)
set(_MICB_ALIGN_MEASURE_ONLY 0)
if(MICB_ALIGN_MEASURE_ONLY)
  set(_MICB_ALIGN_MEASURE_ONLY 1)
endif()
option(MICB_ALIGN_SWEEP "Measure the source direction each capture (for beam-pattern sweeps)" OFF)
set(_MICB_ALIGN_SWEEP 0)
if(MICB_ALIGN_SWEEP)
  set(_MICB_ALIGN_SWEEP 1)
endif()
set(MICB_SWEEP_LEVEL_DEADBAND 0.4 CACHE STRING "dB below which the direction is called unknown")
if(MICB_ALIGN_SWEEP AND NOT MICB_SLIP_FIX)
  message(FATAL_ERROR "MICB_ALIGN_SWEEP needs MICB_SLIP_FIX=ON: it works by replacing the alignment's "
                      "assumed geometry with a measured one.")
endif()
option(MICB_CAL_REPORT "Print the per-element SETUP calibration block" OFF)
set(_MICB_CAL_REPORT 0)
if(MICB_CAL_REPORT)
  set(_MICB_CAL_REPORT 1)
endif()
if(MICB_CAL_REPORT AND NOT MICB_SLIP_FIX)
  message(FATAL_ERROR "MICB_CAL_REPORT needs MICB_SLIP_FIX=ON: the calibration constants come from "
                      "the alignment pass. Use -DMICB_ALIGN_MEASURE_ONLY=ON to measure without "
                      "correcting.")
endif()
if(MICB_MICS EQUAL 1)
  message(STATUS "dsp-mic-bench: single microphone. Next: -DMICB_MICS=2, then -DMICB_MICS=4")
elseif(MICB_MICS EQUAL 2)
  message(STATUS "dsp-mic-bench: MICB_MICS=2 on ONE channel (combine=${MICB_COMBINE}) — "
                 "M0, M1 and C from one capture; read the ARRAY and TRACK lines")
else()
  message(STATUS "dsp-mic-bench: MICB_MICS=${MICB_MICS} across TWO channels (combine=${MICB_COMBINE}) — "
                 "the cross-channel TRACK line is the clocking verdict")
endif()

# ---------------------------------------------------------------------------------------------------
# Measurement geometry. Defaults are documented in include/mic_bench_config.h; these are here because
# they are the ones worth sweeping from the command line.
# ---------------------------------------------------------------------------------------------------
set(MICB_TARGET_FREQ_HZ 750000000 CACHE STRING "PLL target frequency in Hz")
set(MICB_CAPTURE_MS 2000 CACHE STRING "Analysed window per run, in ms (fixed length: comparability)")
set(MICB_WARMUP_MS 1000 CACHE STRING "One-time boot discard in ms, so run 1 is not the mic HPF settling")
set(MICB_TONE_HZ 1000 CACHE STRING "Tone-fit fundamental in whole Hz (0 disables the tone analysis)")
set(MICB_SUMMARY_EVERY 8 CACHE STRING "Print a rolling min/mean/max summary every N runs (0 = off)")
# Liveness only: per-slot level + alive/constant/no-data verdict, nothing else. Use this to bring up a
# newly wired mic before asking any question about the array.
option(MICB_PROBE_ONLY "Report only whether each slot carries a microphone, then stop" OFF)
set(_MICB_PROBE_ONLY 0)
if(MICB_PROBE_ONLY)
  set(_MICB_PROBE_ONLY 1)
  message(STATUS "dsp-mic-bench: MICB_PROBE_ONLY=1 — liveness only, no pair/array/tone records")
endif()

option(MICB_VAD "Wait for speech before each window instead of capturing unconditionally" OFF)
set(_MICB_VAD 0)
if(MICB_VAD)
  set(_MICB_VAD 1)
  message(STATUS "dsp-mic-bench: MICB_VAD=1 — windows are speech-gated, so the noise-floor numbers "
                 "are NOT comparable with an ungated build")
endif()

target_compile_definitions(dsp-mic-bench PRIVATE
  MICB_MICS=${MICB_MICS}
  MICB_COMBINE=${MICB_COMBINE}
  MICB_MIC_CHANNEL=${MICB_MIC_CHANNEL}
  MICB_MIC_CHANNEL_B=${MICB_MIC_CHANNEL_B}
  MICB_SLIP_FIX=${_MICB_SLIP_FIX}
  MICB_ENDFIRE_DELAY=${MICB_ENDFIRE_DELAY}
  MICB_EXPECTED_LAG=${MICB_EXPECTED_LAG}f
  MICB_GAIN_DB_M1=${MICB_GAIN_DB_M1}f
  MICB_GAIN_DB_M2=${MICB_GAIN_DB_M2}f
  MICB_GAIN_DB_M3=${MICB_GAIN_DB_M3}f
  MICB_TRACK_BLOCKS=${MICB_TRACK_BLOCKS}u
  MICB_MAX_LAG=${MICB_MAX_LAG}
  MICB_SPACING_CM=${MICB_SPACING_CM}f
  MICB_CHAN_OFFSET=${MICB_CHAN_OFFSET}f
  MICB_ALIGN_FRAC=${_MICB_ALIGN_FRAC}
  MICB_ALIGN_MEASURE_ONLY=${_MICB_ALIGN_MEASURE_ONLY}
  MICB_CAL_REPORT=${_MICB_CAL_REPORT}
  MICB_ALIGN_SWEEP=${_MICB_ALIGN_SWEEP}
  MICB_SWEEP_LEVEL_DEADBAND=${MICB_SWEEP_LEVEL_DEADBAND}f
  MICB_TARGET_FREQ_HZ=${MICB_TARGET_FREQ_HZ}ULL
  MICB_CAPTURE_MS=${MICB_CAPTURE_MS}u
  MICB_WARMUP_MS=${MICB_WARMUP_MS}u
  MICB_TONE_HZ=${MICB_TONE_HZ}
  MICB_SUMMARY_EVERY=${MICB_SUMMARY_EVERY}u
  MICB_VAD_ENABLE=${_MICB_VAD}
  MICB_PROBE_ONLY=${_MICB_PROBE_ONLY}
)
message(STATUS "dsp-mic-bench: PLL ${MICB_TARGET_FREQ_HZ} Hz, ${MICB_CAPTURE_MS} ms window, "
               "tone ${MICB_TONE_HZ} Hz")

# The global add_compile_options(-O1) in the top-level CMakeLists lands after Release's -O3 and the
# last -O wins, so a target-level override is needed. The capture loop is I/O-bound on the mic clock
# either way; this is for the analysis passes (the tone fit is the only expensive one).
target_compile_options(dsp-mic-bench PRIVATE -O2)

# COMPILE THE METRICS WITHOUT RVV. The RVV float kernels in this tree have already produced confident
# garbage on dsp25 once — the TinySpeech entry in /CLAUDE.md, where every input produced the same
# ranking at 10-100x the right magnitude — and these numbers ARE this demo's entire output, with
# nothing on-chip to check them against. Undefining __riscv_vector for this one translation unit
# selects plain scalar C; the ABI is unchanged so it links normally, and the cost is a few
# milliseconds against a 2-second capture. test/host_mic_quality_test.c checks the same source on the
# host, which is the other half of trusting it.
set_source_files_properties(src/mic_quality.c
  TARGET_DIRECTORY dsp-mic-bench
  PROPERTIES COMPILE_OPTIONS "-march=rv64imafd")

# A stack overflow here does not trap: it silently corrupts automatics or hangs the core (/CLAUDE.md).
# The capture buffers are static for that reason; the largest frames left are the correlation and
# frequency-search arrays at ~1 KB each.
target_compile_options(dsp-mic-bench PRIVATE -Wstack-usage=4096)

# Stock dsp25.ld (DRAM declared at its real 256 MB, stack derived from inside real memory — unlike the
# `llm` variant, see /CLAUDE.md), with a small heap because nothing here allocates: the stock 1 MB
# heap sits inside the single LOAD segment, so uart_tsi zero-fills all of it over the serial link on
# every flash. 16 KiB of stack per hart is several times the largest frame above.
target_link_options(dsp-mic-bench PRIVATE
  -Wl,--defsym=__heap_size=0x10000
  -Wl,--defsym=__stack_size=0x4000
  -Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/dsp-mic-bench.map
)

# glossy (whole-archive) provides the dsp25 HAL including the I2S driver; m provides sin/cos/log10.
target_link_libraries(dsp-mic-bench PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if(PROF_COV)
  target_link_libraries(dsp-mic-bench PRIVATE gcov)
endif()
