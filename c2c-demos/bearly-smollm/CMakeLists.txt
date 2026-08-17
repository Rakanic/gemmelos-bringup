# bearly-smollm: SmolLM2-135M-Instruct (int8 Q8_0) on Bearly ML 25. Standalone — no C2C link.
#
#   .venv/bin/python c2c-demos/bearly-smollm/scripts/export_smollm.py     # once: writes model/*.bin
#   make build CHIP=bearly25 PLATFORM=CHIP TARGET=bearly-smollm EXTRA_CMAKE_ARGS="-DLINKER=llm"
#   make tsi-run TTY=<tty> BINARY=build/c2c-demos/bearly-smollm/bearly-smollm.elf   # ~26 min load
#
# LINKER=llm is REQUIRED and the top-level Makefile does not forward LINKER=, so it has to go inside
# EXTRA_CMAKE_ARGS. It selects platform/bearly25/bearly25-llm.ld, which (a) declares the heap and
# stack as bare symbols instead of NOLOAD sections — otherwise uart_tsi zero-fills the whole ~100 MB
# heap over the serial link on every load — and (b) keeps DRAM at its real 256 MB so the stack
# lands inside memory that exists. See that file's header and the LINKER=llm entry in CLAUDE.md.

# Blobs baked into the ELF by src/blob.S. Override with -DSMOLLM_MODEL_BIN=/abs/path.bin.
set(SMOLLM_MODEL_BIN     "${CMAKE_CURRENT_SOURCE_DIR}/model/model_q80.bin"
    CACHE STRING "Grouped-Q8_0 SmolLM model blob to embed")
set(SMOLLM_TOKENIZER_BIN "${CMAKE_CURRENT_SOURCE_DIR}/model/tokenizer.bin"
    CACHE STRING "Byte-level BPE tokenizer blob to embed")
foreach(_blob "${SMOLLM_MODEL_BIN}" "${SMOLLM_TOKENIZER_BIN}")
  if(NOT EXISTS "${_blob}")
    message(WARNING "bearly-smollm: ${_blob} is missing — run scripts/export_smollm.py first, "
                    "or the .incbin will fail the build.")
  endif()
endforeach()

add_executable(bearly-smollm
  src/main.c
  src/model.c
  src/tokenizer.c
  src/dualcore.c
  src/uncore_probe.c
  src/blob.S
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c   # init_test(): portable PLL + UART bring-up
)

target_include_directories(bearly-smollm PRIVATE
  include
  ${CMAKE_SOURCE_DIR}/bmark-lib
  ${CMAKE_SOURCE_DIR}/driver/rocket-chip-blocks/uart
  ${CMAKE_CURRENT_SOURCE_DIR}/../common        # mem_probe.h, shared with the membw target
)

target_compile_definitions(bearly-smollm PRIVATE
  SMOLLM_MODEL_BIN=${SMOLLM_MODEL_BIN}          # bare paths; blob.S stringizes them for .incbin
  SMOLLM_TOKENIZER_BIN=${SMOLLM_TOKENIZER_BIN}
)

# OBJECT_DEPENDS is REQUIRED, not a nicety: the compiler emits no depfile for `.incbin`, so a
# re-exported model would NOT rebuild this object and the ELF would keep the OLD weights — which
# looks exactly like a model bug. (Learned in dsp-citrinet; verified there by touching the .bin.)
set_source_files_properties(src/blob.S PROPERTIES
  OBJECT_DEPENDS "${SMOLLM_MODEL_BIN};${SMOLLM_TOKENIZER_BIN}")

# Optimization: the top-level CMakeLists does a global add_compile_options(-O1) that lands AFTER
# Release's -O3, and the last -O wins — so without a target-level override the matmul that is 99%
# of the runtime would build at -O1. Target options come after directory options.
set(SMOLLM_OPT "-O3" CACHE STRING "Optimization level (overrides the global -O1)")
target_compile_options(bearly-smollm PRIVATE ${SMOLLM_OPT})

# Vector int8 dot product in matmul(). Bit-identical to the scalar path (int32 accumulation of
# int8 products either way), so the host reference stays an exact oracle.
option(SMOLLM_USE_RVV "Use the RVV grouped-int8 dot product" ON)
set(_SM_RVV 0)
if(SMOLLM_USE_RVV)
  set(_SM_RVV 1)
endif()
target_compile_definitions(bearly-smollm PRIVATE SMOLLM_USE_RVV=${_SM_RVV})

# PLL target. init_test() programs the PLL to (freq / SYS_CLK_FREQ) x 50 MHz and derives the UART
# divisor from it, so this single value drives the clock, the console and the tok/s report.
set(SMOLLM_TARGET_FREQ_HZ 750000000 CACHE STRING "PLL target frequency in Hz")
target_compile_definitions(bearly-smollm PRIVATE
  SMOLLM_TARGET_FREQUENCY_HZ=${SMOLLM_TARGET_FREQ_HZ}ULL)

# Sampling. 0 = greedy, which is deterministic and exactly what scripts/ref_runq.py reproduces —
# keep it there while validating, raise it (e.g. 0.8) for livelier demo output.
set(SMOLLM_TEMPERATURE "0.0f" CACHE STRING "Sampling temperature (0 = greedy)")
set(SMOLLM_TOPK 40 CACHE STRING "Sample from the k most likely tokens when temperature > 0")
set(SMOLLM_MAX_NEW_TOKENS 200 CACHE STRING "Max tokens generated per answer")
target_compile_definitions(bearly-smollm PRIVATE
  SMOLLM_TEMPERATURE=${SMOLLM_TEMPERATURE}
  SMOLLM_TOPK=${SMOLLM_TOPK}
  SMOLLM_MAX_NEW_TOKENS=${SMOLLM_MAX_NEW_TOKENS})

# Non-interactive prompt: generate once and halt instead of reading the UART. Required for Spike,
# which models no UART input. e.g. EXTRA_CMAKE_ARGS="-DSMOLLM_PROMPT=What is a transistor?"
set(SMOLLM_PROMPT "" CACHE STRING "Compile-time prompt; empty = interactive UART loop")
if(NOT SMOLLM_PROMPT STREQUAL "")
  target_compile_definitions(bearly-smollm PRIVATE SMOLLM_PROMPT=\"${SMOLLM_PROMPT}\")
  message(STATUS "bearly-smollm: autorun prompt \"${SMOLLM_PROMPT}\"")
endif()

# Compare per-stage residual sums against the host reference to localize a diverging layer.
# Regenerate include/smollm_golden.h with scripts/ref_runq.py --golden whenever the blob changes.
option(SMOLLM_DEBUG_GOLDEN "Check per-layer sums against include/smollm_golden.h" OFF)
set(_SM_GOLD 0)
if(SMOLLM_DEBUG_GOLDEN)
  set(_SM_GOLD 1)
  message(STATUS "bearly-smollm: SMOLLM_DEBUG_GOLDEN=1")
endif()
target_compile_definitions(bearly-smollm PRIVATE SMOLLM_DEBUG_GOLDEN=${_SM_GOLD})

# Diagnostics for an on-silicon divergence. Both cost one boot-time pass and nothing afterwards.
#   CHECK_BLOB    — FNV-1a the .incbin'd model and compare against the value the exporter recorded.
#                   Separates "the 143 MB did not survive the UART load / DRAM" from "the maths is
#                   wrong", which look identical on the console. Also times a full sequential read.
#   BENCH_KERNELS — time the memory floor and every matmul variant on a real weight tensor, and
#                   cross-check them numerically. Answers "memory system or kernel?" in one flash.
option(SMOLLM_CHECK_BLOB "Verify the model blob checksum at boot" OFF)
option(SMOLLM_BENCH_KERNELS "Benchmark matmul variants at boot" OFF)
set(_SM_CHK 0)
set(_SM_BENCH 0)
if(SMOLLM_CHECK_BLOB)
  set(_SM_CHK 1)
  message(STATUS "bearly-smollm: SMOLLM_CHECK_BLOB=1")
endif()
if(SMOLLM_BENCH_KERNELS)
  set(_SM_BENCH 1)
  message(STATUS "bearly-smollm: SMOLLM_BENCH_KERNELS=1")
endif()
target_compile_definitions(bearly-smollm PRIVATE
  SMOLLM_CHECK_BLOB=${_SM_CHK}
  SMOLLM_BENCH_KERNELS=${_SM_BENCH})

# Boot-time memory-system probe (c2c-demos/common/mem_probe.h — the same code the standalone `membw`
# target sweeps with). Prints the MEASURED core clock, dependent-load latency in L1 and in DRAM, and
# streaming read bandwidth over both the heap and the model blob, in cyc/byte and real MB/s.
#
# ON by default: decode here is 100% weight-bandwidth bound, so these are the numbers that explain
# tok/s, and ~2 s of boot against a ~40 s flash and 10 s/token is free. Turn it off for a demo
# console (SMOLLM_QUIET already silences it).
option(SMOLLM_MEM_PROBE "Probe memory latency/bandwidth at boot" ON)
set(SMOLLM_MEM_PROBE_BYTES 16777216 CACHE STRING "Heap scratch for the probe (>> last-level cache)")
set(_SM_MEM 0)
if(SMOLLM_MEM_PROBE)
  set(_SM_MEM 1)
endif()
target_compile_definitions(bearly-smollm PRIVATE
  SMOLLM_MEM_PROBE=${_SM_MEM}
  SMOLLM_MEM_PROBE_BYTES=${SMOLLM_MEM_PROBE_BYTES}u)

# Which RVV matmul the forward pass uses.
#   1 = lane-float: products stay in lanes, one float reduction per output ROW.  DEFAULT.
#   0 = row-dot:    one integer widening reduction (vwredsum) per group of 64 weights.
# Row-dot is the obvious kernel and it is BROKEN ON THIS SILICON: the on-chip benchmark measured
# max_diff = INT32_MAX against the scalar reference, which is what produced fluent-looking garbage
# ("atimstakingstaking elephstakingarxiv") on the first working flash. Lane-float measured
# max_diff 0. Do not switch this to 0 without re-running SMOLLM_BENCH_KERNELS.
set(SMOLLM_MATMUL_LANEFLOAT 1 CACHE STRING "1 = lane-float matmul, 0 = row-dot (BROKEN on bearly25)")
target_compile_definitions(bearly-smollm PRIVATE SMOLLM_MATMUL_LANEFLOAT=${SMOLLM_MATMUL_LANEFLOAT})

# Split batched-prefill matmuls across both harts. OFF by default and PROVES ITSELF AT BOOT before
# it is used: dsp-moonshine measured hart 1 reading stale data through a correct release/acquire
# pair on dsp25 (AMOs reliable, plain cached lines not — see .claude/plans/009-silicon-rtl-bug-list.md
# #2), and bearly25 has never been checked. src/dualcore.c probes liveness, job delivery and result
# visibility over a 256 KB payload twice with different seeds; on any failure the hooks stay NULL
# and the model runs single-core. Only prefill is split — decode is one token at a time and entirely
# memory-bound, so a second hart has nothing to add there.
option(SMOLLM_DUALCORE "Split batched prefill across both harts (probed at boot)" OFF)
set(_SM_DC 0)
if(SMOLLM_DUALCORE)
  set(_SM_DC 1)
  message(STATUS "bearly-smollm: SMOLLM_DUALCORE=1 (probed at boot, falls back to single-core)")
endif()
# Also split DECODE across both harts. Separate from SMOLLM_DUALCORE because decode is purely
# memory-bound: prefill measured ~0% from the second hart (12,063 -> 12,069 Mcyc), so this is
# expected to buy nothing unless two harts can each hold a line-fill in flight. Measure it; the
# per-answer "hart1 completed N row-ranges" line proves the split actually engaged.
option(SMOLLM_DUALCORE_DECODE "Split decode matmuls across both harts too" OFF)
set(_SM_DCD 0)
if(SMOLLM_DUALCORE_DECODE)
  set(_SM_DCD 1)
  message(STATUS "bearly-smollm: SMOLLM_DUALCORE_DECODE=1")
endif()
target_compile_definitions(bearly-smollm PRIVATE SMOLLM_DUALCORE=${_SM_DC}
                                                 SMOLLM_DUALCORE_DECODE=${_SM_DCD})

# Probe the memory system's CONFIGURATION registers — the part of "it's fixed silicon" that is not
# actually fixed. Reads the ccache Config/WayEnable (a WayEnable still at its reset value means most
# of the L2 is dark and one store fixes it) and optionally SiFive's feature-disable CSR. Each access
# is announced before it happens because a bad access HANGS this chip silently instead of trapping,
# so a hang names its own cause.
option(SMOLLM_PROBE_UNCORE "Print cache-controller / core config registers at boot" OFF)
option(SMOLLM_PROBE_FEATURE_CSR "Also read CSR 0x7c1 (traps if unimplemented)" OFF)
option(SMOLLM_PROBE_CCACHE_ENABLE_WAYS "If ways are disabled, enable them (one-way door)" OFF)
set(_SM_PU 0)
set(_SM_PF 0)
set(_SM_PW 0)
if(SMOLLM_PROBE_UNCORE)
  set(_SM_PU 1)
endif()
if(SMOLLM_PROBE_FEATURE_CSR)
  set(_SM_PF 1)
endif()
if(SMOLLM_PROBE_CCACHE_ENABLE_WAYS)
  set(_SM_PW 1)
  message(STATUS "bearly-smollm: will ENABLE ccache ways if they are off (cannot be undone without reset)")
endif()
target_compile_definitions(bearly-smollm PRIVATE
  SMOLLM_PROBE_UNCORE=${_SM_PU}
  SMOLLM_PROBE_FEATURE_CSR=${_SM_PF}
  SMOLLM_PROBE_CCACHE_ENABLE_WAYS=${_SM_PW})

# Quiet demo console: no banners, no configuration dump, no per-answer statistics — the chip boots
# straight to "You: ". Errors still print (the loaders use printf directly), and prefill still emits
# one dot per pass because ~14 s of silence is indistinguishable from a hang.
option(SMOLLM_QUIET "Demo console: suppress boot and statistics output" OFF)
set(_SM_Q 0)
if(SMOLLM_QUIET)
  set(_SM_Q 1)
  message(STATUS "bearly-smollm: SMOLLM_QUIET=1 (demo console)")
endif()
target_compile_definitions(bearly-smollm PRIVATE SMOLLM_QUIET=${_SM_Q})

# STACK PLACEMENT. On this platform a stack overflow does not trap — it silently fails to store, or
# hangs the core — and Spike backs the whole region with real zeroed RAM so it is invisible in
# simulation. bearly25-llm.ld already derives the stack from the REAL end of DRAM (unlike
# dsp25-llm.ld, which put it ~1.9 GB past the last backed address), but pin it explicitly anyway:
# the symbols are PROVIDE()d so --defsym wins, and having the address in the build log is worth the
# two lines. 0x8FE00000 + 5 harts * 256 KiB = 0x8FF40000, leaving 768 KiB below the 0x90000000 top.
# __heap_end moves with it so sbrk cannot grow the heap into the stacks.
set(SMOLLM_STACK_BASE 0x8FE00000 CACHE STRING "Stack base in real DRAM")
target_link_options(bearly-smollm PRIVATE
  -Wl,--defsym=__stack_size=0x40000               # 256 KiB per hart
  -Wl,--defsym=__stack_start=${SMOLLM_STACK_BASE}
  -Wl,--defsym=__heap_end=${SMOLLM_STACK_BASE}    # sbrk stops below the stacks
  -Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bearly-smollm.map
)

# Catch the next oversized frame at build time instead of as a mysteriously flaky chip. Every frame
# here is a few hundred bytes; the biggest is emit_piece's 193-byte print buffer.
target_compile_options(bearly-smollm PRIVATE -Wstack-usage=8192)

# glossy (whole-archive) supplies the bearly25 HAL + syscalls; m supplies cosf/sinf/expf/powf/sqrtf.
target_link_libraries(bearly-smollm PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if(PROF_COV)
  target_link_libraries(bearly-smollm PRIVATE gcov)
endif()
