# bearly-smollm-c2c: the Bearly ML 25 half of the voice assistant.
#
#   DSP 25 (dsp-citrinet-c2c) speaks -> transcript over the C2C link -> SmolLM2-135M-Instruct
#   prefill + decode here -> the answer goes back over the link (and to this console).
#
# It is the proven `bearly-smollm` demo compiled with SMOLLM_C2C=1, plus one new source file
# (src/stt_link_bml.c) that speaks the protocol in c2c-demos/common/stt_link_proto.h. The SmolLM
# sources are REUSED, not copied, so the model path here and in bearly-smollm cannot drift apart.
#
#   .venv/bin/python c2c-demos/bearly-smollm/scripts/export_smollm.py    # once: writes model/*.bin
#   make build CHIP=bearly25 PLATFORM=CHIP TARGET=bearly-smollm-c2c EXTRA_CMAKE_ARGS="-DLINKER=llm"
#   make tsi-run TTY=<tty> BINARY=build/c2c-demos/bearly-smollm-c2c/bearly-smollm-c2c.elf
#
# 143 MB of weights: ~40 s over the 30 MHz SPI loader, ~26 minutes over uart_tsi. Boot then takes a
# few more seconds (KV cache); only when that is done does this chip announce readiness over the
# link, which is what unblocks the DSP. Start the DSP first or together — it waits indefinitely and
# issues no cross-link write until the announcement arrives.
#
# LINKER=llm is REQUIRED and the top-level Makefile does not forward LINKER=, so it goes inside
# EXTRA_CMAKE_ARGS. It selects platform/bearly25/bearly25-llm.ld, which declares heap and stack as
# bare symbols (otherwise uart_tsi zero-fills ~100 MB of heap over the serial link on every load)
# and keeps DRAM at its real 256 MB so the stack lands inside memory that exists.
#
# ------------------------------------------------------------------------------------------------
# WHY EVERY CACHE VARIABLE BELOW IS PREFIXED `SMOLLM_C2C_`
#
# CMake `option()` and `set(... CACHE ...)` create GLOBAL cache entries, and a second declaration of
# a name that already exists is a NO-OP — the first one wins. c2c-demos/CMakeLists.txt adds
# bearly-smollm before this directory, so reusing its names silently inherited ITS defaults here:
# this target shipped with SMOLLM_MEM_PROBE=ON (a screenful of boot output and 2 s of probing that
# the DSP spends blocked) and SMOLLM_MAX_NEW_TOKENS=200 instead of the 64 configured below — and
# nothing in the build log said so. Never reuse a sibling demo's cache-variable name.
# ------------------------------------------------------------------------------------------------

set(_SM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../bearly-smollm)

# Blobs baked into the ELF by blob.S — shared with bearly-smollm, not duplicated.
set(SMOLLM_C2C_MODEL_BIN     "${_SM_SRC}/model/model_q80.bin"
    CACHE STRING "Grouped-Q8_0 SmolLM model blob to embed")
set(SMOLLM_C2C_TOKENIZER_BIN "${_SM_SRC}/model/tokenizer.bin"
    CACHE STRING "Byte-level BPE tokenizer blob to embed")
foreach(_blob "${SMOLLM_C2C_MODEL_BIN}" "${SMOLLM_C2C_TOKENIZER_BIN}")
  if(NOT EXISTS "${_blob}")
    message(WARNING "bearly-smollm-c2c: ${_blob} is missing — run bearly-smollm/scripts/export_smollm.py "
                    "first, or the .incbin will fail the build.")
  endif()
endforeach()

add_executable(bearly-smollm-c2c
  ${_SM_SRC}/src/main.c
  ${_SM_SRC}/src/model.c
  ${_SM_SRC}/src/tokenizer.c
  ${_SM_SRC}/src/dualcore.c
  ${_SM_SRC}/src/uncore_probe.c
  ${_SM_SRC}/src/blob.S
  src/stt_link_bml.c
  ${CMAKE_SOURCE_DIR}/c2c-demos/common/c2c_shm.c   # the one flush / repeat-write implementation
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c     # init_test(): portable PLL + UART bring-up
)

target_include_directories(bearly-smollm-c2c PRIVATE
  include
  ${_SM_SRC}/include
  ${CMAKE_SOURCE_DIR}/bmark-lib
  ${CMAKE_SOURCE_DIR}/driver/rocket-chip-blocks/uart
  ${CMAKE_SOURCE_DIR}/c2c-demos/common            # stt_link_proto.h, c2c_shm.h, mem_probe.h
)

target_compile_definitions(bearly-smollm-c2c PRIVATE
  SMOLLM_C2C=1
  SMOLLM_MODEL_BIN=${SMOLLM_C2C_MODEL_BIN}        # bare paths; blob.S stringizes them for .incbin
  SMOLLM_TOKENIZER_BIN=${SMOLLM_C2C_TOKENIZER_BIN}
)

# OBJECT_DEPENDS is REQUIRED: the compiler emits no depfile for `.incbin`, so a re-exported model
# would NOT rebuild this object and the ELF would keep the OLD weights — which looks exactly like a
# model bug.
set_source_files_properties(${_SM_SRC}/src/blob.S PROPERTIES
  OBJECT_DEPENDS "${SMOLLM_C2C_MODEL_BIN};${SMOLLM_C2C_TOKENIZER_BIN}")

# ---- link geometry (declared once in c2c-demos/CMakeLists.txt — must match the DSP side) ---------
target_compile_definitions(bearly-smollm-c2c PRIVATE
  STT_LINK_RETURN_ANSWER=${STT_LINK_RETURN_ANSWER_VALUE}
  STT_PROMPT_MAX_BYTES=${STT_PROMPT_MAX_BYTES}u
  STT_ANSWER_MAX_BYTES=${STT_ANSWER_MAX_BYTES}u)

# ---- console -------------------------------------------------------------------------------------
# QUIET is ON here, unlike the standalone demo: this is a spoken-word demo whose console should show
# the conversation and nothing else. It suppresses every boot banner, the configuration dump and the
# per-answer statistics (and, via smollm_config.h, forces the memory probe off). The link's own
# boot chatter follows the same flag — see STT_BML_BOOT_LOG in src/stt_link_bml.c. What still
# prints: the transcript, the token stream, one dot per prefill pass (14 s of silence reads as a
# hung chip), and any link anomaly — a retransmit, a duplicate grant, a failed checksum.
option(SMOLLM_C2C_QUIET "Demo console: show only the conversation" ON)
set(_SM_Q 0)
if(SMOLLM_C2C_QUIET)
  set(_SM_Q 1)
endif()
target_compile_definitions(bearly-smollm-c2c PRIVATE SMOLLM_QUIET=${_SM_Q})

# ---- model ---------------------------------------------------------------------------------------
set(SMOLLM_C2C_OPT "-O3" CACHE STRING "Optimization level (overrides the global -O1)")
target_compile_options(bearly-smollm-c2c PRIVATE ${SMOLLM_C2C_OPT})

option(SMOLLM_C2C_USE_RVV "Use the RVV grouped-int8 dot product" ON)
set(_SM_RVV 0)
if(SMOLLM_C2C_USE_RVV)
  set(_SM_RVV 1)
endif()
# 1 = lane-float (products stay in lanes, one float reduction per output row). Row-dot (0) uses the
# integer widening reduction vwredsum, which is BROKEN ON THIS SILICON — it produced fluent garbage
# on the first working flash. Do not change this without re-running the kernel benchmark.
set(SMOLLM_C2C_MATMUL_LANEFLOAT 1 CACHE STRING "1 = lane-float matmul, 0 = row-dot (BROKEN on bearly25)")
target_compile_definitions(bearly-smollm-c2c PRIVATE
  SMOLLM_USE_RVV=${_SM_RVV}
  SMOLLM_MATMUL_LANEFLOAT=${SMOLLM_C2C_MATMUL_LANEFLOAT})

# PLL target. Keep it equal to the DSP's CN_C2C_TARGET_FREQ_HZ: the link's retransmit periods are
# counted in CLINT ticks, and mtime is derived from the core clock on this silicon, so the timing
# constants in stt_link_proto.h only mean what their comments say if both chips agree.
set(SMOLLM_C2C_TARGET_FREQ_HZ 750000000 CACHE STRING "PLL target frequency in Hz")
target_compile_definitions(bearly-smollm-c2c PRIVATE
  SMOLLM_TARGET_FREQUENCY_HZ=${SMOLLM_C2C_TARGET_FREQ_HZ}ULL)

# Sampling. 0 = greedy/deterministic. Raise it (e.g. 0.8) for livelier demo output once the link is
# trusted — while bringing it up, a wrong answer should not be blamable on the dice.
set(SMOLLM_C2C_TEMPERATURE "0.0f" CACHE STRING "Sampling temperature (0 = greedy)")
set(SMOLLM_C2C_TOPK 40 CACHE STRING "Sample from the k most likely tokens when temperature > 0")
# ---- ANSWER LENGTH — the only thing that meaningfully changes how long an answer takes -----------
# Decode is 100% weight-bandwidth bound: every token streams all 143 MB of .rodata, ~10 s at
# 750 MHz. So wall-clock = tokens x 10 s, and NOTHING else about the decode is worth tuning. Three
# levers, in order of value:
#
#   1. SYSTEM PROMPT (free). An instruct model does what it is told about length. This costs ~15
#      extra prompt tokens ONCE per conversation (multi-turn keeps the KV cache, and prefill is
#      batched 32-at-a-time), against saving whole tokens off every answer.
#   2. STOP_AT_SENTENCE (free). Ends on the first '.', '!' or '?' after MIN_NEW_TOKENS, so the
#      answer stops at a sentence boundary instead of running to <|im_end|>.
#   3. MAX_NEW_TOKENS. A hard backstop only — it truncates mid-word, so it is the worst-looking way
#      to get a short answer. Keep it as a bound on the worst case, not as the mechanism.
#
# Orthogonal to all three: read FEWER BYTES per token. src/model.c supports Q4_0/Q4_1 and
# scripts/export_smollm.py has already written model/model_q41a.bin (84 MB) and model_q4a.bin
# (76 MB) — about 1.7-1.9x fewer bytes, hence that much faster PER TOKEN:
#   -DSMOLLM_C2C_MODEL_BIN=<repo>/c2c-demos/bearly-smollm/model/model_q41a.bin
set(SMOLLM_C2C_SYSTEM_PROMPT
    "You are a helpful voice assistant. Answer in ONE short sentence. Never explain, never list, never repeat the question."
    CACHE STRING "System prompt; the cheapest way to control answer length and style")

# Few-shot examples: OFF, and this is a measured decision rather than a default nobody revisited.
#
# Two worked exchanges were tried as the anti-echo lever and made things WORSE on silicon. A 135M
# model does not read them as "here is the required FORM"; it reads them as conversation context and
# keeps talking about what they mention. With an example about transistors in the prompt, "he"
# answered "A transistor is a fundamental..." and "hall how are you" answered "Hall is a
# transistor." The examples leaked their content into every reply.
#
# They also cost real time: ~60-80 extra prompt tokens turned the first prefill from 1 pass into
# 3-4, and every pass streams all 143 MB of weights (~10 s). A system-prompt instruction is both
# cheaper and, here, better behaved.
option(SMOLLM_C2C_FEWSHOT "Prepend two example exchanges to steer answer style (see comment)" OFF)
set(_SM_FEWSHOT 0)
if(SMOLLM_C2C_FEWSHOT)
  set(_SM_FEWSHOT 1)
endif()
set(SMOLLM_C2C_FEWSHOT_1_Q "Hello, how are you?" CACHE STRING "Few-shot example 1: user")
set(SMOLLM_C2C_FEWSHOT_1_A "Good, thanks. What can I do for you?" CACHE STRING "Few-shot example 1: assistant")
set(SMOLLM_C2C_FEWSHOT_2_Q "What is a transistor?" CACHE STRING "Few-shot example 2: user")
set(SMOLLM_C2C_FEWSHOT_2_A "A switch that uses a small voltage to control a larger current."
    CACHE STRING "Few-shot example 2: assistant")
# Capitalize the ASR transcript and add '.' or '?' before it becomes the prompt. OFF: it went in
# alongside the few-shot experiment and is being reverted with it, so that whatever is judged next is
# judged against the configuration that was known to work.
option(SMOLLM_C2C_POLISH_PROMPT "Capitalize/punctuate the ASR transcript before prompting" OFF)
set(_SM_POLISH 0)
if(SMOLLM_C2C_POLISH_PROMPT)
  set(_SM_POLISH 1)
endif()
target_compile_definitions(bearly-smollm-c2c PRIVATE
  SMOLLM_POLISH_PROMPT=${_SM_POLISH}
  SMOLLM_FEWSHOT=${_SM_FEWSHOT}
  SMOLLM_FEWSHOT_1_Q="${SMOLLM_C2C_FEWSHOT_1_Q}"
  SMOLLM_FEWSHOT_1_A="${SMOLLM_C2C_FEWSHOT_1_A}"
  SMOLLM_FEWSHOT_2_Q="${SMOLLM_C2C_FEWSHOT_2_Q}"
  SMOLLM_FEWSHOT_2_A="${SMOLLM_C2C_FEWSHOT_2_A}")
option(SMOLLM_C2C_STOP_AT_SENTENCE "Stop generating at the first sentence end" ON)
set(_SM_STOP 0)
if(SMOLLM_C2C_STOP_AT_SENTENCE)
  set(_SM_STOP 1)
endif()
# Floor before the sentence stop may fire, so an answer opening with "3.14" or "e.g." is not cut at
# its first period.
set(SMOLLM_C2C_MIN_NEW_TOKENS 6 CACHE STRING "Tokens before STOP_AT_SENTENCE may fire")
# Backstop, not the mechanism. 48 x ~10 s bounds the worst case at ~8 minutes; with the two levers
# above a typical answer should land well under 20 tokens.
set(SMOLLM_C2C_MAX_NEW_TOKENS 48 CACHE STRING "Hard cap on tokens per answer (truncates mid-word)")
# Keep the KV cache across turns so a spoken conversation has memory.
set(SMOLLM_C2C_MULTI_TURN 1 CACHE STRING "1 = remember the conversation across utterances")
# Tokens per prefill pass. Prefill cost is (passes x 143 MB), so what matters is how many PASSES a
# prompt needs, not how many tokens it has: 64 puts any realistic spoken question plus the system
# block in a single pass. Costs ~19 KB of scratch per token (1.2 MB at 64) against a ~100 MB heap.
set(SMOLLM_C2C_MAX_BATCH 64 CACHE STRING "Tokens processed per prefill pass")
target_compile_definitions(bearly-smollm-c2c PRIVATE
  SMOLLM_TEMPERATURE=${SMOLLM_C2C_TEMPERATURE}
  SMOLLM_TOPK=${SMOLLM_C2C_TOPK}
  SMOLLM_MAX_NEW_TOKENS=${SMOLLM_C2C_MAX_NEW_TOKENS}
  SMOLLM_MAX_BATCH=${SMOLLM_C2C_MAX_BATCH}
  SMOLLM_MIN_NEW_TOKENS=${SMOLLM_C2C_MIN_NEW_TOKENS}
  SMOLLM_STOP_AT_SENTENCE=${_SM_STOP}
  SMOLLM_MULTI_TURN=${SMOLLM_C2C_MULTI_TURN}
  SMOLLM_SYSTEM_PROMPT="${SMOLLM_C2C_SYSTEM_PROMPT}")

# ---- diagnostics, all OFF for the demo console ---------------------------------------------------
# CHECK_BLOB (FNV-1a over the 143 MB at boot) is the one worth turning on when something looks wrong:
# it separates "the weights did not survive the load" from "the maths is wrong", which look identical
# on the console. It costs ~8 Gcyc and prints a line, so it is off by default here.
option(SMOLLM_C2C_CHECK_BLOB "Verify the model blob checksum at boot" OFF)
option(SMOLLM_C2C_BENCH_KERNELS "Benchmark matmul variants at boot" OFF)
option(SMOLLM_C2C_MEM_PROBE "Probe memory latency/bandwidth at boot" OFF)
option(SMOLLM_C2C_DEBUG_GOLDEN "Check per-layer sums against include/smollm_golden.h" OFF)
option(SMOLLM_C2C_DUALCORE "Split batched prefill across both harts (probed at boot)" OFF)
option(SMOLLM_C2C_DUALCORE_DECODE "Split decode matmuls across both harts too" OFF)
option(SMOLLM_C2C_PROBE_UNCORE "Print cache-controller / core config registers at boot" OFF)
set(SMOLLM_C2C_MEM_PROBE_BYTES 16777216 CACHE STRING "Heap scratch for the probe (>> last-level cache)")
foreach(_o CHECK_BLOB BENCH_KERNELS MEM_PROBE DEBUG_GOLDEN DUALCORE DUALCORE_DECODE PROBE_UNCORE)
  set(_v 0)
  if(SMOLLM_C2C_${_o})
    set(_v 1)
    message(STATUS "bearly-smollm-c2c: SMOLLM_${_o}=1")
  endif()
  target_compile_definitions(bearly-smollm-c2c PRIVATE SMOLLM_${_o}=${_v})
endforeach()
target_compile_definitions(bearly-smollm-c2c PRIVATE
  SMOLLM_MEM_PROBE_BYTES=${SMOLLM_C2C_MEM_PROBE_BYTES}u
  SMOLLM_PROBE_FEATURE_CSR=0
  SMOLLM_PROBE_CCACHE_ENABLE_WAYS=0)

# STACK PLACEMENT. A stack overflow on this platform does not trap — it silently fails to store, or
# hangs the core — and Spike backs the whole region with real zeroed RAM so it is invisible in
# simulation. bearly25-llm.ld already derives the stack from the REAL end of DRAM, but pin it
# explicitly: the symbols are PROVIDE()d so --defsym wins, and having the address in the build log
# is worth two lines. 0x8FE00000 + 5 harts * 256 KiB = 0x8FF40000, 768 KiB below the 0x90000000 top.
set(SMOLLM_C2C_STACK_BASE 0x8FE00000 CACHE STRING "Stack base in real DRAM")
target_link_options(bearly-smollm-c2c PRIVATE
  -Wl,--defsym=__stack_size=0x40000               # 256 KiB per hart
  -Wl,--defsym=__stack_start=${SMOLLM_C2C_STACK_BASE}
  -Wl,--defsym=__heap_end=${SMOLLM_C2C_STACK_BASE} # sbrk stops below the stacks
  -Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/bearly-smollm-c2c.map
)

target_compile_options(bearly-smollm-c2c PRIVATE -Wstack-usage=8192)

message(STATUS "bearly-smollm-c2c: quiet=${_SM_Q} rvv=${_SM_RVV} multi_turn=${SMOLLM_C2C_MULTI_TURN} "
               "freq=${SMOLLM_C2C_TARGET_FREQ_HZ}")
message(STATUS "bearly-smollm-c2c: answer length -> stop_at_sentence=${_SM_STOP} "
               "min=${SMOLLM_C2C_MIN_NEW_TOKENS} max=${SMOLLM_C2C_MAX_NEW_TOKENS} tokens "
               "(~10 s/token: worst case ~${SMOLLM_C2C_MAX_NEW_TOKENS}0 s)")
message(STATUS "bearly-smollm-c2c: system prompt \"${SMOLLM_C2C_SYSTEM_PROMPT}\"")

# glossy (whole-archive) supplies the bearly25 HAL + syscalls; m supplies cosf/sinf/expf/powf/sqrtf.
target_link_libraries(bearly-smollm-c2c PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if(PROF_COV)
  target_link_libraries(bearly-smollm-c2c PRIVATE gcov)
endif()
