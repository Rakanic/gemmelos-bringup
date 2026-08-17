# membw: standalone memory latency / bandwidth sweep. No C2C link, no accelerator — just DRAM.
#
# Builds for EITHER chip (like hello-wfi); the sweep is pure loads and stores:
#   make build CHIP=dsp25    PLATFORM=CHIP TARGET=membw
#   make build CHIP=bearly25 PLATFORM=CHIP TARGET=membw
#   make tsi-run TTY=<tty> BINARY=build/c2c-demos/membw/membw.elf
#
# The test window is a fixed ABSOLUTE DRAM range (see include/membw_config.h), not a .bss array or a
# heap allocation, so the ELF stays a few KB and uart_tsi has nothing to zero-fill over the serial
# link. Stock linker script, stock heap — nothing here allocates.

add_executable(membw
  src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c   # init_test(): portable PLL + UART bring-up
)

target_include_directories(membw PUBLIC include)
target_include_directories(membw PRIVATE
  ${CMAKE_SOURCE_DIR}/bmark-lib
  ${CMAKE_CURRENT_SOURCE_DIR}/../common          # mem_probe.h, shared with bearly-smollm
)

# The global add_compile_options(-O1) in the top-level CMakeLists lands AFTER Release's -O3 and the
# last -O wins, so without a target-level override the measurement loops would build at -O1 and
# measure the loop instead of the memory. Target options come after directory options.
target_compile_options(membw PRIVATE -O3)

# Frequency the PLL is asked for. The sweep measures the REAL core clock against CLINT mtime and
# prints both, so this only has to be close enough to boot.
set(MEMBW_TARGET_FREQ_HZ 750000000 CACHE STRING "PLL target frequency in Hz")

# Test window. Defaults: 128 MB starting 64 MB into DRAM, split into a 64 MB source half and a
# 64 MB destination half. On a board with more than 256 MB of DRAM these can be raised; the boot
# check writes and reads back the first and last word (and detects aliasing) before anything else
# touches the range, so an over-large window is reported rather than silently measured.
set(MEMBW_BUF_ADDR   0x84000000 CACHE STRING "Base address of the test window")
set(MEMBW_BUF_BYTES  134217728  CACHE STRING "Test window size in bytes (split in half)")
set(MEMBW_MAX_SPAN   67108864   CACHE STRING "Largest latency-curve span in bytes (<= window/2)")
set(MEMBW_STREAM_SPAN 33554432  CACHE STRING "Span used by the streaming tests in bytes")
set(MEMBW_PASSES     2          CACHE STRING "How many times to run the whole sweep")

# Memory-level parallelism sweep: 8 independent pointer chases of this size each (8 x must fit in
# MEMBW_MAX_SPAN). Each must be far larger than the last-level cache on its own or the "misses" it
# offers are hits — 4 MB against a 256 KB LLC is a 16x margin.
set(MEMBW_MLP_SUB_SPAN 4194304 CACHE STRING "Bytes per independent chase chain")

# Bandwidth a link change is meant to deliver; section 6 turns it into "lines in flight required".
set(MEMBW_LINK_TARGET_MBPS 33 CACHE STRING "Target link bandwidth in MB/s for the Little's Law bar")

target_compile_definitions(membw PRIVATE
  MEMBW_TARGET_FREQ_HZ=${MEMBW_TARGET_FREQ_HZ}ULL
  MEMBW_BUF_ADDR=${MEMBW_BUF_ADDR}UL
  MEMBW_BUF_BYTES=${MEMBW_BUF_BYTES}u
  MEMBW_MAX_SPAN=${MEMBW_MAX_SPAN}u
  MEMBW_STREAM_SPAN=${MEMBW_STREAM_SPAN}u
  MEMBW_MLP_SUB_SPAN=${MEMBW_MLP_SUB_SPAN}u
  MEMBW_LINK_TARGET_MBPS=${MEMBW_LINK_TARGET_MBPS}u
  MEMBW_PASSES=${MEMBW_PASSES}
)
message(STATUS "membw: window ${MEMBW_BUF_ADDR} + ${MEMBW_BUF_BYTES} B, PLL ${MEMBW_TARGET_FREQ_HZ} Hz")

# Catch an oversized frame at build time: a stack overflow on this platform does not trap, it
# silently corrupts automatics or hangs the core (CLAUDE.md). Every frame here is a few hundred
# bytes; the largest automatic is a 32-byte label buffer.
target_compile_options(membw PRIVATE -Wstack-usage=2048)

# Keep the loadable image tiny. The stock linker scripts put .heap and .stack inside the single LOAD
# segment, so their sizes land in memsz and uart_tsi zero-fills the difference over the serial link
# on every flash — the default 1 MB heap alone costs ~12 s at 921600 baud for memory this program
# never allocates from. 64 KiB covers newlib's stdio bookkeeping; nothing here calls malloc.
# 16 KiB of stack per hart is ~8x the largest frame (a printf).
target_link_options(membw PRIVATE
  -Wl,--defsym=__heap_size=0x10000
  -Wl,--defsym=__stack_size=0x4000
  -Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/membw.map
)

target_link_libraries(membw PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  m
)

if(PROF_COV)
  target_link_libraries(membw PRIVATE gcov)
endif()
