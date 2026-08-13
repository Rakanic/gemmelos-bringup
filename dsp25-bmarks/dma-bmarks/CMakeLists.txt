# CMake definitions for target `dma-bmarks`.

add_executable(dma-bmarks
  src/main.c
  ${CMAKE_SOURCE_DIR}/bmark-lib/simple_setup.c
)

# The two scratchpad cases ship disabled (bench_config.h defaults them to 0). They are the ones that
# matter for staging weights/activations in the 64 KB SRAM at 0x08000000, and DMA->scratchpad has a
# history on this chip (whisper 2026-08-04 and moonshine 2026-08-06 both recorded hangs, attributed
# to the SRAM being cached while the DMA is not coherent). Expose them so that claim can be re-tested
# directly — the benchmark verifies every transfer and its wait is bounded, so a bad result is a
# printed mismatch rather than a wedge.
option(DMA_BENCH_SCRATCH_CASES "Enable the DRAM<->scratchpad transfer cases" OFF)
if(DMA_BENCH_SCRATCH_CASES)
  target_compile_definitions(dma-bmarks PRIVATE
    DMA_BENCH_ENABLE_CASE_DRAM_TO_SCRATCH=1
    DMA_BENCH_ENABLE_CASE_SCRATCH_TO_DRAM=1)
  message(STATUS "dma-bmarks: DRAM<->scratchpad cases ENABLED")
endif()

target_include_directories(dma-bmarks PUBLIC include)
target_include_directories(dma-bmarks PRIVATE ${CMAKE_SOURCE_DIR}/bmark-lib)

target_link_libraries(dma-bmarks PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
)

if(PROF_COV)
  target_link_libraries(dma-bmarks PRIVATE gcov)
endif()
