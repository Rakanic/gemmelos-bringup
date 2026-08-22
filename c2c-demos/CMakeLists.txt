add_subdirectory(dsp-kws)
add_subdirectory(bearly-kws)
add_subdirectory(dsp-kws-rolling)
add_subdirectory(bearly-kws-rolling)
add_subdirectory(bearly-kws-llama)
add_subdirectory(dsp-i2s-test)
add_subdirectory(dsp-mic-bench)
add_subdirectory(dsp-whisper)
add_subdirectory(dsp-moonshine)
add_subdirectory(dsp-citrinet)
add_subdirectory(bearly-smollm)
# Voice assistant: dsp-citrinet-c2c (mic -> transcript) talks to bearly-smollm-c2c (prompt -> answer)
# over the C2C link. Each reuses its standalone demo's sources; see common/stt_link_proto.h.
#
# The link's payload geometry is declared HERE, once, on purpose. The two chips are separately
# linked binaries that agree on the scratchpad layout only through common/stt_link_proto.h, so these
# must be identical on both sides; declaring them in each demo's CMakeLists would mean two defaults
# that CMake silently resolves in favour of whichever subdirectory is added first.
set(STT_PROMPT_MAX_BYTES 256 CACHE STRING "Max prompt bytes carried over the C2C link (multiple of 4)")
set(STT_ANSWER_MAX_BYTES 1024 CACHE STRING "Max answer bytes carried over the C2C link (multiple of 4)")
option(STT_LINK_RETURN_ANSWER "Send the generated answer back to the DSP console" ON)
set(STT_LINK_RETURN_ANSWER_VALUE 0)
if(STT_LINK_RETURN_ANSWER)
  set(STT_LINK_RETURN_ANSWER_VALUE 1)
endif()
message(STATUS "stt-link: prompt<=${STT_PROMPT_MAX_BYTES}B answer<=${STT_ANSWER_MAX_BYTES}B "
               "return_answer=${STT_LINK_RETURN_ANSWER_VALUE}")

add_subdirectory(dsp-citrinet-c2c)
add_subdirectory(bearly-smollm-c2c)
add_subdirectory(membw)
add_subdirectory(dsp-simpletest)
add_subdirectory(bearly-simpletest)
add_subdirectory(c2c-measure)
add_subdirectory(hello-wfi)
add_subdirectory(dsp-hello-wfi)
add_subdirectory(bearly-hello-wfi)
add_subdirectory(c2c-transfer-dsp)
add_subdirectory(c2c-transfer-bearly)
