/* Embed the Q8_0 model + golden testcase into .rodata so the demo is self-contained under Spike
 * (no filesystem). Paths resolve via the assembler -I (set to the demo dir in CMakeLists). The
 * model is ~44 MB — fits easily in the dsp25 DRAM region. */
#ifndef WHISPER_MODEL_BIN
#define WHISPER_MODEL_BIN "model/whisper_tiny_en_q80.bin"
#endif
#ifndef WHISPER_TESTCASE_BIN
#define WHISPER_TESTCASE_BIN "model/whisper_testcase.bin"
#endif
#ifndef WHISPER_AUDIO_BIN
#define WHISPER_AUDIO_BIN "model/whisper_audio.bin"
#endif

  .section .rodata
  .balign 8
  .globl whisper_model_blob
whisper_model_blob:
  .incbin WHISPER_MODEL_BIN
  .globl whisper_model_blob_end
whisper_model_blob_end:

  .balign 8
  .globl whisper_testcase_blob
whisper_testcase_blob:
  .incbin WHISPER_TESTCASE_BIN
  .globl whisper_testcase_blob_end
whisper_testcase_blob_end:

  .balign 8
  .globl whisper_audio_blob
whisper_audio_blob:
  .incbin WHISPER_AUDIO_BIN
  .globl whisper_audio_blob_end
whisper_audio_blob_end:
