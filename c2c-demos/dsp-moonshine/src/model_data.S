/* Embed the Q8_0 model + golden audio into .rodata so the demo is self-contained under Spike
 * (no filesystem). Paths resolve via the assembler -I (set to the demo dir in CMakeLists). The
 * model is ~35 MB — fits easily in the dsp25 DRAM region (LINKER=llm). */
#ifndef MOONSHINE_MODEL_BIN
#define MOONSHINE_MODEL_BIN "model/moonshine_tiny_q80.bin"
#endif
#ifndef MOONSHINE_AUDIO_BIN
#define MOONSHINE_AUDIO_BIN "model/moonshine_audio.bin"
#endif

  .section .rodata
  .balign 8
  .globl moonshine_model_blob
moonshine_model_blob:
  .incbin MOONSHINE_MODEL_BIN
  .globl moonshine_model_blob_end
moonshine_model_blob_end:

  .balign 8
  .globl moonshine_audio_blob
moonshine_audio_blob:
  .incbin MOONSHINE_AUDIO_BIN
  .globl moonshine_audio_blob_end
moonshine_audio_blob_end:
