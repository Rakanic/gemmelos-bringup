/* Embed the Q8_0 model + golden audio into .rodata so the demo is self-contained under Spike
 * (no filesystem). Paths resolve via the assembler -I (set to the demo dir in CMakeLists). The
 * model is ~11.8 MB — a third of moonshine's, and it still needs the LINKER=llm DRAM region for
 * headroom. */
#ifndef CITRINET_MODEL_BIN
#define CITRINET_MODEL_BIN "model/citrinet_256_q80.bin"
#endif
#ifndef CITRINET_AUDIO_BIN
#define CITRINET_AUDIO_BIN "model/citrinet_audio.bin"
#endif

  .section .rodata
  .balign 8
  .globl citrinet_model_blob
citrinet_model_blob:
  .incbin CITRINET_MODEL_BIN
  .globl citrinet_model_blob_end
citrinet_model_blob_end:

  .balign 8
  .globl citrinet_audio_blob
citrinet_audio_blob:
  .incbin CITRINET_AUDIO_BIN
  .globl citrinet_audio_blob_end
citrinet_audio_blob_end:
