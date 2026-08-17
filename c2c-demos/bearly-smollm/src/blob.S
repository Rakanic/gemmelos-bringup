/*
 * Bake the quantized model and the tokenizer tables straight into the ELF with `.incbin`.
 *
 * `.incbin` is a file copy done by the assembler, so a 143 MB model costs nothing at build time.
 * The C-array route (borai's weights.h) does not scale: borai's 260 KB model is already 1.5 MB of
 * source text, so 143 MB would be several GB of text for GCC to parse.
 *
 * Paths arrive as -D macros from CMake; this file is `.S` (capital) so the C preprocessor runs
 * first and STR() can stringize a bare path into the quoted form `.incbin` wants.
 *
 * Both blobs land in .rodata, which bearly25-llm.ld places in DRAM inside the single PT_LOAD
 * segment — so they travel inside the ELF and the loader writes them to DRAM as part of the normal
 * load, with no separate preload step. Cost depends on the path: ~40 s over the 30 MHz SPI loader,
 * ~26 minutes over uart_tsi at 921600 baud.
 */

#define STR_(x) #x
#define STR(x)  STR_(x)

    .section .rodata
    .balign 64
    .global g_smollm_model
g_smollm_model:
    .incbin STR(SMOLLM_MODEL_BIN)
    .global g_smollm_model_end
g_smollm_model_end:

    .balign 64
    .global g_smollm_tokenizer
g_smollm_tokenizer:
    .incbin STR(SMOLLM_TOKENIZER_BIN)
    .global g_smollm_tokenizer_end
g_smollm_tokenizer_end:
