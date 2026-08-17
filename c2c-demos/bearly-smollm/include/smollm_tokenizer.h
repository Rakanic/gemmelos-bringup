#ifndef SMOLLM_TOKENIZER_H
#define SMOLLM_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

/* GPT-2 style byte-level BPE, the tokenizer SmolLM2 uses (NOT SentencePiece, so none of llama2.c's
 * tokenizer code applies). Deliberately free of any chip dependency — it is compiled into the
 * bearly25 demo AND into test/host_tokenizer_test.c, which diffs it against HuggingFace on the
 * host. See scripts/check_c_tokenizer.py.
 *
 * tokenizer.bin (written by scripts/export_smollm.py) holds:
 *   - a byte -> token id table, with -1 for bytes this vocab cannot represent;
 *   - the merge list as (a, b, ab) id triples IN RANK ORDER, so encoding merges the lowest-rank
 *     adjacent pair — exact BPE, and no string comparisons in the merge loop;
 *   - one raw-bytes "piece" per token for decoding, so the device never handles the byte-level
 *     unicode ('Ġ') alphabet at all. */

int smollm_tok_load(const uint8_t *blob, size_t blob_bytes);

/* Appends the tokens of `text` to out[*n_out], never exceeding `cap` entries. */
void smollm_tok_encode(const char *text, int *out, int *n_out, int cap);

/* The raw bytes a token decodes to (NOT NUL-terminated). */
const uint8_t *smollm_tok_piece(int id, int *len);

int smollm_tok_bos(void);
int smollm_tok_eos(void);
int smollm_tok_vocab_size(void);
int smollm_tok_n_merges(void);

#endif /* SMOLLM_TOKENIZER_H */
