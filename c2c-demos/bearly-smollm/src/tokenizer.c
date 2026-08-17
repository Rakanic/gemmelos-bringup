/* Byte-level BPE tokenizer for SmolLM2 — see include/smollm_tokenizer.h.
 *
 * No chip dependencies on purpose: scripts/check_c_tokenizer.py compiles this exact file for the
 * host and diffs its output against HuggingFace over a corpus, so the code that runs on silicon is
 * the code that was tested. scripts/tokenizer_ref.py is the Python model of the same algorithm. */

#include "smollm_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOK_MAGIC 0x534d544bu  /* 'SMTK' */
#define TOK_HEADER_BYTES 64
#define TOK_MAX_CHUNK 1024     /* longest pre-token chunk we will merge; longer runs are truncated */

typedef struct {
  int32_t a, b;      /* the pair, as token ids */
  int32_t ab;        /* the merged token id; -1 marks an empty slot */
  int32_t rank;      /* position in the merge list — lower merges first */
} MergeSlot;

static struct {
  int vocab_size;
  int n_merges;
  int bos, eos;
  const int32_t *byte_to_id;   /* [256], -1 for bytes with no token in this vocab */
  const uint8_t **piece;       /* [vocab_size] -> the token's raw bytes, inside the blob */
  int32_t *piece_len;
  MergeSlot *merge;
  uint32_t merge_mask;
} g_tok;

static inline uint32_t merge_hash(int32_t a, int32_t b) {
  uint64_t h = (uint64_t)(uint32_t)a * 0x9E3779B97F4A7C15ull;
  h ^= (uint64_t)(uint32_t)b * 0xC2B2AE3D27D4EB4Full;
  h ^= h >> 29;
  return (uint32_t)(h >> 32);
}

/* Merge rank of the pair (a, b), or -1 if it is not a merge; *out gets the merged token id. */
static int merge_lookup(int32_t a, int32_t b, int32_t *out) {
  uint32_t i = merge_hash(a, b) & g_tok.merge_mask;
  for (;;) {
    const MergeSlot *m = &g_tok.merge[i];
    if (m->ab < 0) return -1;
    if (m->a == a && m->b == b) { *out = m->ab; return m->rank; }
    i = (i + 1) & g_tok.merge_mask;
  }
}

int smollm_tok_load(const uint8_t *blob, size_t blob_bytes) {
  uint32_t magic, max_piece;
  int32_t version, vocab_size, n_merges, bos, eos;
  memcpy(&magic, blob + 0, 4);
  memcpy(&version, blob + 4, 4);
  memcpy(&vocab_size, blob + 8, 4);
  memcpy(&n_merges, blob + 12, 4);
  memcpy(&bos, blob + 16, 4);
  memcpy(&eos, blob + 20, 4);
  memcpy(&max_piece, blob + 24, 4);
  if (magic != TOK_MAGIC || version != 1) {
    printf("[smollm] BAD TOKENIZER HEADER magic=0x%08lx version=%ld\r\n",
           (unsigned long)magic, (long)version);
    return -1;
  }
  g_tok.vocab_size = vocab_size;
  g_tok.n_merges = n_merges;
  g_tok.bos = bos;
  g_tok.eos = eos;

  const uint8_t *p = blob + TOK_HEADER_BYTES;
  g_tok.byte_to_id = (const int32_t *)(const void *)p;
  p += 256 * 4;
  const int32_t *triples = (const int32_t *)(const void *)p;
  p += (size_t)n_merges * 3 * 4;

  /* Open-addressed pair -> rank table, sized to keep the load factor under ~40%. */
  uint32_t cap = 1;
  while (cap < (uint32_t)n_merges * 3u) cap <<= 1;
  g_tok.merge_mask = cap - 1;
  g_tok.merge = (MergeSlot *)calloc(cap, sizeof(MergeSlot));
  g_tok.piece = (const uint8_t **)calloc((size_t)vocab_size, sizeof(uint8_t *));
  g_tok.piece_len = (int32_t *)calloc((size_t)vocab_size, sizeof(int32_t));
  if (!g_tok.merge || !g_tok.piece || !g_tok.piece_len) {
    printf("[smollm] tokenizer: out of heap\r\n");
    return -1;
  }
  for (uint32_t i = 0; i < cap; i++) g_tok.merge[i].ab = -1;
  for (int r = 0; r < n_merges; r++) {
    const int32_t a = triples[3 * r], b = triples[3 * r + 1], ab = triples[3 * r + 2];
    uint32_t i = merge_hash(a, b) & g_tok.merge_mask;
    while (g_tok.merge[i].ab >= 0) {
      if (g_tok.merge[i].a == a && g_tok.merge[i].b == b) break;  /* duplicate: keep the lower rank */
      i = (i + 1) & g_tok.merge_mask;
    }
    if (g_tok.merge[i].ab < 0) {
      g_tok.merge[i].a = a; g_tok.merge[i].b = b; g_tok.merge[i].ab = ab; g_tok.merge[i].rank = r;
    }
  }

  for (int i = 0; i < vocab_size; i++) {
    int32_t len;
    memcpy(&len, p, 4);
    p += 4;
    g_tok.piece_len[i] = len;
    g_tok.piece[i] = p;
    p += len;
  }
  const size_t consumed = (size_t)(p - blob);
  if (consumed != blob_bytes) {
    printf("[smollm] TOKENIZER SIZE MISMATCH: parsed %lu of %lu bytes\r\n",
           (unsigned long)consumed, (unsigned long)blob_bytes);
    return -1;
  }
  return 0;   /* the banner is main.c's business — this file only reports FAILURES */
}

/* Byte classes for the pre-tokenizer. Every byte >= 0x80 counts as a letter, which matches \p{L}
 * for actual letters; emoji and CJK punctuation are the known approximation. */
enum { C_SPACE, C_LETTER, C_DIGIT, C_OTHER };

static inline int byte_class(uint8_t b) {
  if (b == ' ' || b == '\t' || b == '\n' || b == '\v' || b == '\f' || b == '\r') return C_SPACE;
  if ((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || b >= 0x80) return C_LETTER;
  if (b >= '0' && b <= '9') return C_DIGIT;
  return C_OTHER;
}

static const char *const CONTRACTIONS[] = { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };

/* End index of the pre-token chunk starting at `i`, reproducing GPT-2's ByteLevel regex
 *   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
 * SmolLM's Digits(individual_digits) pre-tokenizer needs no special case here: it applied when the
 * tokenizer was TRAINED, so no vocab token holds two ASCII digits and BPE leaves a digit run split
 * into single digits regardless. (Modelling it explicitly is actively wrong — it merges "  1" into
 * one two-space token where HF emits two single spaces.) */
static int chunk_end(const uint8_t *s, int n, int i) {
  if (s[i] == '\'') {
    for (unsigned c = 0; c < sizeof(CONTRACTIONS) / sizeof(CONTRACTIONS[0]); c++) {
      const int len = (int)strlen(CONTRACTIONS[c]);
      if (i + len <= n && memcmp(s + i, CONTRACTIONS[c], (size_t)len) == 0) return i + len;
    }
  }
  /* at most ONE literal space (0x20, never \n or \t) may lead a letter/digit/other run */
  int j = (s[i] == ' ' && i + 1 < n && byte_class(s[i + 1]) != C_SPACE) ? i + 1 : i;
  if (byte_class(s[j]) != C_SPACE) {
    const int want = byte_class(s[j]);
    int k = j;
    while (k < n && byte_class(s[k]) == want) k++;
    return k;
  }
  int k = i;
  while (k < n && byte_class(s[k]) == C_SPACE) k++;
  if (k < n && k - i > 1) k--;   /* a run followed by text gives its last space to the next chunk */
  return k;
}

/* Merge one chunk by repeatedly taking the lowest-rank adjacent pair — exact BPE. */
static void encode_chunk(const uint8_t *s, int len, int *out, int *n_out, int cap) {
  static int32_t ids[TOK_MAX_CHUNK];
  int n = 0;
  for (int i = 0; i < len && n < TOK_MAX_CHUNK; i++) {
    const int32_t id = g_tok.byte_to_id[s[i]];
    if (id >= 0) ids[n++] = id;   /* a byte with no token is dropped, exactly as HF does */
  }
  while (n > 1) {
    int best_rank = -1, best_at = -1;
    int32_t best_id = -1;
    for (int i = 0; i < n - 1; i++) {
      int32_t ab;
      const int rank = merge_lookup(ids[i], ids[i + 1], &ab);
      if (rank >= 0 && (best_rank < 0 || rank < best_rank)) {
        best_rank = rank; best_at = i; best_id = ab;
      }
    }
    if (best_at < 0) break;
    ids[best_at] = best_id;
    memmove(&ids[best_at + 1], &ids[best_at + 2], (size_t)(n - best_at - 2) * sizeof(ids[0]));
    n--;
  }
  for (int i = 0; i < n && *n_out < cap; i++) out[(*n_out)++] = (int)ids[i];
}

void smollm_tok_encode(const char *text, int *out, int *n_out, int cap) {
  const uint8_t *s = (const uint8_t *)text;
  const int n = (int)strlen(text);
  int i = 0;
  while (i < n) {
    const int end = chunk_end(s, n, i);
    encode_chunk(s + i, end - i, out, n_out, cap);
    i = end;
  }
}

const uint8_t *smollm_tok_piece(int id, int *len) {
  if (id < 0 || id >= g_tok.vocab_size) { *len = 0; return (const uint8_t *)""; }
  *len = g_tok.piece_len[id];
  return g_tok.piece[id];
}

int smollm_tok_bos(void) { return g_tok.bos; }
int smollm_tok_eos(void) { return g_tok.eos; }
int smollm_tok_vocab_size(void) { return g_tok.vocab_size; }
int smollm_tok_n_merges(void) { return g_tok.n_merges; }
