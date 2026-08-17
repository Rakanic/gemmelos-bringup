/*
 * Host harness around src/model.c + src/tokenizer.c — the SAME files the chip runs.
 *
 * Loads the real model_q80.bin and tokenizer.bin from disk, builds SmolLM2's chat prompt, and
 * greedy-decodes an answer, printing the per-stage sums of the first forward pass. Those sums and
 * the generated text are directly comparable to scripts/ref_runq.py, which is validated against
 * HuggingFace — so this proves the C numerics without a chip, a simulator, or a 26-minute flash.
 *
 * Build and run it with scripts/check_c_forward.py.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "smollm_model.h"
#include "smollm_tokenizer.h"

#define MAX_PROMPT_TOKENS 2048

static const char *SYSTEM_PROMPT =
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

static void stage_hook(int stage, const char *name, float sum) {
  printf("STAGE %d %s %.6e\n", stage, name, (double)sum);
}

static unsigned char *slurp(const char *path, size_t *out_bytes) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); return NULL; }
  fseek(f, 0, SEEK_END);
  const long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *buf = malloc((size_t)n);
  if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { perror(path); fclose(f); return NULL; }
  fclose(f);
  *out_bytes = (size_t)n;
  return buf;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <model_q80.bin> <tokenizer.bin> <prompt> [max_new_tokens]\n", argv[0]);
    return 2;
  }
  const int max_new = (argc > 4) ? atoi(argv[4]) : 24;

  size_t model_bytes = 0, tok_bytes = 0;
  unsigned char *model = slurp(argv[1], &model_bytes);
  unsigned char *tokbin = slurp(argv[2], &tok_bytes);
  if (!model || !tokbin) return 2;

  if (smollm_model_load(model, model_bytes) != 0) return 1;
  if (smollm_tok_load(tokbin, tok_bytes) != 0) return 1;
  if (smollm_model_alloc() != 0) return 1;
  const smollm_config_t *cfg = smollm_model_config();
  printf("config dim=%d hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d vocab=%d "
         "seq_len=%d GS=%d rope_theta=%g eps=%g\n",
         cfg->dim, cfg->hidden_dim, cfg->n_layers, cfg->n_heads, cfg->n_kv_heads, cfg->head_dim,
         cfg->vocab_size, cfg->seq_len, cfg->group_size, (double)cfg->rope_theta,
         (double)cfg->norm_eps);

  /* the same chat template main.c builds */
  static int pt[MAX_PROMPT_TOKENS];
  static char scratch[4096];
  int n = 0;
  pt[n++] = smollm_tok_bos();
  snprintf(scratch, sizeof(scratch), "system\n%s", SYSTEM_PROMPT);
  smollm_tok_encode(scratch, pt, &n, MAX_PROMPT_TOKENS);
  pt[n++] = smollm_tok_eos();
  smollm_tok_encode("\n", pt, &n, MAX_PROMPT_TOKENS);
  pt[n++] = smollm_tok_bos();
  snprintf(scratch, sizeof(scratch), "user\n%s", argv[3]);
  smollm_tok_encode(scratch, pt, &n, MAX_PROMPT_TOKENS);
  pt[n++] = smollm_tok_eos();
  smollm_tok_encode("\n", pt, &n, MAX_PROMPT_TOKENS);
  pt[n++] = smollm_tok_bos();
  smollm_tok_encode("assistant\n", pt, &n, MAX_PROMPT_TOKENS);

  printf("PROMPT %d tokens:", n);
  for (int i = 0; i < n; i++) printf(" %d", pt[i]);
  printf("\n");

  smollm_set_stage_hook(stage_hook);
  const char *sp = getenv("SMOLLM_STAGE_POS");
  smollm_set_stage_pos(sp ? atoi(sp) : 0);
  const char *tl = getenv("SMOLLM_TRACE_LAYER");
  smollm_set_trace_layer(tl ? atoi(tl) : -1);
  float *logits = NULL;
  int pos = 0;

  /* SMOLLM_BATCH=N prefills N tokens per pass over the weights. Batched and sequential do the same
   * arithmetic in the same order, so the logits must come out BIT-identical — which is what makes
   * this worth testing on the host rather than discovering on silicon. */
  const char *bs = getenv("SMOLLM_BATCH");
  const int batch = bs ? atoi(bs) : 1;
  if (batch > 1) {
    for (int i = 0; i < n; ) {
      int chunk = n - i;
      if (chunk > batch) chunk = batch;
      logits = smollm_forward_batch(&pt[i], chunk, pos);
      pos += chunk;
      i += chunk;
    }
  } else {
    for (int i = 0; i < n; i++) logits = smollm_forward(pt[i], pos++);
  }
  smollm_set_stage_hook(NULL);

  int generated = 0;
  static int gen_ids[4096];
  for (;;) {
    int next = 0;
    for (int i = 1; i < cfg->vocab_size; i++) if (logits[i] > logits[next]) next = i;
    /* top-4 per step, so a divergence from the reference can be read as "near tie" vs "wrong" */
    int top[4] = {0, 0, 0, 0};
    for (int r = 0; r < 4; r++) {
      int best = -1;
      for (int i = 0; i < cfg->vocab_size; i++) {
        int seen = 0;
        for (int j = 0; j < r; j++) if (top[j] == i) seen = 1;
        if (!seen && (best < 0 || logits[i] > logits[best])) best = i;
      }
      top[r] = best;
    }
    printf("STEP %d pick=%d", generated, next);
    for (int r = 0; r < 4; r++) printf(" | %d %.6f", top[r], (double)logits[top[r]]);
    printf("\n");
    if (next == smollm_tok_eos() || next == smollm_tok_bos()) break;
    if (generated >= max_new || pos >= cfg->seq_len) break;
    gen_ids[generated++] = next;
    logits = smollm_forward(next, pos++);
  }
  printf("OUTPUT ");
  for (int i = 0; i < generated; i++) {
    int len = 0;
    const uint8_t *piece = smollm_tok_piece(gen_ids[i], &len);
    fwrite(piece, 1, (size_t)len, stdout);
  }
  printf("\nTOKENS");
  for (int i = 0; i < generated; i++) printf(" %d", gen_ids[i]);
  printf("\n");
  return 0;
}
