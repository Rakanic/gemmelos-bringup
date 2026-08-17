/*
 * Host harness around src/tokenizer.c — the SAME file the chip runs.
 *
 * Reads tokenizer.bin and a text file, prints the token ids of each line as "id id id" so
 * scripts/check_c_tokenizer.py can diff them against HuggingFace. Build and run it with that
 * script; it needs no cross toolchain.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "smollm_tokenizer.h"

#define MAX_TOKENS 8192

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <tokenizer.bin> <corpus.txt>\n", argv[0]);
    return 2;
  }
  FILE *f = fopen(argv[1], "rb");
  if (!f) { perror(argv[1]); return 2; }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *blob = malloc((size_t)n);
  if (fread(blob, 1, (size_t)n, f) != (size_t)n) { perror("read"); return 2; }
  fclose(f);

  /* the load banner goes to stderr so stdout stays pure token ids */
  fflush(stdout);
  int saved = dup(fileno(stdout));
  dup2(fileno(stderr), fileno(stdout));
  const int rc = smollm_tok_load(blob, (size_t)n);
  fflush(stdout);
  dup2(saved, fileno(stdout));
  if (rc != 0) return 1;

  FILE *c = fopen(argv[2], "rb");
  if (!c) { perror(argv[2]); return 2; }
  static char line[64 * 1024];
  static int toks[MAX_TOKENS];
  while (fgets(line, sizeof(line), c)) {
    size_t len = strlen(line);
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
    int nt = 0;
    smollm_tok_encode(line, toks, &nt, MAX_TOKENS);
    for (int i = 0; i < nt; i++) printf("%s%d", i ? " " : "", toks[i]);
    printf("\n");
  }
  fclose(c);
  return 0;
}
