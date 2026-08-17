#!/usr/bin/env python3
"""
Compile the DEVICE tokenizer (src/tokenizer.c) for the host and diff it against HuggingFace.

scripts/tokenizer_ref.py checks the *algorithm*; this checks the *code that ships*, which is the
transcription risk the Python model cannot cover.

    .venv/bin/python c2c-demos/bearly-smollm/scripts/check_c_tokenizer.py
    .venv/bin/python c2c-demos/bearly-smollm/scripts/check_c_tokenizer.py --corpus CLAUDE.md
"""

import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)

# Text with the cases the pre-tokenizer has to get right: leading/repeated spaces, digits,
# contractions, punctuation runs, tabs and non-ASCII.
DEFAULT_CORPUS = """Hello, world!
What is the capital of France?
  two leading spaces and a digit 1 here
   three spaces then 42 and 3.14159
don't can't it's they're we've I'm we'll he'd
system\tYou are a helpful AI assistant named SmolLM, trained by Hugging Face
user
assistant
punctuation!!! ??? ...---___ (nested [brackets] {here})
emails: foo.bar@example.com; urls: https://example.com/a?b=c#d
unicode: cafe naive uber - em-dash ... ellipsis
UPPER lower MiXeD CamelCase snake_case kebab-case
int8 quantization with group_size=64 gives ~143 MB for 135M parameters.
0123456789
x
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=os.path.join(DEMO, "model", "tokenizer.bin"))
    ap.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--corpus", help="text file to check (default: a built-in corpus)")
    ap.add_argument("--cc", default=os.environ.get("CC", "cc"))
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "tok")
        cmd = [args.cc, "-O2", "-std=c11", "-Wall", "-Wextra", "-D_POSIX_C_SOURCE=200809L",
               "-I", os.path.join(DEMO, "include"),
               os.path.join(DEMO, "src", "tokenizer.c"),
               os.path.join(DEMO, "test", "host_tokenizer_test.c"),
               "-o", exe]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        corpus_path = args.corpus
        if not corpus_path:
            corpus_path = os.path.join(tmp, "corpus.txt")
            with open(corpus_path, "w", encoding="utf-8") as f:
                f.write(DEFAULT_CORPUS)

        out = subprocess.run([exe, args.tokenizer, corpus_path], check=True,
                             capture_output=True, text=True)
        got_lines = out.stdout.splitlines()
        with open(corpus_path, encoding="utf-8", errors="replace") as f:
            texts = [ln.rstrip("\n").rstrip("\r") for ln in f]

    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(args.model_id)

    bad = 0
    for i, text in enumerate(texts):
        want = hf.encode(text, add_special_tokens=False)
        got = [int(t) for t in got_lines[i].split()] if i < len(got_lines) else []
        if want != got:
            bad += 1
            print(f"MISMATCH line {i + 1}: {text!r}\n  hf  {want}\n  C   {got}")
    print(f"C tokenizer: {len(texts) - bad}/{len(texts)} lines exact vs HF")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
