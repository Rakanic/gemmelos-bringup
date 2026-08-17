#!/usr/bin/env python3
"""
Bit-exact Python model of the tokenizer the DEVICE runs, plus a differential test against HF.

`c2c-demos/bearly-smollm/src/main.c` encodes text with the algorithm implemented here, reading
the same `tokenizer.bin` this file parses. Keeping the reference here means the C can be checked
against HuggingFace on a corpus without a chip in the loop — run:

    .venv/bin/python c2c-demos/bearly-smollm/scripts/tokenizer_ref.py --check

Pre-tokenization is the only approximation. HF applies Digits(individual_digits) and then the
GPT-2 ByteLevel regex

    's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+

which `pretokenize()` reproduces with byte classes, treating every byte >= 0x80 as a letter
(\\p{L} for real letters; wrong for emoji/CJK punctuation, which then merge slightly differently).
Everything after the split — the merge order — is exact, because the export writes merges as
(a, b, ab) id triples in rank order and both sides merge the lowest-rank adjacent pair.
"""

import argparse
import os
import struct
import sys

TOK_MAGIC = 0x534D544B
TOK_HEADER_BYTES = 64

SPACE = 0
LETTER = 1
DIGIT = 2
OTHER = 3

CONTRACTIONS = (b"'s", b"'t", b"'re", b"'ve", b"'m", b"'ll", b"'d")


def cls(b):
    if b in (0x20, 0x09, 0x0A, 0x0B, 0x0C, 0x0D):
        return SPACE
    if (0x41 <= b <= 0x5A) or (0x61 <= b <= 0x7A) or b >= 0x80:
        return LETTER
    if 0x30 <= b <= 0x39:
        return DIGIT
    return OTHER


def pretokenize(text):
    """Split raw bytes into pre-token chunks (list of bytes objects)."""
    s = text.encode("utf-8") if isinstance(text, str) else text
    n = len(s)
    out = []
    i = 0
    while i < n:
        c = cls(s[i])

        if s[i] == 0x27:  # "'" — contractions win over the punctuation run
            hit = next((x for x in CONTRACTIONS if s[i:i + len(x)] == x), None)
            if hit:
                out.append(hit)
                i += len(hit)
                continue

        # " ?\p{L}+" / " ?\p{N}+" / " ?[^\s\p{L}\p{N}]+": at most ONE literal space (0x20, never \n
        # or \t) may lead the run, and the run is a single character class.
        # The Digits(individual_digits) pre-tokenizer needs no special case: it was applied when the
        # tokenizer was TRAINED, so no vocab token ever holds two digits and BPE leaves a digit run
        # split into single digits regardless.
        j = i + 1 if (s[i] == 0x20 and i + 1 < n and cls(s[i + 1]) != SPACE) else i
        if j < n and cls(s[j]) != SPACE:
            want = cls(s[j])
            k = j
            while k < n and cls(s[k]) == want:
                k += 1
            out.append(s[i:k])
            i = k
            continue

        # whitespace run: "\s+(?!\S)|\s+" — a run that is followed by a non-space gives its last
        # character up to the next chunk (which is why " a" is one token but "  a" is two).
        k = i
        while k < n and cls(s[k]) == SPACE:
            k += 1
        if k < n and k - i > 1:
            k -= 1
        out.append(s[i:k])
        i = k
    return out


class DeviceTokenizer:
    def __init__(self, path):
        blob = open(path, "rb").read()
        magic, ver, self.vocab_size, n_merges, self.bos, self.eos, self.max_piece = \
            struct.unpack_from("<IiiiiiI", blob, 0)
        assert magic == TOK_MAGIC and ver == 1, (hex(magic), ver)
        off = TOK_HEADER_BYTES
        self.byte_to_id = list(struct.unpack_from(f"<256i", blob, off))
        off += 256 * 4
        trip = struct.unpack_from(f"<{n_merges * 3}i", blob, off)
        off += n_merges * 3 * 4
        # (a, b) -> (rank, ab); rank is the position in the merge list
        self.merges = {(trip[3 * r], trip[3 * r + 1]): (r, trip[3 * r + 2]) for r in range(n_merges)}
        self.pieces = []
        for _ in range(self.vocab_size):
            (ln,) = struct.unpack_from("<i", blob, off)
            off += 4
            self.pieces.append(blob[off:off + ln])
            off += ln
        assert off == len(blob), (off, len(blob))

    def encode_chunk(self, chunk, out):
        ids = [self.byte_to_id[b] for b in chunk]
        ids = [i for i in ids if i >= 0]  # unencodable bytes are dropped, as in HF
        while len(ids) > 1:
            best_rank, best_at = None, -1
            for i in range(len(ids) - 1):
                m = self.merges.get((ids[i], ids[i + 1]))
                if m is not None and (best_rank is None or m[0] < best_rank):
                    best_rank, best_at = m[0], i
            if best_at < 0:
                break
            ids[best_at:best_at + 2] = [self.merges[(ids[best_at], ids[best_at + 1])][1]]
        out.extend(ids)

    def encode(self, text):
        out = []
        for chunk in pretokenize(text):
            self.encode_chunk(chunk, out)
        return out

    def decode(self, ids):
        return b"".join(self.pieces[i] for i in ids)


CORPUS = [
    "Hello, world!",
    "Once upon a time, there was a little robot named Bolt.",
    "What is the capital of France?",
    "  leading and trailing spaces  ",
    "Write a haiku about RISC-V vector extensions.",
    "The answer is 42 and 3.14159 and 1000000.",
    "def matmul(a, b):\n    return a @ b\n",
    "tabs\tand\nnewlines\r\n\r\nand   multiple   spaces",
    "don't can't it's they're we've I'm we'll he'd",
    "UPPER lower MiXeD CamelCase snake_case kebab-case",
    "emails: foo.bar@example.com; urls: https://example.com/a?b=c#d",
    "unicode: café naïve über Ωμέγα 日本語 — em-dash … ellipsis",
    "punctuation!!! ??? ...---___ (nested [brackets] {here})",
    "A single a.  B ends with digit 7",
    "0123456789",
    "",
    " ",
    "\n",
    "x",
    "system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face",
    "user\nTell me a story about a brave little toaster.",
    "assistant\n",
    "The DSP 25 chip runs at 500 MHz and has 256 MB of DRAM.",
    "int8 quantization with group_size=64 gives ~143 MB for 135M parameters.",
]


def check(tok_path, model_dir, corpus_file=None):
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(model_dir)
    dev = DeviceTokenizer(tok_path)

    corpus = list(CORPUS)
    if corpus_file:
        with open(corpus_file, encoding="utf-8", errors="replace") as f:
            corpus += [ln.rstrip("\n") for ln in f]

    bad = 0
    for text in corpus:
        want = hf.encode(text, add_special_tokens=False)
        got = dev.encode(text)
        if want != got:
            bad += 1
            print(f"MISMATCH {text!r}\n  hf  {want}\n      {[hf.decode([t]) for t in want]}"
                  f"\n  dev {got}\n      {[dev.pieces[t].decode('utf-8', 'replace') for t in got]}")
        rt = dev.decode(got).decode("utf-8", "replace")
        if got and rt != text:
            print(f"ROUNDTRIP {text!r} -> {rt!r}")
    print(f"encode: {len(corpus) - bad}/{len(corpus)} exact vs HF")

    # the chat prompt the device builds, checked against HF's own chat template
    msgs = [{"role": "user", "content": "What is the capital of France?"}]
    want = hf.apply_chat_template(msgs, add_generation_prompt=True)
    if not isinstance(want, list):  # transformers >= 5 returns a BatchEncoding, not a token list
        want = list(want["input_ids"])
    if want and isinstance(want[0], list):
        want = want[0]
    got = ([dev.bos] + dev.encode("system\nYou are a helpful AI assistant named SmolLM, "
                                  "trained by Hugging Face") + [dev.eos] + dev.encode("\n")
           + [dev.bos] + dev.encode("user\n" + msgs[0]["content"]) + [dev.eos] + dev.encode("\n")
           + [dev.bos] + dev.encode("assistant\n"))
    print(f"chat template: {'MATCH' if want == got else 'MISMATCH'}")
    if want != got:
        print(f"  hf  {want}\n  dev {got}")
        bad += 1
    return 1 if bad else 0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=os.path.join(here, "..", "model", "tokenizer.bin"))
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--check", action="store_true", help="differential test against HF")
    ap.add_argument("--encode", help="encode one string and print the tokens")
    ap.add_argument("--corpus-file", help="extra text file; every line is checked against HF")
    args = ap.parse_args()

    if args.encode is not None:
        dev = DeviceTokenizer(args.tokenizer)
        ids = dev.encode(args.encode)
        print(ids)
        print([dev.pieces[i].decode("utf-8", "replace") for i in ids])
        return 0
    return check(args.tokenizer, args.model, args.corpus_file)


if __name__ == "__main__":
    sys.exit(main())
