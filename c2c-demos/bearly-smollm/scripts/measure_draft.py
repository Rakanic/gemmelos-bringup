#!/usr/bin/env python3
"""
Measure the ACCEPTANCE RATE of cheap draft strategies for speculative decoding.

Speculative decoding verifies K drafted tokens in one batched forward pass. On this board that
costs ~1.11 normal decodes (one weight pass + K x arithmetic), so the speedup is

    expected_accepted_tokens / 1.11

and the whole idea lives or dies on how often a draft is right. This script measures that against
the model's OWN greedy output — which is what verification compares to — so the number it prints
is directly the p in the table above.

The constraint that rules out most drafts: decode is memory-bound at ~31.6 cycles per weight byte,
so any draft that reads a model from DRAM costs another pass and cancels the benefit. Only drafts
that fit in cache (a table of tens/hundreds of KB) or cost nothing at all (n-grams over the text
generated so far) are worth measuring.

    .venv/bin/python c2c-demos/bearly-smollm/scripts/measure_draft.py
"""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROMPTS = [
    "What is the capital of France?",
    "Why is the sky blue?",
    "Write one sentence about a robot.",
    "Explain what a transistor does.",
    "Tell me a short story about a lighthouse.",
    "How do I make a cup of tea?",
    "What is 2 + 2?",
    "Describe the ocean in two sentences.",
    "Who wrote Hamlet?",
    "Give me three tips for studying.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--steps", type=int, default=48, help="tokens generated per prompt")
    ap.add_argument("--ngram", type=int, default=2, help="context length for the lookup draft")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).eval()

    # 1. Generate the reference greedy stream for each prompt — this is exactly what the device
    #    would produce, and what a draft has to match.
    runs = []
    for p in PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt")
        ids = (enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "keys") else enc)
        ids = [int(t) for t in (ids[0] if ids.dim() > 1 else ids)]
        with torch.no_grad():
            out = model.generate(torch.tensor([ids]), max_new_tokens=args.steps,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        runs.append((list(ids), [int(t) for t in out[0][len(ids):]]))
        print(f"  {p!r} -> {len(runs[-1][1])} tokens")

    # 2. Build a "distilled bigram": the model's most likely next token for each single token,
    #    computed once on the host. On-device this is a 49152 x 2-byte table = 96 KB, small enough
    #    to stay resident, so drafting costs a single load.
    print("building the bigram table (one batched forward over the vocabulary) ...")
    V = model.config.vocab_size
    bigram = torch.empty(V, dtype=torch.int32)
    with torch.no_grad():
        for lo in range(0, V, 1024):
            hi = min(lo + 1024, V)
            ctx = torch.arange(lo, hi, dtype=torch.long).unsqueeze(1)
            logits = model(ctx).logits[:, -1, :]
            bigram[lo:hi] = logits.argmax(-1).to(torch.int32)

    # 3. Score each strategy against the reference streams.
    stats = defaultdict(lambda: [0, 0])   # name -> [hits, total]

    for prompt_ids, gen in runs:
        history = list(prompt_ids)
        for i, actual in enumerate(gen):
            prev = history[-1]

            # (a) distilled bigram: argmax P(next | last token)
            stats["bigram"][1] += 1
            stats["bigram"][0] += int(bigram[prev].item() == actual)

            # (b) prompt/history lookup: find the last n tokens earlier in the context and propose
            #     whatever followed them (costs nothing but a search over a few hundred tokens)
            n = args.ngram
            pred = None
            if len(history) > n:
                needle = history[-n:]
                for j in range(len(history) - n - 1, -1, -1):
                    if history[j:j + n] == needle:
                        pred = history[j + n]
                        break
            stats[f"lookup-{n}"][1] += 1
            stats[f"lookup-{n}"][0] += int(pred == actual)

            # (c) both: prefer the lookup when it fires, else the bigram
            stats["lookup+bigram"][1] += 1
            stats["lookup+bigram"][0] += int((pred if pred is not None
                                              else bigram[prev].item()) == actual)

            history.append(actual)

    print("\nacceptance rate against the model's own greedy output:")
    print(f"{'strategy':<16} {'accept':>8}   speedup at K=3 (verify = 1.11 decodes)")
    for name, (hit, tot) in sorted(stats.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
        p = hit / max(tot, 1)
        expected = 1 + p + p * p + p ** 3       # tokens per verify with 3 drafts
        print(f"{name:<16} {p * 100:7.1f}%   {expected / 1.11:5.2f}x")
    print("\n(a strategy needs ~30% to be worth the device work; below that the verify overhead "
          "eats the gain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
