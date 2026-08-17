#!/usr/bin/env python3
"""
Compile the DEVICE forward pass (src/model.c + src/tokenizer.c) for the host and check it against
the numpy reference, which is itself checked against HuggingFace.

This is the closest thing to a chip that runs in a minute: same source files, same blob, same
arithmetic. It catches transcription bugs in the C that scripts/ref_runq.py cannot, because that
one models the algorithm rather than the code.

    .venv/bin/python c2c-demos/bearly-smollm/scripts/check_c_forward.py
    .venv/bin/python c2c-demos/bearly-smollm/scripts/check_c_forward.py --prompt "Why is the sky blue?"
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-bin", default=os.path.join(DEMO, "model", "model_q80.bin"))
    ap.add_argument("--tokenizer", default=os.path.join(DEMO, "model", "tokenizer.bin"))
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--cc", default=os.environ.get("CC", "cc"))
    ap.add_argument("--tol", type=float, default=2e-3, help="relative tolerance on stage sums")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "fwd")
        cmd = [args.cc, "-O2", "-std=c11", "-Wall", "-Wextra",
               "-I", os.path.join(DEMO, "include"),
               os.path.join(DEMO, "src", "model.c"),
               os.path.join(DEMO, "src", "tokenizer.c"),
               os.path.join(DEMO, "test", "host_forward_test.c"),
               "-lm", "-o", exe]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        run = subprocess.run([exe, args.model_bin, args.tokenizer, args.prompt, str(args.steps)],
                             check=True, capture_output=True, text=True,
                             env={**os.environ, "SMOLLM_STAGE_POS": "0"})

    c_stages, c_prompt, c_out = {}, [], ""
    for line in run.stdout.splitlines():
        if line.startswith("STAGE "):
            _, idx, name, val = line.split(maxsplit=3)
            c_stages[int(idx)] = (name, float(val))
        elif line.startswith("PROMPT "):
            c_prompt = [int(t) for t in line.split(":", 1)[1].split()]
        elif line.startswith("OUTPUT "):
            c_out = line[len("OUTPUT "):]
        elif line.startswith("STEP ") or line.startswith("TOKENS"):
            pass
        else:
            print("  " + line)

    # the numpy reference, on the same prompt
    from ref_runq import Model, build_prompt, generate
    from tokenizer_ref import DeviceTokenizer
    tok = DeviceTokenizer(args.tokenizer)
    model = Model(args.model_bin)
    ref_prompt = build_prompt(tok, args.prompt)
    trace = []
    model.alloc_kv(len(ref_prompt) + args.steps)
    model.forward(ref_prompt[0], 0, trace=trace)
    ref_out = tok.decode(generate(model, tok, ref_prompt, args.steps, echo=False)) \
        .decode("utf-8", "replace")

    bad = 0
    if c_prompt != ref_prompt:
        print(f"PROMPT MISMATCH\n  C   {c_prompt}\n  ref {ref_prompt}")
        bad += 1
    else:
        print(f"prompt: {len(c_prompt)} tokens, identical")

    worst, worst_at = 0.0, ""
    for i, (name, val) in enumerate(trace):
        if i not in c_stages:
            print(f"stage {i} ({name}) missing from the C output")
            bad += 1
            continue
        cname, cval = c_stages[i]
        rel = abs(cval - val) / max(abs(val), 1e-6)
        if rel > worst:
            worst, worst_at = rel, f"{i} {name}"
        if rel > args.tol:
            print(f"STAGE {i} {name}: C {cval:.6e} vs ref {val:.6e}  rel {rel:.3e}  <-- DIVERGES")
            bad += 1
    print(f"stages: {len(trace)} compared, worst relative difference {worst:.3e} at stage {worst_at}")

    print(f"C   output: {c_out!r}")
    print(f"ref output: {ref_out!r}")
    # The token streams are NOT expected to stay identical, and that is not a failure. int8
    # activation quantization is a step function: a ~1e-6 float difference (the C accumulates
    # matmul groups and attention sequentially, numpy sums them pairwise) eventually lands one
    # activation on the other side of a rounding boundary. Measured at position 1 of this prompt:
    # every op through layer 13's attention agrees to ~1e-6, then the int8 activations feeding
    # `wo` sum to -45 in C and -44 in the reference — ONE value off by one — and that grows to a
    # few percent by layer 28, which is enough to flip a near-tie between tokens. Both evaluations
    # are equally valid; what must hold is stage-exactness at position 0, where attention has a
    # single term and nothing amplifies.
    if c_out.strip() != ref_out.strip():
        print("note: token streams differ (expected — see the comment in this script); "
              "compare the text for sense, not for equality")

    print("PASS" if not bad else f"FAIL ({bad} problems)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
