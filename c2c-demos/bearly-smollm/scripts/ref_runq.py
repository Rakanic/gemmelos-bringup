#!/usr/bin/env python3
"""
Host reference for the on-device forward pass, and the differential test against HuggingFace.

This runs the SAME arithmetic `c2c-demos/bearly-smollm/src/main.c` runs — it reads the exported
`model_q80.bin`, dequantizes nothing up front, quantizes activations per group of GS, and
accumulates each group in int32 before scaling — so an on-chip disagreement is a bug in the C
port, not in the export. Products of two int8 values and their 64-wide group sums are all exactly
representable in float32, so the einsum below is bit-identical to int32 accumulation.

    # correctness of the export + runtime, against HF greedy decoding
    .venv/bin/python c2c-demos/bearly-smollm/scripts/ref_runq.py --check --steps 24

    # what the chip should print for a prompt
    .venv/bin/python c2c-demos/bearly-smollm/scripts/ref_runq.py --prompt "What is a transistor?"

    # golden per-layer sums for token 0, as a C header the device can compare against
    .venv/bin/python c2c-demos/bearly-smollm/scripts/ref_runq.py --golden \\
        ../include/smollm_golden.h
"""

import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer_ref import DeviceTokenizer  # noqa: E402

MODEL_MAGIC = 0x616B3432
EXT_MAGIC = 0x534D4C4D

SYSTEM_PROMPT = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


class Model:
    def __init__(self, path):
        blob = np.memmap(path, dtype=np.uint8, mode="r")
        hdr = blob[:256].tobytes()
        magic, version = struct.unpack_from("<Ii", hdr, 0)
        assert magic == MODEL_MAGIC and version == 2, (hex(magic), version)
        (self.dim, self.hidden_dim, self.n_layers, self.n_heads, self.n_kv_heads,
         self.vocab_size, self.seq_len) = struct.unpack_from("<7i", hdr, 8)
        (self.shared_classifier,) = struct.unpack_from("<B", hdr, 36)
        (self.gs,) = struct.unpack_from("<i", hdr, 37)
        ext, self.rope_theta, self.norm_eps, self.head_dim = struct.unpack_from("<Iffi", hdr, 41)
        assert ext == EXT_MAGIC, f"missing SmolLM extension block (got {ext:#x})"
        (self.quant_mode,) = struct.unpack_from("<i", hdr, 57)
        self.kv_dim = self.head_dim * self.n_kv_heads

        off = 256
        n_norm = (2 * self.n_layers + 1) * self.dim
        norms = np.frombuffer(blob[off:off + n_norm * 4].tobytes(), dtype="<f4")
        off += n_norm * 4
        self.rms_att = norms[:self.n_layers * self.dim].reshape(self.n_layers, self.dim)
        self.rms_ffn = norms[self.n_layers * self.dim:2 * self.n_layers * self.dim].reshape(
            self.n_layers, self.dim)
        self.rms_final = norms[2 * self.n_layers * self.dim:]

        def take(count, rows, cols, q4=False, q41=False):
            """Read `count` quantized (rows, cols) tensors: values then fp32 group scales.

            int4 is nibble-packed with weight j in the low nibble of byte j and weight j+gs/2 in
            the high nibble, so unpacking is two masks and a concatenate — the same layout the
            device relies on to avoid a shuffle."""
            nonlocal off
            qs, ss, ms = [], [], []
            n = rows * cols
            nbytes = n // 2 if q4 else n
            for _ in range(count):
                raw = np.frombuffer(blob[off:off + nbytes].tobytes(),
                                    dtype=np.uint8 if q4 else np.int8)
                off += nbytes
                if q4:
                    raw = raw.reshape(rows, cols // self.gs, self.gs // 2)
                    if q41:            # unsigned nibbles; the affine min is carried separately
                        lo = (raw & 0x0F).astype(np.float32)
                        hi = ((raw >> 4) & 0x0F).astype(np.float32)
                    else:              # symmetric: sign-extend the nibble
                        lo = (((raw & 0x0F).astype(np.int8) ^ 8) - 8).astype(np.float32)
                        hi = ((((raw >> 4) & 0x0F).astype(np.int8) ^ 8) - 8).astype(np.float32)
                    q = np.concatenate([lo, hi], axis=2).astype(np.float32)
                else:
                    q = raw.reshape(rows, cols // self.gs, self.gs).astype(np.float32)
                qs.append(q)
                ss.append(np.frombuffer(blob[off:off + (n // self.gs) * 4].tobytes(), dtype="<f4")
                          .reshape(rows, cols // self.gs))
                off += (n // self.gs) * 4
                if q41:
                    ms.append(np.frombuffer(blob[off:off + (n // self.gs) * 4].tobytes(),
                                            dtype="<f4").reshape(rows, cols // self.gs))
                    off += (n // self.gs) * 4
                else:
                    ms.append(None)
            return list(zip(qs, ss, ms))

        d, hd, L = self.dim, self.hidden_dim, self.n_layers
        # 0 = q8, 1/2 = Q4_0 layers/all, 3/4 = Q4_1 layers/all
        q41 = self.quant_mode in (3, 4)
        q4e = self.quant_mode in (2, 4)     # embedding/classifier packed only in the "all" modes
        q4l = self.quant_mode >= 1          # layer matrices
        self.tokens = take(1, self.vocab_size, d, q4e, q41 and q4e)[0]
        self.wq = take(L, d, d, q4l, q41)
        self.wk = take(L, self.kv_dim, d, q4l, q41)
        self.wv = take(L, self.kv_dim, d, q4l, q41)
        self.wo = take(L, d, d, q4l, q41)
        self.w1 = take(L, hd, d, q4l, q41)
        self.w2 = take(L, d, hd, q4l, q41)
        self.w3 = take(L, hd, d, q4l, q41)
        assert self.shared_classifier, "untied classifier is not supported"
        self.wcls = self.tokens
        assert off == blob.size, f"trailing bytes: parsed {off} of {blob.size}"

    # ---- device-identical primitives -------------------------------------------------------

    def quantize(self, x):
        """Per-group symmetric int8 activation quantization (llama2.c runq quantize())."""
        g = x.reshape(-1, self.gs)
        scale = np.abs(g).max(axis=1) / 127.0
        safe = np.where(scale == 0.0, 1.0, scale)
        q = g / safe[:, None]
        q = np.floor(np.abs(q) + 0.5) * np.sign(q)  # C round(): half away from zero
        return q.astype(np.float32), scale.astype(np.float32)

    def matmul(self, x, w):
        """W(d, n) @ x(n) with per-group int accumulation, exactly as the device does it.

        Q4_1 dequantizes affinely (w = q*d + m), so the group contribution gains a second term:
            sum_j w_j x_j = d * sum_j(q_j x_j) + m * sum_j(x_j)
        and sum_j(x_j) depends only on the ACTIVATION, so it is computed once per matmul rather
        than once per output row — which is what keeps Q4_1 nearly as cheap as Q4_0."""
        xq, xs = self.quantize(x)
        wq, ws, wm = w
        acc = np.einsum("dgk,gk->dg", wq, xq, optimize=True)  # exact: |acc| < 2^24
        out = acc * ws
        if wm is not None:
            out = out + wm * xq.sum(axis=1)[None, :]
        return (out * xs[None, :]).sum(axis=1).astype(np.float32)

    def rmsnorm(self, x, weight):
        ss = float((x.astype(np.float32) ** 2).mean()) + self.norm_eps
        return (weight * (x / np.sqrt(ss))).astype(np.float32)

    def embed(self, token):
        q, s, m = self.tokens
        v = q[token] * s[token][:, None]
        if m is not None:
            v = v + m[token][:, None]
        return v.reshape(-1).astype(np.float32)

    # ---- forward ---------------------------------------------------------------------------

    def alloc_kv(self, steps):
        self.kcache = np.zeros((self.n_layers, steps, self.kv_dim), dtype=np.float32)
        self.vcache = np.zeros((self.n_layers, steps, self.kv_dim), dtype=np.float32)

    def forward(self, token, pos, trace=None):
        hs, kv_mul = self.head_dim, self.n_heads // self.n_kv_heads
        x = self.embed(token)
        if trace is not None:
            trace.append(("embed", float(x.sum())))

        for l in range(self.n_layers):
            xb = self.rmsnorm(x, self.rms_att[l])
            q = self.matmul(xb, self.wq[l])
            k = self.matmul(xb, self.wk[l])
            v = self.matmul(xb, self.wv[l])

            # RoPE over adjacent pairs (the export de-interleaved q/k for this convention)
            i = np.arange(0, self.dim, 2)
            freq = 1.0 / (self.rope_theta ** ((i % hs) / np.float32(hs)))
            fcr, fci = np.cos(pos * freq), np.sin(pos * freq)
            q0, q1 = q[0::2].copy(), q[1::2].copy()
            q[0::2], q[1::2] = q0 * fcr - q1 * fci, q0 * fci + q1 * fcr
            nk = self.kv_dim // 2
            k0, k1 = k[0::2].copy(), k[1::2].copy()
            k[0::2] = k0 * fcr[:nk] - k1 * fci[:nk]
            k[1::2] = k0 * fci[:nk] + k1 * fcr[:nk]

            self.kcache[l, pos], self.vcache[l, pos] = k, v

            xb = np.empty(self.dim, dtype=np.float32)
            for h in range(self.n_heads):
                kvh = h // kv_mul
                qh = q[h * hs:(h + 1) * hs]
                ks = self.kcache[l, :pos + 1, kvh * hs:(kvh + 1) * hs]
                att = (ks @ qh) / np.sqrt(np.float32(hs))
                att = np.exp(att - att.max())
                att /= att.sum()
                vs = self.vcache[l, :pos + 1, kvh * hs:(kvh + 1) * hs]
                xb[h * hs:(h + 1) * hs] = att @ vs

            x = x + self.matmul(xb, self.wo[l])
            xb = self.rmsnorm(x, self.rms_ffn[l])
            hb = self.matmul(xb, self.w1[l])
            hb2 = self.matmul(xb, self.w3[l])
            hb = hb * (1.0 / (1.0 + np.exp(-hb))) * hb2
            x = x + self.matmul(hb, self.w2[l])
            if trace is not None:
                trace.append((f"layer{l}", float(x.sum())))

        x = self.rmsnorm(x, self.rms_final)
        logits = self.matmul(x, self.wcls)
        if trace is not None:
            trace.append(("logits", float(logits.sum())))
        return logits


def build_prompt(tok, user, system=SYSTEM_PROMPT):
    """The chat prompt exactly as the device builds it (checked against apply_chat_template)."""
    ids = [tok.bos] + tok.encode("system\n" + system) + [tok.eos] + tok.encode("\n")
    ids += [tok.bos] + tok.encode("user\n" + user) + [tok.eos] + tok.encode("\n")
    ids += [tok.bos] + tok.encode("assistant\n")
    return ids


def generate(model, tok, prompt_ids, steps, echo=True):
    model.alloc_kv(min(len(prompt_ids) + steps, model.seq_len))
    out = []
    token, pos = prompt_ids[0], 0
    limit = min(len(prompt_ids) + steps, model.seq_len)
    while pos < limit:
        logits = model.forward(token, pos)
        pos += 1
        nxt = prompt_ids[pos] if pos < len(prompt_ids) else int(np.argmax(logits))
        if pos >= len(prompt_ids):
            if nxt == tok.eos:
                break
            out.append(nxt)
            if echo:
                sys.stdout.write(tok.pieces[nxt].decode("utf-8", "replace"))
                sys.stdout.flush()
        token = nxt
    if echo:
        print()
    return out


def check(model, tok, model_id, user, steps):
    """Greedy-decode the same prompt through HF fp32 and report where the two streams diverge."""
    import torch
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    ids = build_prompt(tok, user)
    with torch.no_grad():
        gen = hf.generate(torch.tensor([ids]), max_new_tokens=steps, do_sample=False,
                          pad_token_id=tok.eos)
    want = [int(t) for t in gen[0][len(ids):]]
    want = want[:want.index(tok.eos)] if tok.eos in want else want

    got = generate(model, tok, ids, steps, echo=False)

    print(f"prompt  : {user!r} ({len(ids)} tokens)")
    print(f"hf   fp32: {tok.decode(want).decode('utf-8', 'replace')!r}")
    print(f"dev  q8_0: {tok.decode(got).decode('utf-8', 'replace')!r}")
    n = min(len(want), len(got))
    same = next((i for i in range(n) if want[i] != got[i]), n)
    print(f"identical for {same}/{n} tokens"
          + ("" if same == n else f"; first difference: hf {want[same]} "
                                  f"{tok.pieces[want[same]]!r} vs q8 {got[same]} "
                                  f"{tok.pieces[got[same]]!r}"))
    return same


def write_golden(model, tok, path, user):
    """Per-layer running-sum of the residual stream for the prompt's first token."""
    ids = build_prompt(tok, user)
    model.alloc_kv(len(ids) + 1)
    trace = []
    model.forward(ids[0], 0, trace=trace)
    with open(path, "w") as f:
        f.write("/* Generated by c2c-demos/bearly-smollm/scripts/ref_runq.py --golden. Do not edit.\n"
                " * Sum of the residual stream after each stage of forward(token=%d, pos=0), from the\n"
                " * host reference. SMOLLM_DEBUG_GOLDEN=1 makes the device print the same sums, so a\n"
                " * mismatch localizes the diverging layer instead of just \"the output is wrong\". */\n"
                % ids[0])
        f.write("#ifndef SMOLLM_GOLDEN_H\n#define SMOLLM_GOLDEN_H\n\n")
        f.write(f"#define SMOLLM_GOLDEN_TOKEN {ids[0]}\n")
        f.write(f"#define SMOLLM_GOLDEN_STAGES {len(trace)}\n\n")
        f.write("static const float g_smollm_golden_sums[SMOLLM_GOLDEN_STAGES] = {\n")
        for name, val in trace:
            f.write(f"  {val:.6e}f,  /* {name} */\n")
        f.write("};\n\n#endif /* SMOLLM_GOLDEN_H */\n")
    print(f"wrote {path} ({len(trace)} stages, token {ids[0]})")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-bin", default=os.path.join(here, "..", "model", "model_q80.bin"))
    ap.add_argument("--tokenizer", default=os.path.join(here, "..", "model", "tokenizer.bin"))
    ap.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--check", action="store_true", help="diff against HF fp32 greedy decoding")
    ap.add_argument("--ppl", metavar="TEXT_OR_FILE",
                    help="teacher-forced perplexity over this text — the number to compare "
                         "quantization variants with, instead of eyeballing one generation")
    ap.add_argument("--golden", metavar="HEADER", help="write per-layer golden sums to a C header")
    args = ap.parse_args()

    tok = DeviceTokenizer(args.tokenizer)
    model = Model(args.model_bin)
    print(f"config dim={model.dim} hidden={model.hidden_dim} layers={model.n_layers} "
          f"heads={model.n_heads} kv_heads={model.n_kv_heads} vocab={model.vocab_size} "
          f"seq_len={model.seq_len} head_dim={model.head_dim} GS={model.gs} "
          f"rope_theta={model.rope_theta:g} eps={model.norm_eps:g}")

    if args.ppl:
        text = open(args.ppl).read() if os.path.exists(args.ppl) else args.ppl
        ids = tok.encode(text)[:model.seq_len - 1]
        model.alloc_kv(len(ids) + 1)
        nll, cnt = 0.0, 0
        for i in range(len(ids) - 1):
            logits = model.forward(ids[i], i)
            m = logits.max()
            lse = m + np.log(np.exp(logits - m).sum())
            nll += float(lse - logits[ids[i + 1]])
            cnt += 1
        print(f"tokens={cnt} mean_nll={nll / cnt:.4f} perplexity={np.exp(nll / cnt):.3f}")
        return 0
    if args.golden:
        write_golden(model, tok, args.golden, args.prompt)
        return 0
    if args.check:
        return 0 if check(model, tok, args.model_id, args.prompt, args.steps) else 1
    ids = build_prompt(tok, args.prompt)
    print(f"prompt ({len(ids)} tokens): {args.prompt!r}\n")
    generate(model, tok, ids, args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
