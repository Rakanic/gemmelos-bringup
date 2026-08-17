#!/usr/bin/env python3
"""
Export SmolLM2-135M(-Instruct) into the two blobs `c2c-demos/bearly-smollm` embeds into its ELF:

  model_q80.bin   grouped-Q8_0 weights, llama2.c `--version 2` layout + a small SmolLM extension
                  block in the 256-byte header (rope_theta / norm_eps / head_dim).
  tokenizer.bin   byte-level BPE tables in the form the device actually needs: a byte->id map,
                  the merge list as (a, b, ab) id triples in rank order, and one raw-bytes
                  "piece" per token for decoding.

Usage (from the repo root):

    .venv/bin/python c2c-demos/bearly-smollm/scripts/export_smollm.py \
        --out c2c-demos/bearly-smollm/model

Why this exists instead of `llama2.c/export.py --hf`: that path hardcodes
`n_kv_heads = n_heads` (breaks SmolLM's 9/3 GQA), assumes an untied `lm_head`, drops
`rope_theta` (SmolLM uses 100000, not llama2's 10000), and its tokenizer exporter is
SentencePiece-only while SmolLM uses a GPT-2 style byte-level BPE.

Layout of model_q80.bin (all little-endian; offsets are byte offsets from the start):

     0  u32  magic 0x616b3432 ('ak42')
     4  i32  version = 2
     8  i32  dim
    12  i32  hidden_dim
    16  i32  n_layers
    20  i32  n_heads
    24  i32  n_kv_heads
    28  i32  vocab_size
    32  i32  seq_len            (clamped by --seq-len; only sizes the KV cache)
    36  u8   shared_classifier
    37  i32  group_size         (unaligned, exactly as llama2.c writes it)
    41  u32  ext magic 0x534d4c4d ('SMLM')   <- SmolLM extension, ignored by stock runq.c
    45  f32  rope_theta
    49  f32  norm_eps
    53  i32  head_dim
    57       zero pad to 256
   256       fp32 rmsnorm weights: n_layers*dim att, n_layers*dim ffn, dim final
    ...      then per tensor: int8 values, then fp32 group scales, in the order
             q_tokens, wq[L], wk[L], wv[L], wo[L], w1[L], w2[L], w3[L]
             (no wcls: SmolLM ties the classifier to the embedding)
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

MODEL_MAGIC = 0x616B3432  # 'ak42'
EXT_MAGIC = 0x534D4C4D  # 'SMLM'
TOK_MAGIC = 0x534D544B  # 'SMTK'
TOK_VERSION = 1
TOK_HEADER_BYTES = 64


# ----------------------------------------------------------------------------------------------
# model export


def quantize_q80(w, group_size):
    """Symmetric per-group int8 quantization, identical to llama2.c's quantize_q80()."""
    assert w.size % group_size == 0, f"numel {w.size} not a multiple of group_size {group_size}"
    w = w.astype(np.float32).reshape(-1, group_size)
    wmax = np.abs(w).max(axis=1)
    scale = wmax / 127.0
    # a fully-zero group would divide by zero; its quantized values are zero either way
    safe = np.where(scale == 0.0, 1.0, scale)
    q = np.rint(w / safe[:, None]).astype(np.int8)
    err = float(np.abs(q.astype(np.float32) * scale[:, None] - w).max())
    return q.reshape(-1), scale.astype(np.float32), err


def quantize_q40(w, group_size):
    """Symmetric per-group int4, packed two weights per byte.

    Layout inside a group: byte j holds weight j in the LOW nibble and weight j+group/2 in the HIGH
    nibble (llama.cpp's Q4_0 convention). Unpacking then yields two contiguous halves — `v & 0x0F`
    is weights 0..31 and `v >> 4` is weights 32..63 — so the device never has to de-interleave, which
    is what makes a vector kernel practical.

    Scale is max|w|/7, matching the Q8 path's max|w|/127: it gives up the -8 code so the mapping
    stays symmetric, costing ~7% of the available resolution for a much simpler invariant.
    """
    assert group_size % 2 == 0
    w = w.astype(np.float32).reshape(-1, group_size)
    wmax = np.abs(w).max(axis=1)
    scale = wmax / 7.0
    safe = np.where(scale == 0.0, 1.0, scale)
    q = np.clip(np.rint(w / safe[:, None]), -8, 7).astype(np.int8)
    err = float(np.abs(q.astype(np.float32) * scale[:, None] - w).max())
    half = group_size // 2
    packed = ((q[:, :half] & 0x0F) | ((q[:, half:] & 0x0F) << 4)).astype(np.uint8)
    return packed.reshape(-1), scale.astype(np.float32), err


def quantize_q41(w, group_size):
    """Asymmetric per-group int4: w ~= q*d + m, with q an UNSIGNED nibble in [0, 15].

    Q4_0 spends its 16 codes symmetrically around zero, which fits a distribution that is neither
    symmetric nor zero-centred badly — measured on this model, Q4_0 raised perplexity 44.5 -> 68.3.
    Carrying a per-group minimum costs one extra fp32 per 64 weights (0.5625 -> 0.625 bytes/weight,
    so 1.89x -> 1.70x vs int8) and lets the 16 codes span the group's actual range.

    Packing is the same low/high nibble split as Q4_0, so the device still unpacks into two
    contiguous halves and still loads two groups per 64-byte cache line.
    """
    assert group_size % 2 == 0
    w = w.astype(np.float32).reshape(-1, group_size)
    wmin = w.min(axis=1)
    wmax = w.max(axis=1)
    d = (wmax - wmin) / 15.0
    safe = np.where(d == 0.0, 1.0, d)
    q = np.clip(np.rint((w - wmin[:, None]) / safe[:, None]), 0, 15).astype(np.uint8)
    err = float(np.abs(q.astype(np.float32) * d[:, None] + wmin[:, None] - w).max())
    half = group_size // 2
    packed = (q[:, :half] | (q[:, half:] << 4)).astype(np.uint8)
    return packed.reshape(-1), d.astype(np.float32), wmin.astype(np.float32), err


def permute_reverse(w, n_heads, dim1, dim2):
    """Undo the HF q/k rotation so RoPE can be applied to adjacent pairs (llama2.c convention).

    HF stores q_proj/k_proj for the "split half" RoPE (rotate dims [0:h/2] against [h/2:h]);
    llama2.c rotates adjacent pairs (2i, 2i+1). SmolLM's config says rope_interleaved=false,
    i.e. the HF convention, so the weights must be de-interleaved here.
    """
    return w.reshape(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(0, 2, 1, 3).reshape(dim1, dim2)


QUANT_Q8 = 0          # everything int8 — the original format
QUANT_Q4_LAYERS = 1   # layer matrices Q4_0 (symmetric), tied embedding/classifier int8
QUANT_Q4_ALL = 2      # everything Q4_0
QUANT_Q41_LAYERS = 3  # layer matrices Q4_1 (asymmetric: scale + min), classifier int8
QUANT_Q41_ALL = 4     # everything Q4_1


def export_model(model_dir, out_path, seq_len, group_size, quant_mode=QUANT_Q8, quiet=False):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    sd = model.state_dict()

    dim = cfg.hidden_size
    hidden_dim = cfg.intermediate_size
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    vocab_size = cfg.vocab_size
    head_dim = getattr(cfg, "head_dim", None) or dim // n_heads
    kv_dim = head_dim * n_kv_heads
    # transformers >= 5 keeps RoPE settings in a `rope_parameters` dict instead of `config.rope_theta`;
    # read both so the exporter can't silently fall back to llama2's theta of 10000 (SmolLM: 100000).
    rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    rope_type = rope.get("rope_type", "default")
    assert rope_type == "default", f"RoPE type {rope_type!r} is not supported by the device runtime"
    rope_theta = rope.get("rope_theta", getattr(cfg, "rope_theta", None))
    assert rope_theta is not None, "could not determine rope_theta from the HF config"
    rope_theta = float(rope_theta)
    norm_eps = float(cfg.rms_norm_eps)
    max_seq = int(cfg.max_position_embeddings)
    seq_len = min(seq_len, max_seq)

    assert head_dim * n_heads == dim, f"head_dim*{n_heads} != dim ({head_dim}*{n_heads} != {dim})"
    assert cfg.hidden_act == "silu", f"unsupported activation {cfg.hidden_act}"

    def t(name):
        return sd[name].detach().cpu().float().numpy()

    emb = t("model.embed_tokens.weight")
    if "lm_head.weight" in sd:
        shared_classifier = bool(np.array_equal(emb, t("lm_head.weight")))
    else:
        shared_classifier = True  # tied: HF omits lm_head.weight from the state dict entirely
    if not shared_classifier:
        raise SystemExit("untied lm_head is not supported by this exporter (SmolLM ties them)")

    # fp32 rmsnorm weights, in llama2.c order: all attention norms, all ffn norms, final norm
    norms = [t(f"model.layers.{i}.input_layernorm.weight") for i in range(n_layers)]
    norms += [t(f"model.layers.{i}.post_attention_layernorm.weight") for i in range(n_layers)]
    norms += [t("model.norm.weight")]

    # quantized tensors, in llama2.c order
    wq = [permute_reverse(t(f"model.layers.{i}.self_attn.q_proj.weight"), n_heads, dim, dim)
          for i in range(n_layers)]
    wk = [permute_reverse(t(f"model.layers.{i}.self_attn.k_proj.weight"), n_kv_heads, kv_dim, dim)
          for i in range(n_layers)]
    tensors = [emb]
    tensors += wq
    tensors += wk
    tensors += [t(f"model.layers.{i}.self_attn.v_proj.weight") for i in range(n_layers)]
    tensors += [t(f"model.layers.{i}.self_attn.o_proj.weight") for i in range(n_layers)]
    tensors += [t(f"model.layers.{i}.mlp.gate_proj.weight") for i in range(n_layers)]
    tensors += [t(f"model.layers.{i}.mlp.down_proj.weight") for i in range(n_layers)]
    tensors += [t(f"model.layers.{i}.mlp.up_proj.weight") for i in range(n_layers)]

    while any(w.size % group_size for w in tensors):
        group_size //= 2
        assert group_size >= 8, "no workable group size"
        print(f"BACKOFF: reducing group size to {group_size}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Ii", MODEL_MAGIC, 2))
        f.write(struct.pack("<7i", dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len))
        f.write(struct.pack("<B", int(shared_classifier)))
        f.write(struct.pack("<i", group_size))
        f.write(struct.pack("<Iffi", EXT_MAGIC, rope_theta, norm_eps, head_dim))
        f.write(struct.pack("<i", quant_mode))
        pad = 256 - f.tell()
        assert pad >= 0, "header overflowed 256 bytes"
        f.write(b"\0" * pad)

        for w in norms:
            f.write(w.astype("<f4").tobytes())

        # tensor 0 is the tied embedding/classifier; the rest are layer matrices
        all_modes = {QUANT_Q4_ALL, QUANT_Q41_ALL}
        q41_modes = {QUANT_Q41_LAYERS, QUANT_Q41_ALL}
        q4_modes = {QUANT_Q4_LAYERS, QUANT_Q4_ALL}
        worst = 0.0
        for i, w in enumerate(tensors):
            packed = (quant_mode in all_modes) or (quant_mode != QUANT_Q8 and i > 0)
            kind = "int8"
            if packed and quant_mode in q41_modes:
                q, sc, mn, err = quantize_q41(w, group_size)
                f.write(q.tobytes())
                f.write(sc.astype("<f4").tobytes())
                f.write(mn.astype("<f4").tobytes())   # Q4_1 carries a second fp32 per group
                kind = "int4_1"
            elif packed and quant_mode in q4_modes:
                q, sc, err = quantize_q40(w, group_size)
                f.write(q.tobytes())
                f.write(sc.astype("<f4").tobytes())
                kind = "int4_0"
            else:
                q, sc, err = quantize_q80(w, group_size)
                f.write(q.tobytes())
                f.write(sc.astype("<f4").tobytes())
            worst = max(worst, err)
            if not quiet and (i % 30 == 0 or i == len(tensors) - 1):
                print(f"  quantized {i + 1}/{len(tensors)} {tuple(w.shape)} {kind} "
                      f"max_err={err:.6f}")
        size = f.tell()

    # FNV-1a over the finished blob, emitted as a header so the device can prove the 143 MB
    # actually arrived intact over the UART and stays intact in DRAM. Without it, "the weights are
    # corrupt" and "the arithmetic is wrong" look identical on the console.
    h = 1469598103934665603
    with open(out_path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            for b in chunk:
                h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    check_h = os.path.join(os.path.dirname(os.path.abspath(out_path)), "..", "include",
                           "smollm_blob_check.h")
    with open(check_h, "w") as f:
        f.write("/* Generated by scripts/export_smollm.py. Do not edit.\n"
                " * FNV-1a of model_q80.bin, so the device can verify the blob it was flashed with. */\n"
                "#ifndef SMOLLM_BLOB_CHECK_H\n#define SMOLLM_BLOB_CHECK_H\n\n"
                f"#define SMOLLM_MODEL_FNV1A64 0x{h:016x}ULL\n"
                f"#define SMOLLM_MODEL_BYTES   {size}ULL\n\n"
                "#endif\n")
    print(f"wrote {out_path} ({size / 1e6:.1f} MB), worst group quantization error {worst:.6f}")
    print(f"  fnv1a64 = 0x{h:016x} -> {os.path.normpath(check_h)}")
    meta = dict(quant_mode=quant_mode, dim=dim, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
                n_kv_heads=n_kv_heads, vocab_size=vocab_size, seq_len=seq_len, head_dim=head_dim,
                group_size=group_size, rope_theta=rope_theta, norm_eps=norm_eps,
                shared_classifier=shared_classifier, bytes=size)
    print("  " + json.dumps(meta))
    return meta


# ----------------------------------------------------------------------------------------------
# tokenizer export


def bytes_to_unicode():
    """GPT-2's reversible byte <-> unicode-codepoint map (the 'Ġ'/'Ċ' alphabet)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
        list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def export_tokenizer(model_dir, out_path):
    """Emit tokenizer.bin: byte->id table, merge triples in rank order, raw decode pieces.

    The device never manipulates the byte-level unicode strings — encode starts from the byte->id
    table and merges by (a, b) pair rank, decode emits the stored raw bytes. That makes on-device
    tokenization exactly HF's BPE (given the same pre-tokenizer split) with no string compares in
    the merge loop.
    """
    with open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8") as f:
        tok = json.load(f)
    assert tok["model"]["type"] == "BPE", tok["model"]["type"]
    assert not tok["model"].get("ignore_merges", False), "ignore_merges=True is not implemented"
    assert not tok["model"].get("byte_fallback", False), "byte_fallback=True is not implemented"

    vocab = tok["model"]["vocab"]
    merges = tok["model"]["merges"]
    vocab_size = max(vocab.values()) + 1
    id_of = vocab
    piece_of = {i: s for s, i in vocab.items()}

    b2u = bytes_to_unicode()
    u2b = {u: b for b, u in b2u.items()}

    # byte -> token id for the single-byte-level-char tokens. 21 bytes (some control codes and the
    # never-valid UTF-8 lead bytes 0xC0/0xC1/0xF1-0xFF) have no token in this vocab and no byte
    # fallback exists, so they are unencodable; -1 tells the device to drop them, exactly as HF's
    # tokenizer produces nothing for them.
    byte_to_id = [id_of.get(b2u[b], -1) for b in range(256)]
    unencodable = [b for b in range(256) if byte_to_id[b] < 0]
    if unencodable:
        print(f"  {len(unencodable)} bytes have no token and will be dropped on encode: "
              + " ".join(f"0x{b:02x}" for b in unencodable))

    # merge triples (a, b, ab) in rank order; drop any merge whose result is not in the vocab
    triples = []
    for m in merges:
        a, b = (m if isinstance(m, list) else m.split(" "))
        ab = a + b
        if a not in id_of or b not in id_of or ab not in id_of:
            print(f"  skipping unusable merge {a!r}+{b!r}")
            continue
        triples.append((id_of[a], id_of[b], id_of[ab]))

    specials = {t["id"]: t["content"] for t in tok.get("added_tokens", [])}

    with open(os.path.join(model_dir, "tokenizer_config.json"), encoding="utf-8") as f:
        tcfg = json.load(f)
    bos = id_of.get(tcfg.get("bos_token"), -1)
    eos = id_of.get(tcfg.get("eos_token"), -1)

    # raw decode bytes per token: byte-level chars mapped back to bytes; specials stay literal text
    pieces = []
    max_piece = 0
    for i in range(vocab_size):
        s = piece_of.get(i)
        if s is None:
            raw = b""
        elif i in specials:
            raw = specials[i].encode("utf-8")
        else:
            raw = bytes(u2b[ch] for ch in s)
        pieces.append(raw)
        max_piece = max(max_piece, len(raw))

    with open(out_path, "wb") as f:
        f.write(struct.pack("<IiiiiiI", TOK_MAGIC, TOK_VERSION, vocab_size, len(triples),
                            bos, eos, max_piece))
        f.write(b"\0" * (TOK_HEADER_BYTES - f.tell()))
        f.write(np.asarray(byte_to_id, dtype="<i4").tobytes())
        f.write(np.asarray(triples, dtype="<i4").tobytes())
        for raw in pieces:
            f.write(struct.pack("<i", len(raw)))
            f.write(raw)
        size = f.tell()

    print(f"wrote {out_path} ({size / 1e6:.2f} MB): vocab={vocab_size} merges={len(triples)} "
          f"bos={bos} eos={eos} max_piece={max_piece}")
    return dict(vocab_size=vocab_size, merges=len(triples), bos=bos, eos=eos, bytes=size)


# ----------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local dir")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "model"),
                    help="output directory for model_q80.bin / tokenizer.bin")
    ap.add_argument("--seq-len", type=int, default=512,
                    help="max sequence length baked into the header; sizes the on-chip KV cache "
                         "(n_layers*seq_len*kv_dim*2*4 bytes = 46 KB per token for SmolLM-135M)")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--quant", choices=["q8", "q4-layers", "q4-all", "q41-layers", "q41-all"],
                    default="q8",
                    help="q8 = all int8 (143 MB); q4-layers = layer matrices int4 with the tied "
                         "embedding/classifier left at int8 (90 MB); q4-all = everything int4 "
                         "(76 MB). Decode is weight-bandwidth bound, so these ratios are the "
                         "speedup — but the classifier is the most quantization-sensitive tensor "
                         "in the model, which is why q4-layers exists.")
    ap.add_argument("--skip-model", action="store_true", help="only re-export the tokenizer")
    args = ap.parse_args()

    model_dir = args.model
    if not os.path.isdir(model_dir):
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(args.model, allow_patterns=[
            "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "generation_config.json", "model.safetensors"])
        print(f"model dir: {model_dir}")

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    export_tokenizer(model_dir, os.path.join(out_dir, "tokenizer.bin"))
    if not args.skip_model:
        mode = {"q8": QUANT_Q8, "q4-layers": QUANT_Q4_LAYERS, "q4-all": QUANT_Q4_ALL,
                "q41-layers": QUANT_Q41_LAYERS, "q41-all": QUANT_Q41_ALL}[args.quant]
        name = {"q8": "model_q80.bin", "q4-layers": "model_q4l.bin", "q4-all": "model_q4a.bin",
                "q41-layers": "model_q41l.bin", "q41-all": "model_q41a.bin"}[args.quant]
        export_model(model_dir, os.path.join(out_dir, name), args.seq_len, args.group_size, mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
