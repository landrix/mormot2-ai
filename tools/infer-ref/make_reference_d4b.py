"""Device test fixture of backlog D4b - no download, no reference run.

tests/data/infer-ref/tiny-qwen3-sliding/g16: the config of tiny-qwen3-sliding (a window of 5
on layer 1) with 16 query heads on one KV head - a GQA group of 16, more than the 8 queries
of attention.comp, which a window layer of a flash.comp prefill must still respect. Random
BF16 weights (seeded); the tests compare the GPU with the CPU on this model, so no
transformers output is stored.

    python tools/infer-ref/make_reference_d4b.py
"""
import json
import os
import struct

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SRC = os.path.join(ROOT, "tests", "data", "infer-ref", "tiny-qwen3-sliding")
OUT = os.path.join(SRC, "g16")

HIDDEN, HEADS, KV, HD, FFN, VOCAB, LAYERS = 64, 16, 1, 16, 160, 96, 2


def bf16(a):
    """float32 -> BF16 bits, rounded to nearest even"""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def main():
    rng = np.random.default_rng(20261019)
    cfg = json.load(open(os.path.join(SRC, "config.json"), encoding="utf-8"))
    cfg["num_attention_heads"] = HEADS
    cfg["num_key_value_heads"] = KV
    tensors = {"model.embed_tokens.weight": rng.normal(0, 0.5, (VOCAB, HIDDEN))}
    for l in range(LAYERS):
        p = "model.layers.%d." % l
        tensors[p + "input_layernorm.weight"] = 1 + rng.normal(0, 0.1, HIDDEN)
        tensors[p + "post_attention_layernorm.weight"] = 1 + rng.normal(0, 0.1, HIDDEN)
        tensors[p + "self_attn.q_proj.weight"] = rng.normal(0, 0.15, (HEADS * HD, HIDDEN))
        tensors[p + "self_attn.k_proj.weight"] = rng.normal(0, 0.15, (KV * HD, HIDDEN))
        tensors[p + "self_attn.v_proj.weight"] = rng.normal(0, 0.15, (KV * HD, HIDDEN))
        tensors[p + "self_attn.o_proj.weight"] = rng.normal(0, 0.1, (HIDDEN, HEADS * HD))
        tensors[p + "self_attn.q_norm.weight"] = 1 + rng.normal(0, 0.1, HD)
        tensors[p + "self_attn.k_norm.weight"] = 1 + rng.normal(0, 0.1, HD)
        tensors[p + "mlp.gate_proj.weight"] = rng.normal(0, 0.15, (FFN, HIDDEN))
        tensors[p + "mlp.up_proj.weight"] = rng.normal(0, 0.15, (FFN, HIDDEN))
        tensors[p + "mlp.down_proj.weight"] = rng.normal(0, 0.1, (HIDDEN, FFN))
    tensors["model.norm.weight"] = 1 + rng.normal(0, 0.1, HIDDEN)
    # safetensors: a little-endian u64 header size, the JSON header, the data
    header, blobs, offset = {}, [], 0
    for name in sorted(tensors):
        data = bf16(tensors[name]).tobytes()
        header[name] = {"dtype": "BF16", "shape": list(np.shape(tensors[name])),
                        "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for b in blobs:
            f.write(b)
    with open(os.path.join(OUT, "config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(cfg, f, indent=1)
        f.write("\n")
    print("written:", OUT, offset, "bytes of weights")


if __name__ == "__main__":
    main()
