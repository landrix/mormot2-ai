"""tiny-qwen3/stack: tiny-qwen3 as one stack of a larger checkpoint (backlog B14)

The weights of tests/data/infer-ref/tiny-qwen3 under the prefix "outer.model." in place of
"model.", the head as its own tensor "outer.head.weight" (a copy of the tied embedding), an
unrelated tensor "other.weight", and the same layers again under "broken.model." with one
tensor more (a Q bias the family does not have). config.json holds the settings as objects:

- "outer_config": tiny-qwen3's settings, untied, with the multimodal RoPE of Qwen3-TTS's talker
  (sections 6 + 5 + 5 = head_dim / 2): plain RoPE while the positions are equal
- "outer_config.inner": the same settings again, one level deeper (a dotted path)
- "bad_mrope": sections summing to 15; "zero_mrope": the right sum from a negative and a zero
  section; "scaled": a linear rope_scaling - all refused

Nothing is computed: a stack loaded with CreateStack must give tiny-qwen3's reference
(tensors.bin of tiny-qwen3) bit for bit. Needs numpy only.

    python tools/infer-ref/make_reference_b14.py
"""
import json, os, struct

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "tests", "data", "infer-ref", "tiny-qwen3")
OUT = os.path.join(SRC, "stack")


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    tensors = {}
    for name, t in header.items():
        if name == "__metadata__":
            continue
        a, b = t["data_offsets"]
        tensors[name] = (t["dtype"], t["shape"], data[a:b])
    return tensors


def write_safetensors(path, tensors):
    header, blobs, pos = {}, [], 0
    for name in sorted(tensors):
        dtype, shape, raw = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [pos, pos + len(raw)]}
        blobs.append(raw)
        pos += len(raw)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def main():
    src = read_safetensors(os.path.join(SRC, "model.safetensors"))
    with open(os.path.join(SRC, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    out = {}
    for name, t in src.items():
        assert name.startswith("model."), name
        out["outer." + name] = t
        out["broken." + name] = t
    emb = src["model.embed_tokens.weight"]
    out["outer.head.weight"] = emb
    hidden = cfg["hidden_size"]
    # a bias the family has not: under the layers of the broken stack
    out["broken.model.layers.0.self_attn.q_proj.bias"] = (
        "F32", [cfg["num_attention_heads"] * cfg["head_dim"]],
        b"\0" * 4 * cfg["num_attention_heads"] * cfg["head_dim"])
    out["other.weight"] = ("F32", [3, hidden], struct.pack("<%df" % (3 * hidden),
                                                             *[0.5] * (3 * hidden)))
    inner = {k: v for k, v in cfg.items()
             if k not in ("architectures", "torch_dtype", "model_type")}
    inner["tie_word_embeddings"] = False
    inner["model_type"] = "some_talker"  # not checked for a stack
    half = cfg["head_dim"] // 2
    outer = dict(inner)
    outer["rope_scaling"] = {"interleaved": True, "mrope_section": [6, 5, half - 11],
                             "rope_type": "default", "type": "default"}
    outer["inner"] = dict(inner)
    bad = dict(outer)
    del bad["inner"]
    bad["rope_scaling"] = {"mrope_section": [6, 5, half - 12], "rope_type": "default"}
    zero = dict(bad)
    zero["rope_scaling"] = {"mrope_section": [half + 1, -1, 0], "rope_type": "default"}
    scaled = dict(inner)
    scaled["rope_scaling"] = {"type": "linear", "factor": 2.0}
    config = {"model_type": "some_speech_model", "outer_config": outer,
              "bad_mrope": bad, "zero_mrope": zero, "scaled": scaled}
    os.makedirs(OUT, exist_ok=True)
    write_safetensors(os.path.join(OUT, "model.safetensors"), out)
    with open(os.path.join(OUT, "config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(config, f, indent=1)
        f.write("\n")
    print("written", OUT, len(out), "tensors")


if __name__ == "__main__":
    main()
