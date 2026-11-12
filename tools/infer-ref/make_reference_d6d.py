#!/usr/bin/env python3
"""Reference data for phase D slice D6d of the native inference engine: Gemma 4 from GGUF.

Targets (all of them when called without arguments):
  tiny-gemma4-gguf      tests/data/infer-ref/tiny-gemma4/gguf: both tiny models of D6a
                        (tiny-gemma4 with experts, K=V and the proportional RoPE;
                        tiny-gemma4/dense) as llama.cpp's converter writes the
                        architecture gemma4 (conversion/gemma.py: Gemma4Model; the keys
                        and tensors of ggml-org/gemma-4-26B-A4B-it-GGUF), in BF16 and in
                        Q8_0 as llama-quantize writes it, plus damaged files - each one
                        refused by its own check of the loader
  gemma4-tokens-12b     tests/data/infer-ref/gemma4-tokens/12b: the tokenizer of
                        google/gemma-4-12B-it @ 43c9d01 (the file of its GGUF: 239 merges
                        fewer than 4d7ae49), one case per dropped merge - needs that
                        tokenizer.json in $LAI_GEMMA4_12B_TOKENIZER (sha256 pinned) and
                        the 26B-A4B one in $LAI_GEMMA4_DIR (default
                        vendor/models/gemma-4-26B-A4B-it)
  gemma4-tokenizer-gguf vendor/models/gemma-4-26B-A4B-it/tokenizer.gguf (gitignored): the
                        metadata of ggml-org/gemma-4-26B-A4B-it-GGUF (the Q8_0 file at its
                        pinned revision, read by an HTTP range: the first 40 MiB) without
                        tensors - the tokenizer of the published file for the machines
                        without its 27 GB; sha256 pinned

No model download, no transformers run: the tiny files are written from the tiny
folders, whose values they are expected to give (tests/data/infer-ref/tiny-gemma4); the
token ids come from the tokenizers library.
Followed, written anew (no code taken): conversion/gemma.py (Gemma3Model.modify_tensors,
Gemma4Model.set_gguf_parameters / generate_extra_tensors / filter_tensors /
modify_tensors), conversion/base.py (prepare_tensors: the weights in float32 before any
change, 1-D tensors, norms and the router in F32) and src/llama-arch.cpp of the llama.cpp
commit 2a4daf0e08d6a9b6993ee0d4a9a3cfc35a6f405c.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_reference as mr  # noqa: E402

LLAMA_CPP_COMMIT = "2a4daf0e08d6a9b6993ee0d4a9a3cfc35a6f405c"

# Hugging Face names under model.layers.N. and llama.cpp's under blk.N.
LAYER_NAMES = [
    ("input_layernorm.weight", "attn_norm.weight"),
    ("post_attention_layernorm.weight", "post_attention_norm.weight"),
    ("pre_feedforward_layernorm.weight", "ffn_norm.weight"),
    ("post_feedforward_layernorm.weight", "post_ffw_norm.weight"),
    ("post_feedforward_layernorm_1.weight", "post_ffw_norm_1.weight"),
    ("pre_feedforward_layernorm_2.weight", "pre_ffw_norm_2.weight"),
    ("post_feedforward_layernorm_2.weight", "post_ffw_norm_2.weight"),
    ("layer_scalar", "layer_output_scale.weight"),
    ("self_attn.q_proj.weight", "attn_q.weight"),
    ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.v_proj.weight", "attn_v.weight"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("self_attn.q_norm.weight", "attn_q_norm.weight"),
    ("self_attn.k_norm.weight", "attn_k_norm.weight"),
    ("mlp.gate_proj.weight", "ffn_gate.weight"),
    ("mlp.up_proj.weight", "ffn_up.weight"),
    ("mlp.down_proj.weight", "ffn_down.weight"),
    ("router.proj.weight", "ffn_gate_inp.weight"),
    ("router.scale", "ffn_gate_inp.scale"),
    ("router.per_expert_scale", "ffn_down_exps.scale"),
    ("experts.gate_up_proj", "ffn_gate_up_exps.weight"),
    ("experts.down_proj", "ffn_down_exps.weight"),
]


def text_settings(conf):
    """the text settings with what transformers derives: layer_types (five sliding,
    one full, the last full), the rotary sets, the full head dim and KV heads"""
    t = dict(conf.get("text_config", conf))
    n = t["num_hidden_layers"]
    types = list(t.get("layer_types") or [
        "sliding_attention" if (i + 1) % 6 else "full_attention" for i in range(n)])
    types[-1] = "full_attention"
    t["layer_types"] = types
    rp = t.get("rope_parameters") or {}
    t["rope_sliding"] = rp.get("sliding_attention", {"rope_type": "default",
                                                     "rope_theta": 10000.0})
    t["rope_full"] = rp.get("full_attention", {"rope_type": "proportional",
                                               "partial_rotary_factor": 0.25,
                                               "rope_theta": 1000000.0})
    ghd = t.get("global_head_dim")
    gkv = t.get("num_global_key_value_heads")
    for i, over in (t.get("per_layer_config") or {}).items():
        if types[int(i)] == "full_attention":
            ghd = over.get("head_dim", ghd)
            gkv = over.get("num_key_value_heads", gkv)
    t["full_head_dim"] = ghd
    t["full_kv_heads"] = gkv
    return t


def load_folder(folder):
    """the tensors of the text model under model. as float32 numpy (BF16 widened)"""
    from safetensors.torch import load_file
    sd = {}
    for k, v in load_file(os.path.join(folder, "model.safetensors")).items():
        if not k.startswith("model.language_model."):
            continue  # nothing else in the tiny folders, a tower in the real ones
        sd["model." + k[len("model.language_model."):]] = v.to(torch.float32).numpy()
    return sd


def gguf_tensors(g, t, sd, matrix):
    """the tensors as the converter writes them: no norm shifted (Gemma 4 multiplies by
    the weight as it is), the router scale and per_expert_scale renamed, ".weight"
    appended to layer_scalar and the expert tensors, rope_freqs generated; 1-D tensors,
    norms and ffn_gate_inp.weight in F32, every other matrix in BF16 (the bits of the
    float32 values: exact) or in Q8_0 from ggml's reference quantizer (matrix 'bf16'
    or 'q8', as llama-quantize Q8_0 writes token_embd and the experts too)"""
    ts = []

    def put(name, a):
        a = np.ascontiguousarray(a, np.float32)
        ne = list(reversed(a.shape))
        f32 = a.ndim <= 1 or name.endswith("_norm.weight") or \
            name.endswith("ffn_gate_inp.weight") or not name.endswith(".weight")
        if f32:
            ts.append((name, 0, ne, a.tobytes()))
        elif matrix == "q8":
            rows = a.reshape(-1, a.shape[-1])
            ts.append((name, 8, ne, b"".join(g.quant_ref(r) for r in rows)))
        else:
            ts.append((name, 30, ne, (a.view(np.uint32) >> 16).astype(np.uint16).tobytes()))

    put("token_embd.weight", sd["model.embed_tokens.weight"])
    put("output_norm.weight", sd["model.norm.weight"])
    # rope_freqs: 1 for the first int(head * partial / 2) pairs of a full head, 1e30
    # for the others (generate_extra_tensors), used by the full layers only
    full = t["full_head_dim"]
    rot = int(full * t["rope_full"]["partial_rotary_factor"] / 2)
    put("rope_freqs.weight", np.array([1.0] * rot + [1e30] * (full // 2 - rot), np.float32))
    for i in range(t["num_hidden_layers"]):
        p = f"model.layers.{i}."
        for hf, gg in LAYER_NAMES:
            if p + hf in sd:
                put(f"blk.{i}.{gg}", sd[p + hf])
    return ts


def gguf_kvs(t, name):
    """the keys of ggml-org/gemma-4-26B-A4B-it-GGUF for the text settings t"""
    a = "gemma4."
    n = t["num_hidden_layers"]
    swa = [k == "sliding_attention" for k in t["layer_types"]]
    kvs = [("general.architecture", "str", "gemma4"),
           ("general.name", "str", name),
           (a + "block_count", "u32", n),
           (a + "context_length", "u32", t.get("max_position_embeddings", 131072)),
           (a + "embedding_length", "u32", t["hidden_size"]),
           (a + "feed_forward_length", "u32", t["intermediate_size"]),
           (a + "attention.head_count", "u32", t["num_attention_heads"])]
    kv = t["num_key_value_heads"]
    if t["full_kv_heads"] is not None:
        # an array only when the full layers name their own KV heads
        kvs.append((a + "attention.head_count_kv", "arr",
                    ("i32", [kv if s else t["full_kv_heads"] for s in swa])))
    else:
        kvs.append((a + "attention.head_count_kv", "u32", kv))
    kvs += [(a + "rope.freq_base", "f32", t["rope_full"]["rope_theta"]),
            (a + "rope.freq_base_swa", "f32", t["rope_sliding"]["rope_theta"]),
            (a + "attention.layer_norm_rms_epsilon", "f32", t["rms_norm_eps"])]
    if t.get("enable_moe_block"):
        kvs += [(a + "expert_count", "u32", t["num_experts"]),
                (a + "expert_used_count", "u32", t["top_k_experts"])]
    kvs += [(a + "attention.key_length", "u32", t["full_head_dim"]),
            (a + "attention.value_length", "u32", t["full_head_dim"])]
    if t.get("final_logit_softcapping"):
        kvs.append((a + "final_logit_softcapping", "f32", t["final_logit_softcapping"]))
    kvs += [(a + "attention.sliding_window", "u32", t["sliding_window"]),
            (a + "attention.shared_kv_layers", "u32", t.get("num_kv_shared_layers", 0)),
            (a + "embedding_length_per_layer_input", "u32",
             t.get("hidden_size_per_layer_input") or 0),
            (a + "attention.sliding_window_pattern", "arr", ("bool", swa)),
            (a + "attention.key_length_swa", "u32", t["head_dim"]),
            (a + "attention.value_length_swa", "u32", t["head_dim"])]
    if t.get("enable_moe_block"):
        kvs.append((a + "expert_feed_forward_length", "u32", t["moe_intermediate_size"]))
    kvs += [(a + "rope.dimension_count", "u32", t["full_head_dim"]),
            (a + "rope.dimension_count_swa", "u32", t["head_dim"])]
    return kvs


def tiny_gemma4_gguf():
    print("tiny Gemma 4 as GGUF")
    g = mr.Ggml()
    base = os.path.join(mr.OUT, "tiny-gemma4")
    folder = os.path.join(base, "gguf")
    os.makedirs(folder, exist_ok=True)

    def source(sub):
        src = os.path.join(base, sub)
        conf = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
        return text_settings(conf), load_folder(src)

    moe, msd = source("")
    dense, dsd = source("dense")
    km = gguf_kvs(moe, "tiny-gemma4")
    kd = gguf_kvs(dense, "tiny-gemma4-dense")
    mb = gguf_tensors(g, moe, msd, "bf16")
    db = gguf_tensors(g, dense, dsd, "bf16")
    files = {"bf16.gguf": (km, mb),
             "q8_0.gguf": (km, gguf_tensors(g, moe, msd, "q8")),
             "dense-bf16.gguf": (kd, db),
             "dense-q8_0.gguf": (kd, gguf_tensors(g, dense, dsd, "q8"))}

    def kv(kvs, key, t, v):
        out = [k for k in kvs if k[0] != key]
        return out + [(key, t, v)] if v is not None else out

    # damaged: each one fails a different check of the loader; those refused while
    # the settings are read carry only the embedding, the last norm and rope_freqs
    a = "gemma4."
    head = [x for x in mb if x[0] in ("token_embd.weight", "output_norm.weight",
                                      "rope_freqs.weight")]
    n = moe["num_hidden_layers"]
    files["bad-arch.gguf"] = (kv(km, "general.architecture", "str", "gemma3"), head)
    files["bad-ple.gguf"] = (kv(km, a + "embedding_length_per_layer_input", "u32", 8), head)
    files["bad-shared.gguf"] = (kv(km, a + "attention.shared_kv_layers", "u32", 2), head)
    files["bad-ffn.gguf"] = (kv(km, a + "feed_forward_length", "arr",
                                ("u32", [moe["intermediate_size"]] * n)), head)
    files["bad-scaling.gguf"] = (km + [(a + "rope.scaling.type", "str", "yarn")], head)
    files["bad-rope.gguf"] = (kv(km, a + "rope.dimension_count_swa", "u32", 8), head)
    files["bad-pattern.gguf"] = (kv(km, a + "attention.sliding_window_pattern", "arr",
                                    ("bool", [True, False, True])), head)
    full = moe["full_head_dim"]
    freqs = np.array([1.0] * (full // 2), np.float32)
    freqs[1] = 1e30  # a rotating pair after one that does not
    files["bad-freqs.gguf"] = (km, [x if x[0] != "rope_freqs.weight" else
                                    (x[0], 0, x[2], freqs.tobytes()) for x in head])
    files["bad-head.gguf"] = (km, mb + [("output.weight",) + tuple(
        x[1:]) for x in mb if x[0] == "token_embd.weight"])
    files["bad-missing.gguf"] = (km, [x for x in mb if x[0] != "blk.1.ffn_down_exps.scale"])
    files["bad-extra.gguf"] = (km, mb + [("blk.0.ffn_gate_shexp.weight",) + tuple(x[1:])
                                         for x in mb if x[0] == "blk.0.ffn_gate.weight"])
    files["bad-swa-v.gguf"] = (km, [x for x in mb if x[0] != "blk.2.attn_v.weight"])
    # a full head of 44 with 15 rotating pairs (partial_rotary_factor 0.7 as the
    # converter counts it: int(44 * 0.7 / 2)): 2 * 15 / 44 * 44 is 29.999... in double -
    # the loader must still rotate 15 pairs; random weights, only the settings are read
    rng = np.random.default_rng(20261112)

    def bf16(shape):
        a = (0.08 * rng.standard_normal(shape)).astype(np.float32)
        return (a.view(np.uint32) >> 16).astype(np.uint16).tobytes()

    wide = 44
    hd = moe["hidden_size"]
    nh = moe["num_attention_heads"]
    full_layers = [i for i, k in enumerate(moe["layer_types"]) if k == "full_attention"]
    shapes = {"attn_q.weight": [hd, nh * wide], "attn_k.weight": [hd, wide],
              "attn_output.weight": [nh * wide, hd]}
    r44 = []
    for x in mb:
        name = x[0]
        if name == "rope_freqs.weight":
            x = (name, 0, [wide // 2],
                 np.array([1.0] * 15 + [1e30] * 7, np.float32).tobytes())
        elif name.startswith("blk.") and int(name.split(".")[1]) in full_layers:
            kind = name.split(".", 2)[2]
            if kind in shapes:
                ne = shapes[kind]
                x = (name, 30, ne, bf16(list(reversed(ne))))
            elif kind in ("attn_q_norm.weight", "attn_k_norm.weight"):
                x = (name, 0, [wide], np.ones(wide, np.float32).tobytes())
        r44.append(x)
    k44 = kv(kv(kv(km, a + "attention.key_length", "u32", wide), a + "attention.value_length",
                "u32", wide), a + "rope.dimension_count", "u32", wide)
    files["rope44.gguf"] = (k44, r44)
    # a tokenizer alone: the five controls, the 256 byte tokens, five pieces, one added
    # control; bad: one more byte token that is no <0xHH>
    sp = chr(0x2581)
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"]
    tokens += [f"<0x{b:02X}>" for b in range(256)]
    tokens += ["a", "b", "ab", sp, sp + "a", "<|turn>"]
    types = [3] * 5 + [6] * 256 + [1] * 5 + [3]
    tk = [("general.architecture", "str", "gemma4"),
          ("tokenizer.ggml.model", "str", "gemma4"),
          ("tokenizer.ggml.tokens", "arr", ("str", tokens)),
          ("tokenizer.ggml.scores", "arr", ("f32", [-1000.0] * len(tokens))),
          ("tokenizer.ggml.token_type", "arr", ("i32", types)),
          ("tokenizer.ggml.merges", "arr", ("str", ["a b", sp + " a"])),
          ("tokenizer.ggml.bos_token_id", "u32", 2),
          ("tokenizer.ggml.eos_token_id", "u32", 1),
          ("tokenizer.ggml.add_space_prefix", "bool", False),
          ("tokenizer.ggml.add_bos_token", "bool", True)]
    files["tokenizer.gguf"] = (tk, [])
    for name, extra in (("bad-tokenizer.gguf", "<0xZZ>"), ("bad-tokenizer-x.gguf", "<0X41>")):
        bad = [(k, t, v) for k, t, v in tk if k not in ("tokenizer.ggml.tokens",
                                                         "tokenizer.ggml.scores",
                                                         "tokenizer.ggml.token_type")]
        bad += [("tokenizer.ggml.tokens", "arr", ("str", tokens + [extra])),
                ("tokenizer.ggml.scores", "arr", ("f32", [-1000.0] * (len(tokens) + 1))),
                ("tokenizer.ggml.token_type", "arr", ("i32", types + [6]))]
        files[name] = (bad, [])
    for name, (k, t) in files.items():
        mr.gguf_write(os.path.join(folder, name), k, t)
    manifest = {"dataset": "tiny-gemma4/gguf", "environment": mr.environment(),
                "ggml": {"dir": os.path.relpath(mr.GGML_DIR, mr.ROOT), "sha256": g.libs},
                "llama_cpp_commit": LLAMA_CPP_COMMIT,
                "source": "tiny-gemma4, tiny-gemma4/dense",
                "note": "tiny-gemma4 (bf16, q8_0) and tiny-gemma4/dense (dense-bf16, "
                        "dense-q8_0) as llama.cpp's converter writes gemma4, with the "
                        "keys of ggml-org/gemma-4-26B-A4B-it-GGUF: head_count_kv an "
                        "array per layer where the full layers have their own KV heads "
                        "(tiny-gemma4), a number otherwise (dense); "
                        "sliding_window_pattern; key_length / rope.dimension_count the "
                        "full head, *_swa the sliding one; rope_freqs 1 for the rotating "
                        "pairs of the full head, 1e30 for the others; no attn_v in the "
                        "full layers of tiny-gemma4 (attention_k_eq_v); the norms as "
                        "they are, the router scale as ffn_gate_inp.scale, "
                        "per_expert_scale as ffn_down_exps.scale, layer_scalar as "
                        "layer_output_scale.weight, gate_up_proj and down_proj as "
                        "ffn_gate_up_exps / ffn_down_exps; 1-D tensors, norms and the "
                        "router in F32, the matrices in BF16 (the bits of the folder: "
                        "exact) or Q8_0 (quantize_row_q8_0_ref row by row, token_embd "
                        "and the experts too). The expected values are those of the "
                        "source folders. bad-* (the first nine with the embedding, the "
                        "last norm and rope_freqs only): another architecture, "
                        "per-layer embeddings, shared KV layers, a feed-forward width "
                        "per layer, a rope scaling, a sliding rotary count of 8, a "
                        "sliding_window_pattern of 3 layers, rope_freqs with a rotating "
                        "pair after one that does not, an output.weight (an untied "
                        "head), a missing blk.1.ffn_down_exps.scale, an extra "
                        "blk.0.ffn_gate_shexp.weight, a sliding layer 2 without "
                        "attn_v. rope44: tiny-gemma4 with a full head of 44 (random "
                        "weights in the full layers) and 15 rotating pairs - the count, "
                        "not a share, must reach the RoPE (2 * 15 / 44 * 44 is "
                        "29.999... in double). tokenizer: a gemma4 tokenizer alone (the "
                        "five controls, the 256 byte tokens, a, b, ab, U+2581, U+2581a, "
                        "<|turn>; merges a b, U+2581 a); bad-tokenizer, bad-tokenizer-x: "
                        "one more byte token <0xZZ>, <0X41> (the decoder reads <0x only)",
                "files": {n: mr.sha256_file(os.path.join(folder, n)) for n in sorted(files)}}
    with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")


GEMMA4_12B_TOKENIZER_REVISION = "43c9d014b231474fa7212dbb48a762566c2b255d"
GEMMA4_12B_TOKENIZER_SHA256 =     "9788bdb6a021b3170357cd3587b5d622b1177561ad3c5bf43d73728fd71aeacb"


def plain_tokenizer(path):
    """tokenizer.json without its added tokens: every text is text"""
    from tokenizers import Tokenizer
    with open(path, encoding="utf-8") as f:
        t = json.load(f)
    t["added_tokens"] = []
    return Tokenizer.from_str(json.dumps(t))


def tokens_12b():
    """the tokenizer.json of google/gemma-4-12B-it after the fix of its 210 glitch
    tokens (revision 43c9d01, the one ggml-org/gemma-4-12B-it-GGUF was converted from):
    the vocabulary of 4d7ae49 (26B-A4B), 239 merges fewer, order kept - none of the
    3197 cases of gemma4-tokens changes. One case per dropped merge: the text of its
    two parts, ids of the fixed file (and those of the old one, which joined them)"""
    print("Gemma 4 12B tokenizer cases (the dropped merges)")
    old_path = os.path.join(os.environ.get(
        "LAI_GEMMA4_DIR", os.path.join(mr.ROOT, "vendor", "models", "gemma-4-26B-A4B-it")),
        "tokenizer.json")
    new_path = os.environ.get("LAI_GEMMA4_12B_TOKENIZER", "")
    if not new_path or mr.sha256_file(new_path) != GEMMA4_12B_TOKENIZER_SHA256:
        raise SystemExit("LAI_GEMMA4_12B_TOKENIZER: the tokenizer.json of "
                         f"google/gemma-4-12B-it @ {GEMMA4_12B_TOKENIZER_REVISION} "
                         f"(sha256 {GEMMA4_12B_TOKENIZER_SHA256})")
    old, new = plain_tokenizer(old_path), plain_tokenizer(new_path)
    with open(old_path, encoding="utf-8") as f:
        om = [tuple(m) for m in json.load(f)["model"]["merges"]]
    with open(new_path, encoding="utf-8") as f:
        nm = [tuple(m) for m in json.load(f)["model"]["merges"]]
    kept = set(nm)
    dropped = [m for m in om if m not in kept]
    if [m for m in om if m in kept] != nm or len(dropped) != 239:
        raise ValueError("the fixed merges are not the old ones minus 239")
    flat, offsets, cases = [], [], []
    text_bytes, text_offsets = bytearray(), []
    changed = 0
    for i, (left, right) in enumerate(dropped):
        text = (left + right).replace(chr(0x2581), " ")
        ids = new.encode(text, add_special_tokens=False).ids
        before = old.encode(text, add_special_tokens=False).ids
        if new.decode(ids, skip_special_tokens=False) != text:
            raise ValueError(f"merge {i}: decode differs")
        offsets.append(len(flat))
        flat += ids
        text_offsets.append(len(text_bytes))
        text_bytes += text.encode("utf-8")
        e = {"id": f"dropped_{i:03d}", "text": text, "tokens": len(ids)}
        if before != ids:
            e["old_ids"] = before
            changed += 1
        cases.append(e)
    offsets.append(len(flat))
    text_offsets.append(len(text_bytes))
    blob = mr.Blob()
    blob.add("texts", np.frombuffer(bytes(text_bytes), dtype=np.uint8),
             "UTF-8 of all case texts concatenated")
    blob.add("text_offsets", np.array(text_offsets, dtype=np.int64),
             "case i = texts[text_offsets[i] .. text_offsets[i+1]-1]")
    blob.add("ids", np.array(flat, dtype=np.int64), "all cases concatenated")
    blob.add("offsets", np.array(offsets, dtype=np.int64),
             "case i = ids[offsets[i] .. offsets[i+1]-1]")
    manifest = {
        "dataset": "gemma4-tokens/12b",
        "note": "the tokenizer.json of google/gemma-4-12B-it @ 43c9d01 (its 210 glitch "
                "tokens fixed: 239 merges of 4d7ae49 dropped, the vocabulary and every "
                "other setting the same; ggml-org/gemma-4-12B-it-GGUF holds these "
                "tables). One case per dropped merge: the text of its two parts "
                "(U+2581 as a space), the ids of the fixed file without its added "
                "tokens, no special tokens added; old_ids where the old file gives "
                "others (it joins the parts)",
        "model": {"id": "google/gemma-4-12B-it",
                  "revision": GEMMA4_12B_TOKENIZER_REVISION,
                  "files": {"tokenizer.json": GEMMA4_12B_TOKENIZER_SHA256}},
        "environment": mr.environment(),
        "changed": changed,
        "cases": cases,
    }
    blob.write(os.path.join(mr.OUT, "gemma4-tokens", "12b"), manifest)
    print(f"  {len(cases)} cases, {changed} with other ids than the old file")


GGUF_26B_URL = ("https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/resolve/"
                "1ca99aae748b4b01f609fa51eb23682ee5f00b85/gemma-4-26B-A4B-it-Q8_0.gguf")
GGUF_26B_METADATA_SHA256 = "e947d307c2a5471d292f8ab145f2ae878d71ee0d1e425738d1df00897484659c"


def metadata_end(head):
    """the offset after the last key/value pair of a GGUF v3 header in head"""
    import struct
    pos = 24
    if head[:4] != b"GGUF" or struct.unpack_from("<I", head, 4)[0] != 3:
        raise ValueError("no GGUF v3 file")
    n_kv = struct.unpack_from("<Q", head, 16)[0]
    size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip_value(t, pos):
        if t == 8:
            return pos + 8 + struct.unpack_from("<Q", head, pos)[0]
        if t == 9:
            et, n = struct.unpack_from("<IQ", head, pos)
            pos += 12
            if et == 8:
                for _ in range(n):
                    pos = skip_value(8, pos)
                return pos
            return pos + n * size[et]
        return pos + size[t]

    for _ in range(n_kv):
        pos += 8 + struct.unpack_from("<Q", head, pos)[0]
        t = struct.unpack_from("<I", head, pos)[0]
        pos = skip_value(t, pos + 4)
    return n_kv, pos


def tokenizer_gguf():
    """vendor/models/gemma-4-26B-A4B-it/tokenizer.gguf: the metadata of
    ggml-org/gemma-4-26B-A4B-it-GGUF (Q8_0, pinned revision) without its tensors - the
    tokenizer of a published GGUF file where the 27 GB are not (gitignored, 16 MB)"""
    import urllib.request
    print("Gemma 4 tokenizer GGUF (the metadata of the published file)")
    req = urllib.request.Request(GGUF_26B_URL, headers={"Range": "bytes=0-41943039"})
    with urllib.request.urlopen(req) as r:
        head = r.read()
    n_kv, end = metadata_end(head)
    if end > len(head):
        raise ValueError("the metadata is longer than the range read")
    import struct
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, n_kv) + head[24:end]
    out += b"\0" * ((-len(out)) % 32)
    import hashlib
    sha = hashlib.sha256(out).hexdigest()
    if sha != GGUF_26B_METADATA_SHA256:
        raise ValueError(f"metadata sha256 {sha}, expected {GGUF_26B_METADATA_SHA256}")
    folder = os.environ.get("LAI_GEMMA4_DIR",
                            os.path.join(mr.ROOT, "vendor", "models", "gemma-4-26B-A4B-it"))
    path = os.path.join(folder, "tokenizer.gguf")
    with open(path, "wb") as f:
        f.write(out)
    print(f"  {path}: {n_kv} keys, {len(out)} bytes, sha256 {sha}")


TARGETS = {"tiny-gemma4-gguf": tiny_gemma4_gguf, "gemma4-tokens-12b": tokens_12b,
           "gemma4-tokenizer-gguf": tokenizer_gguf}

if __name__ == "__main__":
    want = sys.argv[1:] or list(TARGETS)
    for w in want:
        if w not in TARGETS:
            raise SystemExit(f"unknown target {w}; known: {', '.join(TARGETS)}")
    for w in want:
        TARGETS[w]()
