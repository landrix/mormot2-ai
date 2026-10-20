#!/usr/bin/env python3
"""Reference data for phase D slice D6 of the native inference engine: Gemma 4.

Targets (all of them when called without arguments):
  tiny-gemma4           tests/data/infer-ref/tiny-gemma4: a seeded 4-layer Gemma 4 in the
                        layout of the published checkpoints (Gemma4ForConditionalGeneration,
                        model_type gemma4, the text settings in text_config, the tensors
                        under model.language_model.): layers sliding (window 5), full,
                        sliding, full; head_dim 16 in the sliding layers, global_head_dim
                        32 with one KV head and attention_k_eq_v (no v_proj) in the full
                        ones; the "proportional" rotary set in the full layers (a quarter
                        of the head rotates); a dense GeGLU block beside 8 experts of which
                        a token takes 3 (enable_moe_block); final_logit_softcapping 3 (it
                        bites on these logits); a layer_scalar per layer; BF16 weights.
                        Every state, the logits, greedy continuations and the rotary
                        factors of both sets for positions 0..63
  tiny-gemma4-dense     tests/data/infer-ref/tiny-gemma4/dense: 6 layers without
                        layer_types and rope_parameters in config.json (the defaults:
                        five sliding, one full; theta 10000 and the proportional set),
                        no experts, attention_k_eq_v false (v_proj in every layer, the
                        full layers keep num_key_value_heads), the same cases

No model download. Same environment, format and rules as make_reference.py; the helpers
of make_reference_d4.py are reused. Every case is checked twice: two runs give the same
bits, and the greedy continuation with the transformers cache equals the one recomputed
without cache. The experts run with the eager implementation (one expert after the
other), as our CPU computes them.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_reference as mr  # noqa: E402
import make_reference_d4 as d4  # noqa: E402
import make_reference_qwen3 as q3  # noqa: E402

TINY_SEED = 20261020
ROPE_POSITIONS = 64
VOCAB = 96


def seeded_weights(model, seed):
    """BF16-exact weights: norms and scales around 1, the router sharp enough
    that its top-k does not hang on a rounding, the layer scalars apart from 1"""
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("router.scale") or \
               name.endswith("per_expert_scale"):
                p.copy_(1.0 + 0.25 * torch.randn_like(p))
            elif name.endswith("router.proj.weight"):
                p.copy_(3.0 * torch.randn_like(p))
            else:
                p.copy_(0.08 * torch.randn_like(p))
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
        for name, b in model.named_buffers():
            if name.endswith("layer_scalar"):
                b.copy_(0.6 + 0.8 * torch.rand_like(b))
                b.copy_(b.to(torch.bfloat16).to(torch.float32))
    model.eval()


def text_model(model):
    return model.model.language_model


def record(model, ids):
    """one forward pass without cache: the input of the first layer, every layer
    output, layer 0 attention and dense MLP output, the final norm and the logits"""
    taken = {}
    hooks = []
    inner = text_model(model)

    def keep(name):
        def hook(_mod, _inp, out):
            taken[name] = (out[0] if isinstance(out, tuple) else out).detach().clone()
        return hook

    hooks.append(inner.embed_tokens.register_forward_hook(keep("embed")))
    for i, layer in enumerate(inner.layers):
        hooks.append(layer.register_forward_hook(keep(f"layer{i:02d}")))
    hooks.append(inner.layers[0].self_attn.register_forward_hook(keep("layer00_attn")))
    hooks.append(inner.layers[0].mlp.register_forward_hook(keep("layer00_mlp")))
    hooks.append(inner.norm.register_forward_hook(keep("norm")))
    try:
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    taken["logits"] = out.logits.detach().clone()
    return {k: v[0].numpy() for k, v in taken.items()}


def router_gap(model, ids, k):
    """the smallest gap between the k-th and the (k+1)-th router logit of any
    token in any layer: the choice must not hang on a rounding"""
    gaps = []
    hooks = []

    def hook(_mod, _inp, out):
        s = torch.sort(out.double(), dim=-1, descending=True).values
        gaps.append(float((s[..., k - 1] - s[..., k]).min()))

    for layer in text_model(model).layers:
        hooks.append(layer.router.proj.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return min(gaps)


def cases(back, seed):
    rng = np.random.default_rng(seed)
    seqs = [rng.integers(3, VOCAB, size=n).tolist() for n in d4.LENGTHS]
    blob = mr.Blob()
    entries = []
    for k, ids in enumerate(seqs):
        a = record(back, ids)
        b = record(back, ids)
        for name in a:
            if not np.array_equal(a[name], b[name]):
                raise ValueError(f"case {k} {name}: two runs differ")
        e = {"case": k, "ids": ids, "tokens": len(ids)}
        blob.add(f"case{k}.ids", np.array(ids, dtype=np.int64))
        for name, v in a.items():
            blob.add(f"case{k}.{name}", v)
        new, margins = d4.greedy(back, ids, d4.GREEDY_TOKENS)
        if q3.greedy_no_cache(back, ids, d4.GREEDY_TOKENS, []) != new:
            raise ValueError(f"case {k}: greedy with and without cache differ")
        blob.add(f"case{k}.greedy", np.array(new, dtype=np.int64))
        blob.add(f"case{k}.greedy_margin", np.array(margins, dtype=np.float32))
        e["greedy"] = new
        e["min_margin"] = min(margins)
        entries.append(e)
    return blob, entries


def rope_factors(rotary, layer_type):
    """cos and sin of positions 0..ROPE_POSITIONS-1 of one layer type ([positions,
    head_dim / 2]: the first half of what the model applies)"""
    pos = torch.arange(ROPE_POSITIONS).unsqueeze(0)
    x = torch.zeros(1, dtype=torch.float32)
    with torch.no_grad():
        cos, sin = rotary(x, pos, layer_type)
    half = cos.shape[-1] // 2
    if not (torch.equal(cos[..., :half], cos[..., half:]) and
            torch.equal(sin[..., :half], sin[..., half:])):
        raise ValueError("the rotary halves differ")
    return cos[0, :, :half].numpy(), sin[0, :, :half].numpy()


def per_layer_form(text):
    """the text settings as save_pretrained writes them: per_layer_config by
    layer index in place of global_head_dim and num_global_key_value_heads"""
    t = dict(text)
    ghd = t.pop("global_head_dim", 512)
    gkv = t.pop("num_global_key_value_heads", None)
    types = t.get("layer_types") or [
        "sliding_attention" if (i + 1) % 6 else "full_attention"
        for i in range(t["num_hidden_layers"])]
    types = list(types)
    types[-1] = "full_attention"
    over = {"head_dim": ghd}
    if t.get("attention_k_eq_v") and gkv is not None:
        over["num_key_value_heads"] = gkv
    t["per_layer_config"] = {str(i): dict(over) for i, k in enumerate(types)
                             if k == "full_attention"}
    return t


def write_folder(model, text, folder):
    """config.json with the text settings in text_config (the form of the
    published checkpoints google/gemma-4-*, or per_layer_config when text has
    one), model.safetensors in BF16 without the tied lm_head"""
    from safetensors.torch import save_file
    os.makedirs(folder, exist_ok=True)
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()
          if k != "lm_head.weight"}
    st = os.path.join(folder, "model.safetensors")
    save_file(sd, st, metadata={"format": "pt"})
    conf = {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4",
            "text_config": dict(text, model_type="gemma4_text", dtype="bfloat16"),
            "tie_word_embeddings": True, "dtype": "bfloat16"}
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(conf, f, indent=1)
        f.write("\n")
    return st


def build(text, seed):
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    cfg = Gemma4Config(text_config=dict(text), tie_word_embeddings=True)
    cfg.text_config._attn_implementation = "eager"
    model = Gemma4ForConditionalGeneration(cfg)
    seeded_weights(model, seed)
    return model


def load_back(folder):
    from transformers import Gemma4ForConditionalGeneration
    back = Gemma4ForConditionalGeneration.from_pretrained(
        folder, dtype=torch.float32, attn_implementation="eager",
        experts_implementation="eager")
    back.eval()
    return back


def check_shapes(back, text):
    """what config.json must give back: the head shapes of every layer"""
    tc = back.config.text_config
    for i, t in enumerate(tc.layer_types):
        a = text_model(back).layers[i].self_attn
        if t == "full_attention":
            want_hd = text.get("global_head_dim", 512)
            kv = text.get("num_global_key_value_heads") \
                if text.get("attention_k_eq_v") else text["num_key_value_heads"]
            if (a.v_proj is None) != bool(text.get("attention_k_eq_v")):
                raise ValueError(f"layer {i}: v_proj and attention_k_eq_v disagree")
        else:
            want_hd = text["head_dim"]
            kv = text["num_key_value_heads"]
        if a.head_dim != want_hd or a.k_proj.out_features != kv * want_hd:
            raise ValueError(f"layer {i}: head {a.head_dim}, k_proj {a.k_proj.out_features}")


def window_acts(folder, back, ids):
    """the reference really slides: the same weights with a window wider than
    the case give other logits"""
    from transformers import AutoConfig, Gemma4ForConditionalGeneration
    c = AutoConfig.from_pretrained(folder)
    c.text_config.sliding_window = 4096
    wide = Gemma4ForConditionalGeneration.from_pretrained(
        folder, config=c, dtype=torch.float32, attn_implementation="eager",
        experts_implementation="eager")
    wide.eval()
    with torch.no_grad():
        a = back(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]
        b = wide(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]
    if torch.equal(a, b):
        raise ValueError("the sliding window changes nothing")
    return float((a - b).abs().max())


def softcap_bites(back, ids, cap):
    """the soft cap changes the logits: some exceed half of it before"""
    taken = {}

    def hook(_mod, _inp, out):
        taken["raw"] = out.detach().clone()

    h = back.lm_head.register_forward_hook(hook)
    try:
        with torch.no_grad():
            back(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        h.remove()
    top = float(taken["raw"].abs().max())
    if top < cap / 2:
        raise ValueError(f"the largest logit {top} is far below the cap {cap}")
    return top


TINY_TEXT = {
    "vocab_size": VOCAB, "hidden_size": 64, "intermediate_size": 64,
    "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
    "head_dim": 16, "global_head_dim": 32, "num_global_key_value_heads": 1,
    "attention_k_eq_v": True, "sliding_window": 5,
    "layer_types": ["sliding_attention", "full_attention", "sliding_attention",
                    "full_attention"],
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25,
                           "rope_theta": 1000000.0}},
    "enable_moe_block": True, "num_experts": 8, "top_k_experts": 3,
    "moe_intermediate_size": 32, "final_logit_softcapping": 3.0,
    "hidden_activation": "gelu_pytorch_tanh", "rms_norm_eps": 1e-6,
    "max_position_embeddings": 512, "hidden_size_per_layer_input": 0,
    "vocab_size_per_layer_input": VOCAB, "num_kv_shared_layers": 0,
    "use_double_wide_mlp": False, "attention_bias": False, "attention_dropout": 0.0,
    "tie_word_embeddings": True, "use_bidirectional_attention": None,
    "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0, "use_cache": True,
    "initializer_range": 0.02,
}

DENSE_TEXT = {
    "vocab_size": VOCAB, "hidden_size": 64, "intermediate_size": 96,
    "num_hidden_layers": 6, "num_attention_heads": 4, "num_key_value_heads": 2,
    "head_dim": 16, "global_head_dim": 32, "num_global_key_value_heads": 1,
    "attention_k_eq_v": False, "sliding_window": 5,
    "enable_moe_block": False, "final_logit_softcapping": 3.0,
    "hidden_activation": "gelu_pytorch_tanh", "rms_norm_eps": 1e-6,
    "max_position_embeddings": 512, "hidden_size_per_layer_input": 0,
    "vocab_size_per_layer_input": VOCAB, "num_kv_shared_layers": 0,
    "use_double_wide_mlp": False, "attention_bias": False, "attention_dropout": 0.0,
    "tie_word_embeddings": True, "use_bidirectional_attention": None,
    "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0, "use_cache": True,
    "initializer_range": 0.02,
}


def tiny_gemma4():
    print("tiny Gemma 4")
    folder = os.path.join(mr.OUT, "tiny-gemma4")
    # the first seed whose router choice keeps a gap of 0.02 in every case
    seed = TINY_SEED
    rng = np.random.default_rng(TINY_SEED)
    probe = [rng.integers(3, VOCAB, size=n).tolist() for n in d4.LENGTHS]
    while True:
        model = build(TINY_TEXT, seed)
        gap = min(router_gap(model, ids, TINY_TEXT["top_k_experts"]) for ids in probe)
        if gap >= 0.02:
            break
        seed += 1
        if seed > TINY_SEED + 200:
            raise ValueError("no seed with a clear router choice")
    st = write_folder(model, TINY_TEXT, folder)
    back = load_back(folder)
    if back.config.text_config._experts_implementation != "eager":
        raise ValueError("experts not eager")
    check_shapes(back, TINY_TEXT)
    blob, entries = cases(back, TINY_SEED)
    longest = entries[-1]["ids"]
    effect = window_acts(folder, back, longest)
    top = softcap_bites(back, longest, TINY_TEXT["final_logit_softcapping"])
    rot = text_model(back).rotary_emb
    for name, kind in (("rope", "sliding_attention"), ("rope2", "full_attention")):
        cos, sin = rope_factors(rot, kind)
        blob.add(f"{name}.cos", cos)
        blob.add(f"{name}.sin", sin)
    manifest = {
        "dataset": "tiny-gemma4",
        "note": "seeded Gemma 4 (Gemma4ForConditionalGeneration, text only), 4 layers: "
                "0 and 2 sliding (window 5, head_dim 16, 2 KV heads, rope theta 10000), "
                "1 and 3 full (global_head_dim 32, 1 KV head, attention_k_eq_v: no "
                "v_proj, proportional rope theta 1e6 with a quarter of the head "
                "rotating); hidden 64, 4 heads, a dense GeGLU of 64 beside 8 experts of "
                "width 32, 3 per token (router: RMSNorm without weight times scale and "
                "hidden^-0.5, softmax, top 3 renormed, times per_expert_scale); "
                "final_logit_softcapping 3; layer_scalar between 0.6 and 1.4; tied head; "
                "BF16 weights. The contract is model.safetensors + config.json in the "
                "form of the published checkpoints (text_config, tensors under "
                "model.language_model., no lm_head). Per case (5, 13, 29 tokens): "
                "embed, layerNN, layer00_attn, layer00_mlp (the dense block), norm, "
                "logits (all positions, soft-capped), greedy (24 tokens, no eos) with "
                "the top-2 margin - checked equal with the transformers cache and "
                "without. rope.cos / rope.sin [64, 8]: the sliding set; rope2.cos / "
                "rope2.sin [64, 16]: the full set (pairs 4..15 do not rotate)",
        "config": json.load(open(os.path.join(folder, "config.json"), encoding="utf-8")),
        "safetensors_sha256": mr.sha256_file(st),
        "environment": mr.environment(),
        "seed": seed,
        "router_gap": gap,
        "window_effect": effect,
        "largest_raw_logit": top,
        "cases": entries,
    }
    blob.write(folder, manifest)


def tiny_gemma4_dense():
    print("tiny Gemma 4, dense with the default layer types and rope")
    folder = os.path.join(mr.OUT, "tiny-gemma4", "dense")
    model = build(DENSE_TEXT, TINY_SEED + 1)
    # as save_pretrained writes it: per_layer_config, no global_head_dim
    st = write_folder(model, per_layer_form(DENSE_TEXT), folder)
    back = load_back(folder)
    tc = back.config.text_config
    if tc.layer_types != ["sliding_attention"] * 5 + ["full_attention"]:
        raise ValueError(f"default layer types {tc.layer_types}")
    check_shapes(back, DENSE_TEXT)
    blob, entries = cases(back, TINY_SEED + 1)
    effect = window_acts(folder, back, entries[-1]["ids"])
    rot = text_model(back).rotary_emb
    for name, kind in (("rope", "sliding_attention"), ("rope2", "full_attention")):
        cos, sin = rope_factors(rot, kind)
        blob.add(f"{name}.cos", cos)
        blob.add(f"{name}.sin", sin)
    manifest = {
        "dataset": "tiny-gemma4/dense",
        "note": "seeded Gemma 4, 6 layers from the defaults (no layer_types: five "
                "sliding, the last full; no rope_parameters: sliding theta 10000, "
                "full proportional theta 1e6 with a quarter rotating), window 5, no "
                "experts, intermediate 96, attention_k_eq_v false (v_proj in every "
                "layer; the full layers keep 2 KV heads with a head of 32), config.json "
                "as save_pretrained writes it (per_layer_config {\"5\": {\"head_dim\": "
                "32}} in place of global_head_dim), "
                "final_logit_softcapping 3, BF16 weights; the cases of tiny-gemma4",
        "config": json.load(open(os.path.join(folder, "config.json"), encoding="utf-8")),
        "safetensors_sha256": mr.sha256_file(st),
        "environment": mr.environment(),
        "window_effect": effect,
        "cases": entries,
    }
    blob.write(folder, manifest)


TARGETS = {"tiny-gemma4": tiny_gemma4, "tiny-gemma4-dense": tiny_gemma4_dense}

if __name__ == "__main__":
    want = sys.argv[1:] or list(TARGETS)
    for t in want:
        if t not in TARGETS:
            raise SystemExit(f"unknown target {t}; known: {', '.join(TARGETS)}")
    torch.set_num_threads(1)
    for t in want:
        TARGETS[t]()
