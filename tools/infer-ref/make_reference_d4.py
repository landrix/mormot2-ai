#!/usr/bin/env python3
"""Reference data for phase D slice D4 of the native inference engine: sliding-window
attention and layers without rotary positions (NoPE).

Targets (all of them when called without arguments):
  tiny-smollm3          tests/data/infer-ref/tiny-smollm3: a seeded 4-layer SmolLM3 whose
                        layers cover every combination - 0 full with RoPE, 1 sliding
                        without RoPE, 2 sliding with RoPE, 3 full without RoPE (window 6,
                        explicit no_rope_layers and layer_types), BF16 weights; every
                        state, the logits and greedy continuations that wrap the window
                        several times
  tiny-qwen3-sliding    tests/data/infer-ref/tiny-qwen3-sliding: a seeded 2-layer Qwen3
                        with use_sliding_window, sliding_window 5 and max_window_layers 1
                        and no layer_types in config.json (the window of layer 1 is
                        derived as Qwen3Config does)

No model download. Same environment, format and rules as make_reference.py; the helpers
of make_reference_qwen3.py are reused. Every case is checked twice: two runs give the
same bits, and the greedy continuation with the transformers cache (a sliding-window
cache for the sliding layers) equals the one recomputed without cache.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_reference as mr  # noqa: E402
import make_reference_qwen3 as q3  # noqa: E402

TINY_SEED = 20261019
GREEDY_TOKENS = 24
LENGTHS = (5, 13, 29)  # below one window, two windows, several


def seeded_weights(model, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.dim() == 1:  # the RMSNorm weights start as ones: make them count
                p.copy_(1.0 + 0.25 * torch.randn_like(p))
            else:
                p.copy_(0.08 * torch.randn_like(p))
            p.copy_(p.to(torch.bfloat16).to(torch.float32))  # BF16-exact
    model.eval()


def write_folder(model, cfg, folder, keep, arch):
    from safetensors.torch import save_file
    os.makedirs(folder, exist_ok=True)
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()
          if k != "lm_head.weight"}
    st = os.path.join(folder, "model.safetensors")
    save_file(sd, st, metadata={"format": "pt"})
    d = cfg.to_dict()
    d["architectures"] = [arch]
    d["torch_dtype"] = "bfloat16"
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({k: d[k] for k in keep + ["architectures", "torch_dtype"] if k in d},
                  f, indent=1)
        f.write("\n")
    return st


def cases(back, vocab, seed):
    rng = np.random.default_rng(seed)
    seqs = [rng.integers(3, vocab, size=n).tolist() for n in LENGTHS]
    blob = mr.Blob()
    entries = []
    for k, ids in enumerate(seqs):
        a = q3.record(back, ids)
        b = q3.record(back, ids)
        for name in a:
            if not np.array_equal(a[name], b[name]):
                raise ValueError(f"case {k} {name}: two runs differ")
        e = {"case": k, "ids": ids, "tokens": len(ids)}
        blob.add(f"case{k}.ids", np.array(ids, dtype=np.int64))
        for name, v in a.items():
            blob.add(f"case{k}.{name}", v)
        # no eos: the continuation runs on past several windows
        new, margins = greedy(back, ids, GREEDY_TOKENS)
        if q3.greedy_no_cache(back, ids, GREEDY_TOKENS, []) != new:
            raise ValueError(f"case {k}: greedy with and without cache differ")
        blob.add(f"case{k}.greedy", np.array(new, dtype=np.int64))
        blob.add(f"case{k}.greedy_margin", np.array(margins, dtype=np.float32))
        e["greedy"] = new
        e["min_margin"] = min(margins)
        entries.append(e)
    return blob, entries


def greedy(model, ids, steps):
    """greedy continuation with the transformers cache and no eos (it runs on
    past several windows), plus each step's top-2 margin"""
    model.generation_config.eos_token_id = None
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids),
                             dtype=torch.long), do_sample=False, max_new_tokens=steps,
                             min_new_tokens=steps, pad_token_id=0, output_logits=True,
                             return_dict_in_generate=True, temperature=None, top_p=None,
                             top_k=None)
    new = out.sequences[0, len(ids):].tolist()
    if len(new) != steps:
        raise ValueError(f"{len(new)} of {steps} tokens")
    margins = []
    for step, logits in enumerate(out.logits):
        top = torch.topk(logits[0].double(), 2)
        if int(top.indices[0]) != new[step]:
            raise ValueError("generate did not pick the argmax")
        margins.append(float(top.values[0] - top.values[1]))
    return new, margins


def window_acts(cls, folder, back, ids):
    """the reference really slides: the same weights with every layer in full
    attention give other logits for the longest case"""
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(folder)
    c.layer_types = ["full_attention"] * c.num_hidden_layers
    c.use_sliding_window = False
    full = cls.from_pretrained(folder, config=c, dtype=torch.float32,
                               attn_implementation="eager")
    full.eval()
    with torch.no_grad():
        a = back(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]
        b = full(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]
    if torch.equal(a, b):
        raise ValueError("the sliding window changes nothing")
    return float((a - b).abs().max())


# ---------------------------------------------------------------- tiny SmolLM3

def tiny_smollm3():
    print("tiny SmolLM3")
    from transformers import SmolLM3Config, SmolLM3ForCausalLM
    cfg = SmolLM3Config(
        vocab_size=96, hidden_size=64, intermediate_size=160, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        rms_norm_eps=1e-6, rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=True, hidden_act="silu", bos_token_id=1, eos_token_id=2,
        pad_token_id=0, use_sliding_window=True, sliding_window=6,
        no_rope_layers=[1, 0, 1, 0],
        layer_types=["full_attention", "sliding_attention", "sliding_attention",
                     "full_attention"],
        attention_bias=False, mlp_bias=False, attn_implementation="eager")
    model = SmolLM3ForCausalLM(cfg)
    seeded_weights(model, TINY_SEED)
    names = [n for n, _ in model.named_parameters()]
    if any(n.endswith(".bias") or "q_norm" in n for n in names):
        raise ValueError("a bias or a per-head norm: not the SmolLM3 layout")
    for i, want in enumerate([True, False, True, False]):
        if bool(model.model.layers[i].self_attn.use_rope) != want:
            raise ValueError(f"layer {i}: use_rope is not {want}")
    folder = os.path.join(mr.OUT, "tiny-smollm3")
    keep = ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
            "rms_norm_eps", "rope_theta", "rope_parameters", "tie_word_embeddings",
            "hidden_act", "bos_token_id", "eos_token_id", "pad_token_id", "model_type",
            "use_sliding_window", "sliding_window", "no_rope_layers",
            "no_rope_layer_interval", "layer_types", "attention_bias", "mlp_bias",
            "rope_scaling"]
    st = write_folder(model, cfg, folder, keep, "SmolLM3ForCausalLM")
    back = SmolLM3ForCausalLM.from_pretrained(folder, dtype=torch.float32,
                                              attn_implementation="eager")
    back.eval()
    if back.config.layer_types != cfg.layer_types or \
       back.config.no_rope_layers != cfg.no_rope_layers:
        raise ValueError("config.json does not keep the layer kinds")
    blob, entries = cases(back, 96, TINY_SEED)
    effect = window_acts(SmolLM3ForCausalLM, folder, back, entries[-1]["ids"])
    manifest = {
        "dataset": "tiny-smollm3",
        "note": "seeded SmolLM3, 4 layers: 0 full attention with RoPE, 1 sliding window "
                "without RoPE, 2 sliding window with RoPE, 3 full attention without "
                "RoPE; window 6 (key j for query i when i - 6 < j <= i), hidden 64, "
                "4 heads / 2 KV heads, head_dim 16 (hidden / heads), no bias, no "
                "per-head norm, tied embeddings, BF16 weights. The contract is "
                "model.safetensors + config.json. Per case (5, 13, 29 tokens): embed, "
                "layerNN, layer00_attn, layer00_mlp, norm, logits (all positions), "
                "greedy (24 tokens, no eos) with the top-2 margin - checked equal "
                "with the transformers cache and without",
        "config": json.load(open(os.path.join(folder, "config.json"), encoding="utf-8")),
        "safetensors_sha256": mr.sha256_file(st),
        "environment": mr.environment(),
        "window_effect": effect,
        "cases": entries,
    }
    blob.write(folder, manifest)


# ---------------------------------------------------------------- tiny Qwen3, sliding

def tiny_qwen3_sliding():
    print("tiny Qwen3 with a sliding window")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(
        vocab_size=96, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=512, rms_norm_eps=1e-6, rope_theta=1000000.0,
        tie_word_embeddings=True, hidden_act="silu", bos_token_id=1, eos_token_id=2,
        pad_token_id=0, use_sliding_window=True, sliding_window=5, max_window_layers=1,
        attn_implementation="eager")
    if cfg.layer_types != ["full_attention", "sliding_attention"]:
        raise ValueError(f"layer types {cfg.layer_types}")
    model = Qwen3ForCausalLM(cfg)
    seeded_weights(model, TINY_SEED + 1)
    folder = os.path.join(mr.OUT, "tiny-qwen3-sliding")
    # no layer_types: the reader derives them from max_window_layers
    keep = ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "max_position_embeddings", "rms_norm_eps", "rope_theta", "rope_parameters",
            "tie_word_embeddings", "hidden_act", "bos_token_id", "eos_token_id",
            "pad_token_id", "model_type", "use_sliding_window", "sliding_window",
            "max_window_layers", "attention_bias", "rope_scaling"]
    st = write_folder(model, cfg, folder, keep, "Qwen3ForCausalLM")
    back = Qwen3ForCausalLM.from_pretrained(folder, dtype=torch.float32,
                                            attn_implementation="eager")
    back.eval()
    if back.config.layer_types != cfg.layer_types:
        raise ValueError("config.json does not give the same layer types")
    blob, entries = cases(back, 96, TINY_SEED + 1)
    effect = window_acts(Qwen3ForCausalLM, folder, back, entries[-1]["ids"])
    manifest = {
        "dataset": "tiny-qwen3-sliding",
        "note": "seeded Qwen3, 2 layers, use_sliding_window with sliding_window 5 and "
                "max_window_layers 1: layer 0 full attention, layer 1 a window of 5 - "
                "config.json has no layer_types. hidden 64, 4 heads / 2 KV heads, "
                "head_dim 16, per-head norms, tied embeddings, BF16 weights. Per case "
                "(5, 13, 29 tokens): the states, logits and greedy (24 tokens, no eos) "
                "as tiny-smollm3",
        "config": json.load(open(os.path.join(folder, "config.json"), encoding="utf-8")),
        "safetensors_sha256": mr.sha256_file(st),
        "environment": mr.environment(),
        "window_effect": effect,
        "cases": entries,
    }
    blob.write(folder, manifest)


TARGETS = {"tiny-smollm3": tiny_smollm3, "tiny-qwen3-sliding": tiny_qwen3_sliding}

if __name__ == "__main__":
    want = sys.argv[1:] or list(TARGETS)
    for t in want:
        if t not in TARGETS:
            raise SystemExit(f"unknown target {t}; known: {', '.join(TARGETS)}")
    torch.set_num_threads(1)
    for t in want:
        TARGETS[t]()
