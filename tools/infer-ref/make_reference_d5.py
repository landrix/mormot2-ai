#!/usr/bin/env python3
"""Reference data for phase D slice D5 of the native inference engine: gpt-oss.

Targets (all of them when called without arguments):
  tiny-gpt-oss          tests/data/infer-ref/tiny-gpt-oss: a seeded 3-layer gpt-oss
                        (sliding, full, sliding with a window of 5), 8 experts of which a
                        token takes 3, YaRN rotary positions (factor 32, truncate false:
                        the ramp between interpolation and extrapolation falls inside the
                        16 dimensions of a head pair), attention sinks, a bias on Q, K, V,
                        the output, the router and the experts, the clamped SwiGLU with a
                        small swiglu_limit so that it clamps, head_dim 24 (not hidden /
                        heads), an untied head, BF16 weights; every state, the logits,
                        greedy continuations and the rotary factors of positions 0..63
  tiny-gpt-oss-narrow   tests/data/infer-ref/tiny-gpt-oss/narrow: 2 layers, experts of
                        width 16 for a hidden width of 64 (a down_proj row is longer than a
                        gate_up_proj row), the same cases

No model download. Same environment, format and rules as make_reference.py; the helpers
of make_reference_qwen3.py and make_reference_d4.py are reused. Every case is checked
twice: two runs give the same bits, and the greedy continuation with the transformers
cache equals the one recomputed without cache. The experts run with the eager
implementation (one expert after the other), as our CPU computes them.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_reference as mr  # noqa: E402
import make_reference_d4 as d4  # noqa: E402

TINY_SEED = 20261020
ROPE_POSITIONS = 64


def seeded_weights(model, seed):
    """BF16-exact weights: norms around 1, sinks of either sign and of a size
    that weighs against the scores, biases large enough to count"""
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("layernorm.weight") or name == "model.norm.weight":
                p.copy_(1.0 + 0.25 * torch.randn_like(p))
            elif name.endswith(".sinks"):
                p.copy_(1.5 * torch.randn_like(p))
            elif name.endswith("o_proj.bias") or name.endswith("down_proj_bias"):
                # added to the residual of every token alike: kept small, or
                # one direction wins and greedy repeats one token
                p.copy_(0.02 * torch.randn_like(p))
            elif name.endswith("bias"):
                p.copy_(0.3 * torch.randn_like(p))
            else:
                p.copy_(0.08 * torch.randn_like(p))
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
    model.eval()


def rope_factors(rotary):
    """cos and sin of positions 0..ROPE_POSITIONS-1 as the model applies them
    ([positions, head_dim / 2], the attention factor of YaRN included)"""
    pos = torch.arange(ROPE_POSITIONS).unsqueeze(0)
    x = torch.zeros(1, dtype=torch.float32)
    with torch.no_grad():
        cos, sin = rotary(x, pos)
    return cos[0].numpy(), sin[0].numpy()


# YaRN variants of the rope settings, rotary factors only: truncate true (the
# correction range on whole dimensions), and an explicit attention_factor with
# other betas
ROPE_VARIANTS = {
    "rope_t": {"truncate": True},
    "rope_a": {"attention_factor": 1.25, "beta_fast": 16.0, "beta_slow": 2.0},
}


def rope_variant(cfg, change):
    import copy
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
    c = copy.deepcopy(cfg)
    c.rope_parameters = dict(c.rope_parameters, **change)
    return GptOssRotaryEmbedding(c)


def tiny_gpt_oss():
    print("tiny gpt-oss")
    from transformers import GptOssConfig, GptOssForCausalLM
    cfg = GptOssConfig(
        vocab_size=96, hidden_size=64, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=24,
        num_local_experts=8, num_experts_per_tok=3, max_position_embeddings=512,
        rms_norm_eps=1e-5, sliding_window=5,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        rope_parameters={"rope_type": "yarn", "rope_theta": 150000.0, "factor": 32.0,
                         "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False,
                         "original_max_position_embeddings": 4096},
        swiglu_limit=0.8, tie_word_embeddings=False, attention_bias=True,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, attn_implementation="eager",
        experts_implementation="eager")
    model = GptOssForCausalLM(cfg)
    seeded_weights(model, TINY_SEED)
    folder = os.path.join(mr.OUT, "tiny-gpt-oss")
    # swiglu_alpha left out of config.json: the real one has none (default 1.702)
    keep = ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "num_local_experts", "num_experts_per_tok", "max_position_embeddings",
            "rms_norm_eps", "sliding_window", "layer_types", "rope_parameters",
            "swiglu_limit", "tie_word_embeddings", "attention_bias", "hidden_act",
            "bos_token_id", "eos_token_id", "pad_token_id", "model_type"]
    st = write_folder(model, cfg, folder, keep)
    back = GptOssForCausalLM.from_pretrained(folder, dtype=torch.float32,
                                             attn_implementation="eager",
                                             experts_implementation="eager")
    back.eval()
    if back.config._experts_implementation != "eager":
        raise ValueError(f"experts {back.config._experts_implementation}")
    if back.config.swiglu_alpha != 1.702 or back.config.swiglu_limit != 0.8:
        raise ValueError("swiglu settings not read back")
    if back.model.rotary_emb.attention_scaling == 1.0:
        raise ValueError("no YaRN attention factor")
    clamped = clamp_share(back, 96, TINY_SEED)
    blob, entries = d4.cases(back, 96, TINY_SEED)
    effect = d4.window_acts(GptOssForCausalLM, folder, back, entries[-1]["ids"])
    cos, sin = rope_factors(back.model.rotary_emb)
    blob.add("rope.cos", cos)
    blob.add("rope.sin", sin)
    blob.add("rope.inv_freq", back.model.rotary_emb.inv_freq.numpy())
    variants = {}
    for name, change in ROPE_VARIANTS.items():
        rot = rope_variant(back.config, change)
        cos2, sin2 = rope_factors(rot)
        if np.array_equal(cos2, cos):
            raise ValueError(f"{name} changes nothing")
        blob.add(f"{name}.cos", cos2)
        blob.add(f"{name}.sin", sin2)
        variants[name] = {"change": change, "attention_scaling": rot.attention_scaling}
    manifest = {
        "dataset": "tiny-gpt-oss",
        "note": "seeded gpt-oss, 3 layers: 0 and 2 a sliding window of 5, 1 full "
                "attention; hidden 64, 4 heads / 2 KV heads, head_dim 24, 8 experts "
                "of width 64, 3 per token (softmax over the top 3 router logits), "
                "YaRN (theta 150000, factor 32, original 4096, beta 32 / 1, truncate "
                "false), attention sinks, bias on q/k/v/o, router and experts, "
                "clamped SwiGLU (limit 0.8, alpha 1.702 by default), untied head, "
                "BF16 weights. The contract is model.safetensors + config.json "
                "(transformers 5 layout: gate_up_proj [experts, hidden, 2 * width] "
                "with gate and up interleaved, down_proj [experts, width, hidden]). "
                "Per case (5, 13, 29 tokens): embed, layerNN, layer00_attn, "
                "layer00_mlp, norm, logits (all positions), greedy (24 tokens, no "
                "eos) with the top-2 margin - checked equal with the transformers "
                "cache and without. rope.cos / rope.sin: [64 positions, 12], the "
                "attention factor included; rope.inv_freq [12]; rope_t / rope_a: the "
                "same for the rope settings changed as rope_variants says",
        "config": json.load(open(os.path.join(folder, "config.json"), encoding="utf-8")),
        "safetensors_sha256": mr.sha256_file(st),
        "environment": mr.environment(),
        "window_effect": effect,
        "attention_scaling": back.model.rotary_emb.attention_scaling,
        "clamped_share": clamped,
        "rope_variants": variants,
        "cases": entries,
    }
    blob.write(folder, manifest)


def clamp_share(model, vocab, seed):
    """the share of gate and up values the clamp changes in the longest case:
    it must clamp, or the limit is not tested"""
    rng = np.random.default_rng(seed + 7)
    ids = rng.integers(3, vocab, size=29).tolist()
    seen = []

    def hook(mod, inp, _out):
        gate_up = inp[0]
        g, u = gate_up[..., ::2], gate_up[..., 1::2]
        lim = mod.limit
        seen.append(float(((g > lim) | (u > lim) | (u < -lim)).float().mean()))

    for layer in model.model.layers:
        # _apply_gate gets gate_up: an instance attribute wraps it to look
        e = layer.mlp.experts
        orig = e._apply_gate

        def wrapped(gate_up, _orig=orig, _mod=e):
            hook(_mod, (gate_up,), None)
            return _orig(gate_up)
        e._apply_gate = wrapped
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        for layer in model.model.layers:
            del layer.mlp.experts._apply_gate
    share = sum(seen) / len(seen)
    if not 0.05 < share < 0.95:
        raise ValueError(f"the clamp changes {share:.3f} of the values")
    return share


def write_folder(model, cfg, folder, keep):
    from safetensors.torch import save_file
    os.makedirs(folder, exist_ok=True)
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    st = os.path.join(folder, "model.safetensors")
    save_file(sd, st, metadata={"format": "pt"})
    d = cfg.to_dict()
    d["architectures"] = ["GptOssForCausalLM"]
    d["torch_dtype"] = "bfloat16"
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({k: d[k] for k in keep + ["architectures", "torch_dtype"] if k in d},
                  f, indent=1)
        f.write("\n")
    return st


def tiny_gpt_oss_narrow():
    """experts narrower than half the hidden width (16 against 64): a row of
    down_proj is longer than one of gate_up_proj"""
    print("tiny gpt-oss, narrow experts")
    from transformers import GptOssConfig, GptOssForCausalLM
    cfg = GptOssConfig(
        vocab_size=96, hidden_size=64, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=512,
        rms_norm_eps=1e-5, sliding_window=5,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={"rope_type": "yarn", "rope_theta": 150000.0, "factor": 32.0,
                         "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False,
                         "original_max_position_embeddings": 4096},
        swiglu_limit=0.8, tie_word_embeddings=False, attention_bias=True,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, attn_implementation="eager",
        experts_implementation="eager")
    model = GptOssForCausalLM(cfg)
    seeded_weights(model, TINY_SEED + 1)
    folder = os.path.join(mr.OUT, "tiny-gpt-oss", "narrow")
    keep = ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "num_local_experts", "num_experts_per_tok", "max_position_embeddings",
            "rms_norm_eps", "sliding_window", "layer_types", "rope_parameters",
            "swiglu_limit", "tie_word_embeddings", "attention_bias", "hidden_act",
            "bos_token_id", "eos_token_id", "pad_token_id", "model_type"]
    st = write_folder(model, cfg, folder, keep)
    back = GptOssForCausalLM.from_pretrained(folder, dtype=torch.float32,
                                             attn_implementation="eager",
                                             experts_implementation="eager")
    back.eval()
    blob, entries = d4.cases(back, 96, TINY_SEED + 1)
    manifest = {
        "dataset": "tiny-gpt-oss/narrow",
        "note": "seeded gpt-oss, 2 layers (window 5, full), 4 experts of width 16 "
                "(less than half the hidden width 64), 2 per token, head_dim 16, "
                "otherwise as tiny-gpt-oss; the same cases",
        "config": json.load(open(os.path.join(folder, "config.json"), encoding="utf-8")),
        "safetensors_sha256": mr.sha256_file(st),
        "environment": mr.environment(),
        "cases": entries,
    }
    blob.write(folder, manifest)


TARGETS = {"tiny-gpt-oss": tiny_gpt_oss, "tiny-gpt-oss-narrow": tiny_gpt_oss_narrow}

if __name__ == "__main__":
    want = sys.argv[1:] or list(TARGETS)
    for t in want:
        if t not in TARGETS:
            raise SystemExit(f"unknown target {t}; known: {', '.join(TARGETS)}")
    torch.set_num_threads(1)
    for t in want:
        TARGETS[t]()
