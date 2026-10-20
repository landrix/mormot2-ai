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
  gpt-oss-20b           tests/data/infer-ref/gpt-oss-20b: openai/gpt-oss-20b in the pinned
                        revision (not downloaded here: the folder $LAI_GPTOSS_DIR, by default
                        vendor/models/gpt-oss-20b, checked against the pinned hashes); four
                        prompts, per prompt the ids, the logits of the last position, 16
                        greedy tokens with the margin and the last row of every state.
                        Needs about 50 GB of RAM: the experts stay in BF16 (MXFP4
                        dequantized: exact), each one taken to float32 when used
  tiny-gpt-oss-mxfp4    tests/data/infer-ref/tiny-gpt-oss/mxfp4: the config of tiny-gpt-oss
                        with its own seed, the experts in MXFP4 as the published
                        checkpoints store them (quantization_config mxfp4, *_blocks and
                        *_scales), split into two files with model.safetensors.index.json;
                        transformers loads it (dequantized: exact in float32) for the cases

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


# the E2M1 values of the 8 codes without sign (bit 3 is the sign)
FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


def mxfp4_quantize(w):
    """[..., n] float32 rows (n a multiple of 32) as the checkpoint stores them:
    blocks [..., n / 32, 16] (value 2j in the low, 2j + 1 in the high nibble of
    byte j) and E8M0 scales [..., n / 32] - round to the nearest value, the
    scale so that the largest magnitude lands in [4, 8) (clipped at 6); a zero
    block gets the exponent 0"""
    shape = w.shape
    v = w.reshape(-1, 32).astype(np.float64)
    amax = np.abs(v).max(axis=1)
    e = np.zeros(len(v), dtype=np.int64)
    nz = amax > 0
    e[nz] = np.floor(np.log2(amax[nz])).astype(np.int64) - 2 + 127
    q = v / np.ldexp(1.0, (e - 127))[:, None]
    mag = np.abs(q)
    code = np.abs(mag[..., None] - FP4.astype(np.float64)).argmin(axis=-1)
    code = code + 8 * (q < 0)
    code[~nz] = 0
    nib = code.astype(np.uint8)
    blocks = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    scales = e.astype(np.uint8)
    return (blocks.reshape(shape[:-1] + (shape[-1] // 32, 16)),
            scales.reshape(shape[:-1] + (shape[-1] // 32,)))


def mxfp4_dequantize(blocks, scales):
    """the float32 rows of mxfp4_quantize's output"""
    lut = np.concatenate([FP4, -FP4])
    lo = lut[blocks & 15]
    hi = lut[blocks >> 4]
    v = np.stack([lo, hi], axis=-1).reshape(blocks.shape[:-1] + (32,))
    # ldexp, not a product: a zero times 2^128 stays zero (no inf * 0)
    v = np.ldexp(v, (scales.astype(np.int32) - 127)[..., None]).astype(np.float32)
    return v.reshape(blocks.shape[:-2] + (-1,)).astype(np.float32)


def tiny_gpt_oss_mxfp4():
    """the experts in MXFP4 as the published checkpoints store them, in two
    files with an index; a zero block with the exponent 0 and one of -0 values"""
    print("tiny gpt-oss, MXFP4 experts in two files")
    from safetensors.torch import save_file
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
    seeded_weights(model, TINY_SEED + 2)
    folder = os.path.join(mr.OUT, "tiny-gpt-oss", "mxfp4")
    os.makedirs(folder, exist_ok=True)
    for f in os.listdir(folder):
        if f.endswith(".safetensors"):
            os.remove(os.path.join(folder, f))
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            ex = layer.mlp.experts
            p = f"model.layers.{i}.mlp.experts."
            # rows are outputs in the checkpoint: gate_up_proj transposed
            gu = ex.gate_up_proj.detach().numpy().transpose(0, 2, 1).copy()
            dn = ex.down_proj.detach().numpy().transpose(0, 2, 1).copy()
            gub, gus = mxfp4_quantize(gu)
            dnb, dns = mxfp4_quantize(dn)
            if i == 0:
                # expert 1, gate row 0, block 0: zero with the exponent 0; up
                # row 0, block 1: all -0 (code 8) with the exponent 255
                gub[1, 0, 0] = 0
                gus[1, 0, 0] = 0
                gub[1, 1, 1] = 0x88
                gus[1, 1, 1] = 255
            for k in [p + "gate_up_proj", p + "down_proj"]:
                del sd[k]
            sd[p + "gate_up_proj_blocks"] = torch.from_numpy(gub)
            sd[p + "gate_up_proj_scales"] = torch.from_numpy(gus)
            sd[p + "down_proj_blocks"] = torch.from_numpy(dnb)
            sd[p + "down_proj_scales"] = torch.from_numpy(dns)
            # the model computes with the values the blocks hold
            ex.gate_up_proj.copy_(torch.from_numpy(
                mxfp4_dequantize(gub, gus).transpose(0, 2, 1).copy()))
            ex.down_proj.copy_(torch.from_numpy(
                mxfp4_dequantize(dnb, dns).transpose(0, 2, 1).copy()))
    # two files: the embedding and layer 0, then the rest
    first = {k: v for k, v in sd.items()
             if k.startswith("model.layers.0.") or k == "model.embed_tokens.weight"}
    files = {"model-00001-of-00002.safetensors": first,
             "model-00002-of-00002.safetensors":
                 {k: v for k, v in sd.items() if k not in first}}
    weight_map = {}
    total = 0
    for name, part in files.items():
        save_file(part, os.path.join(folder, name), metadata={"format": "pt"})
        for k, v in part.items():
            weight_map[k] = name
            total += v.numel() * v.element_size()
    with open(os.path.join(folder, "model.safetensors.index.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump({"metadata": {"total_size": total},
                   "weight_map": dict(sorted(weight_map.items()))}, f, indent=1)
        f.write("\n")
    d = cfg.to_dict()
    keep = ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "num_local_experts", "num_experts_per_tok", "max_position_embeddings",
            "rms_norm_eps", "sliding_window", "layer_types", "rope_parameters",
            "swiglu_limit", "tie_word_embeddings", "attention_bias", "hidden_act",
            "bos_token_id", "eos_token_id", "pad_token_id", "model_type"]
    conf = {k: d[k] for k in keep if k in d}
    conf["architectures"] = ["GptOssForCausalLM"]
    conf["torch_dtype"] = "bfloat16"
    # as gpt-oss-20b writes it
    conf["quantization_config"] = {
        "modules_to_not_convert": ["model.layers.*.self_attn", "model.layers.*.mlp.router",
                                   "model.embed_tokens", "lm_head"],
        "quant_method": "mxfp4"}
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(conf, f, indent=1)
        f.write("\n")
    # dequantize=True: what transformers does on a CPU anyway (without it,
    # it first asks for accelerate)
    from transformers import Mxfp4Config
    back = GptOssForCausalLM.from_pretrained(
        folder, dtype=torch.float32, attn_implementation="eager",
        experts_implementation="eager",
        quantization_config=Mxfp4Config(dequantize=True))
    back = back.float()
    back.eval()
    if back.config._experts_implementation != "eager":
        raise ValueError(f"experts {back.config._experts_implementation}")
    # transformers dequantized exactly what the blocks hold
    for i, layer in enumerate(back.model.layers):
        mine = model.model.layers[i].mlp.experts
        for n in ["gate_up_proj", "down_proj"]:
            a = getattr(layer.mlp.experts, n).detach()
            b = getattr(mine, n).detach()
            if not torch.equal(a.float(), b.float()):
                raise ValueError(f"layer {i} {n}: transformers reads other values")
    clamped = clamp_share(back, 96, TINY_SEED + 2)
    blob, entries = d4.cases(back, 96, TINY_SEED + 2)
    # one expert per layer as transformers dequantized it, rows = outputs
    for i, layer in enumerate(back.model.layers):
        gu = layer.mlp.experts.gate_up_proj.detach()[3].float()
        blob.add(f"experts.l{i}.e3.gate", gu[:, 0::2].T.contiguous().numpy())
        blob.add(f"experts.l{i}.e3.up", gu[:, 1::2].T.contiguous().numpy())
        blob.add(f"experts.l{i}.e3.down",
                 layer.mlp.experts.down_proj.detach()[3].float().T.contiguous().numpy())
    manifest = {
        "dataset": "tiny-gpt-oss/mxfp4",
        "note": "the config of tiny-gpt-oss (seed + 2), the experts in MXFP4 as "
                "gpt-oss-20b stores them: quantization_config mxfp4, "
                "gate_up_proj_blocks [experts, 2 * width, hidden / 32, 16] (rows = "
                "outputs, gate and up rows interleaved; byte j holds value 2j low, "
                "2j + 1 high), gate_up_proj_scales [experts, 2 * width, hidden / 32] "
                "(E8M0, bias 127), down_proj the same over the width; layer 0 expert 1 "
                "has a zero block with the exponent 0 (gate row 0) and one of -0 values "
                "with the exponent 255 (up row 0, block 1). Two files and "
                "model.safetensors.index.json. transformers loads it dequantized (exact "
                "in float32); the cases as tiny-gpt-oss; experts.lN.e3.{gate,up,down}: "
                "expert 3 of layer N as transformers dequantized it, rows = outputs",
        "config": conf,
        "safetensors_sha256": {name: mr.sha256_file(os.path.join(folder, name))
                               for name in files},
        "environment": mr.environment(),
        "clamped_share": clamped,
        "cases": entries,
    }
    blob.write(folder, manifest)


GPTOSS_ID = "openai/gpt-oss-20b"
GPTOSS_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
GPTOSS_SHA256 = {
    "model-00000-of-00002.safetensors":
        "16d0f997dcfc4462089d536bffe51b4bcea2f872f5c430be09ef8ed392312427",
    "model-00001-of-00002.safetensors":
        "4fbe328ab445455d6f58dc73852b85873bd626986310abd91cd4d2ce3245eaea",
    "model-00002-of-00002.safetensors":
        "a18106b209e9ab35c3406db4f6f12a927364a058b21e9d1373d682e20674b303",
    "tokenizer.json":
        "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3",
}
GPTOSS_PROMPTS = [
    ("raw", "The capital of France is"),
    ("german", "Die Handwerkskammer prüft die Rechnung des Betriebs, weil"),
    ("code", "def fibonacci(n):\n    "),
    ("chat", None),  # the chat template of the revision, one user message
]
GPTOSS_CHAT = [{"role": "user", "content": "What is 17 times 23? Answer briefly."}]


class FloatSlices:
    """a BF16 expert tensor that hands out float32 slices: the eager experts of
    transformers index it per expert, the model stays at half the memory"""

    def __init__(self, t):
        self.t = t

    def __getitem__(self, i):
        return self.t[i].float()


def gpt_oss_20b():
    print("gpt-oss-20b")
    from transformers import AutoTokenizer, GptOssForCausalLM, Mxfp4Config
    folder = os.environ.get("LAI_GPTOSS_DIR",
                            os.path.join(mr.ROOT, "vendor", "models", "gpt-oss-20b"))
    for name, want in GPTOSS_SHA256.items():
        got = mr.sha256_file(os.path.join(folder, name))
        if got != want:
            raise ValueError(f"{name}: sha256 {got}, pinned {want}")
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        conf = json.load(f)
    tok = AutoTokenizer.from_pretrained(folder)
    # dequantized to BF16 on a CPU (exact: an E2M1 value times a power of two)
    model = GptOssForCausalLM.from_pretrained(
        folder, dtype=torch.bfloat16, attn_implementation="eager",
        experts_implementation="eager",
        quantization_config=Mxfp4Config(dequantize=True))
    model.eval()
    if model.config._experts_implementation != "eager":
        raise ValueError(f"experts {model.config._experts_implementation}")
    # everything but the expert matrices in float32; those as float32 slices
    with torch.no_grad():
        for layer in model.model.layers:
            ex = layer.mlp.experts
            gu, dn = ex.gate_up_proj.data, ex.down_proj.data
            del ex.gate_up_proj
            del ex.down_proj
            ex.gate_up_proj = FloatSlices(gu)
            ex.down_proj = FloatSlices(dn)
        for p in model.parameters():
            p.data = p.data.float()
        for name, b in model.named_buffers():
            if b.is_floating_point():
                b.data = b.data.float()
    blob = mr.Blob()
    entries = []
    for k, (kind, text) in enumerate(GPTOSS_PROMPTS):
        if text is None:
            ids = tok.apply_chat_template(GPTOSS_CHAT, add_generation_prompt=True,
                                          tokenize=True)
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            rendered = tok.apply_chat_template(GPTOSS_CHAT, add_generation_prompt=True,
                                               tokenize=False)
        else:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            rendered = text
        print(f"  case {k} ({kind}): {len(ids)} tokens", flush=True)
        x = torch.tensor([ids])
        # the last row of every state: the embedding, each layer, the final norm
        taken = {}
        hooks = []

        def keep(name):
            def hook(_mod, _inp, out):
                o = out[0] if isinstance(out, tuple) else out
                taken[name] = o[0, -1].detach().float().clone()
            return hook

        inner = model.model
        hooks.append(inner.embed_tokens.register_forward_hook(keep("embed")))
        for i, layer in enumerate(inner.layers):
            hooks.append(layer.register_forward_hook(keep(f"layer{i:02d}")))
        hooks.append(inner.norm.register_forward_hook(keep("norm")))
        try:
            with torch.no_grad():
                out = model(input_ids=x, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        logits = out.logits[0, -1].float().numpy()
        if k == 0:
            with torch.no_grad():
                again = model(input_ids=x, use_cache=False).logits[0, -1].float().numpy()
            if not np.array_equal(logits, again):
                raise ValueError("two runs differ")
        # greedy with the cache, the margin of every step
        greedy, margins = [], []
        cur = list(ids)
        past = None
        nxt = x
        with torch.no_grad():
            for _ in range(16):
                o = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = o.past_key_values
                row = o.logits[0, -1].float()
                top = torch.topk(row, 2)
                t = int(top.indices[0])
                greedy.append(t)
                margins.append(float(top.values[0] - top.values[1]))
                cur.append(t)
                nxt = torch.tensor([[t]])
        blob.add(f"case{k}.ids", np.asarray(ids, dtype=np.int32))
        blob.add(f"case{k}.logits_last", logits)
        for name, v in taken.items():
            blob.add(f"case{k}.last.{name}", v.numpy(), "the state at the last position")
        blob.add(f"case{k}.greedy", np.asarray(greedy, dtype=np.int32))
        blob.add(f"case{k}.greedy_margin", np.asarray(margins, dtype=np.float32))
        entries.append({"case": k, "kind": kind, "text": rendered, "tokens": len(ids),
                        "greedy_text": tok.decode(greedy),
                        "min_margin": min(margins)})
    manifest = {
        "dataset": "gpt-oss-20b",
        "note": "openai/gpt-oss-20b, MXFP4 experts dequantized by transformers (exact "
                "in BF16 and float32), computed in float32 (the experts taken to "
                "float32 per use), eager attention and experts, one thread. Per case: "
                "ids (tokenizer.json of the revision, no special tokens added; case 3 "
                "through the chat template with the generation prompt), logits_last, "
                "last.embed / last.layerNN / last.norm: the row of the last position "
                "of the embedding, every layer output and the final norm, greedy 16 tokens with the cache and the top-2 margin per step. "
                "Generated on the RTX computer (Windows x86_64, 96 GB): the model needs "
                "more memory than the WSL reference environment has",
        "model": {"id": GPTOSS_ID, "revision": GPTOSS_REVISION, "sha256": GPTOSS_SHA256},
        "config": conf,
        "environment": mr.environment(),
        "cases": entries,
    }
    blob.write(os.path.join(mr.OUT, "gpt-oss-20b"), manifest)


TARGETS = {"tiny-gpt-oss": tiny_gpt_oss, "tiny-gpt-oss-narrow": tiny_gpt_oss_narrow,
           "tiny-gpt-oss-mxfp4": tiny_gpt_oss_mxfp4, "gpt-oss-20b": gpt_oss_20b}

if __name__ == "__main__":
    # gpt-oss-20b only when asked for: it needs the model and 50 GB of RAM
    want = sys.argv[1:] or [t for t in TARGETS if t != "gpt-oss-20b"]
    for t in want:
        if t not in TARGETS:
            raise SystemExit(f"unknown target {t}; known: {', '.join(TARGETS)}")
    torch.set_num_threads(1)
    for t in want:
        TARGETS[t]()
