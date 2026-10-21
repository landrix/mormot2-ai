#!/usr/bin/env python3
"""Reference data for phase D slice D7a of the native inference engine: Qwen3.5
(Gated DeltaNet beside gated full attention).

Targets (all of them when called without arguments):
  tiny-qwen35           tests/data/infer-ref/tiny-qwen35: a seeded 4-layer Qwen3.5 in the
                        layout of the published checkpoints (Qwen3_5ForConditionalGeneration,
                        model_type qwen3_5, the text settings in text_config, the tensors
                        under model.language_model., a tiny vision tower under model.visual.
                        and an mtp tensor beside them that a text model leaves alone): the
                        default layer types (three linear_attention, then full_attention),
                        2 key and 2 value heads of 16 in the DeltaNet layers with a
                        convolution of 4, full attention with an output gate, 4 heads of 32,
                        2 KV heads, a quarter of each head rotating; tied head; BF16 weights.
                        Every state, the logits, greedy continuations, the rotary factors of
                        positions 0..63, and the logits of every position computed one
                        token at a time with the cache (the recurrent form) beside those of
                        one pass (the chunked form)
  tiny-qwen35-text      tests/data/infer-ref/tiny-qwen35/text: Qwen3_5ForCausalLM
                        (model_type qwen3_5_text, the settings at the top, the tensors under
                        model.), 5 layers of explicit layer_types (linear, full, linear,
                        linear, full), 2 key heads read by 4 value heads (repeat_interleave),
                        a convolution of 3, half of each head rotating, an untied head
  qwen35-0.8b           tests/data/infer-ref/qwen35-0.8b: Qwen/Qwen3.5-0.8B at the pinned
                        revision (LAI_QWEN35_DIR, else the Hugging Face cache): ids of a few
                        prompts (plain text and the chat template without thinking),
                        logits of the last position, the last row of every state, greedy
                        continuations with the cache

Same environment, format and rules as make_reference.py. Every tiny case is checked
twice: two runs give the same bits, and the greedy continuation with the transformers
cache equals the one recomputed without cache. Without the fla and causal-conv1d packages
transformers takes its torch paths: torch_chunk_gated_delta_rule for a prompt,
torch_recurrent_gated_delta_rule for a cached step, causal_conv1d_fn / _update.
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

TINY_SEED = 20261021
ROPE_POSITIONS = 64
VOCAB = 96


def seeded_weights(model, seed):
    """BF16-exact weights: the zero-centered norms (1 + weight) around 1, the gated
    norm of the DeltaNet around 1, decays that keep a memory over the cases
    (A between 0.05 and 1), a convolution that mixes the tokens"""
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("linear_attn.norm.weight"):
                p.copy_(1.0 + 0.25 * torch.randn_like(p))
            elif name.endswith("norm.weight"):
                p.copy_(0.25 * torch.randn_like(p))
            elif name.endswith("A_log"):
                p.copy_(torch.empty_like(p).uniform_(0.05, 1.0).log())
            elif name.endswith("dt_bias"):
                p.copy_(0.5 * torch.randn_like(p))
            elif name.endswith("conv1d.weight"):
                p.copy_(0.3 * torch.randn_like(p))
            else:
                p.copy_(0.08 * torch.randn_like(p))
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
    model.eval()


def text_model(model):
    inner = model.model
    return inner.language_model if hasattr(inner, "language_model") else inner


def record(model, ids):
    """one forward pass without cache: the input of the first layer, every layer
    output, the mixer (DeltaNet or attention) and MLP output of layer 0, the final
    norm and the logits"""
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
    first = inner.layers[0]
    mixer = first.linear_attn if first.block_type == "linear_attention" else first.self_attn
    hooks.append(mixer.register_forward_hook(keep("layer00_attn")))
    hooks.append(first.mlp.register_forward_hook(keep("layer00_mlp")))
    hooks.append(inner.norm.register_forward_hook(keep("norm")))
    try:
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    taken["logits"] = out.logits.detach().clone()
    return {k: v[0].numpy() for k, v in taken.items()}


def stepwise(model, ids):
    """the logits of every position, one token at a time with the cache: the
    recurrent form of the delta rule and the convolution update"""
    rows = []
    past = None
    with torch.no_grad():
        for t in ids:
            o = model(input_ids=torch.tensor([[t]]), past_key_values=past, use_cache=True)
            past = o.past_key_values
            rows.append(o.logits[0, -1].clone())
    return torch.stack(rows).numpy()


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
        steps = stepwise(back, ids)
        blob.add(f"case{k}.logits_steps", steps,
                 "the logits of every position, one token at a time with the cache")
        e["steps_vs_pass"] = float(np.abs(steps - a["logits"]).max())
        new, margins = d4.greedy(back, ids, d4.GREEDY_TOKENS)
        if q3.greedy_no_cache(back, ids, d4.GREEDY_TOKENS, []) != new:
            raise ValueError(f"case {k}: greedy with and without cache differ")
        blob.add(f"case{k}.greedy", np.array(new, dtype=np.int64))
        blob.add(f"case{k}.greedy_margin", np.array(margins, dtype=np.float32))
        e["greedy"] = new
        e["min_margin"] = min(margins)
        entries.append(e)
    return blob, entries


def rope_factors(rotary):
    """cos and sin of positions 0..ROPE_POSITIONS-1 ([positions, rotary dims / 2]:
    the first half of what the model applies; the three position rows of a text
    token are equal, so the interleaved sections change nothing)"""
    pos = torch.arange(ROPE_POSITIONS).unsqueeze(0)
    x = torch.zeros(1, dtype=torch.float32)
    with torch.no_grad():
        cos, sin = rotary(x, pos)
    half = cos.shape[-1] // 2
    if not (torch.equal(cos[..., :half], cos[..., half:]) and
            torch.equal(sin[..., :half], sin[..., half:])):
        raise ValueError("the rotary halves differ")
    return cos[0, :, :half].numpy(), sin[0, :, :half].numpy()


def recurrence_acts(back, ids):
    """the recurrent state carries: the same tokens with the DeltaNet decay at
    its fastest (A huge: the state forgets at once) give other logits"""
    with torch.no_grad():
        a = back(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1].clone()
        keep = []
        for layer in text_model(back).layers:
            if layer.block_type == "linear_attention":
                keep.append((layer.linear_attn.A_log, layer.linear_attn.A_log.clone()))
                layer.linear_attn.A_log.fill_(30.0)
        try:
            b = back(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]
        finally:
            for p, v in keep:
                p.copy_(v)
    if torch.equal(a, b):
        raise ValueError("the recurrent state changes nothing")
    return float((a - b).abs().max())


def gate_acts(back, ids):
    """the output gate of the full attention bites: with the gate rows of q_proj
    zeroed (sigmoid 0.5 everywhere) the logits change"""
    with torch.no_grad():
        a = back(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1].clone()
        keep = []
        for layer in text_model(back).layers:
            if layer.block_type == "full_attention":
                at = layer.self_attn
                w = at.q_proj.weight
                keep.append((w, w.clone()))
                v = w.view(at.config.num_attention_heads, 2 * at.head_dim, -1)
                v[:, at.head_dim:, :] = 0
        try:
            b = back(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]
        finally:
            for p, v in keep:
                p.copy_(v)
    if torch.equal(a, b):
        raise ValueError("the output gate changes nothing")
    return float((a - b).abs().max())


def save(sd, folder, extra=None):
    from safetensors.torch import save_file
    os.makedirs(folder, exist_ok=True)
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()}
    if extra:
        sd.update(extra)
    st = os.path.join(folder, "model.safetensors")
    save_file(sd, st, metadata={"format": "pt"})
    return st


def write_json(conf, folder):
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(conf, f, indent=1)
        f.write("\n")


TINY_TEXT = {
    "vocab_size": VOCAB, "hidden_size": 64, "intermediate_size": 96,
    "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
    "head_dim": 32, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 16,
    "linear_value_head_dim": 16, "linear_num_key_heads": 2, "linear_num_value_heads": 2,
    "full_attention_interval": 4, "hidden_act": "silu", "rms_norm_eps": 1e-6,
    "max_position_embeddings": 512, "attention_bias": False, "attention_dropout": 0.0,
    "attn_output_gate": True, "mlp_only_layers": [], "tie_word_embeddings": True,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0,
                        "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                        "mrope_section": [2, 1, 1]},
    "bos_token_id": None, "eos_token_id": 1, "pad_token_id": None, "use_cache": True,
    "initializer_range": 0.02,
}

TINY_VISION = {
    "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
    "out_hidden_size": 64, "patch_size": 16, "spatial_merge_size": 2,
    "temporal_patch_size": 2, "num_position_embeddings": 64,
    "hidden_act": "gelu_pytorch_tanh", "in_channels": 3,
}

TEXT_TEXT = dict(TINY_TEXT)
TEXT_TEXT.update({
    "num_hidden_layers": 5, "linear_num_value_heads": 4, "linear_conv_kernel_dim": 3,
    "tie_word_embeddings": False,
    "layer_types": ["linear_attention", "full_attention", "linear_attention",
                    "linear_attention", "full_attention"],
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0,
                        "partial_rotary_factor": 0.5, "mrope_interleaved": True,
                        "mrope_section": [4, 2, 2]},
})
del TEXT_TEXT["full_attention_interval"]


def load_back(cls, folder):
    back = cls.from_pretrained(folder, dtype=torch.float32, attn_implementation="eager")
    back.eval()
    return back


def check_layers(back, types):
    got = [layer.block_type for layer in text_model(back).layers]
    if got != types:
        raise ValueError(f"layer types {got}")


def finish(folder, back, blob, entries, manifest):
    longest = entries[-1]["ids"]
    manifest["recurrence_effect"] = recurrence_acts(back, longest)
    manifest["gate_effect"] = gate_acts(back, longest)
    cos, sin = rope_factors(text_model(back).rotary_emb)
    blob.add("rope.cos", cos)
    blob.add("rope.sin", sin)
    manifest["environment"] = mr.environment()
    manifest["cases"] = entries
    blob.write(folder, manifest)


def tiny_qwen35():
    print("tiny Qwen3.5")
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    folder = os.path.join(mr.OUT, "tiny-qwen35")
    text = dict(TINY_TEXT)
    cfg = Qwen3_5Config(text_config=dict(text), vision_config=dict(TINY_VISION),
                        tie_word_embeddings=True)
    model = Qwen3_5ForConditionalGeneration(cfg)
    seeded_weights(model, TINY_SEED)
    sd = {k: v for k, v in model.state_dict().items() if k != "lm_head.weight"}
    # the published checkpoints carry the multi-token prediction head beside the
    # model (Qwen3_5PreTrainedModel ignores ^mtp.*): one such tensor
    torch.manual_seed(TINY_SEED + 7)
    mtp = {"mtp.fc.weight": (0.08 * torch.randn(64, 128)).to(torch.bfloat16)}
    st = save(sd, folder, mtp)
    conf = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
            "text_config": dict(text, model_type="qwen3_5_text", dtype="bfloat16"),
            "vision_config": dict(TINY_VISION, model_type="qwen3_5"),
            "image_token_id": 90, "video_token_id": 91, "vision_start_token_id": 92,
            "vision_end_token_id": 93, "tie_word_embeddings": True}
    write_json(conf, folder)
    back = load_back(Qwen3_5ForConditionalGeneration, folder)
    check_layers(back, ["linear_attention"] * 3 + ["full_attention"])
    blob, entries = cases(back, TINY_SEED)
    manifest = {
        "dataset": "tiny-qwen35",
        "note": "seeded Qwen3.5 (Qwen3_5ForConditionalGeneration, text only), 4 layers "
                "of the default types: 0..2 linear_attention (Gated DeltaNet: 2 key and "
                "2 value heads of 16, a depthwise causal convolution of 4 with silu, "
                "q and k L2-normed, beta = sigmoid(b), decay g = -exp(A_log) "
                "softplus(a + dt_bias), a gated RMSNorm per value head, out_proj), 3 "
                "full_attention (4 heads of 32, 2 KV heads, q_proj twice as wide: per "
                "head the query, then the gate; q and k normed; a quarter of each head "
                "rotating with theta 1e7; the output times sigmoid(gate)); RMSNorms "
                "times (1 + weight), the gated one times its weight; hidden 64, SwiGLU "
                "96; tied head; BF16 weights; a tiny vision tower under model.visual. "
                "and one mtp tensor, both outside the text model. Per case (5, 13, 29 "
                "tokens): embed, layerNN, layer00_attn (the DeltaNet output), "
                "layer00_mlp, norm, logits (all positions, one pass: the chunked delta "
                "rule), logits_steps (the same positions one token at a time with the "
                "cache: the recurrent rule), greedy (24 tokens, no eos) with the top-2 "
                "margin - checked equal with the transformers cache and without. "
                "rope.cos / rope.sin [64, 4]",
        "config": json.load(open(os.path.join(folder, "config.json"), encoding="utf-8")),
        "safetensors_sha256": mr.sha256_file(st),
        "seed": TINY_SEED,
    }
    finish(folder, back, blob, entries, manifest)


def tiny_qwen35_text():
    print("tiny Qwen3.5, text layout")
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    folder = os.path.join(mr.OUT, "tiny-qwen35", "text")
    cfg = Qwen3_5TextConfig(**TEXT_TEXT)
    model = Qwen3_5ForCausalLM(cfg)
    seeded_weights(model, TINY_SEED + 1)
    st = save(model.state_dict(), folder)
    conf = dict(TEXT_TEXT, architectures=["Qwen3_5ForCausalLM"],
                model_type="qwen3_5_text", dtype="bfloat16")
    write_json(conf, folder)
    back = load_back(Qwen3_5ForCausalLM, folder)
    check_layers(back, TEXT_TEXT["layer_types"])
    if torch.equal(back.lm_head.weight, text_model(back).embed_tokens.weight):
        raise ValueError("the head is tied")
    blob, entries = cases(back, TINY_SEED + 1)
    manifest = {
        "dataset": "tiny-qwen35/text",
        "note": "seeded Qwen3.5 as Qwen3_5ForCausalLM (model_type qwen3_5_text, the "
                "settings at the top of config.json, tensors under model., lm_head of "
                "its own), 5 layers: linear, full, linear, linear, full; 2 key heads "
                "read by 4 value heads (each key head by two neighbours: "
                "repeat_interleave), a convolution of 3, half of each head rotating; "
                "otherwise as tiny-qwen35; the same kinds of cases",
        "config": json.load(open(os.path.join(folder, "config.json"), encoding="utf-8")),
        "safetensors_sha256": mr.sha256_file(st),
        "seed": TINY_SEED + 1,
    }
    finish(folder, back, blob, entries, manifest)


QWEN35_ID = "Qwen/Qwen3.5-0.8B"
QWEN35_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
QWEN35_SHA256 = {
    "model.safetensors-00001-of-00001.safetensors":
        "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    "tokenizer.json":
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
}
QWEN35_PROMPTS = [
    ("raw", "The capital of France is"),
    ("german", "Die Handwerkskammer prüft die Rechnung des Betriebs, weil"),
    ("code", "def fibonacci(n):\n    "),
    ("chat", None),  # the chat template of the revision, thinking off
]
QWEN35_CHAT = [{"role": "user", "content": "What is 17 times 23? Answer briefly."}]
QWEN35_GREEDY = 24


def qwen35_folder():
    folder = os.environ.get("LAI_QWEN35_DIR")
    if folder:
        return folder
    from huggingface_hub import snapshot_download
    return snapshot_download(QWEN35_ID, revision=QWEN35_REVISION)


def qwen35_08b():
    print("Qwen3.5-0.8B")
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    folder = qwen35_folder()
    for name, want in QWEN35_SHA256.items():
        got = mr.sha256_file(os.path.join(folder, name))
        if got != want:
            raise ValueError(f"{name}: sha256 {got}, pinned {want}")
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        conf = json.load(f)
    tok = AutoTokenizer.from_pretrained(folder)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        folder, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), conf["text_config"]["eos_token_id"]]
    blob = mr.Blob()
    entries = []
    for k, (kind, text) in enumerate(QWEN35_PROMPTS):
        if text is None:
            rendered = tok.apply_chat_template(QWEN35_CHAT, add_generation_prompt=True,
                                               tokenize=False, enable_thinking=False)
            ids = tok(rendered, add_special_tokens=False)["input_ids"]
        else:
            rendered = text
            ids = tok(text, add_special_tokens=False)["input_ids"]
        print(f"  case {k} ({kind}): {len(ids)} tokens", flush=True)
        x = torch.tensor([ids])
        taken = {}
        hooks = []

        def keep(name):
            def hook(_mod, _inp, out):
                o = out[0] if isinstance(out, tuple) else out
                taken[name] = o[0, -1].detach().float().clone()
            return hook

        inner = text_model(model)
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
        greedy, margins = q3.greedy(model, ids, QWEN35_GREEDY, eos)
        blob.add(f"case{k}.ids", np.asarray(ids, dtype=np.int32))
        blob.add(f"case{k}.logits_last", logits)
        for name, v in taken.items():
            blob.add(f"case{k}.last.{name}", v.numpy(), "the state at the last position")
        blob.add(f"case{k}.greedy", np.asarray(greedy, dtype=np.int32))
        blob.add(f"case{k}.greedy_margin", np.asarray(margins, dtype=np.float32))
        entries.append({"case": k, "kind": kind, "text": rendered, "tokens": len(ids),
                        "greedy_text": tok.decode(greedy), "min_margin": min(margins)})
    manifest = {
        "dataset": "qwen35-0.8b",
        "note": "Qwen/Qwen3.5-0.8B (Qwen3_5ForConditionalGeneration, text only), BF16 "
                "weights upcast exactly, computed in float32, eager attention, one "
                "thread, the torch paths of the delta rule. Per case: ids (tokenizer.json "
                "of the revision, no special tokens added; case 3 the chat template with "
                "the generation prompt and enable_thinking false), logits_last, "
                "last.embed / last.layerNN / last.norm: the row of the last position, "
                "greedy up to 24 tokens with the cache (eos: <|im_end|> and "
                "<|endoftext|>) and the top-2 margin per step",
        "model": {"id": QWEN35_ID, "revision": QWEN35_REVISION, "sha256": QWEN35_SHA256},
        "eos": eos,
        "config": conf,
        "environment": mr.environment(),
        "cases": entries,
    }
    blob.write(os.path.join(mr.OUT, "qwen35-0.8b"), manifest)


TARGETS = {"tiny-qwen35": tiny_qwen35, "tiny-qwen35-text": tiny_qwen35_text,
           "qwen35-0.8b": qwen35_08b}

if __name__ == "__main__":
    want = sys.argv[1:] or list(TARGETS)
    for t in want:
        if t not in TARGETS:
            raise SystemExit(f"unknown target {t}; known: {', '.join(TARGETS)}")
    torch.set_num_threads(1)
    for t in want:
        TARGETS[t]()
