#!/usr/bin/env python3
"""Reference data for phase D slice D1b of the native inference engine (Qwen2.5).

Targets (all of them when called without arguments):
  tiny-qwen2     tests/data/infer-ref/tiny-qwen2: a seeded 2-layer Qwen2 (bias on q, k
                 and v, no per-head norm, head_dim = hidden / heads, BF16 weights)
                 with every state, the logits and greedy continuations; gguf/: the
                 same as llama.cpp writes a qwen2 file (Q8_0)
  qwen25-chat    tests/data/infer-ref/qwen25-chat: the chat template of
                 Qwen2.5-0.5B-Instruct over the cases of the Qwen3 gate
  qwen25         tests/data/infer-ref/qwen25: Qwen2.5-0.5B-Instruct in FP32 - states,
                 logits and greedy continuations of a fixed prompt set

Also copies the pinned model files to vendor/models/qwen2.5-0.5b-instruct/ and the
GGUF files to vendor/models/ (gitignored). Same environment, format and rules as
make_reference.py; the helpers of make_reference_qwen3.py are reused.
"""
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_reference as mr  # noqa: E402
import make_reference_qwen3 as q3  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_FILES = ["config.json", "generation_config.json", "tokenizer.json",
               "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors",
               "LICENSE"]
GGUF_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
GGUF_REVISION = "9217f5db79a29953eb74d5343926648285ec7e67"
GGUF_FILES = ["qwen2.5-0.5b-instruct-q8_0.gguf", "qwen2.5-0.5b-instruct-q4_k_m.gguf"]
MODELS = os.path.join(mr.ROOT, "vendor", "models")
MODEL_DIR = os.environ.get("LAI_QWEN25_DIR") or os.path.join(MODELS,
                                                              "qwen2.5-0.5b-instruct")
TINY_SEED = 20261019
GREEDY_TOKENS = 48


def copy_checked(src, dst):
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)
            and mr.sha256_file(dst) == mr.sha256_file(src)):
        shutil.copyfile(src, dst)


def fetch_model():
    from huggingface_hub import snapshot_download, hf_hub_download
    snapshot = snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=MODEL_FILES)
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name in MODEL_FILES:
        copy_checked(os.path.join(snapshot, name), os.path.join(MODEL_DIR, name))
    for name in GGUF_FILES:
        copy_checked(hf_hub_download(GGUF_ID, name, revision=GGUF_REVISION),
                     os.path.join(MODELS, name))
    return snapshot


def model_info(snapshot):
    return {"id": MODEL_ID, "revision": REVISION,
            "files": {n: mr.sha256_file(os.path.join(snapshot, n)) for n in MODEL_FILES},
            "gguf": {"id": GGUF_ID, "revision": GGUF_REVISION,
                     "files": {n: mr.sha256_file(os.path.join(MODELS, n))
                               for n in GGUF_FILES}}}


# ---------------------------------------------------------------- tiny Qwen2

def tiny_qwen2():
    print("tiny Qwen2")
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from safetensors.torch import save_file
    torch.manual_seed(TINY_SEED)
    cfg = Qwen2Config(
        vocab_size=96, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=True,
        hidden_act="silu", bos_token_id=1, eos_token_id=2, pad_token_id=0,
        use_sliding_window=False, attn_implementation="eager")
    model = Qwen2ForCausalLM(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".bias"):
                p.copy_(0.3 * torch.randn_like(p))  # biases that move the result
            elif p.dim() == 1:  # the RMSNorm weights start as ones: make them count
                p.copy_(1.0 + 0.25 * torch.randn_like(p))
            else:
                p.copy_(0.08 * torch.randn_like(p))
            p.copy_(p.to(torch.bfloat16).to(torch.float32))  # BF16-exact, like the model
    model.eval()
    names = [n for n, _ in model.named_parameters()]
    for want in ("model.layers.0.self_attn.q_proj.bias", "model.layers.0.self_attn.k_proj.bias",
                 "model.layers.0.self_attn.v_proj.bias"):
        if want not in names:
            raise ValueError(f"no {want}: not the Qwen2 layout")
    if any(".o_proj.bias" in n or "q_norm" in n for n in names):
        raise ValueError("an o_proj bias or a per-head norm: not the Qwen2 layout")
    folder = os.path.join(mr.OUT, "tiny-qwen2")
    os.makedirs(folder, exist_ok=True)
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()
          if k != "lm_head.weight"}
    st = os.path.join(folder, "model.safetensors")
    save_file(sd, st, metadata={"format": "pt"})
    d = cfg.to_dict()
    # the keys of Qwen2.5's own config.json: no head_dim (hidden / heads)
    keep = ["architectures", "vocab_size", "hidden_size", "intermediate_size",
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "max_position_embeddings", "rms_norm_eps", "rope_theta", "rope_parameters",
            "tie_word_embeddings", "hidden_act", "bos_token_id", "eos_token_id",
            "pad_token_id", "model_type", "use_sliding_window", "sliding_window",
            "max_window_layers", "rope_scaling"]
    d["architectures"] = ["Qwen2ForCausalLM"]
    d["torch_dtype"] = "bfloat16"
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({k: d[k] for k in keep + ["torch_dtype"] if k in d}, f, indent=1)
        f.write("\n")
    back = Qwen2ForCausalLM.from_pretrained(folder, dtype=torch.float32,
                                            attn_implementation="eager")
    back.eval()
    rng = np.random.default_rng(TINY_SEED)
    seqs = [rng.integers(3, 96, size=n).tolist() for n in (5, 9, 13)]
    blob = mr.Blob()
    entries = []
    for k, ids in enumerate(seqs):
        a = q3.record(back, ids)
        b = q3.record(back, ids)
        for name in a:
            if not np.array_equal(a[name], b[name]):
                raise ValueError(f"tiny {k} {name}: two runs differ")
        e = {"case": k, "ids": ids, "tokens": len(ids)}
        blob.add(f"case{k}.ids", np.array(ids, dtype=np.int64))
        for name, v in a.items():
            blob.add(f"case{k}.{name}", v)
        new, margins = q3.greedy(back, ids, 16, [2])
        if q3.greedy_no_cache(back, ids, 16, [2]) != new:
            raise ValueError(f"tiny {k}: greedy with and without cache differ")
        blob.add(f"case{k}.greedy", np.array(new, dtype=np.int64))
        blob.add(f"case{k}.greedy_margin", np.array(margins, dtype=np.float32))
        e["greedy"] = new
        e["min_margin"] = min(margins)
        entries.append(e)
    manifest = {
        "dataset": "tiny-qwen2",
        "note": "seeded Qwen2, 2 layers, hidden 64, 4 heads / 2 KV heads, head_dim 16 "
                "(hidden / heads: config.json has no head_dim, as Qwen2.5's), bias on "
                "q, k and v (none on o), no per-head norm, tied embeddings, BF16 "
                "weights. The contract is model.safetensors + config.json. Per case: "
                "embed, layerNN, layer00_attn, layer00_mlp, norm, logits (all "
                "positions), greedy (16 tokens or up to eos 2) with the top-2 margin",
        "config": json.load(open(os.path.join(folder, "config.json"), encoding="utf-8")),
        "safetensors_sha256": mr.sha256_file(st),
        "environment": mr.environment(),
        "cases": entries,
    }
    blob.write(folder, manifest)


QWEN2_GGUF_LAYER = [
    ("self_attn.q_proj.weight", "attn_q.weight"), ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.v_proj.weight", "attn_v.weight"), ("self_attn.o_proj.weight", "attn_output.weight"),
    ("self_attn.q_proj.bias", "attn_q.bias"), ("self_attn.k_proj.bias", "attn_k.bias"),
    ("self_attn.v_proj.bias", "attn_v.bias"),
    ("mlp.gate_proj.weight", "ffn_gate.weight"), ("mlp.up_proj.weight", "ffn_up.weight"),
    ("mlp.down_proj.weight", "ffn_down.weight"), ("input_layernorm.weight", "attn_norm.weight"),
    ("post_attention_layernorm.weight", "ffn_norm.weight")]


def tiny_qwen2_gguf():
    """tiny-qwen2 as llama.cpp writes a qwen2 file: no attention.key_length (head_dim
    is embedding_length / head_count), the biases as F32 tensors, a tied head left
    out; the matrices quantized by ggml's reference Q8_0 quantizer"""
    print("tiny-qwen2-gguf")
    from safetensors.torch import load_file
    g = mr.Ggml()
    src = os.path.join(mr.OUT, "tiny-qwen2")
    folder = os.path.join(src, "gguf")
    os.makedirs(folder, exist_ok=True)
    cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    sd = {k: v.float().numpy() for k, v in
          load_file(os.path.join(src, "model.safetensors")).items()}
    theta = cfg["rope_parameters"]["rope_theta"] if "rope_parameters" in cfg \
        else cfg["rope_theta"]
    kv = [("general.architecture", "str", "qwen2"),
          ("general.name", "str", "tiny-qwen2"),
          ("qwen2.block_count", "u32", cfg["num_hidden_layers"]),
          ("qwen2.context_length", "u32", cfg["max_position_embeddings"]),
          ("qwen2.embedding_length", "u32", cfg["hidden_size"]),
          ("qwen2.feed_forward_length", "u32", cfg["intermediate_size"]),
          ("qwen2.attention.head_count", "u32", cfg["num_attention_heads"]),
          ("qwen2.attention.head_count_kv", "u32", cfg["num_key_value_heads"]),
          ("qwen2.rope.freq_base", "f32", theta),
          ("qwen2.attention.layer_norm_rms_epsilon", "f32", cfg["rms_norm_eps"])]
    ts = []

    def add(name, a, kind):
        a = np.ascontiguousarray(a, dtype=np.float32)
        ne = list(reversed(a.shape))
        if kind == "f32":
            ts.append((name, 0, ne, a.tobytes()))
        else:
            ts.append((name, 8, ne, b"".join(g.quant_ref(r) for r in a)))

    add("token_embd.weight", sd["model.embed_tokens.weight"], "q8")
    add("output_norm.weight", sd["model.norm.weight"], "f32")
    for i in range(cfg["num_hidden_layers"]):
        for hf, gg in QWEN2_GGUF_LAYER:
            a = sd[f"model.layers.{i}.{hf}"]
            add(f"blk.{i}.{gg}", a, "f32" if a.ndim == 1 else "q8")
    files = {"q8_0.gguf": (kv, ts),
             # a bias missing: the loader must not run without it
             "bad-nobias.gguf": (kv, [t for t in ts if t[0] != "blk.1.attn_k.bias"])}
    for name, (kvs, tens) in files.items():
        mr.gguf_write(os.path.join(folder, name), kvs, tens)
    manifest = {"dataset": "tiny-qwen2/gguf", "environment": mr.environment(),
                "ggml": {"dir": os.path.relpath(mr.GGML_DIR, mr.ROOT), "sha256": g.libs},
                "note": "tiny-qwen2 as a qwen2 GGUF file: matrices and token_embd in Q8_0 "
                        "from ggml's reference quantizer, norms and biases F32, no "
                        "attention.key_length; bad-nobias: blk.1.attn_k.bias left out",
                "files": {n: mr.sha256_file(os.path.join(folder, n)) for n in sorted(files)}}
    with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)


# ---------------------------------------------------------------- Qwen2.5 chat

def chat_to_hf25(messages, tools):
    """as make_reference_qwen3.chat_to_hf, the arguments of a tool call as an
    object: the Qwen2.5 template writes them through tojson (a string would come
    out quoted), as an OpenAI client's JSON text decoded"""
    hm, ht = q3.chat_to_hf(messages, tools)
    for h in hm:
        for c in h.get("tool_calls", []):
            c["function"]["arguments"] = json.loads(c["function"]["arguments"])
    return hm, ht


def chat(snapshot):
    print("Qwen2.5 chat template cases")
    hf = q3.hf_tokenizer(snapshot)
    flat, offsets, cases = [], [], []
    for cid, messages, tools, gen, think in q3.chat_cases():
        hm, ht = chat_to_hf25(messages, tools)
        text = hf.apply_chat_template(hm, tools=ht or None, add_generation_prompt=gen,
                                      tokenize=False)
        ids = hf(text, add_special_tokens=False)["input_ids"]
        offsets.append(len(flat))
        flat += ids
        cases.append({"id": cid, "messages": messages, "tools": tools,
                      "add_generation_prompt": gen, "rendered": text, "tokens": len(ids)})
    offsets.append(len(flat))
    blob = mr.Blob()
    blob.add("ids", np.array(flat, dtype=np.int64), "all cases concatenated")
    blob.add("offsets", np.array(offsets, dtype=np.int64),
             "case i = ids[offsets[i] .. offsets[i+1]-1]")
    manifest = {
        "dataset": "qwen25-chat",
        "note": "the chat template of Qwen2.5-0.5B-Instruct (tokenizer_config.json of the "
                "pinned revision) over the cases of qwen3-chat, rendered by transformers "
                "and tokenized by the HF tokenizer; enable_thinking does not exist there. "
                "Tool call arguments are raw JSON text in the messages and given to the "
                "template as objects (json.loads), which it writes through tojson; tools "
                "as in qwen3-chat",
        "model": model_info(snapshot),
        "environment": mr.environment(),
        "cases": cases,
    }
    blob.write(os.path.join(mr.OUT, "qwen25-chat"), manifest)


# ---------------------------------------------------------------- Qwen2.5-0.5B-Instruct

def prompts():
    """(id, messages or raw text, tools, keep states)"""
    nl = "\n"
    return [
        ("raw_fox", "The quick brown fox", [], True),
        ("chat_de", [q3.msg("user", "Wie verlege ich ein Kupferrohr in der Wand?")], [], True),
        ("chat_en", [q3.msg("user", "What is the capital of France? Answer in one word.")],
         [], False),
        ("chat_system", [q3.msg("system", "Du bist ein hilfreicher Assistent für "
                                          "Handwerker."),
                         q3.msg("user", "Nenne drei Werkzeuge zum Fliesenlegen.")], [], False),
        ("raw_code", "def fibonacci(n):" + nl, [], False),
        ("chat_tool", [q3.msg("user", "Wie ist das Wetter in Berlin?")], [q3.WEATHER], False),
    ]


def qwen25(snapshot):
    print("Qwen2.5-0.5B-Instruct (FP32)")
    from transformers import Qwen2ForCausalLM
    hf = q3.hf_tokenizer(snapshot)
    model = Qwen2ForCausalLM.from_pretrained(snapshot, dtype=torch.float32,
                                             attn_implementation="eager")
    model.eval()
    # plain argmax: generation_config.json asks for a repetition penalty of 1.1,
    # which generate applies even without sampling
    model.generation_config.repetition_penalty = 1.0
    with open(os.path.join(snapshot, "generation_config.json"), encoding="utf-8") as f:
        eos = json.load(f)["eos_token_id"]
    blob = mr.Blob()
    entries = []
    for pid, source, tools, states in prompts():
        if isinstance(source, str):
            text = source
        else:
            hm, ht = chat_to_hf25(source, tools)
            text = hf.apply_chat_template(hm, tools=ht or None, add_generation_prompt=True,
                                          tokenize=False)
        ids = hf(text, add_special_tokens=False)["input_ids"]
        a = q3.record(model, ids)
        b = q3.record(model, ids)
        if not np.array_equal(a["logits"], b["logits"]):
            raise ValueError(f"{pid}: two runs differ")
        new, margins = q3.greedy(model, ids, GREEDY_TOKENS, eos)
        again, _ = q3.greedy(model, ids, GREEDY_TOKENS, eos)
        if again != new:
            raise ValueError(f"{pid}: two greedy runs differ")
        nocache = q3.greedy_no_cache(model, ids, GREEDY_TOKENS, eos)
        e = {"id": pid, "tokens": len(ids), "text": text,
             "greedy": new, "greedy_text": hf.decode(new),
             "min_margin": min(margins), "no_cache_equal": nocache == new}
        if not isinstance(source, str):
            e["messages"] = source
            e["tools"] = tools
        blob.add(f"{pid}.ids", np.array(ids, dtype=np.int64))
        if states:
            for name, v in a.items():
                if name != "logits":
                    blob.add(f"{pid}.{name}", v)
            e["states"] = True
        blob.add(f"{pid}.logits_last", a["logits"][-1],
                 "logits of the last prompt position (the first generated token)")
        blob.add(f"{pid}.greedy", np.array(new, dtype=np.int64))
        blob.add(f"{pid}.greedy_margin", np.array(margins, dtype=np.float32),
                 "top-1 minus top-2 logit per step: small values mark near ties")
        print(f"  {pid}: {len(ids)} + {len(new)} tokens, min margin {min(margins):.4f}, "
              f"no cache equal {nocache == new}")
        entries.append(e)
    manifest = {
        "dataset": "qwen25",
        "note": "Qwen2.5-0.5B-Instruct in FP32 (BF16 weights upcast exactly), eager "
                "attention, one thread. Per prompt: ids (chat prompts through the chat "
                "template with the generation prompt), logits_last, greedy (up to "
                f"{GREEDY_TOKENS} tokens or eos {eos}, HF KV cache) with the top-2 margin "
                "per step. With states: embed, layerNN outputs, layer00_attn, "
                "layer00_mlp, norm",
        "model": model_info(snapshot),
        "environment": mr.environment(),
        "prompts": entries,
    }
    blob.write(os.path.join(mr.OUT, "qwen25"), manifest)


def main():
    mr.setup_determinism()
    what = set(sys.argv[1:]) or {"tiny-qwen2", "qwen25-chat", "qwen25"}
    snapshot = fetch_model() if what - {"tiny-qwen2"} else None
    if "tiny-qwen2" in what:
        tiny_qwen2()
        tiny_qwen2_gguf()
    if "qwen25-chat" in what:
        chat(snapshot)
    if "qwen25" in what:
        qwen25(snapshot)


if __name__ == "__main__":
    main()
