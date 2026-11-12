#!/usr/bin/env python3
"""Reference data for phase D slice D6b of the native inference engine: the published
Gemma 4 checkpoints on the CPU.

Targets (none by default - each needs its model, tens of GB; name them):
  gemma4-26b-a4b   tests/data/infer-ref/gemma4-26b-a4b: google/gemma-4-26B-A4B-it
  gemma4-31b       tests/data/infer-ref/gemma4-31b: google/gemma-4-31B-it
  gemma4-12b       tests/data/infer-ref/gemma4-12b: google/gemma-4-12B-it (model_type
                   gemma4_unified - the text model is a dense Gemma 4)
  upcast-check     no dataset: tiny-gemma4 (both folders) loaded in float32 and loaded in
                   BF16 with upcast_gemma give the same bits - the experts included

Per case: the ids, the logits of the last position, the row of the last position of
every state and a greedy continuation, as make_reference_d7.py writes them for Qwen3.5.
A model is loaded in BF16 and every matrix (Linear, the embedding, the experts) taken to
float32 only inside its product (upcast_gemma): the arithmetic of a float32 load in half
the RAM. LAI_GEMMA_F32=1 loads it in float32 instead and writes <dataset>-f32 - to check
the upcast path on a real checkpoint (the 12B: 48 GB in float32).

The folders: $LAI_GEMMA4_26B_DIR, $LAI_GEMMA4_31B_DIR, $LAI_GEMMA4_12B_DIR, by default
vendor/models/gemma-4-26B-A4B-it, gemma-4-31B-it, gemma-4-12B-it (nothing is
downloaded; the pinned hashes are checked). Same environment and format as
make_reference.py; LAI_THREADS sets the torch threads (default 1; the datasets were
made with 16 - the 12B gives the same bits in both load paths with them).
"""
import json
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_reference as mr  # noqa: E402
import make_reference_qwen3 as q3  # noqa: E402

MODELS = {
    "gemma4-26b-a4b": {
        "id": "google/gemma-4-26B-A4B-it",
        "revision": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        "env": "LAI_GEMMA4_26B_DIR", "folder": "gemma-4-26B-A4B-it",
        "sha256": {
            "model-00001-of-00002.safetensors":
                "1127684971bbca40465435a5cad69d67ad603bf5e61c6dfd5561fae4a3bcfdb3",
            "model-00002-of-00002.safetensors":
                "aab47033e1e8a492ef8e581efae1cf36478d0433567e7729b3c1728bc8970db7",
            "tokenizer.json":
                "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        },
    },
    "gemma4-31b": {
        "id": "google/gemma-4-31B-it",
        "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
        "env": "LAI_GEMMA4_31B_DIR", "folder": "gemma-4-31B-it",
        "sha256": {
            "model-00001-of-00002.safetensors":
                "eeef8791537bc04f110967c513149e037d2a9ae97d49add7291ebfa62806bbfa",
            "model-00002-of-00002.safetensors":
                "018912220f559f7025d60333e0996183cd538aa77ad6f4988a89ce47be681f10",
            "tokenizer.json":
                "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        },
    },
    "gemma4-12b": {
        "id": "google/gemma-4-12B-it",
        "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "env": "LAI_GEMMA4_12B_DIR", "folder": "gemma-4-12B-it",
        "sha256": {
            "model.safetensors":
                "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d",
            "tokenizer.json":
                "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        },
    },
}

GEMMA_PROMPTS = [
    ("raw", "The capital of France is"),
    ("german", "Die Handwerkskammer prüft die Rechnung des Betriebs, weil"),
    ("code", "def fibonacci(n):\n    "),
    ("chat", None),  # the chat template of the revision, thinking off
]
GEMMA_CHAT = [{"role": "user", "content": "What is 17 times 23? Answer briefly."}]
GEMMA_GREEDY = 24


def text_model(model):
    inner = model.model
    return inner.language_model if hasattr(inner, "language_model") else inner


def where_of(folder):
    """tensor name -> the safetensors file that holds it"""
    from safetensors import safe_open
    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            return json.load(f)["weight_map"]
    files = [n for n in os.listdir(folder) if n.endswith(".safetensors")]
    if len(files) != 1:
        raise ValueError(f"{folder}: no index and {len(files)} safetensors files")
    with safe_open(os.path.join(folder, files[0]), "pt") as f:
        return {k: files[0] for k in f.keys()}


def upcast_gemma(model, folder):
    """the model loaded in BF16, computing as if loaded in float32: the matrices of every
    Linear, the embedding and the experts stay BF16 and go to float32 (exactly) inside
    their product; every other parameter is read again from the files in its stored
    type and taken to float32; the buffers are what a float32 load has (the embedding
    scale computed again in float32 - a BF16 load rounds sqrt(hidden) -, a loaded buffer
    such as layer_scalar widened exactly)
    - only the text model is read again: the vision and audio parts (whose names
    transformers maps from other ones in the files) are widened in place - a text
    request does not reach them"""
    import torch.nn.functional as F
    from safetensors import safe_open
    big = set()

    def linear(self, x):
        b = None if self.bias is None else self.bias.float()
        return F.linear(x, self.weight.float(), b)

    def embedding(self, ids):
        e = F.embedding(ids, self.weight, self.padding_idx).float()
        if hasattr(self, "scalar_embed_scale"):  # Gemma4TextScaledWordEmbedding
            e = e * torch.tensor(self.scalar_embed_scale, dtype=torch.float32)
        return e

    def experts(self, hidden_states, top_k_index, top_k_weights):
        # Gemma4TextExperts.forward (eager) with the slices of the expert in float32
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            mask = mask.permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit:
            e = e[0]
            if e == self.num_experts:
                continue
            pos, tok = torch.where(mask[e])
            cur = hidden_states[tok]
            gate, up = F.linear(cur, self.gate_up_proj[e].float()).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = F.linear(h, self.down_proj[e].float())
            h = h * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, h.to(final.dtype))
        return final

    seen_experts = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear):
            mod.forward = types.MethodType(linear, mod)
            big.add(id(mod.weight))
        elif isinstance(mod, torch.nn.Embedding):
            mod.forward = types.MethodType(embedding, mod)
            big.add(id(mod.weight))
        elif type(mod).__name__ == "Gemma4TextExperts":
            mod.forward = types.MethodType(experts, mod)
            big.add(id(mod.gate_up_proj))
            big.add(id(mod.down_proj))
            seen_experts += 1
    for p in model.parameters():
        if id(p) in big and p.dtype != torch.bfloat16:
            raise ValueError(f"a matrix in {p.dtype}")
    where = where_of(folder)
    inner = text_model(model)
    prefix = next(n for n, m in model.named_modules() if m is inner) + "."
    small = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if id(p) in big:
                continue
            if not name.startswith(prefix):
                p.data = p.data.float()  # not on the text path
                continue
            if name not in where:
                raise ValueError(f"{name}: not in the files")
            small.setdefault(where[name], []).append((name, p))
    with torch.no_grad():
        for file, params in small.items():
            with safe_open(os.path.join(folder, file), "pt") as f:
                for name, p in params:
                    t = f.get_tensor(name)
                    if t.shape != p.shape:
                        raise ValueError(f"{name}: {tuple(t.shape)} in the file")
                    p.data = t.float()
        persistent = set(model.state_dict().keys())
        for name, b in list(model.named_buffers()):
            if not b.is_floating_point() or b.dtype == torch.float32:
                continue
            leaf = name.rsplit(".", 1)[-1]
            owner = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            if leaf == "embed_scale" and hasattr(owner, "scalar_embed_scale"):
                setattr(owner, leaf, torch.tensor(owner.scalar_embed_scale,
                                                  dtype=torch.float32))
            elif not name.startswith(prefix):
                setattr(owner, leaf, b.float())  # not on the text path
            elif name in persistent:
                if name not in where:
                    raise ValueError(f"buffer {name}: not in the files")
                with safe_open(os.path.join(folder, where[name]), "pt") as f:
                    setattr(owner, leaf, f.get_tensor(name).float())
            else:
                raise ValueError(f"buffer {name} in {b.dtype}: no float32 value known")
        for name, b in model.named_buffers():
            if b.is_floating_point() and b.dtype != torch.float32:
                raise ValueError(f"buffer {name} still in {b.dtype}")
    return seen_experts


def load(folder, f32):
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        folder, dtype=torch.float32 if f32 else torch.bfloat16,
        attn_implementation="eager", experts_implementation="eager")
    model.eval()
    experts = 0
    if not f32:
        experts = upcast_gemma(model, folder)
    return model, experts


def gemma_real(dataset):
    m = MODELS[dataset]
    print(m["id"], flush=True)
    folder = os.environ.get(m["env"], os.path.join(mr.ROOT, "vendor", "models", m["folder"]))
    for name, want in m["sha256"].items():
        got = mr.sha256_file(os.path.join(folder, name))
        if got != want:
            raise ValueError(f"{name}: sha256 {got}, pinned {want}")
    files = {name: mr.sha256_file(os.path.join(folder, name))
             for name in ["config.json", "tokenizer_config.json", "chat_template.jinja",
                          "generation_config.json"]}
    f32 = os.environ.get("LAI_GEMMA_F32") == "1"
    if f32:
        dataset += "-f32"
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        conf = json.load(f)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(folder)
    model, experts = load(folder, f32)
    print(f"  loaded ({'float32' if f32 else 'BF16, upcast'}; {experts} expert blocks)",
          flush=True)
    eos = [tok.convert_tokens_to_ids("<turn|>"), conf["text_config"]["eos_token_id"]]
    if eos[0] is None or eos[0] == tok.unk_token_id:
        raise ValueError("no <turn|> token")
    blob = mr.Blob()
    entries = []
    for k, (kind, text) in enumerate(GEMMA_PROMPTS):
        if text is None:
            rendered = tok.apply_chat_template(GEMMA_CHAT, add_generation_prompt=True,
                                               tokenize=False, enable_thinking=False)
            ids = tok(rendered, add_special_tokens=False)["input_ids"]
        else:
            # plain text after <bos>: the tokenizer of Gemma 4 adds none (its
            # post_processor is empty), the model is lost without it
            rendered = text
            ids = [tok.bos_token_id] + tok(text, add_special_tokens=False)["input_ids"]
            if tok.bos_token_id != 2 or ids[1] == 2:
                raise ValueError("unexpected <bos>")
        print(f"  case {k} ({kind}): {len(ids)} tokens", flush=True)
        x = torch.tensor([ids])
        taken = {}
        hooks = []

        def keep(name):
            def hook(_mod, _inp, out):
                o = out[0] if isinstance(out, tuple) else out
                if o.dtype != torch.float32:
                    raise ValueError(f"{name}: {o.dtype}")
                taken[name] = o[0, -1].detach().float().clone()
            return hook

        inner = text_model(model)
        routes = []

        def route(_mod, _inp, out):
            # Gemma4TextRouter: (probabilities, top_k_weights, top_k_index), one row per
            # token - the weights renormed and times per_expert_scale, as the experts
            # take them
            _, w, idx = out
            if w.dtype != torch.float32:
                raise ValueError(f"router weights in {w.dtype}")
            routes.append((idx.detach().clone(), w.detach().clone()))

        hooks.append(inner.embed_tokens.register_forward_hook(keep("embed")))
        for i, layer in enumerate(inner.layers):
            hooks.append(layer.register_forward_hook(keep(f"layer{i:02d}")))
            if getattr(layer, "router", None) is not None:
                hooks.append(layer.router.register_forward_hook(route))
        hooks.append(inner.norm.register_forward_hook(keep("norm")))
        try:
            with torch.no_grad():
                out = model(input_ids=x, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        if out.logits.dtype != torch.float32:
            raise ValueError(f"logits: {out.logits.dtype}")
        logits = out.logits[0, -1].float().numpy()
        greedy, margins = q3.greedy(model, ids, GEMMA_GREEDY, eos)
        blob.add(f"case{k}.ids", np.asarray(ids, dtype=np.int32))
        blob.add(f"case{k}.logits_last", logits)
        for name, v in taken.items():
            blob.add(f"case{k}.last.{name}", v.numpy(), "the state at the last position")
        if routes:
            if len(routes) != len(inner.layers):
                raise ValueError(f"{len(routes)} routed layers")
            ids_r = torch.stack([r[0] for r in routes]).to(torch.int32).numpy()
            w_r = torch.stack([r[1] for r in routes]).float().numpy()
            blob.add(f"case{k}.route_ids", ids_r,
                     "the experts of every layer and token [layers, tokens, top_k], in "
                     "the order of torch.topk")
            blob.add(f"case{k}.route_weights", w_r,
                     "their weights (renormed, times per_expert_scale)")
        blob.add(f"case{k}.greedy", np.asarray(greedy, dtype=np.int32))
        blob.add(f"case{k}.greedy_margin", np.asarray(margins, dtype=np.float32))
        entries.append({"case": k, "kind": kind, "text": rendered, "tokens": len(ids),
                        "greedy_text": tok.decode(greedy), "min_margin": min(margins)})
        print(f"    greedy: {tok.decode(greedy)!r}", flush=True)
    how = ("loaded in float32" if f32 else
           "loaded in BF16, every matrix (Linear, embedding, experts) taken to float32 "
           "(exactly) inside its product, every other parameter and buffer in float32 - "
           "the arithmetic of a float32 load")
    manifest = {
        "dataset": dataset,
        "note": f"{m['id']} ({type(model).__name__}, text only), {how}, eager attention "
                f"and experts, {torch.get_num_threads()} threads. Per case: ids (tokenizer.json of the revision; "
                "cases 0-2 the text after <bos> (id 2, put there here: the tokenizer adds "
                "none), case 3 the chat template "
                "with the generation prompt and enable_thinking false), logits_last (after "
                "the soft cap), last.embed / last.layerNN / last.norm: the row of the last "
                "position, for a model with experts route_ids / route_weights (the choice of its router in every layer for every token of the prompt), greedy up to 24 tokens with the cache (eos: <turn|> and the "
                "eos_token_id of text_config) and the top-2 margin per step",
        "model": {"id": m["id"], "revision": m["revision"], "sha256": m["sha256"],
                  "files": files},
        "eos": eos,
        "config": conf,
        "environment": mr.environment(),
        "cases": entries,
    }
    blob.write(os.path.join(mr.OUT, dataset), manifest)


def upcast_check():
    """tiny-gemma4 (experts) and its dense folder: float32 load and BF16 load with
    upcast_gemma give the same logits and states, bit for bit"""
    for sub in ["tiny-gemma4", os.path.join("tiny-gemma4", "dense")]:
        folder = os.path.join(mr.OUT, sub)
        a, _ = load(folder, True)
        b, experts = load(folder, False)
        ids = torch.tensor([[2, 5, 9, 17, 33, 41, 60, 7, 3, 88, 12]])
        with torch.no_grad():
            la = a(input_ids=ids, use_cache=False, output_hidden_states=True)
            lb = b(input_ids=ids, use_cache=False, output_hidden_states=True)
        same = torch.equal(la.logits, lb.logits) and all(
            torch.equal(x, y) for x, y in zip(la.hidden_states, lb.hidden_states))
        print(f"{sub}: {experts} expert blocks, {'SAME-BITS' if same else 'DIFFERENT'}"
              f" (max {float((la.logits - lb.logits).abs().max()):.3g})", flush=True)
        if not same:
            raise SystemExit(1)


TARGETS = {"gemma4-26b-a4b": lambda: gemma_real("gemma4-26b-a4b"),
           "gemma4-31b": lambda: gemma_real("gemma4-31b"),
           "gemma4-12b": lambda: gemma_real("gemma4-12b"),
           "upcast-check": upcast_check}

if __name__ == "__main__":
    want = sys.argv[1:]
    if not want:
        raise SystemExit(f"name the targets: {', '.join(TARGETS)}")
    for t in want:
        if t not in TARGETS:
            raise SystemExit(f"unknown target {t}; known: {', '.join(TARGETS)}")
    torch.set_num_threads(int(os.environ.get("LAI_THREADS", "1")))
    for t in want:
        TARGETS[t]()
