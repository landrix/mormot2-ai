#!/usr/bin/env python3
"""Reference data for the native inference engine (docs/native-inference.md, phase 0).

Writes, below tests/data/infer-ref/:
  bge-m3/          short cases with every hidden state (ids, masks, positions,
                   25 states, CLS before and after L2) plus a right-padded batch
  bge-m3-embed-de/ FP32 dense vectors of the embed-de corpus and queries
  tiny-xlmr/       a seeded 2-layer XLM-R (config + safetensors weights) and its outputs
  ops/             operator fixtures (linear, layernorm, masked softmax, gelu, l2)

and converts the pinned BGE-M3 pytorch_model.bin into an F32 model.safetensors outside
version control (vendor/models/bge-m3/, or $LAI_MODEL_DIR), the format phase 1a reads.

Every dataset is manifest.json + tensors.bin: raw little-endian blobs, one after the other
(4-byte aligned), each listed with dtype, shape (row-major, C order), offset, byte length
and sha256. The reference environment is CPU on WSL aarch64 (see README.md here).
"""
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import transformers
import safetensors
import tokenizers
from safetensors.torch import save_file, load_file
from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel

MODEL_ID = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MODEL_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "sentencepiece.bpe.model",
               "pytorch_model.bin", "1_Pooling/config.json", "modules.json",
               "sentence_bert_config.json"]
MAX_CASE_TOKENS = 16  # including <s> and </s>
TINY_SEED = 20261014

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "tests", "data", "infer-ref")
MODEL_DIR = os.environ.get("LAI_MODEL_DIR") or os.path.join(ROOT, "vendor", "models", "bge-m3")

# short cases: (id, text, note); at most MAX_CASE_TOKENS tokens each, checked below
CASES = [
    ("empty", "", "no text: only <s> </s>"),
    ("whitespace", "   ", "whitespace only"),
    ("trade", "Kupferrohr 15 mm in Wand verlegen", "short German trade text"),
    ("unicode", "Fuge ≥ 3 mm, Größe 30×60, 5 €",
     ">=, umlaut, sharp s, multiplication sign, euro (exactly 16 tokens)"),
    ("multilingual", "Install the radiator. Радиатор. 散热器",
     "English, Russian, Chinese"),
    ("special_literal", "Text mit <s> und <pad>", "literal special-token text: plain text "
     "for the reference tokenizer (parse_special=false); HF differs, see tokenizer_checks"),
]
BATCH = ["Kupferrohr 15 mm in Wand verlegen", "Heizkörper montieren"]
SPECIAL_LITERALS = ["<s>", "</s>", "<pad>", "<unk>", "<mask>", "a<s>b"]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Blob:
    """collects tensors into one tensors.bin and describes them in a manifest"""

    def __init__(self):
        self.parts = []
        self.size = 0
        self.entries = []

    def add(self, name, array, note=None):
        a = np.ascontiguousarray(array)
        if a.dtype == np.uint8:
            dtype = "u8"
        elif a.dtype in (np.float32, np.float64, np.float16):
            a = a.astype("<f4")
            dtype = "f32"
        elif a.dtype.kind in "iub":
            if a.size and (a.min() < -2**31 or a.max() >= 2**31):
                raise ValueError(f"{name}: value outside int32")
            a = a.astype("<i4")
            dtype = "i32"
        else:
            raise ValueError(f"{name}: unsupported dtype {a.dtype}")
        if dtype == "f32" and not np.all(np.isfinite(a)):
            raise ValueError(f"{name}: non-finite value")
        raw = a.tobytes(order="C")
        pad = (-self.size) % 4
        if pad:
            self.parts.append(b"\0" * pad)
            self.size += pad
        e = {"name": name, "dtype": dtype, "shape": list(a.shape),
             "offset": self.size, "bytes": len(raw), "sha256": sha256_bytes(raw)}
        if note:
            e["note"] = note
        self.entries.append(e)
        self.parts.append(raw)
        self.size += len(raw)

    def write(self, folder, manifest):
        os.makedirs(folder, exist_ok=True)
        data = b"".join(self.parts)
        with open(os.path.join(folder, "tensors.bin"), "wb") as f:
            f.write(data)
        manifest["format"] = 1
        manifest["blob"] = {"file": "tensors.bin", "bytes": len(data),
                            "sha256": sha256_bytes(data), "byte_order": "little",
                            "layout": "row-major (C order)"}
        manifest["tensors"] = self.entries
        with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, indent=1, ensure_ascii=True)
            f.write("\n")
        print(f"  {os.path.relpath(folder, ROOT)}: {len(self.entries)} tensors, "
              f"{len(data) / 1048576:.2f} MiB")


def environment():
    return {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "safetensors": safetensors.__version__,
        "sentencepiece": sentencepiece_version(),
        "numpy": np.__version__,
        "device": "cpu",
        "dtype": "float32",
        "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "attn_implementation": "eager",
        "mode": "eval (dropout off)",
        "generator": "tools/infer-ref/make_reference.py",
    }


def sentencepiece_version():
    import sentencepiece
    return sentencepiece.__version__


def setup_determinism():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)


def xlmr_positions(ids, pad_id):
    """XLM-R position ids: non-padding tokens count from pad_id+1, padding gets pad_id"""
    mask = (ids != pad_id).astype(np.int64)
    return np.cumsum(mask, axis=-1) * mask + pad_id


def forward(model, ids, mask):
    with torch.no_grad():
        out = model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask),
                    output_hidden_states=True)
    return [h.numpy() for h in out.hidden_states]


def dense(last_hidden):
    """BGE-M3 dense vector: CLS of the last hidden state, L2-normalised"""
    cls = last_hidden[:, 0, :].astype(np.float64)
    norm = np.linalg.norm(cls, axis=-1, keepdims=True)
    return last_hidden[:, 0, :], (cls / norm).astype(np.float32)


# ---------------------------------------------------------------- BGE-M3

def fetch_model():
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=MODEL_FILES)


def convert(snapshot):
    """pytorch_model.bin -> F32 model.safetensors, checked tensor by tensor"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    src = torch.load(os.path.join(snapshot, "pytorch_model.bin"), map_location="cpu",
                     weights_only=True)
    out = {}
    for k, v in src.items():
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"{k}: not a tensor")
        if not v.is_floating_point():
            raise ValueError(f"{k}: non-float tensor {v.dtype} - decide explicitly")
        out[k] = v.to(torch.float32).contiguous()
    dst = os.path.join(MODEL_DIR, "model.safetensors")
    # one metadata key only: safetensors writes the metadata map in no fixed order,
    # so a second key would make the file (and its recorded sha256) vary per run
    save_file(out, dst, metadata={"format": "pt"})
    back = load_file(dst)
    if set(back) != set(out):
        raise ValueError("safetensors round trip lost tensors")
    for k in out:
        if back[k].dtype != torch.float32 or back[k].shape != out[k].shape \
                or not torch.equal(back[k], out[k]):
            raise ValueError(f"{k}: safetensors round trip differs")
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "sentencepiece.bpe.model"]:
        shutil.copyfile(os.path.join(snapshot, name), os.path.join(MODEL_DIR, name))
    tensors = sorted((k, list(v.shape)) for k, v in out.items())
    return dst, tensors


def load_model(folder):
    model, info = XLMRobertaModel.from_pretrained(
        folder, add_pooling_layer=False, attn_implementation="eager",
        torch_dtype=torch.float32, output_loading_info=True)
    if info["missing_keys"]:
        raise ValueError(f"missing weights would be random: {info['missing_keys']}")
    unexpected = [k for k in info["unexpected_keys"] if not k.startswith("pooler.")]
    if unexpected:
        raise ValueError(f"unexpected weights: {unexpected}")
    model.eval()
    return model, sorted(info["unexpected_keys"])


class RefTokenizer:
    """the reference token ids: SentencePiece on sentencepiece.bpe.model

    SentencePiece never matches its control symbols (<s>, </s>, <pad>) in the input
    text - that is parse_special=false, the behaviour the native tokenizer must have.
    The HF tokenizer (tokenizer.json) does match them: '<s>' is a piece of its Unigram
    vocabulary, so neither split_special_tokens nor encode_special_tokens turn it into
    plain text. XLM-R ids are the fairseq ids: SentencePiece id + 1, except
    <unk> 0 -> 3, <s> 1 -> 0, </s> 2 -> 2; <s> ... </s> wrap every text.
    """

    def __init__(self, snapshot):
        import sentencepiece
        self.sp = sentencepiece.SentencePieceProcessor(
            model_file=os.path.join(snapshot, "sentencepiece.bpe.model"))
        self.hf = AutoTokenizer.from_pretrained(snapshot)
        self.hf_mismatches = 0

    def fairseq(self, pieces):
        special = {0: 3, 1: 0, 2: 2}
        return [special.get(i, i + 1) for i in pieces]

    def encode(self, text, expect_hf_equal=True):
        ids = [0] + self.fairseq(self.sp.encode(text)) + [2]
        hf = self.hf(text, add_special_tokens=True, truncation=False)["input_ids"]
        if expect_hf_equal and hf != ids:
            raise ValueError(f"HF and SentencePiece disagree on {text!r}: {hf} / {ids}")
        if hf != ids:
            self.hf_mismatches += 1
        return ids

    def checks(self):
        fast = self.hf.backend_tokenizer if hasattr(self.hf, "backend_tokenizer") else None
        result = []
        for text in SPECIAL_LITERALS:
            e = {"text": text,
                 "reference": self.fairseq(self.sp.encode(text)),
                 "hf_default": self.hf(text, add_special_tokens=False)["input_ids"],
                 "hf_split_special_tokens": self.hf(text, add_special_tokens=False,
                                                    split_special_tokens=True)["input_ids"]}
            if fast is not None:
                fast.encode_special_tokens = True
                e["hf_encode_special_tokens"] = fast.encode(text, add_special_tokens=False).ids
                fast.encode_special_tokens = False
            result.append(e)
        return result


def check_pooling(snapshot):
    """BGE-M3 dense = CLS + L2: verify it against the revision's own config"""
    with open(os.path.join(snapshot, "1_Pooling", "config.json"), encoding="utf-8") as f:
        pooling = json.load(f)
    modes = {k: v for k, v in pooling.items() if k.startswith("pooling_mode_")}
    if not modes.get("pooling_mode_cls_token") or any(
            v for k, v in modes.items() if k != "pooling_mode_cls_token"):
        raise ValueError(f"pooling is not CLS only: {modes}")
    with open(os.path.join(snapshot, "modules.json"), encoding="utf-8") as f:
        modules = [m["type"] for m in json.load(f)]
    if not any(t.endswith("Normalize") for t in modules):
        raise ValueError(f"no Normalize module: {modules}")
    return {"pooling": modes, "modules": modules}


def bge_m3(snapshot):
    print("BGE-M3: convert pytorch_model.bin -> model.safetensors")
    pooling_config = check_pooling(snapshot)
    st_path, st_tensors = convert(snapshot)
    tok = RefTokenizer(snapshot)
    model_bin, ignored = load_model(snapshot)
    model, _ = load_model(MODEL_DIR)
    cfg = model.config
    pad = cfg.pad_token_id

    blob = Blob()
    cases = []
    for cid, text, note in CASES:
        literal = cid == "special_literal"
        ids = np.array([tok.encode(text, expect_hf_equal=not literal)], dtype=np.int64)
        if ids.shape[1] > MAX_CASE_TOKENS:
            raise ValueError(f"case {cid}: {ids.shape[1]} tokens > {MAX_CASE_TOKENS}")
        mask = np.ones_like(ids)
        pos = xlmr_positions(ids, pad)
        hs = forward(model, ids, mask)
        hs2 = forward(model, ids, mask)
        if any(not np.array_equal(a, b) for a, b in zip(hs, hs2)):
            raise ValueError(f"case {cid}: repeated run is not bit-identical")
        hsb = forward(model_bin, ids, mask)
        if any(not np.array_equal(a, b) for a, b in zip(hs, hsb)):
            raise ValueError(f"case {cid}: converted weights change the forward")
        cls, emb = dense(hs[-1])
        p = f"case/{cid}/"
        blob.add(p + "input_ids", ids[0])
        blob.add(p + "attention_mask", mask[0])
        blob.add(p + "position_ids", pos[0])
        for k, h in enumerate(hs):
            blob.add(p + f"hidden.{k}", h[0],
                     "embedding module output (word+position+token_type, LayerNorm)"
                     if k == 0 else f"output of encoder layer {k}")
        blob.add(p + "cls", cls[0], "last hidden state, token 0, before normalisation")
        blob.add(p + "embedding", emb[0], "dense vector: cls / ||cls||")
        cases.append({"id": cid, "text": text, "note": note, "tokens": int(ids.shape[1])})

    # right-padded batch: rows must equal the single runs on their real tokens
    rows = [tok.encode(t) for t in BATCH]
    width = max(len(r) for r in rows)
    ids = np.array([r + [pad] * (width - len(r)) for r in rows], dtype=np.int64)
    mask = (ids != pad).astype(np.int64)
    hs = forward(model, ids, mask)
    cls, emb = dense(hs[-1])
    blob.add("batch/input_ids", ids)
    blob.add("batch/attention_mask", mask)
    blob.add("batch/position_ids", xlmr_positions(ids, pad))
    for k, h in enumerate(hs):
        blob.add(f"batch/hidden.{k}", h, "padded positions are not meaningful")
    blob.add("batch/embedding", emb)
    batch_diff = []
    for i, r in enumerate(rows):
        one = np.array([r], dtype=np.int64)
        single = dense(forward(model, one, np.ones_like(one))[-1])[1][0]
        blob.add(f"batch/single.{i}/embedding", single,
                 f"row {i} run alone, without padding")
        batch_diff.append(float(np.abs(single - emb[i]).max()))

    manifest = {
        "dataset": "bge-m3",
        "model": {"id": MODEL_ID, "revision": REVISION,
                  "files": {n: sha256_file(os.path.join(snapshot, n)) for n in MODEL_FILES},
                  "converted": {"file": os.path.relpath(st_path, ROOT).replace(os.sep, "/"),
                                "sha256": sha256_file(st_path), "dtype": "F32",
                                "tensors": len(st_tensors),
                                "unused_by_dense": ignored},
                  "config": {k: getattr(cfg, k) for k in [
                      "model_type", "vocab_size", "hidden_size", "num_hidden_layers",
                      "num_attention_heads", "intermediate_size", "hidden_act",
                      "max_position_embeddings", "type_vocab_size", "layer_norm_eps",
                      "pad_token_id", "bos_token_id", "eos_token_id"]}},
        "pipeline": {"pooling": "cls of last_hidden_state (not pooler_output)",
                     "normalize": "l2", "position_ids": "cumsum(ids != pad) * (ids != pad) + pad",
                     "token_type_ids": "all 0 (its embedding row is still added)",
                     "tokenizer": "SentencePiece (sentencepiece.bpe.model) with fairseq "
                                  "ids: id + 1, <unk> 0->3, <s> 1->0, </s> 2->2; "
                                  "<s> text </s>; no truncation; batch padded right",
                     "parse_special": False,
                     "hf_tokenizer_check": "equal to the HF tokenizer for every text "
                                           "without literal special-token text (checked)",
                     "sentence_transformers": pooling_config},
        "environment": environment(),
        "cases": cases,
        "batch": {"texts": BATCH, "max_abs_diff_to_single": batch_diff},
        "tokenizer_checks": tok.checks(),
        "tokenizer_checks_note": "reference = SentencePiece, literal text stays text "
                                 "(parse_special=false, what the native tokenizer must do); "
                                 "hf_default, hf_split_special_tokens and "
                                 "hf_encode_special_tokens still map '<s>' to id 0, because "
                                 "'<s>' is a piece of the Unigram vocabulary",
    }
    blob.write(os.path.join(OUT, "bge-m3"), manifest)
    return tok, model, manifest


def embed_de(tok, model, bge_manifest):
    print("BGE-M3: embed-de corpus (FP32)")
    data = os.path.join(ROOT, "tests", "data", "embed-de")
    sets = {}
    for name in ["corpus", "queries"]:
        rows = []
        with open(os.path.join(data, name + ".jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        sets[name] = rows
    blob = Blob()
    counts = {}
    for name, key in [("corpus", "docs"), ("queries", "queries")]:
        vecs, maxtok = [], 0
        for r in sets[name]:
            ids = np.array([tok.encode(r["text"])], dtype=np.int64)
            if ids.shape[1] > 512:
                raise ValueError(f"{r['id']}: {ids.shape[1]} tokens > 512")
            maxtok = max(maxtok, ids.shape[1])
            hs = forward(model, ids, np.ones_like(ids))
            vecs.append(dense(hs[-1])[1][0])
        blob.add(key, np.stack(vecs), f"row i = {name}.jsonl line i (ids in manifest)")
        counts[key] = {"rows": len(vecs), "ids": [r["id"] for r in sets[name]],
                       "max_tokens": maxtok}
    manifest = {
        "dataset": "bge-m3-embed-de",
        "note": "FP32 reference vectors of tests/data/embed-de. They do NOT replace the "
                "lembed Q8 oracle in embed-de/reference nor the nDCG gate of phase 1b.",
        "model": {"id": MODEL_ID, "revision": REVISION,
                  "converted_sha256": bge_manifest["model"]["converted"]["sha256"]},
        "pipeline": bge_manifest["pipeline"],
        "environment": environment(),
        "sets": counts,
    }
    blob.write(os.path.join(OUT, "bge-m3-embed-de"), manifest)


# ---------------------------------------------------------------- tokenizer

def designed_cases():
    """hand-picked edge cases - (id, text)"""
    c = chr
    cases = [
        ("empty", ""), ("space", " "), ("spaces", "   "), ("tab", c(9)),
        ("newline", c(10)), ("crlf", "a" + c(13) + c(10) + "b"),
        ("lead_space", " Rohr"), ("trail_space", "Rohr "), ("both_spaces", "  Rohr  "),
        ("inner_spaces", "Rohr    DN   15"), ("tabs_inside", "a" + c(9) + c(9) + "b"),
        ("nbsp", "15" + c(0xA0) + "mm"), ("thin_space", "15" + c(0x2009) + "mm"),
        ("narrow_nbsp", "15" + c(0x202F) + "mm"), ("ideographic_space", "a" + c(0x3000) + "b"),
        ("em_space", "a" + c(0x2003) + "b"), ("zwsp", "a" + c(0x200B) + "b"),
        ("zwnj", "a" + c(0x200C) + "b"), ("zwj", "a" + c(0x200D) + "b"),
        ("bom_char", c(0xFEFF) + "Rohr"), ("soft_hyphen", "Kupfer" + c(0xAD) + "rohr"),
        ("nul", "a" + c(0) + "b"), ("bell", "a" + c(7) + "b"), ("del", "a" + c(0x7F) + "b"),
        ("c1_control", "a" + c(0x85) + "b"), ("line_sep", "a" + c(0x2028) + "b"),
        ("combining_acute", "e" + c(0x301)), ("combining_diaeresis", "u" + c(0x308) + "ber"),
        ("precomposed_ue", c(0xFC) + "ber"), ("hangul_jamo", c(0x1100) + c(0x1161)),
        ("hangul_syllable", c(0xAC00)), ("fullwidth_latin", "".join(c(0xFF21 + i) for i in range(5))),
        ("fullwidth_digits", "".join(c(0xFF10 + i) for i in range(10))),
        ("halfwidth_kana", c(0xFF76) + c(0xFF9E)), ("ligature_fi", c(0xFB01) + "x"),
        ("ligature_ffl", "a" + c(0xFB04) + "e"), ("superscript", "m" + c(0xB2) + " m" + c(0xB3)),
        ("fraction", "1" + c(0x2044) + "2 " + c(0xBD)), ("roman_numeral", c(0x2163)),
        ("circled_digit", c(0x2460) + c(0x2461)), ("micro_sign", c(0xB5) + "m"),
        ("ohm_sign", "10 " + c(0x2126)), ("angstrom", c(0x212B)), ("kelvin", "5 " + c(0x212A)),
        ("degree", "20" + c(0xB0) + "C"), ("multiplication", "30" + c(0xD7) + "60"),
        ("minus_sign", c(0x2212) + "5"), ("en_dash", "a" + c(0x2013) + "b"),
        ("em_dash", "a" + c(0x2014) + "b"), ("quotes", c(0x201E) + "Rohr" + c(0x201C)),
        ("guillemets", c(0xAB) + "x" + c(0xBB)), ("ellipsis", "a" + c(0x2026)),
        ("euro", "5 " + c(0x20AC)), ("sharp_s", "Stra" + c(0xDF) + "e"),
        ("capital_sharp_s", "STRA" + c(0x1E9E) + "E"), ("emoji", c(0x1F600)),
        ("emoji_skin", c(0x1F44D) + c(0x1F3FD)), ("emoji_zwj", c(0x1F468) + c(0x200D) + c(0x1F469) + c(0x200D) + c(0x1F467)),
        ("flag", c(0x1F1E9) + c(0x1F1EA)), ("variation_selector", c(0x2764) + c(0xFE0F)),
        ("supplementary_cjk", c(0x20000) + c(0x20001)), ("math_bold", c(0x1D400) + c(0x1D401)),
        ("private_use", c(0xE000) + "x"), ("noncharacter", "a" + c(0xFFFE) + "b"),
        ("replacement_char", c(0xFFFD)), ("cjk", "".join(c(x) for x in [0x6563, 0x70ED, 0x5668])),
        ("japanese", "".join(c(x) for x in [0x914D, 0x7BA1, 0x5DE5, 0x4E8B])),
        ("korean", "".join(c(x) for x in [0xBC30, 0xAD00])), ("arabic", "".join(c(x) for x in [0x0623, 0x0646, 0x0628, 0x0648, 0x0628])),
        ("hebrew", "".join(c(x) for x in [0x05E6, 0x05D9, 0x05E0, 0x05D5, 0x05E8])),
        ("thai", "".join(c(x) for x in [0x0E17, 0x0E48, 0x0E2D])), ("devanagari", "".join(c(x) for x in [0x0928, 0x0932, 0x0940])),
        ("cyrillic", "".join(c(x) for x in [0x0422, 0x0440, 0x0443, 0x0431, 0x0430])),
        ("greek", "".join(c(x) for x in [0x03A9, 0x03BB, 0x03C6])), ("mixed_scripts", "Rohr " + c(0x0422) + c(0x0440) + " " + c(0x7BA1)),
        ("metaspace_char", c(0x2581) + "Rohr"), ("metaspace_only", c(0x2581)),
        ("metaspace_double", "a" + c(0x2581) + c(0x2581) + "b"), ("long_word", "Rohrleitungsbauunternehmensverband" * 3),
        ("long_repeat", "a" * 300), ("long_digits", "1234567890" * 20), ("dots", "." * 50),
        ("url", "https://example.org/a?b=c&d=e"), ("email", "info@example.org"),
        ("code", "x := y + 1; // ok"), ("html", "<b>Rohr</b> &amp; Fitting"),
        ("lt_gt", "a < b > c"), ("brackets", "[DN 15] (Kupfer) {neu}"),
        ("backslash", "C:" + c(92) + "Pfad" + c(92) + "x"), ("apostrophe", "Hauseigent" + c(0xFC) + "mer's"),
        ("numbers_units", "3 x 1,5 mm" + c(0xB2) + " NYM-J"), ("dn", "DN15/DN20/DN25"),
        ("all_caps", "KUPFERROHR VERLEGEN"), ("mixed_case", "kUpFeRrOhR"),
    ]
    # the special-token literals: alone, glued to text, repeated, with spaces
    for lit, name in [("<s>", "bos"), ("</s>", "eos"), ("<pad>", "pad"),
                      ("<unk>", "unk"), ("<mask>", "mask")]:
        cases += [
            (f"lit_{name}", lit), (f"lit_{name}_glued", "a" + lit + "b"),
            (f"lit_{name}_spaced", "a " + lit + " b"), (f"lit_{name}_twice", lit + lit),
            (f"lit_{name}_lead", lit + "Rohr"), (f"lit_{name}_trail", "Rohr" + lit),
        ]
    # around 512 tokens (incl. <s> </s>; "Rohr" = 2 pieces, "in" = 1): tokenization
    # only, truncation is a separate concern - the counts are asserted in tokens()
    cases.append(("tokens_511", "Rohr " * 254 + "in"))
    cases.append(("tokens_512", "Rohr " * 255))
    cases.append(("tokens_513", "Rohr " * 255 + "in"))
    cases.append(("tokens_602", "Rohr " * 300))
    return cases


def generated_cases(n, seed):
    """deterministic random mixes over character pools - (id, text)"""
    import random
    rng = random.Random(seed)
    pools = [
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "".join(chr(x) for x in [0xE4, 0xF6, 0xFC, 0xC4, 0xD6, 0xDC, 0xDF, 0xE9, 0xE8, 0xE7, 0xF1]),
        " " * 6 + chr(9) + chr(10) + chr(0xA0) + chr(0x3000) + chr(0x2009),
        ".,;:-_/()[]{}+*=%&#@!?'" + chr(34),
        "".join(chr(x) for x in range(0x0410, 0x0450)),
        "".join(chr(x) for x in range(0x4E00, 0x4E40)),
        "".join(chr(x) for x in range(0x0300, 0x0310)),
        "".join(chr(x) for x in [0xFF21, 0xFF41, 0xFF10, 0xFB01, 0x2163, 0x2460, 0xB2, 0xBD, 0x2126]),
        "".join(chr(x) for x in [0x1F600, 0x1F44D, 0x200D, 0xFE0F, 0x1F1E9, 0x20000]),
        "".join(chr(x) for x in range(0x0600, 0x0620)) + "".join(chr(x) for x in range(0x05D0, 0x05EA)),
        chr(0x2581) + chr(0x200B) + chr(0xAD) + chr(0xFEFF),
    ]
    weights = [30, 10, 12, 8, 6, 6, 4, 6, 4, 4, 2]
    cases, seen = [], set()
    while len(cases) < n:
        length = rng.randint(1, 40)
        chars = []
        for _ in range(length):
            pool = rng.choices(pools, weights)[0]
            chars.append(rng.choice(pool))
        text = "".join(chars)
        if text in seen:
            continue  # duplicates add nothing
        seen.add(text)
        cases.append((f"gen_{len(cases):04d}", text))
    return cases


def tokens(snapshot):
    print("tokenizer cases")
    tok = RefTokenizer(snapshot)
    data = os.path.join(ROOT, "tests", "data", "embed-de")
    texts = []
    for name in ["corpus", "queries"]:
        with open(os.path.join(data, name + ".jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    texts.append((f"{name}_{r['id']}", r["text"]))
    texts += designed_cases()
    texts += generated_cases(900, TINY_SEED)
    if len({cid for cid, _ in texts}) != len(texts):
        raise ValueError("duplicated case id")
    flat, offsets, cases = [], [], []
    text_bytes, text_offsets = bytearray(), []
    norm_bytes, norm_offsets = bytearray(), []
    hf_differs = 0
    for cid, text in texts:
        ids = tok.encode(text, expect_hf_equal=False)
        hf = tok.hf(text, add_special_tokens=True, truncation=False)["input_ids"]
        if cid.startswith("tokens_") and len(ids) != int(cid[7:]):
            raise ValueError(f"{cid}: {len(ids)} tokens")
        offsets.append(len(flat))
        flat += ids
        text_offsets.append(len(text_bytes))
        text_bytes += text.encode("utf-8")
        norm_offsets.append(len(norm_bytes))
        norm_bytes += tok.sp.normalize(text).encode("utf-8")
        e = {"id": cid, "text": text, "tokens": len(ids),
             "normalized": tok.sp.normalize(text)}
        if hf != ids:
            e["hf_ids"] = hf
            hf_differs += 1
        cases.append(e)
    offsets.append(len(flat))
    text_offsets.append(len(text_bytes))
    norm_offsets.append(len(norm_bytes))
    blob = Blob()
    # the exact UTF-8 bytes: a JSON reader may not keep an escaped NUL
    blob.add("texts", np.frombuffer(bytes(text_bytes), dtype=np.uint8),
             "UTF-8 of all case texts concatenated")
    blob.add("text_offsets", np.array(text_offsets, dtype=np.int64),
             "case i = texts[text_offsets[i] .. text_offsets[i+1]-1]")
    blob.add("normalized", np.frombuffer(bytes(norm_bytes), dtype=np.uint8),
             "UTF-8 of all SentencePiece-normalized texts concatenated")
    blob.add("normalized_offsets", np.array(norm_offsets, dtype=np.int64),
             "case i = normalized[normalized_offsets[i] .. normalized_offsets[i+1]-1]")
    blob.add("ids", np.array(flat, dtype=np.int64), "all cases concatenated")
    blob.add("offsets", np.array(offsets, dtype=np.int64),
             "case i = ids[offsets[i] .. offsets[i+1]-1]")
    manifest = {
        "dataset": "bge-m3-tokens",
        "note": "reference token ids (SentencePiece, parse_special=false, fairseq ids, "
                "<s> text </s>, no truncation) - the tokenizer gate of phase 1a. "
                "normalized = SentencePiece normalize(); hf_ids only where the HF "
                "tokenizer differs (it reads special-token literals as control tokens)",
        "model": {"id": MODEL_ID, "revision": REVISION,
                  "sentencepiece_model_sha256": sha256_file(os.path.join(snapshot, "sentencepiece.bpe.model")),
                  "tokenizer_json_sha256": sha256_file(os.path.join(snapshot, "tokenizer.json"))},
        "environment": environment(),
        "hf_differs": hf_differs,
        "cases": cases,
    }
    blob.write(os.path.join(OUT, "bge-m3-tokens"), manifest)


# ---------------------------------------------------------------- tiny XLM-R

def tiny():
    print("tiny-xlmr")
    torch.manual_seed(TINY_SEED)
    cfg = XLMRobertaConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                           num_attention_heads=4, intermediate_size=64,
                           max_position_embeddings=40, type_vocab_size=1,
                           layer_norm_eps=1e-5, hidden_act="gelu",
                           pad_token_id=1, bos_token_id=0, eos_token_id=2,
                           position_embedding_type="absolute",
                           hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    model = XLMRobertaModel(cfg, add_pooling_layer=False)
    with torch.no_grad():
        # random LayerNorm gamma/beta: the defaults (1/0) would hide a missing scale
        for name, p in model.named_parameters():
            if "LayerNorm" in name:
                p.copy_(torch.randn_like(p) * 0.2 + (1.0 if name.endswith("weight") else 0.0))
    model.eval()
    folder = os.path.join(OUT, "tiny-xlmr")
    os.makedirs(folder, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(folder, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(cfg.to_json_string(use_diff=False))
    model, _ = load_model(folder)  # the checked-in weights, not the in-memory ones

    blob = Blob()
    singles = [[0, 7, 2], [0, 5, 9, 33, 12, 2], [0, 63, 4, 4, 17, 50, 21, 8, 2]]
    for i, seq in enumerate(singles):
        ids = np.array([seq], dtype=np.int64)
        mask = np.ones_like(ids)
        hs = forward(model, ids, mask)
        cls, emb = dense(hs[-1])
        p = f"single/{i}/"
        blob.add(p + "input_ids", ids[0])
        blob.add(p + "position_ids", xlmr_positions(ids, 1)[0])
        for k, h in enumerate(hs):
            blob.add(p + f"hidden.{k}", h[0])
        blob.add(p + "cls", cls[0])
        blob.add(p + "embedding", emb[0])
    width = max(len(s) for s in singles)
    ids = np.array([s + [1] * (width - len(s)) for s in singles], dtype=np.int64)
    mask = (ids != 1).astype(np.int64)
    hs = forward(model, ids, mask)
    blob.add("batch/input_ids", ids)
    blob.add("batch/attention_mask", mask)
    blob.add("batch/position_ids", xlmr_positions(ids, 1))
    for k, h in enumerate(hs):
        blob.add(f"batch/hidden.{k}", h, "padded positions are not meaningful")
    blob.add("batch/embedding", dense(hs[-1])[1])
    manifest = {
        "dataset": "tiny-xlmr",
        "note": "seeded random XLM-R with the BGE-M3 conventions (pad 1, bos 0, eos 2, "
                "type_vocab 1, eps 1e-5, exact GELU, absolute positions); the weights in "
                "model.safetensors are the contract, the seed only documents their origin",
        "seed": TINY_SEED,
        "weights": {"file": "model.safetensors",
                    "sha256": sha256_file(os.path.join(folder, "model.safetensors")),
                    "tensors": sorted((k, list(v.shape)) for k, v in state.items())},
        "config_sha256": sha256_file(os.path.join(folder, "config.json")),
        "environment": environment(),
    }
    blob.write(folder, manifest)


# ---------------------------------------------------------------- operators

def ops():
    """inputs in f32; expected outputs computed in float64 from those f32 inputs"""
    print("ops")
    rng = np.random.default_rng(TINY_SEED)
    blob = Blob()
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    # linear like nn.Linear: y = x W^T + b, W stored [out, in]
    x = f32(rng.standard_normal((3, 5)))
    w = f32(rng.standard_normal((4, 5)))
    b = f32(rng.standard_normal(4))
    blob.add("linear/x", x)
    blob.add("linear/weight", w, "[out, in], used transposed")
    blob.add("linear/bias", b)
    blob.add("linear/y", x.astype(np.float64) @ w.astype(np.float64).T + b)
    # layernorm with non-trivial gamma/beta; row 2 nearly constant, row 3 exactly constant
    x = f32(rng.standard_normal((4, 8)))
    x[2] = f32(3.0 + 1e-4 * rng.standard_normal(8))
    x[3] = 2.5
    g = f32(rng.standard_normal(8) * 0.5 + 1)
    be = f32(rng.standard_normal(8) * 0.5)
    xd = x.astype(np.float64)
    mu = xd.mean(-1, keepdims=True)
    var = ((xd - mu) ** 2).mean(-1, keepdims=True)  # population variance
    blob.add("layernorm/x", x)
    blob.add("layernorm/gamma", g)
    blob.add("layernorm/beta", be)
    blob.add("layernorm/y", (xd - mu) / np.sqrt(var + 1e-5) * g + be, "eps = 1e-5")
    # softmax over the last axis with an additive mask (0 keep, -inf drop)
    s = f32(rng.standard_normal((3, 6)) * 4)
    s[1, 0] = 80.0  # large logit: needs the max subtraction
    keep = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1]], dtype=np.int64)
    sd = np.where(keep == 1, s.astype(np.float64), -np.inf)
    e = np.exp(sd - sd.max(-1, keepdims=True))
    blob.add("softmax/scores", s)
    blob.add("softmax/keep", keep, "1 = attend, 0 = masked")
    blob.add("softmax/y", e / e.sum(-1, keepdims=True), "masked entries are exactly 0")
    # exact GELU (erf), including large magnitudes
    x = f32(np.concatenate([np.linspace(-6, 6, 25), [-30, -10, 10, 30, 0]]))
    blob.add("gelu/x", x)
    # erfc(-z) = 1 + erf(z) without the cancellation that zeroes gelu(-10)
    blob.add("gelu/y", np.array([0.5 * v * math.erfc(-v / math.sqrt(2)) for v in x.astype(np.float64)]),
             "exact: 0.5 x (1 + erf(x / sqrt 2)) = 0.5 x erfc(-x / sqrt 2), not the tanh approximation")
    # L2 normalisation, incl. a tiny-norm row
    x = f32(rng.standard_normal((3, 7)))
    x[2] = f32(rng.standard_normal(7) * 1e-20)
    xd = x.astype(np.float64)
    blob.add("l2/x", x, "row 2 has a tiny norm (1e-20 scale): must not underflow to NaN")
    blob.add("l2/y", xd / np.linalg.norm(xd, axis=-1, keepdims=True))
    blob.write(os.path.join(OUT, "ops"), {"dataset": "ops", "environment": environment(),
                                          "note": "inputs f32, outputs computed in float64 and "
                                                  "stored as f32 (round to nearest)"})


# ---------------------------------------------------------------- GGUF / Q8_0 (phase 1b)

GGML_DIR = os.environ.get("LAI_GGML_DIR") or os.path.join(ROOT, "vendor", "sqlite-ext",
                                                          "aarch64-linux")
GGUF_Q8 = os.environ.get("LAI_GGUF") or os.path.join(ROOT, "vendor", "models",
                                                     "bge-m3-q8_0.gguf")
# the file of reference contract v1 (tests/data/embed-de/reference/manifest.json)
GGUF_Q8_SHA256 = "aa473d51f451a22f0fcf39ba3330c14bed38a385712b1113440f69df4047a173"
Q8_BLOCK = 34  # bytes of one Q8_0 block: FP16 scale + 32 int8
Q4_BLOCK = 18  # bytes of one Q4_0 block: FP16 scale + 32 nibbles
Q5_BLOCK = 22  # bytes of one Q5_0 block: FP16 scale, 32 fifth bits, 32 nibbles
MX_BLOCK = 17  # bytes of one MXFP4 block: E8M0 exponent + 32 nibbles
Q4K_BLOCK = 144  # one Q4_K super-block of 256: d, dmin, 12 bytes scales, 128 nibbles
Q6K_BLOCK = 210  # one Q6_K super-block of 256: 128 + 64 bytes quants, 16 scales, d
Q8K_BLOCK = 292  # one Q8_K block of 256: float d, 256 int8, 16 int16 block sums


class Ggml:
    """ggml's own Q8_0 arithmetic, from the libraries of the lembed oracle build

    quantize_row_q8_0 and ggml_vec_dot_q8_0_q8_0 are the ggml-cpu kernels lembed runs
    (on aarch64 the NEON paths: activations round to nearest even); the weights are
    quantized with quantize_row_q8_0_ref like any GGUF writer would.
    """

    def __init__(self):
        import ctypes
        self.ct = ctypes
        self.base = ctypes.CDLL(os.path.join(GGML_DIR, "libggml-base.so"), mode=ctypes.RTLD_GLOBAL)
        self.cpu = ctypes.CDLL(os.path.join(GGML_DIR, "libggml-cpu.so"), mode=ctypes.RTLD_GLOBAL)
        self.cpu.ggml_cpu_init.restype = None
        self.cpu.ggml_cpu_init()
        vp, fp, i64 = ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64
        for lib, name in [(self.cpu, "quantize_row_q8_0"), (self.base, "quantize_row_q8_0_ref")]:
            getattr(lib, name).argtypes = [fp, vp, i64]
            getattr(lib, name).restype = None
        self.base.dequantize_row_q8_0.argtypes = [vp, fp, i64]
        self.base.dequantize_row_q8_0.restype = None
        self.cpu.ggml_vec_dot_q8_0_q8_0.argtypes = [ctypes.c_int, fp, ctypes.c_size_t, vp,
                                                    ctypes.c_size_t, vp, ctypes.c_size_t,
                                                    ctypes.c_int]
        self.cpu.ggml_vec_dot_q8_0_q8_0.restype = None
        self.base.quantize_row_q4_0_ref.argtypes = [fp, vp, i64]
        self.base.quantize_row_q4_0_ref.restype = None
        self.base.dequantize_row_q4_0.argtypes = [vp, fp, i64]
        self.base.dequantize_row_q4_0.restype = None
        self.cpu.ggml_vec_dot_q4_0_q8_0.argtypes = self.cpu.ggml_vec_dot_q8_0_q8_0.argtypes
        self.cpu.ggml_vec_dot_q4_0_q8_0.restype = None
        self.base.quantize_row_q5_0_ref.argtypes = [fp, vp, i64]
        self.base.quantize_row_q5_0_ref.restype = None
        self.base.dequantize_row_q5_0.argtypes = [vp, fp, i64]
        self.base.dequantize_row_q5_0.restype = None
        self.cpu.ggml_vec_dot_q5_0_q8_0.argtypes = self.cpu.ggml_vec_dot_q8_0_q8_0.argtypes
        self.cpu.ggml_vec_dot_q5_0_q8_0.restype = None
        self.base.quantize_row_mxfp4_ref.argtypes = [fp, vp, i64]
        self.base.quantize_row_mxfp4_ref.restype = None
        self.base.dequantize_row_mxfp4.argtypes = [vp, fp, i64]
        self.base.dequantize_row_mxfp4.restype = None
        self.cpu.ggml_vec_dot_mxfp4_q8_0.argtypes = self.cpu.ggml_vec_dot_q8_0_q8_0.argtypes
        self.cpu.ggml_vec_dot_mxfp4_q8_0.restype = None
        for name in ["quantize_row_q4_K_ref", "quantize_row_q6_K_ref"]:
            getattr(self.base, name).argtypes = [fp, vp, i64]
            getattr(self.base, name).restype = None
        for name in ["dequantize_row_q4_K", "dequantize_row_q6_K"]:
            getattr(self.base, name).argtypes = [vp, fp, i64]
            getattr(self.base, name).restype = None
        self.cpu.quantize_row_q8_K.argtypes = [fp, vp, i64]
        self.cpu.quantize_row_q8_K.restype = None
        for name in ["ggml_vec_dot_q4_K_q8_K", "ggml_vec_dot_q6_K_q8_K"]:
            getattr(self.cpu, name).argtypes = self.cpu.ggml_vec_dot_q8_0_q8_0.argtypes
            getattr(self.cpu, name).restype = None
        self.base.ggml_fp32_to_fp16.argtypes = [ctypes.c_float]
        self.base.ggml_fp32_to_fp16.restype = ctypes.c_uint16
        self.libs = {n: sha256_file(os.path.join(GGML_DIR, n))
                     for n in ["libggml-base.so", "libggml-cpu.so"]}

    def _quant(self, fn, x, block=Q8_BLOCK):
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        if x.size % 32:
            raise ValueError("Q8_0 / Q4_0 need whole blocks")
        out = self.ct.create_string_buffer(x.size // 32 * block)
        fn(x.ctypes.data_as(self.ct.POINTER(self.ct.c_float)), out, x.size)
        return out.raw

    def quant(self, x):
        """activations, as ggml-cpu quantizes them before a Q8_0 matmul"""
        return self._quant(self.cpu.quantize_row_q8_0, x)

    def quant_ref(self, x):
        """weights, as a GGUF writer quantizes them"""
        return self._quant(self.base.quantize_row_q8_0_ref, x)

    def quant_q4(self, x):
        """weights, as a GGUF writer quantizes them to Q4_0"""
        return self._quant(self.base.quantize_row_q4_0_ref, x, Q4_BLOCK)

    def dequant_q4(self, raw, n):
        y = np.zeros(n, dtype=np.float32)
        self.base.dequantize_row_q4_0(raw, y.ctypes.data_as(self.ct.POINTER(self.ct.c_float)), n)
        return y

    def dot_q4(self, n, w, x):
        s = self.ct.c_float()
        self.cpu.ggml_vec_dot_q4_0_q8_0(n, self.ct.byref(s), 0, w, 0, x, 0, 1)
        return s.value

    def quant_q5(self, x):
        """weights, as a GGUF writer quantizes them to Q5_0"""
        return self._quant(self.base.quantize_row_q5_0_ref, x, Q5_BLOCK)

    def dequant_q5(self, raw, n):
        y = np.zeros(n, dtype=np.float32)
        self.base.dequantize_row_q5_0(raw, y.ctypes.data_as(self.ct.POINTER(self.ct.c_float)), n)
        return y

    def dot_q5(self, n, w, x):
        s = self.ct.c_float()
        self.cpu.ggml_vec_dot_q5_0_q8_0(n, self.ct.byref(s), 0, w, 0, x, 0, 1)
        return s.value

    def quant_mx(self, x):
        """weights, as a GGUF writer quantizes them to MXFP4"""
        return self._quant(self.base.quantize_row_mxfp4_ref, x, MX_BLOCK)

    def dequant_mx(self, raw, n):
        y = np.zeros(n, dtype=np.float32)
        self.base.dequantize_row_mxfp4(raw, y.ctypes.data_as(self.ct.POINTER(self.ct.c_float)), n)
        return y

    def dot_mx(self, n, w, x):
        s = self.ct.c_float()
        self.cpu.ggml_vec_dot_mxfp4_q8_0(n, self.ct.byref(s), 0, w, 0, x, 0, 1)
        return s.value

    def kquant(self, kind, x):
        """weights as a GGUF writer stores them: 'q4_K' or 'q6_K' (whole 256 blocks)"""
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        size = Q4K_BLOCK if kind == "q4_K" else Q6K_BLOCK
        out = self.ct.create_string_buffer(x.size // 256 * size)
        getattr(self.base, "quantize_row_%s_ref" % kind)(
            x.ctypes.data_as(self.ct.POINTER(self.ct.c_float)), out, x.size)
        return out.raw

    def kdequant(self, kind, raw, n):
        y = np.zeros(n, dtype=np.float32)
        getattr(self.base, "dequantize_row_%s" % kind)(
            raw, y.ctypes.data_as(self.ct.POINTER(self.ct.c_float)), n)
        return y

    def q8k(self, x):
        """activations, as ggml-cpu quantizes them before a K-quant matmul"""
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        out = self.ct.create_string_buffer(x.size // 256 * Q8K_BLOCK)
        self.cpu.quantize_row_q8_K(x.ctypes.data_as(self.ct.POINTER(self.ct.c_float)),
                                   out, x.size)
        return out.raw

    def kdot(self, kind, n, w, x):
        s = self.ct.c_float()
        getattr(self.cpu, "ggml_vec_dot_%s_q8_K" % kind)(n, self.ct.byref(s), 0, w, 0, x, 0, 1)
        return s.value

    def dequant(self, raw, n):
        y = np.zeros(n, dtype=np.float32)
        self.base.dequantize_row_q8_0(raw, y.ctypes.data_as(self.ct.POINTER(self.ct.c_float)), n)
        return y

    def dot(self, n, a, b):
        s = self.ct.c_float()
        self.cpu.ggml_vec_dot_q8_0_q8_0(n, self.ct.byref(s), 0, a, 0, b, 0, 1)
        return s.value

    def f16(self, v):
        return self.base.ggml_fp32_to_fp16(float(v))

    def linear(self, x, wq, n_in, n_out, bias):
        """y = x W^T + b with Q8_0 weights wq (n_out rows of blocks) - ggml's way"""
        rb = n_in // 32 * Q8_BLOCK
        y = np.zeros((x.shape[0], n_out), dtype=np.float64)
        for r in range(x.shape[0]):
            xq = self.quant(x[r])
            for o in range(n_out):
                y[r, o] = self.dot(n_in, wq[o * rb:(o + 1) * rb], xq)
        return y + bias.astype(np.float64)


def q8():
    """Q8_0 fixtures from ggml itself: FP16 rounding, activation quantization, the
    integer block dot product, dequantization"""
    print("q8")
    g = Ggml()
    rng = np.random.default_rng(TINY_SEED + 8)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    blob = Blob()
    # half precision: ties to even, overflow, subnormals, underflow
    half = [0.0, -0.0, 1.0, -2.0, 1 + 2 ** -11, 1 + 3 * 2 ** -11, 1 + 2 ** -11 + 2 ** -20,
            65504.0, 65519.0, 65520.0, -65520.0, 1e10, 2 ** -14, 2 ** -24, 2 ** -25,
            2 ** -25 + 2 ** -40, 3 * 2 ** -25, 2 ** -26, 1e-8, -1e-8, 0.1, 1 / 3, 1 / 127,
            127 / 127.0 / 127, 2 ** -15 + 2 ** -25, 2 ** -15 + 3 * 2 ** -25]
    half += list(rng.standard_normal(40) * 10.0 ** rng.integers(-9, 5, 40))
    half = f32(half)
    blob.add("half/x", half, "singles, incl. ties, overflow, subnormals")
    blob.add("half/bits", np.array([g.f16(v) for v in half], dtype=np.int64),
             "ggml_fp32_to_fp16: IEEE half bits, round to nearest even")
    # activation quantization: ties (amax 127 -> d 1), a zero block, tiny and large rows
    x = f32(rng.standard_normal((5, 96)))
    x[0, :8] = [127, 2.5, -3.5, 0.5, -0.5, 1.5, -1.5, 126.5]
    x[1, 32:64] = 0
    x[2] *= 1e-6
    x[3] *= 1e4
    x[4, 64:96] = f32(rng.standard_normal(32) * 1e-30)
    blob.add("quant/x", x)
    blob.add("quant/blocks", np.frombuffer(b"".join(g.quant(r) for r in x), dtype=np.uint8)
             .reshape(5, 3 * Q8_BLOCK), "ggml-cpu quantize_row_q8_0 per row")
    # Q8_0 linear: weights from the reference quantizer, activations quantized by ggml-cpu
    n_in, n_out = 96, 5
    x = f32(rng.standard_normal((3, n_in)))
    w = f32(rng.standard_normal((n_out, n_in)) * 0.2)
    b = f32(rng.standard_normal(n_out))
    wq = b"".join(g.quant_ref(r) for r in w)
    blob.add("linear/x", x)
    blob.add("linear/weight", np.frombuffer(wq, dtype=np.uint8).reshape(n_out, -1),
             "[out, in/32 blocks] Q8_0")
    blob.add("linear/bias", b)
    blob.add("linear/y", g.linear(x, wq, n_in, n_out, b),
             "ggml_vec_dot_q8_0_q8_0 (float sums) + bias in float64")
    blob.add("dequant/y", np.stack([g.dequant(wq[o * 3 * Q8_BLOCK:(o + 1) * 3 * Q8_BLOCK], n_in)
                                    for o in range(n_out)]), "dequantize_row_q8_0 of the weights")
    blob.write(os.path.join(OUT, "q8"), {
        "dataset": "q8", "environment": environment(),
        "ggml": {"dir": os.path.relpath(GGML_DIR, ROOT), "sha256": g.libs,
                 "note": "the ggml build of the lembed oracle"}})


def q4():
    """Q4_0 fixtures from ggml itself (phase 3e): the reference quantizer of a GGUF
    writer, dequantization, and the Q4_0 x Q8_0 dot product of ggml-cpu; and Q8_0
    weights from that writer's quantizer (ties away from zero)"""
    print("q4")
    g = Ggml()
    rng = np.random.default_rng(TINY_SEED + 4)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    blob = Blob()
    x = f32(rng.standard_normal((8, 96)))
    # d = max / -8 with the signed value of the first largest magnitude
    x[0, :32] = 0.25
    x[0, 0] = -8.0             # d = 1: x + 8.5 lands on .5 and on whole numbers
    x[0, 1:12] = [0.5, -0.5, 6.5, 7.0, -7.5, 1.0, -1.0, 7.49, -8.0, 3.5, -3.5]
    x[1, :32] = rng.standard_normal(32)
    x[1, 3], x[1, 9] = -5.0, 5.0   # equal magnitudes: the first one (negative) wins
    x[1, 32:64] = rng.standard_normal(32)
    x[1, 40], x[1, 41] = 5.0, -5.0  # the first one (positive) wins
    x[2, :32] = 0                  # a zero block
    x[3] *= 1e-6
    x[4] *= 1e4
    x[5, 64:96] = f32(rng.standard_normal(32) * 1e-30)
    x[6, 32:64] = 7.0              # one value repeated: all at the clip
    blob.add("quant/x", x)
    blob.add("quant/blocks", np.frombuffer(b"".join(g.quant_q4(r) for r in x), dtype=np.uint8)
             .reshape(8, 3 * Q4_BLOCK), "quantize_row_q4_0_ref per row")
    blob.add("dequant/y", np.stack([g.dequant_q4(g.quant_q4(r), 96) for r in x]),
             "dequantize_row_q4_0 of those blocks")
    # Q8_0 weights as a GGUF writer stores them: quantize_row_q8_0_ref rounds ties
    # away from zero (activations: to even); weights widened from BF16 hit ties
    w8 = f32(rng.standard_normal((4, 64)))
    w8[0, :8] = [127, 2.5, -3.5, 0.5, -0.5, 1.5, -1.5, 126.5]  # d = 1: ties
    w8[1] = (w8[1].astype(np.float32).view(np.uint32) & 0xFFFF0000).view(np.float32)
    w8[2] = (w8[2].astype(np.float32).view(np.uint32) & 0xFFFF0000).view(np.float32) * 8
    w8[3, 32:] = 0
    blob.add("q8ref/x", w8, "row 0 ties at d = 1, rows 1-2 BF16 values")
    blob.add("q8ref/blocks", np.frombuffer(b"".join(g.quant_ref(r) for r in w8),
                                           dtype=np.uint8).reshape(4, 2 * Q8_BLOCK),
             "quantize_row_q8_0_ref per row")
    # Q4_0 linear: weights from the reference quantizer, activations quantized by ggml-cpu
    n_in, n_out = 128, 6
    xs = f32(rng.standard_normal((3, n_in)))
    w = f32(rng.standard_normal((n_out, n_in)) * 0.2)
    b = f32(rng.standard_normal(n_out))
    wq = b"".join(g.quant_q4(r) for r in w)
    rb = n_in // 32 * Q4_BLOCK
    y = np.zeros((3, n_out), dtype=np.float64)
    for r in range(3):
        xq = g.quant(xs[r])
        for o in range(n_out):
            y[r, o] = g.dot_q4(n_in, wq[o * rb:(o + 1) * rb], xq)
    blob.add("linear/x", xs)
    blob.add("linear/weight", np.frombuffer(wq, dtype=np.uint8).reshape(n_out, -1),
             "[out, in/32 blocks] Q4_0")
    blob.add("linear/bias", b)
    blob.add("linear/y", y + b.astype(np.float64),
             "ggml_vec_dot_q4_0_q8_0 (float sums) + bias in float64")
    blob.write(os.path.join(OUT, "q4"), {
        "dataset": "q4", "environment": environment(),
        "ggml": {"dir": os.path.relpath(GGML_DIR, ROOT), "sha256": g.libs,
                 "note": "the ggml build of the lembed oracle"}})


def q5():
    """Q5_0 fixtures from ggml itself (backlog B16): weights from the reference
    quantizer of a GGUF writer, their dequantization, and the Q5_0 x Q8_0 dot product
    of ggml-cpu"""
    print("q5")
    g = Ggml()
    rng = np.random.default_rng(TINY_SEED + 5)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    blob = Blob()
    x = f32(rng.standard_normal((6, 96)))
    x[0, :32] = np.linspace(-16, 15, 32)  # d = 1: every q from 0 to 31 once
    x[0, 0] = -16.0
    x[1, :32] = 0                  # a zero block
    x[2] *= 1e-6
    x[3] *= 1e4
    x[4, 32:64] = 7.0              # one value repeated: d = -7 / 16, all at q = 0
    x[5, 64:96] = -np.abs(x[5, 64:96])  # every value below zero: fifth bits clear
    wq = b"".join(g.quant_q5(r) for r in x)
    blob.add("quant/x", x)
    blob.add("quant/blocks", np.frombuffer(wq, dtype=np.uint8).reshape(6, 3 * Q5_BLOCK),
             "quantize_row_q5_0_ref per row")
    blob.add("dequant/y", np.stack([g.dequant_q5(g.quant_q5(r), 96) for r in x]),
             "dequantize_row_q5_0 of those blocks")
    # Q5_0 linear: weights from the reference quantizer, activations quantized by ggml-cpu
    n_in, n_out = 128, 6
    xs = f32(rng.standard_normal((3, n_in)))
    w = f32(rng.standard_normal((n_out, n_in)) * 0.2)
    b = f32(rng.standard_normal(n_out))
    wq = b"".join(g.quant_q5(r) for r in w)
    rb = n_in // 32 * Q5_BLOCK
    y = np.zeros((3, n_out), dtype=np.float64)
    for r in range(3):
        xq = g.quant(xs[r])
        for o in range(n_out):
            y[r, o] = g.dot_q5(n_in, wq[o * rb:(o + 1) * rb], xq)
    blob.add("linear/x", xs)
    blob.add("linear/weight", np.frombuffer(wq, dtype=np.uint8).reshape(n_out, -1),
             "[out, in/32 blocks] Q5_0")
    blob.add("linear/bias", b)
    blob.add("linear/y", y + b.astype(np.float64),
             "ggml_vec_dot_q5_0_q8_0 (float sums) + bias in float64")
    blob.write(os.path.join(OUT, "q5"), {
        "dataset": "q5", "environment": environment(),
        "ggml": {"dir": os.path.relpath(GGML_DIR, ROOT), "sha256": g.libs,
                 "note": "the ggml build of the lembed oracle"}})


def mxfp4():
    """MXFP4 fixtures from ggml itself (backlog D5b): weights from the reference
    quantizer of a GGUF writer, their dequantization, and the MXFP4 x Q8_0 dot
    product of ggml-cpu; the blocks of x rows 6 and 7 carry exponents outside
    the range the engine computes exactly (its loaders refuse them)"""
    print("mxfp4")
    g = Ggml()
    rng = np.random.default_rng(TINY_SEED + 8)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    blob = Blob()
    x = f32(rng.standard_normal((8, 96)))
    # every E2M1 value and its negative at a scale of 1: max 6 gives e = 127
    vals = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    x[0, :32] = vals + [-v for v in vals] + vals[::-1] + [-v for v in vals[::-1]]
    x[0, 32:64] = np.linspace(-6, 6, 32)  # between the values: rounding
    x[1, :32] = 0                  # a zero block
    x[2] *= 1e-3
    x[3] *= 1e3
    x[4, 32:64] = 5.0              # one value repeated
    x[5, 64:96] = -np.abs(x[5, 64:96])  # every value below zero
    x[6] *= 1e-9                   # exponents below the exact range
    x[7] *= 1e7                    # and above it
    wq = b"".join(g.quant_mx(r) for r in x)
    blob.add("quant/x", x)
    blob.add("quant/blocks", np.frombuffer(wq, dtype=np.uint8).reshape(8, 3 * MX_BLOCK),
             "quantize_row_mxfp4_ref per row")
    blob.add("dequant/y", np.stack([g.dequant_mx(g.quant_mx(r), 96) for r in x]),
             "dequantize_row_mxfp4 of those blocks")
    exps = np.frombuffer(wq, dtype=np.uint8).reshape(-1, MX_BLOCK)[:, 0].copy()
    if exps[3] != 0:
        raise ValueError(f"the zero block has the exponent {exps[3]}")
    exps[3] = 127  # a zero block: any exponent, ggml writes 0
    if exps[:18].min() < 104 or exps[:18].max() > 143:
        raise ValueError(f"rows 0..5 leave the exact range: {exps[:18]}")
    if exps[18:21].max() >= 104 or exps[21:24].min() <= 143:
        raise ValueError(f"rows 6 and 7 stay in the exact range: {exps[18:]}")
    # MXFP4 linear: weights from the reference quantizer, activations
    # quantized by ggml-cpu - inputs of 2880 as gpt-oss-20b, no multiple of 256
    n_in, n_out = 2880, 6
    xs = f32(rng.standard_normal((3, n_in)))
    w = f32(rng.standard_normal((n_out, n_in)) * 0.05)
    b = f32(rng.standard_normal(n_out))
    wq = b"".join(g.quant_mx(r) for r in w)
    rb = n_in // 32 * MX_BLOCK
    y = np.zeros((3, n_out), dtype=np.float64)
    for r in range(3):
        xq = g.quant(xs[r])
        for o in range(n_out):
            y[r, o] = g.dot_mx(n_in, wq[o * rb:(o + 1) * rb], xq)
    blob.add("linear/x", xs)
    blob.add("linear/weight", np.frombuffer(wq, dtype=np.uint8).reshape(n_out, -1),
             "[out, in/32 blocks] MXFP4")
    blob.add("linear/bias", b)
    blob.add("linear/y", y + b.astype(np.float64),
             "ggml_vec_dot_mxfp4_q8_0 (float sums) + bias in float64")
    blob.write(os.path.join(OUT, "mxfp4"), {
        "dataset": "mxfp4", "environment": environment(),
        "ggml": {"dir": os.path.relpath(GGML_DIR, ROOT), "sha256": g.libs,
                 "note": "the ggml build of the lembed oracle"}})


def qk():
    """K-quant fixtures from ggml itself (phase 3e-3): Q4_K and Q6_K weights from the
    reference quantizers a GGUF writer uses, their dequantization, Q8_K activations
    of ggml-cpu and the Q4_K / Q6_K x Q8_K dot products"""
    print("qk")
    g = Ggml()
    rng = np.random.default_rng(TINY_SEED + 6)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    blob = Blob()
    # Q8_K activations: the signed largest value sets the scale, ties to even,
    # a zero block, tiny and large values
    x = f32(rng.standard_normal((5, 512)))
    x[0, :8] = [-127, 2.5, -3.5, 0.5, -0.5, 1.5, -1.5, 126.5]  # d = 1: ties
    x[1, 256:] = 0
    x[2] *= 1e-6
    x[3] *= 1e4
    x[4, 10], x[4, 20] = 50.0, -50.0  # equal magnitudes: the first one wins
    blob.add("q8k/x", x)
    blob.add("q8k/blocks", np.frombuffer(b"".join(g.q8k(r) for r in x), dtype=np.uint8)
             .reshape(5, 2 * Q8K_BLOCK), "ggml-cpu quantize_row_q8_K per row")
    n_in, n_out = 512, 7
    xs = f32(rng.standard_normal((3, n_in)))
    xq = [g.q8k(r) for r in xs]
    blob.add("linear/x", xs)
    b = f32(rng.standard_normal(n_out))
    blob.add("linear/bias", b)
    for kind, size in [("q4_K", Q4K_BLOCK), ("q6_K", Q6K_BLOCK)]:
        w = f32(rng.standard_normal((n_out, n_in)) * 0.1)
        w[1] *= 50  # another magnitude
        w[2, :32] = 0  # a zero sub-block
        wq = b"".join(g.kquant(kind, r) for r in w)
        rb = n_in // 256 * size
        y = np.zeros((3, n_out), dtype=np.float64)
        for r in range(3):
            for o in range(n_out):
                y[r, o] = g.kdot(kind, n_in, wq[o * rb:(o + 1) * rb], xq[r])
        blob.add(kind + "/weight", np.frombuffer(wq, dtype=np.uint8).reshape(n_out, -1),
                 "[out, in/256 blocks] " + kind)
        blob.add(kind + "/dequant", np.stack([g.kdequant(kind, wq[o * rb:(o + 1) * rb], n_in)
                                              for o in range(n_out)]),
                 "dequantize_row_" + kind)
        blob.add(kind + "/y", y + b.astype(np.float64),
                 "ggml_vec_dot_%s_q8_K (float sums) + bias in float64" % kind)
    blob.write(os.path.join(OUT, "qk"), {
        "dataset": "qk", "environment": environment(),
        "ggml": {"dir": os.path.relpath(GGML_DIR, ROOT), "sha256": g.libs,
                 "note": "the ggml build of the lembed oracle"}})


# --- a minimal GGUF v3 writer (format: ggml docs/gguf.md)

GT = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6, "bool": 7,
      "str": 8, "arr": 9, "u64": 10, "i64": 11, "f64": 12}
GT_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q",
          12: "d"}


def gguf_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def gguf_value(t, v):
    if t == GT["str"]:
        return gguf_str(v)
    return struct.pack("<" + GT_FMT[t], v)


def gguf_write(path, kvs, tensors, alignment=32):
    """kvs: [(key, type name, value)] - arrays as ("arr", (elem type name, [values]));
    tensors: [(name, ggml type, ne innermost first, raw bytes)]"""
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs))
    for k, t, v in kvs:
        head += gguf_str(k) + struct.pack("<I", GT[t])
        if t == "arr":
            et, vals = v
            head += struct.pack("<IQ", GT[et], len(vals))
            head += b"".join(gguf_value(GT[et], x) for x in vals)
        else:
            head += gguf_value(GT[t], v)
    off, data = 0, b""
    for name, gtype, ne, raw in tensors:
        pad = (-off) % alignment
        data += b"\0" * pad
        off += pad
        head += gguf_str(name) + struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
        head += struct.pack("<IQ", gtype, off)
        data += raw
        off += len(raw)
    head += b"\0" * ((-len(head)) % alignment)
    with open(path, "wb") as f:
        f.write(head + data)


def tiny_tokenizer(vocab):
    """tokenizer.ggml.* metadata of a 64-piece Unigram vocabulary for tiny-gguf: lets
    the tokenizer and the embedder load from the always-present fixture (their real
    gate is BGE-M3); the charsmap is one XCDA unit that matches nothing"""
    space = "\u2581"
    normal = [space] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    normal += [space + w for w in ["a", "ab", "abc", "the", "in", "er", "on", "b", "c"]]
    normal += ["ab", "abc", "bc", "cd", "de", "th", "he", "in", "er", "on", "an", "re",
               "the", "ing", "ion", "and", "ent", "ed", "es", "st", "or", "at", "en"]
    pieces = ["<s>", "<pad>", "</s>", "<unk>"] + normal
    pieces = pieces[:vocab - 1] + ["<mask>"]
    if len(pieces) != vocab or len(set(pieces)) != vocab:
        raise ValueError("tiny vocabulary does not fill the model rows")
    types = [3, 3, 3, 2] + [1] * (vocab - 5) + [3]
    scores = [0.0, 0.0, 0.0, 0.0] + [-1.0 - 0.125 * i for i in range(vocab - 5)] + [0.0]
    charsmap = list(struct.pack("<II", 4, 0))
    return [("tokenizer.ggml.model", "str", "t5"),
            ("tokenizer.ggml.pre", "str", "default"),
            ("tokenizer.ggml.tokens", "arr", ("str", pieces)),
            ("tokenizer.ggml.scores", "arr", ("f32", scores)),
            ("tokenizer.ggml.token_type", "arr", ("i32", types)),
            ("tokenizer.ggml.add_space_prefix", "bool", True),
            ("tokenizer.ggml.remove_extra_whitespaces", "bool", True),
            ("tokenizer.ggml.precompiled_charsmap", "arr", ("u8", charsmap)),
            ("tokenizer.ggml.bos_token_id", "u32", 0),
            ("tokenizer.ggml.eos_token_id", "u32", 2),
            ("tokenizer.ggml.unknown_token_id", "u32", 3),
            ("tokenizer.ggml.add_bos_token", "bool", True),
            ("tokenizer.ggml.add_eos_token", "bool", True)]


def tiny_gguf():
    """tiny-xlmr as llama.cpp's converter writes XLM-R: architecture bert, Q8_0 matrices,
    the first pad + 1 position rows dropped (positions count from 0); expected states
    computed with ggml's Q8_0 kernels for every matmul, everything else in float64 and
    rounded to f32 after each operation like the engine's buffers"""
    print("tiny-gguf")
    g = Ggml()
    src = load_file(os.path.join(OUT, "tiny-xlmr", "model.safetensors"))
    cfg = json.load(open(os.path.join(OUT, "tiny-xlmr", "config.json"), encoding="utf-8"))
    sd = {k: v.numpy().astype(np.float32) for k, v in src.items()}
    h, f, layers = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    heads, pad = cfg["num_attention_heads"], cfg["pad_token_id"]
    eps = cfg["layer_norm_eps"]
    tensors, w = [], {}

    def add_f32(name, a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        tensors.append((name, 0, list(reversed(a.shape)), a.tobytes()))
        w[name] = a

    def add_q8(name, a):
        raw = b"".join(g.quant_ref(r) for r in a)
        tensors.append((name, 8, list(reversed(a.shape)), raw))
        w[name] = raw

    add_f32("position_embd.weight", sd["embeddings.position_embeddings.weight"][pad + 1:])
    add_q8("token_embd.weight", sd["embeddings.word_embeddings.weight"])
    add_f32("token_embd_norm.bias", sd["embeddings.LayerNorm.bias"])
    add_f32("token_embd_norm.weight", sd["embeddings.LayerNorm.weight"])
    add_f32("token_types.weight", sd["embeddings.token_type_embeddings.weight"][0])
    names = {"attn_q": "attention.self.query", "attn_k": "attention.self.key",
             "attn_v": "attention.self.value", "attn_output": "attention.output.dense",
             "attn_output_norm": "attention.output.LayerNorm", "ffn_up": "intermediate.dense",
             "ffn_down": "output.dense", "layer_output_norm": "output.LayerNorm"}
    for i in range(layers):
        for gname in sorted(names):
            hname = f"encoder.layer.{i}.{names[gname]}"
            add_f32(f"blk.{i}.{gname}.bias", sd[hname + ".bias"])
            if gname.endswith("norm"):
                add_f32(f"blk.{i}.{gname}.weight", sd[hname + ".weight"])
            else:
                add_q8(f"blk.{i}.{gname}.weight", sd[hname + ".weight"])
    n_pos = cfg["max_position_embeddings"] - pad - 1
    kvs = [("general.architecture", "str", "bert"),
           ("general.name", "str", "tiny-xlmr"),
           ("bert.block_count", "u32", layers),
           ("bert.context_length", "u32", n_pos),
           ("bert.embedding_length", "u32", h),
           ("bert.feed_forward_length", "u32", f),
           ("bert.attention.head_count", "u32", heads),
           ("bert.attention.layer_norm_epsilon", "f32", eps),
           ("bert.attention.causal", "bool", False),
           ("bert.pooling_type", "u32", 2),
           ("tokenizer.ggml.padding_token_id", "u32", pad),
           ("general.file_type", "u32", 7)]
    kvs += tiny_tokenizer(cfg["vocab_size"])
    folder = os.path.join(OUT, "tiny-gguf")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "model.gguf")
    gguf_write(path, kvs, tensors)

    vocab = cfg["vocab_size"]
    word = np.stack([g.dequant(w["token_embd.weight"][i * h // 32 * Q8_BLOCK:
                                                      (i + 1) * h // 32 * Q8_BLOCK], h)
                     for i in range(vocab)])
    r32 = lambda a: np.asarray(a, dtype=np.float32)

    def ln(x, gname):
        xd = x.astype(np.float64)
        mu = xd.mean(-1, keepdims=True)
        var = ((xd - mu) ** 2).mean(-1, keepdims=True)
        return r32((xd - mu) / np.sqrt(var + eps) * w[gname + ".weight"] + w[gname + ".bias"])

    def lin(x, name, n_in, n_out):
        return r32(g.linear(x, w[name + ".weight"], n_in, n_out, w[name + ".bias"]))

    def fwd(ids, keep):
        n = len(ids)
        pos, seen = [], 0
        for t in ids:
            if t == pad:
                pos.append(0)
            else:
                pos.append(seen)
                seen += 1
        x = r32(word[ids].astype(np.float64) + w["position_embd.weight"][pos] +
                w["token_types.weight"])
        x = ln(x, "token_embd_norm")
        states = [x]
        hd = h // heads
        for i in range(layers):
            p = f"blk.{i}."
            q, k, v = (lin(x, p + n, h, h) for n in ["attn_q", "attn_k", "attn_v"])
            ctx = np.zeros((n, h), dtype=np.float64)
            for hh in range(heads):
                c = slice(hh * hd, (hh + 1) * hd)
                s = q[:, c].astype(np.float64) @ k[:, c].astype(np.float64).T / math.sqrt(hd)
                s = np.where(keep[None, :] == 1, s, -np.inf)
                e = np.exp(s - s.max(-1, keepdims=True))
                ctx[:, c] = (e / e.sum(-1, keepdims=True)) @ v[:, c].astype(np.float64)
            a = lin(r32(ctx), p + "attn_output", h, h)
            x = ln(r32(a.astype(np.float64) + x), p + "attn_output_norm")
            u = lin(x, p + "ffn_up", h, f).astype(np.float64)
            u = r32(0.5 * u * np.vectorize(math.erfc)(-u / math.sqrt(2)))
            d = lin(u, p + "ffn_down", f, h)
            x = ln(r32(d.astype(np.float64) + x), p + "layer_output_norm")
            states.append(x)
        return states

    blob = Blob()
    singles = [[0, 7, 2], [0, 5, 9, 33, 12, 2], [0, 63, 4, 4, 17, 50, 21, 8, 2]]
    for i, seq in enumerate(singles):
        hs = fwd(np.array(seq), np.ones(len(seq), dtype=np.int64))
        blob.add(f"single/{i}/input_ids", np.array(seq))
        for k, s in enumerate(hs):
            blob.add(f"single/{i}/hidden.{k}", s)
        cls = hs[-1][0].astype(np.float64)
        blob.add(f"single/{i}/embedding", cls / np.linalg.norm(cls))
    width = max(len(s) for s in singles)
    batch = [s + [pad] * (width - len(s)) for s in singles]
    blob.add("batch/input_ids", np.array(batch))
    for r, seq in enumerate(batch):
        hs = fwd(np.array(seq), (np.array(seq) != pad).astype(np.int64))
        blob.add(f"batch/{r}/hidden.{layers}", hs[-1], "padded positions are not meaningful")
    blob.write(folder, {
        "dataset": "tiny-gguf",
        "note": "tiny-xlmr converted like llama.cpp converts XLM-R (architecture bert, "
                "positions from 0, Q8_0 matrices and word embedding); expected states use "
                "ggml's Q8_0 kernels (activations quantized per row) for every matmul, "
                "float64 for the rest, f32 after each operation, exact (erf) GELU",
        "model": {"file": "model.gguf", "sha256": sha256_file(path),
                  "from": "tiny-xlmr/model.safetensors"},
        "ggml": {"sha256": g.libs},
        "environment": environment()})


def lembed_tokens():
    """lembed_tokenize_json for every case of bge-m3-tokens: the token oracle of phase 1b"""
    print("lembed tokens")
    import sqlite3
    if sha256_file(GGUF_Q8) != GGUF_Q8_SHA256:
        raise ValueError(f"{GGUF_Q8}: not the file of reference contract v1")
    src = os.path.join(OUT, "bge-m3-tokens")
    man = json.load(open(os.path.join(src, "manifest.json"), encoding="utf-8"))
    raw = open(os.path.join(src, "tensors.bin"), "rb").read()
    t = {e["name"]: e for e in man["tensors"]}
    def arr(name, dt):
        e = t[name]
        return np.frombuffer(raw[e["offset"]:e["offset"] + e["bytes"]], dtype=dt)
    texts, toff = arr("texts", np.uint8).tobytes(), arr("text_offsets", "<i4")
    ref, roff = arr("ids", "<i4"), arr("offsets", "<i4")
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(os.path.join(GGML_DIR, "lembed0"))
    version = db.execute("select lembed_version()").fetchone()[0]
    db.execute("INSERT INTO temp.lembed_models(name, model) VALUES ('m', lembed_model_from_file(?))",
               (GGUF_Q8,))
    flat, offsets, literal, same = [], [], [], 0
    specials = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
    for i, c in enumerate(man["cases"]):
        text = texts[toff[i]:toff[i + 1]].decode("utf-8")
        ids = json.loads(db.execute("select lembed_tokenize_json('m', ?)", (text,)).fetchone()[0])
        offsets.append(len(flat))
        flat += ids
        lit = any(s in text for s in specials)
        literal.append(int(lit))
        if ids == list(ref[roff[i]:roff[i + 1]]):
            same += 1
        elif not lit:
            raise ValueError(f"case {c['id']}: lembed {ids} differs from SentencePiece "
                             f"{list(ref[roff[i]:roff[i + 1]])} without a special literal")
    offsets.append(len(flat))
    # extra cases: pieces of other token types as text (BGE-M3 has the UNUSED
    # piece [PAD250000], id 250000) - the oracle never matches them
    extra = ["[PAD250000]", "x[PAD250000]y", "a [PAD250000] b"]
    ex_bytes, ex_offsets, ex_ids, ex_id_offsets = bytearray(), [], [], []
    for text in extra:
        ex_offsets.append(len(ex_bytes))
        ex_bytes += text.encode("utf-8")
        ids = json.loads(db.execute("select lembed_tokenize_json('m', ?)", (text,)).fetchone()[0])
        if 250000 in ids:
            raise ValueError(f"{text!r}: lembed matched the unused piece")
        ex_id_offsets.append(len(ex_ids))
        ex_ids += ids
    ex_offsets.append(len(ex_bytes))
    ex_id_offsets.append(len(ex_ids))
    blob = Blob()
    blob.add("extra/texts", np.frombuffer(bytes(ex_bytes), dtype=np.uint8),
             "extra case texts: an UNUSED piece of the vocabulary as plain text")
    blob.add("extra/text_offsets", np.array(ex_offsets, dtype=np.int64))
    blob.add("extra/ids", np.array(ex_ids, dtype=np.int64), "lembed ids of the extra cases")
    blob.add("extra/offsets", np.array(ex_id_offsets, dtype=np.int64))
    blob.add("ids", np.array(flat, dtype=np.int64), "all cases concatenated")
    blob.add("offsets", np.array(offsets, dtype=np.int64), "case i = ids[offsets[i] .. offsets[i+1]-1]")
    blob.add("special_literal", np.array(literal, dtype=np.int64),
             "1 = the text contains <s> </s> <pad> <unk> or <mask>: lembed reads it as a "
             "control token (parse_special=true), the native tokenizer must not")
    blob.write(os.path.join(OUT, "lembed-tokens"), {
        "dataset": "lembed-tokens",
        "note": "token ids of the lembed oracle for the cases of bge-m3-tokens (same order); "
                "every case without a special literal equals the SentencePiece reference",
        "cases_from": "bge-m3-tokens", "cases_sha256": man["blob"]["sha256"],
        "cases": len(man["cases"]), "equal_to_sentencepiece": same,
        "special_literal_cases": int(sum(literal)),
        "oracle": {"lembed": version, "sqlite": sqlite3.sqlite_version,
                   "model_file": os.path.basename(GGUF_Q8), "model_sha256": GGUF_Q8_SHA256,
                   "ggml_sha256": Ggml().libs},
        "environment": environment()})


def main():
    setup_determinism()
    what = set(sys.argv[1:]) or {"ops", "tiny", "bge-m3", "tokens", "q8", "q4", "q5",
                                 "mxfp4", "qk", "tiny-gguf", "lembed-tokens"}
    if "ops" in what:
        ops()
    if "tiny" in what:
        tiny()
    if "bge-m3" in what:
        snapshot = fetch_model()
        tok, model, manifest = bge_m3(snapshot)
        embed_de(tok, model, manifest)
    if "tokens" in what:
        tokens(fetch_model())
    if "q8" in what:
        q8()
    if "q4" in what:
        q4()
    if "q5" in what:
        q5()
    if "mxfp4" in what:
        mxfp4()
    if "qk" in what:
        qk()
    if "tiny-gguf" in what:
        tiny_gguf()
    if "lembed-tokens" in what:
        lembed_tokens()


if __name__ == "__main__":
    main()
