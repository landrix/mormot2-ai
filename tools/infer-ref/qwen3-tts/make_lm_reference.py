"""Build the qwen3-tts-lm dataset: embeddings in, logits out, for two Qwen3 stacks.

The format of mormot2-ai's tests/data/infer-ref (manifest.json + tensors.bin,
read by tests/test.infer.ref.pas), so the folder can be dropped there as is.
Both stacks of Qwen3-TTS-12Hz-0.6B-CustomVoice take embedding rows instead of
token ids:
- talker (28 layers): the 22 prefill rows of a sentence, its hidden states
  and the logits of its last row (codec_head, 3072);
- predictor (5 layers): the 16 rows of the code predictor for frame 0, and
  the logits of its 15 heads (lm_head.g on row g + 1, 2048 each).

Every value comes from PyTorch (tools/tts/qwen_ref.py: CPU, float32, eager), except
predictor rows 2..15: the code predictor received them one by one with its KV
cache; they are rows of its codec embeddings (BF16 in the file, widened
exactly), taken here from the same model file.

Standard library only:

    python tools/tts/make_lm_reference.py --ref <qwen_ref.py output> --model <snapshot>
"""
import argparse
import array
import datetime
import hashlib
import json
import os
import struct
import sys

REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
HIDDEN = 1024
GROUPS = 16


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


class SafeTensors:
    """Just enough of a safetensors reader: header, then raw rows."""

    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.base = 8 + n

    def row_f32(self, name, row):
        t = self.header[name]
        if t["dtype"] != "BF16" or len(t["shape"]) != 2:
            sys.exit(f"{name}: {t['dtype']} {t['shape']}, expected a BF16 matrix")
        cols = t["shape"][1]
        self.f.seek(self.base + t["data_offsets"][0] + row * cols * 2)
        bf16 = array.array("H")
        bf16.frombytes(self.f.read(cols * 2))
        if sys.byteorder != "little":
            bf16.byteswap()
        # a BF16 value is the high half of the float32 with the same bits
        f32 = array.array("I", (v << 16 for v in bf16))
        if sys.byteorder != "little":
            f32.byteswap()
        return f32.tobytes()


class Blob:
    """The tensors.bin content and its manifest entries."""

    def __init__(self):
        self.data = bytearray()
        self.tensors = []

    def add(self, name, dtype, shape, raw, note=None):
        count = 1
        for d in shape:
            count *= d
        if len(raw) != count * 4:
            sys.exit(f"{name}: {len(raw)} bytes for shape {shape}")
        entry = {"name": name, "dtype": dtype, "shape": list(shape),
                 "offset": len(self.data), "bytes": len(raw),
                 "sha256": hashlib.sha256(raw).hexdigest()}
        if note:
            entry["note"] = note
        self.tensors.append(entry)
        self.data += raw


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ref", required=True, help="a tools/tts/qwen_ref.py output folder")
    p.add_argument("--model", required=True, help="the model snapshot folder")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "tests",
                                                 "data", "infer-ref", "qwen3-tts-lm"))
    args = p.parse_args()

    with open(os.path.join(args.ref, "manifest.json"), encoding="utf-8") as f:
        ref = json.load(f)

    def tensor(name, shape):
        t = ref["tensors"][name]
        if t["shape"] != shape:
            sys.exit(f"{name}: shape {t['shape']}, expected {shape}")
        with open(os.path.join(args.ref, t["file"]), "rb") as f:
            return f.read()

    with open(os.path.join(args.model, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    tc = config["talker_config"]
    pc = tc["code_predictor_config"]
    if tc["hidden_size"] != HIDDEN or pc["hidden_size"] != HIDDEN:
        sys.exit("unexpected hidden size")

    rows = ref["tensors"]["talker.prefill.embeds"]["shape"][0]
    embeds = tensor("talker.prefill.embeds", [rows, HIDDEN])
    layers = tensor("talker.prefill.layers", [tc["num_hidden_layers"] + 1, rows, HIDDEN])
    logits = tensor("talker.prefill.logits", [tc["vocab_size"]])
    pred2 = tensor("predictor.f0.embeds", [2, HIDDEN])
    pred_logits = tensor("predictor.f0.logits", [GROUPS - 1, pc["vocab_size"]])
    codes = ref["codes"][0]
    state = rows * HIDDEN * 4
    if layers[:state] != embeds:
        sys.exit("talker.prefill.layers[0] differs from talker.prefill.embeds")

    st = SafeTensors(os.path.join(args.model, "model.safetensors"))
    pred16 = bytearray(pred2)
    for g in range(1, GROUPS - 1):
        pred16 += st.row_f32(f"talker.code_predictor.model.codec_embedding.{g - 1}.weight",
                             codes[g])

    blob = Blob()
    blob.add("talker.embed", "f32", [rows, HIDDEN], embeds,
             "input rows (positions 0..%d): sums of text projections and codec "
             "embeddings, the prefill of one sentence" % (rows - 1))
    for k in range(tc["num_hidden_layers"] - 1):
        blob.add("talker.layer%02d" % k, "f32", [rows, HIDDEN],
                 layers[(k + 1) * state:(k + 2) * state])
    blob.add("talker.norm", "f32", [rows, HIDDEN], layers[-state:],
             "output of the final RMSNorm; the raw output of layer %02d is not "
             "available (transformers replaces the last hidden state by the normed "
             "one)" % (tc["num_hidden_layers"] - 1))
    blob.add("talker.logits_last", "f32", [tc["vocab_size"]], logits,
             "talker.codec_head.weight applied to the last row of talker.norm")
    blob.add("predictor.embed", "f32", [GROUPS, HIDDEN], bytes(pred16),
             "row 0: the talker state of frame 0 (talker.norm last row); row 1: "
             "talker.model.codec_embedding[code 0]; row g + 1 (g = 1..14): "
             "talker.code_predictor.model.codec_embedding.{g-1}[code g], widened "
             "from BF16 - codes from predictor.codes")
    blob.add("predictor.codes", "i32", [GROUPS], struct.pack("<%di" % GROUPS, *codes),
             "the greedy codes of frame 0 (argmax of each logits row, code 0 from "
             "the talker); the last one is not fed back")
    blob.add("predictor.logits", "f32", [GROUPS - 1, pc["vocab_size"]], pred_logits,
             "row g: talker.code_predictor.lm_head.{g}.weight applied to the normed "
             "state of position g + 1 - one forward of the 16 rows gives them all "
             "(causal attention)")

    data = bytes(blob.data)
    manifest = {
        "dataset": "qwen3-tts-lm",
        "note": "Two Qwen3 stacks of Qwen3-TTS-12Hz-0.6B-CustomVoice fed with embedding "
                "rows instead of token ids, one sequence from position 0, no padding. "
                "PyTorch in float32 (BF16 weights upcast exactly), eager attention, CPU. "
                "Both are plain Qwen3 decoders (per-head q_norm / k_norm, SwiGLU, no "
                "bias); the multimodal RoPE of the talker (interleaved, sections "
                "[24, 20, 20]) reduces to the plain rotate_half RoPE, its three "
                "positions being equal.",
        "model": {
            "id": REPO,
            "revision": os.path.basename(os.path.normpath(args.model)),
            "files": {
                "config.json": sha256_file(os.path.join(args.model, "config.json")),
                "model.safetensors": sha256_file(os.path.join(args.model, "model.safetensors")),
            },
        },
        "stacks": {
            "talker": {
                "prefix": "talker.model.",
                "head": "talker.codec_head.weight",
                "layers": tc["num_hidden_layers"],
                "hidden_size": tc["hidden_size"],
                "heads": tc["num_attention_heads"],
                "kv_heads": tc["num_key_value_heads"],
                "head_dim": tc["head_dim"],
                "intermediate_size": tc["intermediate_size"],
                "rms_norm_eps": tc["rms_norm_eps"],
                "rope_theta": tc["rope_theta"],
                "vocab_size": tc["vocab_size"],
            },
            "predictor": {
                "prefix": "talker.code_predictor.model.",
                "head": "talker.code_predictor.lm_head.{g}.weight, g = 0..14",
                "layers": pc["num_hidden_layers"],
                "hidden_size": pc["hidden_size"],
                "heads": pc["num_attention_heads"],
                "kv_heads": pc["num_key_value_heads"],
                "head_dim": pc["head_dim"],
                "intermediate_size": pc["intermediate_size"],
                "rms_norm_eps": pc["rms_norm_eps"],
                "rope_theta": pc["rope_theta"],
                "vocab_size": pc["vocab_size"],
            },
        },
        "source": {
            "text": ref["text"],
            "speaker": ref["speaker"],
            "language": ref["language"],
            "prompt_ids": ref["prompt_ids"],
        },
        "environment": {
            "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "torch": ref["torch"],
            "device": ref["device"],
            "dtype": ref["dtype"],
            "attn_implementation": ref["attn_implementation"],
            "generator": "tools/tts/qwen_ref.py, then tools/tts/make_lm_reference.py",
        },
        "blob": {
            "file": "tensors.bin",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "byte_order": "little",
            "layout": "row-major (C order)",
        },
        "tensors": blob.tensors,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "tensors.bin"), "wb") as f:
        f.write(data)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{len(blob.tensors)} tensors, {len(data)} bytes written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
