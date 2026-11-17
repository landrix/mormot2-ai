#!/usr/bin/env python3
"""Qwen3-0.6B as a pure Q5_1 GGUF file (backlog B16, Q5_1): no official one exists.

Every matrix the official Q8_0 file (Qwen/Qwen3-0.6B-GGUF) holds as Q8_0 is quantized
again to Q5_1 from the BF16 weights of model.safetensors (Qwen/Qwen3-0.6B) - like
`llama-quantize --pure ... Q5_1`, the embedding (tied head) included; the F32 tensors
(norms), the metadata and the tokenizer are copied from the Q8_0 file as they are, only
general.file_type becomes 9 (MOSTLY_Q5_1).

The quantizer is gguf-py's Q5_1 (llama.cpp, MIT; the numpy form of ggml's
quantize_row_q5_1_ref), taken from a llama.cpp checkout given by LAI_LLAMA_CPP. Each
source matrix is first checked against its Q8_0 form (cosine above 0.99), so a wrong
name mapping cannot pass.

    LAI_LLAMA_CPP=<llama.cpp checkout> python quantize_q5_1.py <qwen3-0.6b folder>
        <qwen3-0.6b-q8_0.gguf> <out.gguf>

Needs numpy only. Prints the SHA-256 of every input and of the output.
"""
import hashlib
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.environ["LAI_LLAMA_CPP"], "gguf-py"))
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as T, quants  # noqa: E402

ALIGN = 32  # the Q8_0 file sets no general.alignment


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def safetensors(path):
    """name -> (dtype, shape, absolute offset) of a safetensors file"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    base = 8 + n
    return {k: (v["dtype"], v["shape"], base + v["data_offsets"][0])
            for k, v in head.items() if k != "__metadata__"}


def hf_name(name):
    """the safetensors name of a GGUF tensor of the qwen3 architecture"""
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    parts = name.split(".")
    assert parts[0] == "blk", name
    i, what = parts[1], ".".join(parts[2:])
    table = {
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }
    return "model.layers.%s.%s" % (i, table[what])


def gguf_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def main():
    folder, q8path, out = sys.argv[1:4]
    stpath = os.path.join(folder, "model.safetensors")
    st = safetensors(stpath)
    mm = np.memmap(stpath, dtype=np.uint8, mode="r")
    raw = np.memmap(q8path, dtype=np.uint8, mode="r")
    r = gguf.GGUFReader(q8path)
    fields = list(r.fields.values())
    # the key/value region as it is: from the first key to the first tensor info
    kv_start = min(f.offset for f in fields if not f.name.startswith("GGUF."))
    ti_start = min(t.field.offset for t in r.tensors)
    kv = bytearray(raw[kv_start:ti_start].tobytes())
    ft = r.fields["general.file_type"]
    pos = ft.offset - kv_start + 8 + len("general.file_type")
    assert struct.unpack_from("<I", kv, pos)[0] == 4  # GGUF type UINT32
    assert struct.unpack_from("<I", kv, pos + 4)[0] == 7  # MOSTLY_Q8_0
    struct.pack_into("<I", kv, pos + 4, 9)  # MOSTLY_Q5_1
    infos, blobs, offset, worst = [], [], 0, (1.0, "")
    for t in r.tensors:
        if t.tensor_type == T.Q8_0:
            dtype, shape, at = st[hf_name(t.name)]
            assert dtype == "BF16" and list(reversed(shape)) == [int(d) for d in t.shape], t.name
            n = int(np.prod(shape))
            w = (mm[at:at + 2 * n].view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            w = w.reshape(shape)
            q8 = quants.dequantize(t.data, T.Q8_0).reshape(shape)
            cos = float(np.dot(w.ravel().astype(np.float64), q8.ravel()) /
                        (np.linalg.norm(w) * np.linalg.norm(q8)))
            assert cos > 0.99, (t.name, cos)
            worst = min(worst, (cos, t.name))
            data = quants.quantize(w, T.Q5_1).tobytes()
            kind = T.Q5_1
        else:
            assert t.tensor_type == T.F32, t.name
            data = t.data.tobytes()
            kind = T.F32
        info = gguf_string(t.name) + struct.pack("<I", len(t.shape))
        info += b"".join(struct.pack("<Q", int(d)) for d in t.shape)
        info += struct.pack("<IQ", int(kind), offset)
        infos.append(info)
        blobs.append(data)
        offset += len(data)
        offset += -offset % ALIGN
    head = b"GGUF" + struct.pack("<IQQ", 3, len(r.tensors), len(fields) - 3)
    head += bytes(kv) + b"".join(infos)
    head += b"\0" * (-len(head) % ALIGN)
    with open(out, "wb") as f:
        f.write(head)
        for b in blobs:
            f.write(b)
            f.write(b"\0" * (-len(b) % ALIGN))
    print("lowest cosine of a source matrix against its Q8_0 form: %.6f (%s)" % worst)
    for p in (stpath, q8path, out):
        print("sha256 %s  %s" % (sha256(p), os.path.basename(p)))


if __name__ == "__main__":
    main()
