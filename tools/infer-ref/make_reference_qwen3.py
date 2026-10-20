#!/usr/bin/env python3
"""Reference data for phase 2 of the native inference engine (decoder, Qwen3).

Targets (all of them when called without arguments):
  qwen3-unicode  src/infer/mormot.ai.infer.unicode.inc: the code point classes of the
                 Qwen2 pre-tokenizer (\\p{L}, \\p{N}, \\s) and the NFC tables, each
                 checked code point by code point against the Hugging Face tokenizer
  qwen3-tokens   tests/data/infer-ref/qwen3-tokens: the tokenizer gate (byte-level BPE)
  qwen3-chat     tests/data/infer-ref/qwen3-chat: the chat template gate
  tiny-qwen3     tests/data/infer-ref/tiny-qwen3: a seeded 2-layer Qwen3 (BF16 weights)
                 with every state, the logits and greedy continuations
  tiny-qwen3-gguf tests/data/infer-ref/tiny-qwen3-gguf: tiny-qwen3 as GGUF files (Q8_0,
                 Q4_0 and F16 from ggml's quantizers) and damaged variants
  tiny-qwen3moe  tests/data/infer-ref/tiny-qwen3moe: a seeded 3-layer Qwen3-MoE (layers 0
                 and 2 with 8 experts, top 2; layer 1 dense), every state, the logits
                 and greedy continuations; its variant without normed top-k weights;
                 and both as GGUF files with 3-D expert tensors (Q8_0, Q4_0)
  qwen3          tests/data/infer-ref/qwen3: Qwen3-0.6B in FP32 - states, logits and
                 greedy continuations of a fixed prompt set (the gate of phase 2)

Also copies the pinned model files to vendor/models/qwen3-0.6b/ (gitignored, or
$LAI_QWEN3_DIR). Same environment, format and rules as make_reference.py.
"""
import json
import os
import shutil
import sys
import unicodedata

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_reference as mr  # noqa: E402  (Blob, environment, cases, determinism)

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_FILES = ["config.json", "generation_config.json", "tokenizer.json",
               "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors",
               "LICENSE"]
MODEL_DIR = os.environ.get("LAI_QWEN3_DIR") or os.path.join(
    mr.ROOT, "vendor", "models", "qwen3-0.6b")
UNICODE_INC = os.path.join(mr.ROOT, "src", "infer", "mormot.ai.infer.unicode.inc")
TINY_SEED = 20261015
GREEDY_TOKENS = 48

# the code point classes of the pre-tokenizer
CLS_OTHER, CLS_LETTER, CLS_NUMBER, CLS_SPACE = 0, 1, 2, 3


# ---------------------------------------------------------------- model files

def fetch_model():
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=MODEL_FILES)
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name in MODEL_FILES:
        src = os.path.join(snapshot, name)
        dst = os.path.join(MODEL_DIR, name)
        if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)
                and mr.sha256_file(dst) == mr.sha256_file(src)):
            shutil.copyfile(src, dst)
    return snapshot


def model_info(snapshot):
    return {"id": MODEL_ID, "revision": REVISION,
            "files": {n: mr.sha256_file(os.path.join(snapshot, n)) for n in MODEL_FILES}}


def hf_tokenizer(snapshot):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(snapshot)


def plain_tokenizer(snapshot):
    """the tokenizer.json pipeline without its added tokens: every text is text

    That is how the native tokenizer encodes text (parse_special=false): control
    tokens come only from the chat template, never from user text. The HF
    tokenizer matches all 26 added tokens anywhere in a text, also <think> or
    <tool_call>, which are not even flagged special."""
    from tokenizers import Tokenizer
    with open(os.path.join(snapshot, "tokenizer.json"), encoding="utf-8") as f:
        t = json.load(f)
    t["added_tokens"] = []
    return Tokenizer.from_str(json.dumps(t))


def tokenizer_parts(snapshot):
    """the normalizer and the Split pre-tokenizer of tokenizer.json, checked"""
    from tokenizers import normalizers, pre_tokenizers, Regex
    with open(os.path.join(snapshot, "tokenizer.json"), encoding="utf-8") as f:
        t = json.load(f)
    if t["normalizer"] != {"type": "NFC"}:
        raise ValueError(f"unexpected normalizer {t['normalizer']}")
    seq = t["pre_tokenizer"]
    if seq["type"] != "Sequence" or len(seq["pretokenizers"]) != 2:
        raise ValueError("unexpected pre-tokenizer")
    split, byte_level = seq["pretokenizers"]
    if split["type"] != "Split" or split["behavior"] != "Isolated" or split["invert"]:
        raise ValueError(f"unexpected split {split}")
    if byte_level != {"type": "ByteLevel", "add_prefix_space": False,
                      "trim_offsets": False, "use_regex": False}:
        raise ValueError(f"unexpected byte level {byte_level}")
    pattern = split["pattern"]["Regex"]
    return normalizers.NFC(), pre_tokenizers.Split(Regex(pattern), "isolated"), pattern


# ---------------------------------------------------------------- Unicode tables

def classify(split):
    """class of every code point, measured with the HF pre-tokenizer itself

    letter: "a"+c stays one piece (\\p{L}+); number: "cc" splits (\\p{N} is one
    char); space: "c!" splits (U+0020 is known: " !" is one piece by design);
    other: the rest. The regex is the contract, not Python's unicodedata."""
    def n(s):
        return len(split.pre_tokenize_str(s))
    cls = bytearray(0x110000)
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        if n("a" + c) == 1:
            cls[cp] = CLS_LETTER
        elif cp == 0x20:
            cls[cp] = CLS_SPACE
        elif n(c + c) == 2:
            cls[cp] = CLS_NUMBER
        elif n(c + "!") == 2:
            cls[cp] = CLS_SPACE
        else:
            cls[cp] = CLS_OTHER
    return cls


def check_contractions(split, cls):
    """(?i:'s|'t|'re|'ve|'m|'ll|'d): which letters match, measured

    The native code accepts the ASCII letters in both cases plus U+017F LATIN
    SMALL LETTER LONG S for s (Unicode case folding); this proves that set."""
    letters = [cp for cp in range(0x110000) if cls[cp] == CLS_LETTER]

    def fold(ch):
        s = {ord(ch), ord(ch.upper())}
        if ch == "s":
            s.add(0x17F)
        return s
    # one-letter forms together ('s 't 'm 'd), then each position of 're 've 'll
    probes = [("", "", set().union(*(fold(x) for x in "stmd"))),
              ("", "e", fold("r") | fold("v")), ("", "l", fold("l")),
              ("r", "", fold("e")), ("v", "", fold("e")), ("l", "", fold("l"))]
    found = {}
    for before, after, expect in probes:
        got = set()
        for cp in letters:
            probe = "'" + before + chr(cp) + after
            pieces = [p for p, _ in split.pre_tokenize_str(probe + "b")]
            if pieces[0] == probe:
                got.add(cp)
        if got != expect:
            raise ValueError(f"contraction '{before}?{after}: "
                             f"{sorted(hex(x) for x in got)}")
        found[f"'{before}?{after}"] = sorted(got)
    return found


def ranges(values, keep):
    """[(lo, hi, value)] of the runs of equal values for which keep(value)"""
    out, start = [], None
    for cp in range(0x110001):
        v = values[cp] if cp < 0x110000 else None
        if start is not None and v != values[start]:
            out.append((start, cp - 1, values[start]))
            start = None
        if start is None and v is not None and keep(v):
            start = cp
    return out


def nfc_emulate(s, decomp, ccc, comp):
    """NFC as the native code computes it from the tables (UAX #15)"""
    SB, LB, VB, TB, LC, VC, TC = 0xAC00, 0x1100, 0x1161, 0x11A7, 19, 21, 28
    NC, SC = VC * TC, LC * VC * TC
    out = []
    for ch in s:
        cp = ord(ch)
        if SB <= cp < SB + SC:
            si = cp - SB
            out.append(LB + si // NC)
            out.append(VB + (si % NC) // TC)
            if si % TC:
                out.append(TB + si % TC)
        else:
            out += decomp.get(cp, [cp])
    # canonical ordering: stable sort of each run of non-starters
    i = 0
    while i < len(out):
        if ccc[out[i]]:
            j = i
            while j < len(out) and ccc[out[j]]:
                j += 1
            out[i:j] = sorted(out[i:j], key=lambda x: ccc[x])
            i = j
        else:
            i += 1
    # canonical composition
    if not out:
        return ""
    res = [out[0]]
    starter = 0 if ccc[out[0]] == 0 else -1
    last_ccc = 0 if starter == 0 else 256
    for cp in out[1:]:
        c = ccc[cp]
        if starter >= 0:
            s0 = res[starter]
            composed = None
            if LB <= s0 < LB + LC and VB <= cp < VB + VC and starter == len(res) - 1:
                composed = SB + ((s0 - LB) * VC + (cp - VB)) * TC
            elif (SB <= s0 < SB + SC and (s0 - SB) % TC == 0 and TB < cp < TB + TC
                  and starter == len(res) - 1):
                composed = s0 + cp - TB
            elif (last_ccc < c or (last_ccc == 0 and starter == len(res) - 1)) \
                    and (s0, cp) in comp:
                composed = comp[(s0, cp)]
            if composed is not None:
                res[starter] = composed
                continue
        if c == 0:
            starter = len(res)
            last_ccc = 0
        else:
            last_ccc = c
        res.append(cp)
    return "".join(chr(x) for x in res)


def nfc_tables(nfc):
    """decomposition, combining class and composition data for NFC, as the HF
    normalizer applies them

    The HF normalizer uses older Unicode data than Python's unicodedata: code
    points assigned since then have no decomposition, combining class 0 and no
    composition there. So decompositions come from the HF NFD itself, combining
    classes from unicodedata where HF reorders the mark (else 0), composition
    pairs from unicodedata where HF composes them. Then the native algorithm
    (nfc_emulate, the same steps as the Pascal code) must give the HF result for
    every code point, every pair, starters x marks and pairs of marks."""
    from tokenizers import normalizers
    nfd = normalizers.NFD()
    decomp, ccc = {}, [0] * 0x110000
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF or 0xAC00 <= cp <= 0xD7A3:
            continue
        c = chr(cp)
        d = nfd.normalize_str(c)
        if d != c:
            decomp[cp] = [ord(x) for x in d]
    low, high = chr(0x334), chr(0x301)  # combining classes 1 and 230
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        py = unicodedata.combining(chr(cp))
        if not py:
            continue
        if py > 1:  # m before a class-1 mark moves behind it only when HF knows m
            probe = "a" + chr(cp) + low
            moved = nfd.normalize_str(probe) == "a" + low + chr(cp)
        else:
            probe = "a" + high + chr(cp)
            moved = nfd.normalize_str(probe) == "a" + chr(cp) + high
        if moved:
            ccc[cp] = py
    comp = {}
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        m = unicodedata.decomposition(chr(cp))
        if not m or m.startswith("<"):
            continue
        parts = [int(x, 16) for x in m.split()]
        if len(parts) == 2 and nfc.normalize_str(chr(parts[0]) + chr(parts[1])) == chr(cp):
            comp[(parts[0], parts[1])] = cp
    bad = []

    def check(s, kind):
        if nfc.normalize_str(s) != nfc_emulate(s, decomp, ccc, comp):
            bad.append((kind, [hex(ord(x)) for x in s]))
    for cp in range(0x110000):
        if not 0xD800 <= cp <= 0xDFFF:
            check(chr(cp), "single")
    for (a, b) in comp:
        check(chr(a) + chr(b), "pair")
    marks = [cp for cp in range(0x110000) if unicodedata.combining(chr(cp)) if
             not 0xD800 <= cp <= 0xDFFF]
    starters = sorted({a for a, _ in comp} | {0x61, 0x41, 0x1100, 0xAC00, 0x915})
    for a in starters:
        for m in marks:
            check(chr(a) + chr(m), "mark")
    sample = marks[::5]
    for m1 in sample:
        for m2 in sample:
            check("a" + chr(m1) + chr(m2), "order")
            check(chr(m1) + "a" + chr(m2), "order")
    if bad:
        raise ValueError(f"native NFC differs from HF: {len(bad)} cases, first {bad[:10]}")
    return decomp, ccc, comp, len(marks), len(starters)


def pascal_array(name, values, typ, per_line=10, fmt="${:x}"):
    lines = [f"  {name}: array[0..{len(values) - 1}] of {typ} = ("]
    for i in range(0, len(values), per_line):
        chunk = ", ".join(fmt.format(v) for v in values[i:i + per_line])
        lines.append("    " + chunk + ("," if i + per_line < len(values) else ");"))
    return lines


def unicode_inc(snapshot):
    print("Unicode tables")
    nfc, split, pattern = tokenizer_parts(snapshot)
    cls = classify(split)
    contractions = check_contractions(split, cls)
    # Python's view, for the record only: the measured classes are the contract
    differs = 0
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        cat = unicodedata.category(chr(cp))
        py = (CLS_LETTER if cat[0] == "L" else CLS_NUMBER if cat[0] == "N" else
              CLS_SPACE if chr(cp).isspace() or cat in ("Zs", "Zl", "Zp") else CLS_OTHER)
        if py != cls[cp]:
            differs += 1
    print(f"  classes: {differs} code points differ from Python's categories")
    cls_ranges = ranges(cls, lambda v: v != CLS_OTHER)
    decomp, ccc, comp, nmarks, nstarters = nfc_tables(nfc)
    ccc_ranges = ranges(ccc, lambda v: v != 0)
    dcp = sorted(decomp)
    dofs, ddata = [], []
    for cp in dcp:
        dofs.append(len(ddata))
        ddata += decomp[cp]
    dofs.append(len(ddata))
    pairs = sorted(comp.items())
    # below this code point NFC changes nothing: no mark, no Hangul part, no second
    # half of a composition and no code point whose NFC differs from itself
    quick = min([cp for cp in range(0x110000) if ccc[cp]] + [p[0][1] for p in pairs] +
                [0x1100] + [cp for cp in range(0xD800) if nfc.normalize_str(chr(cp)) != chr(cp)])
    out = [
        "// Unicode data of the Qwen2/Qwen3 tokenizer - GENERATED, do not edit",
        "// - written by tools/infer-ref/make_reference_qwen3.py (target qwen3-unicode)",
        f"//   from {MODEL_ID} revision {REVISION}",
        "// - code point classes measured with the pre-tokenizer of tokenizer.json",
        "//   (tokenizers " + __import__("tokenizers").__version__ + "): 1 = \\p{L}, 2 = \\p{N},",
        "//   3 = \\s; unlisted = other; " + str(differs) + " code points differ from",
        "//   Python's unicodedata " + unicodedata.unidata_version + " categories",
        "// - NFC: canonical decompositions (full, ordered; Hangul syllables are",
        "//   algorithmic), canonical combining classes and primary composites from",
        "//   unicodedata " + unicodedata.unidata_version + ", checked against the HF normalizer for",
        f"//   every code point, every pair and {nstarters} starters x {nmarks} marks",
        "// - derived from the Unicode Character Database, (c) Unicode, Inc.,",
        "//   Unicode License v3 - see NOTICE",
        "",
        "const",
        f"  UNI_CLASS_COUNT = {len(cls_ranges)};",
        f"  UNI_CCC_COUNT = {len(ccc_ranges)};",
        f"  UNI_DECOMP_COUNT = {len(dcp)};",
        f"  UNI_DECOMP_DATA_COUNT = {len(ddata)};",
        f"  UNI_COMP_COUNT = {len(pairs)};",
        f"  UNI_NFC_QUICK_LIMIT = ${quick:x};",
        "",
    ]
    out += pascal_array("UNI_CLASS_LO", [r[0] for r in cls_ranges], "cardinal")
    out += pascal_array("UNI_CLASS_HI", [r[1] for r in cls_ranges], "cardinal")
    out += pascal_array("UNI_CLASS_KIND", [r[2] for r in cls_ranges], "byte", 20, "{}")
    out += pascal_array("UNI_CCC_LO", [r[0] for r in ccc_ranges], "cardinal")
    out += pascal_array("UNI_CCC_HI", [r[1] for r in ccc_ranges], "cardinal")
    out += pascal_array("UNI_CCC_VALUE", [r[2] for r in ccc_ranges], "byte", 20, "{}")
    out += pascal_array("UNI_DECOMP_CP", dcp, "cardinal")
    out += pascal_array("UNI_DECOMP_OFS", dofs, "word", 12, "{}")
    out += pascal_array("UNI_DECOMP_DATA", ddata, "cardinal")
    out += pascal_array("UNI_COMP_FIRST", [p[0][0] for p in pairs], "cardinal")
    out += pascal_array("UNI_COMP_SECOND", [p[0][1] for p in pairs], "cardinal")
    out += pascal_array("UNI_COMP_RESULT", [p[1] for p in pairs], "cardinal")
    text = "\n".join(out) + "\n"
    text.encode("ascii")  # the include must stay ASCII
    with open(UNICODE_INC, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
    print(f"  {os.path.relpath(UNICODE_INC, mr.ROOT)}: {len(cls_ranges)} class ranges, "
          f"{len(ccc_ranges)} ccc ranges, {len(dcp)} decompositions, {len(pairs)} pairs")
    return {"pattern": pattern, "contractions": contractions, "class_ranges": len(cls_ranges)}


# ---------------------------------------------------------------- tokenizer gate

ADDED = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>",
         "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>",
         "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
         "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>",
         "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
         "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>",
         "<think>", "</think>"]


def qwen_designed_cases():
    """edge cases of the Qwen2 split regex, byte-level BPE and NFC - (id, text)"""
    c = chr
    nl, cr, tab = c(10), c(13), c(9)
    cases = [
        # contractions: (?i:'s|'t|'re|'ve|'m|'ll|'d), U+017F folds to s
        ("contr_its", "it's"), ("contr_caps", "IT'S WE'RE THEY'VE I'M YOU'LL HE'D"),
        ("contr_mixed", "We'Re they'vE i'M you'Ll he'D"), ("contr_dont", "don't won't"),
        ("contr_long_s", "x'" + c(0x17F) + " y'" + c(0x17F) + "elf"),
        ("contr_glued", "'s's'S'd't'm"), ("contr_word", "Rohr'sche 'self 'rex 'velo"),
        ("contr_curly", "it" + c(0x2019) + "s"), ("contr_alone", "'"), ("contr_end", "abc'"),
        ("contr_space", "a ' s"), ("contr_digit", "'1 '2s"),
        # numbers: one \p{N} per piece
        ("digits", "12345"), ("digits_grouped", "1.000.000,50"), ("digits_mixed", "DN15x20"),
        ("digits_arabic", "".join(c(x) for x in [0x663, 0x664, 0x665])),
        ("digits_super", "x" + c(0xB2) + c(0xB3)), ("digits_roman", c(0x2163) + c(0x2164)),
        ("digits_fraction", c(0xBD) + c(0xBE)), ("digits_fullwidth", c(0xFF11) + c(0xFF12)),
        # newlines and whitespace runs
        ("nl_double", "a" + nl + nl + "b"), ("nl_spaced", "a " + nl + " b"),
        ("crlf_double", "a" + cr + nl + cr + nl + "b"), ("ws_nl_ws", "  " + nl + "  "),
        ("nl_trailing", "a" + nl), ("nl_only", nl * 3), ("cr_only", cr),
        ("ws_mix", "a  " + nl + nl + "  b"), ("ws_tab_nl", " " + tab + nl),
        ("ls", "x" + c(0x2028) + "y"), ("nel", "x" + c(0x85) + "y"),
        ("vt_ff", "a" + c(11) + c(12) + "b"), ("spaces_inner", "a   b"),
        ("tabs_inner", "a" + tab + tab + "b"), ("spaces_trail", "end   "),
        ("spaces_lead", "   start"), ("space_digit", "a 1"), ("space_punct", "a !"),
        ("nbsp_word", c(0xA0) + "Rohr"), ("ideo_space_word", c(0x3000) + "x"),
        ("tab_word", tab + "Rohr"), ("nl_word", nl + "Rohr"),
        ("indent", "def f(x):" + nl + "    return x" + nl + nl + "\tpass"),
        # punctuation runs
        ("punct_run", "!!!"), ("punct_mix", "?!."), ("punct_nl", "..." + nl + nl),
        ("space_punct_run", " ..."), ("punct_inner", "a...b"),
        ("punct_word", "!Rohr ?Rohr"), ("em_dash_spaced", "a " + c(0x2014) + " b"),
        ("json", '{"a": [1, 2.5, "x"], "b": null}'),
        ("pascal", "if a <> b then" + nl + "  Exit;"),
        # NFC: composition, reordering, singletons, exclusions, Hangul
        ("nfc_e_acute", "e" + c(0x301)), ("nfc_two_marks", "A" + c(0x30A) + c(0x301)),
        ("nfc_reorder", c(0x1E0B) + c(0x323)), ("nfc_reorder2", "q" + c(0x307) + c(0x323)),
        ("nfc_singleton_mark", "a" + c(0x344)), ("nfc_ohm", c(0x2126)),
        ("nfc_angstrom", c(0x212B)), ("nfc_kelvin", c(0x212A)),
        ("nfc_jamo_lvt", c(0x1100) + c(0x1161) + c(0x11A8)),
        ("nfc_lv_t", c(0xAC00) + c(0x11A8)), ("nfc_exclusion", c(0x958)),
        ("nfc_tibetan", c(0xF73)), ("nfc_cgj", "a" + c(0x34F) + c(0x301)),
        ("nfc_blocked", "a" + c(0x316) + c(0x301)), ("nfc_starter_block", "a" + "b" + c(0x301)),
        ("nfc_greek", c(0x3B1) + c(0x313) + c(0x301) + c(0x345)),
        ("nfc_vietnamese", "Vie" + c(0x323) + c(0x302) + "t"),
        ("nfc_already", c(0xE9) + c(0xFC) + c(0xC5)),
        ("nfc_mark_alone", c(0x301) + "a"), ("nfc_marks_only", c(0x301) + c(0x302)),
        ("nfc_devanagari_nukta", c(0x915) + c(0x93C)),
        ("nfc_kaithi", c(0x11099) + c(0x110BA)), ("nfc_compat_untouched", c(0xFB01) + c(0x2460)),
        # byte level: everything maps to bytes, no unknown token
        ("bytes_controls", "".join(c(x) for x in range(0, 32))), ("bytes_del", c(0x7F)),
        ("bytes_latin1", "".join(c(x) for x in range(0xA0, 0x100))),
        ("bytes_4byte", c(0x10FFFF) + c(0x1F600) + c(0xE0001)),
        ("bytes_private", c(0xF0000) + c(0x100000)),
        ("long_letters", "a" * 5000), ("long_punct", "!" * 3000), ("long_spaces", " " * 1000),
        ("long_digits_q", "1" * 500), ("long_newlines", nl * 400),
        ("long_word_q", "Rohrleitungsbauunternehmensverband" * 20),
        ("long_mixed", ("Kupferrohr 15 mm, " + c(0xFC) + "ber Putz; 3x DN20!" + nl) * 60),
        ("long_cjk", c(0x4E00) * 400 + c(0x7BA1) * 400),
    ]
    # the added tokens as literal text: plain text for the native tokenizer
    for i, lit in enumerate(ADDED):
        cases += [(f"lit_{i:02d}", lit), (f"lit_{i:02d}_glued", "a" + lit + "b"),
                  (f"lit_{i:02d}_spaced", "a " + lit + " b")]
    cases.append(("lit_chat", "<|im_start|>user" + nl + "Hallo<|im_end|>" + nl))
    cases.append(("lit_partial", "<|im_start <|im_end| <think <|endoftext|"))
    return cases


def qwen_generated_cases(n, seed):
    """random mixes over the regex classes - (id, text)"""
    import random
    rng = random.Random(seed)
    pools = [
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "0123456789" + chr(0x663) + chr(0xB2) + chr(0x2163),
        " " * 8 + chr(9) + chr(10) * 3 + chr(13) + chr(0xA0) + chr(0x3000) + chr(0x2028),
        "'" * 6 + "sStTrReEvVmMlLdD" + chr(0x17F),
        ".,;:-_/()[]{}+*=%&#@!?<>|" + chr(34) + chr(92),
        "".join(chr(x) for x in range(0x0300, 0x0320)) + chr(0x323) + chr(0x345),
        "".join(chr(x) for x in [0x1100, 0x1101, 0x1161, 0x1162, 0x11A8, 0x11A9, 0xAC00]),
        "".join(chr(x) for x in range(0x0915, 0x0920)) + chr(0x93C),
        "".join(chr(x) for x in [0xE4, 0xF6, 0xFC, 0xDF, 0xE9, 0x212B, 0x2126, 0xC5]),
        "".join(chr(x) for x in [0x1F600, 0x1F44D, 0x200D, 0xFE0F, 0x20000, 0x4E00]),
        "".join(chr(x) for x in range(0x0410, 0x0430)),
    ]
    weights = [30, 8, 14, 8, 10, 6, 4, 3, 8, 4, 5]
    cases, seen = [], set()
    while len(cases) < n:
        length = rng.randint(1, 48)
        text = "".join(rng.choice(rng.choices(pools, weights)[0]) for _ in range(length))
        if text in seen:
            continue
        seen.add(text)
        cases.append((f"qgen_{len(cases):04d}", text))
    return cases


def tokens(snapshot):
    print("Qwen3 tokenizer cases")
    nfc, _, _ = tokenizer_parts(snapshot)
    plain = plain_tokenizer(snapshot)
    hf = hf_tokenizer(snapshot)
    texts = []
    data = os.path.join(mr.ROOT, "tests", "data", "embed-de")
    for name in ["corpus", "queries"]:
        with open(os.path.join(data, name + ".jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    texts.append((f"{name}_{r['id']}", r["text"]))
    texts += mr.designed_cases()
    texts += qwen_designed_cases()
    texts += mr.generated_cases(900, mr.TINY_SEED)
    texts += qwen_generated_cases(900, TINY_SEED)
    if len({cid for cid, _ in texts}) != len(texts):
        raise ValueError("duplicated case id")
    flat, offsets, cases = [], [], []
    text_bytes, text_offsets = bytearray(), []
    norm_bytes, norm_offsets = bytearray(), []
    hf_differs = 0
    for cid, text in texts:
        ids = plain.encode(text, add_special_tokens=False).ids
        hf_ids = hf(text, add_special_tokens=False)["input_ids"]
        norm = nfc.normalize_str(text)
        literal = any(a in text for a in ADDED)
        if hf_ids != ids:
            if not literal:
                raise ValueError(f"{cid}: HF differs without an added-token literal")
            hf_differs += 1
        # decoding the ids gives back the normalized text, byte for byte
        back = plain.decode(ids, skip_special_tokens=False)
        if back != norm:
            raise ValueError(f"{cid}: decode differs")
        offsets.append(len(flat))
        flat += ids
        text_offsets.append(len(text_bytes))
        text_bytes += text.encode("utf-8")
        norm_offsets.append(len(norm_bytes))
        norm_bytes += norm.encode("utf-8")
        e = {"id": cid, "text": text, "tokens": len(ids)}
        if norm != text:
            e["normalized"] = norm
        if hf_ids != ids:
            e["hf_ids"] = hf_ids
        cases.append(e)
    offsets.append(len(flat))
    text_offsets.append(len(text_bytes))
    norm_offsets.append(len(norm_bytes))
    blob = mr.Blob()
    blob.add("texts", np.frombuffer(bytes(text_bytes), dtype=np.uint8),
             "UTF-8 of all case texts concatenated")
    blob.add("text_offsets", np.array(text_offsets, dtype=np.int64),
             "case i = texts[text_offsets[i] .. text_offsets[i+1]-1]")
    blob.add("normalized", np.frombuffer(bytes(norm_bytes), dtype=np.uint8),
             "UTF-8 of all NFC-normalized texts concatenated (HF normalizer)")
    blob.add("normalized_offsets", np.array(norm_offsets, dtype=np.int64),
             "case i = normalized[normalized_offsets[i] .. normalized_offsets[i+1]-1]")
    blob.add("ids", np.array(flat, dtype=np.int64), "all cases concatenated")
    blob.add("offsets", np.array(offsets, dtype=np.int64),
             "case i = ids[offsets[i] .. offsets[i+1]-1]")
    manifest = {
        "dataset": "qwen3-tokens",
        "note": "reference token ids: tokenizer.json without its added tokens (every "
                "text is text, parse_special=false), no special tokens added. decode(ids) "
                "= normalized text, checked. hf_ids only where the HF tokenizer differs: "
                "it reads the 26 added-token literals in text as control tokens",
        "model": model_info(snapshot),
        "environment": mr.environment(),
        "hf_differs": hf_differs,
        "cases": cases,
    }
    blob.write(os.path.join(mr.OUT, "qwen3-tokens"), manifest)


# ---------------------------------------------------------------- chat template gate

def tool(name, description, parameters):
    """a tool as TLlmTool carries it: name, description, raw JSON parameters"""
    return {"name": name, "description": description, "parameters": parameters}


def call(cid, name, arguments):
    return {"id": cid, "name": name, "arguments": arguments}


def msg(role, content, tool_calls=None, tool_call_id=None, reasoning=None):
    """a message as TLlmMessage carries it; reasoning is TLlmMessage.Reasoning
    (none when empty)"""
    m = {"role": role, "content": content}
    if tool_calls:
        m["tool_calls"] = tool_calls
    if tool_call_id:
        m["tool_call_id"] = tool_call_id
    if reasoning:
        m["reasoning"] = reasoning
    return m


WEATHER = tool("get_weather", "Aktuelles Wetter f\u00fcr einen Ort",
               '{"type":"object","properties":{"city":{"type":"string",'
               '"description":"Stadt, z. B. M\u00fcnchen"},"unit":{"type":"string",'
               '"enum":["celsius","fahrenheit"]}},"required":["city"]}')
NUMBERS = tool("numbers", "Zahlen und Escapes",
               '{ "type" : "object", "properties" : { "a" : { "type":"number", '
               '"minimum":0.5, "maximum":1E5, "multipleOf":1e-5, "default":-0.0, '
               '"examples":[100.0, 1.50, 0.0001, 2.5E+20, 1e16, 1e15, -0, '
               '12345678901234567890, 3, true, false, null] }, '
               '"s": {"type":"string", "pattern":"^[a-z]+\\\\d$", '
               '"description":"Zeile1\\nZeile2\\t\\"q\\" \\u0001 \\u00fc \\u20ac '
               '\\ud83d\\ude00 </script> / \\/"}}, "additionalProperties":false}')
NO_DESC = tool("ping", "", '{"type":"object","properties":{}}')
NO_PARAMS = tool("now", "Uhrzeit", "")


def chat_cases():
    nl = "\n"
    u = lambda t: msg("user", t)  # noqa: E731
    a = lambda t, calls=None: msg("assistant", t, calls)  # noqa: E731
    s = lambda t: msg("system", t)  # noqa: E731
    tr = lambda cid, t: msg("tool", t, tool_call_id=cid)  # noqa: E731
    r = lambda t, why, calls=None: msg("assistant", t, calls, reasoning=why)  # noqa: E731
    cases = [
        ("user", [u("Hallo")], [], True, None),
        ("user_no_think", [u("Hallo")], [], True, False),
        ("user_think", [u("Hallo")], [], True, True),
        ("user_no_prompt", [u("Hallo")], [], False, None),
        ("system_user", [s("Du bist ein Assistent f\u00fcr Handwerker."),
                         u("Wie verlege ich ein Kupferrohr?")], [], True, False),
        ("multi_turn", [u("Hallo"), a("Hallo! Wie kann ich helfen?"),
                        u("Erkl\u00e4re DN15.")], [], True, None),
        ("history_think", [u("2+2?"), a("<think>" + nl + "einfach" + nl + "</think>" + nl
                                        + nl + "4"), u("Und 3+3?")], [], True, None),
        ("history_think_only", [u("x"), a("<think>abc</think>"), u("y")], [], True, False),
        ("last_assistant", [u("Sag ja."), a("Ja.")], [], False, None),
        ("last_assistant_think", [u("Sag ja."), a("<think>" + nl + "kurz" + nl + "</think>"
                                                   + nl + nl + "Ja.")], [], False, None),
        ("assistant_newlines", [u("a"), a(nl + nl + "b" + nl), u("c")], [], True, None),
        ("empty_user", [u("")], [], True, None),
        ("unicode_user", [u("Gr\u00f6\u00dfe 30\u00d760, 5 \u20ac \U0001F600 "
                            "\u7ba1 \u0420\u0430\u0434\u0438\u0430\u0442\u043e\u0440")],
         [], True, False),
        ("second_system", [s("A"), u("b"), s("C"), u("d")], [], True, None),
        ("tools", [s("Sei knapp."), u("Wie ist das Wetter in Berlin?")], [WEATHER], True,
         False),
        ("tools_no_system", [u("Wetter?")], [WEATHER, NO_DESC, NO_PARAMS], True, None),
        ("tools_numbers", [u("x")], [NUMBERS], True, None),
        ("tool_flow", [u("Wie ist das Wetter in Berlin?"),
                       a("<think>" + nl + "Ich frage das Wetter ab." + nl + "</think>" + nl
                         + nl + "Ich schaue nach.",
                         [call("c1", "get_weather", '{"city": "Berlin"}')]),
                       tr("c1", '{"temp": 21, "sky": "sonnig"}')], [WEATHER], True, None),
        ("tool_flow_no_content", [u("Wetter in Ulm und Bonn?"),
                                  a("", [call("c1", "get_weather", '{"city":"Ulm"}'),
                                         call("c2", "get_weather",
                                              '{ "city" : "Bonn", "unit":"celsius" }')]),
                                  tr("c1", "18 Grad"), tr("c2", "20 Grad")],
         [WEATHER], True, False),
        ("tool_flow_answered", [u("Wetter Ulm?"),
                                a("", [call("c1", "get_weather", '{"city":"Ulm"}')]),
                                tr("c1", "18 Grad"), a("In Ulm sind es 18 Grad."),
                                u("Danke! Und morgen?")], [WEATHER], True, None),
        ("tool_flow_think_chain", [u("Plane."),
                                   a("<think>a</think>Schritt 1",
                                     [call("c1", "now", "{}")]),
                                   tr("c1", "12:00"),
                                   a("<think>b</think>Schritt 2")], [NO_PARAMS], False, None),
        # TLlmMessage.Reasoning, sent as reasoning_content
        ("reasoning_tool_flow", [u("Wie ist das Wetter in Berlin?"),
                                 r("Ich schaue nach.", "Ich frage das Wetter ab.",
                                   [call("c1", "get_weather", '{"city": "Berlin"}')]),
                                 tr("c1", '{"temp": 21}')], [WEATHER], True, None),
        ("reasoning_no_content", [u("Wetter Ulm?"),
                                  r("", nl + "Ulm abfragen." + nl + nl,
                                    [call("c1", "get_weather", '{"city":"Ulm"}')]),
                                  tr("c1", "18 Grad")], [WEATHER], True, None),
        ("reasoning_before_query", [u("2+2?"), r("4", "einfach"), u("Und 3+3?")], [],
         True, None),
        ("reasoning_last_no_prompt", [u("Sag ja."), r("Ja.", "kurz")], [], False, None),
        ("reasoning_chain", [u("Plane."),
                             r("Schritt 1", "a", [call("c1", "now", "{}")]),
                             tr("c1", "12:00"), r("Schritt 2", "b")], [NO_PARAMS], False,
         None),
    ]
    return cases


def chat_to_hf(messages, tools):
    hm = []
    for m in messages:
        h = {"role": m["role"], "content": m["content"]}
        if "tool_calls" in m:
            h["tool_calls"] = [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"], "arguments": c["arguments"]}}
                               for c in m["tool_calls"]]
        if "tool_call_id" in m:
            h["tool_call_id"] = m["tool_call_id"]
        if "reasoning" in m:
            h["reasoning_content"] = m["reasoning"]
        hm.append(h)
    ht = []
    for t in tools:
        f = {"name": t["name"]}
        if t["description"]:
            f["description"] = t["description"]
        if t["parameters"]:
            f["parameters"] = json.loads(t["parameters"])
        ht.append({"type": "function", "function": f})
    return hm, ht


def render(hf, messages, tools, gen, think):
    hm, ht = chat_to_hf(messages, tools)
    kw = {}
    if think is not None:
        kw["enable_thinking"] = think
    return hf.apply_chat_template(hm, tools=ht or None, add_generation_prompt=gen,
                                  tokenize=False, **kw)


def chat(snapshot):
    print("Qwen3 chat template cases")
    hf = hf_tokenizer(snapshot)
    flat, offsets, cases = [], [], []
    for cid, messages, tools, gen, think in chat_cases():
        text = render(hf, messages, tools, gen, think)
        ids = hf(text, add_special_tokens=False)["input_ids"]
        offsets.append(len(flat))
        flat += ids
        e = {"id": cid, "messages": messages, "tools": tools,
             "add_generation_prompt": gen, "rendered": text, "tokens": len(ids)}
        if think is not None:
            e["enable_thinking"] = think
        cases.append(e)
    offsets.append(len(flat))
    blob = mr.Blob()
    blob.add("ids", np.array(flat, dtype=np.int64), "all cases concatenated")
    blob.add("offsets", np.array(offsets, dtype=np.int64),
             "case i = ids[offsets[i] .. offsets[i+1]-1]")
    manifest = {
        "dataset": "qwen3-chat",
        "note": "the chat template of the pinned revision (tokenizer_config.json), "
                "rendered by transformers and tokenized by the HF tokenizer. Messages "
                "as TLlmMessage carries them (tool call arguments are raw JSON text, "
                "inserted verbatim), reasoning (TLlmMessage.Reasoning) as reasoning_content; "
                "tools as TLlmTool (name, description, raw JSON "
                "parameters), sent as {type: function, function: {name, description "
                "(when not empty), parameters (when not empty, json.loads)}}. No "
                "content holds an added-token literal: there the HF tokenizer would "
                "read control tokens, the native template keeps text",
        "model": model_info(snapshot),
        "environment": mr.environment(),
        "cases": cases,
    }
    blob.write(os.path.join(mr.OUT, "qwen3-chat"), manifest)


# ---------------------------------------------------------------- model forward

def record(model, ids):
    """one forward pass without cache: the input of the first layer, every layer
    output, layer 0 attention and MLP output, the final norm and the logits"""
    taken = {}
    hooks = []
    inner = model.model

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


def greedy(model, ids, steps, eos):
    """greedy continuation with the HF KV cache, plus each step's top-2 margin"""
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids),
                             dtype=torch.long), do_sample=False, max_new_tokens=steps,
                             eos_token_id=eos, pad_token_id=eos[0], output_logits=True,
                             return_dict_in_generate=True, temperature=None, top_p=None,
                             top_k=None)
    new = out.sequences[0, len(ids):].tolist()
    margins = []
    for step, logits in enumerate(out.logits):
        top = torch.topk(logits[0].double(), 2)
        if int(top.indices[0]) != new[step]:
            raise ValueError("generate did not pick the argmax")
        margins.append(float(top.values[0] - top.values[1]))
    return new, margins


def greedy_no_cache(model, ids, steps, eos):
    seq = list(ids)
    for _ in range(steps):
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([seq]), use_cache=False).logits[0, -1]
        t = int(torch.argmax(logits))
        seq.append(t)
        if t in eos:
            break
    return seq[len(ids):]


# ---------------------------------------------------------------- tiny Qwen3

def tiny_qwen3():
    print("tiny Qwen3")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from safetensors.torch import save_file
    torch.manual_seed(TINY_SEED)
    cfg = Qwen3Config(
        vocab_size=96, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        max_position_embeddings=512, rms_norm_eps=1e-6, rope_theta=1000000.0,
        tie_word_embeddings=True, attention_bias=False, hidden_act="silu",
        bos_token_id=1, eos_token_id=2, pad_token_id=0, use_sliding_window=False,
        attn_implementation="eager")
    model = Qwen3ForCausalLM(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1:  # the RMSNorm weights start as ones: make them count
                p.copy_(1.0 + 0.25 * torch.randn_like(p))
            else:
                p.copy_(0.08 * torch.randn_like(p))
            p.copy_(p.to(torch.bfloat16).to(torch.float32))  # BF16-exact, like the model
    model.eval()
    folder = os.path.join(mr.OUT, "tiny-qwen3")
    os.makedirs(folder, exist_ok=True)
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()
          if k != "lm_head.weight"}
    st = os.path.join(folder, "model.safetensors")
    save_file(sd, st, metadata={"format": "pt"})
    d = cfg.to_dict()
    keep = ["architectures", "vocab_size", "hidden_size", "intermediate_size",
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
            "max_position_embeddings", "rms_norm_eps", "rope_theta", "rope_parameters",
            "tie_word_embeddings", "attention_bias", "hidden_act", "bos_token_id",
            "eos_token_id", "pad_token_id", "model_type", "use_sliding_window",
            "sliding_window", "rope_scaling"]
    d["architectures"] = ["Qwen3ForCausalLM"]
    d["torch_dtype"] = "bfloat16"
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({k: d[k] for k in keep + ["torch_dtype"] if k in d}, f, indent=1)
        f.write("\n")
    # reload from the files: the contract is model.safetensors + config.json
    back = Qwen3ForCausalLM.from_pretrained(folder, dtype=torch.float32,
                                            attn_implementation="eager")
    back.eval()
    rng = np.random.default_rng(TINY_SEED)
    seqs = [rng.integers(3, 96, size=n).tolist() for n in (5, 9, 13)]
    blob = mr.Blob()
    entries = []
    for k, ids in enumerate(seqs):
        a = record(back, ids)
        b = record(back, ids)
        for name in a:
            if not np.array_equal(a[name], b[name]):
                raise ValueError(f"tiny {k} {name}: two runs differ")
        e = {"case": k, "ids": ids, "tokens": len(ids)}
        blob.add(f"case{k}.ids", np.array(ids, dtype=np.int64))
        for name, v in a.items():
            blob.add(f"case{k}.{name}", v)
        new, margins = greedy(back, ids, 16, [2])
        if greedy_no_cache(back, ids, 16, [2]) != new:
            raise ValueError(f"tiny {k}: greedy with and without cache differ")
        blob.add(f"case{k}.greedy", np.array(new, dtype=np.int64))
        blob.add(f"case{k}.greedy_margin", np.array(margins, dtype=np.float32))
        e["greedy"] = new
        e["min_margin"] = min(margins)
        entries.append(e)
    manifest = {
        "dataset": "tiny-qwen3",
        "note": "seeded Qwen3, 2 layers, hidden 64, 4 heads / 2 KV heads, head_dim 32 "
                "(not 64/4), tied embeddings, BF16 weights. The contract is "
                "model.safetensors + config.json. Per case: embed (input of layer 0), "
                "layerNN (output of layer NN), layer00_attn (o_proj output), "
                "layer00_mlp, norm (final RMSNorm), logits (all positions), greedy "
                "(16 tokens or up to eos 2, HF KV cache; equal without cache) with the "
                "top-2 logit margin per step",
        "config": json.load(open(os.path.join(folder, "config.json"), encoding="utf-8")),
        "safetensors_sha256": mr.sha256_file(st),
        "environment": mr.environment(),
        "cases": entries,
    }
    blob.write(folder, manifest)
    tiny_untied(model, cfg, folder, seqs[0])


def tiny_untied(model, cfg, folder, ids):
    """the same tiny model with its own lm_head (tie_word_embeddings false):
    the untied head path, which Qwen3-0.6B does not take"""
    from transformers import Qwen3ForCausalLM
    from safetensors.torch import save_file
    torch.manual_seed(TINY_SEED + 1)
    head = (0.08 * torch.randn(cfg.vocab_size, cfg.hidden_size)).to(torch.bfloat16)
    sub = os.path.join(folder, "untied")
    os.makedirs(sub, exist_ok=True)
    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()
          if k != "lm_head.weight"}
    sd["lm_head.weight"] = head.contiguous()
    st = os.path.join(sub, "model.safetensors")
    save_file(sd, st, metadata={"format": "pt"})
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        c = json.load(f)
    c["tie_word_embeddings"] = False
    with open(os.path.join(sub, "config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(c, f, indent=1)
        f.write("\n")
    back = Qwen3ForCausalLM.from_pretrained(sub, dtype=torch.float32,
                                            attn_implementation="eager")
    back.eval()
    if torch.equal(back.lm_head.weight, back.model.embed_tokens.weight):
        raise ValueError("untied head equals the embedding")
    a = record(back, ids)
    blob = mr.Blob()
    blob.add("case0.ids", np.array(ids, dtype=np.int64))
    blob.add("case0.logits", a["logits"])
    blob.add("case0.norm", a["norm"])
    blob.write(sub, {"dataset": "tiny-qwen3/untied",
                     "note": "tiny-qwen3 with its own lm_head (tie_word_embeddings "
                             "false): logits and final norm of case0",
                     "safetensors_sha256": mr.sha256_file(st),
                     "environment": mr.environment()})


# ---------------------------------------------------------------- Qwen3-0.6B

def prompts():
    """(id, messages or raw text, tools, enable_thinking, keep states)"""
    nl = "\n"
    return [
        ("raw_fox", "The quick brown fox", [], None, True),
        ("chat_de", [msg("user", "Wie verlege ich ein Kupferrohr in der Wand?")], [], False,
         True),
        ("chat_en", [msg("user", "What is the capital of France? Answer in one word.")], [],
         False, False),
        ("chat_system", [msg("system", "Du bist ein hilfreicher Assistent f\u00fcr Handwerker."),
                         msg("user", "Nenne drei Werkzeuge zum Fliesenlegen.")], [], False,
         False),
        ("chat_think", [msg("user", "Was ist 17 * 23?")], [], None, False),
        ("raw_code", "def fibonacci(n):" + nl, [], None, False),
        ("chat_tool", [msg("user", "Wie ist das Wetter in Berlin?")], [WEATHER], False,
         False),
    ]


def qwen3(snapshot):
    print("Qwen3-0.6B (FP32)")
    from transformers import Qwen3ForCausalLM
    hf = hf_tokenizer(snapshot)
    model = Qwen3ForCausalLM.from_pretrained(snapshot, dtype=torch.float32,
                                             attn_implementation="eager")
    model.eval()
    with open(os.path.join(snapshot, "generation_config.json"), encoding="utf-8") as f:
        eos = json.load(f)["eos_token_id"]
    blob = mr.Blob()
    entries = []
    for pid, source, tools, think, states in prompts():
        if isinstance(source, str):
            text = source
        else:
            text = render(hf, source, tools, True, think)
        ids = hf(text, add_special_tokens=False)["input_ids"]
        a = record(model, ids)
        b = record(model, ids)
        if not np.array_equal(a["logits"], b["logits"]):
            raise ValueError(f"{pid}: two runs differ")
        new, margins = greedy(model, ids, GREEDY_TOKENS, eos)
        again, _ = greedy(model, ids, GREEDY_TOKENS, eos)
        if again != new:
            raise ValueError(f"{pid}: two greedy runs differ")
        nocache = greedy_no_cache(model, ids, GREEDY_TOKENS, eos)
        e = {"id": pid, "tokens": len(ids), "text": text,
             "greedy": new, "greedy_text": hf.decode(new),
             "min_margin": min(margins), "no_cache_equal": nocache == new}
        if not isinstance(source, str):
            e["messages"] = source
            e["tools"] = tools
            if think is not None:
                e["enable_thinking"] = think
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
        "dataset": "qwen3",
        "note": "Qwen3-0.6B in FP32 (BF16 weights upcast exactly), eager attention, one "
                "thread. Per prompt: ids (chat prompts through the chat template with "
                "the generation prompt), logits_last, greedy (up to "
                f"{GREEDY_TOKENS} tokens or eos {eos}, HF KV cache) with the top-2 margin "
                "per step. With states: embed, layerNN outputs, layer00_attn, "
                "layer00_mlp, norm. no_cache_equal: the same greedy tokens without cache "
                "- a mismatch there marks a near tie inside the reference itself",
        "model": model_info(snapshot),
        "environment": mr.environment(),
        "prompts": entries,
    }
    blob.write(os.path.join(mr.OUT, "qwen3"), manifest)


QWEN3_GGUF_LAYER = [
    ("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"),
    ("self_attn.v_proj", "attn_v"), ("self_attn.o_proj", "attn_output"),
    ("self_attn.q_norm", "attn_q_norm"), ("self_attn.k_norm", "attn_k_norm"),
    ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"),
    ("mlp.down_proj", "ffn_down"), ("input_layernorm", "attn_norm"),
    ("post_attention_layernorm", "ffn_norm")]


def tiny_qwen3_gguf():
    """tiny-qwen3 as llama.cpp's converter and quantizer write it (architecture qwen3,
    tensor names blk.N.*, settings as qwen3.* metadata, a tied head left out), with
    the matrices quantized by ggml itself - plus damaged variants the loader refuses"""
    print("tiny-qwen3-gguf")
    from safetensors.torch import load_file
    g = mr.Ggml()
    folder = os.path.join(mr.OUT, "tiny-qwen3-gguf")
    os.makedirs(folder, exist_ok=True)

    def load(sub):
        src = os.path.join(mr.OUT, "tiny-qwen3", *sub)
        cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
        # BF16 in the file: widening to float32 is exact
        raw = load_file(os.path.join(src, "model.safetensors"))
        sd = {k: v.float().numpy() for k, v in raw.items()}
        return cfg, sd

    def metadata(cfg, arch="qwen3"):
        return [("general.architecture", "str", arch),
                ("general.name", "str", "tiny-qwen3"),
                (arch + ".block_count", "u32", cfg["num_hidden_layers"]),
                (arch + ".context_length", "u32", cfg["max_position_embeddings"]),
                (arch + ".embedding_length", "u32", cfg["hidden_size"]),
                (arch + ".feed_forward_length", "u32", cfg["intermediate_size"]),
                (arch + ".attention.head_count", "u32", cfg["num_attention_heads"]),
                (arch + ".attention.head_count_kv", "u32", cfg["num_key_value_heads"]),
                (arch + ".rope.freq_base", "f32", cfg["rope_parameters"]["rope_theta"]),
                (arch + ".attention.layer_norm_rms_epsilon", "f32", cfg["rms_norm_eps"]),
                (arch + ".attention.key_length", "u32", cfg["head_dim"]),
                (arch + ".attention.value_length", "u32", cfg["head_dim"])]

    def tensors(cfg, sd, matrix):
        """matrix: 'q8', 'q4' or 'f16' for the 2-D weights; norms stay F32"""
        out = []

        def add(name, a, kind):
            a = np.ascontiguousarray(a, dtype=np.float32)
            ne = list(reversed(a.shape))
            if kind == "f32":
                out.append((name, 0, ne, a.tobytes()))
            elif kind == "f16":
                out.append((name, 1, ne, a.astype(np.float16).tobytes()))
            elif kind == "q8":
                out.append((name, 8, ne, b"".join(g.quant_ref(r) for r in a)))
            else:
                out.append((name, 2, ne, b"".join(g.quant_q4(r) for r in a)))

        add("token_embd.weight", sd["model.embed_tokens.weight"],
            "q8" if matrix == "q4" else matrix)
        add("output_norm.weight", sd["model.norm.weight"], "f32")
        if "lm_head.weight" in sd:
            add("output.weight", sd["lm_head.weight"], "q8" if matrix == "q4" else matrix)
        for i in range(cfg["num_hidden_layers"]):
            for hf, gg in QWEN3_GGUF_LAYER:
                a = sd[f"model.layers.{i}.{hf}.weight"]
                add(f"blk.{i}.{gg}.weight", a, "f32" if a.ndim == 1 else matrix)
        return out

    cfg, sd = load([])
    files = {}
    files["q8_0.gguf"] = (metadata(cfg), tensors(cfg, sd, "q8"))
    files["q4_0.gguf"] = (metadata(cfg), tensors(cfg, sd, "q4"))
    ucfg, usd = load(["untied"])
    files["untied-f16.gguf"] = (metadata(ucfg), tensors(ucfg, usd, "f16"))
    # damaged: each one fails a different check of the loader
    kv, ts = metadata(cfg), tensors(cfg, sd, "q8")
    files["bad-arch.gguf"] = (metadata(cfg, "llama"), ts)
    files["bad-rope.gguf"] = (kv + [("qwen3.rope.scaling.type", "str", "linear")], ts)
    files["bad-extra.gguf"] = (kv, ts + [("blk.0.attn_q.bias", 0, [128],
                                          np.zeros(128, np.float32).tobytes())])
    files["bad-missing.gguf"] = (kv, [t for t in ts if t[0] != "blk.1.ffn_down.weight"])
    inf = [list(t) for t in ts]
    for t in inf:
        if t[0] == "blk.0.ffn_up.weight":
            raw = bytearray(t[3])
            raw[34 * 3:34 * 3 + 2] = b"\x00\x7c"  # the scale of block 3: +Inf
            t[3] = bytes(raw)
    files["bad-scale.gguf"] = (kv, [tuple(t) for t in inf])
    shape = [list(t) for t in ts]
    for t in shape:
        if t[0] == "blk.0.attn_q.weight":
            t[2] = list(reversed(t[2]))  # [in, out] instead of [out, in]: same bytes
    files["bad-shape.gguf"] = (kv, [tuple(t) for t in shape])
    files["bad-ropedim.gguf"] = (kv + [("qwen3.rope.dimension_count", "u32", 16)], ts)
    # an untied Q8_0 file: token_embd only read per token, never a matrix
    emb = [list(t) for t in tensors(ucfg, usd, "q8")]
    for t in emb:
        if t[0] == "token_embd.weight":
            raw = bytearray(t[3])
            raw[34 * 5:34 * 5 + 2] = bytes([0x00, 0x7e])  # the scale of block 5: NaN
            t[3] = bytes(raw)
    files["bad-embscale.gguf"] = (metadata(ucfg), [tuple(t) for t in emb])
    # tokenizers the BPE loader refuses: another pre-tokenizer, a byte token
    # (type 6) after the vocabulary - the 256 GPT-2 byte-level characters first
    alphabet = ([chr(c) for c in range(ord("!"), ord("~") + 1)] +
                [chr(c) for c in range(0xa1, 0xad)] + [chr(c) for c in range(0xae, 0x100)])
    alphabet += [chr(256 + i) for i in range(256 - len(alphabet))]

    def tokenizer(pre, extra_type):
        toks = alphabet + ["<|im_end|>"]
        return [("general.architecture", "str", "qwen3"),
                ("tokenizer.ggml.model", "str", "gpt2"),
                ("tokenizer.ggml.pre", "str", pre),
                ("tokenizer.ggml.tokens", "arr", ("str", toks)),
                ("tokenizer.ggml.token_type", "arr", ("i32", [1] * 256 + [extra_type])),
                ("tokenizer.ggml.merges", "arr", ("str", []))]
    files["tok-bad-pre.gguf"] = (tokenizer("llama-bpe", 3), [])
    files["tok-bad-type.gguf"] = (tokenizer("qwen2", 6), [])
    for name, (kvs, tens) in files.items():
        mr.gguf_write(os.path.join(folder, name), kvs, tens)
    manifest = {"dataset": "tiny-qwen3-gguf", "environment": mr.environment(),
                "ggml": {"dir": os.path.relpath(mr.GGML_DIR, mr.ROOT), "sha256": g.libs},
                "note": "tiny-qwen3 (and its untied variant) as GGUF: q8_0 and q4_0 "
                        "matrices from ggml's reference quantizers (token_embd Q8_0 in "
                        "both), untied-f16 with output.weight in F16; bad-*: another "
                        "architecture, RoPE scaling, an extra tensor, a missing one, an "
                        "infinite Q8_0 scale, a transposed shape, RoPE over 16 of 32 "
                        "dimensions, a NaN scale in the embedding of an untied file",
                "files": {n: mr.sha256_file(os.path.join(folder, n)) for n in sorted(files)}}
    with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)


# ---------------------------------------------------------------- tiny Qwen3-MoE

QWEN3MOE_SPARSE = [0, 2]  # the layers with experts; layer 1 is dense (mlp_only_layers)


def tiny_qwen3moe():
    """a seeded Qwen3-MoE: the router (softmax, top-k, normed or not), the experts,
    a dense layer between sparse ones - transformers' Qwen3MoeForCausalLM"""
    print("tiny Qwen3-MoE")
    # the first seed whose router keeps a clear gap between the k-th and the
    # (k+1)-th logit for every token: the expert choice must not hang on rounding
    for seed in range(TINY_SEED + 2, TINY_SEED + 202):
        model, cfg = moe_model(seed)
        gap = min(router_margin(model, ids) for ids in moe_cases())
        if gap >= MOE_ROUTER_GAP:
            break
    else:
        raise ValueError("no seed with a clear router")
    print(f"  seed {seed}, smallest router logit gap {gap:.4f}")
    folder = os.path.join(mr.OUT, "tiny-qwen3moe")
    write_moe(model, cfg, folder, "tiny-qwen3moe", True, seed)
    # the same weights, the top-k weights not normed (norm_topk_prob false)
    write_moe(model, cfg, os.path.join(folder, "unnormed"), "tiny-qwen3moe/unnormed",
              False, seed)


MOE_ROUTER_GAP = 0.05


def moe_cases():
    rng = np.random.default_rng(TINY_SEED + 2)
    return [rng.integers(3, 96, size=n).tolist() for n in (5, 9, 13)]


def moe_model(seed):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3MoeConfig(
        vocab_size=96, hidden_size=64, intermediate_size=160, moe_intermediate_size=64,
        num_experts=8, num_experts_per_tok=2, norm_topk_prob=True, decoder_sparse_step=1,
        mlp_only_layers=[1], num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, max_position_embeddings=512, rms_norm_eps=1e-6,
        rope_theta=1000000.0, tie_word_embeddings=True, attention_bias=False,
        hidden_act="silu", bos_token_id=1, eos_token_id=2, pad_token_id=0,
        use_sliding_window=False, attn_implementation="eager")
    model = Qwen3MoeForCausalLM(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1:
                p.copy_(1.0 + 0.25 * torch.randn_like(p))
            elif name.endswith("mlp.gate.weight"):
                # a router that decides: its logits spread over several units
                p.copy_(0.6 * torch.randn_like(p))
            else:
                p.copy_(0.08 * torch.randn_like(p))
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
    model.eval()
    return model, cfg


def write_moe(model, cfg, folder, dataset, normed, seed):
    from transformers import Qwen3MoeForCausalLM
    os.makedirs(folder, exist_ok=True)
    cfg.norm_topk_prob = normed
    model.config.norm_topk_prob = normed
    # BF16 in the file, like the checkpoints (the weights are BF16-exact)
    model.to(torch.bfloat16)
    model.save_pretrained(folder, safe_serialization=True)
    model.to(torch.float32)
    st = os.path.join(folder, "model.safetensors")
    # the config as the checkpoints write it (num_experts, not transformers 5's
    # num_local_experts), minus what only training reads
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        d = json.load(f)
    for k in ("transformers_version", "router_aux_loss_coef", "output_router_logits",
              "use_cache", "initializer_range", "attention_dropout", "dtype"):
        d.pop(k, None)
    d["num_experts"] = d.pop("num_local_experts")
    d["torch_dtype"] = "bfloat16"
    d["architectures"] = ["Qwen3MoeForCausalLM"]
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, indent=1)
        f.write("\n")
    for extra in ("generation_config.json",):
        if os.path.exists(os.path.join(folder, extra)):
            os.remove(os.path.join(folder, extra))
    back = Qwen3MoeForCausalLM.from_pretrained(folder, dtype=torch.float32,
                                               attn_implementation="eager")
    back.eval()
    if back.config.norm_topk_prob != normed:
        raise ValueError("norm_topk_prob not taken over")
    seqs = moe_cases()
    if not normed:
        seqs = seqs[:1]
    blob = mr.Blob()
    entries = []
    for k, ids in enumerate(seqs):
        a = record(back, ids)
        b = record(back, ids)
        for name in a:
            if not np.array_equal(a[name], b[name]):
                raise ValueError(f"{dataset} {k} {name}: two runs differ")
        e = {"case": k, "ids": ids, "tokens": len(ids)}
        blob.add(f"case{k}.ids", np.array(ids, dtype=np.int64))
        for name, v in a.items():
            blob.add(f"case{k}.{name}", v)
        new, margins = greedy(back, ids, 16, [2])
        if greedy_no_cache(back, ids, 16, [2]) != new:
            raise ValueError(f"{dataset} {k}: greedy with and without cache differ")
        blob.add(f"case{k}.greedy", np.array(new, dtype=np.int64))
        blob.add(f"case{k}.greedy_margin", np.array(margins, dtype=np.float32))
        e["greedy"] = new
        e["min_margin"] = min(margins)
        e["router_gap"] = router_margin(back, ids)
        entries.append(e)
    manifest = {
        "dataset": dataset,
        "note": "seeded Qwen3-MoE, 3 layers (0 and 2: 8 experts of 64, top 2, softmax "
                "router, norm_topk_prob " + str(normed).lower() + "; 1: dense SwiGLU of "
                "160), hidden 64, 4 heads / 2 KV heads, head_dim 32, tied embeddings, BF16 "
                "weights. The contract is model.safetensors + config.json (experts stored "
                "one by one, as the Hugging Face checkpoints). States as tiny-qwen3; "
                "router_gap: the smallest gap between the k-th and the (k+1)-th router "
                f"logit over every token and sparse layer (seed {seed}: the first one "
                f"with every gap >= {MOE_ROUTER_GAP})",
        "config": json.load(open(os.path.join(folder, "config.json"), encoding="utf-8")),
        "safetensors_sha256": mr.sha256_file(st),
        "environment": mr.environment(),
        "cases": entries,
    }
    blob.write(folder, manifest)


def router_margin(model, ids):
    """the smallest gap between the k-th and the (k+1)-th router logit: a tiny
    gap would make the expert choice depend on rounding"""
    gaps = []
    hooks = []

    def hook(mod, inp, out):
        top = torch.topk(out[0].double(), mod.top_k + 1, dim=-1).values
        gaps.append(float((top[:, mod.top_k - 1] - top[:, mod.top_k]).min()))

    for i in QWEN3MOE_SPARSE:
        hooks.append(model.model.layers[i].mlp.gate.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return min(gaps)


QWEN3MOE_GGUF_LAYER = [
    ("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"),
    ("self_attn.v_proj", "attn_v"), ("self_attn.o_proj", "attn_output"),
    ("self_attn.q_norm", "attn_q_norm"), ("self_attn.k_norm", "attn_k_norm"),
    ("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm")]


def tiny_qwen3moe_gguf():
    """tiny-qwen3moe as llama.cpp writes a qwen3moe file: the router as
    ffn_gate_inp (F32), the experts of a layer as one 3-D tensor each
    (ffn_gate_exps, ffn_up_exps [experts, out, in]; ffn_down_exps), the dense layer
    with ffn_gate/up/down - and a damaged variant"""
    print("tiny-qwen3moe-gguf")
    from safetensors.torch import load_file
    g = mr.Ggml()
    folder = os.path.join(mr.OUT, "tiny-qwen3moe", "gguf")
    os.makedirs(folder, exist_ok=True)
    src = os.path.join(mr.OUT, "tiny-qwen3moe")
    cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
    sd = {k: v.float().numpy() for k, v in
          load_file(os.path.join(src, "model.safetensors")).items()}
    arch = "qwen3moe"
    kv = [("general.architecture", "str", arch),
          ("general.name", "str", "tiny-qwen3moe"),
          (arch + ".block_count", "u32", cfg["num_hidden_layers"]),
          (arch + ".context_length", "u32", cfg["max_position_embeddings"]),
          (arch + ".embedding_length", "u32", cfg["hidden_size"]),
          (arch + ".feed_forward_length", "u32", cfg["intermediate_size"]),
          (arch + ".attention.head_count", "u32", cfg["num_attention_heads"]),
          (arch + ".attention.head_count_kv", "u32", cfg["num_key_value_heads"]),
          (arch + ".rope.freq_base", "f32", cfg["rope_parameters"]["rope_theta"]),
          (arch + ".attention.layer_norm_rms_epsilon", "f32", cfg["rms_norm_eps"]),
          (arch + ".attention.key_length", "u32", cfg["head_dim"]),
          (arch + ".attention.value_length", "u32", cfg["head_dim"]),
          (arch + ".expert_count", "u32", cfg["num_experts"]),
          (arch + ".expert_used_count", "u32", cfg["num_experts_per_tok"]),
          (arch + ".expert_feed_forward_length", "u32", cfg["moe_intermediate_size"])]

    def tensors(matrix):
        out = []

        def add(name, a, kind):
            a = np.ascontiguousarray(a, dtype=np.float32)
            ne = list(reversed(a.shape))
            rows = a.reshape(-1, a.shape[-1])
            if kind == "f32":
                out.append((name, 0, ne, a.tobytes()))
            elif kind == "q8":
                out.append((name, 8, ne, b"".join(g.quant_ref(r) for r in rows)))
            else:
                out.append((name, 2, ne, b"".join(g.quant_q4(r) for r in rows)))

        add("token_embd.weight", sd["model.embed_tokens.weight"], "q8")
        add("output_norm.weight", sd["model.norm.weight"], "f32")
        for i in range(cfg["num_hidden_layers"]):
            p = f"model.layers.{i}."
            for hf, gg in QWEN3MOE_GGUF_LAYER:
                a = sd[p + hf + ".weight"]
                add(f"blk.{i}.{gg}.weight", a, "f32" if a.ndim == 1 else matrix)
            if i in QWEN3MOE_SPARSE:
                add(f"blk.{i}.ffn_gate_inp.weight", sd[p + "mlp.gate.weight"], "f32")
                n = cfg["num_experts"]
                for hf, gg in (("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps"),
                               ("down_proj", "ffn_down_exps")):
                    a = np.stack([sd[p + f"mlp.experts.{e}.{hf}.weight"] for e in range(n)])
                    add(f"blk.{i}.{gg}.weight", a, matrix)
            else:
                for hf, gg in (("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"),
                               ("down_proj", "ffn_down")):
                    add(f"blk.{i}.{gg}.weight", sd[p + f"mlp.{hf}.weight"], matrix)
        return out

    files = {"q8_0.gguf": (kv, tensors("q8")), "q4_0.gguf": (kv, tensors("q4"))}
    # damaged: the experts of layer 2 one short
    short = []
    for t in tensors("q8"):
        if t[0] == "blk.2.ffn_up_exps.weight":
            ne = list(t[2])
            ne[2] -= 1
            t = (t[0], t[1], ne, t[3][:len(t[3]) * ne[2] // (ne[2] + 1)])
        short.append(t)
    files["bad-experts.gguf"] = (kv, short)
    for name, (kvs, tens) in files.items():
        mr.gguf_write(os.path.join(folder, name), kvs, tens)
    manifest = {"dataset": "tiny-qwen3moe/gguf", "environment": mr.environment(),
                "ggml": {"dir": os.path.relpath(mr.GGML_DIR, mr.ROOT), "sha256": g.libs},
                "note": "tiny-qwen3moe as qwen3moe GGUF files: q8_0 and q4_0 matrices from "
                        "ggml's reference quantizers (token_embd Q8_0, norms and routers "
                        "F32), the experts of layers 0 and 2 as 3-D tensors, layer 1 dense; "
                        "bad-experts: ffn_up_exps of layer 2 with 7 instead of 8 experts",
                "files": {n: mr.sha256_file(os.path.join(folder, n)) for n in sorted(files)}}
    with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)


def main():
    mr.setup_determinism()
    what = set(sys.argv[1:]) or {"qwen3-unicode", "qwen3-tokens", "qwen3-chat",
                                 "tiny-qwen3", "tiny-qwen3-gguf", "tiny-qwen3moe", "qwen3"}
    snapshot = fetch_model() if what - {"tiny-qwen3", "tiny-qwen3-gguf",
                                        "tiny-qwen3moe"} else None
    if "qwen3-unicode" in what:
        unicode_inc(snapshot)
    if "qwen3-tokens" in what:
        tokens(snapshot)
    if "qwen3-chat" in what:
        chat(snapshot)
    if "tiny-qwen3" in what:
        tiny_qwen3()
    if "tiny-qwen3-gguf" in what:
        tiny_qwen3_gguf()
    if "tiny-qwen3moe" in what:
        tiny_qwen3moe()
        tiny_qwen3moe_gguf()
    if "qwen3" in what:
        qwen3(snapshot)


if __name__ == "__main__":
    main()
