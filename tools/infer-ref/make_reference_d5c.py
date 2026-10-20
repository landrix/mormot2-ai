#!/usr/bin/env python3
"""Reference data for phase D slice D5c of the native inference engine: the tokenizer of
gpt-oss (o200k, the pre-tokenizer of tiktoken's o200k_base in tokenizer.json) and its
Harmony chat format.

Targets (all of them when called without arguments):
  o200k-unicode   src/infer/mormot.ai.infer.unicode.o200k.inc: the code point classes of
                  the o200k pre-tokenizer regex (upper and title case letters, lower case
                  letters, modifier and other letters, marks, numbers, white space), each
                  measured code point by code point with the regex of the tokenizers
                  library; the letters, numbers and white space must be those of the
                  Qwen2 table (mormot.ai.infer.unicode.inc), the same regex engine
  gpt-oss-tokens  tests/data/infer-ref/gpt-oss-tokens: the tokenizer gate - per case the
                  ids of tokenizer.json without its added tokens (every text is text), the
                  pieces of the pre-tokenizer as byte offsets, decoding checked
  gpt-oss-chat    tests/data/infer-ref/gpt-oss-chat: the Harmony gate - conversations and
                  tool schemas through chat_template.jinja of the revision, rendered by
                  transformers with a fixed date, and their ids; 400 generated schemas over
                  the branches of the TypeScript rendering; what transformers refuses is
                  marked as an error

Both need the tokenizer files of openai/gpt-oss-20b in the pinned revision (not
downloaded here: the folder $LAI_GPTOSS_DIR, by default vendor/models/gpt-oss-20b,
checked against the pinned hashes); the weights are not needed. Same environment, format
and rules as make_reference.py; the cases of make_reference.py and
make_reference_qwen3.py are reused.
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_reference as mr  # noqa: E402
import make_reference_d5 as d5  # noqa: E402
import make_reference_qwen3 as q3  # noqa: E402

# the files of the revision this slice reads, beyond those of make_reference_d5.py
GPTOSS_TOKENIZER_SHA256 = {
    "tokenizer.json": d5.GPTOSS_SHA256["tokenizer.json"],
    "tokenizer_config.json":
        "9279e942392b742d633c7adbb89ebe002c98399db8926a7af5125c726f404070",
    "special_tokens_map.json":
        "dd5e191d20c12d2fee1da5bae14ca1db0f5f4215300af691f23cdee97120a293",
    "chat_template.jinja":
        "a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146",
    "generation_config.json":
        "f9970ada892d2d1f72e3ed0a6535ccebadd11897318794ca671d8c7014c957da",
}
O200K_INC = os.path.join(mr.ROOT, "src", "infer", "mormot.ai.infer.unicode.o200k.inc")
QWEN_INC = os.path.join(mr.ROOT, "src", "infer", "mormot.ai.infer.unicode.inc")
TOKENS_SEED = 20261021

# the classes of the o200k regex, as TO200kClass numbers them
C_OTHER, C_UPPER, C_LOWER, C_LETTER, C_MARK, C_NUMBER, C_SPACE = range(7)
CLASS_NAMES = ["other", "upper", "lower", "letter", "mark", "number", "space"]

O200K_PATTERN = (
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+"
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
    "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}|"
    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")


def gpt_oss_folder():
    folder = os.environ.get("LAI_GPTOSS_DIR",
                            os.path.join(mr.ROOT, "vendor", "models", "gpt-oss-20b"))
    for name, want in GPTOSS_TOKENIZER_SHA256.items():
        got = mr.sha256_file(os.path.join(folder, name))
        if got != want:
            raise ValueError(f"{name}: sha256 {got}, pinned {want}")
    return folder


def model_info():
    return {"id": d5.GPTOSS_ID, "revision": d5.GPTOSS_REVISION,
            "files": GPTOSS_TOKENIZER_SHA256}


def tokenizer_parts(folder):
    """the pipeline of tokenizer.json, checked: no normalizer, Split(o200k, Isolated)
    + ByteLevel, BPE with ignore_merges, ByteLevel decoder"""
    from tokenizers import pre_tokenizers, Regex
    with open(os.path.join(folder, "tokenizer.json"), encoding="utf-8") as f:
        t = json.load(f)
    if t["normalizer"] is not None:
        raise ValueError(f"unexpected normalizer {t['normalizer']}")
    seq = t["pre_tokenizer"]
    if seq["type"] != "Sequence" or len(seq["pretokenizers"]) != 2:
        raise ValueError("unexpected pre-tokenizer")
    split, byte_level = seq["pretokenizers"]
    if (split["type"] != "Split" or split["behavior"] != "Isolated" or split["invert"]
            or split["pattern"] != {"Regex": O200K_PATTERN}):
        raise ValueError(f"unexpected split {split}")
    if byte_level != {"type": "ByteLevel", "add_prefix_space": False,
                      "trim_offsets": True, "use_regex": False}:
        raise ValueError(f"unexpected byte level {byte_level}")
    if t["decoder"]["type"] != "ByteLevel" or t["post_processor"]["type"] != "ByteLevel":
        raise ValueError("unexpected decoder or post-processor")
    m = t["model"]
    if m["type"] != "BPE" or not m["ignore_merges"] or m["byte_fallback"]:
        raise ValueError("unexpected model")
    return pre_tokenizers.Split(Regex(O200K_PATTERN), "isolated"), t


# ---------------------------------------------------------------- Unicode classes

def members(pattern, cps):
    """the code points of cps one regex class matches, measured with the Split
    pre-tokenizer of the tokenizers library (removed: what stays did not match)"""
    from tokenizers import pre_tokenizers, Regex
    split = pre_tokenizers.Split(Regex(pattern), "removed")
    out = set()
    chunk_len = 4096
    for i in range(0, len(cps), chunk_len):
        chunk = cps[i:i + chunk_len]
        s = "".join(chr(c) for c in chunk)
        kept = bytearray(len(chunk))
        for piece, (a, b) in split.pre_tokenize_str(s):
            if s[a:b] != piece:
                raise ValueError("offsets are not in characters")
            for k in range(a, b):
                kept[k] = 1
        out.update(c for k, c in enumerate(chunk) if not kept[k])
    return out


def qwen_classes():
    """the letters, numbers and white space of mormot.ai.infer.unicode.inc"""
    with open(QWEN_INC, encoding="ascii") as f:
        text = f.read()

    def array(name):
        m = re.search(name + r": array\[0\.\.\d+\] of \w+ = \((.*?)\);", text, re.S)
        return [int(v.replace("$", "0x"), 0) for v in m.group(1).replace("\n", " ")
                .split(",")]
    lo, hi, kind = array("UNI_CLASS_LO"), array("UNI_CLASS_HI"), array("UNI_CLASS_KIND")
    sets = {1: set(), 2: set(), 3: set()}
    for a, b, k in zip(lo, hi, kind):
        sets[k].update(range(a, b + 1))
    return sets[1], sets[2], sets[3]


def classify():
    """the class of every code point, from the sets of the regex, each checked"""
    cps = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    m = {k: members(p, cps) for k, p in [
        ("L", r"\p{L}"), ("N", r"\p{N}"), ("S", r"\s"), ("M", r"\p{M}"),
        ("U", r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]"), ("W", r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]"),
        ("P", r"[^\r\n\p{L}\p{N}]"), ("O", r"[^\s\p{L}\p{N}]")]}
    cls = bytearray(0x110000)
    for cp in cps:
        L, N, S, M = cp in m["L"], cp in m["N"], cp in m["S"], cp in m["M"]
        U, W = cp in m["U"], cp in m["W"]
        if S:
            if L or N or U or W:
                raise ValueError(f"U+{cp:04X}: white space in another class")
            c = C_SPACE
        elif N:
            if L or U or W:
                raise ValueError(f"U+{cp:04X}: number in another class")
            c = C_NUMBER
        elif L:
            if not (U or W) or M:
                raise ValueError(f"U+{cp:04X}: letter outside both letter sets")
            c = C_LETTER if U and W else C_UPPER if U else C_LOWER
        elif U or W:
            if not (U and W and M):
                raise ValueError(f"U+{cp:04X}: not a mark but in a letter set")
            c = C_MARK
        else:
            if M:
                raise ValueError(f"U+{cp:04X}: mark outside the letter sets")
            c = C_OTHER
        # the two negated classes of the regex follow from these
        if (cp in m["P"]) != (cp not in (10, 13) and c in (C_OTHER, C_MARK, C_SPACE)):
            raise ValueError(f"U+{cp:04X}: [^\\r\\n\\p{{L}}\\p{{N}}] differs")
        if (cp in m["O"]) != (c in (C_OTHER, C_MARK)):
            raise ValueError(f"U+{cp:04X}: [^\\s\\p{{L}}\\p{{N}}] differs")
        cls[cp] = c
    letters, numbers, spaces = qwen_classes()
    for name, want, got in [
            ("letters", letters, {cp for cp in cps if cls[cp] in (C_UPPER, C_LOWER, C_LETTER)}),
            ("numbers", numbers, {cp for cp in cps if cls[cp] == C_NUMBER}),
            ("white space", spaces, {cp for cp in cps if cls[cp] == C_SPACE})]:
        if want != got:
            raise ValueError(f"{name} differ from the Qwen2 table: "
                             f"{sorted(hex(x) for x in want ^ got)[:10]}")
    return cls


def check_contractions(split, cls):
    """(?i:'s|'t|'re|'ve|'m|'ll|'d) after a word: which code points match, measured

    The native code accepts the ASCII letters in both cases plus U+017F LATIN SMALL
    LETTER LONG S for s, as for the Qwen2 regex; this proves that set here."""
    cand = [cp for cp in range(0x110000)
            if cls[cp] in (C_UPPER, C_LOWER, C_LETTER, C_MARK)]

    def fold(ch):
        s = {ord(ch), ord(ch.upper())}
        if ch == "s":
            s.add(0x17F)
        return s
    probes = [("", "", set().union(*(fold(x) for x in "stmd"))),
              ("", "e", fold("r") | fold("v")), ("", "l", fold("l")),
              ("r", "", fold("e")), ("v", "", fold("e")), ("l", "", fold("l"))]
    found = {}
    for before, after, expect in probes:
        got = set()
        for cp in cand:
            # "ab" is a word of lower case letters: the suffix follows it directly
            probe = "ab'" + before + chr(cp) + after
            if split.pre_tokenize_str(probe + "!")[0][0] == probe:
                got.add(cp)
        if got != expect:
            raise ValueError(f"contraction '{before}?{after}: "
                             f"{sorted(hex(x) for x in got)}")
        found[f"'{before}?{after}"] = sorted(got)
    return found


def unicode_inc():
    print("o200k Unicode classes")
    folder = gpt_oss_folder()
    split, _ = tokenizer_parts(folder)
    cls = classify()
    contractions = check_contractions(split, cls)
    counts = {CLASS_NAMES[k]: sum(1 for cp in range(0x110000) if cls[cp] == k)
              for k in range(1, 7)}
    rng = q3.ranges(cls, lambda v: v != C_OTHER)
    import tokenizers
    out = [
        "// Unicode classes of the o200k pre-tokenizer (gpt-oss) - GENERATED, do not edit",
        "// - written by tools/infer-ref/make_reference_d5c.py (target o200k-unicode)",
        f"//   from {d5.GPTOSS_ID} revision {d5.GPTOSS_REVISION}",
        "// - measured code point by code point with the regex of tokenizers " +
        tokenizers.__version__ + ":",
        "//   1 = upper (\\p{Lu} \\p{Lt}), 2 = lower (\\p{Ll}), 3 = letter of both sets",
        "//   (\\p{Lm} \\p{Lo}), 4 = mark (\\p{M}), 5 = \\p{N}, 6 = \\s; unlisted = other",
        "// - the letters (1..3), numbers and white space equal those of",
        "//   mormot.ai.infer.unicode.inc (checked by the generator)",
        "// - derived from the Unicode Character Database, (c) Unicode, Inc.,",
        "//   Unicode License v3 - see NOTICE",
        "",
        "const",
        f"  UNI_O200K_COUNT = {len(rng)};",
        "",
    ]
    out += q3.pascal_array("UNI_O200K_LO", [r[0] for r in rng], "cardinal")
    out += q3.pascal_array("UNI_O200K_HI", [r[1] for r in rng], "cardinal")
    out += q3.pascal_array("UNI_O200K_KIND", [r[2] for r in rng], "byte", 20, "{}")
    text = "\n".join(out) + "\n"
    text.encode("ascii")
    with open(O200K_INC, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
    print(f"  {os.path.relpath(O200K_INC, mr.ROOT)}: {len(rng)} ranges, {counts}")
    return {"counts": counts, "contractions": contractions}


# ---------------------------------------------------------------- tokenizer gate

ADDED = ["<|startoftext|>", "<|endoftext|>", "<|return|>", "<|constrain|>", "<|channel|>",
         "<|start|>", "<|end|>", "<|message|>", "<|call|>", "<|endofprompt|>",
         "<|reserved_200000|>"]


def o200k_designed_cases():
    """edge cases of the o200k split regex - (id, text)"""
    c = chr
    nl, cr, tab = c(10), c(13), c(9)
    cases = [
        # upper case runs, then lower case: [U]*[W]+ and [U]+[W]*
        ("case_camel", "CamelCaseWordsHereXML"), ("case_acronym", "HTTPServer XMLHttpRequest"),
        ("case_caps", "DIN EN 1057 KUPFERROHR"), ("case_iphone", "iPhone eBay macOS"),
        ("case_single", "A b C d"), ("case_mixed", "aBcDeF AbCdEf"),
        ("case_contr", "DON'T don't Don'T it'S WE'RE we'Re THEY'VE I'M YOU'LL HE'D"),
        ("case_contr_upper_run", "ABC's ABC'S McDonald's O'Neil"),
        ("case_contr_long_s", "x'" + c(0x17F) + " X'" + c(0x17F) + "elf"),
        ("case_contr_alone", "'s 't 're 'S"), ("case_contr_digit", "1's 2'S"),
        ("case_contr_glued", "a'sb's'S'd"), ("case_contr_space", "a 's"),
        ("case_umlaut", "\u00c4rger \u00dcBER \u00f6l STRA\u00dfE Stra\u00dfe"),
        ("case_title", c(0x1C5) + "a " + c(0x1C8) + c(0x1C8) + "x " + c(0x1C4) + c(0x1C6)),
        ("case_greek", "\u0391\u03b8\u03ae\u03bd\u03b1 \u0391\u0398\u0397\u039d\u0391"),
        ("case_cyrillic", "\u041c\u043e\u0441\u043a\u0432\u0430 \u041c\u041e\u0421\u041a\u0412\u0410"),
        # modifier and other letters belong to both sets
        ("lm_mod", "a" + c(0x2B0) + "b " + c(0x2B0) + "A" + c(0x2B0)),
        ("lo_cjk", "\u7ba1\u9053\u5de5\u7a0b ABC\u7ba1 \u7ba1abc"),
        ("lo_hebrew", "\u05e9\u05dc\u05d5\u05dd A\u05e9"),
        ("lo_thai", "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35"),
        ("lo_arabic", "\u0645\u0631\u062d\u0628\u0627 \u0661\u0662\u0663\u0664"),
        # marks: in both letter sets, and in the prefix and other classes
        ("mark_after_upper", "A" + c(0x301) + "B" + c(0x301) + "c"),
        ("mark_after_lower", "e" + c(0x301) + "t" + c(0x301) + "E"),
        ("mark_alone", c(0x301) + " " + c(0x301) + c(0x302) + "A"),
        ("mark_upper_space", c(0x301) + "A B"), ("mark_punct", "!" + c(0x301) + "a"),
        ("mark_digit", "1" + c(0x301) + "2"), ("mark_space_word", " " + c(0x301) + "abc"),
        ("mark_devanagari", "\u0915\u093f\u0924\u093e\u092c \u0915\u0947"),
        ("mark_decomposed", "Vie" + c(0x323) + c(0x302) + "t U" + c(0x308) + "BER"),
        # numbers in groups of three
        ("num_groups", "1 12 123 1234 12345 123456 1234567"),
        ("num_mixed", "DN15x20 v1.2.3 3.14159 1,000,000"),
        ("num_other", c(0x663) + c(0x664) + c(0x665) + c(0x666) + " " + c(0xB2) + c(0xB3) +
         c(0xB9) + c(0xBC)),
        ("num_letters", "abc123def4567"),
        # punctuation runs take newlines and slashes after them
        ("slash_nl", "a/" + nl + "/b" + nl + "//" + nl + nl + "x"),
        ("path_nl", "src/infer/" + nl + "tests/" + cr + nl),
        ("url", "https://example.com/a/b?c=d&e=f#g"),
        ("comment", "x = 1 // note" + nl + "/* block */" + nl),
        ("punct_space_nl", " ..." + nl + " !!/" + nl),
        ("punct_cr", "end." + cr + cr + nl + "next"),
        # white space
        ("ws_runs", "a  b   c" + tab + tab + "d" + nl + nl + "  e" + nl),
        ("ws_trailing", "end  " + nl + "  "), ("ws_only", "   " + nl + "  " + tab),
        ("ws_unicode", "a" + c(0xA0) + "b" + c(0x3000) + "c" + c(0x2028) + "d" + c(0x85)),
        # code and markup
        ("code_py", "def fibonacci(n):" + nl + "    if n < 2:" + nl + "        return n" + nl),
        ("code_pas", "procedure TFoo.Bar(const aValue: RawUtf8);" + nl + "begin" + nl +
         "  inherited;" + nl + "end;"),
        ("json", '{"name": "get_weather", "arguments": {"city": "M\u00fcnchen"}}'),
        ("markdown", "## Title" + nl + nl + "- **bold** _it_ `code`" + nl + "| a | b |"),
        ("emoji", "\U0001F600\U0001F44D\U0001F3FD \u2764\ufe0f \U0001F468\u200d\U0001F469"),
        ("long_upper", "A" * 3000), ("long_mixed_case", "aB" * 2000),
        ("long_marks", "a" + c(0x301) * 2000), ("long_digits", "7" * 1000),
        ("long_slash_nl", ("/" + nl) * 500),
    ]
    for i, lit in enumerate(ADDED):
        cases += [(f"lit_{i:02d}", lit), (f"lit_{i:02d}_glued", "a" + lit + "b")]
    cases.append(("lit_harmony", "<|start|>assistant<|channel|>analysis<|message|>x<|end|>"))
    cases.append(("lit_call", "<|start|>assistant to=functions.f<|channel|>commentary "
                              "<|constrain|>json<|message|>{}<|call|>"))
    return cases


def o200k_generated_cases(n, seed):
    """random mixes over the classes of the o200k regex - (id, text)"""
    import random
    rng = random.Random(seed)
    pools = [
        "abcdefghijklmnopqrstuvwxyz\u00e4\u00f6\u00fc\u00df\u00e9",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ\u00c4\u00d6\u00dc\u00c9",
        "".join(chr(x) for x in [0x1C5, 0x1C8, 0x1CB, 0x1F2, 0x1F88, 0x1F8F]),
        "".join(chr(x) for x in [0x2B0, 0x2B2, 0x2C6, 0x3005, 0x4E00, 0x7BA1, 0x5D0, 0xE01]),
        "".join(chr(x) for x in range(0x0300, 0x0310)) + chr(0x93F) + chr(0x20DD),
        "0123456789" + chr(0x663) + chr(0xB2) + chr(0x2163) + chr(0xBD),
        " " * 8 + chr(9) + chr(10) * 3 + chr(13) + chr(0xA0) + chr(0x3000) + chr(0x2028),
        "'" * 6 + "sStTrReEvVmMlLdD" + chr(0x17F),
        ".,;:-_/()[]{}+*=%&#@!?<>|" + chr(34) + chr(92) + "/" * 4,
        "".join(chr(x) for x in [0x1F600, 0x1F44D, 0x200D, 0xFE0F, 0x20000, 0x2764]),
        "".join(chr(x) for x in range(0x0410, 0x0450, 3)),
    ]
    weights = [25, 14, 3, 6, 6, 8, 14, 8, 10, 3, 5]
    cases, seen = [], set()
    while len(cases) < n:
        length = rng.randint(1, 48)
        text = "".join(rng.choice(rng.choices(pools, weights)[0]) for _ in range(length))
        if text in seen:
            continue
        seen.add(text)
        cases.append((f"ogen_{len(cases):04d}", text))
    return cases


def plain_tokenizer(t, ignore_merges=True):
    """tokenizer.json without its added tokens: every text is text"""
    from tokenizers import Tokenizer
    t = json.loads(json.dumps(t))
    t["added_tokens"] = []
    t["model"]["ignore_merges"] = ignore_merges
    return Tokenizer.from_str(json.dumps(t))


def ignore_merges_cases(t, plain, merging, limit):
    """vocabulary tokens that BPE alone would split: one piece each, valid UTF-8"""
    from tokenizers import decoders
    dec = decoders.ByteLevel()
    vocab = t["model"]["vocab"]
    cases = []
    for text_bl, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
        try:
            text = dec.decode([text_bl])
        except Exception:
            continue
        if "\ufffd" in text or not text:
            continue
        if plain.encode(text).ids != [tid]:
            continue
        if merging.encode(text).ids != [tid]:
            cases.append((f"ignore_merges_{tid}", text))
            if len(cases) == limit:
                break
    return cases


def tokens():
    print("gpt-oss tokenizer cases")
    folder = gpt_oss_folder()
    split, t = tokenizer_parts(folder)
    plain = plain_tokenizer(t)
    merging = plain_tokenizer(t, ignore_merges=False)
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(folder)
    texts = []
    data = os.path.join(mr.ROOT, "tests", "data", "embed-de")
    for name in ["corpus", "queries"]:
        with open(os.path.join(data, name + ".jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    texts.append((f"{name}_{r['id']}", r["text"]))
    texts += mr.designed_cases()
    texts += q3.qwen_designed_cases()
    texts += [("o200k_" + cid, text) for cid, text in o200k_designed_cases()]
    texts += ignore_merges_cases(t, plain, merging, 24)
    texts += mr.generated_cases(600, mr.TINY_SEED)
    texts += q3.qwen_generated_cases(600, q3.TINY_SEED)
    texts += o200k_generated_cases(1200, TOKENS_SEED)
    ids_seen = [cid for cid, _ in texts]
    if len(set(ids_seen)) != len(ids_seen):
        raise ValueError(f"duplicated case id {[x for x in ids_seen if ids_seen.count(x) > 1]}")
    flat, offsets, cases = [], [], []
    pieces, piece_offsets = [], []
    text_bytes, text_offsets = bytearray(), []
    hf_differs = merges_differ = 0
    for cid, text in texts:
        ids = plain.encode(text, add_special_tokens=False).ids
        hf_ids = hf(text, add_special_tokens=False)["input_ids"]
        literal = any(a in text for a in ADDED) or "<|reserved_" in text
        if hf_ids != ids:
            if not literal:
                raise ValueError(f"{cid}: HF differs without an added-token literal")
            hf_differs += 1
        if merging.encode(text, add_special_tokens=False).ids != ids:
            merges_differ += 1
        if plain.decode(ids, skip_special_tokens=False) != text:
            raise ValueError(f"{cid}: decode differs")
        # the pieces as UTF-8 byte offsets of their ends, checked to cover the text
        piece_offsets.append(len(pieces))
        pos = 0
        for piece, (a, b) in split.pre_tokenize_str(text):
            if a != pos or text[a:b] != piece:
                raise ValueError(f"{cid}: pieces do not cover the text")
            pos = b
            pieces.append(len(text[:b].encode("utf-8")))
        if pos != len(text):
            raise ValueError(f"{cid}: pieces end early")
        offsets.append(len(flat))
        flat += ids
        text_offsets.append(len(text_bytes))
        text_bytes += text.encode("utf-8")
        e = {"id": cid, "text": text, "tokens": len(ids)}
        if hf_ids != ids:
            e["hf_ids"] = hf_ids
        cases.append(e)
    offsets.append(len(flat))
    piece_offsets.append(len(pieces))
    text_offsets.append(len(text_bytes))
    blob = mr.Blob()
    blob.add("texts", np.frombuffer(bytes(text_bytes), dtype=np.uint8),
             "UTF-8 of all case texts concatenated")
    blob.add("text_offsets", np.array(text_offsets, dtype=np.int64),
             "case i = texts[text_offsets[i] .. text_offsets[i+1]-1]")
    blob.add("ids", np.array(flat, dtype=np.int64), "all cases concatenated")
    blob.add("offsets", np.array(offsets, dtype=np.int64),
             "case i = ids[offsets[i] .. offsets[i+1]-1]")
    blob.add("pieces", np.array(pieces, dtype=np.int64),
             "the end of every pre-tokenizer piece, in bytes from the start of its text")
    blob.add("piece_offsets", np.array(piece_offsets, dtype=np.int64),
             "case i = pieces[piece_offsets[i] .. piece_offsets[i+1]-1]")
    manifest = {
        "dataset": "gpt-oss-tokens",
        "note": "reference token ids: tokenizer.json without its added tokens (every "
                "text is text, parse_special=false), no special tokens added, no "
                "normalizer; decode(ids) = text, checked. pieces: the Split "
                "pre-tokenizer (o200k regex) of the tokenizers library. hf_ids only "
                "where the HF tokenizer differs: it reads the added-token literals in "
                "text as control tokens. merges_differ: cases whose ids change when "
                "ignore_merges is off (BPE alone would split a word of the vocabulary)",
        "model": model_info(),
        "environment": mr.environment(),
        "hf_differs": hf_differs,
        "merges_differ": merges_differ,
        "cases": cases,
    }
    blob.write(os.path.join(mr.OUT, "gpt-oss-tokens"), manifest)
    print(f"  {len(cases)} cases, {len(flat)} ids, {len(pieces)} pieces, "
          f"HF differs {hf_differs}, ignore_merges matters in {merges_differ}")


# ---------------------------------------------------------------- Harmony chat template

CHAT_DATE = "2026-10-20"


def fixed_date():
    """transformers renders strftime_now with datetime.now(): a fixed day instead"""
    import datetime as dt
    import transformers.utils.chat_template_utils as ctu

    class Fixed(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 20, 12, 0, 0)
    ctu.datetime = Fixed


def chat_to_hf(messages, tools):
    """TLlmMessage / TLlmTool as the gpt-oss template reads them: arguments as an
    object (the template writes them with tojson), a description always (the
    template concatenates it: an empty one is "")"""
    hm = []
    for m in messages:
        h = {"role": m["role"], "content": m["content"]}
        if "tool_calls" in m:
            h["tool_calls"] = [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"],
                                             "arguments": json.loads(c["arguments"] or "{}")}}
                               for c in m["tool_calls"]]
        if "tool_call_id" in m:
            h["tool_call_id"] = m["tool_call_id"]
        if "reasoning" in m:
            h["thinking"] = m["reasoning"]
        hm.append(h)
    ht = []
    for t in tools:
        f = {"name": t["name"], "description": t["description"]}
        if t["parameters"]:
            f["parameters"] = json.loads(t["parameters"])
        ht.append({"type": "function", "function": f})
    return hm, ht


def harmony_cases():
    """(id, messages, tools, add_generation_prompt, options)"""
    nl = "\n"
    u = lambda t: q3.msg("user", t)  # noqa: E731
    a = lambda t, calls=None: q3.msg("assistant", t, calls)  # noqa: E731
    s = lambda t: q3.msg("system", t)  # noqa: E731
    tr = lambda cid, t: q3.msg("tool", t, tool_call_id=cid)  # noqa: E731
    r = lambda t, why, calls=None: q3.msg("assistant", t, calls, reasoning=why)  # noqa: E731
    W, P, NOW = q3.WEATHER, q3.NO_DESC, q3.NO_PARAMS
    return [
        ("user", [u("Hallo")], [], True, {}),
        ("user_no_prompt", [u("Hallo")], [], False, {}),
        ("effort_low", [u("Hallo")], [], True, {"reasoning_effort": "low"}),
        ("effort_high", [u("Hallo")], [], True, {"reasoning_effort": "high"}),
        ("identity", [u("Hallo")], [], True, {"model_identity": "Du bist ein Assistent."}),
        ("system_user", [s("Du bist ein Assistent f\u00fcr Handwerker."),
                         u("Wie verlege ich ein Kupferrohr?")], [], True, {}),
        ("empty_system", [s(""), u("x")], [], True, {}),
        ("multi_turn", [u("Hallo"), a("Hallo! Wie kann ich helfen?"),
                        u("Erkl\u00e4re DN15.")], [], True, {}),
        ("last_assistant", [u("Sag ja."), a("Ja.")], [], False, {}),
        ("last_assistant_prompt", [u("Sag ja."), a("Ja.")], [], True, {}),
        ("newlines", [u(nl + "a" + nl + nl), a(nl + "b" + nl), u("c" + nl)], [], True, {}),
        ("empty_user", [u("")], [], True, {}),
        ("unicode_user", [u("Gr\u00f6\u00dfe 30\u00d760, 5 \u20ac \U0001F600 \u7ba1")],
         [], True, {}),
        ("tools", [s("Sei knapp."), u("Wie ist das Wetter in Berlin?")], [W], True, {}),
        ("tools_no_system", [u("Wetter?")], [W, P, NOW], True, {}),
        ("tool_flow", [u("Wie ist das Wetter in Berlin?"),
                       a("", [q3.call("c1", "get_weather", '{"city": "Berlin"}')]),
                       tr("c1", '{"temp": 21, "sky": "sonnig"}')], [W], True, {}),
        ("tool_flow_analysis", [u("Wetter in Ulm?"),
                                a("Ich frage das Wetter ab.",
                                  [q3.call("c1", "get_weather", '{"city":"Ulm"}')]),
                                tr("c1", "18 Grad")], [W], True, {}),
        ("tool_flow_answered", [u("Wetter Ulm?"),
                                a("Ich frage nach.",
                                  [q3.call("c1", "get_weather", '{"city":"Ulm"}')]),
                                tr("c1", "18 Grad"), a("In Ulm sind es 18 Grad."),
                                u("Danke! Und morgen?")], [W], True, {}),
        ("tool_two_results", [u("Uhrzeit?"), a("", [q3.call("c1", "now", "")]),
                              tr("c1", "12:00"), tr("c1", "noch einmal 12:00")],
         [NOW], True, {}),
        ("tool_chain", [u("Plane."), a("", [q3.call("c1", "now", "{}")]),
                        tr("c1", "12:00"),
                        a("", [q3.call("c2", "get_weather",
                                       '{"city": "M\u00fcnchen", "unit": "celsius"}')]),
                        tr("c2", 'Sonne "hell"' + nl + "\tkalt \\ \u00fc")],
         [NOW, W], True, {}),
        ("tool_last_no_prompt", [u("Wetter?"),
                                 a("", [q3.call("c1", "get_weather", '{"city":"Ulm"}')])],
         [W], False, {}),
        ("tool_numbers", [u("x"), a("", [q3.call("c1", "numbers",
                                                 '{"a": 1.50, "b": [1E5, -0, true]}')]),
                          tr("c1", "ok")], [q3.NUMBERS], True, {}),
        # TLlmMessage.Reasoning, sent as thinking
        ("thinking_tool_flow", [u("Wetter in Ulm?"),
                                r("", "Ich frage das Wetter ab.",
                                  [q3.call("c1", "get_weather", '{"city":"Ulm"}')]),
                                tr("c1", "18 Grad")], [W], True, {}),
        ("thinking_answered", [u("Wetter Ulm?"),
                               r("", "Ich frage nach.",
                                 [q3.call("c1", "get_weather", '{"city":"Ulm"}')]),
                               tr("c1", "18 Grad"), r("In Ulm sind es 18 Grad.", "fertig"),
                               u("Danke!")], [W], True, {}),
        # content and thinking before a call: transformers refuses the pair, the
        # native template writes the analysis and a commentary preamble (as
        # gpt-oss wrote them) - tested in HarmonyTemplateRules, not here
        ("thinking_last_no_prompt", [u("Sag ja."), r("Ja.", "kurz" + nl)], [], False, {}),
        ("thinking_last_prompt", [u("Sag ja."), r("Ja.", "kurz")], [], True, {}),
        ("thinking_middle", [u("a"), r("b", "r"), u("c")], [], True, {}),
        ("thinking_chain", [u("Plane."),
                            r("", "erst die Uhrzeit", [q3.call("c1", "now", "{}")]),
                            tr("c1", "12:00"),
                            r("", nl + "dann das Wetter" + nl,
                              [q3.call("c2", "get_weather", '{"city":"Bonn"}')]),
                            tr("c2", "Sonne")], [NOW, W], True, {}),
        ("thinking_last_call_no_prompt", [u("Wetter?"),
                                          r("", "abfragen",
                                            [q3.call("c1", "get_weather",
                                                     '{"city":"Ulm"}')])], [W], False, {}),
    ]


def harmony_schemas():
    """designed tool schemas for the TypeScript rendering of the template - (id, schema)"""
    obj = lambda props, req=None, **kw: dict(  # noqa: E731
        {"type": "object", "properties": props}, **({"required": req} if req else {}), **kw)
    return [
        ("scalars", obj({"s": {"type": "string"}, "n": {"type": "number"},
                         "i": {"type": "integer"}, "b": {"type": "boolean"},
                         "x": {}, "nul": {"type": "null"}}, ["s", "b"])),
        ("described", obj({"city": {"type": "string", "description": "Stadt"},
                           "empty": {"type": "string", "description": ""}}, ["city"])),
        ("enum", obj({"unit": {"type": "string", "enum": ["c", "f"], "default": "c"},
                      "mixed": {"type": "string", "enum": ["a", 1, 2.5, True, None]},
                      "empty_enum": {"type": "string", "enum": []}})),
        ("nullable", obj({"s": {"type": "string", "nullable": True},
                          "a": {"type": "array", "items": {"type": "string"}, "nullable": True},
                          "n": {"type": "number", "nullable": True}})),
        ("arrays", obj({"s": {"type": "array", "items": {"type": "string"}},
                        "n": {"type": "array", "items": {"type": "number"}},
                        "i": {"type": "array", "items": {"type": "integer"}},
                        "b": {"type": "array", "items": {"type": "boolean"}},
                        "none": {"type": "array"}, "empty": {"type": "array", "items": {}},
                        "nested": {"type": "array", "items": {"type": "array",
                                                              "items": {"type": "string"}}},
                        "objs": {"type": "array", "items": obj({"k": {"type": "string"}})},
                        "tuple": {"type": "array", "items": [{"type": "string"}]},
                        "union": {"type": "array", "items": {"type": ["object", "object"]}},
                        "long": {"type": "array", "items": obj(
                            {"aaaaaaaaaa": {"type": "string"}, "bbbbbbbbbb": {"type": "number"},
                             "cccccccccc": {"type": "boolean"}})}})),
        ("type_list", obj({"a": {"type": ["string", "null"]}, "b": {"type": ["integer"]},
                           "c": {"type": []}, "d": {"type": ["number", "string", "boolean"]}})),
        ("one_of", obj({"v": {"oneOf": [{"type": "string", "description": "Name"},
                                        {"type": "number", "default": 3}]},
                        "o": {"oneOf": [obj({"a": {"type": "string"}}), {"type": "string"}],
                              "default": "x"},
                        "single": {"oneOf": [{"type": "boolean"}]}}, ["v"])),
        ("objects", obj({"o": obj({"a": {"type": "string"}, "b": {"type": "number"}}, ["a"]),
                         "e": {"type": "object"}, "deep": obj({"x": obj({"y": {}})})})),
        ("defaults", obj({"n": {"type": "number", "default": 10},
                          "f": {"type": "number", "default": 0.5},
                          "s": {"type": "string", "default": "a\"b\u00fc"},
                          "b": {"type": "boolean", "default": False},
                          "z": {"type": "string", "default": None},
                          "l": {"type": "array", "default": [1, "x"]},
                          "d": {"type": "object", "default": {"k": 1}}})),
        ("unicode", obj({"gr\u00f6\u00dfe": {"type": "string",
                                             "description": "\u20ac \U0001F600"}})),
        ("no_properties", {"type": "object"}),
        ("empty_properties", {"type": "object", "properties": {}}),
        ("not_required", obj({"a": {"type": "string"}}, [])),
        # item types of 59 characters (any[] by the limit of 50) and of 49 (kept)
        ("item_limit", obj({"t59": {"type": "array", "items": {"type": [
            "string", "number", "boolean", "null", "object", "array", "integer"]}},
            "t49": {"type": "array", "items": {"type": [
                "string", "number", "boolean", "null", "object", "array"]}}})),
        # zeros are false: no " | null", no description
        ("falsy", obj({"a": {"type": "string", "nullable": -0.0},
                       "b": {"type": "string", "nullable": 0.0, "description": -0.0},
                       "c": {"type": "array", "nullable": 0}, "d": {"type": "string",
                                                                  "nullable": 1}})),
    ]


def random_schema(rng, depth=0):
    """a parameter schema over the branches of render_typescript_type"""
    words = ["a", "b", "city", "n\u00e4me", "x_y", "Long Name"]
    kinds = ["string", "number", "integer", "boolean", "object", "array", "null", "list",
             "oneOf", "none", "enum"]
    k = rng.choice(kinds if depth < 3 else kinds[:4] + ["null", "none"])
    sp = {}
    if k == "list":
        sp["type"] = rng.sample(["string", "number", "null", "object", "boolean"],
                                rng.randint(0, 3))
    elif k == "oneOf":
        sp["oneOf"] = [random_schema(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    elif k == "enum":
        sp["type"] = "string"
        sp["enum"] = [rng.choice(["c", "f", "\u00fc", 1, 2.5, True, None, ""])
                      for _ in range(rng.randint(0, 3))]
    elif k == "array":
        sp["type"] = "array"
        r = rng.random()
        if r < 0.6:
            sp["items"] = random_schema(rng, depth + 1)
        elif r < 0.7:
            sp["items"] = {}
        elif r < 0.8:
            sp["items"] = [random_schema(rng, depth + 1)]
    elif k == "object":
        sp["type"] = "object"
        if rng.random() < 0.8:
            sp["properties"] = {rng.choice(words) + str(i): random_schema(rng, depth + 1)
                                for i in range(rng.randint(0, 3))}
        if rng.random() < 0.5 and sp.get("properties"):
            sp["required"] = rng.sample(list(sp["properties"]),
                                        rng.randint(0, len(sp["properties"])))
    elif k != "none":
        sp["type"] = k
    if rng.random() < 0.2:
        sp["nullable"] = rng.choice([True, False, 0, 1])
    if rng.random() < 0.3:
        sp["description"] = rng.choice(["Ein Wert", "", "a // b\nc", 5])
    if rng.random() < 0.25:
        sp["default"] = rng.choice([1, 0.5, "x", True, None, [1, 2], {"k": "v"}, -0.0])
    return sp


def random_tools(n, seed):
    import random
    rng = random.Random(seed)
    out = []
    for i in range(n):
        props = {f"p{j}": random_schema(rng) for j in range(rng.randint(1, 4))}
        params = {"type": "object", "properties": props}
        if rng.random() < 0.6:
            params["required"] = rng.sample(list(props), rng.randint(0, len(props)))
        out.append((f"random_{i:03d}", params))
    return out


def chat():
    print("gpt-oss chat template cases")
    folder = gpt_oss_folder()
    fixed_date()
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(folder)
    with open(os.path.join(folder, "chat_template.jinja"), encoding="utf-8") as f:
        if hf.chat_template != f.read():
            raise ValueError("the tokenizer does not use chat_template.jinja")
    flat, offsets, cases = [], [], []

    def add(cid, messages, tools, gen, options):
        hm, ht = chat_to_hf(messages, tools)
        e = {"id": cid, "messages": messages, "tools": tools, "add_generation_prompt": gen,
             "options": options}
        try:
            text = hf.apply_chat_template(hm, tools=ht or None, add_generation_prompt=gen,
                                          tokenize=False, **options)
        except Exception as x:  # what transformers refuses, the native one must too
            e["error"] = f"{type(x).__name__}: {x}"
            text = None
        offsets.append(len(flat))
        if text is not None:
            if CHAT_DATE not in text:
                raise ValueError(f"{cid}: the date is not fixed")
            ids = hf(text, add_special_tokens=False)["input_ids"]
            flat.extend(ids)
            e["rendered"] = text
            e["tokens"] = len(ids)
        cases.append(e)

    for cid, messages, tools, gen, options in harmony_cases():
        add(cid, messages, tools, gen, options)
    user = [q3.msg("user", "x")]
    for cid, params in harmony_schemas() + random_tools(400, TOKENS_SEED):
        tool = q3.tool("f_" + cid, "Schema " + cid, json.dumps(params, ensure_ascii=False))
        add("schema_" + cid, user, [tool], True, {})
    offsets.append(len(flat))
    blob = mr.Blob()
    blob.add("ids", np.array(flat, dtype=np.int64), "all cases concatenated")
    blob.add("offsets", np.array(offsets, dtype=np.int64),
             "case i = ids[offsets[i] .. offsets[i+1]-1] (none for an error case)")
    errors = sum(1 for e in cases if "error" in e)
    manifest = {
        "dataset": "gpt-oss-chat",
        "note": "the chat template of the revision (chat_template.jinja), rendered by "
                "transformers with the date fixed to " + CHAT_DATE + " (strftime_now) "
                "and tokenized by the HF tokenizer. Messages as TLlmMessage carries them "
                "(tool call arguments are raw JSON text, sent as their object; reasoning "
                "as thinking); tools as "
                "TLlmTool (name, description - always sent, parameters as the object). "
                "options: the template kwargs (reasoning_effort, model_identity). "
                "schema_*: one tool each, its parameters through the TypeScript "
                "rendering of the template; random_*: generated over its branches. "
                "error: transformers refused the case (the native template must too)",
        "date": CHAT_DATE,
        "model": model_info(),
        "environment": mr.environment(),
        "errors": errors,
        "cases": cases,
    }
    blob.write(os.path.join(mr.OUT, "gpt-oss-chat"), manifest)
    print(f"  {len(cases)} cases ({errors} refused), {len(flat)} ids")


TARGETS = {"o200k-unicode": unicode_inc, "gpt-oss-tokens": tokens, "gpt-oss-chat": chat}

if __name__ == "__main__":
    want = sys.argv[1:] or list(TARGETS)
    for name in want:
        if name not in TARGETS:
            raise SystemExit(f"unknown target {name}; known: {', '.join(TARGETS)}")
    for name in want:
        TARGETS[name]()
