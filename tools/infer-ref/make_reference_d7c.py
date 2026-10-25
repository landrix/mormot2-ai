#!/usr/bin/env python3
"""Reference data for phase D slice D7c of the native inference engine: the tokenizer of
Qwen3.5 and its chat format.

Targets (all of them when called without arguments):
  qwen35-tokens   tests/data/infer-ref/qwen35-tokens: the tokenizer gate - per case the
                  ids of tokenizer.json without its added tokens (every text is text),
                  the NFC text the decoding gives back, and the pieces of the Split
                  pre-tokenizer (its regex joins the marks \\p{M} to the letters)
  qwen35-chat     tests/data/infer-ref/qwen35-chat: the template gate - conversations
                  and tool schemas through chat_template.jinja of the revision, rendered
                  by transformers, and their ids; what transformers refuses is marked as
                  an error

It needs the tokenizer files of Qwen/Qwen3.5-0.8B in the pinned revision (not downloaded
here: the folder $LAI_QWEN35_DIR, by default vendor/models/qwen3.5-0.8b, checked against
the pinned hashes); the weights are not needed. Same environment, format and rules as
make_reference.py; the cases of make_reference.py, make_reference_qwen3.py,
make_reference_d5c.py and make_reference_d6c.py are reused.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_reference as mr  # noqa: E402
import make_reference_qwen3 as q3  # noqa: E402
import make_reference_d5c as d5c  # noqa: E402
import make_reference_d6c as d6c  # noqa: E402

QWEN35_ID = "Qwen/Qwen3.5-0.8B"
QWEN35_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
QWEN35_SHA256 = {
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json":
        "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    "chat_template.jinja":
        "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
}
QWEN35_PATTERN = ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|"
                  "\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
TOKENS_SEED = 20261025


def qwen35_folder():
    folder = os.environ.get("LAI_QWEN35_DIR",
                            os.path.join(mr.ROOT, "vendor", "models", "qwen3.5-0.8b"))
    for name, want in QWEN35_SHA256.items():
        got = mr.sha256_file(os.path.join(folder, name))
        if got != want:
            raise ValueError(f"{name}: sha256 {got}, pinned {want}")
    return folder


def model_info():
    return {"id": QWEN35_ID, "revision": QWEN35_REVISION, "files": QWEN35_SHA256}


def tokenizer_parts(folder):
    """the pipeline of tokenizer.json, checked: NFC, Split(Qwen3.5, Isolated) +
    ByteLevel, plain BPE, ByteLevel decoder, no truncation or padding"""
    from tokenizers import normalizers, pre_tokenizers, Regex
    with open(os.path.join(folder, "tokenizer.json"), encoding="utf-8") as f:
        t = json.load(f)
    if t["normalizer"] != {"type": "NFC"}:
        raise ValueError(f"unexpected normalizer {t['normalizer']}")
    if t.get("truncation") is not None or t.get("padding") is not None:
        raise ValueError("truncation or padding set")
    seq = t["pre_tokenizer"]
    if seq["type"] != "Sequence" or len(seq["pretokenizers"]) != 2:
        raise ValueError("unexpected pre-tokenizer")
    split, byte_level = seq["pretokenizers"]
    if (split["type"] != "Split" or split["behavior"] != "Isolated" or split["invert"]
            or split["pattern"] != {"Regex": QWEN35_PATTERN}):
        raise ValueError(f"unexpected split {split}")
    if byte_level != {"type": "ByteLevel", "add_prefix_space": False,
                      "trim_offsets": False, "use_regex": False}:
        raise ValueError(f"unexpected byte level {byte_level}")
    if t["decoder"]["type"] != "ByteLevel" or t["post_processor"]["type"] != "ByteLevel":
        raise ValueError("unexpected decoder or post-processor")
    m = t["model"]
    if m["type"] != "BPE" or m["ignore_merges"] or m["byte_fallback"]:
        raise ValueError("unexpected model")
    return normalizers.NFC(), pre_tokenizers.Split(Regex(QWEN35_PATTERN), "isolated"), t


def added_tokens(t):
    return [a["content"] for a in t["added_tokens"]]


def plain_tokenizer(t):
    """tokenizer.json without its added tokens: every text is text"""
    from tokenizers import Tokenizer
    t = json.loads(json.dumps(t))
    t["added_tokens"] = []
    return Tokenizer.from_str(json.dumps(t))


def qwen35_designed_cases():
    """the marks the Qwen3.5 regex joins to the letters - (id, text)"""
    c = chr
    nl = c(10)
    acute, grave, circ, dia = c(0x301), c(0x300), c(0x302), c(0x308)
    return [
        ("mark_compose", "e" + acute + " a" + grave + " o" + dia + " n" + c(0x303)),
        ("mark_stack", "a" + acute + circ + c(0x323) + "b" + grave + grave + grave),
        ("mark_no_compose", "q" + acute + " x" + c(0x30A) + " j" + c(0x30C)),
        ("mark_alone", acute),
        ("mark_lead", acute + "abc"),
        ("mark_after_digit", "1" + acute + "2" + c(0x20E3) + " 3" + c(0xFE0F) + c(0x20E3)),
        ("mark_after_space", " " + acute + "x " + acute + acute),
        ("mark_after_punct", "!" + acute + "?" + acute + "a" + " ." + acute),
        ("mark_after_newline", nl + acute + "x" + nl + nl + acute),
        ("mark_after_tab", c(9) + acute + c(9) + "y"),
        ("mark_after_apostrophe", "it's" + acute + " don'" + acute + "t 'll" + acute),
        ("mark_runs", "a" + acute * 20 + " " + acute * 30),
        ("devanagari", "हिन्दी भाषा में लिखा गया पाठ"),
        ("tamil", "தமிழ் மொழி"),
        ("bengali", "বাংলা ভাষা"),
        ("thai", "ภาษาไทยเป็นภาษาที่สวยงาม"),
        ("arabic_harakat", "مَدْرَسَةٌ كَبِيرَةٌ"),
        ("hebrew_niqqud", "שָׁלוֹם עֲלֵיכֶם"),
        ("myanmar", "မြန်မာဘာသာ"),
        ("vietnamese_nfd", "Tie" + c(0x302) + acute + "ng Vie" + c(0x323) + c(0x302) + "t"),
        ("korean_jamo", c(0x1100) + c(0x1161) + c(0x11A8) + " " + c(0x1112) + c(0x1161)),
        ("emoji_vs16", c(0x2764) + c(0xFE0F) + " " + c(0x263A) + c(0xFE0F) + "x"),
        ("emoji_zwj", c(0x1F468) + c(0x200D) + c(0x1F469) + c(0x200D) + c(0x1F467)),
        ("emoji_skin", c(0x1F44D) + c(0x1F3FD) + " ok"),
        ("enclosing", "A" + c(0x20DD) + " B" + c(0x20DE) + " 7" + c(0x20E3)),
        ("mark_code", "x" + acute + " = y" + dia + " + 1;" + nl + "    z" + grave + "()"),
        ("mark_number_letter", "5" + c(0x301) + "km 2" + c(0x308) + "x"),
        ("mark_cjk", "管" + c(0x3099) + " か" + c(0x3099) + "き" + c(0x309A)),
        ("de_nfd", "Mu" + dia + "ller pru" + dia + "ft Gro" + dia + "ße"),
        ("de_sentence", "Die Kupferrohre im Heizungskeller müssen bis Freitag geprüft "
                        "werden – bitte mit Druckprotokoll."),
    ]


def qwen35_generated_cases(n, seed):
    """random mixes over the classes of the regex, marks among them - (id, text)"""
    import random
    rng = random.Random(seed)
    marks = [0x300, 0x301, 0x302, 0x308, 0x323, 0x30A, 0x93F, 0x94D, 0x93E, 0xE31, 0xE34,
             0xE48, 0x64B, 0x64E, 0x650, 0x5B4, 0x5BC, 0xFE0F, 0x20E3, 0x20DD, 0x3099,
             0x1AB0, 0x1DC0, 0xBCD, 0x9CD]
    pools = [
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüß",
        "".join(chr(x) for x in marks),
        " " * 10 + chr(9) * 2 + chr(10) * 3 + chr(13) + chr(0xA0) + chr(0x3000),
        "0123456789" + chr(0x663) + chr(0x967) + chr(0x2163),
        ".,;:-_/()[]{}+*=%&#@!?<>|'" + chr(34) + chr(92) + "'''",
        "sStTmMdDlLrRvVeE'",
        "".join(chr(x) for x in [0x915, 0x92E, 0x932, 0xE01, 0xE14, 0x627, 0x644, 0x5D0,
                                 0x5E9, 0x4E00, 0x7BA1, 0xAC00, 0x1F600, 0x1F44D, 0x200D]),
    ]
    weights = [26, 22, 14, 10, 10, 8, 10]
    cases, seen = [], set()
    while len(cases) < n:
        length = rng.randint(1, 48)
        text = "".join(rng.choice(rng.choices(pools, weights)[0]) for _ in range(length))
        if text in seen:
            continue
        seen.add(text)
        cases.append((f"q35gen_{len(cases):04d}", text))
    return cases


def tokens():
    print("Qwen3.5 tokenizer cases")
    folder = qwen35_folder()
    nfc, split, t = tokenizer_parts(folder)
    added = added_tokens(t)
    plain = plain_tokenizer(t)
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
    texts += [("o200k_" + cid, text) for cid, text in d5c.o200k_designed_cases()]
    texts += [("gemma_" + cid, text) for cid, text in d6c.gemma_designed_cases([])]
    texts += [("q35_" + cid, text) for cid, text in qwen35_designed_cases()]
    texts += [(f"q35_lit_{i:02d}", "a" + lit + " " + lit + "b") for i, lit in enumerate(added)]
    texts += mr.generated_cases(300, mr.TINY_SEED)
    texts += q3.qwen_generated_cases(300, q3.TINY_SEED)
    texts += d5c.o200k_generated_cases(300, d5c.TOKENS_SEED)
    texts += qwen35_generated_cases(1500, TOKENS_SEED)
    ids_seen = [cid for cid, _ in texts]
    if len(set(ids_seen)) != len(ids_seen):
        raise ValueError(f"duplicated case id {[x for x in ids_seen if ids_seen.count(x) > 1]}")
    flat, offsets, cases = [], [], []
    pieces, piece_offsets = [], []
    text_bytes, text_offsets = bytearray(), []
    norm_bytes, norm_offsets = bytearray(), []
    hf_differs = normalized = marks = 0
    import unicodedata
    for cid, text in texts:
        ids = plain.encode(text, add_special_tokens=False).ids
        hf_ids = hf(text, add_special_tokens=False)["input_ids"]
        norm = nfc.normalize_str(text)
        if hf_ids != ids:
            if not any(a in text for a in added):
                raise ValueError(f"{cid}: HF differs without an added-token literal")
            hf_differs += 1
        # decoding the ids gives back the normalized text, byte for byte
        if plain.decode(ids, skip_special_tokens=False) != norm:
            raise ValueError(f"{cid}: decode differs")
        # the pieces of the normalized text, as UTF-8 byte offsets of their ends
        piece_offsets.append(len(pieces))
        pos = 0
        for piece, (a, b) in split.pre_tokenize_str(norm):
            if a != pos or norm[a:b] != piece:
                raise ValueError(f"{cid}: pieces do not cover the text")
            pos = b
            pieces.append(len(norm[:b].encode("utf-8")))
        if pos != len(norm):
            raise ValueError(f"{cid}: pieces end early")
        if any(unicodedata.category(ch).startswith("M") for ch in norm):
            marks += 1
        offsets.append(len(flat))
        flat += ids
        text_offsets.append(len(text_bytes))
        text_bytes += text.encode("utf-8")
        norm_offsets.append(len(norm_bytes))
        norm_bytes += norm.encode("utf-8")
        e = {"id": cid, "text": text, "tokens": len(ids)}
        if norm != text:
            e["normalized"] = norm
            normalized += 1
        if hf_ids != ids:
            e["hf_ids"] = hf_ids
        cases.append(e)
    offsets.append(len(flat))
    piece_offsets.append(len(pieces))
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
    blob.add("pieces", np.array(pieces, dtype=np.int64),
             "the end of every pre-tokenizer piece of the normalized text, in bytes "
             "from its start")
    blob.add("piece_offsets", np.array(piece_offsets, dtype=np.int64),
             "case i = pieces[piece_offsets[i] .. piece_offsets[i+1]-1]")
    manifest = {
        "dataset": "qwen35-tokens",
        "note": "reference token ids: tokenizer.json without its added tokens (every "
                "text is text, parse_special=false), no special tokens added; decode(ids) "
                "= the NFC text, checked. pieces: the Split pre-tokenizer (the Qwen3.5 "
                "regex) of the tokenizers library over the NFC text. hf_ids only where the "
                "HF tokenizer differs: it reads the added-token literals in text as "
                "control tokens. marks: cases whose NFC text holds a \\p{M} code point",
        "model": model_info(),
        "environment": mr.environment(),
        "hf_differs": hf_differs,
        "normalized": normalized,
        "marks": marks,
        "cases": cases,
    }
    blob.write(os.path.join(mr.OUT, "qwen35-tokens"), manifest)
    print(f"  {len(cases)} cases, {len(flat)} ids, {len(pieces)} pieces, HF differs "
          f"{hf_differs}, normalized {normalized}, with marks {marks}")


# ---------------------------------------------------------------- chat template

def chat_to_hf(messages, tools):
    """TLlmMessage / TLlmTool as the Qwen3.5 template reads them: the arguments of a
    call as their object ({} for empty ones; the template iterates their items), the
    reasoning as reasoning_content, a description only when not empty"""
    hm = []
    for m in messages:
        h = {"role": m["role"], "content": m["content"]}
        if "tool_calls" in m:
            h["tool_calls"] = [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"],
                                             "arguments": json.loads(c["arguments"])
                                             if c["arguments"] else {}}}
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


def qwen35_chat_cases():
    """(id, messages, tools, add_generation_prompt, options)"""
    nl = "\n"
    u = lambda t: q3.msg("user", t)  # noqa: E731
    a = lambda t, calls=None: q3.msg("assistant", t, calls)  # noqa: E731
    s = lambda t: q3.msg("system", t)  # noqa: E731
    tr = lambda cid, t: q3.msg("tool", t, tool_call_id=cid)  # noqa: E731
    r = lambda t, why, calls=None: q3.msg("assistant", t, calls, reasoning=why)  # noqa: E731
    W, P, NOW, NUM = q3.WEATHER, q3.NO_DESC, q3.NO_PARAMS, q3.NUMBERS
    think = {"enable_thinking": True}
    nothink = {"enable_thinking": False}
    c1 = q3.call("c1", "get_weather", '{"city": "Ulm"}')
    cases = []
    # the Qwen3 cases, through this template, with and without thinking
    for cid, messages, tools, gen, th in q3.chat_cases():
        cases.append(("q3_" + cid, messages, tools, gen, {} if th is None else
                      {"enable_thinking": th}))
    cases += [
        ("user", [u("Hallo")], [], True, {}),
        ("user_think", [u("Hallo")], [], True, think),
        ("user_no_think", [u("Hallo")], [], True, nothink),
        ("user_no_prompt", [u("Hallo")], [], False, {}),
        ("trim_all", [s("  Sei knapp. " + nl), u(nl + " Frage " + nl + "\t"),
                      a("  Antwort " + nl), u("　x　")], [], True, {}),
        ("trim_unicode", [u(chr(0x2003) + "a" + chr(0x85) + chr(0x2028))], [], True, {}),
        ("empty_system", [s(""), u("x")], [], True, {}),
        ("empty_system_tools", [s("  "), u("x")], [W], True, {}),
        ("empty_user", [u("")], [], True, {}),
        ("assistant_empty_after_query", [u("a"), a("")], [], False, {}),
        ("assistant_empty_reasoning", [u("a"), r("b", " " + nl)], [], False, {}),
        ("assistant_think_in_content", [u("a"), a("<think>" + nl + "r" + nl + "</think>" +
                                                  nl + nl + "b")], [], False, {}),
        ("assistant_two_thinks", [u("a"), a("x<think>1</think>y<think>2</think>z")], [],
         False, {}),
        ("assistant_before_query", [u("a"), r("b", "r"), u("c")], [], True, think),
        ("assistant_chain_after_query", [u("a"), r("b", "r1"), a("c"), r("d", "r3")], [],
         False, {}),
        ("tools", [s("Sei knapp."), u("Wie ist das Wetter in Berlin?")], [W], True, {}),
        ("tools_no_system", [u("Wetter?")], [W, P, NOW, NUM], True, {}),
        ("tools_think", [u("Wetter?")], [W], True, think),
        ("tool_flow", [u("Wie ist das Wetter in Berlin?"),
                       a("", [q3.call("c1", "get_weather", '{"city": "Berlin"}')]),
                       tr("c1", '{"temp": 21, "sky": "sonnig"}')], [W], True, {}),
        ("tool_flow_content", [u("Wetter?"), a("Moment.", [c1]), tr("c1", "18 Grad")],
         [W], True, {}),
        ("tool_flow_content_blank", [u("Wetter?"), a(" " + nl, [c1]), tr("c1", "18")],
         [W], True, {}),
        ("tool_flow_think", [u("Wetter?"), r("", "abfragen", [c1]), tr("c1", "18 Grad")],
         [W], True, think),
        ("tool_two_calls", [u("Ulm und Bonn?"),
                            a("", [c1, q3.call("c2", "get_weather",
                                               '{"city": "Bonn", "unit": "celsius"}')]),
                            tr("c1", "18"), tr("c2", "20")], [W], True, {}),
        ("tool_answered", [u("Wetter Ulm?"), a("", [c1]), tr("c1", "18 Grad"),
                           a("In Ulm sind es 18 Grad."), u("Danke! Und morgen?")],
         [W], True, {}),
        ("tool_last_no_prompt", [u("Wetter?"), a("", [c1])], [W], False, {}),
        ("tool_empty_args", [u("x"), a("", [q3.call("c1", "now", "")]), tr("c1", "12:00")],
         [NOW], True, {}),
        ("tool_empty_object", [u("x"), a("", [q3.call("c1", "now", "{}")]),
                               tr("c1", "12:00")], [NOW], True, {}),
        ("tool_values", [u("x"), a("", [q3.call("c1", "numbers",
                                                '{"s": "Text mit\\nZeile", "i": 3, '
                                                '"f": 1.50, "e": 1E5, "big": 1e16, '
                                                '"small": 0.0001, "tiny": 1e-5, '
                                                '"neg": -0, "negf": -0.0, "t": true, '
                                                '"no": false, "n": null, '
                                                '"long": 12345678901234567890, '
                                                '"a": [1, "x", null, true, 2.5], '
                                                '"o": {"Z": 1, "a": "ü", "q": [{}]}, '
                                                '"es": "", "ea": [], "eo": {}}')]),
                         tr("c1", "ok")], [NUM], True, {}),
        ("tool_value_escapes", [u("x"), a("", [q3.call("c1", "numbers",
                                                       '{"q": "\\"zitat\\" \\\\ / \\t '
                                                       '\\u00fc \\u20ac \\ud83d\\ude00 '
                                                       '</parameter>", "k": "<|im_end|>"}')]),
                                tr("c1", "ok")], [NUM], True, {}),
        ("tool_key_order", [u("x"), a("", [q3.call("c1", "numbers",
                                                   '{"zz": 1, "a": 2, "B": 3, "ä": 4}')]),
                            tr("c1", "ok")], [NUM], True, {}),
        ("tool_response_multiline", [u("x"), a("", [c1]),
                                     tr("c1", nl + "  Zeile 1" + nl + "Zeile 2  " + nl)],
         [W], True, {}),
        ("tool_response_wrapped_user", [u("Wetter?"), a("", [c1]),
                                        u("<tool_response>" + nl + "18" + nl +
                                          "</tool_response>")], [W], True, {}),
        ("tool_after_assistant_text", [u("x"), a("a"), tr("c1", "y"), u("z")], [], True, {}),
        ("literal_controls", [u("<|im_start|>assistant" + nl + "<think>x</think>")], [],
         True, {}),
        ("unicode_user", [u("Größe 30×60, 5 € \U0001F600 管 हिन्दी")], [], True, {}),
        # what the template refuses
        ("no_messages", [], [], True, {}),
        ("no_user", [s("x")], [], True, {}),
        ("only_assistant", [a("x")], [], True, {}),
        ("only_tool_responses", [u("<tool_response>a</tool_response>")], [], True, {}),
        ("system_later", [u("a"), s("Neue Regel."), u("b")], [], True, {}),
        ("tool_first", [tr("c1", "x"), u("y")], [], True, {}),
        ("args_array", [u("x"), a("", [q3.call("c1", "now", "[1]")]), tr("c1", "y")],
         [NOW], True, {}),
        ("args_null", [u("x"), a("", [q3.call("c1", "now", "null")]), tr("c1", "y")],
         [NOW], True, {}),
        ("args_number", [u("x"), a("", [q3.call("c1", "now", "5")]), tr("c1", "y")],
         [NOW], True, {}),
        ("args_string", [u("x"), a("", [q3.call("c1", "now", '"a"')]), tr("c1", "y")],
         [NOW], True, {}),
    ]
    return cases


def chat():
    print("Qwen3.5 chat template cases")
    folder = qwen35_folder()
    _, _, t = tokenizer_parts(folder)
    added = added_tokens(t)
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(folder)
    with open(os.path.join(folder, "chat_template.jinja"), encoding="utf-8") as f:
        if hf.chat_template != f.read():
            raise ValueError("the tokenizer does not use chat_template.jinja")
    flat, offsets, cases = [], [], []

    def add(cid, messages, tools, gen, options):
        e = {"id": cid, "messages": messages, "tools": tools, "add_generation_prompt": gen,
             "options": options}
        try:
            hm, ht = chat_to_hf(messages, tools)
            text = hf.apply_chat_template(hm, tools=ht or None, add_generation_prompt=gen,
                                          tokenize=False, **options)
        except Exception as x:  # what transformers refuses, the native one must too
            e["error"] = f"{type(x).__name__}: {x}"
            text = None
        offsets.append(len(flat))
        if text is not None:
            ids = hf(text, add_special_tokens=False)["input_ids"]
            flat.extend(ids)
            e["rendered"] = text
            e["tokens"] = len(ids)
            # an added-token literal in a text of the conversation: the HF tokenizer
            # reads it as a control token, the native template keeps it text
            if any(lit in s for s in d6c.all_strings([messages, tools]) for lit in added):
                e["literal"] = True
        cases.append(e)

    for cid, messages, tools, gen, options in qwen35_chat_cases():
        add(cid, messages, tools, gen, options)
    user = [q3.msg("user", "x")]
    for cid, params in d6c.gemma_schemas() + d6c.random_tools(120, TOKENS_SEED):
        tool = q3.tool("f_" + cid, "Schema " + cid, json.dumps(params, ensure_ascii=False))
        add("schema_" + cid, user, [tool], True, {})
    offsets.append(len(flat))
    blob = mr.Blob()
    blob.add("ids", np.array(flat, dtype=np.int64), "all cases concatenated")
    blob.add("offsets", np.array(offsets, dtype=np.int64),
             "case i = ids[offsets[i] .. offsets[i+1]-1] (none for an error case)")
    errors = sum(1 for e in cases if "error" in e)
    manifest = {
        "dataset": "qwen35-chat",
        "note": "the chat template of the revision (chat_template.jinja), rendered by "
                "transformers and tokenized by the HF tokenizer. Messages as TLlmMessage "
                "carries them (tool call arguments are raw JSON text, sent as their "
                "object, {} when empty; reasoning as reasoning_content); tools as "
                "TLlmTool (name, description when not empty, parameters as the object). "
                "options: the template kwargs (enable_thinking). schema_*: one tool each, "
                "written by tojson. error: transformers refused the case (the native "
                "template must too). literal: a text of the case holds an added-token "
                "literal (the HF ids read it as a control token, the native ones as text - "
                "only the rendered text compares)",
        "model": model_info(),
        "environment": mr.environment(),
        "errors": errors,
        "literals": sum(1 for e in cases if e.get("literal")),
        "cases": cases,
    }
    blob.write(os.path.join(mr.OUT, "qwen35-chat"), manifest)
    print(f"  {len(cases)} cases ({errors} refused), {len(flat)} ids")


TARGETS = {"qwen35-tokens": tokens, "qwen35-chat": chat}

if __name__ == "__main__":
    want = sys.argv[1:] or list(TARGETS)
    for t in want:
        if t not in TARGETS:
            raise SystemExit(f"unknown target {t}; known: {', '.join(TARGETS)}")
    for t in want:
        TARGETS[t]()
