#!/usr/bin/env python3
"""Reference data for phase D slice D6c of the native inference engine: the tokenizer of
Gemma 4 and its chat format.

Targets (all of them when called without arguments):
  gemma4-tokens   tests/data/infer-ref/gemma4-tokens: the tokenizer gate - per case the
                  ids of tokenizer.json without its added tokens (every text is text) and
                  the text decoding gives back where it is not the input (a U+2581 in
                  the input decodes to a space)
  gemma4-chat     tests/data/infer-ref/gemma4-chat: the template gate - conversations and
                  tool schemas through chat_template.jinja of the revision, rendered by
                  transformers, and their ids; 400 generated schemas over the branches of
                  format_parameters; what transformers refuses is marked as an error

It needs the tokenizer files of google/gemma-4-26B-A4B-it in the pinned revision (not
downloaded here: the folder $LAI_GEMMA4_DIR, by default vendor/models/gemma-4-26B-A4B-it,
checked against the pinned hashes); the weights are not needed. gemma-4-31B-it and
gemma-4-12B-it ship the same tokenizer.json and chat_template.jinja (same SHA-256). Same
environment, format and rules as make_reference.py; the cases of make_reference.py,
make_reference_qwen3.py and make_reference_d5c.py are reused.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_reference as mr  # noqa: E402
import make_reference_qwen3 as q3  # noqa: E402
import make_reference_d5c as d5c  # noqa: E402

GEMMA4_ID = "google/gemma-4-26B-A4B-it"
GEMMA4_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
GEMMA4_SHA256 = {
    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    "tokenizer_config.json":
        "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633",
    "chat_template.jinja":
        "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
    "generation_config.json":
        "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
}
TOKENS_SEED = 20261021


def gemma4_folder():
    folder = os.environ.get("LAI_GEMMA4_DIR",
                            os.path.join(mr.ROOT, "vendor", "models", "gemma-4-26B-A4B-it"))
    for name, want in GEMMA4_SHA256.items():
        got = mr.sha256_file(os.path.join(folder, name))
        if got != want:
            raise ValueError(f"{name}: sha256 {got}, pinned {want}")
    return folder


def model_info():
    return {"id": GEMMA4_ID, "revision": GEMMA4_REVISION, "files": GEMMA4_SHA256}


def added_tokens(folder):
    with open(os.path.join(folder, "tokenizer.json"), encoding="utf-8") as f:
        return [a["content"] for a in json.load(f)["added_tokens"]]


def plain_tokenizer(folder):
    """tokenizer.json without its added tokens: every text is text"""
    from tokenizers import Tokenizer
    with open(os.path.join(folder, "tokenizer.json"), encoding="utf-8") as f:
        t = json.load(f)
    t["added_tokens"] = []
    return Tokenizer.from_str(json.dumps(t))


def gemma_designed_cases(added):
    """edge cases of the SentencePiece-style pipeline - (id, text)"""
    c = chr
    nl, cr, tab, sp = c(10), c(13), c(9), c(0x2581)
    cases = [
        # spaces: every one a U+2581, runs merged as far as the vocabulary goes
        ("sp_runs", "a" + "".join(" " * k + "b" for k in range(1, 40))),
        ("sp_lead_long", " " * 70 + "x"), ("sp_trail_long", "x" + " " * 70),
        ("sp_only_long", " " * 100), ("sp_between_punct", "a , b . c ! d"),
        ("sp_literal", "a" + sp + "b " + sp + " " + sp + sp + "c"),
        ("sp_literal_lead", sp * 5 + "x"),
        # other white space stays itself: tabs, newlines and their long runs
        ("nl_runs", "a" + "".join(nl * k + "b" for k in range(1, 35))),
        ("tab_runs", "a" + "".join(tab * k + "b" for k in range(1, 35))),
        ("cr_runs", "a" + "".join((cr + nl) * k + "b" for k in range(1, 10))),
        ("ws_mix_runs", (" " + tab + nl) * 20), ("nbsp_runs", c(0xA0) * 9 + "x"),
        # byte fallback: characters outside the vocabulary
        ("bytes_private", c(0xF0000) + c(0x100000) + c(0xE123)),
        ("bytes_controls", "".join(c(x) for x in range(0, 32)) + c(0x7F)),
        ("bytes_c1", "".join(c(x) for x in range(0x80, 0xA0))),
        ("bytes_cjk_ext", "".join(c(x) for x in range(0x2A700, 0x2A710))),
        ("bytes_glued", "Rohr" + c(0xF0001) + "leitung" + c(0xF0002)),
        ("bytes_spaced", " " + c(0xF0003) + " " + c(0xF0004) + " "),
        ("bytes_noncharacters", c(0xFDD0) + c(0xFFFF) + c(0x1FFFE)),
        ("bytes_surrogate_area", c(0xFFFD) + c(0xFFFC) + c(0xFFFB)),
        # the byte tokens as literals are text
        ("lit_byte_token", "<0x0A><0xFF> <0x41>"),
        # German and code, as the model will see them
        ("de_sentence", "Die Kupferrohre im Heizungskeller müssen bis Freitag "
                        "geprüft werden – bitte mit Druckprotokoll."),
        ("de_compound", "Rohrleitungsbauunternehmensverbandsvorsitzender"),
        ("code_indent", "def f(x):" + nl + "    if x:" + nl + "        return 1" + nl +
                        tab + "pass" + nl),
        ("json_tool", '{"name": "get_weather", "arguments": {"city": "München"}}'),
        ("long_text", ("Der Monteur verlegt " + nl + "  DN 15 Kupferrohr, 3,5 m." + tab) * 120),
        ("long_one_word", "Rohr" * 1500),
    ]
    # the added tokens as text: alone, glued, spaced, in a template-like run
    for i, lit in enumerate(added):
        cases += [(f"lit_{i:02d}", lit), (f"lit_{i:02d}_glued", "a" + lit + "b"),
                  (f"lit_{i:02d}_spaced", " " + lit + " ")]
    cases.append(("lit_turn", "<bos><|turn>user" + nl + "Hallo<turn|>" + nl +
                  "<|turn>model" + nl + "<|channel>thought" + nl + "<channel|>"))
    cases.append(("lit_call", '<|tool_call>call:f{a:<|"|>x<|"|>}<tool_call|>'
                  "<|tool_response>response:f{value:1}<tool_response|>"))
    cases.append(("lit_unused", "<unused0><unused99><start_of_image><end_of_turn>"))
    return cases


def gemma_generated_cases(n, seed):
    """random mixes over what the pipeline treats apart - (id, text)"""
    import random
    rng = random.Random(seed)
    pools = [
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüß",
        " " * 12 + chr(0x2581) * 2,
        chr(9) * 3 + chr(10) * 4 + chr(13) + chr(0xA0) + chr(0x3000),
        "0123456789.,;:-_/()[]{}+*=%&#@!?<>|'" + chr(34) + chr(92),
        "".join(chr(x) for x in [0xF0000, 0x100001, 0xE000, 0x2A700, 0x1, 0x7F, 0x85]),
        "".join(chr(x) for x in [0x1F600, 0x1F44D, 0x200D, 0xFE0F, 0x4E00, 0x7BA1, 0xAC00]),
        "".join(chr(x) for x in range(0x0410, 0x0450, 3)) + "".join(
            chr(x) for x in range(0x0300, 0x0308)),
    ]
    weights = [34, 18, 10, 14, 6, 10, 8]
    cases, seen = [], set()
    while len(cases) < n:
        length = rng.randint(1, 64)
        text = "".join(rng.choice(rng.choices(pools, weights)[0]) for _ in range(length))
        if text in seen:
            continue
        seen.add(text)
        cases.append((f"ggen_{len(cases):04d}", text))
    return cases


def tokens():
    print("Gemma 4 tokenizer cases")
    folder = gemma4_folder()
    added = added_tokens(folder)
    plain = plain_tokenizer(folder)
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
    texts += [("gemma_" + cid, text) for cid, text in gemma_designed_cases(added)]
    texts += mr.generated_cases(400, mr.TINY_SEED)
    texts += q3.qwen_generated_cases(400, q3.TINY_SEED)
    texts += d5c.o200k_generated_cases(400, d5c.TOKENS_SEED)
    texts += gemma_generated_cases(1200, TOKENS_SEED)
    ids_seen = [cid for cid, _ in texts]
    if len(set(ids_seen)) != len(ids_seen):
        raise ValueError(f"duplicated case id {[x for x in ids_seen if ids_seen.count(x) > 1]}")
    flat, offsets, cases = [], [], []
    text_bytes, text_offsets = bytearray(), []
    hf_differs = decoded_differs = byte_fallback = 0
    first_byte = plain.token_to_id("<0x00>")
    for cid, text in texts:
        ids = plain.encode(text, add_special_tokens=False).ids
        hf_ids = hf(text, add_special_tokens=False)["input_ids"]
        if hf_ids != ids:
            if not any(a in text for a in added):
                raise ValueError(f"{cid}: HF differs without an added-token literal")
            hf_differs += 1
        decoded = plain.decode(ids, skip_special_tokens=False)
        want = text.replace(chr(0x2581), " ")
        if decoded != want:
            raise ValueError(f"{cid}: decode gives {decoded!r}")
        if any(first_byte <= i < first_byte + 256 for i in ids):
            byte_fallback += 1
        offsets.append(len(flat))
        flat += ids
        text_offsets.append(len(text_bytes))
        text_bytes += text.encode("utf-8")
        e = {"id": cid, "text": text, "tokens": len(ids)}
        if decoded != text:
            e["decoded"] = decoded
            decoded_differs += 1
        if hf_ids != ids:
            e["hf_ids"] = hf_ids
        cases.append(e)
    offsets.append(len(flat))
    text_offsets.append(len(text_bytes))
    blob = mr.Blob()
    blob.add("texts", np.frombuffer(bytes(text_bytes), dtype=np.uint8),
             "UTF-8 of all case texts concatenated")
    blob.add("text_offsets", np.array(text_offsets, dtype=np.int64),
             "case i = texts[text_offsets[i] .. text_offsets[i+1]-1]")
    blob.add("ids", np.array(flat, dtype=np.int64), "all cases concatenated")
    blob.add("offsets", np.array(offsets, dtype=np.int64),
             "case i = ids[offsets[i] .. offsets[i+1]-1]")
    manifest = {
        "dataset": "gemma4-tokens",
        "note": "reference token ids: tokenizer.json without its added tokens (every "
                "text is text, parse_special=false), no special tokens added; decode(ids) "
                "is the text with U+2581 as a space, checked - 'decoded' where that is "
                "not the text. hf_ids only where the HF tokenizer differs: it reads the "
                "added-token literals in text as control tokens. byte_fallback: cases "
                "with at least one <0xNN> token",
        "model": model_info(),
        "environment": mr.environment(),
        "hf_differs": hf_differs,
        "decoded_differs": decoded_differs,
        "byte_fallback": byte_fallback,
        "cases": cases,
    }
    blob.write(os.path.join(mr.OUT, "gemma4-tokens"), manifest)
    print(f"  {len(cases)} cases, {len(flat)} ids, HF differs {hf_differs}, "
          f"decoded differs {decoded_differs}, byte fallback in {byte_fallback}")


# ---------------------------------------------------------------- chat template

def chat_to_hf(messages, tools):
    """TLlmMessage / TLlmTool as the Gemma 4 template reads them: arguments as an
    object (None for empty ones), a tool_call_id on every tool message, a name where
    TLlmMessage.Name is set, the reasoning as "reasoning", a description always"""
    hm = []
    for m in messages:
        h = {"role": m["role"], "content": m["content"]}
        if "tool_calls" in m:
            h["tool_calls"] = [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"],
                                             "arguments": json.loads(c["arguments"])
                                             if c["arguments"] else None}}
                               for c in m["tool_calls"]]
        if m["role"] == "tool":
            h["tool_call_id"] = m.get("tool_call_id", "")
        if m.get("name"):
            h["name"] = m["name"]
        if "reasoning" in m:
            h["reasoning"] = m["reasoning"]
        hm.append(h)
    ht = []
    for t in tools:
        f = {"name": t["name"], "description": t["description"]}
        if t["parameters"]:
            f["parameters"] = json.loads(t["parameters"])
        ht.append({"type": "function", "function": f})
    return hm, ht


def gemma_chat_cases():
    """(id, messages, tools, add_generation_prompt, options)"""
    nl = "\n"
    u = lambda t: q3.msg("user", t)  # noqa: E731
    a = lambda t, calls=None: q3.msg("assistant", t, calls)  # noqa: E731
    s = lambda t: q3.msg("system", t)  # noqa: E731
    tr = lambda cid, t: q3.msg("tool", t, tool_call_id=cid)  # noqa: E731
    r = lambda t, why, calls=None: q3.msg("assistant", t, calls, reasoning=why)  # noqa: E731
    W, P, NOW, NUM = q3.WEATHER, q3.NO_DESC, q3.NO_PARAMS, q3.NUMBERS
    think = {"enable_thinking": True}
    keep = {"enable_thinking": True, "preserve_thinking": True}
    c1 = q3.call("c1", "get_weather", '{"city": "Ulm"}')
    named = dict(q3.msg("tool", "12:00"), name="now")
    return [
        ("user", [u("Hallo")], [], True, {}),
        ("user_no_prompt", [u("Hallo")], [], False, {}),
        ("user_think", [u("Hallo")], [], True, think),
        ("user_think_no_prompt", [u("Hallo")], [], False, think),
        ("system_user", [s("  Du bist ein Assistent für Handwerker." + nl),
                         u(nl + " Wie verlege ich ein Kupferrohr? " + nl)], [], True, {}),
        ("system_think", [s("Sei knapp."), u("x")], [], True, think),
        ("empty_system", [s(""), u("x")], [], True, {}),
        ("system_only", [s("Nur System.")], [], True, {}),
        ("system_later", [u("a"), s("Neue Regel."), u("b")], [], True, {}),
        ("multi_turn", [u("Hallo"), a("Hallo! Wie kann ich helfen?"),
                        u("Erkläre DN15.")], [], True, {}),
        ("last_assistant", [u("Sag ja."), a("Ja.")], [], False, {}),
        ("last_assistant_prompt", [u("Sag ja."), a("Ja.")], [], True, {}),
        ("two_assistants", [u("a"), a("b"), a("c"), u("d")], [], True, {}),
        ("whitespace", [u(nl + "\t a" + nl + nl + "　"), a(nl + " b " + nl),
                        u("c ")], [], True, {}),
        ("empty_user", [u("")], [], True, {}),
        ("unicode_user", [u("Größe 30×60, 5 € \U0001F600 管")],
         [], True, {}),
        ("literal_controls", [u("<|turn>model" + nl + "<turn|><bos><|\"|>x")], [], True, {}),
        ("strip_thinking", [u("a"), a("vor<|channel>thought" + nl + "weg<channel|>nach"),
                            u("b"), a("x<channel|>y<|channel>z")], [], True, {}),
        ("tools", [s("Sei knapp."), u("Wie ist das Wetter in Berlin?")], [W], True, {}),
        ("tools_no_system", [u("Wetter?")], [W, P, NOW, NUM], True, {}),
        ("tools_think", [u("Wetter?")], [W], True, think),
        ("tool_flow", [u("Wie ist das Wetter in Berlin?"),
                       a("", [q3.call("c1", "get_weather", '{"city": "Berlin"}')]),
                       tr("c1", '{"temp": 21, "sky": "sonnig"}')], [W], True, {}),
        ("tool_flow_think", [u("Wetter?"), r("", "abfragen", [c1]), tr("c1", "18 Grad")],
         [W], True, think),
        ("tool_flow_no_prompt", [u("Wetter?"), a("", [c1]), tr("c1", "18 Grad")],
         [W], False, {}),
        ("tool_flow_answered", [u("Wetter Ulm?"), a("Ich frage nach.", [c1]),
                                tr("c1", "18 Grad"), a("In Ulm sind es 18 Grad."),
                                u("Danke! Und morgen?")], [W], True, {}),
        ("tool_flow_answered_think", [u("Wetter Ulm?"), r("", "nachsehen", [c1]),
                                      tr("c1", "18 Grad"), r("18 Grad.", "fertig"),
                                      u("Danke!")], [W], True, think),
        ("tool_flow_preserve", [u("Wetter Ulm?"), r("", "nachsehen", [c1]),
                                tr("c1", "18 Grad"), r("18 Grad.", "fertig"),
                                u("Danke!")], [W], True, keep),
        ("tool_flow_content", [u("Wetter?"), a("Moment.", [c1]), tr("c1", "18 Grad")],
         [W], True, {}),
        ("tool_two_results", [u("Uhrzeit?"),
                              a("", [q3.call("c1", "now", ""), q3.call("c2", "now", "")]),
                              tr("c1", "12:00"), tr("c2", "12:01")], [NOW], True, {}),
        ("tool_name_fallback", [u("x"), a("", [q3.call("c1", "now", "")]),
                                tr("other", "?"), named], [NOW], True, {}),
        ("tool_chain", [u("Plane."), a("", [q3.call("c1", "now", "{}")]),
                        tr("c1", "12:00"),
                        a("", [q3.call("c2", "get_weather",
                                       '{"city": "München", "unit": "celsius"}')]),
                        tr("c2", 'Sonne "hell"' + nl + "\tkalt " + chr(92) + " ü")],
         [NOW, W], True, {}),
        ("tool_last_no_prompt", [u("Wetter?"), a("", [c1])], [W], False, {}),
        ("tool_last_prompt", [u("Wetter?"), a("", [c1])], [W], True, {}),
        ("tool_numbers", [u("x"), a("", [q3.call("c1", "numbers",
                                                 '{"b": [1E5, -0, true, null, 0.5, '
                                                 '{"Z": 1, "a": "s"}], "A": 1.50}')]),
                          tr("c1", "ok")], [NUM], True, {}),
        ("tool_null_args", [u("x"), a("", [q3.call("c1", "now", "null")]),
                            tr("c1", "ok")], [NOW], True, {}),
        ("thinking_middle", [u("a"), r("b", "r"), u("c")], [], True, think),
        ("thinking_last", [u("a"), r("b", nl + "r" + nl)], [], False, think),
        ("thinking_last_prompt", [u("a"), r("b", "r")], [], True, {}),
    ]


def gemma_schemas():
    """designed tool schemas over the branches of format_parameters - (id, schema)"""
    obj = lambda props, req=None, **kw: dict(  # noqa: E731
        {"type": "object", "properties": props}, **({"required": req} if req else {}), **kw)
    return [
        ("scalars", obj({"s": {"type": "string"}, "n": {"type": "number"},
                         "i": {"type": "integer"}, "b": {"type": "boolean"},
                         "x": {}, "nul": {"type": "null"}}, ["s", "b"])),
        ("described", obj({"city": {"type": "string", "description": "Stadt"},
                           "empty": {"type": "string", "description": ""},
                           "num": {"type": "string", "description": 5}}, ["city"])),
        ("enum", obj({"unit": {"type": "string", "enum": ["c", "f"], "default": "c"},
                      "mixed": {"type": "string", "enum": ["a", 1, 2.5, True, None]},
                      "empty_enum": {"type": "string", "enum": []},
                      "int_enum": {"type": "integer", "enum": [1, 2]}})),
        ("nullable", obj({"s": {"type": "string", "nullable": True},
                          "a": {"type": "array", "items": {"type": "string"}, "nullable": True},
                          "n": {"type": "number", "nullable": 0}})),
        ("arrays", obj({"s": {"type": "array", "items": {"type": "string"}},
                        "none": {"type": "array"}, "empty": {"type": "array", "items": {}},
                        "nested": {"type": "array", "items": {"type": "array",
                                                              "items": {"type": "string"}}},
                        "objs": {"type": "array", "items": obj({"k": {"type": "string"},
                                                                "J": {"type": "number"}},
                                                               ["k"])},
                        "tuple": {"type": "array", "items": [{"type": "string"}]},
                        "union": {"type": "array", "items": {"type": ["object", "null", 1]}},
                        "extra": {"type": "array", "items": {"description": "d",
                                                             "enum": ["x"], "nullable": None,
                                                             "minimum": 1.5,
                                                             "props": {"Z": 1, "a": None}}},
                        "props_list": {"type": "array", "items": {"properties": [1]}}})),
        ("objects", obj({"o": obj({"a": {"type": "string"}, "b": {"type": "number"}}, ["a"]),
                         "e": {"type": "object"},
                         "free": {"type": "object", "additionalProperties": {"type": "string"},
                                  "description": "frei", "minProperties": 1,
                                  "required": ["x"]},
                         "null_props": {"type": "object", "properties": None},
                         "deep": obj({"x": obj({"y": {}})})})),
        ("case_order", obj({"b": {"type": "string"}, "B_": {"type": "string"},
                            "a_b": {"type": "string"}, "aB": {"type": "string"},
                            "A": {"type": "string"}, "a": {"type": "string"},
                            "Ärger": {"type": "string"}, "ä": {"type": "string"},
                            "zz": {"type": "string"}})),
        ("type_case", obj({"s": {"type": "String"}, "o": {"type": "Object",
                                                         "properties": {"q": {}}},
                           "n": {"type": None}, "num": {"type": 3}})),
        ("unicode", obj({"größe": {"type": "string",
                                             "description": "€ \U0001F600"}})),
        ("no_properties", {"type": "object"}),
        ("empty_properties", {"type": "object", "properties": {}}),
        ("not_required", obj({"a": {"type": "string"}}, [])),
        ("no_type", {"properties": {"a": {"type": "string"}}, "required": ["a"]}),
        ("required_scalars", obj({"a": {"type": "string"}}, ["a", 1, True, None])),
    ]


def random_schema(rng, depth=0):
    """a property schema over the branches of format_parameters"""
    words = ["a", "b", "City", "näme", "x_y", "Long Name", "_k"]
    kinds = ["string", "number", "integer", "boolean", "object", "array", "null", "none",
             "enum", "upper"]
    k = rng.choice(kinds if depth < 3 else kinds[:4] + ["null", "none"])
    sp = {}
    if k == "enum":
        sp["type"] = "string"
        sp["enum"] = [rng.choice(["c", "f", "ü", 1, 2.5, True, None, ""])
                      for _ in range(rng.randint(0, 3))]
    elif k == "array":
        sp["type"] = "array"
        x = rng.random()
        if x < 0.6:
            sp["items"] = random_schema(rng, depth + 1)
            if rng.random() < 0.3:
                sp["items"]["type"] = rng.sample(["string", "number", "null"],
                                                 rng.randint(0, 2))
        elif x < 0.7:
            sp["items"] = {}
        elif x < 0.8:
            sp["items"] = [random_schema(rng, depth + 1)]
    elif k == "object":
        sp["type"] = rng.choice(["object", "OBJECT"])
        if rng.random() < 0.8:
            sp["properties"] = {rng.choice(words) + str(i): random_schema(rng, depth + 1)
                                for i in range(rng.randint(0, 3))}
        elif rng.random() < 0.5:
            sp["additionalProperties"] = rng.choice([False, {"type": "string"}])
        if rng.random() < 0.5 and sp.get("properties"):
            sp["required"] = rng.sample(list(sp["properties"]),
                                        rng.randint(0, len(sp["properties"])))
    elif k == "upper":
        sp["type"] = rng.choice(["String", "NUMBER", "Array"])
    elif k != "none":
        sp["type"] = k
    if rng.random() < 0.2:
        sp["nullable"] = rng.choice([True, False, 0, 1])
    if rng.random() < 0.3:
        sp["description"] = rng.choice(["Ein Wert", "", "a // b\nc", 5, "<|\"|>"])
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
        if rng.random() < 0.1:
            del params["type"]
        out.append((f"random_{i:03d}", params))
    return out


def all_strings(value):
    """every string in a case: texts, keys, and those of the JSON texts in it"""
    if isinstance(value, str):
        yield value
        try:
            inner = json.loads(value)
        except ValueError:
            return
        if not isinstance(inner, str):
            yield from all_strings(inner)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from all_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from all_strings(v)


def chat():
    print("Gemma 4 chat template cases")
    folder = gemma4_folder()
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
            ids = hf(text, add_special_tokens=False)["input_ids"]
            flat.extend(ids)
            e["rendered"] = text
            e["tokens"] = len(ids)
            # an added-token literal in a text of the conversation: the HF tokenizer
            # reads it as a control token, the native template keeps it text
            if any(lit in t for t in all_strings([messages, tools]) for lit in added):
                e["literal"] = True
        cases.append(e)

    added = added_tokens(folder)
    for cid, messages, tools, gen, options in gemma_chat_cases():
        add(cid, messages, tools, gen, options)
    user = [q3.msg("user", "x")]
    for cid, params in gemma_schemas() + random_tools(400, TOKENS_SEED):
        tool = q3.tool("f_" + cid, "Schema " + cid, json.dumps(params, ensure_ascii=False))
        add("schema_" + cid, user, [tool], True, {})
    offsets.append(len(flat))
    blob = mr.Blob()
    blob.add("ids", np.array(flat, dtype=np.int64), "all cases concatenated")
    blob.add("offsets", np.array(offsets, dtype=np.int64),
             "case i = ids[offsets[i] .. offsets[i+1]-1] (none for an error case)")
    errors = sum(1 for e in cases if "error" in e)
    manifest = {
        "dataset": "gemma4-chat",
        "note": "the chat template of the revision (chat_template.jinja), rendered by "
                "transformers and tokenized by the HF tokenizer. Messages as TLlmMessage "
                "carries them (tool call arguments are raw JSON text, sent as their "
                "object, None when empty; every tool message with its tool_call_id; name "
                "where set; reasoning as reasoning); tools as TLlmTool (name, description "
                "- always sent, parameters as the object). options: the template kwargs "
                "(enable_thinking, preserve_thinking). schema_*: one tool each, its "
                "parameters through format_parameters; random_*: generated over its "
                "branches. error: transformers refused the case (the native template "
                "must too). literal: a text of the case holds an added-token literal "
                "(the HF ids read it as a control token, the native ones as text - "
                "only the rendered text compares)",
        "model": model_info(),
        "environment": mr.environment(),
        "errors": errors,
        "literals": sum(1 for e in cases if e.get("literal")),
        "cases": cases,
    }
    blob.write(os.path.join(mr.OUT, "gemma4-chat"), manifest)
    print(f"  {len(cases)} cases ({errors} refused), {len(flat)} ids")


TARGETS = {"gemma4-tokens": tokens, "gemma4-chat": chat}

if __name__ == "__main__":
    want = sys.argv[1:] or list(TARGETS)
    for t in want:
        if t not in TARGETS:
            raise SystemExit(f"unknown target {t}; known: {', '.join(TARGETS)}")
    for t in want:
        TARGETS[t]()
