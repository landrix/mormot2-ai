#!/usr/bin/env python3
"""Reference data for phase D slice D7h of the native inference engine: the chat format
of Qwen3.8 - its template and how its response_template reads an answer.

Targets (all of them when called without arguments):
  qwen38-chat     tests/data/infer-ref/qwen38-chat: the template gate - conversations and
                  tool schemas through chat_template.jinja of the revision, rendered by
                  transformers, and their ids; what transformers refuses is marked as an
                  error. The cases of qwen35-chat (make_reference_d7c.py) with the template
                  kwargs of Qwen3.8 on top (reasoning_effort, preserve_thinking)
  qwen38-parse    tests/data/infer-ref/qwen38-parse: the parser gate - per case a prompt
                  (rendered as above) and a model answer, both as ids, and what
                  transformers' parse_response makes of the answer with the
                  response_template of tokenizer_config.json and the tools of the case

It needs the tokenizer files of Qwen/Qwen3.8-27B in the pinned revision (not downloaded
here: the folder $LAI_QWEN38_DIR, by default vendor/models/qwen3.8-27b, checked against
the pinned hashes); the weights are not needed. Same environment, format and rules as
make_reference.py. The tokenizer of Qwen3.8 is that of Qwen3.5 with seven added audio
tokens at the end (checked here against $LAI_QWEN35_DIR): the native tests encode with
the Qwen3.5 tokenizer.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_reference as mr  # noqa: E402
import make_reference_qwen3 as q3  # noqa: E402
import make_reference_d6c as d6c  # noqa: E402
import make_reference_d7c as d7c  # noqa: E402

QWEN38_ID = "Qwen/Qwen3.8-27B"
QWEN38_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
QWEN38_SHA256 = {
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json":
        "207c03c41a06a9c3b0cc527d8a56115159c1eac6b59a4ffae85b646e773d074e",
    "chat_template.jinja":
        "af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375",
}
# the added tokens of Qwen3.8 that Qwen3.5 lacks (ids after all of Qwen3.5's)
QWEN38_EXTRA_TOKENS = ["<|audio_start|>", "<|audio_end|>", "<tts_pad>", "<tts_text_bos>",
                       "<tts_text_eod>", "<tts_text_bos_single>", "<|audio_pad|>"]
SCHEMA_SEED = 20261025  # the schemas of qwen35-chat


def qwen38_folder():
    folder = os.environ.get("LAI_QWEN38_DIR",
                            os.path.join(mr.ROOT, "vendor", "models", "qwen3.8-27b"))
    for name, want in QWEN38_SHA256.items():
        got = mr.sha256_file(os.path.join(folder, name))
        if got != want:
            raise ValueError(f"{name}: sha256 {got}, pinned {want}")
    return folder


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def model_info():
    return {"id": QWEN38_ID, "revision": QWEN38_REVISION, "files": QWEN38_SHA256}


def check_tokenizer(folder):
    """tokenizer.json of Qwen3.8 = that of Qwen3.5 plus QWEN38_EXTRA_TOKENS, appended"""
    with open(os.path.join(folder, "tokenizer.json"), encoding="utf-8") as f:
        t8 = json.load(f)
    with open(os.path.join(d7c.qwen35_folder(), "tokenizer.json"), encoding="utf-8") as f:
        t5 = json.load(f)
    for k in t5:
        if k != "added_tokens" and t5[k] != t8[k]:
            raise ValueError(f"tokenizer.json: {k} differs from Qwen3.5")
    n = len(t5["added_tokens"])
    if t8["added_tokens"][:n] != t5["added_tokens"]:
        raise ValueError("tokenizer.json: the added tokens of Qwen3.5 differ")
    extra = t8["added_tokens"][n:]
    if [a["content"] for a in extra] != QWEN38_EXTRA_TOKENS:
        raise ValueError(f"tokenizer.json: unexpected added tokens {extra}")
    last = max(a["id"] for a in t5["added_tokens"])
    if [a["id"] for a in extra] != list(range(last + 1, last + 1 + len(extra))):
        raise ValueError("tokenizer.json: the extra tokens are not appended")
    return t8


def hf_tokenizer(folder):
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(folder)
    with open(os.path.join(folder, "chat_template.jinja"), encoding="utf-8") as f:
        if hf.chat_template != f.read():
            raise ValueError("the tokenizer does not use chat_template.jinja")
    if getattr(hf, "response_template", None) is None:
        raise ValueError("the tokenizer has no response_template")
    return hf


# ---------------------------------------------------------------- chat template

def qwen38_chat_cases():
    """(id, messages, tools, add_generation_prompt, options): those of Qwen3.5, then
    the options and rules of Qwen3.8"""
    nl = "\n"
    u = lambda t: q3.msg("user", t)  # noqa: E731
    a = lambda t, calls=None: q3.msg("assistant", t, calls)  # noqa: E731
    s = lambda t: q3.msg("system", t)  # noqa: E731
    tr = lambda cid, t: q3.msg("tool", t, tool_call_id=cid)  # noqa: E731
    r = lambda t, why, calls=None: q3.msg("assistant", t, calls, reasoning=why)  # noqa: E731
    W, NOW, NUM = q3.WEATHER, q3.NO_PARAMS, q3.NUMBERS
    c1 = q3.call("c1", "get_weather", '{"city": "Ulm"}')
    off = {"enable_thinking": False}
    cases = [("q35_" + cid, m, t, g, o) for cid, m, t, g, o in d7c.qwen35_chat_cases()]
    for effort in ["xhigh", "medium", "low"]:
        for th in [None, True, False]:
            o = {"reasoning_effort": effort}
            if th is not None:
                o["enable_thinking"] = th
            tag = {None: "", True: "_on", False: "_off"}[th]
            cases += [
                (f"effort_{effort}{tag}", [u("Hallo")], [], True, o),
                (f"effort_{effort}{tag}_system", [s(" Sei knapp. "), u("Hallo")], [], True, o),
                (f"effort_{effort}{tag}_empty_system", [s(" "), u("Hallo")], [], True, o),
                (f"effort_{effort}{tag}_tools", [s("Sei knapp."), u("Wetter?")], [W], True,
                 o),
            ]
    history = [u("a"), r("b", "r1"), u("c"), r("d", " r2 " + nl), a("e"),
               a("", [c1]), tr("c1", "18"), r("f", "r4")]
    cases += [
        ("effort_high", [u("Hallo")], [], True, {"reasoning_effort": "high"}),
        ("effort_high_off", [u("Hallo")], [], True,
         {"reasoning_effort": "high", "enable_thinking": False}),
        ("effort_upper", [u("Hallo")], [], True, {"reasoning_effort": "XHIGH"}),
        ("no_system_off", [u("Hallo")], [], True, off),
        ("empty_system_off", [s(""), u("Hallo")], [], True, off),
        ("empty_system_tools_off", [s(" "), u("Hallo")], [W], True, off),
        ("preserve_default", history, [W], False, {}),
        ("preserve_true", history, [W], False, {"preserve_thinking": True}),
        ("preserve_false", history, [W], False, {"preserve_thinking": False}),
        ("preserve_false_prompt", history + [u("g")], [W], True,
         {"preserve_thinking": False}),
        ("preserve_false_off", history + [u("g")], [W], True,
         {"preserve_thinking": False, "enable_thinking": False}),
        ("think_in_content", [u("a"), a("<think>" + nl + "r" + nl + "</think>" + nl + nl +
                                        "b"), u("c")], [], True, {}),
        ("think_in_content_reasoning", [u("a"), r("<think>x</think>b", "r"), u("c")], [],
         True, {}),
        ("args_empty_string", [u("x"), a("", [q3.call("c1", "now", '""')]), tr("c1", "y")],
         [NOW], True, {}),
        ("args_values", [u("x"), a("", [q3.call("c1", "numbers",
                                                '{"t": true, "no": false, "n": null, '
                                                '"i": -3, "f": 100000.0, "e": 1E5, '
                                                '"big": 1e16, "s": "True", "sn": "null", '
                                                '"ns": "5", "a": [true, null], '
                                                '"o": {"k": false}}')]),
                         tr("c1", "ok")], [NUM], True, {}),
        ("args_whitespace", [u("x"), a("", [q3.call("c1", "numbers",
                                                     '{"s": "  a  ", "m": "\\n x \\n"}')]),
                             tr("c1", "ok")], [NUM], True, {}),
    ]
    return cases


def render(hf, messages, tools, gen, options):
    hm, ht = d7c.chat_to_hf(messages, tools)
    return hf.apply_chat_template(hm, tools=ht or None, add_generation_prompt=gen,
                                  tokenize=False, **options)


def chat():
    print("Qwen3.8 chat template cases")
    folder = qwen38_folder()
    t = check_tokenizer(folder)
    added = d7c.added_tokens(t)
    hf = hf_tokenizer(folder)
    flat, offsets, cases = [], [], []

    def add(cid, messages, tools, gen, options):
        e = {"id": cid, "messages": messages, "tools": tools, "add_generation_prompt": gen,
             "options": options}
        try:
            text = render(hf, messages, tools, gen, options)
        except Exception as x:  # what transformers refuses, the native one must too
            e["error"] = f"{type(x).__name__}: {x}"
            text = None
        offsets.append(len(flat))
        if text is not None:
            ids = hf(text, add_special_tokens=False)["input_ids"]
            flat.extend(ids)
            e["rendered"] = text
            e["tokens"] = len(ids)
            if any(lit in s for s in d6c.all_strings([messages, tools]) for lit in added):
                e["literal"] = True
        cases.append(e)

    for cid, messages, tools, gen, options in qwen38_chat_cases():
        add(cid, messages, tools, gen, options)
    user = [q3.msg("user", "x")]
    for cid, params in d6c.gemma_schemas() + d6c.random_tools(120, SCHEMA_SEED):
        tool = q3.tool("f_" + cid, "Schema " + cid, json.dumps(params, ensure_ascii=False))
        add("schema_" + cid, user, [tool], True, {})
    offsets.append(len(flat))
    if max(flat) >= 248070:
        raise ValueError("an id of the extra tokens: the Qwen3.5 tokenizer would differ")
    blob = mr.Blob()
    blob.add("ids", np.array(flat, dtype=np.int64), "all cases concatenated")
    blob.add("offsets", np.array(offsets, dtype=np.int64),
             "case i = ids[offsets[i] .. offsets[i+1]-1] (none for an error case)")
    errors = sum(1 for e in cases if "error" in e)
    manifest = {
        "dataset": "qwen38-chat",
        "note": "the chat template of the revision (chat_template.jinja), rendered by "
                "transformers and tokenized by the HF tokenizer (that of Qwen3.5 plus "
                "seven added audio tokens, none of them in an id here). Messages as "
                "TLlmMessage carries them (tool call arguments are raw JSON text, sent as "
                "their value, {} when empty; reasoning as reasoning_content); tools as "
                "TLlmTool. options: the template kwargs (enable_thinking, "
                "reasoning_effort, preserve_thinking) - one that is missing is undefined "
                "for the template (thinking on, effort xhigh, thinking preserved). q35_*: "
                "the cases of qwen35-chat. schema_*: one tool each. error: transformers "
                "refused the case (the native template must too). literal: a text of the "
                "case holds an added-token literal (only the rendered text compares). "
                "chat_template, qwen35_chat_template: chat_template.jinja of Qwen3.8 "
                "and of Qwen3.5 (make_reference_d7c.py), as the files carry them",
        "model": model_info(),
        "environment": mr.environment(),
        "errors": errors,
        "literals": sum(1 for e in cases if e.get("literal")),
        # the templates the client tells apart (TChatTemplate.Recognizes)
        "chat_template": hf.chat_template,
        "qwen35_chat_template": read_text(os.path.join(d7c.qwen35_folder(),
                                                       "chat_template.jinja")),
        "cases": cases,
    }
    blob.write(os.path.join(mr.OUT, "qwen38-chat"), manifest)
    print(f"  {len(cases)} cases ({errors} refused), {len(flat)} ids")


# ---------------------------------------------------------------- response parsing

TYPED = q3.tool("typed", "Typen", json.dumps({
    "type": "object", "properties": {
        "i": {"type": "integer"}, "n": {"type": "number"}, "b": {"type": "boolean"},
        "o": {"type": "object"}, "a": {"type": "array"}, "s": {"type": "string"},
        "ni": {"type": ["integer", "null"]}, "any": {"anyOf": [{"type": "integer"},
                                                               {"type": "string"}]},
        "u": {}}}))


def qwen38_parse_cases():
    """(id, messages, tools, options, answer): the prompt is the conversation with the
    generation prompt; answer is what the model writes, up to its stop token"""
    nl = "\n"
    u = [q3.msg("user", "Wie ist das Wetter in Berlin?")]
    W, T = q3.WEATHER, TYPED
    off = {"enable_thinking": False}
    E = "<|im_end|>"

    def call(name, *params, close=True):
        body = "<tool_call>" + nl + "<function=" + name + ">" + nl
        for k, v in params:
            body += "<parameter=" + k + ">" + nl + v + nl + "</parameter>" + nl
        body += "</function>" + nl
        if close:
            body += "</tool_call>"
        return body

    berlin = call("get_weather", ("city", "Berlin"))
    think = "Ich frage das Wetter ab." + nl + "</think>" + nl + nl
    return [
        ("answer_off", u, [], off, "Es ist sonnig." + E),
        ("answer_off_trailing", u, [], off, "Es ist sonnig.  " + nl + E),
        ("answer_think", u, [], {}, "Kurz nachdenken." + nl + "</think>" + nl + nl +
         "Es ist sonnig." + E),
        ("answer_think_empty", u, [], {}, nl + "</think>" + nl + nl + "Sonnig." + E),
        ("answer_think_cut", u, [], {}, "Ich denke noch"),
        ("answer_no_stop", u, [], off, "Es ist"),
        ("call_off", u, [W], off, berlin + E),
        ("call_think", u, [W], {}, think + berlin + E),
        ("call_content", u, [W], off, "Ich schaue nach." + nl + nl + berlin + E),
        ("call_two", u, [W], off, berlin + nl + call("get_weather", ("city", "Bonn"),
                                                   ("unit", "celsius")) + E),
        ("call_no_think_end", u, [W], {}, "Ich frage das Wetter ab." + nl + nl + berlin + E),
        ("call_no_call_end", u, [W], off, call("get_weather", ("city", "Ulm"), close=False) +
         call("get_weather", ("city", "Bonn")) + E),
        ("call_multiline", u, [W], off, call("get_weather", ("city", "Zeile 1" + nl +
                                                             "Zeile 2")) + E),
        ("call_spaces", u, [W], off, call("get_weather", ("city", "  Berlin  ")) + E),
        ("call_no_args", u, [q3.NO_PARAMS], off, call("now") + E),
        ("call_unknown_tool", u, [W], off, call("get_time", ("zone", "UTC")) + E),
        ("call_unknown_key", u, [W], off, call("get_weather", ("town", "Berlin")) + E),
        ("call_no_tools", u, [], off, berlin + E),
        ("typed", u, [T], off, call("typed", ("i", "42"), ("n", "2.5"), ("b", "true"),
                                    ("o", '{"k": [1, 2]}'), ("a", "[1, \"x\"]"),
                                    ("s", "007"), ("ni", "null"), ("any", "7"),
                                    ("u", "frei")) + E),
        ("typed_whitespace", u, [T], off, call("typed", ("i", " 42 "), ("b", " True "),
                                               ("n", " -0.5 ")) + E),
        ("typed_number_int", u, [T], off, call("typed", ("n", "3")) + E),
        ("typed_string_null", u, [T], off, call("typed", ("s", "null")) + E),
        ("typed_object_bad", u, [T], off, call("typed", ("o", "{kein json}")) + E),
        ("typed_integer_bad", u, [T], off, call("typed", ("i", "zwölf")) + E),
        ("typed_unicode", u, [T], off, call("typed", ("s", "Größe 30×60 € \U0001F600")) + E),
        ("text_tool_call_tag", u, [W], off, "Das Tag <tool_call> ist hier Text." + E),
        # a value without </parameter>: it ends before </function> or the next parameter
        ("open_end_function", u, [W], off, "<tool_call>" + nl + "<function=get_weather>" +
         nl + "<parameter=city>" + nl + "Berlin" + nl + "</function>" + nl +
         "</tool_call>" + E),
        ("open_end_next", u, [W], off, "<tool_call>" + nl + "<function=get_weather>" + nl +
         "<parameter=city>" + nl + "Berlin" + nl + "<parameter=unit>" + nl + "celsius" +
         nl + "</parameter>" + nl + "</function>" + nl + "</tool_call>" + E),
        ("open_end_typed", u, [T], off, "<tool_call>" + nl + "<function=typed>" + nl +
         "<parameter=i>" + nl + "7" + nl + "<parameter=o>" + nl + '{"k": 1}' + nl +
         "</function>" + nl + "</tool_call>" + E),
        ("open_end_spaces", u, [W], off, "<tool_call>" + nl + "<function=get_weather>" +
         nl + "<parameter=city>" + nl + "  Ulm  " + nl + "</function>" + nl +
         "</tool_call>" + E),
        ("open_end_compact", u, [W, q3.NO_PARAMS], off,
         "<tool_call><function=get_weather><parameter=city>Berlin</function>"
         "<tool_call><function=now></function></tool_call>" + E),
    ]


def parse():
    print("Qwen3.8 response parsing cases")
    folder = qwen38_folder()
    check_tokenizer(folder)
    hf = hf_tokenizer(folder)
    flat, offsets, cases = [], [], []
    for cid, messages, tools, options, answer in qwen38_parse_cases():
        prompt = render(hf, messages, tools, True, options)
        _, ht = d7c.chat_to_hf(messages, tools)
        parsed = hf.parse_response(answer, prefix=prompt, tools=ht or None)
        p_ids = hf(prompt, add_special_tokens=False)["input_ids"]
        a_ids = hf(answer, add_special_tokens=False)["input_ids"]
        if hf.decode(a_ids) != answer:
            raise ValueError(f"{cid}: the answer does not decode back")
        offsets += [len(flat), len(flat) + len(p_ids)]
        flat += p_ids + a_ids
        calls = []
        for c in parsed.get("tool_calls", []):
            f = c["function"]
            calls.append({"name": f["name"],
                          "arguments": json.dumps(f["arguments"], ensure_ascii=False)})
        cases.append({"id": cid, "messages": messages, "tools": tools, "options": options,
                      "answer": answer, "content": parsed.get("content", ""),
                      "reasoning": parsed.get("reasoning_content", ""), "calls": calls,
                      "parsed": parsed})
    offsets.append(len(flat))
    blob = mr.Blob()
    blob.add("ids", np.array(flat, dtype=np.int64), "prompt and answer of every case")
    blob.add("offsets", np.array(offsets, dtype=np.int64),
             "case i: prompt ids[offsets[2i] .. offsets[2i+1]-1], answer "
             "ids[offsets[2i+1] .. offsets[2i+2]-1]")
    manifest = {
        "dataset": "qwen38-parse",
        "note": "per case a prompt (the conversation, tools and options through "
                "chat_template.jinja with the generation prompt) and an answer as the "
                "model would write it, both tokenized by the HF tokenizer; parsed: what "
                "transformers' parse_response makes of the answer with the "
                "response_template of tokenizer_config.json, the prompt as prefix and the "
                "tools of the case (their schemas type the arguments); content, "
                "reasoning and calls (name, arguments as json.dumps) taken from it",
        "model": model_info(),
        "environment": mr.environment(),
        "response_template": hf.response_template,
        "cases": cases,
    }
    blob.write(os.path.join(mr.OUT, "qwen38-parse"), manifest)
    print(f"  {len(cases)} cases, {len(flat)} ids")


TARGETS = {"qwen38-chat": chat, "qwen38-parse": parse}

if __name__ == "__main__":
    want = sys.argv[1:] or list(TARGETS)
    for t in want:
        if t not in TARGETS:
            raise SystemExit(f"unknown target {t}; known: {', '.join(TARGETS)}")
    for t in want:
        TARGETS[t]()
