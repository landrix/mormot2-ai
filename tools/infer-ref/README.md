# infer-ref — reference data for the native inference

`make_reference.py` generates the reference data the native engine
([docs/native-inference.md](../../docs/native-inference.md)) is checked against. The tool
runs only during provisioning, never at run time and never in a test. The tests read the
checked-in results under [tests/data/infer-ref/](../../tests/data/infer-ref/).

## Reference environment

The authoritative reference is **CPU on WSL aarch64**: Python 3.12.13, torch 2.14.1+cpu,
transformers 5.19.0. The full package resolution is in
[requirements.txt](requirements.txt). The environment lives outside the repo:

```bash
uv venv -p 3.12 ~/.venvs/lai-infer-ref
VIRTUAL_ENV=~/.venvs/lai-infer-ref uv pip install \
  --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu
VIRTUAL_ENV=~/.venvs/lai-infer-ref uv pip install -r tools/infer-ref/requirements.txt
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference.py            # everything
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference.py ops tiny   # without model
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference.py tokens     # tokenizer only
```

Targets: `ops`, `tiny`, `bge-m3` (model, cases, corpus vectors), `tokens` (tokenizer gate),
plus for phase 1b `q8`, `tiny-gguf` and `lembed-tokens`, for phase 3e `q4` (Q4_0 from
ggml itself: `quantize_row_q4_0_ref`, `dequantize_row_q4_0`, `ggml_vec_dot_q4_0_q8_0`),
for backlog B16 `q5` (Q5_0: `quantize_row_q5_0_ref`, `dequantize_row_q5_0`,
`ggml_vec_dot_q5_0_q8_0`) and for backlog D5b `mxfp4` (MXFP4: `quantize_row_mxfp4_ref`,
`dequantize_row_mxfp4`, `ggml_vec_dot_mxfp4_q8_0`).
With `bge-m3` or `tokens` the script loads BAAI/bge-m3 in the fixed revision
`5617a9f61b028005a4858fdac845db406aefb181` (about 2.3 GB into the Hugging Face cache).

The phase 1b targets compute with **ggml itself**, not with a reimplementation: they load
`libggml-base.so` and `libggml-cpu.so` from `vendor/sqlite-ext/aarch64-linux/` (the build
of the lembed oracle, redirectable with `LAI_GGML_DIR`) and call their exported functions
(`quantize_row_q8_0`, `ggml_vec_dot_q8_0_q8_0`, `quantize_row_q8_0_ref`,
`dequantize_row_q8_0`, `ggml_fp32_to_fp16`). The SHA-256 hashes of the libraries are in the
manifests. `lembed-tokens` also loads `lembed0.so` into Python's `sqlite3` and the
GGUF `vendor/models/bge-m3-q8_0.gguf` (redirectable with `LAI_GGUF`); the script aborts
if its SHA-256 is not the one of reference contract v1. Invocation:

```bash
LD_LIBRARY_PATH=$PWD/vendor/sqlite-ext/aarch64-linux ~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference.py q8 q4 q5 mxfp4 tiny-gguf lembed-tokens
```

## Sliding window and NoPE (phase D, D4)

`make_reference_d4.py` uses the same environment and the helpers of `make_reference_qwen3.py`
and loads no model: it generates `tiny-smollm3` (every combination of window and RoPE) and
`tiny-qwen3-sliding` (windows from `max_window_layers`). Greedy runs without EOS over
24 tokens, equal with the transformers cache and without a cache; it also checks that the
logits differ without a window (the reference really measures a window).

```bash
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference_d4.py          # both
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference_d4.py tiny-smollm3
```

## gpt-oss (phase D, D5)

`make_reference_d5.py` uses the same environment and the helpers of `make_reference_d4.py`
and loads no model: it generates `tiny-gpt-oss` - three layers (window 5, full, window 5),
8 experts of which a token takes 3, YaRN rotary positions, attention sinks, a bias on Q,
K, V, the output, the router and the experts, and a `swiglu_limit` of 0.8, so that the
clamp of the SwiGLU changes about a third of the values (checked by the generator).
transformers runs with the eager attention and the eager experts (one expert after the
other). Besides the states and the greedy continuations it stores the rotary factors of
positions 0..63 (`rope.*`), and those of two YaRN variants (`rope_t`: `truncate` true,
`rope_a`: an `attention_factor` and other betas) for the settings test. The target
`tiny-gpt-oss-narrow` writes `tiny-gpt-oss/narrow/`: experts of width 16 for a hidden width
of 64.

`tiny-gpt-oss-mxfp4` writes `tiny-gpt-oss/mxfp4/`: the same configuration with its own seed,
the experts quantized to MXFP4 and stored the way `openai/gpt-oss-20b` stores them
(`quantization_config` mxfp4, `*_blocks` and `*_scales`), split into two files with a
`model.safetensors.index.json`. Layer 0 gets a zero block with the exponent 0 and a block of
-0 values with the exponent 255. transformers loads the folder dequantized (exact in float32,
`Mxfp4Config(dequantize=True)`: what it does on a CPU anyway, without asking for
accelerate); the script checks that it reads the same values it wrote.

```bash
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference_d5.py          # the tiny ones
```

`tiny-gpt-oss-gguf` writes `tiny-gpt-oss/gguf/` from `tiny-gpt-oss/mxfp4`: the tensors and
metadata keys llama.cpp's converter writes for gpt-oss (those of `ggml-org/gpt-oss-20b-GGUF`),
the MXFP4 blocks repacked into ggml's layout and read back with ggml's own
`dequantize_row_mxfp4` (so it needs the ggml build, `LD_LIBRARY_PATH` as for `make_reference.py
q8`); matrices in BF16 or in Q8_0, one file with other YaRN betas, and damaged files.

`gpt-oss-20b` runs only when named. It reads `openai/gpt-oss-20b` in the revision
`6cee5e81ee83917806bbde320786a8fb61efebee` from `$LAI_GPTOSS_DIR` (by default
`vendor/models/gpt-oss-20b`; the script downloads nothing and checks the pinned hashes).
The experts stay in BF16 (the MXFP4 values are exact there) and are taken to float32 one
expert at a time as transformers uses them; everything else computes in float32. That needs
about 50 GB of RAM - more than the WSL reference has - so this dataset is generated on the
RTX computer (Windows x86_64, 96 GB) in an environment with the same pins
(`requirements.txt`, uv 0.11.18, Python 3.12.13, torch 2.14.1+cpu):

```bash
LAI_GPTOSS_DIR=/path/to/gpt-oss-20b python tools/infer-ref/make_reference_d5.py gpt-oss-20b
```

## The tokenizer of gpt-oss (phase D, D5c)

`make_reference_d5c.py` reads only the tokenizer files of `openai/gpt-oss-20b` in the same
revision (`$LAI_GPTOSS_DIR`, by default `vendor/models/gpt-oss-20b`; checked against the
pinned hashes, no weights needed) and loads no model:

- `o200k-unicode` writes `src/infer/mormot.ai.infer.unicode.o200k.inc`: the classes of the
  o200k Split regex (upper and title case, lower case, modifier and other letters, marks,
  numbers, white space), each set measured code point by code point with the regex engine
  of the tokenizers library, also the two negated classes; the letters, numbers and white
  space must equal those of `mormot.ai.infer.unicode.inc`, and the case folding of the
  contractions is measured as for Qwen2.
- `gpt-oss-tokens` writes the tokenizer gate: the texts of `qwen3-tokens` plus o200k edge
  cases and 1200 generated mixtures over the classes of the regex; per case the ids of
  `tokenizer.json` without its added tokens and the pieces of its pre-tokenizer.
- `gpt-oss-chat` writes the Harmony gate: conversations and tool schemas through the
  `chat_template.jinja` of the revision, rendered by transformers (`apply_chat_template`,
  the template kwargs `reasoning_effort` and `model_identity`) with the date fixed - the
  script replaces the `datetime` that transformers' `strftime_now` calls - and their ids;
  400 schemas generated over the branches of the TypeScript rendering. A case transformers
  refuses is stored with its error: the native template must refuse it too.

```bash
LAI_GPTOSS_DIR=/path/to/gpt-oss-20b python tools/infer-ref/make_reference_d5c.py
```

## Decoder stacks (backlog B14)

`make_reference_b14.py` needs only Python: it creates `tiny-qwen3/stack/` - the weights of
`tiny-qwen3` under the prefix `outer.model.`, its own head, a foreign tensor, a
broken stack with one tensor too many and in `config.json` the settings as objects
(with the multimodal RoPE of a talker, a faulty one and a scaled one). Nothing is
computed: `CreateStack` must give the bits of `tiny-qwen3`.

```bash
python tools/infer-ref/make_reference_b14.py
```

## A wide GQA group with a window (backlog D4b)

`make_reference_d4b.py` needs numpy (the environment above): it creates
`tiny-qwen3-sliding/g16/` - the config of `tiny-qwen3-sliding` with 16 query heads on one KV
head, random BF16 weights from a fixed seed. Nothing is computed: the Vulkan and CUDA tests
compare the device with the CPU on it; a group of 16 is more than the 8 queries
`attention.comp` holds, which a window layer of a `flash.comp` prefill must respect.

```bash
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference_d4b.py
```

## Qwen2.5 (phase D, D1b)

`make_reference_qwen2.py` uses the same environment and the helpers of
`make_reference_qwen3.py`. It loads Qwen/Qwen2.5-0.5B-Instruct in the revision
`7ae557604adf67be50417f59c2c2f167def9a775` into `vendor/models/qwen2.5-0.5b-instruct/`
(redirectable with `LAI_QWEN25_DIR`) and from Qwen/Qwen2.5-0.5B-Instruct-GGUF (revision
`9217f5db79a29953eb74d5343926648285ec7e67`) the files `q8_0` and `q4_k_m` into
`vendor/models/`.

```bash
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference_qwen2.py          # everything
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference_qwen2.py tiny-qwen2
```

Targets: `tiny-qwen2` (with `gguf/`), `qwen25-chat`, `qwen25`. Greedy with `repetition_penalty`
1: the model's `generation_config.json` asks for 1.1, and `generate` applies it even without
sampling - the reference is meant to be the pure argmax.

## Qwen3 (phase 2)

`make_reference_qwen3.py` uses the same environment and writes the Qwen3 datasets. It
loads Qwen/Qwen3-0.6B in the revision `c1899de289a04d12100db370d81485cdf75e47ca` (about
1.5 GB) and copies the files to `vendor/models/qwen3-0.6b/` (gitignored, redirectable with
`LAI_QWEN3_DIR`).

```bash
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference_qwen3.py          # everything
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference_qwen3.py qwen3-unicode
```

Targets:
- `qwen3-unicode` writes `src/infer/mormot.ai.infer.unicode.inc`. It measures every
  code point class against the pre-tokenizer of tokenizer.json and builds the NFC tables so
  that a reimplementation of the Pascal algorithm matches the HF normalizer for every code
  point, every pair and starter times character. The HF normalizer is based on older
  Unicode data than Python, so it is the contract and not `unicodedata`.
- `qwen3-tokens` and `qwen3-chat` are the gates of 2a.
- `tiny-qwen3moe` belongs to phase D (D2, MoE).
- `tiny-qwen3` and `qwen3` belong to 2b and 2c. The real model needs about 6 GB RAM
  and runs for several minutes on one thread.

## What the script pins down

- **FP32, eval(), one thread, deterministic algorithms, attention "eager".** Every
  case runs twice and must come out bit-identical. Two runs of the script write
  byte-identical blobs and a byte-identical `model.safetensors`; only the timestamp in the
  manifests changes.
- **Pooling from the revision itself:** `1_Pooling/config.json` must ask for CLS only and
  `modules.json` must contain a `Normalize`, otherwise the script aborts.
- **Weights as safetensors.** The revision contains only `pytorch_model.bin` (pickle). The
  script loads it with `weights_only=True` and writes an F32 `model.safetensors` to
  `vendor/models/bge-m3/` (gitignored, redirectable with `LAI_MODEL_DIR`), together with
  tokenizer and configuration. Then it loads the file again and compares tensor by
  tensor. The forward pass from both sources must also be bit-identical. Missing weights
  abort; only the pooler (`pooler.dense.*`) stays unused, because BGE-M3 dense takes the
  CLS state, not `pooler_output`.
- **Token IDs from SentencePiece**, not from the HF tokenizer. SentencePiece never recognizes
  its control symbols (`<s>`, `</s>`, `<pad>`) in the text. That is `parse_special=false`, and
  this is how the native tokenizer must work. The HF tokenizer, in contrast, recognizes `<s>`
  even with `split_special_tokens` or `encode_special_tokens`, because `<s>` is an entry of its
  Unigram vocabulary (measured, see `tokenizer_checks` in the manifest). For every text
  without such a literal the script checks that both give the same IDs. Mapping to
  XLM-R: SentencePiece ID + 1, except `<unk>` 0→3, `<s>` 1→0, `</s>` 2→2.

Regenerate only when the model, the environment or the cases change. Then rewrite all
datasets together, update `requirements.txt` and [UPSTREAM_BASE](../../UPSTREAM_BASE)
and state in the commit what changed.
