# infer-ref — Referenzdaten für die native Inferenz

`make_reference.py` erzeugt die Referenzdaten, gegen die die native Engine
([docs/native-inference.md](../../docs/native-inference.md)) geprüft wird. Das Werkzeug
läuft nur bei der Bereitstellung, nie zur Laufzeit und nie im Test. Die Tests lesen die
eingecheckten Ergebnisse unter [tests/data/infer-ref/](../../tests/data/infer-ref/).

## Referenzumgebung

Verbindlich ist **CPU auf WSL aarch64**: Python 3.12.13, torch 2.14.1+cpu,
transformers 5.19.0. Die vollständige Paketauflösung steht in
[requirements.txt](requirements.txt). Die Umgebung liegt außerhalb des Repos:

```bash
uv venv -p 3.12 ~/.venvs/lai-infer-ref
VIRTUAL_ENV=~/.venvs/lai-infer-ref uv pip install \
  --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu
VIRTUAL_ENV=~/.venvs/lai-infer-ref uv pip install -r tools/infer-ref/requirements.txt
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference.py            # alles
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference.py ops tiny   # ohne Modell
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference.py tokens     # nur Tokenizer
```

Ziele: `ops`, `tiny`, `bge-m3` (Modell, Fälle, Korpusvektoren), `tokens` (Tokenizer-Gate),
dazu für Phase 1b `q8`, `tiny-gguf` und `lembed-tokens` und für Phase 3e `q4` (Q4_0 aus
ggml selbst: `quantize_row_q4_0_ref`, `dequantize_row_q4_0`, `ggml_vec_dot_q4_0_q8_0`).
Mit `bge-m3` oder `tokens` lädt das Skript BAAI/bge-m3 in der festen Revision
`5617a9f61b028005a4858fdac845db406aefb181` (rund 2,3 GB in den Hugging-Face-Cache).

Die Ziele der Phase 1b rechnen mit **ggml selbst**, nicht mit einer Nachbildung: Sie laden
`libggml-base.so` und `libggml-cpu.so` aus `vendor/sqlite-ext/aarch64-linux/` (der Build
des lembed-Orakels, mit `LAI_GGML_DIR` umlenkbar) und rufen deren exportierte Funktionen
auf (`quantize_row_q8_0`, `ggml_vec_dot_q8_0_q8_0`, `quantize_row_q8_0_ref`,
`dequantize_row_q8_0`, `ggml_fp32_to_fp16`). Die SHA-256 der Bibliotheken stehen in den
Manifesten. `lembed-tokens` lädt zusätzlich `lembed0.so` in Pythons `sqlite3` und das
GGUF `vendor/models/bge-m3-q8_0.gguf` (mit `LAI_GGUF` umlenkbar); das Skript bricht ab,
wenn dessen SHA-256 nicht die des Referenzvertrags v1 ist. Aufruf:

```bash
LD_LIBRARY_PATH=$PWD/vendor/sqlite-ext/aarch64-linux ~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference.py q8 q4 tiny-gguf lembed-tokens
```

## Qwen2.5 (Phase D, D1b)

`make_reference_qwen2.py` nutzt dieselbe Umgebung und die Helfer von
`make_reference_qwen3.py`. Es lädt Qwen/Qwen2.5-0.5B-Instruct in der Revision
`7ae557604adf67be50417f59c2c2f167def9a775` nach `vendor/models/qwen2.5-0.5b-instruct/` (mit
`LAI_QWEN25_DIR` umlenkbar) und aus Qwen/Qwen2.5-0.5B-Instruct-GGUF (Revision
`9217f5db79a29953eb74d5343926648285ec7e67`) die Dateien `q8_0` und `q4_k_m` nach
`vendor/models/`.

```bash
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference_qwen2.py          # alles
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference_qwen2.py tiny-qwen2
```

Ziele: `tiny-qwen2` (mit `gguf/`), `qwen25-chat`, `qwen25`. Greedy mit `repetition_penalty`
1: die `generation_config.json` des Modells verlangt 1,1, und `generate` wendet sie auch ohne
Sampling an - die Referenz soll das reine Argmax sein.

## Qwen3 (Phase 2)

`make_reference_qwen3.py` nutzt dieselbe Umgebung und schreibt die Qwen3-Datensätze. Es
lädt Qwen/Qwen3-0.6B in der Revision `c1899de289a04d12100db370d81485cdf75e47ca` (rund
1,5 GB) und kopiert die Dateien nach `vendor/models/qwen3-0.6b/` (gitignored, mit
`LAI_QWEN3_DIR` umlenkbar).

```bash
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference_qwen3.py          # alles
~/.venvs/lai-infer-ref/bin/python tools/infer-ref/make_reference_qwen3.py qwen3-unicode
```

Ziele:
- `qwen3-unicode` schreibt `src/infer/mormot.ai.infer.unicode.inc`. Es misst jede
  Codepoint-Klasse am Pre-Tokenizer von tokenizer.json und baut die NFC-Tabellen so,
  dass eine Nachbildung des Pascal-Algorithmus den HF-Normalisierer für jeden Codepoint,
  jedes Paar und Starter mal Zeichen trifft. Der HF-Normalisierer beruht auf älteren
  Unicode-Daten als Python, deshalb ist er der Vertrag und nicht `unicodedata`.
- `qwen3-tokens` und `qwen3-chat` sind die Gates von 2a.
- `tiny-qwen3moe` gehört zu Phase D (D2, MoE).
- `tiny-qwen3` und `qwen3` gehören zu 2b und 2c. Das echte Modell braucht rund 6 GB RAM
  und läuft auf einem Thread einige Minuten.

## Was das Skript festlegt

- **FP32, eval(), ein Thread, deterministische Algorithmen, Attention „eager".** Jeder
  Fall läuft zweimal und muss bitgleich herauskommen. Zwei Läufe des Skripts schreiben
  bytegleiche Blobs und ein bytegleiches `model.safetensors`; nur der Zeitstempel in den
  Manifesten ändert sich.
- **Pooling aus der Revision selbst:** `1_Pooling/config.json` muss nur CLS verlangen und
  `modules.json` ein `Normalize` enthalten, sonst bricht das Skript ab.
- **Gewichte als safetensors.** Die Revision enthält nur `pytorch_model.bin` (Pickle). Das
  Skript lädt es mit `weights_only=True` und schreibt ein F32-`model.safetensors` nach
  `vendor/models/bge-m3/` (gitignored, mit `LAI_MODEL_DIR` umlenkbar), zusammen mit
  Tokenizer und Konfiguration. Danach lädt es die Datei erneut und vergleicht Tensor für
  Tensor. Außerdem muss der Forward aus beiden Quellen bitgleich sein. Fehlende Gewichte
  brechen ab; ungenutzt bleibt nur der Pooler (`pooler.dense.*`), weil BGE-M3 dense den
  CLS-Zustand nimmt, nicht `pooler_output`.
- **Token-IDs aus SentencePiece**, nicht aus dem HF-Tokenizer. SentencePiece erkennt seine
  Steuersymbole (`<s>`, `</s>`, `<pad>`) nie im Text. Das ist `parse_special=false`, und
  so muss der native Tokenizer arbeiten. Der HF-Tokenizer erkennt `<s>` dagegen auch mit
  `split_special_tokens` oder `encode_special_tokens`, weil `<s>` ein Eintrag seines
  Unigram-Vokabulars ist (gemessen, siehe `tokenizer_checks` im Manifest). Für jeden Text
  ohne solches Literal prüft das Skript, dass beide dieselben IDs liefern. Abbildung auf
  XLM-R: SentencePiece-ID + 1, außer `<unk>` 0→3, `<s>` 1→0, `</s>` 2→2.

Neu erzeugen nur, wenn sich Modell, Umgebung oder Fälle ändern. Dann alle Datensätze
gemeinsam neu schreiben, `requirements.txt` und [UPSTREAM_BASE](../../UPSTREAM_BASE)
nachziehen und im Commit nennen, was sich geändert hat.
