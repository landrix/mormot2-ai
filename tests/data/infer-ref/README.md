# infer-ref — Referenzdaten der nativen Inferenz

Erzeugt von [tools/infer-ref/make_reference.py](../../../tools/infer-ref/make_reference.py),
die Qwen3-Datensätze von
[tools/infer-ref/make_reference_qwen3.py](../../../tools/infer-ref/make_reference_qwen3.py)
(Umgebung und Festlegungen: [tools/infer-ref/README.md](../../../tools/infer-ref/README.md)).
Gelesen von der Suite `infer.tests` (`tests/test.infer.ref.pas`). Keine dieser Dateien wird
von Hand geändert.

## Format

Jeder Datensatz ist ein Ordner mit `manifest.json` und `tensors.bin`:
- `tensors.bin` enthält die Tensoren als rohe Little-Endian-Werte, hintereinander,
  4-Byte-ausgerichtet;
- `manifest.json` beschreibt jeden Tensor mit `name`, `dtype` (`f32`, `i32` oder `u8`),
  `shape` (zeilenweise, C-Reihenfolge), `offset`, `bytes` und `sha256`. Dazu kommen die
  Hashes der ganzen Datei, Modellrevision, Umgebung und Pipeline, außerdem der Zeitpunkt der
  Erzeugung. Die Blobs sind reproduzierbar, die Manifeste wegen dieses Zeitstempels nicht.

Der Lader prüft beim Öffnen den Hash der Datei, Lage und Länge jedes Tensors und das
Produkt der Form, bevor er irgendetwas liest. Ein Rechenergebnis wird nie über einen Hash
verglichen, sondern elementweise: `|a − r| ≤ atol + rtol·|r|`, dazu Kosinus und Normen.

## Datensätze

| Ordner | Inhalt | Größe |
|---|---|---|
| `ops/` | Operator-Fixtures: Linear (`x·Wᵀ + b`), LayerNorm (nicht-triviale γ/β, fast konstante und konstante Zeile), maskiertes Softmax (großer Logit), exaktes GELU (erf), L2 (Zeile mit Norm ~1e-20). Eingaben f32, Erwartung in float64 gerechnet. | 5 KiB |
| `tiny-xlmr/` | XLM-R mit 2 Layern, Breite 32, 4 Köpfe, Vokabular 64, BGE-M3-Konventionen (pad 1, bos 0, eos 2, eps 1e-5, exaktes GELU). Vertrag sind `model.safetensors` und `config.json`, nicht der Seed. Drei Sequenzen mit allen Zuständen, dazu derselbe Inhalt als rechts gepaddeter Batch, ebenfalls mit allen Zuständen. | 106 KiB |
| `bge-m3/` | Sechs kurze Fälle (≤ 16 Tokens inkl. `<s>`/`</s>`): leer, nur Leerzeichen, Handwerkstext, Unicode, mehrsprachig, Sondertoken-Literal. Je Fall IDs, Maske, Positionen, alle 25 Zustände (0 = Ausgang des Embedding-Moduls nach LayerNorm, 1–24 = Ausgang von Layer k), CLS und normierter Vektor. Dazu ein rechts gepaddeter Batch aus zwei Texten mit allen Zuständen, jeder Text auch als Einzellauf. Prüfung der Sondertoken-Literale unter `tokenizer_checks`. | 7,9 MiB |
| `bge-m3-tokens/` | Token-IDs als Gate für den Tokenizer: alle 320 Texte aus embed-de, 128 entworfene Grenzfälle (Leerraum aller Art, Steuer- und Nullbreitenzeichen, kombinierende Zeichen, Vollbreite, Ligaturen, Emoji mit ZWJ, mehrere Schriften, U+2581 im Text, alle Sondertoken-Literale einzeln/angrenzend/doppelt, 511/512/513/602 Tokens) und 900 erzeugte Unicode-Mischungen. Je Fall Text, normalisierter Text (SentencePiece) und die IDs; wo der HF-Tokenizer abweicht, zusätzlich `hf_ids`. Text und normalisierter Text liegen zusätzlich byte-genau im Blob (`texts`, `normalized` als `u8`): Ein JSON-Leser darf ein maskiertes NUL verlieren (mORMot liest es als `?`). | 0,4 MiB |
| `bge-m3-embed-de/` | FP32-Vektoren des Korpus [embed-de](../embed-de/) (262 Dokumente, 58 Anfragen, Zeile = Dateizeile). Sie ersetzen weder das lembed-Q8-Orakel in `embed-de/reference/` noch das nDCG-Gate von Phase 1b. | 1,25 MiB |
| `q8/` | Q8_0-Arithmetik aus ggml selbst (Build des lembed-Orakels): FP16-Rundung (`ggml_fp32_to_fp16`, mit Gleichständen, Überlauf, Subnormalen), Quantisierung von Aktivierungszeilen durch ggml-cpu (Gleichstände zur geraden Zahl, Nullblock, winzige und große Werte), Q8_0-Linear über `ggml_vec_dot_q8_0_q8_0`, Dequantisierung. | 7 KiB |
| `tiny-gguf/` | `tiny-xlmr` als GGUF, so wie llama.cpp XLM-R umwandelt: Architektur `bert`, Positionen ab 0 (die ersten pad + 1 Zeilen fallen weg), Matrizen und Wort-Embedding in Q8_0. Vertrag ist `model.gguf`. Erwartete Zustände: jede Matrixmultiplikation mit den Q8_0-Kerneln von ggml, der Rest in float64, nach jeder Operation auf f32 gerundet, exaktes GELU. Drei Sequenzen mit allen Zuständen, dazu ein gepaddeter Batch. | 46 KiB |
| `lembed-tokens/` | Token-IDs des lembed-Orakels (`lembed_tokenize_json`, GGUF des Referenzvertrags v1) für alle Fälle von `bge-m3-tokens`, in derselben Reihenfolge. `special_literal` markiert die 30 Fälle mit `<s>`, `</s>`, `<pad>`, `<unk>` oder `<mask>` im Text: lembed liest sie als Steuertokens (`parse_special = true`), der native Tokenizer als Text. Alle übrigen 1318 Fälle sind gleich der SentencePiece-Referenz (vom Generator geprüft). | 0,15 MiB |

| `qwen3-tokens/` | Token-IDs als Gate für den Qwen3-Tokenizer (Byte-Level-BPE): alle Texte aus embed-de, die Grenzfälle von `bge-m3-tokens`, 168 Qwen-spezifische Fälle (Kontraktionen samt U+017F, Ziffern, Zeilenumbrüche und Leerraum-Läufe, Satzzeichen, NFC mit Umordnung, Ausschlüssen und Hangul, alle Bytes, lange Läufe, die 26 Zusatztoken als Text) und 1800 erzeugte Mischungen. Referenz ist tokenizer.json **ohne** Zusatztoken: Jeder Text ist Text. `hf_ids`, wo der HF-Tokenizer abweicht (79 Fälle mit Token-Literalen). Text und NFC-Text auch byte-genau im Blob. | 0,6 MiB |
| `qwen3-chat/` | 21 Gespräche durch das Chat-Template der Revision (transformers rendert, der HF-Tokenizer zerlegt): mit und ohne System, Thinking an und aus, Verlauf mit `<think>`, Tools mit Zahlen und Escapes, Tool-Aufrufe und -Antworten. Gerenderter Text und IDs. | 0,02 MiB |
| `tiny-qwen3/` | Qwen3 mit 2 Layern, Breite 64, 4 Köpfen auf 2 KV-Köpfen, `head_dim` 32 (nicht 64/4), verbundenen Embeddings und BF16-Gewichten. Vertrag sind `model.safetensors` und `config.json`. Drei Sequenzen mit allen Zuständen, allen Logits und 16 Greedy-Tokens samt Abstand der zwei besten Logits je Schritt. | 0,3 MiB |
| `qwen3-tts-lm/` | Übernommen von flydev ([mormot2-qwen-tts](https://github.com/flydev-fr/mormot2-qwen-tts), Erzeuger in [tools/infer-ref/qwen3-tts/](../../../tools/infer-ref/qwen3-tts/)): die zwei Qwen3-Stapel von Qwen3-TTS-12Hz-0.6B-CustomVoice mit Embedding-Zeilen statt Token-Ids. Talker (28 Layer): 22 Eingabezeilen, alle Zustände, Logits der letzten Zeile (3072). Code-Predictor (5 Layer): 16 Eingabezeilen, die 15 Logit-Zeilen seiner Köpfe (je 2048) und die Greedy-Codes. Braucht das Modell (nicht im Repo); Regressionsdaten für den Einstieg mit Embeddings (Backlog B14). | 2,7 MiB |
| `tiny-qwen2/` | Qwen2 mit 2 Layern, Breite 64, 4 Köpfen auf 2 KV-Köpfen, `head_dim` 16 (Hidden / Köpfe: `config.json` nennt keins), Bias auf Q, K, V (ungleich null), keine Norm je Kopf, BF16. Drei Sequenzen wie `tiny-qwen3`; `gguf/` als qwen2-Datei (Q8_0, ohne `key_length`) und eine ohne `attn_k.bias`. Von [make_reference_qwen2.py](../../../tools/infer-ref/make_reference_qwen2.py). | 0,4 MiB |
| `qwen25-chat/` | Die Fälle von `qwen3-chat` durch das Chat-Template von Qwen2.5-0.5B-Instruct (Revision `7ae5576…`), von transformers gerendert und vom HF-Tokenizer zerlegt; Argumente von Tool-Aufrufen als Objekt an das Template. | 0,01 MiB |
| `qwen25/` | Qwen2.5-0.5B-Instruct in FP32: sechs Prompts (roh und Chat, einer mit Werkzeug), Ids, letzte Logits, 48 Greedy-Tokens mit Abstand, Zustände für zwei; `repetition_penalty` 1. | 8 MiB |
| `tiny-qwen3moe/` | Qwen3-MoE mit 3 Layern (0 und 2: 8 Experten der Breite 64, Top-2, Softmax-Router; 1: dicht), sonst wie `tiny-qwen3`; der erste Seed mit mindestens 0,05 zwischen k-tem und (k+1)-tem Router-Logit jedes Tokens. Drei Sequenzen wie `tiny-qwen3`; `unnormed/` dieselben Gewichte mit `norm_topk_prob` false (Fall 0); `gguf/` Q8_0 und Q4_0 mit 3-D-Experten-Tensoren und eine beschädigte Datei. | 2,1 MiB |
| `qwen3/` | Qwen3-0.6B in FP32: sieben Prompts (roher Text, Chat mit und ohne System, Thinking, Code, Tool-Aufruf), je IDs, Logits der letzten Position und bis zu 48 Greedy-Tokens samt Abstand je Schritt; für zwei Prompts alle 28 Layer-Zustände. Die Gewichte liegen nicht im Repo (`vendor/models/qwen3-0.6b/`). | 7,8 MiB |

Gemessen am 2026-10-14:
- Beim Tiny-Modell sind die Batch-Zeilen bitgleich mit den Einzelläufen. Bei BGE-M3 weicht
  die gepaddete Zeile um höchstens 1,3e-7 ab (Summationsreihenfolge), die ungepaddete gar
  nicht.
- Die FP32-Vektoren liegen beim Kosinus mindestens bei 0,99868 zum lembed-Q8-Orakel
  (Mittel 0,99929).
