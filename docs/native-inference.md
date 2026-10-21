# Native Inferenz in Pascal — Plan

> **Status (2026-10-14):** Phase 0, 1a und 1b erledigt: BGE-M3 läuft nativ von Text bis
> Vektor, in FP32 gegen PyTorch und aus demselben Q8-GGUF wie lembed gegen das Orakel
> (Token-IDs, Kosinus, nDCG@10 - siehe dort). 1c auf aarch64 und x86_64 (AVX2) erledigt:
> auf x86_64 1,6-mal (Linux) bis doppelt (Windows) so schnell wie lembed, auf aarch64 gleichauf bis schneller, jeweils
> mit weniger Speicher; unter Linux x86_64 (die Produktion) alle Suiten und das
> Korpus-Gate grün, Soak-Test über 90 Minuten bestanden; im Backend der empfohlene
> Provider (`AI_EMBED_PROVIDER=native`). Phase 1c ist damit abgeschlossen.
> Phase 2 (Decoder Qwen3) bis 2c erledigt: Tokenizer und Chat-Template treffen
> transformers in jedem Referenzfall, und die Greedy-Ausgabe von Qwen3-0.6B in FP32 ist
> für alle sieben Referenz-Prompts tokengleich - das Gate von Phase 2. 2d ist seit
> 2026-10-15 erledigt: `mormot.ai.llm.native` stellt das Modell hinter `ILlmClient`, der
> Agent-Loop ruft damit Tools auf. Phase 2 ist abgeschlossen; als Nächstes kommt
> Phase 3 (Geschwindigkeit).
> Grundlage ist der Entwurf des Nutzers
> („Plan: Native Embedding- und LLM-Inferenz in Pascal", 2026-10-14). Er wurde mit dem
> bestehenden Plan für Embeddings ([native-embeddings.md](native-embeddings.md),
> „Option D") zusammengeführt und mit Codex abgestimmt, das gegen Code, Modelldatei und
> Referenzdaten geprüft hat. Die Plattformbefunde unten sind am 2026-10-14 gemessen.
> Einordnung: [CONCEPT.md §5](../CONCEPT.md).

## Ziel

Eine eigene Inferenz-Engine in reinem Object Pascal für alle Plattformen, die mORMot2
unterstützt (FPC und Delphi). Sie beginnt mit **BGE-M3** (Embeddings, ersetzt
`lembed0` + `libllama`/`ggml`) und danach kommen **Decoder-LLMs** wie Qwen3. Zur Laufzeit
gibt es keine Abhängigkeit von fremden Repositories und keine Copyleft-Vorlagen.

**Was sich damit ändert:** Bisher war neural-api das strategische Backend für native
Inferenz (CONCEPT.md §5). Das entfällt. Das Beitragsangebot
[neural-api#225](https://github.com/joaopauloschuler/neural-api/issues/225) ist seit
2026-10-14 als „not planned" geschlossen. Die eigene Engine übernimmt beides: Embedder
**und** generatives `ILlmClient`. „Option D" aus native-embeddings.md geht als Phase 1
in diesen Plan ein.

## Architektur

Jede Schicht hat eine schmale, nicht-generische Schnittstelle. Neue Formate, Backends,
Tokenizer oder Modelle kommen dazu, ohne dass Bestehendes geändert wird.

Stil: wie der Rest der Extension **im Stil von mORMot2** — Regeln in
[CONCEPT.md §1](../CONCEPT.md). Für die Engine heißt das vor allem: keine Generics und
keine anonymen Methoden (Delphi 7), mORMot-Bausteine für Threads und Puffer, keine
Heap-Allokation in den Rechenschleifen.

Ablage (entschieden): im Unterordner **`src/infer/`** dieses Repos. Das ist eine klare
Grenze, ohne zweites Repo und ohne tiefe Verzeichnishierarchie. mORMot2 selbst ordnet
ebenso nach `src/<bereich>/`. Die Build-Skripte müssen den Pfad zusätzlich aufnehmen;
heute kennen sie nur `src/`.

| Unit | Inhalt |
|---|---|
| `mormot.ai.infer.tensor` | Tensor, Datentypen (F32, F16, BF16, Q8_0, Q4_0 …), Formen |
| `mormot.ai.infer.memory` | Memory-Mapping (Fenster für Win32 erst bei Modellen über 1,5 GB), Scratch-Puffer |
| `mormot.ai.infer.loader.safetensors` | safetensors-Leser |
| `mormot.ai.infer.loader.gguf` | GGUF-v3-Leser, robust gegen beschädigte Dateien |
| `mormot.ai.infer.backend` | Backend-Schnittstelle, Fähigkeitsabfrage, Fallback |
| `mormot.ai.infer.backend.cpu` | Pascal-Referenz + SIMD-Kernel (NEON, AVX2) |
| `mormot.ai.infer.tokenizer.ugm` | Unigram/SentencePiece inkl. `precompiled_charsmap` |
| `mormot.ai.infer.tokenizer.bpe` | Byte-Level-BPE (Qwen) |
| `mormot.ai.infer.model.registry` | Modelle nach `model_type` |
| `mormot.ai.infer.model.xlmr` | XLM-RoBERTa / BERT-Encoder (BGE-M3) |
| `mormot.ai.infer.model.qwen` | Qwen-Decoder |
| `mormot.ai.infer.pipeline.embed` | Pooling (CLS / Mean / Last-Token), Normierung |
| `mormot.ai.infer.pipeline.generate` | KV-Cache, Sampler-Kette, Stop-Behandlung |

Öffentliche Adapter liegen neben den bestehenden in `src/`:
- `mormot.ai.embed.native` mit `TNativeEmbedder: IEmbedder`. Das Interface passt
  unverändert.
- `mormot.ai.llm.native` mit `TNativeLlmClient: ILlmClient`. Completion und Streaming
  einschließlich Tool-Call-Daten passen. Die Engine muss aber das Chat-Template, das
  Parsen von Tool-Argumenten, Stop-Tokens und das Puffern unvollständiger UTF-8-Sequenzen
  selbst leisten. Inferenz-Einstellungen kommen über eine optionale Zusatzschnittstelle,
  **nicht** über die HTTP-Felder von `TLlmProviderConfig`.

**Die native Engine ist eine Implementierung unter mehreren.** Hinter denselben
Schnittstellen `ILlmClient` und `IEmbedder` gibt es heute schon:

| Weg | Chat (`ILlmClient`) | Embeddings (`IEmbedder`) |
|---|---|---|
| OpenAI-kompatibel: OpenAI, **IONOS AI Model Hub**, **Ollama**, LiteLLM, vLLM … | `TLlmClient` mit `OpenAIConfig`/`OllamaConfig`/`LiteLLMConfig`/`LlmConfigFromEnv` (beliebige `BaseUrl` + Key) | `TProviderEmbedder` über `/embeddings` |
| Anthropic | `TAnthropicClient` | – |
| lokal mit C-Bibliothek | – | `TLembedEmbedder` |
| **nativ (dieser Plan)** | `TNativeLlmClient` | `TNativeEmbedder` |

Welcher Weg gilt, entscheidet die Konfiguration des Konsumenten, nie ein `{$ifdef}`. Ein
Vektorraum bleibt dabei an **ein** Modell gebunden (Referenzvertrag): bge-m3 über IONOS
und bge-m3 lokal sind nur dann austauschbar, wenn ihre Vektoren den Paritätstest gegen
die Referenz bestehen. Sonst muss neu indiziert werden. Im Konsumenten fehlt heute noch
die Auswahl `provider` für Embeddings (die Backend-Fabrik kennt nur `none`/`lembed`).

### Was die Engine hinter `ILlmClient` leisten muss

Alles, was heute auf `ILlmClient` und `IEmbedder` aufsetzt, bleibt unverändert und
funktioniert mit der Engine, **sobald** sie die jeweilige Fähigkeit liefert:
- Agent-Loop `TLlmAgent`;
- MCP-Brücke `mormot.ai.agent.mcp`;
- Structured Output `ChatStructured`;
- RAG `TLlmRag` und das MCP-Tool `search_docs`;
- nativer Store.

Die Fähigkeit selbst muss die Engine erbringen. Ein Cloud-Anbieter liefert sie über sein
API mit:

| Fähigkeit | über einen Anbieter | nativ |
|---|---|---|
| Chat + Streaming | ✓ | Phase 2: Tokens gehen direkt aus der Dekodier-Schleife an den Streaming-Callback |
| Tool-Calls (Agent-Loop, MCP-Tools) | ✓ | Phase 2. Was die Engine dafür leisten muss:<br>• das Tool-Schema ins Chat-Template legen<br>• die Tool-Call-Ausgabe des Modells (bei Qwen3 JSON in `<tool_call>`) in `TLlmToolCall` übersetzen<br>Der Agent-Loop bleibt gleich. |
| Structured Output | ✓ (`response_format`) | erst Prompt + Validierung. **Garantiert** nur mit eingeschränktem Dekodieren (JSON-Schema → Grammatik → Maske über die Logits), eigener Schritt nach Phase 2. |
| Vision | ✓ | nicht geplant (bräuchte einen Bild-Encoder, etwa Qwen-VL) |
| Embeddings für RAG | ✓ (`TProviderEmbedder`) | Phase 1 (`TNativeEmbedder`) |

### Designentscheidungen

**Konfigurierbare Blöcke statt einer Klasse je Modell.** Ein Transformer-Block wird aus
der Modellkonfiguration zusammengesetzt:

| Eigenschaft | Varianten |
|---|---|
| Normalisierung | LayerNorm, RMSNorm |
| Position der Norm | Pre-Norm, Post-Norm |
| FFN | GELU, SwiGLU |
| Positionen | absolut, RoPE |
| Attention | MHA, GQA; optional Q/K-Norm |
| Kopfgröße | aus der Konfiguration (`head_dim`), **nicht** `hidden / heads` |
| Maske | bidirektional, kausal |
| Bias | an/aus je Projektion |

Qwen3 ist trotzdem **nicht** nur „llama2.c plus Konfiguration". Es braucht Q/K-Norm, und
Qwen3-0.6B hat `head_dim = 128`, obwohl 1024 / 16 = 64 wäre. Die Registry entsteht erst
mit dem zweiten Modell: kein Ausbau auf Vorrat.

**Backend ohne Generics und ohne anonyme Methoden.** Delphi 7 kennt beides nicht,
FPC 3.2.2 keine anonymen Methoden. Der Aufbau:
- **eine** Schnittstelle mit typisierten Operationen (MatMul, Add, LayerNorm, RMSNorm,
  RoPE, Softmax, GELU, SiLU, Attention), umgesetzt von einer virtuellen Basisklasse;
- intern Tabellen von Prozedurzeigern für die Skalar- und SIMD-Kernel, **einmal** beim
  Start per CPU-Erkennung von mORMot2 gewählt;
- **kein** Interface-Objekt je Operation.

**Fallback je Operation.** Jedes Backend meldet seine Unterstützung je Operation **und**
Datentyp/Layout/Form. Auf die CPU-Referenz wird nur bei „nicht unterstützt" ausgewichen,
nie nach einem Laufzeitfehler. Für die GPU-Phase gehören Gerätezuordnung, Pufferresidenz
und Scratch-Speicher vorher in den Entwurf: Ein stiller CPU-Fallback mitten im Graphen
kostet sonst Kopien und Synchronisation, die niemand sieht.

**CPU-Referenz ist Pflicht.** Jede Operation existiert zuerst in reinem Pascal. Sie läuft
überall, auch unter Delphi 7, und ist der Maßstab für alle optimierten Pfade.

**Threads.** `TSynParallelProcess` passt für grobe Zeilenblöcke, hat aber zwei
Eigenschaften, die man kennen muss: Aufrufe auf **einem** Pool laufen nacheinander, und
die Ausführung ist synchron. Deshalb:
- ein eigener Executor mit expliziter Zuteilung;
- Schwellen, unter denen nicht parallelisiert wird;
- nie denselben Pool rekursiv aufrufen.

**Tokenizer.** Ein Kern mit Adaptern für `tokenizer.json` (Hugging Face) und die
GGUF-Metadaten. Gleiche Charsmap-Bytes beweisen noch nicht dasselbe Verhalten:
Whitespace, Sondertokens und Nachbearbeitung werden einzeln gegen die Referenz geprüft.
Nutzertext wird mit `parse_special = false` tokenisiert (Referenzvertrag v1).

**Positionen bei BGE-M3.** Die Originalkonfiguration hat 8194 Positionen, im GGUF stehen
8192, weil der XLM-R-Offset (+2) beim Konvertieren entfernt wurde. Diese Behandlung
bleibt je Format ausdrücklich im Code und wird nie „passend gerechnet".

## Phasen

Reihenfolge (mit Codex abgestimmt): Im FP32-Pfad lassen sich Fehler am leichtesten
finden. Der nutzbare Ersatz für lembed (GGUF/Q8) kommt **vor** dem Decoder, damit der
erste Gewinn nicht hinter dem größten Umfang wartet.

### Phase 0 — Fundament

- ~~Lizenz festlegen~~ — **entschieden 2026-10-14**: mORMot-Dreifachlizenz
  (MPL 1.1 / GPL 2.0 / LGPL 2.1) für das ganze Repo, auch für `src/infer/`.
- Python-Referenzskript mit `transformers`, eingecheckt unter `tools/`. Folgendes wird
  festgelegt: Python- und Paketversionen, Modell-Revision und Hashes, FP32,
  `eval()`-Modus, Attention-Implementierung, Tokenizer-Einstellungen. Die verbindliche
  Referenzumgebung ist **CPU auf WSL aarch64**; GPU-Ausgaben dienen später nur als
  Vergleich.
- Eingecheckt (`tests/data/infer-ref/`):
  - Manifest;
  - Token-IDs, Masken, Positionen;
  - End-Embeddings;
  - **vollständige** Zwischentensoren für wenige kurze Fälle (16 Tokens × 24 Layer ≈
    1,5 MiB je Fall).

  Hashes oder Stichwerte genügen nicht: Sie lokalisieren keinen numerischen Fehler und
  ändern sich schon bei harmlosen Rundungsunterschieden.
- **Nicht** eingecheckt (gitignored, bei Bedarf erzeugt): Aktivierungen über den ganzen
  Korpus, Attention-Matrizen, Modelldateien.
- Winzige deterministische Modell- und Operator-Fixtures, damit die tägliche Testsuite
  ohne die 2-GB-Gewichte läuft.
- Testrahmen: mORMot `TSynTestCase` wie die bestehenden Suiten. Vergleiche über maximale
  absolute und relative Abweichung, Kosinus **und** Norm. Kosinus allein übersieht
  Skalierungsfehler.

**Fertig, wenn:** die Referenzdaten für BGE-M3 vorliegen und ein Test gegen die winzigen
Fixtures läuft.

**Erledigt (2026-10-14).** Ergebnis:
- Werkzeug [tools/infer-ref/](../tools/infer-ref/) mit fester Umgebung (Python 3.12.13,
  torch 2.14.1+cpu, transformers 5.19.0); Daten in
  [tests/data/infer-ref/](../tests/data/infer-ref/) (rund 7 MiB): Operator-Fixtures,
  Tiny-XLM-R, sechs BGE-M3-Fälle mit allen 25 Zuständen plus gepaddetem Batch,
  FP32-Vektoren des Korpus embed-de. Modellrevision, Datei-Hashes und Umgebung stehen in
  den Manifesten und in [UPSTREAM_BASE](../UPSTREAM_BASE).
- Zwei Läufe erzeugen bytegleiche Blobs und ein bytegleiches `model.safetensors`.
- Suite `infer.tests` (`scripts/run-fpc-infer-tests.sh`, `-Suite infer` in beiden
  Delphi-Skripten), 149 Assertions grün unter FPC 3.2.4 und 3.2.2 (aarch64-linux),
  Delphi 13 (Win64) und Delphi 7 (Win32). Sie prüft den **Testrahmen**, noch keinen
  Pascal-Forward:
  - Unversehrtheit (Hashes);
  - Abweisen beschädigter Daten;
  - Gegenproben für `InferCompare` (Skalierung, Ausreißer, NaN/Inf);
  - innere Konsistenz der Referenzen.

  Gegenproben: Eine ausgebaute Normierung, ein gekipptes Byte, eine ausgebaute
  Überlaufsperre im Lader und eine überzogene Kosinus-Schwelle lassen den Lauf
  fehlschlagen.

Befunde, die in Phase 1 eingehen:
- **Kein safetensors bei BGE-M3.** Die Revision enthält nur `pytorch_model.bin`
  (Pickle). Das Werkzeug konvertiert nach F32-safetensors (`vendor/models/bge-m3/`,
  gitignored) und prüft Tensoren und Forward gegen die Quelle. Phase 1a liest diese
  Datei.
- **Sondertokens im Text.** `tokenizer.json` führt `<s>` als Eintrag des
  Unigram-Vokabulars. Der HF-Tokenizer erkennt es deshalb auch mit
  `split_special_tokens` oder `encode_special_tokens` im Text (gemessen, Manifest
  `tokenizer_checks`). Die Referenz-IDs kommen
  aus SentencePiece (`sentencepiece.bpe.model`), das Steuersymbole nie im Text erkennt,
  mit fairseq-Offset. Für alle Texte ohne solches Literal sind beide gleich (geprüft).
  Der native Tokenizer muss die Steuer-Einträge beim Zerlegen auslassen.
- **FP32 gegen lembed-Q8:** Kosinus mindestens 0,99868, im Mittel 0,99929 über die 262
  Korpusdokumente.
- **Batch gegen Einzellauf:** Beim Tiny-Modell bitgleich. Bei BGE-M3 weicht die
  gepaddete Zeile um höchstens 1,3e-7 ab (Summationsreihenfolge). Die Engine-Tests
  vergleichen deshalb mit Toleranz, nicht bitgenau.

### Phase 1 — BGE-M3 als Ersatz für lembed

**1a. FP32 auf der CPU**
- Tensor-Kern, safetensors-Leser mit Memory-Mapping, naive MatMul in FP32.
- XLM-RoBERTa-Forward (nachgeschlagen in candle und `modeling_xlm_roberta.py`).
- Unigram-Tokenizer aus `tokenizer.json`, CLS-Pooling und L2-Normalisierung. Steuer-
  Einträge (`<s>`, `</s>`, `<pad>`, `<unk>`, `<mask>`) werden im Text nicht erkannt
  (Befund aus Phase 0). Verhalten wie SentencePiece, nicht wie der HF-Tokenizer: Der
  weicht auch bei C1-Steuerzeichen, Hangul-Jamo und kombinierenden Zeichen ab.

**Fertig, wenn:**
- die Token-IDs identisch sind;
- jeder Layer innerhalb fester absoluter und relativer Toleranzen liegt;
- das End-Embedding Kosinus ≥ 0,9999 und eine Norm von 1 ± 1e-5 hat;
- Batch und Einzelaufruf dasselbe liefern;
- Padding die Ergebnisse nicht verändert.

**Erledigt (2026-10-14).** Units unter `src/infer/`: `tensor`, `memory`,
`loader.safetensors`, `backend`, `backend.cpu`, `model.xlmr`, `tokenizer.ugm`,
`pipeline.embed` (`TInferEmbedder`). Ergebnis gegen die Referenz:
- Tokenizer: alle 1348 Fälle von `bge-m3-tokens` mit identischen IDs und identischem
  normalisierten Text (320 Korpustexte, 128 Grenzfälle, 900 Unicode-Mischungen), auch
  unter Delphi 7/Win32. Portiert aus llama.cpp (MIT, Herkunft in `UPSTREAM_BASE`).
- Encoder: alle 25 Zustände der sechs Fälle und des gepaddeten Batchs innerhalb
  atol 1e-4 / rtol 1e-3; gemessen höchstens 1,7e-5 absolut in den Zuständen und
  1,8e-7 im End-Embedding (Kosinus 0,999999999999).
- Text bis Vektor in einem gepaddeten Batch: Kosinus ≥ 0,9999 je Text; mehr als 512
  Tokens werden abgewiesen, nie gekürzt.
- Suite `infer.tests`: 752 Assertions grün unter FPC 3.2.4/3.2.2 und Delphi 13; unter
  Delphi 7 laufen alle Teile außer dem echten Modell (SKIP: die 2,3-GB-F32-Datei passt
  nicht ganz in einen 32-Bit-Prozess, das kommt mit Q8 in 1b).
- Die Referenzrechnung summiert durchgehend in double (jeder Faktor wird vor der
  Multiplikation erweitert), damit x87, SSE und NEON dieselbe Referenz liefern. Ein
  späterer SIMD-Pfad in FP32 wird gegen diese Referenz geprüft.
- Tempo (skalar, ein Thread, aarch64): rund 35 s für die sechs Fälle und den Batch.
  Geschwindigkeit ist Thema von 1c.

**1b. GGUF/Q8_0 — dasselbe Asset wie lembed** (vorher „Schritte 3 und 4" in
native-embeddings.md)
- GGUF-Leser, Q8_0-Dequantisierung bzw. Q8_0×F32-Kernel.
- Tokenizer-Adapter für die GGUF-Metadaten.

**Fertig, wenn:**
- die Token-IDs auf mindestens 1000 Unicode-Fällen identisch zu lembed sind. Wörtliche
  Sondertoken-Zeichenfolgen (`<s>`, `</s>`, `<pad>`) im Text werden **getrennt** gegen das
  gewollte Verhalten `parse_special = false` geprüft, nicht gegen lembed (das sie als
  Steuertokens liest);
- beschädigte Dateien sicher abgelehnt werden;
- der Kosinus gegen die Referenzvektoren aus `tests/data/embed-de` (lembed) ≥ 0,999 ist;
- **nDCG@10 ≥ 0,8863** erreicht wird, also höchstens 1 Prozentpunkt unter 0,8963.

Der Referenzvertrag v1 ist an **genau dieses** Q8-GGUF, 512 Tokens und das Verhalten der
Sondertokens gebunden. Ein bestandener FP32-Test (1a) beweist deshalb noch **keine**
Verträglichkeit mit bereits gespeicherten Vektoren. Beide Referenzen behalten getrennte
Identitäten (Vertrag, Fingerprint).

**Erledigt (2026-10-14).** Neu: `loader.gguf`; GGUF-Konstruktoren in `model.xlmr`,
`tokenizer.ugm` und `pipeline.embed` (`TInferEmbedder.CreateFromGguf`); Q8_0 als
Blocktyp in `tensor`; `Linear` nimmt typisierte Gewichte (`TInferWeight`). Ergebnis:
- **Rechenweg, gemessen vor dem Bau** (PyTorch mit den Q8-Gewichten gegen die
  lembed-Vektoren der 320 Korpustexte): Q8-Gewichte mit F32-Aktivierungen erreichen
  nur Kosinus 0,99829 (ein Text unter 0,999). Erst wenn die Aktivierungen wie bei ggml
  vor jeder Matrixmultiplikation auf Q8_0 quantisiert werden, liegt jeder Text darüber
  (min 0,99949). Die FP16-Tanh-GELU-Tabelle von ggml war schlechter (min 0,99883),
  daher bleibt die exakte GELU. Die Engine rechnet also wie ggml-cpu auf aarch64:
  Skala `max|x|/127` in FP16, `x·(1/d)` auf die nächste gerade Zahl gerundet
  (NEON und AVX runden so, die C-Referenz von ggml nicht), ganzzahlige Blocksummen.
  Der AVX-Pfad bildet den Kehrwert als `127/max|x|` und kann einen Grenzfall um eine
  Stufe verschieben: bytegleich zum aarch64-Orakel, nicht zu jedem ggml-Build.
- **Positionen:** Der Konverter von llama.cpp lässt die ersten pad + 1 Zeilen weg
  (8192 statt 8194). Die Regel steht je Format im Code (`FirstPosition`: pad + 1 bei
  safetensors, 0 bei GGUF) und ist mit Gegenprobe getestet.
- **Token-IDs:** alle 1318 Fälle ohne Sondertoken-Literal identisch zu lembed, alle
  1348 identisch zu SentencePiece. Die 30 Literal-Fälle prüft der Test getrennt: lembed
  liest dort in 24 Fällen Steuertokens, die Engine Text (`parse_special = false`).
  Stücke vom Typ UNUSED (`[PAD250000]`) erkennt auch lembed nie im Text (Extra-Fälle).
- **Vektoren und Gate** (ganzer Korpus, `LAI_INFER_GATE=1`): Kosinus gegen lembed
  mindestens 0,99953 (alle 320 Texte über 0,999), **nDCG@10 0,8989**
  (Gate 0,8863, lembed 0,8963).
- **Beschädigte Dateien:** rund 30 gezielte Defekte (Zählungen, Typen, NUL, Schachtelung,
  Ausrichtung, Überlappung, Lücken, Reihenfolge), jede Kürzung der Tiny-Datei und 4096
  gekippte Header-Bytes enden in `EInferError`, nie in einem anderen Fehler.
- **Fixtures aus ggml selbst** (`q8`, `tiny-gguf`, `lembed-tokens`): FP16-Rundung,
  Aktivierungsquantisierung byte-genau, Q8_0-Skalarprodukt, ein Tiny-XLM-R als GGUF
  mit 64-Stück-Tokenizer.
- **Plattformen:** `infer.tests` grün unter FPC 3.2.4/3.2.2, Delphi 13 und Delphi 7.
  Unter Delphi 7/Win32 läuft jetzt auch das echte Modell: Das Q8-GGUF (634 MB) passt
  ganz in den Adressraum, Mapping in Fenstern braucht erst ein größeres Modell.
- **Tempo** (skalar, ein Thread, aarch64): der ganze Korpus in rund 51 min;
  lembed braucht rund 72 s. Das ist Thema von 1c.

**1c. Tempo, Messung, Umschalten** (vorher „Schritte 5 und 6")
- Minimal-SIMD für das Q8_0-Skalarprodukt (NEON, AVX2) und Threads über Zeilenblöcke.

**Fertig, wenn:**
- der Durchsatz mindestens 80 % von lembed erreicht, bei höchstens 125 % Peak-RSS;
- der Durchsatz unter Dauerlast stabil ist: Soak-Test über 90 Minuten unter Linux x86_64,
  gedrosselt auf 16 Threads, jeder Vektor über alle Durchgänge bitgleich (`embed-bench …
  soak90 16`). Die Erstindizierung von 2 Mio Texten wird daraus hochgerechnet, nicht
  abgewartet: 28 h Volllast wären für den Rechner unnötige Last (geändert 2026-10-14, mit
  dem Nutzer abgestimmt);
- der Build unter FPC 3.2.2/3.2.4 und Delphi 7 bis 13 grün ist (Delphi 7 ohne
  Leistungsziel).

Danach wird **nativ** zum Default. lembed bleibt als expliziter Fallback und als
unabhängiges Orakel erhalten, bis die nativen Gates auf allen Zielplattformen stehen.

**Stand 2026-10-14: Teil a (aarch64) erledigt.** Neu: `kernels` (NEON, Befehle als
Kodierung, weil FPC 3.2 keine NEON-Registerlisten assembliert), `threads` (eigener Pool),
`TInferCpuFastBackend` in `backend.cpu`, der Adapter `mormot.ai.embed.native`
(`TNativeEmbedder`, `SharedNativeEmbedder`) und `demos/infer/embed-bench.lpr`.
- **Zwei Rechenverträge** (mit Codex entschieden): `Exact = true` rechnet bitgleich zur
  skalaren double-Referenz, nur auf allen Kernen. Der Default rechnet wie ggml in
  float-Spuren (Q8_0-Linear: SDOT-Teilsummen, je Block mit dw·dx skaliert, am Zeilenende
  reduziert; Attention in float; GELU über eine kubische Hermite-Tabelle, Fehler < 1e-9)
  und wird mit Toleranz gegen die Referenz geprüft. Exakt bleiben in beiden Modi die
  Quantisierung der Aktivierungen (byte-gleich zu ggml, NEON wie Pascal) und die
  Integer-Blocksummen. Ein Aufrufer, der den Pool belegt findet, rechnet mit denselben
  Kernen auf seinem Thread: dieselben Bits, unabhängig von der Last.
- **erf neu:** eigene Polynomanpassung (`tools/infer-ref/fit_erf.py`, mpmath),
  relativer Fehler < 1e-14, rund 20 ns statt bis zu 200 ns je Wert.
- **Qualität (schneller Pfad, ganzer Korpus):** nDCG@10 **0,9004** (lembed 0,8963,
  Gate 0,8863), Kosinus gegen lembed mindestens 0,99949; der Korpus in 34 s statt
  51 min auf der Referenz.
- **Durchsatz** (Snapdragon X, WSL aarch64, ganzer Korpus, im Wechsel gemessen, Modell
  auf dem Linux-Dateisystem - über `/mnt/d` kostet das Mapping rund 25 %):

  | | lembed | nativ | Anteil |
  |---|---|---|---|
  | 4 Kerne (taskset, je 4 Threads), Runde 1 / 2 | 267 / 216 Tokens/s | 284 / 280 | 106 % / 130 % |
  | ganze Maschine, je Default, Runde 1 / 2 | 251 / 222 | 285 / 238 | 113 % / 107 % |
  | 1 Kern, 40 Texte | 114 | 96 | 84 % |
  | Peak-RSS | 760 MB | 480 MB | 63 % |

  Default sind zwei Drittel der Kerne: Alle Kerne spinnen zu lassen ist in der VM
  langsamer. Unter Dauerlast drosselt der Laptop; deshalb wechseln die Läufe ab.
- **Hochrechnung Erstindizierung:** 5,6 Texte/s bei ⌀ 51 Tokens ergibt für 2 Mio Texte
  rund 100 h auf diesem Laptop (Hochrechnung; das Gate ist der Soak-Test, siehe oben).
- **Reviews:** Codex und Claude-Finder fanden u. a. einen ABA-Fehler im Pool (Aufträge
  mit 8, 2, 8 Einträgen konnten Count und Base mischen - jetzt Seqlock), einen
  Überlauf des Ticketzählers (jetzt Neustart bei 2^30 im Ruhezustand), den verlorenen
  `Exact`-Modus im seriellen Attention-Pfad, einen Delphi-7-Startfehler (signalisierendes
  NaN durch die x87-FPU) und Tests, die NaN durchließen. Alle behoben, mit Gegenprobe.

**Stand 2026-10-14: Teil b (x86_64, AVX2) erledigt.**
- **Kernel** in `kernels` unter `{$ifdef CPUX64}{$ifdef ASMX64AVX1}`, Auswahl zur Laufzeit
  bei AVX2 + FMA + F16C: der FP32-Kernel 4×2 (vpsignb/vpmaddubsw/vpmaddwd wie ggml, FMA
  mit dw·dx), der Quantisierer (byte-gleich zu Pascal), die exakten Integer-Blocksummen
  (vpmovsxbw + vpmaddwd, auch für -128) und die beiden float-Attention-Kernel.
- **Eine ABI für alle:** Jeder Kernel bekommt genau einen Record mit seinen Parametern
  und Platz für die Register, die die ABI zurückverlangt (unter Win64 auch xmm6 bis
  xmm15). Der Asm-Körper ist damit für Win64 und System V, FPC und Delphi derselbe; eine
  Pascal-Hülle mit der gewohnten Signatur füllt den Record.
- **F16C als Bytes:** FPC 3.2.2 kennt `vcvtph2ps`/`vcvtps2ph` nicht; per objdump geprüft.
- **Vertrag des FP32-Kernels:** Aktivierungen liegen in -127..127 (der Quantisierer
  schreibt nie -128). Der vpsignb-Trick von ggml verlangt das; der Test hält sich daran.
- **LayerNorm auf allen Kernen**, zeilenweise, je Zeile die Referenzarithmetik: bitgleich
  in beiden Modi. Auf dem i9 fiel sie damit von 26 % auf 3 % der Zeit.
- **Messung** (i9-14900K, Windows 11, FPC 3.2.2 win64; Zahlen in der README): nativ
  1039 Tokens/s gegen lembed 524 (×1,98), 377 MB gegen 514 MB Spitzenspeicher, Kosinus
  ≥ 0,9991. Default (22 Threads) ist zugleich der beste Wert.
- **Windows drosselt fensterlose Prozesse** (Power Throttling/EcoQoS, z. B. per SSH oder
  als Dienst gestartet): Beide Engines verlieren dann etwa die Hälfte (lembed 187, nativ
  417). `embed-bench` schaltet das für sich ab; ein Windows-Dienst, der viel einbettet,
  sollte es ebenso tun.
- **Erstindizierung, hochgerechnet:** 20 Texte/s bei ⌀ 51 Tokens ergeben für 2 Mio Texte
  rund 28 h auf dem i9.

- **Linux x86_64** (Ubuntu 24.04 unter WSL2 auf dem i9, FPC 3.2.2 wie im Docker-Build,
  2026-10-14): MCP 1257, LLM 656, Inferenz 828 mit Korpus-Gate (nDCG@10 0,8994, Kosinus
  ≥ 0,99918), der System-V-Pfad damit erstmals auf Hardware gelaufen. Durchsatz nativ
  861 gegen lembed 544 Tokens/s (×1,58), 439 gegen 755 MB.
- **Fund dabei:** Die Fixture `tests/data/infer-ref/tiny-gguf/model.gguf` fiel unter
  `*.gguf` in `.gitignore` und fehlte in jedem frischen Klon (26 Fehlschläge). Jetzt mit
  Ausnahme eingecheckt.

- **Soak-Test bestanden** (2026-10-14, Linux x86_64 unter WSL2 auf dem i9, FPC 3.2.2,
  16 Threads, `embed-bench … soak90 16`): 90 Minuten, 248 volle Durchgänge über den
  Korpus plus 268 Texte, zusammen 79.628 Texte und 4,05 Mio Tokens. Durchsatz über alle
  neun 10-Minuten-Intervalle 744 bis 752 Tokens/s, RSS durchgehend 437 MB (Spitze 442):
  kein Einbruch und kein Wachstum beobachtet. **0 Vektoren** weichen vom ersten Durchgang
  ab (79.308 bitgenaue Vergleiche). CPU bei 88 bis 90 °C, Lüfter 100 %; Microcode 0x12F
  (die Raptor-Lake-Korrektur ab 0x12B ist drauf). Nach Überschlag (rund 25.000
  Pool-Tickets je Text, zusammen rund 2,0 Mrd) ist der Ticket-Neustart bei 2^30 dabei
  einmal gelaufen; gezählt hat das Programm ihn nicht.
- **Erstindizierung, hochgerechnet:** aus dem Soak 14,7 Texte/s mit 16 Threads, für
  2 Mio Texte rund 38 h unter Linux; aus der Einzelmessung mit dem Default von 22
  Threads (861 Tokens/s) rund 33 h, über Dauer nicht gemessen.

- **Umschalten im Backend** (2026-10-14): Provider `native` in
  `landrix.server.ai.Embedder` - braucht nur `AI_EMBED_MODEL_PATH`, `AI_EMBED_THREADS`
  optional; `.env.sample`, Docker-Doku und Entrypoint führen ihn als empfohlenen Weg,
  lembed bleibt Rückfall und Orakel. Der Feature-Schalter bleibt per Default `none`
  (kein Modell im RAM, solange niemand AI einschaltet). Ein echter Warmup-Test lädt
  dabei die 32-KB-Fixture `tiny-gguf`.
- **Load Governor** (`mormot.ai.governor`, im Backend `GetBulkEmbedder` mit
  `AI_BULK_*`): Tastgrad, Temperaturschwelle über Linux-hwmon, Durchsatz-Heuristik.
  Review-Fund dabei: sysfs meldet für jede Datei 4096 Bytes, `StringFromFile` lieferte
  deshalb leere Strings - die Temperatur war nie lesbar; jetzt `StringFromFileNoSize`,
  mit Regressionstest gegen das echte `/sys` und Gegenprobe.

**Phase 1c ist abgeschlossen.** Offen bleibt nur der erste Aufrufer des Governors (ein
Indizier-Job im Backend) und ein Lauf auf echter Sensor-Hardware.

### Phase 2 — Decoder

- Gerüst nach llama2.c (MIT), umgebaut auf konfigurierbare Blöcke.
- Erstes Modell **Qwen3-0.6B**: Q/K-Norm, `head_dim` aus der Konfiguration, GQA.
- Byte-Level-BPE, KV-Cache, Greedy-Sampling.
- Chat-Template der unterstützten Modelle fest in Pascal.
- `mormot.ai.llm.native` hinter `ILlmClient`, mit Streaming und Tool-Calls.

**Fertig, wenn:** Greedy-Sampling in FP32 für einen festen Prompt-Satz tokengleich zu
`transformers` ist.

**Stand 2026-10-14:** 2-0 bis 2c erledigt, das Gate ist erfüllt. Offen ist 2d.

- **2-0, Referenzdaten** (`tools/infer-ref/make_reference_qwen3.py`, Qwen/Qwen3-0.6B in
  der Revision `c1899de289a04d12100db370d81485cdf75e47ca`):
  - `qwen3-tokens`: 2416 Fälle für den Tokenizer;
  - `qwen3-chat`: 21 Gespräche für das Chat-Template, mit Tools, Tool-Aufrufen und
    Thinking;
  - `tiny-qwen3`: ein winziges Qwen3 mit 2 Layern, BF16-Gewichten und `head_dim` 32
    statt 64/4, mit allen Zuständen;
  - `qwen3`: sieben Prompts des echten Modells in FP32, mit Zuständen, Logits und je
    48 Greedy-Tokens samt dem Abstand der zwei besten Logits pro Schritt.
- **2a, Tokenizer** (`mormot.ai.infer.unicode`, `mormot.ai.infer.tokenizer.bpe`):
  - NFC und die Zeichenklassen des Split-Regex (`\p{L}`, `\p{N}`, `\s`) stehen als
    erzeugte Tabellen im Code. Sie sind am HF-Tokenizer selbst gemessen und über alle
    Codepoints dagegen geprüft.
  - Befund dabei: Der NFC des HF-Tokenizers beruht auf älteren Unicode-Daten als Python
    (15.0), mit 1129 Abweichungen. Die Tabellen folgen deshalb dem Tokenizer, nicht der
    neuesten Unicode-Version.
  - Das Regex läuft nicht über eine Regex-Engine, sondern als handgeschriebener Abgleich
    seiner sieben Alternativen. Ein Fuzzing mit 200.000 Fällen gegen den HF-Pre-Tokenizer
    ergab keine Abweichung.
  - BPE arbeitet mit derselben Merge-Reihenfolge wie die tokenizers-Bibliothek: Rang,
    dann Position, veraltete Einträge werden übersprungen.
- **2a, Chat-Template** (`mormot.ai.infer.chat`): Das Jinja-Template von Qwen3 ist in
  Pascal nachgebaut. Tools werden wie `json.dumps` von Python serialisiert
  (`PythonJson`).
  - Eine Abweichung ist gewollt: Steuertokens kommen nur aus dem Template, Nachrichtentext
    bleibt immer Text. Ein `<|im_end|>` im Text eines Nutzers schließt also nicht seinen
    Turn.
  - Floats mit mehr als 15 signifikanten Stellen weist `PythonJson` ab, statt die
    binäre Rundung zu raten.
- **2b, Forward** (`mormot.ai.infer.model.qwen3`):
  - Aufbau: RMSNorm, Q/K-Norm je Kopf, RoPE, GQA, kausale Attention, SwiGLU, verbundener
    oder eigener LM-Head.
  - BF16-Gewichte werden beim Laden exakt auf F32 erweitert.
  - Abweichung von der Referenz: Zustände und Logits weichen höchstens um 9e-5 ab
    (Kosinus 0,99999999998).
  - Rotationswinkel: Sie folgen den float32-Schritten der Referenz.
- **2c, KV-Cache und Greedy:**
  - Logits über den Cache sind bitgleich zu einem vollen Durchlauf.
  - Alle sieben Prompts sind tokengleich, auch einer mit nur 0,0046 Abstand zwischen den
    zwei besten Tokens und der Tool-Aufruf.
  - Gegenprobe: Schlüssel an Position 0 statt an ihrer Cache-Position machen den
    Cache-Test und das Greedy-Gate rot.
  - Zeiten auf dem Snapdragon X (WSL, FP32, Pascal-Skalarcode): 4 + 48 Tokens in
    13 s, 193 + 20 Tokens in 46 s. Phase 2 ist bewusst auf Korrektheit gebaut,
    Geschwindigkeit ist Phase 3.
- **2d, erledigt 2026-10-15:** `mormot.ai.llm.native` (`TNativeLlmClient`) hinter
  `ILlmClient`.
  - Prompt: das Chat-Template mit geöffnetem Assistant-Turn. `enable_thinking` kommt
    pro Anfrage aus `Extra.chat_template_kwargs` (wie bei vLLM), sonst gilt die
    Voreinstellung des Clients (an, wie beim Modell).
  - Antwort: `TQwen3OutputParser` liest Token für Token.
    - Ein `<think>` als erstes Token öffnet die Denkphase. Sie landet wie bei vLLM als
      `reasoning_content` in `Raw`, nie in `Content`.
    - Jeder `<tool_call>`-Block aus striktem JSON mit genau
      `{"name": …, "arguments": {…}}` wird ein `TLlmToolCall`. Die Argumente stehen wie
      bei `json.dumps`, so rendert das Template sie im Verlauf zurück; nur ein Float, den
      `json.dumps` runden würde (mehr als 15 Stellen oder außerhalb des
      Double-Bereichs), bleibt so, wie das Modell ihn schrieb - ein Tool, das seine
      Argumente als Double liest, rundet oder verweigert diesen Wert. Alles andere
      bleibt sichtbar als Text.
    - Ein Token kann mitten in einem UTF-8-Zeichen enden; diese Bytes warten auf das
      nächste Token. Ungültiges UTF-8 wird U+FFFD.
    - Zeilenumbrüche: am Anfang, am Ende der Denkphase und vor einem gültigen
      Tool-Call fallen sie weg (das Template schreibt dort eigene), am Ende einer
      normalen Antwort und vor einem Block, der Text bleibt, bleiben sie.
  - `ChatComplete` und `ChatStream` laufen über denselben Weg, die gestreamte Antwort
    ist also die vollständige in Stücken.
  - Stop: `<|im_end|>` und `<|endoftext|>`. Kontextgrenze: Das letzte Token wird nicht
    zurückgeführt, ein Prompt aus P Tokens lässt MaxPositions − P + 1 neue zu.
  - Bewusst noch nicht: Sampling (Temperature wird ignoriert, gedekodiert wird greedy),
    Prompt-Cache (beides seit 3b/3c da), Bilder und `ResponseFormat` (beides wird mit
    Fehler abgelehnt statt still übergangen).
  - Tests: Parser, Prompt und Dekodierung ohne echtes Modell (winziger Byte-Tokenizer,
    das winzige Qwen3 an der Kontextgrenze); mit Qwen3-0.6B die Referenz-Prompts
    (`chat_en` „Paris“, `chat_tool` ruft `get_weather` mit `{"city": "Berlin"}`,
    `chat_think` gibt genau die Referenz-Denkphase, `chat_de` gestreamt genau den
    Referenztext) und ein Agent-Loop mit Tool. Gegenproben schlagen an, u. a.: Kontext
    um ein Token zu knapp, Zeilenumbruch am Anfang behalten oder am Ende verworfen,
    Think-Erkennung aus, zusätzlicher (auch leerer) Schlüssel im Tool-Call, ein
    4-Byte-Zeichen zu früh ausgegeben.

Qwen3-Embedding ist **kein** Nebenprodukt: Es braucht einen eigenen, separat trainierten
Checkpoint und eigene Eingabekonventionen, nicht nur Last-Token-Pooling. Es gehört zu
„Danach".

### Phase 3 — Quantisierung und Geschwindigkeit

- Q4_0, später K-Quants (ggml, MIT, ausgewählt übernommen).
- Volle SIMD-Kernel (AVX2, NEON) und Multithreading.
- Sampler: Temperature, Top-k, Top-p, Stop-Tokens.
- Benchmarks je Plattform.

**Fertig, wenn:** quantisierte Modelle die Referenztests mit festgelegter Toleranz
bestehen und die Benchmarks dokumentiert sind.

**3a, erledigt 2026-10-15: Geschwindigkeit des Decoders.**

- Befund (`demos/infer/llm-bench`, Snapdragon X, 8 Threads): F32 schaffte im Prefill
  2,7 und beim Dekodieren 1,6 Tokens/s. 96 bzw. 93 % der Zeit gingen in `Linear`, denn
  der F32-Zweig des schnellen Backends war noch die skalare Referenzschleife.
- Umgesetzt:
  - **Q8_0-Gewichte:** `TQwen3Model.Create(…, idtQ8_0)` quantisiert jede Matrix beim
    Laden zeilenweise, byte-gleich zu ggml (auch den gebundenen Kopf). Das Embedding
    liest dann je Token die Zeile aus der Datei. Ein NaN, ein Unendlich oder ein
    Gewicht jenseits des Q8_0-Bereichs wird beim Laden abgelehnt.
  - **F32 im Float-Modus:** `LinearTileWork` rechnet jede Kachel mit `InferDotsF32`
    (NEON, AVX2); `Exact` bleibt die bitgleiche Referenzschleife.
  - **Attention als Backend-Operation** `CausalAttention` (`ioCausalAttention`): Das
    Referenz-Backend rechnet wie in Phase 2 in Double. Das schnelle verteilt Kopf mal
    8-Zeilen-Block auf den Pool, mit `Exact` bitgleich, sonst mit Float-Kerneln.
- Messung (200 / 1000 Tokens Prompt, dann 32 neue; ein Lauf, der Laptop drosselt unter
  Dauerlast):

  | | Prefill | Dekodieren | Speicher-Peak |
  |---|---|---|---|
  | F32 vorher, 200 | 2,7 Tokens/s | 1,6 Tokens/s | 3,8 GB |
  | F32, 200 | 103 Tokens/s | 24 Tokens/s | 3,8 GB |
  | F32, 1000 | 87 Tokens/s | 5 Tokens/s | 4,2 GB |
  | Q8_0, 200 | 246 Tokens/s | 36 Tokens/s | 2,2 GB |
  | Q8_0, 1000 | 157 Tokens/s | 20 Tokens/s | 2,6 GB |

  Mit Q8_0 und langem Prompt gehen inzwischen über die Hälfte der Zeit in Normen,
  RoPE, SwiGLU und die Attention: Das ist der nächste Hebel.
- Treue:
  - F32 im Float-Modus: Greedy bleibt auf allen sieben Referenz-Prompts tokengleich.
  - Q8_0: Gate in `Qwen3Q8AgainstReference`. Kosinus der Logits ≥ 0,98, höchstens
    3 Logits Abweichung, Greedy gleich bis zum ersten Schritt, an dem die Referenz
    weniger als 0,5 Vorsprung hatte (gemessen höchstens 0,27). Am empfindlichsten ist
    `chat_tool` (193 Tokens): in exakter Arithmetik Kosinus 0,9935 und 1,72 Abweichung,
    auf aarch64 und x86_64 bitgleich; mit NEON 0,9965 / 1,36, mit AVX2 0,9868 / 2,11.
    Float gegen exakte Q8_0-Rechnung allein: Kosinus 0,998 bis 0,999, bis zu 0,95
    Abweichung, auf beiden Plattformen. Vermutete Ursache, nicht innen gemessen: Eine
    geänderte Rundung kippt einzelne int8-Aktivierungen der nächsten Schicht. Ein
    kaputter Quantisierer verfehlt das Gate
    weit (verschobene Zeilen: Kosinus 0,23, Abweichung 24).
  - Befund: Mit Q8_0 beantwortet Qwen3-0.6B die Wetterfrage von `chat_tool` selbst
    („Das …“), statt `get_weather` aufzurufen (Referenzabstand 0,22). Ein Versuch mit
    quantisierten Gewichten und F32-Aktivierungen kippt genauso, ein F32-Kopf hilft
    nicht: Ursache ist die Rundung der Gewichte, nicht unser Code. `TNativeLlmClient`
    lädt deshalb standardmäßig F32, Q8_0 ist eine Option. (Seit B15, 2026-10-19, ist
    Q8_0 die Vorgabe und F32 die Wahl für Tests und Vergleiche.)
- Gegenproben, die anschlagen: Q8-Zeilen verschoben, Attention ohne Skalierung,
  falsche KV-Kopf-Zuordnung, F32-Kachel zu kurz, Float-Attention im Exact-Modus,
  Referenzzweig mit einem Schlüssel zu wenig, Q8-Kopf gleich dem Embedding, fehlende
  NaN- und Bereichsprüfung, Bias-Abfrage weg, SIMD-Pfad bei null Eingaben.
**3b, erledigt 2026-10-15: Sampler** (`mormot.ai.infer.sampler`).

- Die Warper von Hugging Face in ihrer Reihenfolge: Temperature, Top-k, Top-p auf
  dem, was Top-k übrig lässt, Min-p; dann eine Ziehung. Top-k behält wie transformers
  auch die Tokens, die mit dem k-ten gleichauf liegen. Temperature 0 (und jede unter
  1e-30) ist exakt das bisherige Greedy (`InferArgMax`, aus dem Modell hierher
  umgezogen).
- Exakt ohne das Vokabular zu sortieren: Die Wahrscheinlichkeiten werden über alle
  Tokens summiert, dann sinkt eine Schwelle, bis alle Tokens, die die Filter behalten,
  darüber liegen; nur diese werden sortiert.
- Wiederholbar: ein eigener splitmix64-Generator. Derselbe Seed liefert auf jedem
  Compiler und jeder CPU denselben Zufallsstrom, also dieselben Tokens, solange die
  Logits gleich sind (SIMD und `Exp` können eine Ziehung genau auf einer Grenze
  verschieben). Der Test prüft eine unabhängig in Python berechnete Folge; FPC
  aarch64 und Delphi 7 ziehen sie gleich. Mit torch identische Ziehungen gibt es
  nicht (anderer Generator), nur dieselbe Verteilung.
- NaN, Unendlich und Einstellungen außerhalb des Bereichs werden abgelehnt, über
  ihre Bits (ein Vergleich mit NaN löst unter FPC `EInvalidOp` aus).
- Client: Die Vorgaben kommen aus `generation_config.json` des Modells, wie bei vLLM
  (Qwen3: 0,6 / Top-k 20 / Top-p 0,95; ohne Datei Greedy). Eine Anfrage überschreibt
  sie mit `Temperature` (0 = Greedy) und `Extra` `top_p`, `top_k`, `min_p`, `seed`,
  den Feldnamen von OpenAI und vLLM; ohne Seed zieht jede Anfrage einen zufälligen.
  Die Filter formen nur eine gesampelte Ziehung: Ohne `generation_config.json` sind
  die Vorgaben Greedy, und eine Anfrage mit nur `top_p` bleibt Greedy (vLLM würde dann
  mit Temperature 1 sampeln). Boolesche Werte gelten nicht als Zahl.
- Tests: Verteilungen über 70.000 Ziehungen gegen die exakten Wahrscheinlichkeiten
  (jeder Filter einzeln und kombiniert, Top-p gegen die ganze Masse), NaN und -Inf,
  die feste Seed-Folge, ein Vokabular von 151.936 Tokens gegen einen per Bisektion
  bestimmten Nucleus, `generation_config.json` und die Overrides; mit Qwen3-0.6B:
  derselbe Seed gibt dieselbe Antwort. Gegenproben (Top-p gegen die Kandidaten, Seed
  ignoriert, Temperature nicht angewandt, Top-k um eins daneben) schlagen an.

**3c, erledigt 2026-10-15: Prompt-Cache.**

- Ein Agent-Loop oder Chat schickt mit jeder Runde den ganzen Verlauf, eine Runde
  länger. Der KV-Cache einer Position hängt nur von den Tokens bis dahin ab; was am
  Anfang gleich ist, muss also nicht neu gerechnet werden.
- `TQwen3Context` merkt sich das Token jeder gecachten Position. `TQwen3Model.Reuse`
  behält den gemeinsamen Anfang mit dem neuen Prompt (höchstens alle Tokens bis auf
  das letzte, das für die Logits gerechnet wird; nichts aus dem Kontext eines anderen
  Modells) und vergisst den Rest.
- `NativeDecode` nimmt optional einen Kontext und meldet, wie viele Prompt-Tokens aus
  ihm kamen. `TNativePromptCache` hält die Kontexte fertiger Anfragen (thread-sicher;
  der mit dem längsten gemeinsamen Anfang gewinnt, der am längsten unbenutzte fliegt
  raus). `TNativeLlmClient.Cache` hat einen Platz; `Cache.Slots := 0` schaltet ab
  und gibt den Speicher frei. Der Kontext einer fehlgeschlagenen Anfrage wird
  verworfen, nicht zurückgelegt.
- Die Antwort bleibt dieselbe: Die Logits nach einem wiederverwendeten Anfang sind
  bitgleich zu einem frischen Durchlauf, auf dem Referenz- und dem schnellen Backend,
  auch mit Q8_0 (jede Zeile wird unabhängig von der Zahl der Zeilen gerechnet).
- Speicher: ein Kontext hält 2 × Layers × KvHeads × HeadDim Singles je Token seiner
  Kapazität, bei Qwen3-0.6B 224 KB je Token, also 896 MB für 4096 Tokens, dazu
  Zwischenpuffer für den längsten Prompt (etwa 60 KB je Token). Darum
  standardmäßig nur ein Platz.
- Messung (Snapdragon, F32): Der Agent-Loop des Tests (Tool-Aufruf, dann Antwort)
  braucht mit Cache 2,5 s statt 3,5 s; die zweite Runde übernimmt 153 Tokens.
- Tests: bitgleiche Logits für gemeinsamen Anfang, gleichen und kürzeren Prompt;
  `Truncate`, `Id`, ein fremdes Modell; der Pool (längster Anfang, LRU, Slots, Clear);
  `NativeDecode` im gehaltenen Kontext gibt dieselbe Antwort wie frisch; mit
  Qwen3-0.6B trifft die zweite Agent-Runde den Cache und antwortet ohne Cache gleich.
  Neun Gegenproben (Eigentümer ignoriert, ganzer Prompt behalten, IDs nicht
  gespeichert, Truncate nach oben, erster statt längster Treffer, LRU verkehrt,
  volle Liste neu angelegt, Slots kürzen verkehrt, Client ohne Cache) schlagen an.

**3d, erledigt 2026-10-15: Normen, RoPE, SwiGLU und das Wachsen des KV-Caches.**

- Gemessen vorher (Q8_0, 200 Tokens): 45 % des Prefills und 19 % des Dekodierens lagen
  außerhalb der Matrixprodukte und der Attention, in RMSNorm, RoPE und SwiGLU, die im
  Modell auf einem Thread liefen. `llm-bench` weist jetzt jede Operation einzeln aus.
- RMSNorm, SwiGLU und RoPE sind Backend-Operationen (`ioRmsNorm`, `ioSwiGlu`,
  `ioRope`). Die Referenz rechnet genau wie bisher das Modell; das schnelle Backend
  verteilt Zeilen bzw. Abschnitte auf den Pool, mit derselben Arithmetik. Damit bleibt
  alles bitgleich, in beiden Modi, und kein Gate ändert sich. SIMD lohnt hier nicht
  mehr: Zusammen bleiben die drei unter 10 %.
- RoPE: Cos und Sin rechnet `Forward` einmal je Position, statt je Layer und je Q und K
  (56-mal dasselbe bei Qwen3-0.6B).
- KV-Cache: Er wuchs beim ersten Wachsen auf genau die Prompt-Länge, sodass schon das
  erste dekodierte Token den ganzen Cache kopierte (bei 1000 Tokens rund 300 ms). Er
  wächst jetzt auf das Doppelte und mindestens ein Viertel über den Bedarf.
- Messung (Snapdragon, 8 Threads, ein Lauf; vorher am selben Tag gemessen):

  | | Prefill | Dekodieren | Speicher-Peak |
  |---|---|---|---|
  | F32, 200, vorher | 167 Tokens/s | 30 Tokens/s | 3,8 GB |
  | F32, 200 | 203 Tokens/s | 33 Tokens/s | 3,8 GB |
  | F32, 1000 | 138 Tokens/s | 24 Tokens/s | 4,1 GB |
  | Q8_0, 200, vorher | 282 Tokens/s | 53 Tokens/s | 2,1 GB |
  | Q8_0, 200 | 498 Tokens/s | 64 Tokens/s | 2,1 GB |
  | Q8_0, 1000 | 240 Tokens/s | 40 Tokens/s | 2,4 GB |

  Bei 1000 Tokens geht jetzt die Hälfte des Prefills in die Attention; sie ist der
  nächste Hebel bei langen Prompts.
- Tests: RMSNorm, SwiGLU und RoPE in allen Modi bitgleich zur Referenz (unter, an und
  über den Schwellen, in place) und die Referenz gegen die Formeln; die Qwen3-Gates
  unverändert grün. Gegenproben (Zeilen- und Abschnittsversatz, falsche RoPE-Zeile,
  Position ohne Vorgänger, SiLU ohne Single-Rundung) schlagen an.

**3f, macOS erledigt 2026-10-15: Benchmark auf dem Apple M2 Pro** (FPC 3.2.4, 7 Threads,
NEON-Dotprod; zwei Läufe, bis 12 % auseinander). Die Infer-Suite läuft dort mit allen
Modellen grün (2010 Prüfungen, wie auf dem Snapdragon).

| | Prefill | Dekodieren | Speicher-Peak |
|---|---|---|---|
| F32, 200 | 162–181 Tokens/s | 42 Tokens/s | 3,8 GB |
| Q8_0, 200 | 515–562 Tokens/s | 70–75 Tokens/s | 2,1 GB |
| F32, 1000 | 147–160 Tokens/s | 31–32 Tokens/s | 4,1 GB |
| Q8_0, 1000 | 371–408 Tokens/s | 44–47 Tokens/s | 2,4 GB |

Der M2 Pro dekodiert 9 bis 33 % schneller als der Snapdragon (vermutlich die höhere
Speicherbandbreite, nicht gemessen); `llm-bench` misst den Speicher-Peak jetzt auch unter macOS
(`getrusage`).

**3e-1, erledigt 2026-10-15: Q4_0 (experimentell).**

- `idtQ4_0`: 32 Werte je Block, eine FP16-Skala, Nibbles (18 Bytes). Der Quantisierer
  `InferQuantizeRowQ4_0` liefert die Bytes von ggmls `quantize_row_q4_0_ref` (d = der
  vorzeichenbehaftete erste Wert größten Betrags / −8, q = min(15, trunc(x/d + 8,5)),
  jeder Schritt in Single), `InferRowToF32` dequantisiert, der GGUF-Leser nimmt Q4_0
  (ggml-Typ 2) an.
- `Linear` mit Q4_0: Ein Q4_0-Block ist verlustfrei ein Q8_0-Block mit derselben Skala
  (q − 8). Das Referenz-Backend entpackt die Matrix, das schnelle je Kachel von 16
  Zeilen; danach rechnen die Q8_0-Kernel unverändert, `Exact` bleibt bitgleich.
- Qwen3: `TQwen3Model.Create(…, idtQ4_0)` quantisiert die Layer nach Q4_0, den Kopf nach
  Q8_0 (wie llama.cpp den Ausgabekopf feiner hält).
- Prüfung gegen ggml selbst (neues Referenzziel `q4` in `tools/infer-ref`): Bytes des
  Quantisierers (erster größter Betrag mit Vorzeichen, Summen genau auf .5 und ganzen
  Zahlen, Nullblock, winzige und große Werte, Kappung bei 15), Dequantisierung, Linear
  gegen `ggml_vec_dot_q4_0_q8_0`.
- Ergebnis auf Qwen3-0.6B: zu grob. Die Logits liegen bei Kosinus 0,76 bis 0,96 (Fehler
  bis 9,6), Greedy verlässt die Referenz teils schon beim zweiten Token, `chat_tool`
  ruft das Tool nicht mehr auf. Das Gate prüft deshalb nur Schranken, die ein kaputter
  Quantisierer weit verfehlt.
- **Offen: eigene Kernel.** Ohne sie dekodiert Q4_0 halb so schnell wie Q8_0 (31 statt 63
  Tokens/s, Snapdragon): Das Entpacken in Pascal je Kachel und Token kostet mehr, als die
  halbe Speicherbandbreite spart. ggml löst das mit fusionierten Kerneln (die Nibbles in
  den Registern entpackt und gleich multipliziert, NEON und AVX2) und auf ARM zusätzlich
  mit Repacking (Q4_0-Zeilen beim Laden zu 4×4 oder 8×4 verschränkt, ein Kernel rechnet
  4 oder 8 Zeilen). Sinnvoll vor allem für größere Qwen3-Modelle (4B, 8B), bei denen 4 Bit
  qualitativ reichen. Der Speicher sinkt erst mit fertig quantisierten GGUF-Dateien
  spürbar (heute 1,9 statt 2,1 GB, weil die BF16-Datei gemappt bleibt).
- *B2 (2026-10-17):* Das Entpacken war der Engpass, nicht die Rechnung. `InferQ4ToQ8`
  entpackt jetzt acht Nibbles je Int64, nur bitweise (`Q4Minus8`, auf x86, x64 und
  aarch64; sonst byteweise wegen unausgerichteter Zugriffe), dazu `Q4UnpackNeon` für
  jedes aarch64. Decode, Qwen3-0.6B Q4_0, 32 Tokens: Snapdragon X 30 → 57 (Pascal)
  → 70–78 Tokens/s (NEON), Q8_0 dort 58–66; M2 Pro 28,5 → 84. Ein fusionierter Kernel
  ist damit für das Ziel „≥ Q8_0“ nicht nötig. Auf dem i9-14900K reicht schon das
  Pascal-Entpacken: 64 → 79 Tokens/s, Q8_0 dort 71–73; ein AVX2-Entpacker ist
  zurückgestellt (höchstens etwa 1 ms je Token).
- Weiter (Entscheid 2026-10-15): 3e-2 Qwen3 direkt aus GGUF laden, 3e-3 Q4_K/Q6_K (die
  Mischung Q4_K_M) mit fusionierten NEON- und AVX2-Kerneln nach ggml-Vorbild.

**3e-2, erledigt 2026-10-15: Qwen3 direkt aus GGUF** (`TQwen3Model.CreateFromGguf`).

- Die Einstellungen kommen aus den Metadaten (`qwen3.block_count`, `…embedding_length`,
  `…attention.key_length` als Kopfbreite, `…context_length`, `…rope.freq_base`,
  `…layer_norm_rms_epsilon`), die Tensoren unter llama.cpps Namen (`token_embd`,
  `blk.N.attn_q` … `ffn_down`, `output_norm`, `output`). Fehlt `output.weight`, ist
  der Kopf an `token_embd` gebunden. Q8_0- und Q4_0-Matrizen werden direkt aus der
  gemappten Datei benutzt (jede FP16-Skala beim Laden auf endlich geprüft), F32 ebenso,
  F16 und BF16 auf F32 erweitert; das Embedding wird je Token aus dem Dateityp gelesen.
- Abgelehnt: eine andere Architektur, RoPE-Skalierung, Sliding Window, ein fehlender,
  zusätzlicher oder falsch geformter Tensor, eine unendliche Skala.
- Zwei Rundungsfunde dabei, beide gegen ggml selbst geprüft:
  - Q8_0-**Gewichte** quantisiert ein GGUF-Writer mit `quantize_row_q8_0_ref`:
    Gleichstände werden von der Null weg gerundet (`roundf`), bei den Aktivierungen
    zur geraden Zahl. Das Laden aus safetensors nutzte bisher die Aktivierungsregel,
    und BF16-Gewichte (8 Mantissenbits) treffen solche Gleichstände tatsächlich. Neu:
    `InferQuantizeRowQ8_0Ref` für Gewichte; das Q8_0-Gate liegt danach etwas näher an
    FP32 (`chat_tool`: Kosinus 0,9972 statt 0,9965 mit NEON).
  - Q4_0: Die gebauten ggml-Bibliotheken rechnen `x · (1/d) + 8,5` als fused
    multiply-add (GCC fasst es auf aarch64 und mit `-march=native` auf x86 zusammen),
    also mit einer Rundung statt zweien. Der Quantisierer rechnet jetzt ebenso (Produkt
    exakt in Double); mit zwei Rundungen wichen beim winzigen Modell einige Bytes ab.
- Tests ohne großes Modell: ein winziges Qwen3 als GGUF (`tiny-qwen3-gguf`, von ggmls
  Quantisierern geschrieben): Einstellungen wie `config.json`, jede Matrix byte-gleich
  zu unserer Quantisierung (Q8_0 und Q4_0), Logits gegen FP32, `Exact` bitgleich, F16
  mit eigenem `output.weight`, acht beschädigte Dateien abgelehnt.
- Mit Qwens eigener Datei (`Qwen/Qwen3-0.6B-GGUF`, `Qwen3-0.6B-Q8_0.gguf`, Revision
  `23749fe`, SHA-256 `9465e63a…`, als `vendor/models/qwen3-0.6b-q8_0.gguf`): Sie stammt
  aus anderen Gewichten als die BF16-Veröffentlichung unserer Referenz
  (`token_embd` ist byte-gleich zu ggmls Quantisierer darauf, in den Layern weichen die
  Skalen der meisten Blöcke und die Werte um einige Prozent ab). Geprüft wird deshalb:
  jede Matrix mit Kosinus ≥ 0,999 an ihren BF16-Gewichten (gemessen 0,9998; eine falsch
  zugeordnete läge nahe 0) und das Q8_0-Gate mit Greedy-Schwelle 1 statt 0,5 (`raw_fox`
  weicht an einem Schritt ab, an dem die Referenz 0,71 sicher war). Auch Qwens Datei ruft
  bei `chat_tool` das Tool nicht auf.
- Laden: unter Windows 0,4 s (die Datei wird gemappt, jede Skala einmal gelesen), in der
  WSL auf `/mnt/d` gut 4 s - statt etwa 13 s Quantisieren aus safetensors.
- Abgelehnt werden auch RoPE über nur einen Teil des Kopfes
  (`qwen3.rope.dimension_count`) und eine nicht endliche Skala im Embedding, auch wenn
  der Kopf nicht daran gebunden ist (Review-Fund).

**3e-2b, erledigt 2026-10-15: Tokenizer und Client aus einer GGUF-Datei.**

- `TBpeTokenizer.CreateFromGguf` liest `tokenizer.ggml.*`: Modell `gpt2` mit
  Pre-Tokenizer `qwen2` (die Split-Regex von Qwens `tokenizer.json`), die normalen
  Tokens (Typ 1) als BPE-Vokabular, Kontroll- und benutzerdefinierte Tokens (3, 4) als
  Added Tokens, ungenutzte (5, `[PAD…]`) als kein Token, die Merges als „a b“. GGUF
  speichert keinen Normalizer; NFC wird angewandt wie in Qwens `tokenizer.json`
  (llama.cpp selbst normalisiert nicht). Jeder andere Tokenizer, ein anderer
  Pre-Tokenizer oder ein unbekannter Token-Typ werden abgelehnt.
- `TNativeLlmClient.CreateFromGguf(Datei)`: Modell, Tokenizer und Template aus einer
  Datei; Modellname aus dem Dateinamen; Sampling Greedy, bis `Sampling` gesetzt wird
  (eine GGUF-Datei hat keine `generation_config.json`).
- Tests: der Tokenizer aus Qwens Q8_0-Datei gleicht dem aus `tokenizer.json` in jedem
  Token, jedem Added Token und allen 2416 Gate-Fällen; der Client aus der Datei antwortet
  auf `chat_en` „Paris“ mit derselben Prompt-Länge wie die Referenz und streamt gültiges
  UTF-8; Unigram-, `llama-bpe`- und Byte-Token-Tokenizer werden abgelehnt. Gegenproben
  (benutzerdefinierte Tokens verworfen, unbekannte Typen übergangen, Pre-Tokenizer
  ungeprüft, Vokabular um eins zu kurz) schlagen an.

**3e-3a, erledigt 2026-10-15: K-Quants Q4_K und Q6_K (Korrektheit).**

- Die Typen der verbreiteten Q4_K_M-Dateien: `idtQ4_K` (Superblock zu 256 Werten in 8
  Teilblöcken mit 6-Bit-Skala und -Minimum, 144 Bytes) und `idtQ6_K` (16 Gruppen mit
  8-Bit-Skala, 6-Bit-Werte, 210 Bytes); der GGUF-Leser nimmt sie an (ggml-Typen 12, 14).
- `InferRowToF32` dequantisiert sie mit den Single-Schritten von ggmls
  `dequantize_row_*` (bitgleich geprüft). Aktivierungen werden wie bei ggml als Q8_K
  quantisiert (`InferQuantizeRowQ8_K`: Skala aus dem ersten größten Wert mit Vorzeichen,
  Gleichstände zur geraden Zahl, Blocksummen je 16).
- `Linear`: je Teilblock exakte Ganzzahlsummen, bei Q4_K die Minima über die
  Blocksummen der Aktivierungen; die Skalen in Double. Referenz und schnelles Backend
  rechnen gleich (das schnelle verteilt die Kacheln auf den Pool); SIMD-Kernel folgen in
  3e-3b.
- Das Modell nutzt Q4_K- und Q6_K-Matrizen aus GGUF direkt, die Skalen (bei Q6_K am
  Blockende) beim Laden geprüft.
- Datei: `unsloth/Qwen3-0.6B-GGUF`, `Qwen3-0.6B-Q4_K_M.gguf`, Revision `50968a4`,
  SHA-256 `ac2d9771…`, als `vendor/models/qwen3-0.6b-q4_k_m.gguf`: Q4_K, Q6_K für
  `token_embd` (also den gebundenen Kopf) und je die Hälfte von `attn_v` und
  `ffn_down`, mit Importance-Matrix quantisiert. Lädt in 2,9 s; jede Matrix mit Kosinus
  ≥ 0,997 an den BF16-Gewichten.
- Gate gegen FP32: Kosinus 0,867 (`chat_tool`) bis 0,990, Fehler bis 9,0 - deutlich
  besser als Q4_0 (0,66 bis 0,96), aber Greedy weicht auch hier ab, wo die Referenz
  1,37 sicher war (`chat_en` antwortet „Paris.“ mit Punkt).
- Tests gegen ggml selbst (Referenzziel `qk`): Q8_K-Bytes, Q4_K- und Q6_K-Dequantisierung
  bitgleich, `Linear` gegen `ggml_vec_dot_q4_K_q8_K` / `_q6_K_q8_K`; zufällige Blöcke:
  `Exact` und Float bitgleich zur Referenz, nahe F32 auf den dequantisierten Zeilen.
  Gegenproben (Skala/Minimum von Teilblock 4-7, vertauschte Nibbles, Minima addiert,
  Q6_K-Hochbits, Q8_K-Gleichstände und erster Größter) schlagen an.

**3e-3b, erledigt 2026-10-16: K-Quant-Kernel (NEON, AVX2).**

- Befund (ein Thread, `perf`): Mit SIMD nur für die Teilsummen gingen 46 % der Zeit in
  das Pascal-Fertigrechnen (6-Bit-Skalen je Teilblock auspacken, multiplizieren), 10 % in
  Half-Float-Aufrufe, 22 % in die Kernel.
- Fusionierte Kernel wie bei ggml: je Superblock liefern sie zwei exakte int32, bei
  Q4_K `isum` (Σ Skala · Teilsumme) und `msum` (Σ Minimum · Blocksummen der
  Aktivierungen), bei Q6_K `isum` (Σ Skala · (q − 32) · x über die Gruppen). Die Skalen
  werden mit ggmls Masken in einem Zug ausgepackt. NEON: SDOT auf den Nibbles bzw. den
  aus `ql`/`qh` gebauten 6-Bit-Werten (Opcodes aus dem GNU-Assembler); AVX2:
  `vpmaddubsw` (u8 × s8) und `vpmaddwd` gleich mit der Skala, vollständig ausgerollt,
  nur ymm0..ymm5. Pascal multipliziert danach nur noch zwei Half-Float-Skalen je Block
  (aus der Tabelle). Weil die Ganzzahlen exakt sind, bleibt `Exact` bitgleich zur
  Referenz; die Referenz rechnet die Teilsummen weiter ausführlich in Pascal.
- FPC 3.2 kennt `vmovq`/`vpmovzxbw` mit 64-Bit-Speicheroperand nicht: geladen wird
  über ein Allzweckregister.
- Messung (Snapdragon X, 8 Threads, ein Lauf):

  | | Prefill | Dekodieren | Speicher-Peak |
  |---|---|---|---|
  | Q4_K_M-GGUF, 200 | 290 Tokens/s | 73 Tokens/s | 475 MB |
  | Q4_K_M-GGUF, 1000 | 178 Tokens/s | 44 Tokens/s | 744 MB |
  | Q8_0-GGUF, 200 | 545 Tokens/s | 65 Tokens/s | 707 MB |
  | Q8_0 aus safetensors, 200 | 457 Tokens/s | 66 Tokens/s | 2068 MB |

  Q4_K_M dekodiert damit schneller als Q8_0 (halbe Bandbreite). Im Prefill liegt es
  noch zurück: Der Kernel rechnet eine Aktivierungszeile je Aufruf und packt dieselben
  Gewichte für jede Zeile neu aus (offen: mehrere Zeilen je Kernel).
- Tests: Kernel gegen Pascal bitgleich (auch 513 Blöcke, über die Stückelung zu 512
  hinaus), `Exact` gleich Referenz, auf aarch64 (NEON) und unter x86-64-Emulation
  (AVX2, FPC 3.2.2 Win64 und Delphi 13). Gegenproben (Skalen auf 4 Bit maskiert,
  16 statt 32 · Blocksumme, je in NEON und AVX2; hohe Nibbles gegen die falschen
  Aktivierungen; Minima addiert; Stückelung um Blöcke statt Bytes) schlagen an.

- Nativ auf x86-64 (i9-14900K, Windows 11, FPC 3.2.2, 22 Threads, 2026-10-16): Delphi 13
  (2255) und Delphi 7 (1401) grün; `llm-bench` 200 Tokens: Q8_0-GGUF 757 / 72 Tokens/s,
  Q4_K_M 455 / 78 Tokens/s (Prefill / Decode), 1000 Tokens: Q8_0 540 / 44, Q4_K_M 369 / 47.
  `llm-bench` schaltet unter Windows die EcoQoS-Drossel ab (wie `embed-bench`).

- Offen in Phase 3, als agentenfertige Einträge in [backlog.md](backlog.md): K-Quant-Prefill
  (B1, ein Versuch ohne eigenen Kernel war langsamer), Q4_0-Kernel (B2); beide
  erledigt 2026-10-17.
- *B1, erster Teil (2026-10-16):* `TInferKSums2` und `Q4KSums2NeonDot`.
  - Eine Q4_K-Gewichtszeile läuft gegen zwei Q8_K-Tokenzeilen. Nibbles, Skalen und Minima
    werden pro Block einmal ausgepackt, die SDOTs laufen gegen beide Zeilen.
  - `LinearTileWork` nimmt die Tokenzeilen paarweise.
  - `DotQ4K2` rechnet jedes Ergebnis in der Double-Arithmetik von `DotK`, `Exact` bleibt
    bitgleich.
  - Qwen3-0.6B Q4_K_M, Prefill bei 200 Tokens: Snapdragon X 275 → 299 Tokens/s (drei
    A/B-Paare), M2 Pro 271 → 294.
  - Q6_K genauso (`Q6KSums2NeonDot`, 2026-10-17): die 6-Bit-Werte und Gruppenskalen
    einmal je Halbblock für beide Zeilen; `DotK2` rechnet beide K-Typen. Prefill bei
    200 Tokens: Snapdragon X 291 → 311 Tokens/s, M2 Pro 291 → 305.
  - AVX2 (2026-10-17): `Q4KSums2Avx2Asm` und `Q6KSums2Avx2Asm`, Nibbles bzw. 6-Bit-Werte
    und Skalen-Broadcast einmal für beide Zeilen (ymm0..ymm7, xmm6/7 unter Win64 im
    X-Bereich gesichert). i9-14900K, Q4_K_M-Prefill bei 200 Tokens: 462 → 579 Tokens/s.
    Der SysV-Einsprung (Linux x86_64) ist nur statisch geprüft.

- *B16 erledigt (2026-10-19): Q5_0.* llama.cpp schreibt Q5_0, wo eine K-Quant-Zeile kein
  Vielfaches von 256 ist: Qwens `qwen2.5-0.5b-instruct-q4_k_m.gguf` (Hidden 896) hat 133
  Q5_0-Tensoren, das Embedding eingeschlossen, daneben nur 12 Q4_K, 12 Q6_K und 13 Q8_0.
  - `idtQ5_0` und `TInferBlockQ5_0` (22 Bytes: FP16-Skala, 32 fünfte Bits als
    Little-Endian-Wort, 32 Nibbles wie Q4_0; Wert = d · (q − 16)); `InferRowToF32`,
    GGUF-Typ 6, `INFER_BLOCK_DTYPES` als gemeinsame Liste der Blocktypen, die der Decoder
    an Ort und Stelle liest.
  - CPU: `InferQ5ToQ8` packt verlustfrei nach Q8_0 aus (q − 16 passt in ein int8), wie
    Q4_0 je Kachel; danach dieselben Q8_0-Kernel, `Exact` bitgleich mit der Referenz.
    Ohne eigenen SIMD-Code: 8 Werte je Int64, die fünften Bits über eine Tabelle von
    256 Einträgen.
  - Vulkan (`linear8.comp` auch mit `-DIDS`, `gemm8.comp`, `linear.comp`) und CUDA
    (`linear`, `linear8`, `gemm8`, `linear8ids`): `dtype` 5, die Gewichte im Shader
    ausgepackt wie die anderen Typen. `matvec8.comp` kennt Q5_0 nicht, ein Decode-Schritt
    nimmt dort `linear8`. Die Adreno X1-85 rechnete die fünften Bits aus dem ganzen
    `qh`-Wort, um j verschoben, falsch (NVIDIA richtig); gelesen wird deshalb das eine
    Byte von `qh`, das die acht Werte brauchen - auf beiden Backends gleich.
  - *Gemessen* auf der RTX 5090 (i9-14900K), Qwen2.5-0.5B-Instruct, 200 Tokens Prompt, 64
    neue, Qwens Q8_0-Datei gegen seine Q4_K_M-Datei:

    | | Q8_0 | Q4_K_M (meist Q5_0) |
    |---|---|---|
    | Speicher-Peak CPU | 718 MB | 542 MB |
    | CPU Prefill / Decode | 931 / 98 Tokens/s | 891 / 101 Tokens/s |
    | CUDA Decode | 462 Tokens/s | 447 Tokens/s |
    | Vulkan Decode | 418 Tokens/s | 386 Tokens/s |

    Der Nutzen ist, dass die Datei überhaupt lädt, und ein Viertel weniger Speicher;
    schneller wird ein so kleines Modell nicht (der Kopf bleibt Q8_0, 136 Mio. der 494
    Mio. Parameter). Vulkan verliert 8 %, weil `matvec8` fehlt.
  - *Tests:* `Q5LikeGgml` (Datensatz `q5` aus ggml selbst: Blöcke von
    `quantize_row_q5_0_ref` mit jedem q von 0 bis 31 samt Clip, Nullblock, winzigen und
    großen Zeilen, einem wiederholten Wert, lauter gelöschten fünften Bits, exakt dequantisiert wie
    `dequantize_row_q5_0`; Linear wie `ggml_vec_dot_q5_0_q8_0`), `LinearQ5MatchesReference`
    (Auspacken gegen die Definition für alle Nibble-Bytes und fünften Bits, `Exact` bitgleich,
    896 und 4352 Eingänge), auf CUDA und Vulkan `LinearMatchesReference` mit Q5_0 (auch 32,
    96 und 896 Eingänge, ungerade Blockzahlen) und `LinearIdsMatchesReference` (F32, Q8_0,
    Q5_0, Q4_K über Experten-Ids, Ids doppelt, x je Platz und gemeinsam - von Codex als
    Lücke gemeldet), `Qwen25GgufMatchesCpu` auch mit Q4_K_M (Logits-Kosinus 0,9986 auf der
    RTX 5090, 0,9988 auf der Adreno). `Qwen25Gguf` lädt Q4_K_M: jede Matrix gegen Qwens
    Q8_0-Datei (niedrigster Kosinus 0,9972, eine Q4_K-Matrix), nicht gegen die
    BF16-Gewichte - beide GGUF-Dateien stammen aus einem anderen Checkpoint als
    `model.safetensors` (schon Q8_0 kommt bei Layer 9 `k_proj` nur auf 0,81; ggml's eigenes
    `dequantize_row_q5_0` liefert dieselben Werte wie unseres). Gegen die FP32-Referenz:
    Logits-Kosinus 0,968 bis 0,991, größter Fehler 3,9.
  - Grün auf dem RTX-Rechner unter FPC und Delphi 13 (4786 Assertions) und Delphi 7 (3169),
    auf der Adreno X1-85 (Vulkan-Suite 787) und dem M2 Pro (MoltenVK, 4025). Gegenprobe:
    12 eingebaute Fehler (fünftes Bit der oberen Hälfte, falsches `qh`-Byte, Vorzeichen
    in `Q5Minus16`, Q5_0-Kacheln als Q4_0 ausgepackt; auf CUDA und Vulkan hohe Nibbles nie,
    fünftes Bit verschoben, falsches `qh`-Byte, Zeilen zu 20 bzw. 24 Bytes, Nibbles ab dem
    falschen Byte) - alle scheitern. Review: Claude-Finder ein veralteter Kommentar, Codex
    die zwei Testlücken oben; behoben.
  - *Offen, nur bei Bedarf:* Q5_1 (llama.cpp nimmt es als Ersatz für Q5_K, Q5_K_M-Dateien
    brauchen also zuerst Q5_K); `matvec8.comp` für Q5_0; Auspacken mit NEON/AVX2.

### Phase 4 — GPU: Vulkan zuerst, CUDA danach

Entscheidung 2026-10-15: **Vulkan zuerst** (herstellerübergreifend: NVIDIA, AMD, Intel,
Qualcomm Adreno; unter Linux zusätzlich lavapipe als CPU-Treiber für Tests), **CUDA bleibt
zweites Backend** (Backlog B3). Für beide gilt: **keine mitgelieferten DLLs** — zur
Laufzeit wird nur der Loader des Treibers geladen (`vulkan-1.dll`, `libvulkan.so.1`).

- **Bindung:** eigene schlanke Unit nur mit den benötigten Typen und Funktionen
  (Instanz, Gerät, Queue, Puffer, Speicher, Shader-Modul, Compute-Pipeline,
  Descriptor-Sets, Command-Buffer, Fence), geschrieben nach den Khronos-Headern
  (`vulkan_core.h` / `vk.xml`, Lizenz „Apache-2.0 OR MIT", genutzt wird der MIT-Zweig).
  PasVulkan (`src/Vulkan.pas`) dient nur zum Nachschlagen. Grund für die eigene Unit:
  Delphi 7 bis 13 und FPC, kein 40 000-Zeilen-Generat im Repo.
- **Shader:** GLSL-Compute-Shader unter `tools/vk/`, beim Bauen mit `glslangValidator`
  nach SPIR-V übersetzt und als Pascal-Konstanten eingecheckt (generierte `.inc`). Der
  Compiler wird nur zum Erzeugen gebraucht, nicht zur Laufzeit und nicht für den Build.
- **Tests:**
  - lavapipe unter WSL: Korrektheit ohne GPU, läuft in jeder FPC-Suite mit.
  - RTX 5090 auf dem RTX-Rechner: Leistung.
  - Apple M2 Pro über MoltenVK.
  - Ohne Vulkan-Loader überspringen die Tests.
- **Apple-GPU (M2):** Apple bietet nativ nur Metal, kein Vulkan.
  - *Erster Schritt, erledigt 2026-10-16:* dasselbe Vulkan-Backend über MoltenVK (Vulkan
    auf Metal, Apache-2.0).
    - *Instanz:* Sie fordert `VK_KHR_portability_enumeration` an, wenn der Loader die
      Erweiterung meldet, und setzt dazu `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR`.
    - *Gerät:* Es aktiviert `VK_KHR_portability_subset`, wenn es die Erweiterung meldet
      (`TInferVulkan.Portability`).
    - *Loader:* Er kommt vom System, nicht aus dem Repo: Vulkan SDK oder Homebrew
      (`brew install molten-vk vulkan-loader`). `dlopen` mit bloßem Namen durchsucht
      `/opt/homebrew/lib` nicht, daher versucht `TInferVulkan` auf macOS diesen Pfad als
      Zweites.
    - *Getestet:* auf dem M2 Pro mit MoltenVK 1.4.2 und Loader 1.4.363.0. Die
      Vulkan-Suite ist mit `LAI_REQUIRE_VULKAN=1` grün (181 Assertions, auch Qwen3-0.6B
      und `TNativeLlmClient` auf dem Gerät). Die Subgroups haben 32 Lanes, der
      Subgroup-Pfad von `linear8` läuft also.
  - *Metal-Argumentpuffer aus (2026-10-16):* MoltenVK nutzt standardmäßig Metal-Argumentpuffer
    für Deskriptorsätze.
    - Dann beendet jede Pipeline-Barriere den Metal-Compute-Encoder.
    - Eine kleine Sonde misst je Dispatch mit Barriere rund 25 µs, ohne Argumentpuffer
      rund 5 µs.
    - `TInferVulkan` schaltet die Argumentpuffer deshalb über `VK_EXT_layer_settings` ab:
      Einstellung `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS` = 0 für den Layer `MoltenVK`.
      Das geschieht nur zusammen mit der Portability-Instanz und nur, wenn die
      gleichnamige Umgebungsvariable nicht gesetzt ist.
    - Unsere Shader binden höchstens acht Puffer und brauchen die Argumentpuffer nicht.
    - Im A/B-Test (je drei Läufe, 200 Tokens) stieg der Decode von 62,3 auf 65,8 Tokens/s
      (+5,6 %); der Prefill sank von 306 auf 303 Tokens/s (−1,0 %).
    - Die Tabelle unten ist vor dieser Änderung gemessen.
  - *Zeitstempel:* MoltenVK setzt auf Apple-GPUs nur zwischen Metal-Encodern einen
    Zeitstempel. Alle Stempel eines Batches tragen deshalb denselben Wert.
    - `TInferVulkan` erkennt das nur auf einem Portability-Gerät: Zwei aufeinanderfolgende
      Stempel mit demselben Wert genügen, auch über die Command Buffer von `BatchFlush`
      hinweg. Danach meldet `HasTimestamps` false, und `Profile` misst nichts mehr.
    - Ein natives Gerät entscheidet das nie, weil ein Tick dort lang sein kann
      (Codex-Review).
    - Der Test akzeptiert das nur auf einem Portability-Gerät.
  - *Leistung (Qwen3-0.6B Q8_0, `llm-bench`, 64 neue Tokens, Tokens/s):*

    | Kontext | Prefill GPU | Prefill CPU | Decode GPU | Decode CPU |
    |---|---|---|---|---|
    | 200 | 305 | 563 | 63 | 73 |
    | 1000 | 275 | 409 | 53 | 46 |
    | 4000 | 194 | 145 | 34 | 21 |

    GPU heißt MoltenVK auf dem M2 Pro, CPU heißt NEON-SDOT mit 7 Threads.
    - Bei kurzem Kontext ist die GPU langsamer als die CPU.
    - Ab etwa 1000 Tokens dekodiert sie schneller, bei 4000 Tokens auch der Prefill.
    - Der Prefill liegt weit unter dem, was die GPU rechnen könnte. Die Kernel sind auf
      die RTX abgestimmt (Backlog B8).
  - *Kernel-Sonde (Wanduhr, ein Kernel 50-mal je Batch, Q8_0):*
    - Decode-Linears (eine Zeile) erreichen 40 bis 57 GB/s Gewichtsbandbreite, der LM-Head
      76 GB/s. Der M2 Pro hat rund 200 GB/s.
    - Der Prefill-Linear (200 Zeilen) erreicht rund 400 GFLOPS. Der M2 Pro schafft in FP32
      rund 6,8 TFLOPS.
    - Der Subgroup-Build von `linear8` ist auf Apple im Prefill langsamer (rund 260 GFLOPS).
  - *Decode-Kernel `matvec8.comp` (2026-10-16):* ein eigener Kernel für eine Tokenzeile mit
    Q8_0-Gewichten.
    - Je 32 Invocations rechnen eine Ausgabe, und jede Invocation nimmt ganze
      34-Byte-Blöcke. Die Skala kommt aus einem Wort; die 32 Quants kommen aus 8 Wörtern,
      bei einem Block an Adresse 0 mod 4 per 16-Bit-Verschiebung.
    - Er hat einen Subgroup-Build wie `linear8`.
    - Der LM-Head steigt in der Sonde von 76 auf rund 120 GB/s.
    - Qwen3-0.6B Q8_0 dekodiert auf dem M2 Pro mit 90,2 statt 65,8 Tokens/s bei 200
      Tokens Kontext, mit 68,2 statt 53,6 bei 1000 und mit 40,0 statt 34,3 bei 4000. Die
      GPU dekodiert damit bei jeder Länge schneller als die CPU (73, 46, 21).
    - Vier statt einer Ausgabe je Gruppe waren 4 % langsamer (weniger Workgroups).
    - Auf der RTX 5090 ist der Kernel 5 % langsamer als `linear8`. Mit vier Ausgaben je
      Gruppe waren es 10 %.
    - `TInferVulkanBackend.UseMatvec` ist deshalb nur auf Apple-GPUs (Vendor-ID `$106B`)
      voreingestellt.
    - `LinearMatchesReference` prüft auf jedem Gerät beide Kernel. Die Gegenprobe (Skala aus
      der falschen Worthälfte) lässt den Test scheitern.
  - *`matvec8.comp` für K-Quants (2026-10-16):*
    - *Einheit:* Jede Invocation nimmt Einheiten aus 32 Gewichten: bei Q8_0 einen Block, bei
      Q4_K einen Unterblock, bei Q6_K ein Viertel einer Hälfte unter zwei Skalen.
    - *Lesen:* Die Skalen liest sie einmal je Einheit, die Quants als ganze Wörter.
      Q8_0- und Q6_K-Blöcke sind nur 2-Byte-ausgerichtet. Liegt ein Wert auf 2 mod 4,
      setzt `word2` ihn verzweigungsfrei aus zwei Wörtern zusammen.
    - *Verzweigung:* Mit einer echten Verzweigung dekodierte Q8_0 nur 46 Tokens/s, weil
      Nachbar-Invocations abwechselnd beide Fälle nehmen.
    - *Eigener Build je Typ:* Der Shader wird pro Gewichtstyp übersetzt (`-DDTYPE`), sechs
      Varianten mit den Subgroup-Builds. Ein Shader für alle drei Typen kostete Register;
      Q8_0 dekodierte damit 88 statt 98 Tokens/s.
    - *Qwen3-0.6B auf dem M2 Pro, Decode bei 200 Tokens Kontext (Tokens/s):*

      | Format | vorher | jetzt | CPU (NEON) |
      |---|---|---|---|
      | Q8_0 | 90,2 | 98,3 | 73 |
      | Q4_K_M | 56 (`linear8`) | 83,6 | 75 |

      Bei 1000 Tokens dekodiert Q8_0 mit 73,0 statt 68,2.
    - *Gegenproben:* Q4_K ohne die oberen Bits des Minimums, Q6_K mit falscher zweiter
      Skala und `word2` ohne Verschiebung lassen den Test scheitern.
  - *Prefill-Kernel `gemm8.comp` (2026-10-16):* ein gekachelter Kernel ab 16 Tokenzeilen
    (`GEMM_ROWS`), für alle Gewichtstypen von `linear8`.
    - *Kachel:* Eine Workgroup aus 256 Invocations rechnet 64 Zeilen × 64 Ausgaben.
    - *Shared Memory:* Je Schritt von 32 Eingaben liegen die Aktivierungen und die
      dequantisierten Gewichte dort als `vec4`, zusammen 16 KB.
    - *Rechnung:* Jede Invocation addiert daraus 4 Zeilen × 4 Ausgaben. Ein geladener Wert
      dient so 64 Produkten statt 8.
    - *Sonde:* Der Prefill-Linear steigt auf dem M2 Pro von rund 400 auf 1300 GFLOPS.
    - *Qwen3-0.6B Q8_0, Prefill (Tokens/s):*

      | Gerät | 200 | 1000 | 4000 |
      |---|---|---|---|
      | M2 Pro vorher | 303 | 272 | 193 |
      | M2 Pro mit `gemm8` | 1125 | 889 | 381 |
      | M2 Pro CPU (NEON) | 563 | 409 | 145 |
      | RTX 5090 vorher | 10 000 | 9 500 | 7 150 |
      | RTX 5090 mit `gemm8` | 13 600 | 14 550 | 9 840 |

      Der Decode bleibt unverändert.
    - `TInferVulkanBackend.UseGemm` ist überall voreingestellt, wo das Gerät Workgroups
      mit 256 Invocations erlaubt. Vulkan garantiert nur 128, ohne diese Möglichkeit bleibt es
      bei `linear8` (Codex-Review).
    - `LinearMatchesReference` prüft `gemm8` und `linear8` für viele Zeilen auf jedem
      Gerät, mit einer Form über mehrere Kacheln in beiden Richtungen (70 × 512 × 130).
    - Die Gegenprobe (falsche Zeile im Akkumulator) lässt 30 Assertions scheitern.
  - *Prefill-Attention `flash.comp` (2026-10-16):* Die Sonde maß `attention.comp` auf dem
    M2 Pro mit 220 GFLOPS im Prefill (256 Zeilen gegen 4000 Schlüssel, 37 ms je Schicht)
    und mit 63 GFLOPS im Decode bei 4000 Tokens.
    - *Ursache:* Eine Workgroup bedient nur 8 Queries, und im Score-Teil liest jede
      Invocation einen eigenen Schlüssel (512 Byte Abstand zur Nachbarin).
    - *Aufbau:* `flash.comp` hält bis zu 32 Queries im Shared Memory. Es lädt Schlüssel und
      Werte in Kacheln zu 16 gebündelt und führt die Online-Softmax pro Query.
    - *Ausgabe:* Sie ist dieselbe wie bei `attention.comp`, einschließlich der Splits für
      `attnmerge.comp`.
    - *Voraussetzungen:* Kopfdimension bis 128, 256 Invocations, 27 KB Shared Memory.
    - *Sonde:* 580 statt 220 GFLOPS.
    - *Qwen3-0.6B Q8_0, Prefill auf dem M2 Pro:* 1254 statt 1125 Tokens/s bei 200 Tokens,
      1197 statt 889 bei 1000, 708 statt 381 bei 4000.
    - *RTX 5090:* Dort ist der Kernel langsamer (5840 statt 9880 bei 4000 Tokens).
      `TQwen3Vulkan.UseFlash` ist deshalb nur auf Apple-GPUs voreingestellt.
    - *Test:* `AttentionRangesMatchCpu` prüft beide Kernel mit und ohne Splits, auch unter
      lavapipe (Tiny-Qwen3, Kopfdimension 32).
    - *Gegenprobe:* Ohne Neuskalierung der Summen scheitert der Test.
    - Offen bleibt die Decode-Attention bei langem Kontext (63 GFLOPS).
  - *Zweiter Schritt, nur wenn MoltenVK spürbar bremst:* ein eigenes Metal-Backend. Das wäre
    ein Objective-C-Laufzeit-Aufruf über `objc_msgSend` (kein Xcode-Projekt, keine
    mitgelieferte Bibliothek), mit Shadern in MSL statt GLSL.

Stufen:

- **4a** Loader, Instanz, Gerätewahl (diskret vor integriert vor CPU), Puffer, eine
  Compute-Pipeline; Test: ein trivialer Shader gegen die CPU.
- **4b** `Linear` auf der GPU (F32, Q8_0, Q4_K): Gewichte einmal hochgeladen und
  gecacht, Aktivierungen je Aufruf hin und zurück — als Differenztest gegen die
  Referenz, noch kein Tempogewinn beim Dekodieren (Übertragung und Fence je Aufruf).
- **4c** Norm, RoPE, SwiGLU, Softmax, Attention mit KV-Cache auf der GPU.
- **4d** der ganze Qwen3-Forward auf der GPU, Tensoren bleiben dort; nur Tokens hinein,
  Logits heraus. Erst hier zählt die Leistung.

**4a erledigt (2026-10-16):** `src/infer/mormot.ai.infer.vulkan.pas` (Bindung, Gerätewahl,
Puffer in host-sichtbarem Speicher, Compute-Pipeline, `Run` mit Fence), Shader-Werkzeug
`tools/vk/build-shaders.sh` (glslangValidator, ohne Root installierbar per
`apt-get download glslang-tools` + `dpkg -x`), Test `TTestInferVulkan`: Summe auf dem
Gerät bitgleich zur CPU. Grün auf lavapipe (WSL, FPC 3.2.4) und auf der RTX 5090 mit
Delphi 13 (Win64) und Delphi 7 (Win32). Quellversionen in `UPSTREAM_BASE`.

- Falle: FPC und Delphi schalten Gleitkomma-Ausnahmen scharf; die Threads, die ein
  Treiber startet (lavapipe/LLVM), erben das, und ein Trap darin legt den Prozess still
  (Futex-Hänger in `vkCreateDevice`). Jeder Einstieg in den Treiber maskiert deshalb alle
  Ausnahmen und stellt sie danach wieder her; unter Delphi Win32 zusätzlich MXCSR.
- Struktur-Offsets in `VkPhysicalDeviceLimits` sind aus dem gepinnten `vulkan_core.h`
  berechnet (nicht aus dem Gedächtnis); 32-Bit-Layout auf der RTX 5090 mit Delphi 7 belegt.
- `LAI_VULKAN_DEVICE=<Index>` wählt ein Gerät, `LAI_REQUIRE_VULKAN=1` macht das SKIP
  ohne Gerät zum Fehler.

**4b erledigt (2026-10-16):** `TInferVulkanBackend` (`src/infer/mormot.ai.infer.backend.vulkan.pas`)
ist das schnelle CPU-Backend mit `Linear` auf dem Gerät für F32, Q8_0, Q4_K und Q6_K
(`tools/vk/linear.comp`: entpackt je Element wie `InferRowToF32`, rechnet in float gegen
F32-Aktivierungen). Gewichte und Bias werden beim ersten Aufruf hochgeladen und bleiben
auf dem Gerät (Schlüssel: Adresse, Typ, Größe), Aktivierungen gehen je Aufruf hin und
zurück; ohne Shader-Typ, über dem Gerätelimit oder bei belegtem Gerät rechnet die CPU.
Gerätespeicher mit Staging-Puffer (diskrete GPU) bzw. direkt gemappt (integrierte GPU,
lavapipe). Test: relativer Fehler ≤ 1e-5 gegen die Referenz-F32 der entpackten Gewichte,
auch > 65535 Ausgaben, Cache bitgleich; grün auf lavapipe und RTX 5090 (Delphi 13, Delphi 7).

- Falle: Puffergrößen auf ganze 4-Byte-Wörter runden - der Shader liest Bytes über eine
  `uint`-Sicht, und das letzte Wort von 210-Byte-Q6_K-Blöcken lag sonst außerhalb (der
  Treiber liefert dort 0).
- Falle (Testdaten): zufällige Bytes als FP16-Skalen enthalten signalisierende NaNs; x87
  (Delphi 7) wirft darauf `EInvalidOp` - Testgewichte immer mit endlichen Skalen erzeugen.
- Noch kein Tempogewinn: jeder Aufruf wartet auf das Gerät. Das kommt mit 4c/4d.

**4c/4d erledigt (2026-10-16):** `TQwen3Vulkan` (`src/infer/mormot.ai.infer.model.qwen3.vulkan.pas`)
rechnet den ganzen Qwen3-Forward auf dem Gerät: RMSNorm, Projektionen, Q/K-Norm, RoPE,
kausale Attention (GQA) über einen KV-Cache auf dem Gerät, SwiGLU, Residuals, Final-Norm und
Kopf - als **ein** aufgezeichneter Batch je Chunk (bis 256 Tokens), also ein Warten je
Token. Embeddings und RoPE-Faktoren gehen über einen gemappten Puffer hinauf, die Logits
herunter. Shader `tools/vk/{rmsnorm,rope,swiglu,attention}.comp`, Batch-API in
`mormot.ai.infer.vulkan` (Deskriptor-Sets aus einem je Batch zurückgesetzten Pool).
Tests: Tiny-Qwen3 (F32 und Q8_0-GGUF) gegen die CPU-Referenz über 300 Tokens (zwei Chunks),
Decode, Reuse, ClearWeights, Greedy; Qwen3-0.6B Q4_K_M/Q8_0 gegen das schnelle CPU-Backend
mit erzwungener Tokenfolge: Logits-Kosinus ≥ 0,997 bzw. 0,999, Argmax 16 von 16 (RTX 5090).

RTX 5090 (Treiber 616.64, Windows 11), `llm-bench … vulkan`, naive Shader:

| | Prefill 200 | Decode 200 | Prefill 1000 | Decode 1000 | CPU i9 (Prefill/Decode 200) |
|---|---|---|---|---|---|
| Q4_K_M | 1001 Tok/s | 121 Tok/s | 954 Tok/s | 91 Tok/s | 455 / 78 Tok/s |
| Q8_0 | 1612 Tok/s | 126 Tok/s | 1507 Tok/s | 94 Tok/s | 757 / 72 Tok/s |

- Weit unter dem, was die Karte kann (0,6B in Q8_0 wären bandbreitenbegrenzt deutlich über
  1000 Tok/s Decode): `linear.comp` liest Bytes einzeln und entpackt je Element, `attention.comp`
  rechnet eine Arbeitsgruppe je Kopf und Zeile seriell über die Schlüssel. **Nächster Schritt
  4e:** schnelle Shader (vektorisierte Loads, Blöcke statt Elemente, Subgroup-Reduktionen,
  Matrix-Kacheln für Prefill, Attention mit geteilten Schlüssel-Blöcken), dann `TNativeLlmClient`
  auf Wunsch über die GPU.
- Grenzen werden vorab geprüft: Chunk-Zeilen begrenzt durch die Logits aller Zeilen und die
  65535 Arbeitsgruppen je Dimension; KV-Kapazität auf `maxStorageBufferRange` geklemmt. (Der
  Scores-Puffer, der anfangs die Kontextlänge begrenzte, entfiel mit der Flash-Attention in 4e.)

**4e, erster Teil (2026-10-16):** gemessen mit GPU-Zeitstempeln (`LAI_VULKAN_PROFILE=1` in
`llm-bench … vulkan`: GPU-Zeit je Operation, Host-Zeit für Aufzeichnen und Warten), dann:

- `tools/vk/linear8.comp`: 4 Ausgaben × 32 Lanes je Arbeitsgruppe, jede Lane entpackt 8
  aufeinanderfolgende Gewichte aus ganzen Wörtern und rechnet sie gegen bis zu 8 Token-Zeilen
  (Prefill nutzt jedes entpackte Gewicht 8-mal); `linear.comp` bleibt für F32 mit einer
  Eingangsbreite, die kein Vielfaches von 8 ist.
- Rücklese-Puffer (Logits, Staging) in `HOST_CACHED`-Speicher: Die Logits aus nicht
  gecachtem (write-combined) Speicher zu lesen kostete auf der RTX 5090 rund 2 ms je Token.
- Fusionen: `qkrope.comp` (Q/K-Norm, RoPE, K/V in den Cache - vorher sechs Dispatches),
  `addnorm.comp` (Residual und folgende RMSNorm); Barrieren nur, wo ein Schritt vom vorigen
  abhängt (Q/K/V und Gate/Up laufen nebeneinander). Rund 480 Dispatches mit Barriere je
  Token wurden so zu rund 300, die meisten ohne eigene Wartezeit.
- `attention.comp`: Anfrage-Kopf in Shared Memory, Schlüssel als `vec4`.

| RTX 5090 | Prefill 200 | Decode 200 | Prefill 1000 | Decode 1000 |
|---|---|---|---|---|
| Q4_K_M | 8337 Tok/s | 303 Tok/s | 6335 Tok/s | 196 Tok/s |
| Q8_0 | 8756 Tok/s | 320 Tok/s | 6581 Tok/s | 204 Tok/s |

- Falle: `VkPhysicalDeviceLimits` enthält `size_t`-Felder - `timestampPeriod` liegt unter
  32 Bit bei 416, unter 64 Bit bei 424 (gefunden von Codex; Test `ProfilesBatches`).

**4e, zweiter Teil (2026-10-16):** die Attention für lange Kontexte.

- Split-K (Flash-Decoding): Beim Dekodieren gab es nur eine Arbeitsgruppe je Kopf (16 auf
  einer Karte mit 170 SMs). Die Schlüssel einer Zeile werden jetzt in Bereiche von rund 64
  geteilt, solange es wenige Arbeitsgruppen gibt (höchstens 2048 insgesamt);
  `tools/vk/attnmerge.comp` führt die Teilergebnisse per Log-Sum-Exp zusammen. Decode bei 1000
  Tokens: 204 → 341 Tok/s.
- Flash-Attention: Eine Arbeitsgruppe rechnet bis zu 8 Anfragen, nämlich die Köpfe eines
  KV-Kopfs (GQA) mal Zeilen. Jeder gelesene Schlüssel und Wert dient allen. Die Schlüssel laufen
  in Kacheln zu 128 durch, ihre Scores bleiben im Shared Memory (16 KB, das garantierte
  Minimum), der Softmax läuft online (laufendes Maximum, Summen neu skaliert). Der globale
  Scores-Puffer und mit ihm die Kontextgrenze entfallen. Attention-Zeit beim Prefill von 4000
  Tokens: 829 → 185 ms.
- Lehrgeld: Die erste Kachel-Variante behielt den globalen Scores-Puffer und war langsamer
  (1139 ms). Nicht der K/V-Verkehr begrenzte, sondern die Score-Ladevorgänge im Werte-Schritt.
- `TQwen3Vulkan.SplitKeys` stellt die Bereichsgröße ein (0 = keine Bereiche). Der Test
  `AttentionRangesMatchCpu` fährt beide Extreme, denn mit der Standard-Heuristik bekommt das
  Tiny-Modell nie mehr als eine Kachel je Arbeitsgruppe. Die Gegenprobe (ohne Neuskalierung)
  fiel erst mit diesem Test auf.

| RTX 5090, Q8_0 | Prefill 200 | Decode 200 | Prefill 1000 | Decode 1000 | Prefill 4000 | Decode 4000 |
|---|---|---|---|---|---|---|
| 4e, Teil 1 | 8756 | 320 | 6581 | 204 | 3264* | - |
| 4e, Teil 2 | 9087 | 323 | 8818 | 319 | 6777 | 287 |

\* mit Split-K gemessen. Q4_K_M: Prefill 1000 8345, Decode 1000 299 Tok/s. Decode schwankt
zwischen Läufen um etwa ±5 %.


**4e, dritter Teil (2026-10-16):** Subgroup-Reduktionen in `linear8.comp`.

- Die Instanz fordert Vulkan 1.1 an, wenn der Loader es meldet
  (`vkEnumerateInstanceVersion`); Gerät und `vkGetPhysicalDeviceProperties2` müssen mitziehen,
  sonst bleibt es bei 1.0. Mit 1.1 werden die Subgroup-Eigenschaften abgefragt
  (`TInferVulkan.SubgroupSize`/`SubgroupOperations`/`SubgroupStages`).
- `linear8.comp` wird zweimal gebaut. Die zweite Fassung (`-DSUBGROUP`, SPIR-V 1.3,
  `SPV_LINEAR8_SG`) summiert die 32 Teilsummen je Ausgabe mit `subgroupClusteredAdd` statt über
  Shared Memory mit 5 Barrieren. Sie wird nur genommen bei Subgroups von 32 bis 128 (Zweierpotenz)
  mit Basic-, Arithmetic- und Clustered-Operationen im Compute-Shader. lavapipe hat kleinere
  Subgroups und nimmt die alte Fassung; `LinearMatchesReference` prüft auf Geräten mit Subgroups
  beide (`TInferVulkanBackend.UseSubgroups`). Gegenprobe (Cluster 16 statt 32) auf der RTX: 46
  Fehlschläge.
- Falle (Codex): Vulkan garantiert keinen Zusammenhang zwischen `gl_LocalInvocationID` und der
  Lane in einer Subgroup. Der Subgroup-Pfad leitet Lane und Ausgabe deshalb aus `gl_SubgroupID`
  und `gl_SubgroupInvocationID` ab, und zwar nur bei voll besetzten Subgroups von mindestens 32.
  Sonst summiert auch diese Fassung über Shared Memory, ebenso auf Geräten, die zur Laufzeit
  kleinere Subgroups nutzen, als sie melden.
- Gewinn nur beim Prefill: RTX 5090, Q8_0, Prefill 200/1000/4000 Tokens 9087/8818/6777 →
  9839/9412/7112 Tok/s (+5 bis +8 %). Das Dekodieren bleibt bei rund 320 Tok/s.
- Beim Dekodieren liegen je Token rund 290 Dispatches bei etwa 2,5 ms GPU-Zeit, also knapp
  9 µs je Dispatch. Die Untergrenze liegt bei etwa 4 µs (`swiglu`, `qkrope`). Die Bandbreite wäre
  für 0,6B in Q8_0 nach rund 0,4 ms erschöpft. Weiter kommt man nur mit weniger Dispatches
  (Fusionen: Gate/Up/SwiGLU, Q/K/V in einem Dispatch) und weniger Host-Arbeit je Token.

**Dekodieren, erster Teil (2026-10-16):** Gemessen je Token (Q8_0, 200 Tokens Kontext):
GPU 2,47 ms, Host-Aufzeichnung 0,27 ms, `InferArgMax` über 151936 Logits 0,27 ms.

- `InferArgMax` vergleicht die Float-Bits als Ganzzahlen, ohne Sprung nach dem Vorzeichen.
  Logits sind etwa zur Hälfte negativ, die Sprungvorhersage lag also jeden zweiten Wert daneben.
  Ergebnis: 265 → 94 µs. Vertrag unverändert (erster Nicht-NaN-Wert, dann nur echt größere,
  -0 = +0). Davon profitiert auch der CPU-Pfad.
  - Falle 1 (Claude-Review): `cardinal(0) - x` rechnet FPC auf 64 Bit in Int64; die Maske ist
    deshalb ein Produkt `(b shr 31) * $7fffffff`. Gegenprobe: 24 Fehlschläge.
  - Falle 2 (Codex): Der Test darf zufällige Bitmuster nicht über ein `single`-Ergebnis
    erzeugen, denn eine signalisierende NaN löst im x87 von Delphi Win32 eine Ausnahme aus.
- `TInferVulkan.BatchFlush`: ein Ring aus 8 Command-Buffern. Ein fertiges Stück geht ohne
  Fence an die Queue, während der Host weiter aufzeichnet. Die Barriere am Anfang des nächsten
  Buffers und die Fence am Schluss decken alle früheren Einreichungen ab (Submission Order).
  `TQwen3Vulkan` schickt nach Schicht 0 und dann nach jeder vierten. Für Fehlerpfade gilt:
  Abbruch wartet auf das Gerät, ein gescheitertes Submit gibt den Kontext auf, `fRecording`
  verhindert ein doppeltes `vkEndCommandBuffer` (Codex).

| RTX 5090, Q8_0, Decode | 200 | 1000 | 4000 Tokens Kontext |
|---|---|---|---|
| vorher | 305 | 322 | 290 |
| jetzt | 360 | 375 | 329 Tok/s |

- Gemessen und verworfen (2026-10-16): Fusionen von Gate/Up/SwiGLU (ein `linear8`-Dispatch im
  GLU-Modus, Gewichte hintereinander in einem Puffer) und von Q/K/V (ein Dispatch, `qkrope` und
  `attention` mit Zeilenabstand). Ergebnis: Decode 360/382/332 statt 360/375/329, also im
  Rauschen, Prefill etwa 1,5 % langsamer. Grund: SwiGLU fällt weg (−7,4 ms auf 64 Tokens), aber
  der fusionierte Kern wird um ungefähr dasselbe langsamer. Die erste Fassung, in der jede Lane
  beide Zeilen rechnete, halbierte zudem die Parallelität. Q/K/V liefen ohnehin ohne Barriere
  nebeneinander. Die Annahme „feste Kosten je Dispatch“ trägt also nicht: Die Zeit steckt in der
  Laufzeit der Kerne, die beim Dekodieren an der Latenz hängen.
- Offen: Linear beim Dekodieren mit mehr Parallelität je Matrix-Vektor-Produkt (rund 10 µs je
  Gruppe), `addnorm` mit Subgroups (8 µs, 55 je Token), Argmax auf der GPU (spart 94 µs und das
  Lesen von 600 KB).

**`TNativeLlmClient` auf der GPU (2026-10-16):**

```pascal
gpu := TInferVulkanBackend.Create;
client := TNativeLlmClient.CreateFromGguf('qwen3-0.6b-q8_0.gguf', '', gpu);
client.Device := TQwen3Vulkan.Create(client.Model, gpu); // gehört dann dem Client
llm := client; // ILlmClient: ChatComplete, ChatStream, Agent, Tools wie bisher
```

- Der Client bleibt frei von Vulkan, denn die Brücke liegt im Modell. Die abstrakte Klasse
  `TQwen3Device` (in `mormot.ai.infer.model.qwen3`) rechnet ein `TQwen3Model` anderswo.
  `TQwen3Model.Device` legt fest, wo `NewContext` neue Kontexte anlegt. `Forward`, `Reuse` und
  `Greedy` folgen dem Kontext: Ein Kontext eines Geräts wird dort gerechnet.
- `TQwen3Vulkan` ist ein `TQwen3Device`, `TQwen3VkContext` ein `TQwen3Context`. Länge, Token-IDs,
  `Common` und `Truncate` stammen aus der Basis, deshalb läuft der Prompt-Cache des Clients
  unverändert, mit dem KV-Cache auf der Karte.
- `TNativeLlmClient.Device`: Das Gerät gehört dem Client. Der Wechsel leert den Prompt-Cache
  und wird abgewiesen, solange Anfragen laufen (Zähler unter Lock; Claude- und Codex-Review).
  `Release` verwirft Kontexte eines anderen Geräts. `TQwen3Vulkan.Destroy` setzt
  `Model.Device` zurück. Die alten Kontexte und das alte Gerät werden außerhalb des Spinlocks
  freigegeben (`TNativePromptCache.Take`), und derselbe Wechsel zweimal gibt nichts frei
  (Codex-Nachreview).
- Test `NativeClientOnDevice` (RTX 5090, Qwen3-0.6B Q8_0, greedy):
  - Antwort auf der GPU und auf dem Backend wortgleich.
  - Die GPU-Antwort entsteht ohne einen Backend-`Linear`-Aufruf (`GpuLinears` bleibt gleich).
  - Der nächste Gesprächsschritt kommt aus dem Prompt-Cache auf dem Gerät.
  - `Device := nil` führt zurück aufs Backend.
  - Gegenprobe: Mit reinen CPU-Kontexten scheitert der Test.

**KV-Cache in Halbgenauigkeit (2026-10-17):**

- Beim Dekodieren liest die Attention für jedes neue Token den ganzen KV-Cache. Bei
  Qwen3-0.6B sind das in F32 rund 115 KB je Token Kontext für K und noch einmal so viel für
  V, bei 4000 Tokens knapp 1 GB je Schritt. `TQwen3Vulkan.KvF16` (Vorgabe an) hält K und V
  als Halbgenauigkeit, zwei Werte in einem Wort (`packHalf2x16`/`unpackHalf2x16`, Kern-GLSL,
  keine 16-Bit-Erweiterung nötig). Gebaut als Varianten `-DKVF16` von `qkrope.comp`,
  `attention.comp` und `flash.comp` (`SPV_*_F16`); in `qkrope` schreibt jede Invocation
  ganze Wörter, also zwei Dimensionen, daher Kopfdimension durch 4 teilbar (wie bisher
  verlangt). Das F32-SPIR-V der Attention ist dabei bis auf die Nummerierung gleich
  geblieben.
- Ein Kontext übernimmt das Format, wenn sein Cache angelegt wird, und wieder bei seinem
  nächsten `Forward`, solange er leer ist (nach `Reset`); `TQwen3VkContext.KvF16` zeigt es.
  Halb so viel Speicher heißt auch: doppelt so viele Tokens unter `maxStorageBufferRange`.
- Genauigkeit: Die Logits weichen mit F16 bis 1,1e-3 relativ von der CPU ab (F32-Cache: unter
  1e-4). `AttentionRangesMatchCpu` prüft beide Formate mit eigener Toleranz (1e-4 bzw. 4e-3),
  über `attention.comp` und `flash.comp`, mit und ohne Splits, Kopfdimension 32 und 256, und
  lässt den Cache einmal mit Inhalt wachsen. Qwen3-0.6B gegen die CPU unverändert
  (Kosinus 0,9990 bei Q8_0, 0,9972 bei Q4_K_M; Argmax 16 von 16).
- Gegenprobe, alle acht scheitern: vertauschte Halbwerte in `qkrope` (V), in `attention`
  (K und V getrennt) und in `flash`, die zweite Hälfte der Rotation an der falschen Stelle,
  der F32-`flash` für einen F16-Cache, das Umkopieren beim Wachsen mit 2 Byte je Wert und ein
  leerer Kontext, der sein altes Format behält. `NativeClientOnDevice` verlangt wortgleiche
  Antworten über einen ganzen Text und bleibt deshalb beim F32-Cache.
- RTX 5090, Qwen3-0.6B Q8_0, Decode in Tokens/s (zwei Läufe, gleich bis auf 1 %); der Prefill
  bleibt unverändert:

  | Kontext | 200 | 1000 | 4000 | 8000 |
  |---|---|---|---|---|
  | F32-Cache | 351 | 373 | 325 | 265 |
  | F16-Cache | 361 | 383 | 346 | 287 |

  Auf der RTX bremst die Bandbreite wenig (1,8 TB/s). `llm-bench` schaltet mit `LAI_KV_F32=1`
  auf den F32-Cache zurück.
- *M2 Pro (MoltenVK, 2026-10-17):* kaum Wirkung. Decode mit F32- und F16-Cache: 107,6/107,2
  (200 Tokens), 77,7/78,3 (1000), 42,9/43,4 (4000), 27,6/28,0 (8000), der Prefill
  unverändert. Der Einbruch bei langem Kontext kommt dort also nicht von der Bandbreite
  (bei 8000 Tokens 1,8 GB je Token in F32, bei 200 GB/s rund 9 ms von 36), sondern von der
  Attention selbst; die Vorgabe bleibt an (dort neutral, auf der RTX schneller, halber
  Speicher).

**Decode-Attention auf Apple-GPUs: `attndec.comp` (2026-10-17):**

- *Befund:* Eine Probe, die die Attention einer Schicht allein misst (Qwen3-0.6B, 1 Zeile,
  F16-Cache), zeigte auf dem M2 Pro rund 490 us je Schicht bei 4000 Tokens, 33 GB/s statt
  der rund 200 der Karte. Bei nur 63 Schlüsseln kosteten 512 Arbeitsgruppen schon 283 us:
  Eine Arbeitsgruppe von `attention.comp` (128 Invocations, 16 KB Shared Memory, rund 10 Barrieren
  je Kachel) dauert dort lange, vermutlich läuft je GPU-Kern nur eine gleichzeitig, und jede
  Speicherlatenz schlägt durch. Eine Variante mit gemeinsam geladenen Kacheln (zusammenhängend
  gelesen, gepolstert gegen Bankkonflikte) änderte daran wenig (40 GB/s).
- *Kernel:* `attndec.comp` braucht weder Shared Memory noch Barrieren. Eine Arbeitsgruppe ist
  eine Subgroup von 32; Lane l hält Slot l (4 Dimensionen) der bis zu 8 Queries in Registern
  und liest Slot l jeder Schlüssel- und Wertzeile (eine Zeile von 128 Halbwerten: 256 Byte am
  Stück über die 32 Lanes). Der Score ist die `subgroupAdd` der Teilprodukte, die Softmax
  läuft online je Schlüssel in Registern. Bindungen, Push-Konstanten und Ausgabe wie
  `attention.comp`, `attnmerge.comp` fügt die Bereiche zusammen. Vulkan 1.1 mit
  Subgroup-Arithmetik (`SubgroupLinear`), Kopfdimension bis 128, F32- und F16-Build.
- *Eine Arbeitsgruppe, eine Subgroup:* Vulkan garantiert ohne `requireFullSubgroups` nicht,
  dass 32 Invocations eine einzige Subgroup bilden (Codex-Fund); die Hardware tut es bei Apple
  (SIMD-Gruppen von 32) und NVIDIA (Warps von 32). `HasDecodeAttention` gilt deshalb nur dort,
  bei Subgroup-Größe 32. Zerfällt eine Gruppe trotzdem, speichert der Kernel NaN statt
  Teilsummen, ohne Rechenschritt danach: In `y` bleibt es sicher stehen; durch `attnmerge.comp`
  ist es nicht garantiert, denn Vulkan erlaubt Compilern, bei Arithmetik NaN-freie Operanden
  anzunehmen (Codex). Der eigentliche Schutz ist die Freigabe nur für Apple und NVIDIA. Ein
  Ausweichpfad über Shared Memory im selben Kernel kostete auf dem M2 Pro 17
  bis 32 %, auch wenn er nicht genommen wurde (schon das Shared-Memory-Array bremst), und
  entfiel.
- *Aufteilung:* Am schnellsten waren auf dem M2 Pro 256 Arbeitsgruppen (bei Qwen3-0.6B 32
  Bereiche je KV-Kopf), bei 4000 wie bei 8000 Tokens (188 statt 489 us je Schicht bei 4000,
  351 statt 916 bei 8000, 88 bis 93 GB/s). Mit 2 oder 4 Schlüsseln vorab geladen lief er
  langsamer (Register). Bei 200 Tokens ist `attention.comp` etwas schneller (52 gegen 60 us),
  deshalb erst ab `DecodeKeys` (512) Schlüsseln.
- *RTX 5090:* dort ist `attention.comp` doppelt so schnell (18 gegen 37 us bei 4000 Tokens
  und 64 Bereichen; der neue Kernel bräuchte tausende Arbeitsgruppen).
  `TQwen3Vulkan.UseDecodeAttention` ist deshalb nur auf Apple-GPUs voreingestellt, wie
  `UseFlash` und `UseMatvec`.
- *Ergebnis*, Qwen3-0.6B Q8_0 auf dem M2 Pro, Decode in Tokens/s (zwei Läufe im Wechsel,
  gleich bis auf 1 %):

  | Kontext | 200 | 1000 | 4000 | 8000 |
  |---|---|---|---|---|
  | `attention.comp` | 107,2 | 78,6 | 43,4 | 28,0 |
  | `attndec.comp` | 107,1 | 101,0 | 74,3 | 56,6 |

- *Tests:* `AttentionRangesMatchCpu` dekodiert die Schritte 2 und 3 über `attndec.comp`, mit
  einem Bereich und mit vielen, beide Cache-Formate; `Qwen3GgufMatchesCpu` dekodiert Q4_K_M
  darüber, jeweils wo das Gerät ihn hat (M2 Pro, RTX 5090; nicht Adreno und lavapipe).
  Gegenprobe: ohne `subgroupAdd`, alle Lanes aus Slot 0, Summen nicht umskaliert, Queries
  nicht skaliert, Maximum und Summe im `part[]` vertauscht, der F32-Kernel für einen
  F16-Cache: alle sechs scheitern. Ohne die Begrenzung der Bereiche auf die Zahl der
  Schlüssel bleibt der Test grün - zu Recht: leere Bereiche schreiben Summe 0, und
  `attnmerge.comp` übergeht sie; die Begrenzung spart nur leere Arbeitsgruppen.
- *CUDA (2026-10-17):* `TQwen3Cuda.KvF16`, ebenfalls Vorgabe an. CUDA adressiert 16-Bit-Werte
  direkt, deshalb ohne Wortpaare: `qkrope` und `attention` in `tools/cuda/ops.cu` sind
  Templates über den Typ des Caches (`qkrope16`, `attention16`), umgerechnet per PTX-`cvt`
  (`cvt.rn.f16.f32`, kein `cuda_fp16.h`). Das PTX der F32-Kernel ist bis auf Namen und
  Sprungmarken gleich geblieben. Tests: `TinyQwen3MatchesCpu` mit F32- und Q8_0-Modell je in
  beiden Formaten (mit Graph-Decode, Reuse und Wachsen des Caches), `TinyHeadDim256MatchesCpu`
  wechselt das Format im selben Kontext; F16-Fehler auf der RTX höchstens 4,7e-4.
  Gegenprobe: abgeschnittenes Halbwort beim Schreiben, vertauschte Halbwerte und Wörter beim
  Lesen der Schlüssel, Wert vom Nachbarn, die F32-Attention für einen F16-Cache, Umkopieren
  mit 2 Byte, ein leerer Kontext mit altem Format: alle sieben scheitern.

  | RTX 5090, Q8_0, Decode | 200 | 1000 | 4000 | 8000 |
  |---|---|---|---|---|
  | CUDA, F32-Cache | 405 | 404 | 347 | 252 |
  | CUDA, F16-Cache | 418 | 415 | 383 | 312 |

  Der Prefill gewinnt bei 4000 und 8000 Tokens 4 bis 6 % (10 045 statt 9644, 7460 statt
  7047 Tokens/s). Bei 8000 Tokens ist CUDA mit F16-Cache jetzt auch dort schneller als
  Vulkan (312 gegen 288), wo es mit F32-Cache langsamer war (252 gegen 267).

**Fertig, wenn:** das GPU-Backend dieselben Referenztests besteht wie die CPU (mit
Float-Toleranzen) und auf der RTX 5090 deutlich schneller dekodiert.

#### CUDA (Backlog B3)

In Scheiben: C1 Fundament, C2 `Linear`, C3 die übrigen Operationen, C4 Qwen3 ganz auf der
Karte, C5 Tempo.

- *C1, erledigt 2026-10-17:* `mormot.ai.infer.cuda`.
  - *Bindung:* die Driver API, zur Laufzeit geladen (`nvcuda.dll`, `libcuda.so.1`), nur
    was Compute braucht: Gerät, Primärkontext (`cuDevicePrimaryCtxRetain` statt
    `cuCtxCreate`, dessen Signatur sich mit CUDA 13 geändert hat), Module aus PTX,
    Gerätespeicher, `cuLaunchKernel`. Jede öffentliche Methode macht den Kontext auf dem
    rufenden Thread aktuell. Nur 64-Bit; sonst ist `TInferCuda.Available` falsch.
  - *Kernel:* CUDA C unter `tools/cuda/`, mit `nvcc` nach PTX (`compute_75`), eingecheckt
    als `cardinal`-Arrays (`mormot.ai.infer.cuda.ptx.inc`, `tools/cuda/build-kernels.sh`).
    Der Treiber übersetzt das PTX beim Laden für seine GPU. `nvcc` 13.4.92 kommt aus
    NVIDIAs pip-Paket und läuft auch auf aarch64: gebaut wird auf dem Laptop, der
    RTX-Rechner bleibt Testrechner. Das PTX (ISA 9.4) verlangt einen Treiber für CUDA 13.4
    oder neuer (PTX ist von NVIDIAs Minor-Version-Kompatibilität ausgenommen) und
    mindestens Turing.
  - *Test:* `TTestInferCuda` (`AddMatchesCpu` bitgleich über 1 000 003 und 17 Werte,
    `RefusesMisuse` mit dem Fehlernamen des Treibers, `UploadDownload`), ohne Treiber
    SKIP, `LAI_REQUIRE_CUDA=1` macht daraus einen Fehler. RTX 5090 (Compute 12.0, Treiber
    CUDA 13.4): 11 Assertions grün unter FPC 3.2.2 und Delphi 13. Gegenprobe (Kernel
    subtrahiert) lässt 2 scheitern.
- *C2, erledigt 2026-10-17:* `TInferCudaBackend` (`mormot.ai.infer.backend.cuda`), das
  schnelle CPU-Backend mit `Linear` auf der Karte für F32, Q8_0, Q4_K und Q6_K, gebaut wie
  `TInferVulkanBackend` (Gewichts-Cache, Scratch-Puffer, CPU bei belegtem Gerät).
  - *Kernel:* `tools/cuda/linear.cu`, die CUDA-Form von `linear.comp`: ein Block aus 128
    Threads je Ausgabe und Tokenzeile, die Gewichte je Element dequantisiert wie
    `InferRowToF32`, Warp-Shuffle-Reduktion; Halbfloat per `cvt.f32.f16` (ohne
    `cuda_fp16.h`, das Header braucht, die im pip-Paket fehlen). Zuerst korrekt, Tempo in C5.
  - *Test:* `LinearMatchesReference` (4 Typen × 6 Formen bis 70 001 Ausgaben, Bias, Cache
    bitgleich, Q4_0 auf der CPU). Grün auf der RTX 5090 unter FPC 3.2.2 und Delphi 13
    (Windows) und unter FPC 3.2.2 in der WSL des RTX-Rechners (Linux, `libcuda.so.1` der
    WSL; die ganze Inferenz-Suite 2572 Assertions). Gegenprobe (je ein Fehler in Q8_0, Q4_K
    und Q6_K) lässt 18 scheitern, F32 bleibt grün.
- *C3, erledigt 2026-10-17:* `tools/cuda/ops.cu` mit `rmsnorm`, `addnorm`, `swiglu`,
  `qkrope` (Kopf-Norm, Rotation, KV-Speichern) und `attention` (kausal, GQA; zuerst
  korrekt: die Scores in einem Scratch auf der Karte, Softmax, dann die Werte - Kacheln in
  C5). `TInferCuda.Enqueue` startet ohne Warten (ein Stream, die Reihenfolge bleibt),
  `Synchronize` wartet, `TInferCuBuffer.CopyTo` kopiert auf der Karte.
  - *Test:* `OpsMatchCpu` gegen `RmsNorm`, `Add`, `SwiGlu`, `Rope` und `CausalAttention` der
    CPU-Referenz (Prompt, Zeilen nach einem Präfix, Decode über 300 Schlüssel, Kopf 256,
    der Cache außerhalb der neuen Zeilen unverändert), `qkrope` und `attention` direkt
    hintereinander; `UploadDownload` prüft `CopyTo`. RTX 5090: 112 Assertions grün unter
    FPC 3.2.2 und Delphi 13. Gegenprobe (Teiler der Norm, Vorzeichen der Rotation,
    Kausalgrenze) lässt 14 von 23 scheitern.
- *C4, erledigt 2026-10-17:* `TQwen3Cuda` (`mormot.ai.infer.model.qwen3.cuda`), die
  CUDA-Form von `TQwen3Vulkan` hinter demselben `TQwen3Device`-Vertrag: der ganze
  Forward-Pass auf der Karte, `TNativeLlmClient` nimmt sie über `Device`.
  - *Ablauf:* je Chunk und Schicht `EnqueueLinear` (Q, K, V, O, Gate, Up, Down; ohne
    Warten, Gewichte aus dem Cache des Backends), `qkrope`, `attention`, `addnorm`
    (Residuum plus Norm der nächsten Schicht), `swiglu`; zuletzt Norm, Kopf und ein
    einziger Download der Logits. `GpuLock` hält die Karte über den ganzen Pass; `Linear`
    anderer Threads rechnet derweil auf der CPU.
  - *Speicher:* Chunks bis 256 Zeilen, halbiert, bis die Scores der Attention unter 256 MB
    bleiben (eine Zeile allein kann darüber liegen, bis C5 die Attention ohne Scores-Puffer
    bringt). Der KV-Cache wächst wie auf der CPU und kopiert die vorhandenen Token auf der
    Karte um (`CopyTo`); der Scores-Puffer bekommt ein Viertel Reserve, sonst würde jeder
    Decode-Schritt neu anlegen.
  - *Test:* `TinyQwen3MatchesCpu` (F32 und Q8_0: Prompt über zwei Chunks, Decode,
    Präfix-Wiederverwendung, `ClearWeights`, Greedy, Cache-Wachstum mit Inhalt),
    `TinyHeadDim256MatchesCpu`, `Qwen3GgufMatchesCpu` (Qwen3-0.6B Q4_K_M: Logit-Cosinus
    mindestens 0,9972, Q8_0 0,9991, Argmax 16 von 16) und `NativeClientOnDevice` (Antwort
    wortgleich zum Backend, nächster Schritt aus dem Prompt-Cache auf der Karte). RTX 5090:
    159 Assertions grün unter FPC 3.2.2 und Delphi 13 und unter FPC 3.2.2 in der WSL
    (Linux, ganze Inferenz-Suite 2645). Gegenprobe (Scores-Abstand, Norm der falschen
    Schicht, letzte Zeile, Länge nach Decode, Cache-Wachstum ohne Kopie) scheitert in jedem
    der fünf Fälle.
- *C5, Tempo:* `llm-bench … cuda` misst wie `vulkan` (gemeinsamer Pfad über
  `TQwen3Device`). Ausgangslage auf der RTX 5090 (Qwen3-0.6B, Prefill / Decode in Token/s):

  | Modell, Prompt | Vulkan | CUDA nach C4 |
  |---|---|---|
  | Q8_0, 200 | 13 683 / 358 | 1 859 / 222 |
  | Q8_0, 1000 | 14 581 / 364 | 1 494 / 109 |
  | Q4_K_M, 200 | 13 008 / 332 | 1 543 / 219 |
  | Q4_K_M, 1000 | 13 799 / 347 | 1 283 / 108 |

  - *C5a, erledigt 2026-10-17:* `attention` ist jetzt die CUDA-Form von `attention.comp`
    (Blöcke aus bis zu 8 Abfragen - Köpfe einer KV-Gruppe mal Zeilen -, Schlüsselkacheln
    zu 128, Online-Softmax, die Schlüssel einer Zeile beim Decode in Bereiche geteilt) mit
    `attnmerge`; dieselbe Aufteilung wie Vulkan (`SplitKeys`, Vorgabe 64). Der Scores-Puffer
    und seine Chunk-Grenze entfallen. Decode bei 1000 Token Kontext: Q8_0 109 → 249, Q4_K_M
    108 → 213 Token/s (so schnell wie bei 200). Test: `OpsMatchCpu` mit Kopfblöcken und
    1 bis 5 Bereichen (auch leere späte Bereiche), `TinyQwen3MatchesCpu` dekodiert
    abwechselnd mit und ohne Aufteilung; RTX 5090 175 Assertions grün unter FPC 3.2.2 und
    Delphi 13. Gegenprobe (Gewicht im Merge, Bereich um einen Schlüssel zu kurz, kein
    Umskalieren, Merge fehlt) scheitert in allen vier Fällen.
  - *C5b, erledigt 2026-10-17:* die Matrixprodukte als CUDA-Formen von `linear8.comp`
    (4 Ausgaben je Block, eine je Warp, bis 8 Zeilen, Warp-Shuffle-Summe) und `gemm8.comp`
    (Kachel 64 × 64, Aktivierungen und entquantisierte Gewichte im Shared Memory), gewählt
    wie auf Vulkan (`LinearShape`: `gemm8` ab 16 Zeilen, sonst `linear8`, der einfache
    Kernel bleibt Rückfall; `FastLinear` schaltet auf ihn allein). Gewichtspuffer werden
    auf ganze 32-Bit-Wörter aufgerundet: die Kernel lesen Wörter, das letzte einer
    Q8_0- oder Q6_K-Zeile reichte bis 2 Byte hinter die Daten (Codex-Fund).

    | Modell, Prompt | Vulkan | CUDA nach C5a | CUDA nach C5b |
    |---|---|---|---|
    | Q8_0, 200 | 13 670 / 361 | 1 922 / 243 | 12 335 / 266 |
    | Q8_0, 1000 | 14 634 / 375 | 1 895 / 249 | 12 893 / 273 |
    | Q4_K_M, 200 | 13 017 / 332 | 1 589 / 226 | 11 358 / 274 |
    | Q4_K_M, 1000 | 13 922 / 346 | 1 548 / 213 | 11 887 / 281 |

    Test: `LinearMatchesReference` jede Form mit und ohne `FastLinear`, dazu F32 mit 520,
    100 und 36 Eingängen (alle drei Kernel); RTX 5090 265 Assertions grün unter FPC 3.2.2
    und Delphi 13 und unter FPC 3.2.2 in der WSL (Linux, ganze Inferenz-Suite 2751). Gegenprobe (Bytes in `weights8`, Q4_K-Minimum, letzte Zeile in `gemm8`,
    Shuffle über 16 Lanes, `gemm8` für ungeeignete Eingänge) scheitert in allen fünf Fällen.
  - *C5c, erledigt 2026-10-17:* ein Decode-Schritt als CUDA-Graph. Die Messung zeigte, wohin
    die fehlende Millisekunde ging: der Host brauchte 1,8 ms für die rund 370 Kernelstarts
    eines Tokens, die Karte wartete zwischen ihnen. `TInferCuda.CaptureBegin`/`CaptureEnd`
    zeichnen den Schritt auf (Stream-Capture) und starten ihn als einen Graphen; der Graph
    des vorigen Schritts übernimmt die neuen Parameter (`cuGraphExecUpdate`), bei anderer
    Form wird er neu gebaut. Dafür laufen alle Kernel und Kopien auf einem eigenen,
    nicht-blockierenden Stream, gewartet wird nur auf ihn (`cuStreamSynchronize`): eine
    zweite Instanz im selben Primärkontext, in einem anderen Thread, darf während einer
    Aufzeichnung kopieren und warten (Codex-Fund; mit blockierendem Stream und Kopien auf
    dem Legacy-Stream scheiterte sie mit `CUDA_ERROR_STREAM_CAPTURE_IMPLICIT`).
    `TQwen3Cuda.UseGraphs` (Vorgabe an). Dazu die beiden inneren Schleifen der Attention
    entrollt (`#pragma unroll 8`, Decode +10 %); dasselbe in `linear8` brachte nichts und
    blieb draußen.

    | Modell, Prompt | Vulkan | CUDA nach C5b | CUDA nach C5c |
    |---|---|---|---|
    | Q8_0, 200 | 13 683 / 358 | 12 335 / 266 | 12 392 / 404 |
    | Q8_0, 1000 | 14 596 / 376 | 12 893 / 273 | 13 319 / 407 |
    | Q4_K_M, 200 | 12 993 / 331 | 11 358 / 274 | 11 367 / 360 |
    | Q4_K_M, 1000 | 13 893 / 349 | 11 887 / 281 | 12 193 / 362 |

    Test: `GraphRunsCapture` (Aufzeichnen, Aktualisieren mit anderen Parametern, andere
    Form, Verweigerung von Kopien und Warten während der Aufzeichnung, Abbruch, eine zweite
    Instanz in einem anderen Thread), `TinyQwen3MatchesCpu` dekodiert erst Kernel für
    Kernel, dann als Graph. RTX 5090: 279 Assertions grün unter FPC 3.2.2 und Delphi 13
    und unter FPC 3.2.2 in der WSL (Linux, ganze Inferenz-Suite 2765). `CopyTo` zwischen
    Puffern zweier Instanzen wird abgelehnt (zwei Streams, Codex-Fund).
    Gegenprobe (kein Update, kein Start, Abbruch ohne Ende der Aufzeichnung, Kopie auf dem
    Legacy-Stream, blockierender Stream) scheitert in allen fünf Fällen.

### Phase D — weitere Decoder-Familien

Plan, Kandidaten und Reihenfolge: [backlog.md](backlog.md), Abschnitt B4 (MoE, Sliding
Window, gpt-oss-20b, Gemma 4, Gated DeltaNet, Kolibri).

**D1 erledigt (2026-10-17): gemeinsame Decoder-Typen.** Reiner Umbau, die Rechnung ist
unverändert (alle Qwen3-Tests ohne Änderung grün, Decode-Tempo gleich).

- `mormot.ai.infer.decoder`: `TDecoderModel` (Gewichte laden und prüfen, Forward, Reuse,
  Greedy, Übergabe an ein Gerät), `TDecoderContext` (KV-Cache einer Sequenz),
  `TDecoderDevice` (GPU), `TDecoderConfig`. Neu in der Konfiguration ist `Spec`, die Art
  jedes Layers: `Mixer` (heute nur `dmAttention`) und `Ffn` (heute nur `dfDense`). Forward
  verzweigt je Layer darauf (`AttentionBlock`, `DenseBlock`); eine Familie, die `Spec` leer
  lässt, bekommt überall Attention und SwiGLU.
- `mormot.ai.infer.model.qwen3` ist nur noch die Familie: Einstellungen aus `config.json`
  oder den GGUF-Metadaten, Tensornamen, Gewichte. Die alten Namen (`TQwen3Context`,
  `TQwen3Config`, `TQwen3Device` ...) bleiben als Aliase.
- `mormot.ai.infer.tokenizer`: `TInferTokenizer`, die Schnittstelle, die Template und
  Parser brauchen (`TBpeTokenizer` erbt davon). `mormot.ai.infer.chat`: `TChatTemplate`
  (eine Familie überschreibt `Build`). `mormot.ai.llm.native`: `TNativeOutputParser` setzt
  die Antwort zusammen (Inhalt, Reasoning, Tool-Aufrufe, Usage); eine Familie überschreibt
  `Token` (welche Ids stoppen, Reasoning oder einen Tool-Aufruf öffnen) und `FlushOpen`.
- Zwei Registrierungen: `RegisterDecoderModel` (Modellklasse nach `model_type` bzw.
  `general.architecture`; `NewDecoderModel`, `NewDecoderModelFromGguf`) und
  `RegisterNativeFamily` (Tokenizer, Template und Parser zur Modellklasse).
  `TNativeLlmClient` wählt darüber: `TNativeLlmClient.Create(Ordner)` liest `model_type`,
  `CreateFromGguf` die Architektur.
- Geräte (`TQwen3Vulkan`, `TQwen3Cuda`) nehmen ein `TDecoderModel` und lehnen Modelle ab,
  deren Layer nicht alle Attention mit dichtem Block sind (`HasOnly`).

*Eine Familie andocken:* eine Unit mit `TXxxModel = class(TDecoderModel)`, die `ModelType`,
`Architecture`, `LoadConfig`, `LoadGgufConfig` (beide enden mit `CheckConfig`),
`GgufName` und `LoadWeights` überschreibt und sich in `initialization` mit
`RegisterDecoderModel` anmeldet; braucht sie andere Layer, ergänzt sie `TDecoderMixer` bzw.
`TDecoderFfn`, füllt `Spec` und bekommt in `Forward` einen eigenen Zweig. Für den Client
dazu Tokenizer, Template und Parser (oder die vorhandenen) mit `RegisterNativeFamily`.

*Tests:* `DecoderFamilies` (Registrierung, `NewDecoderModel` aus Ordner und GGUF, `Spec`,
fremde `model_type`/Architektur abgelehnt) und `NativeFamilies` (Klassen von Qwen3,
unvollständige Registrierungen ignoriert, Parser ohne Tokenizer abgelehnt). Grün unter
FPC 3.2.4 und 3.2.2 (WSL, 2600 Assertions), auf dem RTX-Rechner mit echtem Modell unter
FPC 3.2.2, Delphi 13 und Delphi 7. Gegenprobe: elf eingebaute Fehler (Layer-Liste nicht gefüllt, `HasOnly` immer falsch, Registry findet keine Klasse nach `model_type` bzw. Architektur, fremder `model_type` bzw. fremde Architektur angenommen, Parser schließt offene Blöcke nicht, Zeilenumbrüche vor einem Tool-Aufruf bleiben, Parser ohne Tokenizer, Familiensuche ignoriert die Klasse, Familie ohne Template registriert) - alle scheitern.

**D2 erledigt (2026-10-17): Mixture of Experts auf der CPU.**

- `TDecoderFfn` hat `dfMoe`; `TDecoderConfig` dazu `Experts`, `ExpertsUsed`,
  `ExpertIntermediate` und `NormTopK`, `TDecoderLayer` den Router (F32, auch wenn die
  Matrizen beim Laden quantisiert werden) und die Experten als je ein Gewicht.
- `MoeBlock`: RMSNorm, Router-Logits, je Token Softmax (in double), die `ExpertsUsed`
  höchsten (bei Gleichstand die kleinere Id), auf Summe 1 normiert, wenn `NormTopK`. Dann
  Experte für Experte in Id-Reihenfolge (wie die Referenz addiert): seine Tokens gesammelt,
  ein SwiGLU-Block über sie, mit dem Router-Gewicht in ihre Zeilen addiert. Decode rechnet so
  nur die gewählten Experten, Prefill einen Block je Experte über alle seine Tokens.
- `TQwen3MoeModel` (`model_type` `qwen3_moe`, GGUF-Architektur `qwen3moe`): Experten aus
  safetensors einzeln (`mlp.experts.E.gate_proj` ...), aus GGUF als Scheiben der 3-D-Tensoren
  `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` (`TDecoderModel.GgufExperts`, Q8_0,
  Q4_0, Q4_K, Q6_K, in place); dichte Layer nach `mlp_only_layers` und
  `decoder_sparse_step`, im GGUF dort, wo `ffn_gate_inp` fehlt. Im GGUF steht keine
  Normierung: wie llama.cpp für qwen3moe immer normiert. Tokenizer, Template und Parser
  sind die von Qwen3.
- GPU-Geräte lehnen MoE-Modelle ab (`HasOnly`), das kommt mit D3.

*Referenz:* `tiny-qwen3moe` (transformers 5.19 `Qwen3MoeForCausalLM`): 3 Layer, 0 und 2 mit
8 Experten der Breite 64 und Top-2, Layer 1 dicht; der erste Seed ab 20261017, bei dem
zwischen dem k-ten und dem (k+1)-ten Router-Logit jedes Tokens mindestens 0,05 liegen (sonst
hinge die Expertenwahl an der Rundung; gewählt 20261018 mit 0,17). Dazu eine Variante ohne
Normierung und GGUF-Dateien (Q8_0, Q4_0 aus ggml) mit 3-D-Experten.

*Tests:* `TinyMoeMatchesReference` (alle Zustände bis 3e-6, Cache und Prefill in zwei
Teilen bitgleich zum ganzen Durchlauf, Greedy, ohne Normierung), `TinyMoeQuantized` (Q8_0
beim Laden: höchstens 0,21 neben FP32 bei Kosinus 0,9991 - der Router des Tiny-Modells ist
scharf, Token 5 von Fall 1 rückt in Layer 2 um 0,2; GGUF: jeder Experte Byte für Byte wie
unsere Quantisierung, Q4_0-Logits neben unseren), `TinyMoeRefused` (fehlende Schlüssel,
mehr Experten je Token als Experten, Breiten und Experten, die nicht zur Datei passen),
`Qwen3MoeChatClientGguf` (Qwen3-30B-A3B Q4_K_M hinter `TNativeLlmClient`, wo die Datei
liegt). Gegenprobe: 14 eingebaute Fehler (Wahrscheinlichkeiten nicht normiert, `NormTopK` übergangen, niedrigster Logit gewählt, ein Experte doppelt, Router-Gewicht übergangen, Summe nicht geleert, Ausgabe in die gesammelte statt die eigene Zeile, jeder Experte dieselbe GGUF-Scheibe, Form der Experten ungeprüft, mehr Experten je Token als Experten, `mlp_only_layers` übergangen, `decoder_sparse_step` um eins verschoben, im GGUF jeder Layer mit Experten, Router quantisiert) - alle scheitern, der doppelte Experte durch Absturz (der Sammelpuffer läuft über; `Route` wählt nie doppelt).

*Qwen3-30B-A3B* (`Qwen/Qwen3-30B-A3B-GGUF`, `Qwen3-30B-A3B-Q4_K_M.gguf`, 18,6 GB; Q4_K, Q6_K
für einen Teil von `attn_v` und `ffn_down_exps`, Router F32) auf dem RTX-Rechner (i9-14900K,
22 Threads, CPU): geladen in 3,8 s, 17,7 GB Spitze; Prefill 85 Tokens/s, Decode 16
Tokens/s bei 200 Tokens Prompt (zum Vergleich Qwen3-0.6B Q8_0: 759 und 70). Antwortet
`chat_en` mit „Paris“. Grün unter FPC 3.2.2, Delphi 13 (je 3165 Assertions) und Delphi 7
(1978, der 30B-Test braucht einen 64-Bit-Prozess).

**D3a erledigt (2026-10-17): Mixture of Experts auf CUDA.**

- `TQwen3Cuda` nimmt Layer mit Experten. Der Router rechnet auf dem Gerät, seine Logits
  kommen herunter (ein Warten je MoE-Layer), die Wahl trifft der Host wie auf der CPU
  (`TDecoderModel.Route`). Die Token-Zeilen und Gewichte jedes Experten gehen in einem
  Upload hoch; dann Experte für Experte in Id-Reihenfolge: `gatherrows` sammelt seine
  Zeilen, die vorhandenen Linear-Kernel rechnen Gate, Up und Down
  (`TInferCudaBackend.EnqueueLinearAt`: Gewichte an einer Geräteadresse), `swiglu`,
  `scatteradd` addiert mit dem Router-Gewicht in die Zeilen seiner Tokens (`tools/cuda/ops.cu`:
  `fill0`, `gatherrows`, `scatteradd`).
- Die Experten eines Layers liegen als ein Puffer auf dem Gerät, wenn sie im Speicher des
  Hosts hintereinander liegen (ein 3-D-Tensor im GGUF): jeder Experte ein Versatz. Sonst
  (safetensors) ein Puffer je Experte. Einzelne Puffer für die 18432 Experten von
  Qwen3-30B-A3B hätten die 2-MB-Granularität von CUDA mehr als verdoppelt (rund 40 GB).
- Kein CUDA-Graph mit Experten: der Host wählt in jedem Layer.

*Tests:* `TinyMoeMatchesCpu` (F32 mit einem Puffer je Experte, Q8_0 aus GGUF in einem Puffer,
Q8_0 beim Laden, Cache in Halbfloats; 300 Tokens Prompt in zwei Chunks, Decode-Schritte,
`ClearWeights`), `Qwen3MoeGgufMatchesCpu` (Qwen3-30B-A3B gegen die schnelle CPU, Token für
Token vorgegeben: unter FPC Argmax 16 von 16 und kleinster Kosinus 0,992, unter Delphi 13 mit
anderen CPU-Kerneln 15 von 16 und 0,988 - bei knappen Router-Entscheidungen kann ein Experte
kippen; Schwelle 0,98 und 14 von 16).
Toleranzen des scharfen Tiny-Routers: Halbfloat-Cache bis 4,6e-3, Q8_0 beim Laden Kosinus
0,9988 (die CPU quantisiert auch die Aktivierungen). Gegenprobe: neun eingebaute Fehler
(`scatteradd` ohne Router-Gewicht, `gatherrows` nimmt die Zeile des Blocks, Ausgabe nicht
geleert, alle Experten die Token des ersten, alle Experten auf derselben Adresse, Experten
immer als ein Puffer genommen, ein Graph trotz Experten, Router-Gewichte nicht übernommen,
die Router-Logits der ersten Zeile für alle) - alle scheitern, einer durch einen hängenden
Kernel (Lesen über den Puffer hinaus). Review: Codex fand, dass `InferCompare` NaN und
unendliche Werte aus dem Kosinus herausnimmt; die neuen Tests prüfen `NonFinite` mit.

*Qwen3-30B-A3B Q4_K_M auf der RTX 5090 (CUDA):* hochgeladen in 2,8 s; Prefill 384 Tokens/s
bei 200 Tokens Prompt, 494 bei 2000 (CPU: 85 und 67); Decode 33 bis 35 Tokens/s (CPU: 16 und
11). Der Decode hängt an den drei Wartepunkten je MoE-Layer (Router herunter, zwei Uploads)
und an rund 45 Kernel-Starts je MoE-Layer ohne Graph - der nächste Hebel ist die Wahl auf
dem Gerät (ein Route-Kernel, Linear mit Experten-Ids aus einem Gerätepuffer, dann wieder ein
Graph).

**D3b erledigt (2026-10-19): Mixture of Experts auf Vulkan.**

- `TQwen3Vulkan` nimmt Layer mit Experten. Der Router rechnet im Batch, seine Logits gehen in
  einen lesbaren Puffer, dann wird der Batch abgeschickt und abgewartet (ein Warten je
  MoE-Layer); der Host wählt wie auf der CPU (`TDecoderModel.Route`), schreibt Token-Zeilen und
  Gewichte jedes Experten in einen Host-Puffer und beginnt den nächsten Batch, der sie auf das
  Gerät kopiert. Dann Experte für Experte in Id-Reihenfolge: `gatherrows.comp`, Gate und Up
  (`TInferVulkanBackend.RecordLinearAt`), `swiglu.comp`, Down, `scatteradd.comp`; vorher leert
  `fill0.comp` die Ausgabe. Wo die Zeilen eines Experten beginnen, kommt als Push-Konstante
  (`First`), nicht als Versatz einer Bindung.
- Bindungen mit Versatz: `TInferVulkan.BatchDispatchAt` bindet einen Ausschnitt eines Puffers
  (Versatz ein Vielfaches von `minStorageBufferOffsetAlignment`, Bereich höchstens
  `maxStorageBufferRange`; `MinStorageAlignment` liest das Limit an Byte 296 + 328 bzw.
  + 320 unter 32 Bit). So bleiben die Experten eines Layers ein Puffer, wenn sie im GGUF ein
  Tensor sind und jeder Experte an einem gebundenen Versatz liegt (mindestens 4 Byte, die
  Shader lesen Wörter) - einzelne Puffer für 18432 Experten stießen sonst an
  `maxMemoryAllocationCount` (unter Windows oft 4096). Sonst (safetensors, Q8_0 beim Laden)
  ein Puffer je Experte. Der Bereich muss in eine Bindung passen, der Puffer als Ganzes nicht.
- Ein Batch hat höchstens 4096 Deskriptor-Sets, ein Experte braucht sechs: Bei Modellen mit
  Hunderten aktiver Experten in einem Prefill-Chunk wird der Batch vor jedem Layer, jedem
  Experten und dem Kopf geteilt, wenn der Platz knapp wird (`TInferVulkan.BatchRoom`,
  `TQwen3Vulkan.NeedRoom`).

*Tests:* `BindsAtOffset` (Ausschnitt ab Index 0, Bytes davor und dahinter unverändert;
falsch ausgerichtet, zu lang, überlaufender Versatz und falsche Längen werden abgelehnt),
`TinyMoeMatchesCpu` wie auf CUDA, dazu der GGUF-Fall mit höchstens 40 Dispatches je Batch
(die Experten eines Layers über mehrere Batches) und die Prüfung, dass das GGUF einen Puffer
je Layer nimmt (`ExpertBuffers`), `Qwen3MoeGgufMatchesCpu` (Qwen3-30B-A3B gegen die schnelle
CPU, Token für Token und mit der Expertenwahl der CPU, siehe unten). Grün auf der RTX 5090
(FPC, Delphi 13, Delphi 7) und der Adreno X1-85. Gegenprobe: dreizehn eingebaute Fehler
(Shader ohne Router-Gewicht, Zeile des Blocks statt der Token-Zeile, `First` übergangen,
Ausgabe nicht geleert, alle Experten am Versatz 0, `RecordLinearAt` ohne Versatz, `Bind` ohne
Versatz, `Bind` ohne Prüfung der Ausrichtung, Batch nie geteilt, GGUF nie ein Puffer,
Router-Gewichte nicht übernommen, Router-Logits der ersten Zeile für alle, Gewichte aus den
Token-Zeilen kopiert) - alle scheitern. Review: Codex fand das Limit der Deskriptor-Sets,
einen Überlauf in der Bereichsprüfung und einen Kopf in mehr Teilen als ein sehr kleiner Batch
fasst (jetzt eine klare Meldung), der Claude-Finder die Ausrichtung unter 4 Byte und dass der
Test den Ein-Puffer-Weg nicht nachwies; alles behoben.

*Die Expertenwahl im Vergleich vorgegeben (`TDecoderRouteTape`).* Vorher maß der 30B-Vergleich
Rechengenauigkeit und Routing in einem Kosinus: kippt bei einer knappen Router-Entscheidung
ein Experte, weichen die Logits ab, ohne dass etwas falsch ist (Schwelle damals 0,98 und 14
von 16). Jetzt zeichnet `TDecoderModel.Route` beim CPU-Lauf Experten und Gewichte jedes Tokens
und Layers auf (`drmRecord`); beim Lauf auf dem Gerät rechnet `Route` die eigene Wahl aus den
Logits des Geräts, vergleicht sie und gibt die aufgezeichnete weiter (`drmReplay`). Ein Lauf
prüft so beides getrennt: die Logits nur noch die Rechnung (Kosinus mindestens 0,99, Argmax
mindestens 15 von 16), die abweichenden Wahlen das Routing - erlaubt nur bei knappem Abstand
zwischen k-tem und bestem ausgelassenem Router-Logit (höchstens 0,1). Gemessen auf CUDA und
Vulkan: unter FPC Kosinus 0,9995, Argmax 16 von 16, 162 von 1344 Wahlen anders bei höchstens
0,052; unter Delphi 13 0,9975, 15 von 16, 121 anders bei höchstens 0,071 - die übrige
Abweichung unter Delphi kommt aus seiner CPU-Rechnung (Backlog B13), nicht aus dem Routing.
`RouteTapeReplays` prüft das Band auf der CPU: dieselben Tokens wiedergegeben geben dieselben
Bits; die Wahl anderer Tokens erzwungen zählt Abweichungen und ändert die Logits; ein leeres
Band löst eine Exception aus; Route direkt gerufen gibt Experten und Gewichte der
Aufzeichnung bitgleich aus. Weil die Router-Logits einer falschen Zeile für sich plausibel sind
(kleiner Abstand), gilt zusätzlich: höchstens ein Viertel der Wahlen anders (gemessen 12 und
9 %, mit diesem Fehler 47 %). Gegenprobe des Bands: sechs Fehler (eigene Wahl statt der
aufgezeichneten weitergegeben, eigene Gewichte, Abweichungen nicht gezählt, Abstand vom
höchsten statt vom k-ten Logit, Router-Logits der ersten Zeile auf CUDA und auf Vulkan) - alle
scheitern; der mit den eigenen Gewichten überlebte zuerst, deshalb der direkte Test von Route.
Review: Claude-Finder und Codex fanden ein `RouteTape := nil` auf einem nicht geladenen Modell
im `finally` der Tests und einen Kommentar, der mehr behauptete als geprüft (jetzt: Aufzeichnen
lässt die Logits bitgleich); behoben.

*Qwen3-30B-A3B Q4_K_M auf der RTX 5090 (Vulkan):* hochgeladen in 2,3 s, Spitze 17,9 GB;
Prefill 489 Tokens/s bei 200 Tokens Prompt, 642 bei 2000; Decode 43 bis 44 Tokens/s. CUDA im
selben Lauf: Prefill 383, Decode 41 - Vulkan ist hier vorn, weil ein MoE-Layer auf CUDA drei
Wartepunkte hat (Router herunter, zwei Uploads), auf Vulkan einen. Auf dem M2 Pro (MoltenVK)
ist die volle Suite grün (2942 Assertions, das Tiny-MoE auf der GPU).

**D1b erledigt (2026-10-19): Qwen2 / Qwen2.5 als Familie.**

- `TQwen2Model` (`model_type` und GGUF-Architektur `qwen2`, nachgeschlagen in
  `modeling_qwen2.py`): derselbe Decoder wie Qwen3 mit einem Bias auf den Q-, K- und
  V-Projektionen (keiner auf O) und ohne RMSNorm je Kopf. `config.json` nennt kein `head_dim`
  und die GGUF-Datei keine `attention.key_length`: Hidden durch Köpfe, mit Prüfung. Qwen3
  verlangt `head_dim` weiter (`TQwen3Model.HeadNorm`).
- `TDecoderLayer` trägt `QBias`, `KBias`, `VBias` (nil ohne); die CPU gibt sie an `Linear`,
  die Head-Norms entfallen, wenn `QNorm` nil ist. Vulkan (`RecordLinear` mit Bias) und CUDA
  (`EnqueueLinear(At)` mit Bias) laden die Biases in `UploadWeights` hoch, damit weder ein
  Batch noch ein CUDA-Graph hochladen muss; `qkrope.comp` und der CUDA-Kernel `qkrope` haben
  einen Schalter `norm` (0: q und k bleiben, `qn`/`kn` werden nicht gelesen).
- `TQwen2ChatTemplate`: das Template von Qwen2.5 (Revision `7ae5576…`): immer ein System-Turn
  (ohne System-Nachricht der Standardtext von Qwen), Werkzeuge darin, Argumente eines
  Tool-Aufrufs durch `tojson` (`PythonJson`), `<tool_response>` als Text - Qwen2.5 hat dafür
  kein Token. `TQwen2OutputParser`: der Parser von Qwen3 ohne `<think>`.
- Der GGUF-Loader nennt nicht unterstützte Typen beim Namen (`unsupported ggml type 6 (Q5_0)`).

*Referenz:* `tools/infer-ref/make_reference_qwen2.py` - `tiny-qwen2` (2 Layer, Bias ungleich
null, `head_dim` 16 aus 64/4, BF16, als GGUF Q8_0 ohne `key_length` und eine Datei ohne
`attn_k.bias`), `qwen25-chat` (die Fälle von `qwen3-chat` durch das Qwen2.5-Template),
`qwen25` (Qwen2.5-0.5B-Instruct FP32, sechs Prompts; `generate` mit `repetition_penalty` 1,
weil die `generation_config.json` 1,1 verlangt und `generate` sie auch ohne Sampling
anwendet).

*Tests:* `TinyQwen2MatchesReference` (alle Zustände, Greedy, GGUF, fehlender Bias abgelehnt),
`Qwen25MatchesReference` (Zustände und letzte Logits bis 2e-3, Greedy aller sechs Prompts
identisch), `Qwen25Gguf` (Qwens Q8_0: Kosinus 0,990 bis 0,995, größter Fehler 2,6 - mehr als
bei Qwen3-0.6B; Greedy gleich bis zu einer Stelle, an der die Referenz weniger als 3 sicher
war; Q4_K_M abgelehnt, siehe Backlog B16), `Qwen25ChatTemplateMatchesReference` (gleiche Ids
und gleicher Text wie transformers; das Qwen3-Template lehnt den Tokenizer ohne `<think>` ab),
`Qwen25ChatClient` (`TNativeLlmClient`: „Paris“, ein Tool-Aufruf `get_weather` mit
`{"city": "Berlin"}`), auf Vulkan und CUDA `TinyQwen2MatchesCpu` und `Qwen25GgufMatchesCpu`
(Kosinus 0,999, Argmax 16 von 16 auf RTX 5090 und Adreno X1-85), der CUDA-Op-Test `qkrope`
auch ohne Norm. Grün auf der RTX 5090 unter FPC und Delphi 13 (3606 Assertions) und Delphi 7
(2174). Gegenprobe: dreizehn eingebaute Fehler (kein Q-Bias auf der CPU, Head-Norms nie,
head_dim aus Hidden / 2, `attn_v.bias` falsch benannt, kein Standard-System-Turn, Argumente
wörtlich, Assistent mit Aufrufen als Text, `<tool_response>` als Id, Parser hält auf einer
Think-Id, auf Vulkan kein K-Bias und immer die Head-Norm, auf CUDA kein Bias und nie die
Head-Norm) - alle scheitern. Review: Claude-Finder und Codex ohne Befund; der Finder vermisste den Test
für leere Argumente (`{}`), jetzt da.

**D7b erledigt (2026-10-21): der Prefill von Qwen3.5 auf allen Kernen.** Erst gemessen: Nach
D7a lagen im Prefill von 512 Tokens (Qwen3.5-0.8B, F32, RTX-Rechner) 5,8 s von 9,4 s in der
Rekurrenz und 1,6 s in der Convolution - beide seriell, der Rest lief schon auf allen Kernen.

- *Delta-Regel als Operation des Backends* (`TInferBackend.DeltaRule`): die Referenz in
  `TInferCpuBackend` rechnet Kopf für Kopf über alle Tokens (der Zustand eines Kopfes, 64 KB
  bei 128 x 128, bleibt im Cache, statt je Token alle 16 Köpfe zu durchlaufen) mit
  double-Summen wie D7a. `TInferCpuFastBackend` gibt jedem Value-Kopf ein Item auf dem Pool;
  mit `Exact` bitgleich zur Referenz, sonst der Float-Kern `InferDeltaRowsF32` (NEON und AVX2;
  je Zeile von S zwei Durchgänge: Zerfall und Vorhersage `S^T k`, dann Korrektur und Ausgabe
  `S^T q`). Eine key_dim, die kein Vielfaches von 16 ist, rechnet die Referenz.
- *Vorbereitung als Items* (`TDecoderModel.DeltaPrep`): je Q- oder K-Kopf (Convolution, silu,
  L2-Norm) und je Value-Kopf (seine V-Kanäle, beta, Zerfall, silu(z)) ein Item auf
  `Backend.RunItems`, jedes rechnet wie die serielle Schleife (bitgleich). Die Convolution läuft
  jetzt Token für Token entlang der Zeilen; vorher Kanal für Kanal quer über alle Zeilen, mit
  einem Sprung von 24 KB je Token. Der Zustand der Items liegt im Kontext, nicht im Modell:
  ein Modell bedient mehrere Kontexte zugleich.
- *Die Chunk-Form ist nicht übernommen.* Danach macht die Rekurrenz 3-4 % des Prefills aus.
  Die Chunk-Form von transformers (Matrixprodukte je Block von 64 Tokens) rechnet mehr und
  lohnt sich erst mit Tensor-Kernen - eine Option für D7e.
- *Messung* (RTX-Rechner, i9-14900K, 22 Threads, FPC 3.2.2, Bestwert aus drei Läufen,
  512 Tokens):

  | | D7a | D7b ohne Float-Kern | D7b |
  |---|---|---|---|
  | Prefill F32 | 9,36 s (55 tok/s) | 1,32 s | 1,10 s (465 tok/s) |
  | Prefill Q8_0 | 9,78 s (52 tok/s) | 2,05 s | 1,81 s (282 tok/s) |
  | davon Rekurrenz (F32) | 5,8 s | 0,26 s | 0,04 s |
  | Decode Q8_0 | 57,8 ms/Token | 39,9 ms/Token | 37,4 ms/Token |

  Auf dem Snapdragon X (WSL, 8 Threads) Q8_0 von 18,0 s auf 1,86 s. Die Werte dort streuen
  stark (Energiesteuerung des Laptops), deshalb gilt die Tabelle des RTX-Rechners.
- *Tests:* `TTestInferFast.DeltaRuleMatchesReference` (die Referenz gegen die ausgeschriebene
  Definition mit S in double; der exakte Fast-Backend bitgleich für Ausgabe und Zustand; der
  Float-Kern innerhalb 1e-5 + 1e-4 relativ; geteilte Key-Köpfe, Zeilen breiter als ihre Köpfe,
  ein einziger Value-Kopf, eine key_dim ohne volle Lanes; der Kern schreibt nichts über seine
  Zeilen hinaus). Das Tiny-Modell `tiny-qwen35/text` hat jetzt Value-Köpfe von 8 Werten für
  Keys von 16: Vorher hatten alle Testmodelle (auch Qwen3.5-0.8B) dk = dv, eine Verwechslung
  der beiden fiel nicht auf. Qwen3.5-0.8B bleibt in F32 bei 4e-5 (Logits); in Q8_0 sinkt der
  Kosinus der Logits im Chat-Fall auf 0,9984 (x86_64) bzw. 0,9988 (aarch64), Greedy
  identisch - die Schranke im Test jetzt 0,997 statt 0,998, damit sie nicht auf der Kante
  steht.
  Grün in WSL (FPC 3.2.4, 6156 Assertions; vier weitere scheitern nur an
  `LAI_REQUIRE_MODEL=1` ohne gpt-oss-20b auf dem Laptop), auf dem RTX-Rechner unter FPC und
  Delphi 13 (7853) und Delphi 7 (4936, das echte Modell braucht 64 Bit), auf dem M2 Pro
  (Infer 5737, LLM 730).
  Gegenprobe: 23 eingebaute Fehler (beta und Zerfall im Kern vertauscht, kein Zerfall in der
  Referenz, `Exact` übergangen, Köpfe gekachelt, kein silu(z), K skaliert, keine Items der
  Value-Köpfe, der letzte Kopf fehlt, die Zeilenbreite übergangen, V-, z- und Ausgabe-Offsets
  mit dk statt dv, beta und Zerfall roh, dazu acht Fehler aus D7a am umgebauten Code) - alle
  scheitern; drei davon erst mit dem neuen Tiny-Modell. Review: Claude-Finder und Codex in
  zwei Runden; behoben: ein Index auf ein leeres Feld bei Kernel 1, die Q8_0-Schranke auf der
  Kante, veraltete Angaben in README und Unit-Kopf.

**D7a erledigt (2026-10-21): Qwen3.5 auf der CPU (Gated DeltaNet).** Die Rechnung von
Qwen3.5 (dicht: 0.8B bis 27B) gegen `transformers` 5.19 (`modeling_qwen3_5.py`, nachgebaut, kein
Code übernommen), zunächst auf der CPU, seriell und aus safetensors. Modell:
`Qwen/Qwen3.5-0.8B` Revision `2fc06364715b967f1860aea9cf38778875588b17`.

- *Gated DeltaNet* (`dmDeltaNet`, `TDecoderModel.DeltaNetBlock`): `in_proj_qkv` durch eine
  kausale Depthwise-Convolution (`linear_conv_kernel_dim`, ohne Bias) mit silu; Q und K je Kopf
  L2-normiert (eps 1e-6), Q mal key_dim^-0,5; beta = sigmoid(`in_proj_b`), der Zerfall
  g = -exp(`A_log`) * softplus(`in_proj_a` + `dt_bias`); je Value-Kopf eine Matrix
  S [key_dim, value_dim]: S mal exp(g), dann die Delta-Regel S += k (beta (v - S^T k))^T,
  gelesen mit q; ein Key-Kopf dient mehreren benachbarten Value-Köpfen (`repeat_interleave`).
  Danach RMSNorm je Value-Kopf mal silu(`in_proj_z`) und `out_proj`. Gerechnet wird Token für
  Token (die rekurrente Form), auch im Prefill - transformers nimmt dort die Chunk-Form; beide
  liegen bei den Tiny-Modellen unter 1e-5 auseinander. (D7b hat die Chunk-Form nicht
  übernommen, siehe dort.)
- *Zustand im Kontext:* je DeltaNet-Layer die letzten `kernel - 1` Eingaben der Convolution und
  die Matrizen S; keine KV-Zeilen für diese Layer. Der Zustand steht für die ganze Sequenz:
  `Restorable` nur 0 und `Length`, `Reuse` rechnet sonst von vorn. Ein `Forward`, der in den
  Layern scheitert, lässt den Zustand halb fortgeschrieben - dann ist nur noch 0 erreichbar,
  und ein weiteres `Forward` ohne `Reset` wird abgelehnt. Rücksetzpunkte (Checkpoints des
  Zustands für den Prompt-Cache) kommen mit D7d.
- *Volle Attention mit Ausgangsgate* (`Spec.OutputGate`): `q_proj` liefert je Kopf erst die
  Query, dann ebenso viele Gate-Werte; die Attention-Ausgabe mal sigmoid(gate) vor `o_proj`.
  *Teil-RoPE* (`Config.RopeDims`): nur die ersten int(head_dim * `partial_rotary_factor`) Werte
  jedes Kopfes drehen (rotate_half innerhalb dieses Teils, die Frequenzen über diesen Teil);
  `TInferBackend.Rope` hat dafür `aRotDim`. Die multimodalen Abschnitte (`mrope_section`)
  ändern für Text nichts: alle drei Positionen eines Tokens sind gleich.
- *Familie* (`mormot.ai.infer.model.qwen35`): `TQwen35Model` (`qwen3_5`, die Einstellungen in
  `text_config`, die Tensoren unter `model.language_model.`; Bildteil und `mtp.*` bleiben
  liegen) und `TQwen35TextModel` (`qwen3_5_text`, unter `model.`). Jede RMSNorm außer der
  gegateten rechnet mal (1 + Gewicht): beim Laden in float32 eingefaltet, -exp(`A_log`)
  ebenso. Der Head ist gebunden, wie das oberste `tie_word_embeddings` sagt (transformers
  liest bei `qwen3_5` nur dieses, Vorgabe false; geprüft). Beim Quantisieren bleiben
  `in_proj_b` und `in_proj_a` F32. Abgelehnt: eine andere Aktivierung als silu,
  `attention_bias`, `attn_output_gate` false, `mlp_only_layers`, ein anderer `rope_type`, ein
  unbekannter RoPE-Schlüssel, ein ungerader drehender Teil, kein `rope_theta`. Die RoPE-Werte
  liest es wie transformers (`convert_rope_params_to_dict`): `rope_scaling` statt
  `rope_parameters`, wenn gegeben und nicht leer; `type` statt `rope_type`; was dort fehlt, von
  oben (`partial_rotary_factor`, `rope_theta`); ein Modell nur aus voller Attention ist
  zulässig.
  GPU-Geräte lehnen Qwen3.5 ab (Mixer, `dxOutputGate`, `dxPartialRope`).
- *Referenz* (`make_reference_d7.py`): `tiny-qwen35` (Layout der veröffentlichten Checkpoints
  mit Bildteil und einem `mtp`-Tensor daneben, drei DeltaNet-Layer und eine volle, BF16) und
  `tiny-qwen35/text` (`Qwen3_5ForCausalLM`: eigener Head, zwei Value- je Key-Kopf, Kernel 3,
  halber Kopf dreht) - je Fall alle Zustände, die Logits eines Durchgangs und Token für Token
  mit Cache, Greedy mit und ohne Cache gleich, dazu der Beleg, dass Rekurrenz und Gate wirken;
  `qwen35-0.8b` (vier Prompts, letzte Zeilen aller Zustände, Logits, Greedy).
- *Tests* (`test.infer.qwen35`): Tiny alle Zustände bis 1e-4, Token für Token bitgleich zum
  Prefill und in zwei Stücken, Embedding-Zeilen, Greedy; der Zustand (Rücksetzpunkte, Zweig,
  Fortsetzen, gescheiterter `Forward`); schneller Pfad, Q8_0 und Q4_0; die Ablehnungen;
  Qwen3.5-0.8B in F32: Logits gemessen bis 4e-5 genau (Schranke im Test 1e-3), Greedy in
  allen vier Fällen identisch; in Q8_0 (beim Laden) Kosinus der Logits gemessen 0,9990 bis
  0,9999 (Schranke 0,998), Greedy identisch. Die Hilfen der Decoder-Tests
  stehen jetzt in `TTestInferDecoder` (`CheckLastRows` für die letzten Zeilen eines echten
  Modells).
  Grün in WSL (FPC 3.2.4), auf der RTX 5090 unter FPC und Delphi 13 (7814 Assertions) und
  Delphi 7 (4900, das echte Modell braucht 64 Bit) und auf dem M2 Pro (5698).
  Gegenprobe: 25 eingebaute Fehler (Convolution-Gewichte verkehrt, kein silu, keine L2-Norm,
  Q nicht skaliert, kein Zerfall, kein beta, Köpfe gekachelt statt verschränkt, kein Gate,
  Query und Gate vertauscht, Frequenzen über den ganzen Kopf, Norm ohne 1 +, Zustand oder
  Convolution nicht genullt, beliebige Rücksetzpunkte, ein halb fortgeschriebener Zustand
  angenommen, kein silu(z), Head immer gebunden, kein Schieben der Convolution, positiver
  Zerfall, kein `dt_bias`, die gegatete Norm mit 1 +, RoPE über den ganzen Kopf, der Faktor
  von oben übergangen, oben vor `rope_parameters`, die Bindung aus `text_config`) - alle
  scheitern. Review: Claude-Finder und Codex in zwei Runden; behoben: `partial_rotary_factor`
  und `rope_theta` von oben, `rope_scaling` (auch leer) und `type`, ein Modell nur mit voller
  Attention, ein ungeschütztes `Exp` bei beta, ein überlappendes `MoveFast`, ein fehlender
  Test der Bindungsregel, `.D[] :=` in einem Test (der AArch64-Fehler von FPC auf dem Mac).

**D6c erledigt (2026-10-21): Gemma 4 als Chat-Modell (Tokenizer, Template, Parser).** Gemma 4
hinter `ILlmClient`, zunächst auf der CPU und aus safetensors (die echten Gewichte kommen mit
D6b). Quelle: `google/gemma-4-26B-A4B-it` Revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
(`tokenizer.json`, `chat_template.jinja`); `gemma-4-31B-it` und `gemma-4-12B-it` haben dieselben
Dateien (gleiche SHA-256). Nachgebaut, kein Code übernommen.

- *Tokenizer* (`mormot.ai.infer.tokenizer.spbpe`, `TSpBpeTokenizer`): BPE im Stil von
  SentencePiece - jedes Leerzeichen wird U+2581 (Replace-Normalizer), kein Schnitt vor dem BPE
  (der Split auf " " findet danach keins mehr), der ganze Text ist ein Wort; die Symbole sind die
  Zeichen des Vokabulars, ein fremdes Zeichen seine UTF-8-Bytes als `<0xNN>` (Byte-Fallback,
  alle 256 Byte-Tokens Pflicht, also nie `<unk>`); Merges nach Rang, bei Gleichstand links
  zuerst. Decodieren: U+2581 als Leerzeichen, `<0xNN>` als Byte. Abgelehnt: eine andere
  Pipeline, `truncation`/`padding` (die Bibliothek würde damit kürzen), ein Vokabeltext mit
  zwei Ids, ein Merge außerhalb des Vokabulars oder doppelt. Ungültiges UTF-8 wird je Byte
  U+FFFD. GGUF folgt mit D6d.
- *Template* (`TGemma4ChatTemplate` in `mormot.ai.infer.chat`): `<bos>`, ein System-Turn bei
  Thinking, Tools oder einer ersten System-Nachricht (`<|think|>`, die getrimmten Anweisungen,
  jedes Tool als `<|tool>declaration:NAME{...}<tool|>`), dann die Turns. Ein Assistenten-Turn
  schreibt sein Reasoning als `<|channel>thought ... <channel|>` (nur nach der letzten
  Nutzer-Nachricht, mit `preserve_thinking` auch vor Aufrufen), die Aufrufe als
  `<|tool_call>call:NAME{key:value}<tool_call|>`, die Ergebnisse der folgenden Tool-Nachrichten
  als `<|tool_response>response:NAME{value:...}<tool_response|>`, danach den Inhalt ohne
  Channel-Blöcke; der Modell-Turn läuft nach den Ergebnissen weiter. Strings stehen zwischen
  `<|"|>`-Tokens, Schlüssel in der Reihenfolge von Jinjas `dictsort` (Python `lower()` für
  ASCII, Latin-1, Latin Extended-A, Griechisch, Kyrillisch - ein anderer Großbuchstabe im
  Schlüssel wird abgelehnt statt geraten, ebenso ein großes Sigma, dessen Kleinbuchstabe vom
  Wortende abhängt), Typnamen groß (nur ASCII), Zahlen wie Python.
  Thinking ist aus, solange eine Anfrage es nicht verlangt (`TChatTemplate.DefaultEnableThinking`,
  für Gemma false; der Client übernimmt es als Vorgabe). Abweichungen mit Absicht:
  Steuer-Tokens als Ids, jeder Text bleibt Text; eine Tool-Nachricht ohne Aufruf davor wird
  abgelehnt (das Template lässt sie still fallen); was das Template als Python-repr schriebe
  (Liste als Beschreibung oder Typ) oder nicht nehmen könnte, wird abgelehnt.
- *Parser* (`TGemma4OutputParser` in `mormot.ai.llm.native`): der Thought-Channel wird
  Reasoning (die Zeile `thought` fällt weg; ein anderer, leerer oder abgeschnittener Kanal
  bleibt Inhalt, wie bei transformers, und ein Steuer-Token danach wirkt wie im Inhalt), jeder `<|tool_call>`-Block ein Aufruf, wenn seine Argumente
  im Dialekt von transformers lesbar sind (`ParseGemma4ToolCall`: Strings zwischen den
  `<|"|>`-Tokens, ein `\w+`-Schlüssel direkt nach `{` oder `,` in Anführungszeichen, dann
  striktes JSON, ein Objekt; `\w` als Teilmenge des regex-Moduls - eine Nicht-ASCII-Ziffer
  etwa nicht -, so wird nie ein Aufruf angenommen, den transformers ablehnt), sonst bleibt
  er Text; `<turn|>`, `<|tool_response>` und `<eos>`
  halten an. Neu `TNativeOutputParser.Prompt`: `NativeDecode` gibt dem Parser den Prompt, denn
  nach Tool-Ergebnissen mit Thinking endet der Prompt in einem offenen Thought-Channel.
- *Client:* `TGemma4Model` und `TGemma4TextModel` sind als Familie registriert
  (`TSpBpeTokenizer`, `TGemma4ChatTemplate`, `TGemma4OutputParser`); `preserve_thinking` aus
  `chat_template_kwargs` (`TChatTemplateOptions.PreserveThinking`).
- *Referenz* (`make_reference_d6c.py`): `gemma4-tokens` - 3197 Fälle (Korpus, die Fälle der
  früheren Tokenizer, eigene für Leerzeichenläufe, Byte-Fallback, Literale, 1200 erzeugte),
  Ids der tokenizers-Bibliothek ohne Added Tokens; `gemma4-chat` - 452 Gespräche und
  Tool-Schemas (52 entworfen, 400 erzeugt über die Zweige von `format_parameters`), Text und Ids
  von transformers 5.19.
- *Tests* (`test.infer.gemma4`): die Pipeline mit Tiny-Tokenizern (Merges über das Leerzeichen,
  Byte-Fallback, Merges von Byte-Tokens, ungültiges UTF-8, die Ablehnungen), alle
  Tokenizer-Fälle id- und textgleich, alle Chat-Fälle textgleich (Tiny-Tokenizer, ohne Modell)
  und idgleich (61 Fälle mit einem Steuer-Token-Literal im Text nur textgleich: dort liest der
  HF-Tokenizer ein Steuer-Token, wir Text - geprüft, dass unsere Ids den Text ergeben und
  weniger Steuer-Tokens enthalten), die Ablehnungen und die `dictsort`-Tabelle, der
  Argument-Dialekt, der Parser (Kanal, Aufrufe, Stopps, Schnitte, offener Kanal im Prompt), die
  Optionen und die registrierte Familie.
  Gegenprobe: 24 eingebaute Fehler (Leerzeichen nicht ersetzt, Byte-Fallback falsch,
  U+2581 nicht decodiert, falsche Position in der Merge-Queue, `dictsort` mit Großschreibung,
  kein Trim, Generation-Prompt ohne leeren Kanal, kein fortgesetzter Modell-Turn, Tool-Name
  immer `unknown`, Reasoning vor der letzten Nutzer-Nachricht, Komma vorn, `preserve_thinking`
  aus, Schlüssel ohne Anführungszeichen, `<|tool_response>` hält nicht, Prompt-Kanal
  übersehen, Kanalzeile im Reasoning, Thinking-Vorgabe an, Option verloren, verwaiste
  Tool-Nachricht geschluckt, jeder Kanal als Reasoning, großes Sigma geraten, Zahlen als
  `\w`, Literal als Steuer-Token) - 23 scheitern; der übrige (Zahlen über `json.dumps` statt
  `str()`) ist gleichwertig, Python schreibt beide gleich. Review: Claude-Finder und Codex in
  vier Runden; gefunden und behoben u. a. Byte-Literale, die unter Unicode-Delphi falsch
  kodieren, `truncation`/`padding` und doppelte Vokabeltexte, ein anderer oder leerer Kanal
  als Reasoning, ein ungültiger Kanal-Kopf, der einen Aufruf verschluckte, `\w` laxer als
  transformers, das Schluss-Sigma, ungeprüfte Literal-Fälle.

**D6a erledigt (2026-10-20): Gemma 4 auf der CPU.** Die Rechnung von Gemma 4 (26B-A4B, 31B)
gegen `transformers` 5.19 (`modeling_gemma4.py`, nachgebaut, kein Code übernommen), zunächst auf
der CPU und aus safetensors.

- *Decoder:* Die Layer dürfen jetzt verschieden geformt sein - `Spec.HeadDim` und `Spec.KvHeads`
  je Layer (Gemma 4: Sliding-Layer mit `head_dim` 256, volle Layer mit `global_head_dim` 512 und
  eigenen KV-Köpfen), der KV-Cache mit eigener Breite je Layer. `Spec.KIsV`: V ist die
  K-Projektion (`attention_k_eq_v`, kein `v_proj`), roh vor Norm und Rotation übernommen. Ein
  zweiter Rotationssatz (`Spec.Rope2`, `Rope2Theta`, `Rope2Partial`): die „proportionale“ RoPE
  der vollen Layer - die Frequenzen über die ganze Kopfdimension, nur der Anteil
  `partial_rotary_factor` dreht, der Rest hat die Frequenz 0. `AttentionScale` (Gemma: 1, die
  Köpfe sind normiert; `CausalAttention` hat dafür `aScale`), `ValueNorm` (RMSNorm ohne Gewicht
  über jeden V-Kopf), `SandwichNorm` (Norm nach der Attention und nach dem FFN, jeweils vor dem
  Residual), `Activation` mit GeGLU (`gelu_pytorch_tanh`, neue Backend-Operation
  `GeluTanhGlu`), `EmbedScale` (sqrt(hidden) in float32), `LogitSoftcap`
  (tanh(x / c) * c), `Scalar` je Layer (der ganze Zustand mal `layer_scalar`, jeder Wert).
- *MoE von Gemma 4* (`dfDenseMoe`): der dichte GeGLU-Block und die Experten nebeneinander auf
  demselben Residual - dicht auf der Pre-FFN-Norm, die Experten auf eigener Norm, jede Ausgabe
  normiert, die Summe noch einmal; der Router auf einer RMSNorm ohne Gewicht mal `router.scale`
  und hidden^-0,5 (in ein Normgewicht gefaltet), Softmax über alle Experten, Top-k renormiert,
  jedes Gewicht mal `per_expert_scale` (`Route(..., aExpertScale)`). Die Experten liegen als
  `gate_up_proj [E, 2I, H]` (erst Gate-, dann Up-Zeilen) und `down_proj [E, H, I]` vor; F32 zeigt
  in die einmal verbreiterten Tensoren, Q8_0/Q4_0 quantisieren je Experte die Zeilen der Scheibe
  (`QuantizedRows`).
- *Familie* `mormot.ai.infer.model.gemma4`: `TGemma4Model` liest das Layout der veröffentlichten
  Checkpoints (`Gemma4ForConditionalGeneration`, `model_type` gemma4, Einstellungen in
  `text_config`, Tensoren unter `model.language_model.`; Bild- und Audioteil bleiben liegen),
  `TGemma4TextModel` das von `Gemma4ForCausalLM` (`gemma4_text`). `per_layer_config` (so schreibt
  `save_pretrained`) ersetzt `global_head_dim` und `num_global_key_value_heads` ganz, wie in
  `Gemma4TextConfig`; der letzte Layer ist immer voll. Abgelehnt: Per-Layer-Embeddings und
  geteilter KV (E2B, E4B), ein ungebundener Kopf, Bias, bidirektionale Attention über alles,
  eine andere Aktivierung oder RoPE-Art, ein RoPE-Eintrag `null` (transformers baut dann keinen
  Satz und scheitert im Durchlauf). Die GPU-Geräte lehnen jedes Extra jenseits von gpt-oss ab
  (`DECODER_EXTRAS_GPTOSS`), bis D6e kommt.
- *Referenz* (`make_reference_d6.py`): `tiny-gemma4` - 4 Layer (Sliding mit Fenster 5, voll,
  Sliding, voll), Kopfdimension 16 (Sliding, 2 KV-Köpfe) bzw. 32 (voll, 1 KV-Kopf, K=V),
  die proportionale RoPE, dichter
  Block neben 8 Experten (Top-3, der Router scharf genug: Abstand zwischen k-tem und
  (k+1)-tem Logit mindestens 0,022), Softcap 3 (greift: größtes Roh-Logit 3,02), Skalare
  0,6-1,4; `tiny-gemma4/dense` - 6 Layer aus den Vorgaben (ohne `layer_types` und
  `rope_parameters`), ohne Experten, K≠V, `config.json` so, wie `save_pretrained` sie schreibt
  (`per_layer_config`).
- *Tests:* `Gemma4MatchesReference` (jeder Zustand bis 1e-4 bei Kosinus 0,99999999999, beide
  Rotationssätze, Cache und zwei Chunks bitgleich zum ganzen Durchlauf, Zeilen aus `Embed`
  geben die Logits der Ids, Greedy über 24 Tokens gleich, ein Skalar 0 nullt den Zustand; das
  dichte Modell ebenso), `Gemma4Quantized` (jede quantisierte Expertenzeile Byte für Byte die
  Quantisierung der F32-Zeile; mit der Expertenwahl des F32-Laufs nachgespielt: Q8_0 bis 0,64
  bei Kosinus 0,989, Q4_0 Kosinus 0,705 - das Tiny-Modell ist chaotisch, normierte Köpfe mit
  Skala 1 geben eine scharfe Softmax), `Gemma4Refused` (die Ablehnungen; angenommen:
  `per_layer_config` nach Layer-Art, ein letzter Sliding-Layer wird voll und gibt die Zustände
  der Referenz; `GeluTanhGlu` des schnellen Backends parallel bitgleich zur Referenz).
- Grün auf dem RTX-Rechner unter FPC und Delphi 13 (6992 Assertions) und Delphi 7 (4521), unter
  WSL (5299) und auf dem M2 Pro (5345; LLM-Suite 730). Gegenprobe: 24 eingebaute Fehler (V-Norm,
  Skala 1, die Norm nach der Attention, nach dem FFN, nach dem dichten Block, nach den Experten,
  der Router auf der falschen Norm, `per_expert_scale`, der Layer-Skalar, ein Skalar 0 als
  „keiner“, die Embedding-Skala, das Softcap, SwiGLU statt GeGLU, alle Paare drehen, die
  Exponenten über den gedrehten Anteil, eine KV-Breite für alle Layer, die GELU-Konstante,
  `NormTopK` aus, der letzte Layer nicht voll, K=V aus, `per_layer_config` übergangen, Gate- und
  Up-Scheibe vertauscht, die Router-Wurzel vergessen, V aus Q kopiert) - alle scheitern. Review:
  Claude-Finder und Codex fanden je den Skalar 0 als „keiner“ und das übergangene
  `per_layer_config` (so schreibt `save_pretrained`), der Finder dazu die fehlende Prüfung der
  Backend-Operation und schwache Schwellen; im Nachreview Codex, dass ein `per_layer_config: null`
  die globalen Schlüssel schon abschaltet und ein RoPE-Eintrag `null` in transformers scheitert
  (jetzt abgelehnt), der Finder den fehlenden Test für einen Skalar 0. Behoben. Der Mac fand
  nebenbei, dass der D5e-Test `TThreadID` mit 0 verglich (unter Darwin ein Zeiger).

**D5e erledigt (2026-10-20): Issue #1 (flydev) - normierte Zeile vom Gerät, Stapel ohne Kopf,
paralleles Laden.** flydev hat `mormot2-qwen-tts` auf `21aa4bb` gezogen und drei Dinge gemeldet
(<https://github.com/landrix/mormot2-ai/issues/1>): Sein Nachbau des Vulkan-Durchlaufs brach
mit den neuen Push-Konstanten still (falsche Codes, kein Fehler); der Talker braucht auf dem
Gerät die normierte letzte Zeile (die Eingabe des Code-Predictors), und der Predictor nimmt je
Schritt einen anderen seiner 15 Köpfe; `CreateStack` konvertiert auf einem Thread (Q8_0: 6,1 s
gegen 0,56 s mit seinem Pool).

- `TDecoderModel.ForwardEmbeds(..., aNormed)` (und `TDecoderDevice.ForwardEmbeds`,
  `TQwen3Vulkan`, `TQwen3Cuda`): `aNormed <> nil` bekommt die letzte Zeile nach der finalen
  RMSNorm (`[Hidden]`) - genau das, was der Kopf multipliziert. Mit den Logits der letzten
  Zeile, mit denen jeder Zeile (dann wird jede Zeile normiert und die letzte gelesen) oder ohne
  Logits; auf dem Gerät nur im letzten Chunk, auf Vulkan im selben Rücklesepuffer hinter den
  Logits, auf CUDA ein Download aus dem Normpuffer (ein Decode-Schritt bleibt CUDA-Graph; ein
  Wechsel zwischen mit und ohne Zeile baut den Graph einmal neu). Damit braucht flydev den
  nachgebauten Vulkan-Durchlauf nicht mehr.
- `CreateStack(..., aHead = '')`: ein Stapel ohne Kopf (`HasHead` false) - keine Logits, nur
  die normierte Zeile; eine Anfrage mit Logits wird abgelehnt, bevor sich der Cache ändert
  (CPU und Gerät). Ein solcher Stapel lädt keinen Kopf, auch keinen gebundenen; das Gerät lädt
  keinen hoch. `aEmbedding` und `aHead` dürfen beide leer sein (der Predictor von Qwen3-TTS).
- `TInferBackend.RunItems(aJob, aCount)`: unabhängige Einträge auf den Threads des Backends,
  wenn es einen freien Pool hat (`TInferCpuFastBackend` und damit Vulkan und CUDA), sonst der
  Reihe nach. Das Laden nutzt es: Q8_0 und Q4_0 quantisieren Zeile für Zeile auf dem Pool, und
  jede Zeile Block für Block über 32 floats auf dem Stack (kein Zeilenpuffer, kein Heap in den
  Threads - flydevs Hinweis); BF16 und F16 werden in Stücken von 4096 Werten verbreitert. Die
  Bytes sind dieselben wie auf einem Thread (Q8_0 und Q4_0 quantisieren jeden Block für sich);
  ein NaN wird weiter mit Name, Zeile und Spalte abgelehnt, vom Pool als `EInferError` mit
  derselben Meldung.
- *Gemessen* (Talker und Predictor von Qwen3-TTS-12Hz-0.6B, BF16-Checkpoint, zweimal geladen):
  auf dem RTX-Rechner (i9, 22 Threads) Q8_0 5,3-6,4 s vorher, 0,37-0,38 s nachher; F32
  0,65-1,2 s vorher, 0,38-0,43 s nachher. Auf der Snapdragon X unter WSL (8 Threads, Modell im
  Linux-Dateisystem) Q8_0 10-16 s vorher, 2,8-5,9 s nachher.
- *Tests:* `StackFromCheckpoint` (die normierte Zeile ist bitgleich der Zustand `norm` der
  letzten Zeile, mit und ohne Logits, die Logits unverändert; ein Stapel ohne Kopf gibt
  dieselben Bits, lehnt Logits ab, der Cache bleibt), `LoadOnAllCores` (F32, Q8_0 und Q4_0 aus
  safetensors und F16 aus GGUF auf dem Pool Byte für Byte wie auf einem Thread; ein NaN in der
  letzten Zeile vom Pool abgelehnt, mit Zeile und Spalte), auf Vulkan und CUDA
  `NormedRowOnDevice` (mit den letzten Logits, mit allen, ohne, zwei Einzelschritte, über zwei
  Chunks, Stapel ohne Kopf, gegen die CPU) und in `Qwen3TtsOnDevice` die normierte Zeile des
  Talkers gegen PyTorch und der Predictor ohne Kopf Schritt für Schritt, seine 15 Köpfe auf der
  CPU angewandt, gegen flydevs Referenz (alle Greedy-Codes gleich).
- Grün auf dem RTX-Rechner unter FPC und Delphi 13 (6776 Assertions) und Delphi 7 (4305), unter
  WSL mit Lavapipe (5083) und auf dem M2 Pro (MoltenVK, 5129; LLM-Suite 730). Gegenprobe: 15
  eingebaute Fehler (`RunItems` nie auf dem Pool, falscher Block- und Stück-Offset, Rest zu kurz,
  falscher Quelltyp, die erste statt der letzten Zeile auf der CPU, die Norm ohne Logits
  übersprungen, Logits ohne Kopf durchgelassen, auf Vulkan und CUDA Zeile 0 bei allen Logits,
  die Norm ohne Logits übersprungen und die Zeile an den ersten statt den letzten Chunk, auf
  Vulkan vom Anfang des Rücklesepuffers gelesen) - alle scheitern. Review: Codex ohne Befund;
  der Claude-Finder bemerkte, dass `LoadOnAllCores` auf einem Kern scheiterte und nicht zeigte,
  dass der Pool wirklich rechnet (jetzt: Überspringen, 64 Einträge zu 1 ms, die auf anderen
  Threads laufen müssen), Codex im Nachreview, dass ein nie ausgeführter Eintrag als fremder
  Thread zählte (jetzt: jeder Eintrag muss laufen). Behoben.

**D5d2 erledigt (2026-10-20): gpt-oss auf Vulkan.** `TQwen3Vulkan` nimmt gpt-oss an wie
`TQwen3Cuda` seit D5d1.

- *Sinks:* `attention.comp`, `flash.comp` und `attndec.comp` binden die Sinks als Binding 5
  (Push `hasSinks`; ohne wird ein beliebiger Puffer gebunden), `attnmerge.comp` als Binding
  2 - im Maximum und im Nenner wie auf CUDA und der CPU.
- *Biases:* O und der Router über `RecordLinear`; die Experten beim Host-Weg über
  `RecordLinearAt(..., aBias, aBiasFirst)`, beim Gerät-Weg über `RecordLinearIds(...,
  aBias)`. Die Zeile eines Experten liegt nicht an einem Bindungs-Versatz (der müsste auf
  `minStorageBufferOffsetAlignment` liegen, beim schmalen Modell sind es 64 Byte je Zeile),
  sondern als Index im Push: `biasOff` in `linear`/`linear8`/`gemm8`/`matvec8`,
  `ids[slot] * nout` in `linear8 -DIDS`. `TLinearPush` hat dafür `BiasOff`.
- *SwiGLU:* `swiglu.comp` mit `alpha` und `limit` im Push (`TSwiGluPush`), per Vergleich
  begrenzt wie die CPU.
- *MXFP4:* Typ 6 in `linear8`/`gemm8` (und `linear`, dort nicht erreichbar: MXFP4 geht immer
  über `linear8` oder `gemm8`, wie Q5_0); `matvec8` bleibt bei Q8_0/K-Quants. Bei den
  Exponenten 0 und 1 ist die Skala `2^(e - 128)` subnormal, und Vulkan darf einen
  subnormalen Operanden ohne `DenormPreserve` auf 0 setzen - auch wenn das Gewicht selbst
  normal ist (4 · 2^-128 = 2^-126). Dort rechnen die Shader `ldexp(k * 0.25, e - 126)`
  (exakt für jeden Wert der Tabelle, der Exponent nicht unter -126, wo SPIR-V `Ldexp` auch
  auf 0 setzen darf). Eine Grenze bleibt: Gewichte, die selbst subnormal sind (e 0 mit k 1
  bis 3, e 1 mit k 1, um 10^-38), darf solche Hardware beim Produkt weiter auf 0 setzen.
- *Genauigkeit:* gpt-oss-20b gegen transformers: Kosinus 0,9999999 für Text, 0,999997 /
  max. 0,048 für den Chatfall, alle 16 Greedy-Tokens der vier Fälle gleich.
- *Tempo (RTX 5090, GGUF, Prompt 200):* Prefill 1120 Tokens/s (1413 bei 2000), Decode 127;
  mit Wahl auf dem Host 83.
- *Tests:* `LinearMatchesReference` mit MXFP4; `GptOssMatchesCpu` (aus `GptOssRefused`) wie
  auf CUDA, dazu Q8_0 beim Laden über `matvec8.comp` (eine Zeile liest die Biases ihres
  Experten am Offset), mit und ohne Subgroups, und `CheckAttentionRanges('tiny-gpt-oss')` - die Sinks in
  `attention`, `flash` und `attndec`, ein Bereich und einer je Schlüssel, beide Cache-Formate,
  Zurückgehen im Kontext; `GptOss20bOnDevice` wie auf CUDA.
- Gegenprobe: 30 eingebaute Fehler - Sink nicht im Nenner (`attention`, `flash`, `attndec`),
  Sink von Kopf 0, `attnmerge` ohne Sink, SwiGLU ohne +1, ohne Grenze für Up, ohne alpha,
  MXFP4-Nibbles vertauscht (`linear8`, `gemm8`), E8M0 doppelt, `linear8 -DIDS` mit dem Bias
  des Merkmals statt des Experten, `biasOff` übergangen (`linear8`, `gemm8`, beide Zweige
  von `matvec8`), das Backend setzt `BiasOff` nicht oder bindet keinen Bias
  (`RecordLinearAt`, `RecordLinearIds`), kein MXFP4-Shader, kein O-Bias, kein Router-Bias
  (Host- und Gerät-Weg), Gate-Bias von Experte 0, kein Gate-Bias auf dem Gerät-Weg, keine
  Sinks, `attnmerge` ohne Sinks, die schlichte SwiGLU. Zwei überlebten zuerst: der Sink von
  Kopf 0 nur im Maximum (äquivalent - jedes M über allen Exponenten stabilisiert gleich;
  im Nenner gefangen) und der Shared-Memory-Zweig von `matvec8` (auf der RTX 5090 läuft
  der Subgroup-Build; jetzt ein Testfall ohne Subgroups). Alle übrigen scheitern direkt.
- Review: Codex fand, dass Vulkan die subnormale MXFP4-Skala der Exponenten 0 und 1 vor
  dem Produkt auf 0 setzen darf (jetzt `ldexp`, ein exakter Test mit Gewichten 2^-126 und
  Eingaben 2^126; die alte Fassung scheitert daran auf dem M2 Pro unter MoltenVK mit 0 statt
  32, NVIDIA und Lavapipe behalten subnormale Werte), dann dass `ldexp` selbst unter -126
  auf 0 setzen darf (jetzt `ldexp(k * 0.25, e - 126)`); der Claude-Finder nur einen
  veralteten Satz der Doku. Im letzten Nachreview beider ohne Befund bis auf die oben
  genannte Grenze.
- Grün auf dem RTX-Rechner unter FPC und Delphi 13 (je 6733 Assertions; gpt-oss-20b auf
  Vulkan und CUDA unter beiden mit denselben Antworten) und Delphi 7 (4277), unter WSL
  aarch64 mit Lavapipe (Infer 5055, LLM 741) und macOS M2 Pro mit MoltenVK - dort laufen
  `flash.comp`, `attndec.comp` und `matvec8.comp` von selbst (Infer 5101, LLM 730).

**D5d1 erledigt (2026-10-20): gpt-oss auf CUDA.** `TQwen3Cuda` nimmt gpt-oss an; die
Prüfung auf `TDecoderModel.Extras` ist dort weg (auf Vulkan seit D5d2).

- *Sinks:* `attention`/`attention16` bekommen die Sinks je Kopf als letzten Parameter
  (0: keine); mit einem Bereich kommt der Sink beim Abschluss in Maximum und Nenner
  (`M = max(m, sink)`, `y = acc * exp(m - M) / (l * exp(m - M) + exp(sink - M))`, wie die
  CPU: ein Sink weit über allen Scores läuft nicht über), mit mehreren einmal in
  `attnmerge` (das Maximum beginnt beim Sink).
- *Biases:* O über `EnqueueLinear`, der Router in beiden Wegen (Wahl auf dem Host und auf dem
  Gerät), die Experten beim Host-Weg über `EnqueueLinearAt` (die Zeile des Experten im
  Bias-Puffer), beim Gerät-Weg in `linear8ids` (`bias[ids[slot] * nout + o]`, neuer letzter
  Parameter, 0 für keinen). Alle Biases und Sinks gehen beim Anlegen hoch (ein Graph kann
  nicht hochladen).
- *SwiGLU:* `swiglu(g, u, n, alpha, limit)` - mit `limit > 0` die begrenzte Form von gpt-oss
  in den Schritten von `TInferCpuBackend.SwiGlu`, sonst wie bisher.
- *MXFP4:* Typ 6 in `linear`, `linear8`, `gemm8` und `linear8ids` - die Blöcke so gelesen,
  wie sie liegen (17 Byte je 32 Werte, `byteAt`/`bytes8` an jeder Adresse), Wert =
  `kvalues[q] * 2^(e - 128)` wie ggml (`E8M0_TO_FP32_HALF`, unter e = 2 subnormal), jedes
  Produkt exakt in float. Die Experten eines Layers liegen am Stück: Die Wahl auf dem Gerät
  greift (`linear8ids`), ein Decode-Schritt bleibt ein Graph.
- *Genauigkeit:* Das Gerät rechnet die Aktivierungen in float, die CPU vor MXFP4- und
  Q8_0-Gewichten in Q8_0. Gegen transformers (die Fälle von `TTestInferQwen3.GptOss20b`,
  safetensors-Checkpoint) liegt das Gerät darum näher: Logits mit Kosinus 0,9999999 für Text
  und 0,99989 / max. 0,37 für den Chatfall (die CPU 0,993 / 3,3), alle 16 Greedy-Tokens der
  vier Fälle gleich.
- *Tempo (RTX 5090, GGUF mit Q8_0-Matrizen, Prompt 200):* Prefill 861 Tokens/s (1156 bei
  2000), Decode 124 Tokens/s; mit Wahl auf dem Host 72; die CPU (22 Threads) 133 / 21.
- *Tests:* `OpsMatchCpu` mit der begrenzten SwiGLU (Grenze 7 und 0,8) und Sinks (ein Bereich,
  fünf, mit Fenster; ein Sink weit über allen Scores); `LinearMatchesReference` mit MXFP4
  (Zeilen ohne 4-Byte-Ausrichtung, alle Kernel) und einem exakten Fall für die
  Randexponenten 0, 1, 2, 127, 128, 200; `GptOssMatchesCpu` (aus `GptOssRefused`):
  tiny-gpt-oss in F32 (Wahl auf dem Host), aus dem Q8_0-GGUF und dem MXFP4-Checkpoint (Wahl
  auf dem Gerät, gegen die Wahl des Hosts, auch als Graph), mit F16-Cache und das schmale
  Modell, gegen die CPU; `GptOss20bOnDevice`: gpt-oss-20b gegen die transformers-Referenz,
  dann über `TNativeLlmClient` auf dem Gerät - 391, der Aufruf von `get_weather`, die
  Antwort mit den 17 Grad des Tools.
- `llm-bench` registriert gpt-oss (`mormot.ai.infer.model.gptoss` in den uses).
- Gegenprobe: 24 eingebaute Fehler - Sink übergangen (Attention, `attnmerge`), Sink von Kopf
  0, SwiGLU ohne Grenze für Up oder Gate, ohne +1, ohne alpha, MXFP4-Nibbles vertauscht
  (`linear8` und `linear`), E8M0 doppelt (normal und subnormal), 18 Byte je MXFP4-Block,
  `linear8ids` ohne Bias oder mit dem Bias von Experte 0, das Backend lässt ihn fallen, kein
  MXFP4-Kernel, kein O-Bias, kein Router-Bias (Host- und Gerät-Weg), alle Experten mit dem
  Bias von Experte 0, kein Gate-Bias auf dem Gerät-Weg, die schlichte SwiGLU (beide Wege),
  keine Sinks - alle scheitern (der subnormale Exponent nur am exakten Randfall).
- Review: Codex fand, dass der Sink nicht ins Maximum ging (ein Sink weit über allen
  Scores ließ `exp(sink - m)` überlaufen, die Ausgabe wurde 0) - jetzt wie die CPU, mit
  einem Testfall großer Sinks, an dem die alte Formel scheitert -, dass die begrenzte
  SwiGLU mit `fminf` ein NaN verlor (jetzt Vergleiche wie auf der CPU, Test `SwiGluNaN`),
  im 20b-Test eine Prüfung, die nicht scheitern konnte (der Client rechnet jetzt auf dem
  CUDA-Backend), das Freigeben bei einer Ausnahme und fehlende NaN-Prüfung der Logits; der
  Claude-Finder, dass die drei Experten-Biases nur gemeinsam geprüft wurden (jetzt jeder
  für sich). Eine Grenze bleibt dokumentiert: Liegt ein Sink mehr als etwa 103 über allen
  Scores, ist der Anteil der Schlüssel kleiner als das kleinste float - 0, wie die Gewichte
  des schnellen CPU-Backends. Im letzten Nachreview beider ohne Befund.
- Grün auf dem RTX-Rechner unter FPC und Delphi 13 (je 6431 Assertions, gpt-oss-20b auf
  CUDA unter beiden mit denselben Antworten) und Delphi 7 (4003, ohne CUDA: ein
  32-Bit-Prozess lädt den Treiber nicht), unter WSL aarch64 (Infer 4840, LLM 741) und
  macOS M2 Pro (Infer 4826, LLM 730).

**D5c3 erledigt (2026-10-20): das Reasoning früherer Züge geht ans Template zurück.**
`TLlmMessage.Reasoning` trägt die Überlegung eines Assistenten-Zugs, `TLlmChatResponse.Reasoning`
und `TLlmStreamDelta.ReasoningDelta` liefern sie (bisher nur als `reasoning_content` in `Raw`,
das bleibt). `TLlmAgent` übernimmt sie in den Zug mit den Tool-Aufrufen.

- *Templates:* gpt-oss schreibt sie als `thinking` des Templates - die Analyse vor den
  Aufrufen (außer ein späterer Zug antwortet ohne Aufruf) und im letzten Zug ohne
  Generation-Prompt vor der Antwort; sonst fällt sie weg. Hat ein Aufruf-Zug Reasoning
  *und* Inhalt - gpt-oss schreibt manchmal eine Analyse und eine Vorrede auf `commentary`,
  der Parser macht daraus Reasoning und Inhalt -, folgt der Analyse die Vorrede als
  `commentary`-Nachricht, wie das Modell sie schrieb; das Template lehnt dieses Paar ab,
  die Agentenschleife bräche sonst im nächsten Schritt ab (bewusster Unterschied). Qwen3 nimmt sie als `reasoning_content`: der
  `<think>`-Block der Züge nach der letzten Nutzerfrage, der Inhalt wird dann nicht mehr
  nach einem `<think>` durchsucht. Qwen2.5 kennt kein Reasoning und übergeht das Feld.
  Leer heißt keins (das Template unterschiede „vorhanden, aber leer“; `TLlmMessage` kann
  das nicht ausdrücken).
- *Wire:* `ParseOpenAIChatResponse` und der SSE-Stream lesen `reasoning_content` (vLLM,
  DeepSeek), ist es leer oder fehlt, `reasoning` (neuere vLLM, Ollama, OpenRouter), nur als
  String. `AddMessage` schickt es zurück, unter `TLlmProviderConfig.ReasoningField`: leer
  heißt `reasoning_content` - DeepSeek verlangt das im Thinking-Modus in jeder späteren
  Anfrage mit `tools` (sonst HTTP 400), vLLM gibt es an das Template -, `OllamaConfig`
  setzt `reasoning`, den einzigen Namen, den Ollamas OpenAI-Endpunkt liest
  (`LlmConfigFromEnv`: Umgebungsvariable `LLM_REASONING_FIELD`). Vorhanden ist
  es nur, wenn ein Anbieter es geliefert hat; wer einen solchen Verlauf an OpenAI selbst
  schickt, muss das Feld leeren (OpenAI lehnt unbekannte Felder ab). Der Anthropic-Wire
  liest und sendet es nicht (ein Thinking-Block bräuchte seine Signatur).
- *Fixtures:* `qwen3-chat` fünf Fälle mehr (26: Reasoning vor einem Aufruf mit und ohne
  Inhalt, vor der letzten Nutzerfrage, letzte Antwort, Kette; `qwen25-chat` dieselben, Qwen2.5
  übergeht sie), `gpt-oss-chat` sieben mehr (444, davon 106 abgelehnt: Aufruf mit
  Reasoning, spätere Antwort, letzte Antwort mit und ohne Generation-Prompt, Zwischenzug,
  Kette, letzter Aufruf); die alten Fälle unverändert, gleiche Pins.
- *Tests:* die Template-Tests über beide Fixtures, dazu Qwen3 mit Reasoning und `<think>`
  im Inhalt (nur im Text prüfbar: in der Referenz wäre `<think>` ein Zusatztoken) und
  Qwen2.5 mit Reasoning (dieselben Ids); `HarmonyTemplateRules` mit Reasoning und Vorrede;
  `ReasoningRoundTrip` (Wire), `StreamReasoning`
  (SSE), der Agent-Test gibt das Reasoning zurück, die Parser liefern `Response.Reasoning`
  und `ReasoningDelta` gleich dem `Raw`; `GptOss20bChat` gibt die Überlegung vor dem Aufruf
  zurück und prüft die Analyse im Prompt.
- Gegenprobe: 19 eingebaute Fehler (Agent vergisst das Reasoning, Wire liest keins, kein
  Rückfall auf `reasoning` bei fehlendem oder leerem `reasoning_content`, ein Objekt als
  Reasoning, Wire sendet keins, Feldname übergangen, Ollama mit `reasoning_content`, SSE
  und Native-Parser ohne `ReasoningDelta`/`Response.Reasoning`, Qwen3 übergeht es oder
  zerlegt den Inhalt trotzdem, Harmony nimmt den Inhalt vor dem Reasoning, verliert die
  Vorrede oder die Analyse, behält sie nach einer Antwort, keine Analyse vor der letzten
  Antwort, Analyse vor jeder Antwort) - alle scheitern.
- Review: Codex und der Claude-Finder fanden beide, dass ein Aufruf-Zug von gpt-oss mit
  Analyse *und* Vorrede (Reasoning und Inhalt) die Agentenschleife im nächsten Schritt
  am Template scheitern ließ - jetzt die Vorrede als `commentary`; Codex dazu, dass
  DeepSeek `reasoning_content` mit Tools zurückverlangt (der Entwurf schickte es nicht)
  und dass Ollama nur `reasoning` liest (`ReasoningField`, auch über
  `LLM_REASONING_FIELD`); der Finder einen leeren `reasoning_content` neben `reasoning`.
  Alles behoben, im dritten Nachreview beider ohne Befund.
- Grün auf dem RTX-Rechner unter FPC und Delphi 13 (je 6244 Assertions; gpt-oss-20b
  bekommt die Überlegung vor dem Aufruf zurück und antwortet weiter mit den 17 Grad des
  Tools) und Delphi 7 (4001), unter WSL aarch64 (Infer 4838, LLM 741), macOS M2 Pro (Infer
  4824, LLM 730); die LLM-Suite zuletzt unter Delphi 13 (740) und Delphi 7 (728).

**D5c2 erledigt (2026-10-20): gpt-oss als Chat-Modell (Harmony).** gpt-oss-20b läuft hinter
`ILlmClient`: `TNativeLlmClient` erkennt es an `model_type` bzw. an der GGUF-Architektur
(`RegisterNativeFamily(TGptOssModel, TBpeTokenizer, THarmonyChatTemplate,
THarmonyOutputParser)`).

- *Template* (`THarmonyChatTemplate`): das `chat_template.jinja` der Revision `6cee5e8…` in
  Pascal - Systemnachricht (Identität, Wissensstand, Datum, Reasoning-Effort, Kanäle, der
  Hinweis auf `functions`), Entwicklernachricht (die Anweisungen einer ersten
  Systemnachricht, die Tools als TypeScript-Namespace), dann die Züge: eine Antwort als
  `final`, ein Zug mit Tool-Aufrufen als `analysis` (sein Inhalt, außer ein späterer Zug
  antwortet ohne Aufruf) plus `assistant to=functions.NAME<|channel|>commentary json`, ein
  Tool-Ergebnis als `functions.NAME to=assistant` mit dem Inhalt als JSON-String, der letzte
  Zug ohne Generation-Prompt mit `<|return|>`. Die Tool-Schemas laufen durch die
  TypeScript-Umsetzung des Templates samt ihrer Eigenheiten, nachgebaut auf einem Baum der
  Python-Werte (`THarmonyJson`, aufgezeichnet vom vorhandenen `PythonJson`): `str()` für
  Enum- und Typlisten (`True`, `None`), `tojson` für Defaults; `" | \n"` zwischen
  oneOf-Varianten und 20 Leerzeichen vor ihrem Default, 16 vor dem Typ einer Eigenschaft
  eines verschachtelten Objekts (Ausdrücke ohne `-` im Tag, `lstrip_blocks` greift nur vor
  Block-Tags); der Merker für Objekt-Varianten wird in der Schleife gesetzt, wo Jinja ihn
  einschließt - er wirkt nie; ohne `items` findet `['items']` die Methode `dict.items`
  (wahr, aber ohne Typ: dasselbe `any[]`). Was Python als `repr` schriebe (ein Array in
  einem Enum) oder das Template nicht nehmen kann (eine Beschreibung, die kein String ist,
  `required`, `oneOf` oder `enum`, die kein Array sind), lehnt es ab. Optionen je Anfrage
  (`TChatTemplateOptions`, `EncodeWith`/`RenderWith`): `ReasoningEffort` (low, medium,
  high), `ModelIdentity`, `CurrentDate` (sonst heute, lokal wie `strftime_now`).
- *Bewusste Unterschiede:* Steuertokens als Ids, jeder Text bleibt Text (das Template lehnt
  einen Inhalt mit `<|channel|>…` ab - hier ist er Text); eine leere Tool-Beschreibung wird
  `// ` (transformers scheitert ohne); mehrere Aufrufe in einem Zug bekommen je eine
  Nachricht (das Template schreibt nur den ersten) und ein Ergebnis den Namen des Aufrufs
  mit seiner `ToolCallId` (das Template: den letzten Aufruf); eine zweite Systemnachricht
  wird abgelehnt (das Template lässt sie fallen); ein leerer Effort oder eine leere Identität
  ist die Vorgabe (transformers schriebe sie leer); `parameters`, die als wahr gelten, aber
  kein Objekt sind, werden abgelehnt (das Template schriebe `() => any`), falsche wie `null`
  ergeben wie dort keine Parameter.
- *Parser* (`THarmonyOutputParser`): die Nachrichten der Antwort - Kopf bis `<|message|>`
  (`<|channel|>` und Kanal, `to=` vor oder nach dem Kanal, Inhaltstyp mit `<|constrain|>`),
  Rumpf bis `<|end|>`, `<|call|>` oder `<|return|>` (die beiden halten an, ebenso
  `<|endoftext|>`). `analysis` wird Reasoning (`reasoning_content` wie bei vLLM), `final`
  und `commentary` ohne Empfänger der Inhalt, eine Nachricht an `functions.NAME` ein
  Tool-Aufruf, wenn der Rumpf ein JSON-Objekt ist (in der Form von `json.dumps`, wie das
  Template sie zurückschreibt), sonst Text im Inhalt; ein anderer Empfänger (ein
  eingebautes Tool von OpenAI, keins angeboten) geht ins Reasoning.
- *Client:* `NativeTemplateOptions` liest `reasoning_effort` als Feld der Anfrage (OpenAI) oder
  aus `chat_template_kwargs` (vLLM), dazu `model_identity`; zwei verschiedene Werte oder einer,
  der kein String ist, werden abgelehnt - für jede Familie, auch Qwen (wie bei vLLM); ein
  leerer Wert zählt als keiner; `TNativeLlmClient.ReasoningEffort` ist die Vorgabe.
  `NativeDecode` zieht nur Ids, die der Tokenizer kennt: gpt-oss hat 201 088 Zeilen in der
  Einbettung für 200 019 Tokens, Qwen3 151 936 für 151 669; die Füllzeilen von gpt-oss haben
  kleine, aber keine Null-Gewichte (Norm im Mittel 0,029 gegen 0,279), ein Zug dorthin brach
  bisher mit „outside the tokenizer“ ab.
- *Fixture* `gpt-oss-chat` aus `make_reference_d5c.py`: 22 Gespräche (Effort, Identität,
  System, Mehrzüger, letzte Antwort mit und ohne Generation-Prompt, Zeilenumbrüche,
  Unicode, Tools, Tool-Abläufe mit und ohne spätere Antwort, zwei Ergebnisse, Ketten) und 415
  Tool-Schemas (15 entworfene, 400 erzeugte über die Zweige der TypeScript-Umsetzung), von
  transformers mit festem Datum gerendert; 106 lehnt transformers ab (alle `"// " + Nicht-String`).
- *Tests:* `HarmonyTemplateText` (alle 437 Fälle textgleich, die abgelehnten abgelehnt - mit
  einem Mini-Tokenizer, ohne Modell), `HarmonyTemplateIds` (idgleich mit dem HF-Tokenizer),
  `HarmonyTemplateRules` (zwei Aufrufe in einem Zug, das heutige Datum, zehn Ablehnungen,
  Identität und Effort), `HarmonyParser` (Kanäle, Aufrufe, Empfänger vor und nach dem Kanal,
  Präambel, kein Objekt, eingebautes Tool, Steuertoken im Text, Abbrüche, Stream),
  `HarmonyRequestOptions`, `DecodeDrawsTokenizerIds` (Tiny-gpt-oss mit einem Tokenizer von 80
  Ids: nur diese, bis zur ersten Füllzeile dieselben Züge), `GptOss20bChat` (das echte Modell,
  greedy, Effort low: „17 × 23“ mit Reasoning und der Antwort 391, ein Aufruf
  `get_weather` mit `{"city": "Berlin"}`, nach dem Ergebnis eine Antwort mit dessen 17 Grad).
- Gegenprobe: 24 eingebaute Fehler im Template (oneOf-Trenner und -Einrückung, verschachtelte
  Objekte, die Grenze von 50 Zeichen, `str()` von true, -0.0 als wahr, `required`
  übergangen, Analyse trotz späterer Antwort, Tool-Ergebnis nicht als JSON-String, kein
  `<|return|>`, Aufrufe nach einer Antwort behalten, die Tools-Zeile, die Leerzeile nach den
  Anweisungen, leere Argumente, oneOf-Default als `str()`), im Parser (Analyse als Inhalt,
  andere Empfänger als Inhalt, `<|call|>` hält nicht an, Empfänger nur vor dem Kanal, ein
  abgebrochener oder ungültiger Aufruf verschwindet) und im Client (Effort der Anfrage
  übergangen, die ganze Einbettung gezogen, Prompt ohne Optionen) - alle scheitern. Drei
  überlebten zuerst (die Grenze von 50, -0.0, ein Empfänger auf `commentary`): Referenz und
  Test hatten keinen Fall dafür, jetzt zwei Schemas mehr und der Test auf `commentary`.
- Review: Codex und der Claude-Finder fanden beide, dass ein von `<|end|>` abgebrochener Kopf
  stehen blieb und seinen Empfänger an die nächste Nachricht weitergab (ein erfundener
  Tool-Aufruf möglich) - jetzt vergessen, mit Test; der Finder dazu `parameters` `null`
  (ein MCP-Tool ohne Schema ließ die Anfrage scheitern, jetzt keine Parameter) und leere
  Optionen (jetzt die Vorgabe).
- Grün auf dem RTX-Rechner unter FPC und Delphi 13 (je 6218 Assertions, gpt-oss-20b unter
  beiden mit denselben Antworten; Laden 34 s, je Anfrage 4-5 s) und Delphi 7 (3977), unter
  WSL aarch64 (4814) und macOS M2 Pro (4830).
- Das Reasoning früherer Züge kam hier noch nicht zurück - nachgezogen mit D5c3.

**D5c1 erledigt (2026-10-20): der Tokenizer von gpt-oss (o200k).** `TBpeTokenizer` liest das
`tokenizer.json` von `openai/gpt-oss-20b` (Revision `6cee5e8…`) und die Tokenizer-Metadaten
von GGUF-Dateien mit dem Pre-Tokenizer `gpt-4o`; Harmony, Parser und die Familie folgen in
D5c2.

- *Pipeline:* kein Normalizer (Qwen hat NFC), der Split-Ausdruck von o200k (wörtlich aus
  `tokenizer.json`, `O200K_SPLIT_REGEX`), ByteLevel, BPE mit `ignore_merges` (ein Stück, das
  selbst ein Token ist, wird nicht zerlegt). Der Konstruktor nimmt Normalizer (NFC oder
  keiner) und Ausdruck (Qwen2 oder o200k) in jeder Kombination; anderes lehnt er ab.
  `ignore_merges` ändert bei o200k nichts: kein Token des Vokabulars zerfällt durch BPE allein
  (der Generator hat jedes Token geprüft, das als Text ohne Ersatzzeichen U+FFFD dekodiert und
  ein einzelnes Stück ist), es zählt also nur für andere Vokabulare und ist
  mit einem Kunst-Tokenizer getestet.
- *Ausdruck:* von Hand wie der von Qwen2, als leftmost-first über die sieben Alternativen;
  `[^\r\n\p{L}\p{N}]?U*W+` mit dem Rückgabeschritt von `U*` (W+ ist dann das letzte Zeichen des
  Laufs, das in W liegt), Großbuchstaben-Läufe, Ziffern zu dritt, `[\r\n/]*` nach Satzzeichen,
  Kontraktionen nur als Anhang an ein Wort. Die Klassen dafür liegen in einer zweiten Tabelle
  (`mormot.ai.infer.unicode.o200k.inc`, `O200kClassOf`): Groß- und Titelbuchstaben,
  Kleinbuchstaben, Modifikator- und andere Buchstaben (in beiden Mengen), Marken (in beiden
  Mengen, aber kein `\p{L}`), Ziffern, Leerraum. Gemessen Code Point für Code Point mit der
  Regex-Engine von tokenizers 0.23.3, jede Menge einzeln, auch die beiden negierten Klassen;
  Buchstaben, Ziffern und Leerraum sind exakt die der Qwen2-Tabelle, die Faltung der
  Kontraktionen wieder ASCII plus U+017F.
- *GGUF:* `ggml-org/gpt-oss-20b-GGUF` hat dieselben 199 998 Tokens und 446 189 Merges, die 21
  Steuertokens als control und 1069 unbenutzte Füllzeilen bis zu den 201 088 Zeilen der
  Einbettung (kein Token). GGUF speichert weder Normalizer noch `ignore_merges`: `gpt-4o`
  bekommt `ignore_merges` wie das `tokenizer.json` von gpt-oss. llama.cpp selbst nimmt dort
  eine Näherung des Ausdrucks und kein `ignore_merges` - wir folgen Hugging Face.
- *Fixture* `gpt-oss-tokens` aus `make_reference_d5c.py`: 3095 Fälle (embed-de, die Fälle von
  `bge-m3-tokens` und `qwen3-tokens`, 79 o200k-eigene - Groß/klein, Titelbuchstaben, Marken,
  Ziffergruppen, Schrägstriche mit Zeilenumbruch, Harmony-Literale - und 2400 gemischte: je 600 aus `make_reference.py` und
  `make_reference_qwen3.py`, 1200 über die o200k-Klassen),
  je die Ids von `tokenizer.json` ohne Added Tokens (jeder Text ist Text) und die Stücke des
  Pre-Tokenizers als Byte-Offsets. Das `tokenizer.json` liegt nicht im Repo
  (`vendor/models/gpt-oss-20b/`, 27 MB).
- *Tests* (`test.infer.harmony`): `O200kClasses` (Stichproben, jede Klasse gegen die
  Qwen2-Tabelle für alle Code Points, die Zahlen des Generators), `O200kSplitMatchesReference`
  (alle Fälle stückgleich, ohne Modell), `O200kPipeline` (`ignore_merges`, kein NFC, die
  Kombinationen, drei Ablehnungen), `O200kIdsMatchReference` (alle Fälle idgleich, Dekodieren
  gibt den Text zurück), `O200kTokens` (Größen, die Harmony-Steuertokens, Literale bleiben
  Text), `O200kGguf` (die echte Datei: jedes Token gleich, alle Fälle idgleich).
- Grün auf dem RTX-Rechner unter FPC und Delphi 13 (je 6140 Assertions, mit der echten
  GGUF-Datei), unter WSL aarch64 (4745) und auf dem M2 Pro (4761).
- Gegenprobe: 20 eingebaute Fehler im Ausdruck (Rückgabeschritt von `U*`, die Klassen von U,
  W, Präfix und Satzzeichen, Kontraktion, Ziffern zu viert, kein `/`, kein optionales
  Leerzeichen), in der Pipeline (NFC immer, `ignore_merges` nie oder nicht gelesen, GGUF
  `gpt-4o` mit NFC, das Paar `vocab.json` + `merges.txt` ohne NFC) und in der ASCII-Tabelle -
  15 scheitern direkt, das Paar ohne NFC lokal mit dem Qwen3-Modell (in der
  Mutationsumgebung fehlt es). Vier überleben und sind gleichwertig: Präfix zuerst oder
  zuletzt (in Alternative 1 und in 2) und eine Marke als Präfix (eine Marke liegt in U und W, das Stück endet an derselben
  Stelle), `W*` nach `U+` (Alternative 2 kommt nur zum Zug, wenn kein W folgt).
- Review: Codex ohne Befund (verglich alle Referenztexte mit einer eigenen Nachbildung), im
  Nachreview zwei falsche Angaben dieser Doku (jetzt korrigiert); der
  Claude-Finder ohne Befund im Code (300 000 Zufallsketten gegen eine Backtracking-Engine),
  aber vier ungenaue Doku-Stellen (eine veraltete Lücke im Backlog, die Prüfung von
  `ignore_merges` zu weit beschrieben, „identisch mit Hugging Face“ statt mit der
  tokenizers-Bibliothek ohne Added Tokens, die Zusammensetzung der Fälle) - korrigiert.

**D5b2 erledigt (2026-10-20): gpt-oss aus GGUF.** `TGptOssModel` liest die Architektur
`gpt-oss` von llama.cpp; geprüft mit einem Tiny-GGUF und mit `ggml-org/gpt-oss-20b-GGUF`
(Revision `15dcfc7…`, `gpt-oss-20b-MXFP4.gguf`, aus derselben Checkpoint-Revision `6cee5e8`
konvertiert).

- *Metadaten* wie im veröffentlichten File: Größen, `expert_*`, `attention.sliding_window`
  (die geraden Layer gleiten, wie `set_swa_pattern(2)`), `rope.scaling.type` yarn mit
  `factor`, `original_context_length` und `yarn_beta_fast`/`_slow` (fehlen sie: 32 und 1;
  die veröffentlichte Datei hat sie, geschrieben von einem neueren Konverter als dem in
  UPSTREAM_BASE gepinnten llama.cpp-Stand, der die Betas nur aus den Kontextparametern nimmt).
  llama.cpp legt die SwiGLU-Grenze 7 und alpha 1,702 im Code fest (kein Schlüssel) - wir
  ebenso. `truncate` false wie der Checkpoint; llama.cpp selbst rundet die Grenzen des
  YaRN-Korrekturbereichs (floor/ceil, also wie `truncate` true) und weicht damit leicht von
  transformers ab. Abgelehnt: ein `rope.scaling.attn_factor` oder `yarn_log_multiplier`,
  ein anderer Skalierungstyp, ein fehlendes `output.weight`.
- *Tensoren:* 19 je Schicht - `attn_q/k/v/output` mit Bias, `attn_sinks.weight`,
  `attn_norm`, `post_attention_norm`, `ffn_gate_inp` mit Bias, `ffn_gate_exps`/`ffn_up_exps`/
  `ffn_down_exps` (Gate und Up schon getrennt, MXFP4 in place über `GgufExperts`) mit ihren
  Biases. Die MXFP4-Blöcke einer GGUF-Datei prüft `CheckScales` (ein Block außerhalb des
  exakten Bereichs muss nur Nullen halten).
- *Fixture* `tiny-gpt-oss/gguf` aus `make_reference_d5.py`: `tiny-gpt-oss/mxfp4` so, wie der
  Konverter schreibt, mit den Schlüsseln der echten Datei; das Umpacken der Blöcke liest
  ggmls eigenes `dequantize_row_mxfp4` als die Werte des Checkpoints zurück. `bf16.gguf`
  (Matrizen in BF16), `q8_0.gguf` (Matrizen und Embedding in Q8_0 wie die echte Datei),
  `betas.gguf` (Betas 16 und 2) und sieben kaputte Varianten.
- *Tests:* `GptOssGguf` - gegen den safetensors-Ordner mit `swiglu_limit` 7 (die Grenze, die
  llama.cpp setzt): dieselben Einstellungen, alle Experten Byte für Byte gleich, die Logits
  aller Positionen bitgleich; `q8_0.gguf` bis 0,042 (Kosinus 0,99985); die Betas der Datei;
  sieben Ablehnungen. `GptOss20b` prüft nun auch die echte GGUF-Datei: alle 768 Experten
  Byte für Byte gleich dem Umpacken aus safetensors, gegen transformers Logits bei Kosinus
  0,9997-0,9999 (Chat-Turn 0,990, max. 3,1; die Q8_0-Matrizen kommen zum Q8_0-Effekt der
  Aktivierungen dazu), Greedy 16 von 16 außer Fall 0 (weicht bei Schritt 3 ab, wo die
  Referenz nur 0,067 Vorsprung hatte). Laden 4,8 s (in place) statt 20 s.
- Gegenprobe: 11 eingebaute Fehler (Betas nicht gelesen, die ungeraden Layer gleiten,
  andere SwiGLU-Grenze, `truncate` true, `attn_factor` oder gebundener Kopf angenommen,
  Normen vertauscht, Gate und Up oder ihre Biases vertauscht, Tensorzahl, MXFP4-Skalen nicht
  geprüft) - alle scheitern. Der gebundene Kopf überlebte erst: der Test suchte im Fehlertext
  nach "tied", und das steht schon im Dateinamen `bad-tied.gguf`.
- Review: Codex ohne Befund (prüfte auch die Fixture-Hashes und die Tensoren gegen
  safetensors); der Claude-Finder fand, dass der gepinnte llama.cpp-Stand die Beta-Schlüssel
  nicht kennt (jetzt dokumentiert: die echte Datei hat sie), einen verrutschten Kommentar und
  eine zu weiche Erwartung im Fall ohne Index (jetzt mit dem Dateinamen).
- Grün auf dem RTX-Rechner unter FPC und Delphi 13 (je 6060 Assertions, die echte
  GGUF-Datei unter beiden mit denselben Werten) und Delphi 7 (3839) und auf dem M2 Pro
  (4690).

**D5b erledigt (2026-10-20): MXFP4 und gpt-oss-20b auf der CPU.** Das veröffentlichte
`openai/gpt-oss-20b` (Revision `6cee5e8…`, MXFP4-Experten, drei Teildateien) läuft auf der CPU
und stimmt mit transformers in float32 überein; GGUF für gpt-oss folgt (D5b2), dann
Tokenizer und Harmony (D5c) und die Geräte (D5d).

- *Typ* `idtMXFP4` (`TInferBlockMxfp4` wie ggml `block_mxfp4`: ein E8M0-Exponent, 16 Bytes
  E2M1-Nibbles, Wert j unten und j + 16 oben, 17 Bytes je 32 Werte); GGUF-Typ 39.
  `InferRowToF32` gibt 2^(e-127) mal den E2M1-Wert, exakt; ab e = 253 kann das über Single
  hinausgehen, dann wird die Unendlichkeit als Bits geschrieben (Delphi würde bei einem
  Float-Überlauf auslösen).
- *Rechnung:* ein MXFP4-Block wird zu einem Q8_0-Block - die verdoppelten E2M1-Werte
  (0, 1, 2, 3, 4, 6, 8, 12 und negativ, wie `kvalues_mxfp4` von ggml) als int8, 2^(e-128) als
  FP16-Skala. Das ist exakt für e 104..143 (FP16 hält 2^-24..2^15), und die vorhandenen
  Q8_0-Kernel rechnen damit, mit Q8_0-Aktivierungen wie ggml. gpt-oss-20b nutzt e 115..136
  (alle 597 Mio. Blöcke gezählt). Ein Block außerhalb des Bereichs muss nur Nullen halten
  (+0 oder -0), sonst lehnen die Lader die Datei ab. Der schnelle Pfad entpackt je Kachel
  (16 Ausgaben) wie bei Q4_0/Q5_0; dafür gibt es `InferMxfp4Unpack` (AVX2 `vpshufb`, NEON
  `tbl`, gleich der Pascal-Routine für jedes Byte und jeden Exponenten).
- *Teildateien:* `TSafeTensorsFile.OpenFolder` nimmt `model.safetensors`, sonst die Dateien
  von `model.safetensors.index.json` (die Reihenfolge von transformers);
  `CreateFromIndex` prüft jede Datei wie bisher und die `weight_map` gegen die Dateien in
  beide Richtungen: kein Tensor fehlt, keiner ist zu viel, keiner liegt in einer anderen
  Datei, kein Name doppelt, nur schlichte Dateinamen neben dem Index (kein Pfad, kein
  Laufwerk, nichts Verstecktes). Eine Datei, die die Map nicht nennt, gehört nicht zum
  Checkpoint und wird nicht geöffnet. Der Decoder öffnet seinen Ordner darüber.
- *gpt-oss:* `quantization_config` mit `quant_method` mxfp4 wird angenommen (anderes
  abgelehnt); die Experten kommen dann aus `gate_up_proj_blocks`/`_scales` und
  `down_proj_blocks`/`_scales` - Zeilen sind Ausgaben, Gate und Up abwechselnd, Byte j hält
  Wert 2j unten und 2j + 1 oben. Beim Laden in die Blockform von ggml umgepackt
  (`TDecoderModel.OwnMxfp4`, ein Feld je Matrixart für alle Experten einer Schicht) und
  geprüft; sie bleiben MXFP4, was `aWeights` auch verlangt (die übrigen Matrizen folgen
  `aWeights`).
- *Fixtures:* `mxfp4` aus ggml selbst (`quantize_row_mxfp4_ref`, `dequantize_row_mxfp4`,
  `ggml_vec_dot_mxfp4_q8_0` des lembed-Orakels); `tiny-gpt-oss/mxfp4` - die Konfiguration von
  `tiny-gpt-oss` mit MXFP4-Experten wie im echten Checkpoint, zwei Dateien mit Index, ein
  Nullblock mit e 0 und ein -0-Block mit e 255; transformers liest beim Laden genau die
  Werte, die der Generator schrieb. `gpt-oss-20b`: vier Prompts (Text, Deutsch, Code, ein
  Chat-Turn über das Template), je Ids, Logits der letzten Position, die letzte Zeile jedes
  Zustands und 16 Greedy-Tokens. Erzeugt auf dem RTX-Rechner (Windows, 96 GB) in einer
  Umgebung mit denselben Pins: die Experten bleiben BF16 (die MXFP4-Werte sind dort exakt)
  und werden je Expert erst beim Gebrauch zu float32 - so reichen rund 50 GB statt 84.
- *Ergebnis gpt-oss-20b* (RTX-Rechner, schnelles Backend, F32 um die MXFP4-Experten): Text,
  Deutsch und Code mit Logits bei Kosinus 0,99999 (max. 0,12), Greedy 16 von 16; der
  Chat-Turn (78 Tokens mit Harmony-Tokens) bei 0,9933 (max. 3,27), Greedy ebenfalls 16 von
  16. Das ist der Q8_0-Effekt, kein Fehler: transformers selbst mit Q8_0-quantisierten
  Experten-Eingaben liegt beim Chat-Turn bei 0,99258 (max. 3,32) und bei den anderen bei
  0,9999 (gemessen mit einem Hilfsskript, das in transformers die Experten-Eingaben wie
  `quantize_row_q8_0` quantisiert; nicht im Repo). Laden 20 s.
- *Tempo* (RTX-Rechner, i9, 64 Prompt-Tokens, 32 neue): F32 um die Experten Prefill 39,
  Decode 5,3 Tokens/s; Q8_0 um die Experten Prefill 42, Decode 7,6. Gemessen je Token beim
  Decode: die 288 MXFP4-Linears 160 ms mit dem Pascal-Entpacken, 75 ms mit dem
  SIMD-Entpacken (darum gibt es den Kernel); die übrigen Q8_0-Matrizen samt Kopf 41 ms.
- *Tests:* `Mxfp4LikeGgml` (Entpacken bitgleich zu ggml, auch außerhalb des exakten Bereichs;
  die Q8_0-Blöcke genau dort exakt; Linear gegen ggml), `LinearMxfp4MatchesReference`
  (Entpacken gegen die Definition für jedes Byte und jeden Exponenten, Kernel = Pascal,
  exakt = Referenz bitgleich, float in Toleranz), `SafeTensorsIndex` (27 Prüfungen),
  `GptOssMxfp4` (Expert 3 jeder Schicht wie transformers, die Nullblöcke angenommen, alle
  Zustände, schneller Pfad, Q8_0 um MXFP4, sechs Ablehnungen), `GptOss20b` (übersprungen
  ohne `vendor/models/gpt-oss-20b`).
- Review: Codex fand im Index-Lader die Größengrenze erst nach dem Lesen der ganzen Datei
  (jetzt vorher über die Dateigröße) und einen doppelten `weight_map`-Schlüssel oben
  (mORMot nähme den ersten, Python den letzten: jetzt abgelehnt); der Claude-Finder zwei
  Doku-Zahlen und dass der Test die Exponenten der Nullblöcke nicht prüfte (jetzt 0 und 255).
- Grün auf dem RTX-Rechner unter FPC und Delphi 13 (je 5813 Assertions, gpt-oss-20b unter
  Delphi 13 mit denselben Werten wie FPC) und Delphi 7 (3815; dort ohne das 20b-Modell, 32 Bit),
  auf dem M2 Pro (4666) und unter WSL aarch64 (4647, der NEON-Kernel).
- Gegenprobe: 28 eingebaute Fehler (Exponent 0/1, Nibble vertauscht, keine oder zu späte
  Unendlichkeit, ein falscher E2M1-Wert, der Bereich ab 105, das Entpacken nach Q8_0 auf
  beiden Pfaden und im AVX2-Kernel, jede Prüfung des Index, Nibble-Paare, Gate/Up-Zeilen,
  Down-Zeilen, die Skala des Nachbarblocks, der Exponentenbereich, -0 als Wert, jede
  Quantisierung, die Biases ...) - alle scheitern. Der Bereich ab 105 überlebte erst, weil
  der Test die Konstante selbst nutzte; er prüft jetzt mit festen Grenzen.

**D5a erledigt (2026-10-19): gpt-oss auf der CPU.** Die Rechnung von gpt-oss, gegen
transformers 5.19 mit einem Tiny-Modell; die veröffentlichten Gewichte (MXFP4), GGUF, der
o200k-Tokenizer, Harmony und die Geräte folgen (D5b bis D5e, [backlog.md](backlog.md)).

- `mormot.ai.infer.model.gptoss`: `TGptOssModel` (`model_type` `gpt_oss`) liest
  `config.json` - `rope_parameters` (transformers 5) oder `rope_scaling` mit `rope_theta`
  oben (transformers 4, so steht es bei `openai/gpt-oss-20b`), `layer_types` (sonst gleiten
  die geraden Layer, wie `GptOssConfig`), `sliding_window` (sonst 128), `swiglu_limit` und
  `swiglu_alpha` (sonst 7 und 1,702; die echte Konfiguration nennt kein alpha). Ohne
  RoPE-Angaben das YaRN von `GptOssConfig` (32 über 4096, Betas 32/1, `truncate` false);
  `null` wie in transformers: `factor` = `max_position_embeddings` / Original,
  `attention_factor` keiner, `head_dim` = Hidden / Köpfe, `num_key_value_heads` = Köpfe.
  Abgelehnt (in D5a): `quantization_config` (seit D5b: MXFP4), `attention_bias` false, gebundene
  Embeddings, ein anderer `rope_type`, ein RoPE-Schlüssel, den wir nicht lesen (`mscale`
  ...), ein `factor` von 0 oder darunter.
- *Experten im Layout von transformers 5:* `gate_up_proj` ist `[Experten, Hidden, 2 *
  Breite]` mit Gate und Up abwechselnd in den Spalten, `down_proj` `[Experten, Breite,
  Hidden]` - beide gegenüber `nn.Linear` transponiert. Beim Laden je Experte in eine
  Gate-, Up- und Down-Matrix umsortiert (`TDecoderModel.MatrixOf`: F32 übernommen, sonst
  zeilenweise nach Q8_0/Q4_0 quantisiert wie `Matrix`), die Biases ebenso getrennt.
- *Decoder:* `TDecoderLayer` hat `OBias`, `Sinks`, `RouterBias` und flache
  `ExpertGateBias`/`ExpertUpBias`/`ExpertDownBias`; `TDecoderConfig` `SwiGluLimit`/
  `SwiGluAlpha` und die YaRN-Felder (`RopeFactor`, `RopeOriginal`, `RopeBetaFast`/`Slow`,
  `RopeTruncate`, `RopeAttention`). `TDecoderModel.Extras` meldet, was ein Modell über
  Attention, RMSNorm und die einfache SwiGLU hinaus braucht; `TQwen3Vulkan` und
  `TQwen3Cuda` lehnen ein Modell mit Extras ab, bis sie es rechnen.
- *Backend:* `CausalAttention` nimmt `aSinks` (ein Logit je Kopf geht in jedes Softmax ein,
  ohne Wert: die Gewichte der Schlüssel summieren sich auf weniger als 1), Referenz und
  schneller Pfad; `SwiGlu` mit `aLimit > 0` die begrenzte Variante (Gate oben, Up beidseitig
  begrenzt, `(u + 1) * g * sigmoid(alpha * g)`, jeder Schritt in float32 gerundet). Der
  Router wählt wie bisher per Softmax über alle Logits mit Renormierung - das ist dasselbe
  wie das Softmax über die gewählten Logits von gpt-oss.
- *YaRN* in `CheckConfig` (`YarnFrequencies`): die Frequenzen unter dem Korrekturbereich
  bleiben (Extrapolation), die darüber werden durch den Faktor geteilt (Interpolation),
  eine Rampe dazwischen; die Bereichsgrenzen in double wie in Python, jeder Tensorschritt in
  float32. `Rotary` multipliziert cos und sin mit dem Aufmerksamkeitsfaktor (0,1 ln f + 1
  oder `attention_factor`). Da die Geräte ihre Faktoren über `Rotary` holen, rechnen sie
  YaRN schon mit.
- *Fixture* `tiny-gpt-oss` aus `tools/infer-ref/make_reference_d5.py`: 3 Layer (Fenster 5,
  voll, Fenster 5), 8 Experten, top-3, `head_dim` 24, YaRN mit `truncate` false (die Rampe
  liegt mitten in den 12 Frequenzen), Sinks, alle Biases, `swiglu_limit` 0,8 (der Clamp
  ändert etwa ein Drittel der Werte, vom Generator geprüft); dazu die RoPE-Faktoren der
  Positionen 0..63 und zweier Varianten (`truncate` true; `attention_factor` mit anderen
  Betas). Die Biases auf dem Residualpfad (`o_proj`, `down_proj`) sind klein gehalten,
  sonst gewinnt eine Richtung und Greedy wiederholt ein Token. `tiny-gpt-oss/narrow`: 2
  Layer mit Experten der Breite 16 bei Hidden 64 (eine Zeile von `down_proj` ist länger als
  eine von `gate_up_proj`).
- *Tests:* `GptOssMatchesReference` (YaRN-Faktoren, alle Zustände, Cache = voller Lauf
  bitgleich, Prefill in zwei Stücken, Greedy über 24 Tokens), `GptOssQuantized` (F32 auf
  dem schnellen Backend, Q8_0 exakt und float; Q8_0 gegen FP32 bis 0,49 auseinander,
  Kosinus 0,9968 - der Clamp bei 0,8 und der Top-3-Router sind in diesem Tiny-Modell
  scharf), `GptOssConfig` (transformers-4-Form bitgleich, die Vorgaben, beide YaRN-Varianten
  gegen ihre Referenz, 14 Ablehnungen, ein übriger Tensor), in `test.infer.fast` Sinks auf
  allen Pfaden gegen die Definition (ein Sink von -1e30 nimmt nichts, einer von 80 alles)
  und die begrenzte SwiGLU bitgleich auf allen Pfaden und in float32-Schritten,
  `GptOssRefused` auf Vulkan und CUDA.
- Review: Claude-Finder und Codex fanden beide den zu kleinen Zeilenpuffer in
  `LoadExperts` (eine Zeile von `down_proj` hat Hidden Werte, der Puffer fasste 2 * Breite -
  bei Hidden > 2 * Breite ein Heap-Überlauf; behoben, `narrow` prüft es) und `factor: null`,
  das YaRN still abschaltete; der Finder die beiden Bench-Demos, die `CausalAttention` und
  `SwiGlu` noch mit den alten Signaturen überschrieben; Codex `attention_factor: null` und
  im Nachreview `head_dim`/`num_key_value_heads: null`, die transformers ableitet. Das letzte
  Nachreview ohne Befund.
- Grün auf dem RTX-Rechner unter FPC und Delphi 13 (je 5440 Assertions) und Delphi 7 (3665)
  und auf dem M2 Pro (4515). Gegenprobe: 37 eingebaute Fehler (Sink nicht im Nenner auf
  beiden Pfaden, Sinks verworfen, jede der drei Begrenzungen und das +1 der SwiGLU, jeder
  Bias weg oder vom falschen Experten, YaRN ohne Faktor, ohne Rampenbegrenzung, ohne
  Aufrundung, ohne Aufmerksamkeitsfaktor, Gate und Up oder ihre Biases vertauscht, Down
  nicht transponiert, falsche Vorgaben, die Zählung der Tensoren, der kleine Puffer, die
  Ablehnung auf den Geräten ...) - alle scheitern. Ein Test für den Vorrang des Thetas in
  `rope_parameters` kam erst durch die Gegenprobe dazu.
- *Falle (macOS):* FPC 3.2.4 auf aarch64-darwin übersetzt `TDocVariantData.D[name] := wert`
  so, dass 0 gespeichert wird (Konstante wie Variable; `I[]` und `AddOrUpdateValue` mit einem
  Variant stimmen). Unser Quellcode nutzt den Setter nicht; die Tests schreiben Fließkommawerte
  über `SetFloat` (Variant).

**D4b erledigt (2026-10-19): Sliding Window und NoPE auf Vulkan und CUDA.** Bisher lehnten
beide Geräte solche Modelle ab (D4).

- *Fenster:* `attention.comp`, `flash.comp`, `attndec.comp` und `attention`/`attention16` in
  `tools/cuda/ops.cu` bekommen `window` (0: keins). Jede Abfrage hatte schon ihren
  Schlüsselbereich (`j0`/`j1`, über die Splits verteilt); er beginnt jetzt bei
  `total - window` statt bei 0, die Splits teilen nur noch das Fenster. Schlüssel davor
  werden nicht gelesen, nicht bloß maskiert. Die Bereiche wachsen weiter mit der Zeile, also
  stimmen `lo`/`hi` je Block in `flash.comp` und `attention.comp` unverändert.
- *NoPE:* `qkrope.comp` und `qkrope`/`qkrope16` bekommen `rope` (0: q und k ungedreht, wie die
  CPU sie lässt - nicht mit den Faktoren 1 und 0 gerechnet, so stimmen auch NaN und
  Unendlich).
- *Cache:* auf dem Gerät hält der Cache jeder Schicht alle Zeilen (wie
  `TDecoderContext.FullCache` auf der CPU); ein Gerätekontext kann so auf jede Länge zurück
  (`Restorable` gilt für alle). Der schmale Cache der CPU spart nur Speicher; er kommt aufs
  Gerät, wenn ein Modell zeigt, dass es sich lohnt (bei gpt-oss-20b, dessen halbe Layer ein
  Fenster von 128 haben, spart er bei langem Kontext knapp die Hälfte des KV-Speichers).
- *Eine Zeile je Block bei Fenster-Layern:* ein Block rechnet die Werte aller Schlüssel
  `lo`..`hi` seiner Abfragen, die außerhalb des Bereichs einer Abfrage mit Gewicht 0 - und
  0 mal einen NaN- oder unendlichen Wert ergibt NaN (Codex im Review). Ein Fenster-Layer
  mit mehreren Zeilen läuft deshalb mit `RowBlock` 1 (alle Abfragen eines Blocks teilen
  einen Bereich) über `attention.comp` bzw. `attention`, mit eigenen Splits; Layer ohne
  Fenster wie bisher. Eine Maske in der Wertschleife wäre die direkte Lösung gewesen, kostet
  aber unter Vulkan auf der RTX 5090 zu viel (Qwen3-0.6B Q8_0, Prefill 4000 Tokens 9750 ->
  5490 Tokens/s, Decode -18 %; als Auswahl auf das Gewicht noch -32 % und -12 %, je Kachel
  verzweigt ebenso), unter CUDA 4-10 %. Am Kausalrand eines Prompt-Chunks summieren die
  früheren Zeilen eines Blocks die Werte späterer Zeilen weiter mit Gewicht 0, wie seit
  Phase 4c - das träfe nur ein Modell, dessen Werte selbst schon NaN oder unendlich sind.
  Gemessen danach unverändert: Vulkan Prefill 4000 9694 gegen 9750, Decode 345 gegen 342;
  CUDA 10075 gegen 9950, 371 gegen 376 (Rauschen). Der Kopfblock eines Fenster-Layers ist
  dabei der von `attention.comp` (höchstens 8 Abfragen), auch wenn der Chunk `flash.comp`
  mit 32 rechnet - Fixture `tiny-qwen3-sliding/g16` (16 Köpfe auf einem KV-Kopf, aus
  `make_reference_d4b.py`) prüft das; mit dem Kopfblock von `flash.comp` scheitert es.
- `TQwen3Vulkan`/`TQwen3Cuda` nehmen Fenster und NoPE je Layer aus `Config.Spec`; die
  Ablehnung ist weg. Push-Records und Kernel-Parameter um `Window` bzw. `Rope` ergänzt,
  SPIR-V und PTX neu erzeugt.
- *Tests:* `SlidingWindowMatchesCpu` auf beiden Geräten mit `tiny-qwen3-sliding` (Fenster 5)
  und `tiny-smollm3` (Fenster 6 und NoPE-Layer): Prompt über zwei Chunks, das Fenster läuft
  dutzendfach um, beide Cache-Formate, ein Bereich und ein Bereich je Schlüssel, Vulkan
  dazu `flash.comp` und `attndec.comp`, CUDA die Decode-Schritte als Graph; danach zurück auf
  3 Tokens (hinter das Fenster) und 20 andere von dort, gegen eine Neuberechnung der CPU.
  Unter Vulkan läuft das über `CheckAttentionRanges`, jetzt für jede Familie von
  `NewDecoderModel` und mit diesem Rücksprung auch für die bisherigen Modelle.
  `OpsMatchCpu` (CUDA) prüft die Kernel einzeln: Fenster 3, 5, 64, 1 (nur sich selbst) und
  eines über alle Schlüssel, in Bereichen, mit und ohne Kopf-Norm, NoPE allein und mit
  Fenster; die Schlüssel und Werte vor dem Fenster sind NaN - gelesen, schlügen sie durch.
- Grün auf dem RTX-Rechner unter FPC und Delphi 13 (je 5182 Assertions) und Delphi 7 (3407),
  auf der Adreno X1-85 (Vulkan-Suite 860, die Fenster-Tests danach noch einmal mit `g16`)
  und dem M2 Pro (MoltenVK, 4257). Gegenprobe: 13 eingebaute Fehler - das Fenster übergangen
  in `attention.comp`, `flash.comp`, `attndec.comp` und `ops.cu`, um eins zu weit in
  `attention.comp`, die Drehung trotz NoPE in `qkrope.comp` (q und F16-Schlüssel) und
  `ops.cu`, kein Fenster bzw. immer RoPE aus Pascal (Vulkan und CUDA), die gemeinsamen
  Splits für Fenster-Layer, der Kopfblock von `flash.comp` für einen Fenster-Layer - alle
  scheitern.
- Review: der Claude-Finder fand einen veralteten Testnamen im D4-Abschnitt; Codex das
  Gewicht 0 mal NaN über den gemeinsamen Schlüsselbereich eines Blocks (oben, gelöst über
  eine Zeile je Block); im Nachreview fanden beide den zu großen Kopfblock (behoben, mit
  `g16` als Test). Das letzte Nachreview ohne Befund.

**B13 erledigt (2026-10-19): CPU-Rechnung unter Delphi 13 und FPC bitgleich.**
Anlass: Qwen3-30B-A3B gab auf der CPU unter Delphi 13 andere Logits als unter FPC (gegen die
GPU mit vorgegebener Expertenwahl Kosinus 0,9975 und Argmax 15 von 16 statt 0,9995 und 16).

- *Eingegrenzt* mit einem Prüfprogramm, das unter beiden Compilern auf dem RTX-Rechner
  (AVX2) dieselben Eingaben rechnet und jedes Ergebnis bitweise vergleicht: alle Operationen
  der beiden CPU-Backends (Linear F32 und Q8_0, RmsNorm, SwiGLU, RoPE, Attention, Softmax),
  das Tiny-Modell in F32 und Q8_0 und Qwen3-0.6B über alle 28 Layer waren schon bitgleich.
  Abweichend nur der MoE-Block, ab dem MLP-Ausgang von Layer 0.
- *Ursache:* die Gewichtung der Experten, `dst^[i] := dst^[i] + w * src^[i]` in `single`.
  Delphi Win64 rechnet ohne `{$EXCESSPRECISION OFF}` einen `single`-Ausdruck in `double` und
  rundet einmal, FPC rundet nach dem Produkt und nach der Summe (wie die float32-Referenz).
  Das RTL-`Exp` im Router war es nicht.
- *Behoben:* alle 24 Units unter `src/infer` setzen die Direktive (nur Delphi x64; Delphi 7
  rechnet auf dem x87 und bleibt außen vor), die Regel steht in `CONCEPT.md` §1. Danach
  bitgleich: alle Operationen, die Tiny-Modelle (dicht und MoE) und Qwen3-30B-A3B über alle
  48 Layer bis zu den Logits, Prefill und Decode-Schritt.
- *Tempo (Delphi 13, CPU, `llm-bench`, Mittel aus zwei Läufen):* Qwen3-0.6B Q8_0 Decode
  68,7 -> 71,1 Tokens/s, Prefill 658 -> 720; Qwen3-30B-A3B Q4_K_M Decode 11,9 -> 13,6,
  Prefill 62 -> 75 - die Umwandlungen nach `double` und zurück fallen weg (Anteil von `add`
  beim Prefill von 9 auf 2 %).
- *Tests:* die Delphi-Ausnahme in `Qwen3MoeGgufMatchesCpu` (CUDA und Vulkan) ist weg: Argmax
  16 von 16 statt mindestens 15. Delphi 13 kommt jetzt auf dieselben Zahlen wie FPC (Kosinus
  0,99948, 162 von 1344 Routen anders gewählt, größter Router-Abstand 0,052). Gegenprobe:
  ohne die Direktive in `decoder.pas` scheitert der Test unter Delphi 13 auf beiden GPUs
  (15 von 16, Kosinus 0,9975).
- Auch `mormot.ai.vectorstore.native` setzt die Direktive (`DotRow` summiert
  `single`-Produkte; die Tests dort halten 1e-6 ein und bleiben grün, 301 unter Delphi 13).
- Grün auf dem RTX-Rechner unter FPC und Delphi 13 (je 4996 Assertions) und Delphi 7 (3325).
  FPC übersetzt die Direktive nicht, für M2 Pro und Adreno ändert sich nichts. Review:
  Claude-Finder und Codex ohne Befund; der Finder nannte den Vektorspeicher, nachgezogen.

**B15 erledigt (2026-10-19): Q8_0 als Vorgabe, `MaxContext`, Quickstart-Demo.**
Anlass war neural-api (`LoadModel(dir)` und `ChatReply`, int8 als Vorgabe): der Einstieg soll
bei uns genauso kurz sein.

- *Vorgabe (Nutzerentscheidung):* `TNativeLlmClient.Create(Ordner)` quantisiert die Matrizen
  jetzt beim Laden zu Q8_0 - ein Viertel des Speichers von F32, rund doppeltes Decode-Tempo.
  Wer Token-Gleichheit mit transformers braucht (Tests, Vergleiche), übergibt `idtF32`; die
  Client-Tests tun das jetzt ausdrücklich, `Qwen3ChatClientQ8` prüft die Vorgabe. Der Befund
  von Phase 2 bleibt richtig: Q8_0 kann eine knappe Entscheidung kippen (die Wetterfrage).
  GGUF-Dateien betrifft das nicht, sie laden ihre Typen wie gespeichert.
- *Kontext:* Der KV-Cache wuchs schon immer nur mit dem Gespräch (Verdoppeln, bis
  `MaxPositions`, auf CPU und GPU). Neu ist `TNativeLlmClient.MaxContext` (0 = die Länge des
  Modells, 40960 bei Qwen3): ein längerer Prompt wird abgelehnt, eine längere Antwort endet
  mit `lfrLength`; damit bleibt der Cache unter etwa dem Doppelten des Deckels. Darunter
  nimmt `NativeDecode` ein `aMaxContext`. Tests: `DecodeOnTinyModel` (dieselben Regeln mit
  Deckel 100, ein Deckel über `MaxPositions` erweitert nichts), `Qwen3ChatClientQ8`
  (Deckel unter dem Prompt abgelehnt, Deckel = Prompt gibt genau ein Token).
- *Demo:* `demos/llm/llm-quickstart.lpr` - Ordner oder GGUF-Datei, eine Frage, die Antwort
  gestreamt, wahlweise `vulkan` oder `cuda`; dazu ein Schnipsel von zehn Zeilen im README.
  Gelaufen auf dem RTX-Rechner unter FPC, Delphi 13 und Delphi 7 auf CPU, Vulkan und CUDA,
  mit Ordner und GGUF, auch Qwen3-30B-A3B auf CUDA; der README-Schnipsel unter FPC in der
  WSL. Eingetragen in `build-demo.sh` und `build-delphi-demos.ps1`.
- Grün auf dem RTX-Rechner unter FPC und Delphi 13 (je 4996 Assertions) und Delphi 7 (3325)
  und auf dem M2 Pro (4175). Gegenprobe: Vorgabe zurück auf F32, Deckel übergangen, Deckel
  über `MaxPositions` hinaus, der Client reicht `MaxContext` nicht durch - alle vier
  scheitern. Review: Claude-Finder ohne Befund; Codex: ein späteres Senken von `MaxContext`
  verkleinert die Kontexte im Prompt-Cache nicht - jetzt am Property beschrieben, mit
  `Cache.Slots := 0` als Abhilfe.

**B14 Punkte 1, 2, 4 und 5 erledigt (2026-10-19): BF16/F16-Zeilen, Tokenizer aus `vocab.json` + `merges.txt`, F16-KV-Cache, Puffer je Worker.**
Die übrigen Rückmeldungen flydevs aus dem TTS-Projekt (Backlog B14).

- *Punkt 1:* `InferRowToF32` verbreitert BF16 und F16, Bit für Bit über `PCardinalArray` (ein
  NaN kommt nie in die FPU, wichtig unter Delphi 7). Der Decoder nimmt dafür jetzt
  `InferRowToF32` statt eigener Schleifen (`WidenTensor` über den ganzen Tensor als eine Zeile,
  `WidenRow` entfällt); dieselbe Rechnung, also nichts gekostet. Test `RowToF32Widens`: Null
  und minus Null, Normalzahlen, kleinste Subnormale, größter endlicher Wert, beide
  Unendlichkeiten, NaN mit Nutzlast, Zeile 1 von 2 und kein Wert über `aCols` hinaus.
- *Punkt 2:* `TBpeTokenizer.CreateFromVocabMerges(vocab.json, merges.txt)`, die Dateien von
  transformers' `Qwen2Tokenizer`. Das ältere Paar trägt die Pipeline nicht in sich; die
  Klasse kann nur die von Qwen2 (NFC, Split mit der Qwen2-Regex, ByteLevel). Statt GPT-2 als
  Vorgabe (so stand es im Vorhaben) prüft der Konstruktor deshalb, was der Ordner sagt: eine
  `tokenizer_config.json` daneben muss `Qwen2Tokenizer(Fast)` nennen und darf kein
  `add_prefix_space` setzen. Nachgebaut wie transformers 5.19 lädt (am Quellcode und mit
  Versuchen geprüft):
  - Zusatz-Tokens aus `added_tokens_decoder`, ohne ihn (ältere Ordner) aus
    `added_tokens.json`; transformers nummeriert sie fortlaufend hinter dem Vokabular neu,
    deshalb müssen die Ids genau so liegen, und kein Inhalt darf schon im Vokabular stehen.
  - Jedes benannte Spezial-Token muss ein Zusatz-Token sein: ein fehlendes hinge
    transformers mit neuer Id an, einen Eintrag des Vokabulars erkennt es im Text als Ganzes
    (`Encode` nicht). Geprüft werden jeder Schlüssel auf `_token`,
    `extra_special_tokens`, `additional_special_tokens`, `model_specific_special_tokens`, aus
    der Config und (ohne Decoder) aus `special_tokens_map.json`. Fehlt einer von
    `unk_token`, `eos_token` und `pad_token`, gilt für ihn die Vorgabe von `Qwen2Tokenizer`
    (`<|endoftext|>`).
    Bewusst ohne transformers' Vorrangregeln zwischen den Dateien und Listen (die
    Review-Runden fanden darin immer neue Sonderfälle): geprüft wird alles, was gelesen wird.
    Ein Ordner, dessen Dateien sich widersprechen, kann so abgelehnt werden, obwohl
    transformers ihn lädt - angenommen wird keiner, dessen Ids abweichen.
  - Abgelehnt, weil transformers dann etwas anderes liest oder Tokens anfügt, die `Encode`
    nicht anfügt: eine `tokenizer.json` daneben (dafür ist `Create` da), eine Datei mit
    `tokenizer.model`, `tiktoken.model` oder `tekken.json` im Namen; in der Config
    `fast_tokenizer_files`, ein eingebettetes `vocab` oder `merges`, `add_bos_token` oder
    `add_eos_token`; zwei Zusatz-Tokens mit demselben Inhalt (transformers zählt dann um).
  - `merges.txt` wie der Rust-Parser von tokenizers, den transformers 5.19 nimmt: jede
    Zeile, die mit `#version` beginnt, übersprungen; Leerzeilen, umgebende Leerzeichen und
    mehr als ein Leerzeichen abgelehnt; LF oder CRLF.
  Qwen3-TTS liefert genau dieses Paar ohne `tokenizer.json`: `vocab.json` ist bytegleich mit
  der von Qwen3-0.6B, `merges.txt` gleich bis auf die fehlende Kopfzeile, dazu 7
  Zusatz-Tokens mehr (`<tts_pad>` = 151671 usw.). transformers 5.19.0 auf dem TTS-Ordner gibt
  für alle 2416 Fälle von `qwen3-tokens` die Referenz-Ids (bis auf die bekannten Fälle, in
  denen es Steuer-Token-Literale im Text als Steuer-Tokens liest). Tests:
  `VocabMergesFiles` (ohne Modell, 53 Prüfungen: jede Regel oben angenommen und abgelehnt)
  und `VocabMergesMatchTokenizerJson` (das Paar von Qwen3-0.6B, ohne seine `tokenizer.json`
  kopiert, = sein `tokenizer.json`: jeder Token, jeder Zusatz-Token, die Referenz-Ids; das
  Paar von Qwen3-TTS nach SHA-256 an der Revision `85e237c`: 151676 Tokens, die
  Qwen3-Tokens unverändert, die Referenz-Ids).
- *Punkt 4 (Entscheidung 2026-10-19: Vorgabe bleibt F16, keine Umschaltung nach Länge):*
  Gemessen mit `llm-bench` (Qwen3-0.6B Q8_0, RTX 5090, 64 neue Tokens, `LAI_KV_F32=1` für
  F32):

  | Kontext | CUDA F16 / F32 | Vulkan F16 / F32 |
  |---|---|---|
  | 200 | 403 / 392 Tokens/s (+3 %) | 362 / 355 (+2 %) |
  | 4000 | 372 / 344 (+8 %) | 347 / 328 (+6 %) |
  | 16000 | 235 / 178 (+32 %) | 212 / 183 (+16 %) |

  Dazu der halbe Speicher. Der Fehler (Logits etwa 1e-3 statt 1e-4) schadet nicht, wo jeder
  Schritt in einer Token-Id endet; er wächst, wo floats Schritt für Schritt zurückgespeist
  werden (`ForwardEmbeds`-Schleifen eines Sprachmodells: Talker von Qwen3-TTS 7e-3 statt
  3e-5, bei flydev 2,3e-2). Eine Umschaltung nach Länge hätte einen Kontext je nach Verlauf
  verschieden rechnen lassen und spart bei kurzem Kontext ohnehin wenig. Stattdessen sagen
  `KvF16` (Vulkan, CUDA) und `ForwardEmbeds`, was F16 bringt und kostet und wann man es
  abschaltet.
- *Punkt 5:* `TInferThreadPool.RunWorkers` mit `TOnInferWorkerItem = procedure(aItem,
  aWorker)`: `aWorker` ist 0 für den aufrufenden Thread, 1..`Workers` für die des Pools,
  ein Index ist während eines Laufs genau ein Thread. Ein Job legt `Workers + 1` Puffer vor
  dem Lauf an und nimmt den seines Index. `Run` bleibt, wie er war (gemeinsames `Start`).
  Gemessen auf dem RTX-Rechner (32 Threads, 20000 Items mit je 16 KB Zwischenpuffer): Delphi 13
  67 ms mit Allokation je Item gegen 12 ms mit Puffer je Index; FPC 7 ms so oder so (sein
  Heap arbeitet je Thread) - flydevs Befund ist also eine Delphi-Sache, die Regel steht jetzt
  am Pool. Test `ThreadPoolWorkerIndex`: jedes Item einmal, Index in 0..`Workers`, derselbe
  Thread für einen Index über 3000 Läufe, kein Thread mit zwei Indizes, die Worker waren
  beteiligt, Ausnahme erreicht `RunWorkers`, `Run` danach, ohne Worker alles mit Index 0.
- Grün auf dem RTX-Rechner unter FPC und Delphi 13 (je 4985 Assertions) und Delphi 7 (3318),
  auf der Adreno X1-85 (Vulkan-Suite 805; die neuen CPU-Tests nativ auf Windows ARM) und dem
  M2 Pro (4164). Gegenprobe: 42 eingebaute Fehler über alle Runden (BF16/F16-Verschiebung und
  Zeilenversatz, halber Tensor im Decoder, jede Regel des Tokenizers einzeln abgeschaltet,
  Worker-Indizes vertauscht oder konstant, der Job mit Index nach `Run` behalten) - alle
  scheitern.
- Review: Claude-Finder und Codex fanden in fünf Runden vor allem, wie genau transformers 5.19
  einen Ordner ohne `tokenizer.json` lädt (Vorrang der Dateien, Vorgaben von
  `Qwen2Tokenizer`, Neunummerierung der Zusatz-Tokens, eine `tokenizer.json` daneben), dazu
  den vom Scheduler abhängigen Pool-Test (jetzt `WaitForPeer`). Den Abschluss machte auf
  Wunsch des Nutzers ein Fable-Reviewer ohne Codex, mit eigenen Versuchen in transformers:
  `merges.txt` liest in 5.19 der Rust-Parser von tokenizers (jede `#version`-Zeile
  übersprungen, Leerzeilen und Ränder abgelehnt), und ein Spezial-Token aus dem Vokabular
  erkennt transformers im Text als Ganzes - beides übernommen, das Nachreview ohne Befund.

**B14 Punkte 3 und 6 erledigt (2026-10-19): Embedding-Zeilen, Decoder-Stapel, `ForgetWeights`.**
Anlass ist flydevs TTS-Projekt (Backlog B14): Er musste den Durchlauf von `TQwen3Vulkan` nachbauen,
um fertige Embeddings statt Token-Ids einzuspeisen, und stieß auf veraltete Gewichte im
GPU-Cache.

- `TDecoderModel.ForwardEmbeds`: Zeilen `[n, Hidden]` statt Ids, auf CPU, Vulkan und CUDA
  (`TDecoderDevice.ForwardEmbeds`, `Chunk` nimmt Ids oder Zeilen). Zeilen aus `Embed` geben
  die Logits der Ids bitgleich; Ids und Zeilen lassen sich über Chunks mischen (Text mit
  eingeschobenen Bild- oder Audiozeilen). Eine Position mit Zeile hat im Cache die Id -1:
  `Common` trifft sie nie, `Reuse` hält dort an.
- `TDecoderModel.CreateStack`: ein Decoder als ein Stapel eines größeren Checkpoints - die
  Einstellungen aus einem Objekt von `config.json` (Pfad mit Punkten, z. B.
  `talker_config.code_predictor_config`), die Tensoren unter einem Präfix statt `model.`,
  eigenes Embedding (oder keins: dann nur `ForwardEmbeds`) und eigener Kopf; geprüft wird nur,
  dass jeder Tensor unter `<Präfix>layers.` verstanden ist. Für einen Stapel nimmt
  `TQwen3Model` eine multimodale RoPE (`mrope_section`, Summe = head_dim / 2, Typ default)
  als gewöhnliche RoPE: dieselben Winkel, solange die drei Positionskomponenten gleich sind
  (eine Folge der Reihe nach, wie beim Talker von Qwen3-TTS; Codex hat es am Quellcode von
  Qwen3-TTS bestätigt).
- `TInferBackend.ForgetWeights(aFrom, aTo)`: Vulkan und CUDA verwerfen jede Gerätekopie, deren
  Host-Speicher den Bereich berührt, und erhöhen `WeightEpoch` (wer Gewichte hochgeladen hat,
  schlägt neu nach). Ein `TDecoderModel` vergisst seine Gewichte beim Freigeben auf seinem
  Backend, ein `TQwen3Vulkan`/`TQwen3Cuda` auf dem seinen, `TXlmrModel` ebenso. *Neue Regel:*
  ein Modell vor seinem Backend freigeben (die Bibliothek tut es überall so).
- Beim Test trat die Falle wirklich auf: Nach dem Freigeben des Talkers lagen die verbreiterten
  BF16-Gewichte des Predictors an denselben Adressen mit denselben Größen - die GPU rechnete
  mit den Gewichten des Talkers (Logits-Kosinus 0,04), bis das Gerät beim Freigeben vergaß.
- *Tests:* `EmbeddingRows` (bitgleich, gemischt, Ids im Cache, `Reuse`, `Common` mit
  negativer Id), `StackFromCheckpoint` (Fixture `tiny-qwen3/stack` aus
  `make_reference_b14.py`: dieselben Bits wie das Modell, Pfad mit Punkten, ohne Embedding;
  abgelehnt: ein unbekannter Layer-Tensor, MRoPE mit fehlender Frequenz oder einem
  Abschnitt, der nicht positiv ist, skalierte RoPE,
  fehlendes Objekt, fehlender Kopf, kein Präfix), `Qwen3TtsStacks` gegen flydevs Referenz
  `qwen3-tts-lm` (PyTorch float32; Modell nach Hash aus dem Manifest): jeder Zustand des
  Talkers bis 1,8e-4, seine Logits bis 1,3e-5, die 15 Logit-Zeilen des Code-Predictors bis
  8,2e-5, alle Greedy-Codes gleich; auf CUDA, Vulkan (RTX 5090) und der Adreno
  `Qwen3TtsOnDevice` (Talker-Logits bis 2,9e-5 mit KV-Cache in floats), `EmbeddingRowsOnDevice`
  (bitgleich zu den Ids, gemischt, 300 Zeilen über zwei Chunks) und `ForgetWeightsServesNew`
  (alte Kopie ohne den Aufruf, neue danach, andere Gewichte bleiben, Gerät und Modell vergessen
  beim Freigeben).
- Grün auf dem RTX-Rechner unter FPC und Delphi 13 (4882 Assertions) und Delphi 7 (3215), auf
  der Adreno X1-85 (Vulkan-Suite 805) und dem M2 Pro (MoltenVK, 4071, ohne das TTS-Modell).
  Gegenprobe: 13 eingebaute Fehler (Ids 0 statt -1 auf CPU und CUDA, `Common` trifft negative
  Ids, Präfix falsch abgeschnitten, Layer-Tensoren nicht gezählt, MRoPE-Summe nicht geprüft,
  model_type auch beim Stapel geprüft, jeder Chunk ab Zeile 0 auf Vulkan und CUDA, beim
  Zusammenschieben Puffer nicht verschoben, nur im Bereich beginnende Gewichte vergessen,
  Gerät und Modell vergessen beim Freigeben nichts) - alle scheitern. Review: der
  Claude-Finder nannte die neue Freigabe-Reihenfolge, negative Ids in `Common` und
  ungeprüfte MRoPE-Abschnitte (im Nachreview: dazu ein Test); Codex den fehlenden 32-Bit-Skip des TTS-Tests (3,6 GB), die
  ungetestete Chunk-Grenze und das Zusammenschieben im Cache. Behoben.

**D3c erledigt (2026-10-19): Expertenwahl auf dem Gerät beim Decode.** Ein Decode-Schritt
(eine Token-Zeile) wählt die Experten auf CUDA und Vulkan selbst; der Host wartet nicht mehr
je MoE-Layer. Prefill wählt weiter auf dem Host (dort zählt der Durchsatz der Experten-Blöcke,
nicht die Wartepunkte).

- Ein Route-Kernel bzw. -Shader rechnet `TDecoderModel.Route` auf den Router-Logits nach: die
  k höchsten Logits, bei Gleichstand die niedrigere Id (dieselben Experten wie der Host bei
  denselben Logits), Softmax und Normierung auf CUDA in double wie der Host, auf Vulkan in
  float (Vulkan muss keine doubles können); Ids und Gewichte nach Id sortiert in einen
  Gerätepuffer, so summiert das Gerät in Id-Reihenfolge wie die CPU.
- `linear8ids` (CUDA, `tools/cuda/linear.cu`) bzw. `linear8.comp` mit `-DIDS` (Vulkan): die
  Rechnung von `linear8` für eine Zeile, Arbeitsgruppe y ist ein Platz der k gewählten
  Experten, die Matrix des Experten bei Id mal Abstand im Puffer aller Experten eines Layers;
  Gate und Up lesen dieselbe Zeile, Down die Zeile seines Platzes.
  `TInferCudaBackend.EnqueueLinearIds`, `TInferVulkanBackend.RecordLinearIds`.
- `moecombine` summiert die k Ausgaben mit ihren Gewichten in Platz-Reihenfolge - dieselbe
  Rechnung wie `fill0` und ein `scatteradd` je Experte.
- Ein Decode-Schritt ist damit auf CUDA wieder ein CUDA-Graph (die Wahl liegt im Graphen) und
  auf Vulkan ein Batch ohne Zwischen-Submit.
- Nur wenn jeder MoE-Layer seine Experten in einem Puffer hat (heute bei GGUF-Dateien), den
  das Gerät ganz bindet (Vulkan: höchstens `maxStorageBufferRange` bzw. `MaxLinearBytes`), in
  einem Typ mit `linear8ids`-Form (Blocktypen; F32 bei einer durch 8 teilbaren Breite),
  höchstens 64 aus höchstens 4096 Experten gewählt werden und kein `RouteTape` gesetzt ist (`DeviceRoutable`,
  `RouteOnDevice`); sonst wie bisher auf dem Host. `CheckRoute` (Test und Diagnose) liest
  nach jedem Route die Logits und die Wahl zurück und vergleicht sie mit `TDecoderModel.Route`.

*Gemessen:* Qwen3-30B-A3B Q4_K_M auf der RTX 5090, 200 Tokens Prompt, 64 neue, derselbe Build
mit `LAI_ROUTE_HOST=1` (Wahl auf dem Host) und ohne:

| | Decode, Wahl auf dem Host | Decode, Wahl auf dem Gerät | Prefill |
|---|---|---|---|
| CUDA | 38,9 Tokens/s | 97,3 Tokens/s | 383 (unverändert) |
| Vulkan | 42,9 Tokens/s | 90,4 Tokens/s | 490 (unverändert) |

*Tests:* `RouteMatchesHost` auf CUDA und Vulkan (Kernel bzw. Shader gegen `RouteRef`, die
ausgeschriebene Wahl des Hosts, die auf CUDA zuerst gegen `TDecoderModel.Route` geprüft wird;
8 aus 128 mit und ohne Normierung, 2 aus 8, 4 aus 64; Zufallswerte, Gleichstände an der
k-Grenze, lauter gleiche Logits, große und sehr negative Werte; ein NaN-Gewicht zählt als
Fehler). `TinyMoeMatchesCpu`: das GGUF (Experten in einem Puffer) wählt auf dem Gerät, die
anderen Fälle auf dem Host; Decode-Schritte mit `CheckRoute` (jede Wahl gleich der des Hosts)
und ohne, auf CUDA als Graph, jeweils gegen die Wahl des Hosts auf demselben Gerät (CUDA
relativ 1e-6, Vulkan 1e-5 - dort kommen die Gewichte aus float). Gegen die CPU prüfen die
ersten Schritte nur per Kosinus (0,999), die späteren gar nicht: mit Q8_0 quantisiert die CPU
auch die Aktivierungen, an einer knappen Router-Entscheidung kippt dann ein Experte - mit der
Wahl des Hosts auf dem Gerät genauso.
`Qwen3MoeGgufMatchesCpu`: nach dem Band acht Decode-Schritte mit eigener Wahl und `CheckRoute`:
384 Zeilen, keine andere Wahl als der Host, Gewichte auf CUDA bitgleich, auf Vulkan bis
9e-8. Grün auf dem RTX-Rechner unter FPC und Delphi 13 (4325 Assertions) und Delphi 7 (2869),
auf der Adreno X1-85 und dem M2 Pro (MoltenVK, 3723 Assertions, das Tiny-MoE wählt dort auf dem
Gerät). Gegenprobe: vierzehn eingebaute Fehler (Route auf CUDA und Vulkan: Gleichstand an die höhere
Id, nie normiert, auf CUDA nicht nach Id sortiert; `linear8ids`: Down liest Zeile 0, Experte
Id + 1, auf Vulkan immer Experte 0; `moecombine` ohne Gewichte; Gerätewahl trotz `RouteTape`;
Vulkan nie routbar) - alle scheitern. Review: Codex fand zwei Vulkan-Pipelines, die der
Destruktor nicht freigab, und dass ein NaN-Gewicht den Maximumvergleich passiert hätte; der
Claude-Finder, dass `GpuLinearIds` auf Vulkan `MaxLinearBytes` übersah. Behoben.

**D4 erledigt (2026-10-19): Sliding Window, Cache-Semantik, NoPE.** Auf der CPU; die
GPU-Geräte lehnten solche Modelle zunächst mit klarer Meldung ab (seit D4b rechnen sie sie).

- `TDecoderLayerSpec` hat `Window` (0: kein Fenster) und `NoRope`. Ein Token sieht die
  letzten `Window` Tokens samt sich selbst - Schlüssel j für Abfrage i, wenn
  i - Window < j <= i, wie transformers maskiert. `TInferBackend.CausalAttention` hat dafür
  `aWindow`; Schlüssel vor dem Fenster werden nicht gelesen (der Test füllt sie mit NaN).
  `NoRope` lässt die Drehung von Q und K in dem Layer weg.
- Der Cache eines Fenster-Layers hält nur, was das nächste Token braucht: Zeile 0 ist die
  Position `CachedBase`, vor neuen Zeilen schiebt `Slide` die noch gebrauchten
  (aPast - Window + 1 .. aPast - 1) an den Anfang. Der Puffer hat das Doppelte des Fensters
  (eine Verschiebung je Window Tokens), wächst für einen großen Prefill-Chunk und schrumpft
  beim nächsten Schritt von höchstens einem Fenster wieder auf das Doppelte. Ein Chunk
  überschreibt so nie einen Schlüssel, den eine seiner Zeilen noch braucht.
  `TDecoderContext.FullCache` behält alle Zeilen und maskiert nur - die Referenz.
- Rücksetzpunkte: `Restorable(t)` gilt für 0, für die volle Länge und für jede Länge, deren
  Fenster noch im Puffer liegt; `Truncate` lehnt andere ab, `Usable` gibt den längsten
  gemeinsamen Anfang, zu dem der Cache zurück kann (sonst 0), `Reuse` und der Prompt-Cache
  des Clients wählen danach - ein Gespräch, das hinter das Fenster zurückspringt, wird neu
  gerechnet statt mit fehlenden Schlüsseln.
- Konfiguration wie transformers 5.19: Qwen3/Qwen2 mit `use_sliding_window`
  (`sliding_window`, sonst 4096; `null` heißt kein Fenster) auf `layer_types` oder den
  Layern ab `max_window_layers` (sonst 28); Qwen3-MoE auf allen Layern (Qwen3MoeConfig
  kennt keine Layer-Auswahl). Ein Layer-Typ `sliding_attention` ohne Fenster wird
  abgelehnt, wie transformers ihn nicht maskieren kann.
- `TSmolLM3Model` (`smollm3`, nachgeschlagen in `modeling_smollm3.py` und
  `configuration_smollm3.py`): ohne Bias und Kopf-Norm, `head_dim` aus Hidden / Köpfe,
  NoPE nach `no_rope_layers` (sonst jeder `no_rope_layer_interval`-te Layer), Fenster nach
  `layer_types` (auch ohne `use_sliding_window`, wie das Modell maskiert) oder auf den
  NoPE-Layern. Nur aus `config.json`: llama.cpp legt die NoPE-Layer für smollm3 fest, statt
  sie zu lesen - GGUF wird abgelehnt, bis es gegen die Referenz geprüft ist. Keine
  Chat-Familie (Template und Parser von SmolLM3 fehlen noch).

*Referenz:* `tools/infer-ref/make_reference_d4.py` (ohne Modell-Download): `tiny-smollm3`
(4 Layer: voll mit RoPE, Fenster ohne RoPE, Fenster mit RoPE, voll ohne RoPE; Fenster 6) und
`tiny-qwen3-sliding` (Fenster 5 ab `max_window_layers` 1, ohne `layer_types` in der
Datei). Je drei Sequenzen (5, 13, 29 Tokens) mit allen Zuständen und 24 Greedy-Tokens ohne
EOS - das Fenster läuft mehrfach um; der Erzeuger prüft Greedy mit und ohne Cache von
transformers gleich und dass die Logits ohne Fenster abweichen (um bis zu 1,54 bzw. 0,93).

*Tests:* `DecoderOpsMatchReference` (je Attention-Fall ein anderes Fenster, zusammen 1 bis 12, Referenz
und Float-Pfad gegen die Definition, Schlüssel außerhalb des Fensters NaN; ein Fenster über
alle Schlüssel gibt dieselben Bits wie keins), `SlidingWindowMatchesReference` (alle
Zustände, Greedy durch den schmalen Cache), `SlidingWindowCache` - das Gate, auf der
Referenz und dem schnellen Backend: 53 Positionen in Chunks von 1, 2, 5, 6, 7, 13 und
gemischt geben die Bits des vollen Caches; einzeln gefüttert hält ein Fenster-Layer 12
Zeilen (das Doppelte), ein voller alle; nach jeder Länge 0 bis 40 eine Verzweigung von 8
Tokens gegen die Neuberechnung bitgleich (Truncate, wo `Restorable` es erlaubt - geprüft
gegen die Regel -, sonst `Reuse` ab 0); nach einem Prefill von 40 schrumpft der Puffer beim
nächsten Schritt auf 12 Zeilen bei gleichen Bits; `FullCache` kann jede Länge, ist auf
einem gefüllten Kontext gesperrt. `SlidingWindowConfig` (Ableitungen und Ablehnungen wie
oben, jede mit ihrer Meldung), auf Vulkan und CUDA `SlidingWindowRefused` (Fenster allein,
NoPE allein; seit D4b ersetzt durch `SlidingWindowMatchesCpu`). Grün auf dem RTX-Rechner unter FPC und Delphi 13 (4281 Assertions) und
Delphi 7 (2848), lokal auf der Adreno. Gegenprobe: sechzehn eingebaute Fehler (Float-Pfad liest
ab Schlüssel 0, Referenzpfad ohne Fenster, Fenster um eins zu kurz, `Slide` verwirft einen
gebrauchten Schlüssel, `Slide` verschiebt nichts, `Restorable` um eins zu großzügig,
`Truncate` ohne Prüfung, `Reuse` nach `Common`, NoPE übergangen, Fenster nicht an die
Attention, schmaler Cache nie benutzt, Qwen3-Fenster ab `max_window_layers` + 1,
`no_rope_layers` verkehrt, Intervall um eins verschoben, Vulkan nimmt ein Fenster, Vulkan nimmt NoPE).
Review: der Claude-Finder fand Abweichungen von transformers (fehlende Schlüssel mit
Standardwerten, `null`, SmolLM3 ohne `use_sliding_window`, Qwen3-MoE auf allen Layern),
Ablehnungstests ohne Meldung und einen Puffer, der nach einem großen Prefill nicht
schrumpfte; Codex eine Vergrößerung, die bei Speichermangel Schlüssel und Werte
verschieden groß ließ, eine Kontextwahl im Prompt-Cache, die das letzte Token mitzählte,
und Geräte-Tests, die Fenster und NoPE nur zusammen prüften. Alles behoben, die Fixes
nachgeprüft.

### Danach — nach Bedarf

Reranker (BGE-Reranker), Sparse-/ColBERT-Ausgabe von BGE-M3, Qwen3-Embedding; weitere
Decoder siehe Phase D.

Eine eigene Spur daneben ist die Verteilung großer Jobs auf ungenutzte Rechner:
[distributed-execution.md](distributed-execution.md). Ihr Executor ist diese Engine.

## Plattformen und Tests

### Messung am 2026-10-14

| Plattform | Compiler | Ergebnis |
|---|---|---|
| WSL Linux aarch64 | FPC 3.2.4 (Baseline) + 3.2.2 | grün (MCP 1257, LLM 656, Inferenz 1215, alle Demos) |
| WSL2 Linux x86_64 auf dem RTX-Rechner (Ubuntu 24.04) | FPC 3.2.2 (wie Docker) | grün aus einem sauberen `git archive`: MCP 1257, LLM 656, Inferenz 828 mit Korpus-Gate |
| macOS M2, aarch64-darwin | FPC 3.2.4-rc2 per SSH | grün: MCP 1257, LLM 647 (der vec0-Test wird mangels Extension übersprungen); am 2026-10-15 Inferenz 1597 mit allen Modellen (BGE-M3, Qwen3-0.6B), mit Korpus-Gate 1604, nDCG@10 0.9004 wie auf dem Snapdragon. Dabei gefunden: mORMot füllt `CpuFeatures` auf aarch64 nur unter Linux, unter macOS liefen deshalb die Pascal- statt der SDOT-Kernel; `HasArmDotProd` fragt dort `hw.optional.arm.FEAT_DotProd` per sysctl ab. Am 2026-10-16 Inferenz 2629 einschließlich der Vulkan-Suite auf der M2-Pro-GPU über MoltenVK (Homebrew `molten-vk` 1.4.2, `vulkan-loader` 1.4.363.0, `LAI_REQUIRE_VULKAN=1`). |
| Windows ARM, aarch64-win64 | FPC 3.3.1 (fpcupdeluxe, `D:\bin\fpc`) | grün: MCP 1257 und LLM, **aber** nur mit einer ARM64-`sqlite3-64.dll` (sqlite.org 3.51.2) neben dem Binary. mORMot2 hat für dieses Ziel kein statisches SQLite. Ein Skript gibt es noch nicht. |
| Windows x64 | Delphi 13 | grün (MCP 1257, LLM 659, Inferenz 1205) |
| Windows x64 unter Prism | FPC 3.2.2 | nur Kompatibilitätsprobe, nicht für Leistungsaussagen |
| Windows 11 x64, i9-14900K, 96 GB, RTX 5090 (32 GB, Treiber 616.64), per SSH (`ai-rtx5090`) | FPC 3.2.2 win64 (lokal übersetzt, nativ ausgeführt) | grün, Stand `bfd2507` + README-Umbau: MCP 1257, LLM 656 (mit vec0), Inferenz 821 (ohne konvertiertes FP32-Modell, dessen Tests überspringen); das Delphi-13-Testbinary dort ebenfalls grün (Inferenz 818 vor den letzten Tests). Delphi 13 ist installiert, CUDA-Toolkit (`nvcc`) nicht. |
| Windows x86 | **Delphi 7** | grün nach dem Umbau (MCP 1253, LLM 647, Inferenz 669; vec0 übersprungen, es gibt kein 32-Bit-Bundle), siehe unten |

### Rollen im Testplan

Compiler-Verträglichkeit, numerische Korrektheit und Leistung werden **getrennt**
geprüft. Grüne Ergebnisse aus übersprungenen Tests darf es nicht geben:
- Pflichttests ohne Modell sind von modellabhängigen Läufen getrennt.
- Übersprungenes wird ausdrücklich gemeldet.
- Ungetestete mORMot-Ziele gelten als „ungeprüft".

| Plattform | Rolle |
|---|---|
| WSL aarch64, FPC 3.2.4 + 3.2.2 | Hauptkorrektheit; Differenztests Skalar ↔ NEON |
| macOS M2 | Mapping, Dateisystem, NEON-Korrektheit; native Benchmarks |
| Windows x64 (RTX-Rechner) | AVX2-Korrektheit und native x64-Leistung (auch POPCNT für den Store); CUDA-Differenztests und -Leistung |
| Linux x64 | Deploy-Ziel (Docker); auf dem RTX-Rechner über WSL2 (Ubuntu 24.04, FPC 3.2.2, Konto `rtx5090`), seit 2026-10-14 eingerichtet |
| Delphi 13 Win64 | Compiler/ABI und Laufzeitkorrektheit |
| Delphi 7 Win32 | Build plus kleine Tests zu RTTI, Unicode, Numerik und Speichergrenzen; **kein** Leistungs-Gate |
| Windows ARM, FPC 3.3.1 | experimentell, bis Skript und SQLite-DLL geregelt sind |
| Windows x64 unter Prism | Kompatibilitätsprobe |

## Delphi 7

> **Erledigt (2026-10-14):** alle 22 Units bauen, beide Suiten laufen unter Delphi 7 grün
> (`scripts/run-delphi7-tests.ps1`), ebenso weiter unter FPC 3.2.4/3.2.2, Delphi 13, macOS
> und Windows ARM. Umgesetzt wie unten geplant; dazu ein Fund: der stdio-Transport hätte
> unter Delphi 7 eingehendes UTF-8 doppelt kodiert (`StringToUtf8` auf einem AnsiString) —
> behoben in `McpReadLine`, abgesichert durch `StdioReadsUtf8Unchanged`, das erstmals den
> echten Lesepfad prüft. Vorher dazugekommen: `RegistriesKeepTheirContract` und
> `TypedToolCarriesManagedFields` als Netz für den Umbau, danach
> `GenericShellMatchesRecordTool`. Fallen für die Engine: siehe „Praxis" unten.

**Befund vor dem Umbau (2026-10-14, dcc32 gegen mORMot2 master):**
- mORMot2-Kern, `mormot.db.raw.sqlite3.static` und `mormot.net.client` bauen.
- Von den 22 landrixai-Units bauen **11** unverändert: `llm`, `llm.types`, `llm.openai`,
  `llm.anthropic`, `llm.sse`, `embeddings`, `embed.provider`, `vectorstore`, `rag`,
  `sqlite`, `mcp.tools`.

Die Hindernisse:

| Ursache | Stellen | Folge |
|---|---|---|
| `TMcpToolBase<T: record>` (generische Klasse, `mcp.pas:1068`) | 1, dazu die Spezialisierung `rag.tool.pas:75`; im Konsumenten rund 200 Nutzungen | `mcp`, `mcp.stdio`, `mcp.legacy`, `mcp.server`, `rag.tool`, `llm.structured`, `agent.mcp` |
| generische Registries `IKeyValue<>`/`TPair<>` aus `mormot.core.collections` (`mcp.pas:1153 ff.`, `:3386 ff.`) | 12 | MCP-Kern |
| `exit(wert)` | ~37 (`mcp.server` 21, `mcp` 12, `agent`, `agent.mcp`, `vectorstore.native`) | u. a. `agent` |
| Record mit Methoden (`TTopK`, `vectorstore.native.pas:267`) | 1 | `vectorstore.native` |
| `for … in` | 4 (`mcp.pas`) | MCP-Kern |
| UTF-8-BOM | `mcp.server.pas` | `mcp.server` |
| fehlendes `uses mormot.core.unicode` (`Utf8ToString`) | 2 | `embed.lembed`, `vectorstore.sqlitevec` |

Hinter den ersten Fehlern können weitere liegen.

**Vorgehen (mit Codex abgestimmt):**
1. **Mechanisch:**
   - `exit(x)` ersetzen;
   - BOM entfernen und `uses` ergänzen;
   - `TTopK` nach mORMot-Muster umstellen
     (`{$ifdef USERECORDWITHMETHODS}record{$else}object{$endif}`);
   - `for … in` ersetzen.
2. **Registries:** für **alle** Compiler auf eine nicht-generische mORMot-Struktur
   (`TSynDictionary`) umstellen. Ein Codepfad ist besser als `{$ifdef}`-Zweige.
3. **Typisierte Tools:** eine nicht-generische Basis, die einen RTTI-Deskriptor nimmt.
   - `TypeInfo` allein liefert unter Delphi 7 **keine** Record-Feld-Metadaten. Die
     Basis nimmt deshalb registrierte `TRttiCustom`-Beschreibungen
     (`Rtti.RegisterFromText`, mit dessen Layout-Regeln).
   - Initialisierung und Freigabe verwalteter Felder (Strings, dynamische Arrays) der
     Argument-Records müssen dabei erhalten bleiben.
   - `TMcpToolBase<T>` bleibt unter `HASGENERICS` als dünne Hülle darüber, sodass der
     Konsument unverändert bleibt.
4. **Tests:**
   - eigenes Skript `run-delphi7-tests.ps1` (Win32, dcc32, explizite Pfade; die
     installierte `dcc32.cfg` zeigt auf einen falschen Pfad);
   - Programme, die Generics brauchen, werden ausgeklammert;
   - Pflicht sind Laufzeittests zu Schema-Erzeugung, Serialisierung, Unicode, Store und
     Numerik, nicht nur „baut".
5. Das Backend des Konsumenten bleibt FPC-only und ist **nicht** Teil dieser Portierung.

Hauptrisiko: das Ganze als reine Syntaxarbeit zu behandeln, während Schema-Erzeugung,
Serialisierung oder die Lebensdauer verwalteter Felder zur Laufzeit brechen. Aufwand:
einige Tage.

**Praxis (beim Umbau gelernt, gilt auch für die Engine):**
- Keine Array-Konstruktoren (`TFoo.Create(1, 2)`), kein `Default()`, kein `as` vom
  Interface auf die Klasse (`ObjectFromInterface`), kein `TThread.Start` (`TSynThread`),
  keine Konstante `-2147483648` (`Low(integer)`), kein `UnicodeString` ohne
  `HASVARUSTRING`, keine globale `FormatSettings` (vor Delphi XE).
- Kein UTF-8-BOM und keine Nicht-ASCII-Zeichen in String-Literalen: Delphi liest eine
  Datei ohne BOM als ANSI.
- Nach dem ersten Fehler meldet dcc32 oft Folgefehler in fremden Units (auch in mORMot) —
  immer den ersten Fehler beheben.
- Delphi 7 kürzt den Programmnamen am ersten Punkt (`mcp.tests` → `mcp.exe`) und braucht
  FastMM4 als eigene Bibliothek.

**Engine unter Delphi 7 / Win32:**
- Reine Pascal-Referenz, keine SIMD-Pflicht. Assembler wäre möglich, ist aber kein Ziel.
- Große Gewichte als Q8_0/F16; Normparameter und Ähnliches bleiben F32. Akkumuliert wird
  in FP32.
- Kontext, Batch, Scratch und KV-Cache werden begrenzt. Das Modell wird nie komplett auf
  FP32 entpackt.
- Adressraum: 2 GB, mit LAA auf 64-Bit-Windows 4 GB. Der Mapping-Helfer von mORMot lehnt
  auf CPU32 Ansichten über `MaxInt` ab, deshalb mappt `mormot.ai.infer.memory` in Fenstern.
- Gate: nur Korrektheit, keine Leistung.

## Vorlagen und Lizenzen

| Bereich | Projekt | Lizenz | Nutzung |
|---|---|---|---|
| Decoder-Gerüst | llama2.c (Karpathy) | MIT | portieren |
| Modelldetails | candle (Hugging Face) | MIT / Apache 2.0 | nachschlagen; Code nur aus dem MIT-Zweig |
| Spezifikation | transformers (Hugging Face) | Apache 2.0 | **nur** nachschlagen |
| Tokenizer | tokenizers (Hugging Face) | Apache 2.0 | **nur** nachschlagen (Clean-Room) |
| Quantisierung, GGUF, CPU-Kernel | llama.cpp / ggml | MIT | ausgewählt portieren |
| CUDA-Kernel | calm (zeux), llama.cpp | MIT | ausgewählt portieren |

**Nicht verwenden:** PasLLM (AGPL 3.0) und neural-api (LGPL 2.1) als Vorlage.

**Pflichten:**
- Bei Portierung den Copyright-Hinweis der Vorlage mitführen und die Herkunft je Datei
  festhalten.
- Apache-2.0-Code wird **nicht** übernommen: Er ist mit GPLv2, einem Zweig der
  Dreifachlizenz, unverträglich und lässt sich nicht einfach umetikettieren.
- Vor jeder Übernahme die aktuelle LICENSE-Datei der Vorlage prüfen. (Keine
  Rechtsberatung.)

**Modelle:** BGE-M3 (MIT), Qwen3 (Apache 2.0, nur als Gewichte genutzt, nicht als Code).

## Richtwerte

| | |
|---|---|
| BGE-M3 | 24 Layer, Hidden 1024, 16 Köpfe, FFN 4096, ca. 568M Parameter |
| BGE-M3 Speicher | FP32 ca. 2,2 GB, Q8_0 ca. 600 MB |
| Generierung | durch die Speicherbandbreite begrenzt: Tokens/s ≈ Bandbreite / Modellgröße |
| lembed heute (Snapdragon X, WSL) | ~25 Texte/s kurz, ~3,6–5,7 Texte/s bei ⌀ 51 Tokens |

## Offene Entscheidungen

1. ~~Lizenz~~ — **entschieden 2026-10-14**: mORMot-Dreifachlizenz für das ganze Repo,
   auch für die Engine (`LICENSE`).
2. ~~Zeitpunkt der Delphi-7-Portierung~~ — **erledigt 2026-10-14**, vor dem ersten
   Engine-Code: die Engine entsteht von Anfang an unter `run-delphi7-tests.ps1`.
3. ~~Maximale Kontextlänge für Embeddings~~ — **entschieden 2026-10-17**: 512 Tokens wie im
   Vertrag v1 (Begründung: `docs/backlog.md` B7).
