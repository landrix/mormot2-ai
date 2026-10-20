# Backlog — native Inferenz und Umfeld

Offene Themen, jedes so beschrieben, dass ein Agent ohne Vorwissen aus dieser Session
loslegen kann. Reihenfolge = grobe Priorität. Stand 2026-10-16 (landrixai `9a58adb`).

**Für jeden Eintrag gilt** (Regeln aus `CONCEPT.md` §1 und `docs/native-inference.md`):

- Stil wie mORMot2: nicht-generischer Kern, kein `exit(wert)`, kein `with`, keine
  anonymen Methoden/Inline-Variablen, ASCII-Literale, Parameter `a…`, Felder `f…`.
  Kommentare im Code Englisch, Doku Deutsch.
- Compiler: FPC 3.2.2 und 3.2.4 (Referenz: WSL, `scripts/run-fpc-infer-tests.sh`), Delphi 13
  (`scripts/run-delphi-tests.ps1 -Suite infer`) und Delphi 7
  (`scripts/run-delphi7-tests.ps1 -Suite infer -FastMM4 'D:\Projekte\src\utils'`).
  FPC zuerst; die x64-Delphi-Suite läuft auf dem ARM-Laptop nur unter Emulation und
  braucht dort Stunden — nativ auf dem RTX-Rechner (`ssh ai-rtx5090`, Exe + `tests/data`
  + Modelle nach `D:\Temp\ai-infer`, Daten werden vom Exe-Pfad aufwärts gesucht).
- Jeder Kernel bekommt Gegenproben (absichtlich eingebauter Fehler → Test muss
  fehlschlagen) und ein Review (Claude-Finder + Codex).
- Vorlagen nur mit verträglicher Lizenz (Tabelle „Vorlagen und Lizenzen" in
  `native-inference.md`); PasLLM (AGPL) und neural-api (LGPL) nicht als Vorlage.

---

## B1 — K-Quant-Prefill: fusionierter Mehrzeilen-Kernel

**Ziel:** Q4_K_M-Prefill näher an Q8_0 bringen. Heute (Snapdragon X, 200 Tokens):
Q4_K_M 290 Tokens/s, Q8_0 545 Tokens/s; i9-14900K: 455 gegen 757.

**Ausgangslage:** `LinearTileWork` in `src/infer/mormot.ai.infer.backend.cpu.pas` rechnet
K-Quants je Ausgabe und Tokenzeile einzeln über `DotK` mit dem fusionierten Kernel
`InferQ4KSums` / `InferQ6KSums` (`src/infer/mormot.ai.infer.kernels.pas`, NEON als
`.long`-Opcodes, AVX2 als Inline-Asm). Q8_0 dagegen nutzt `InferQ8Linear4x2`
(4 Gewichtszeilen × 2 Tokenzeilen, jede geladene Zeile zweimal verwendet).

**Verworfen (2026-10-16, gemessen):** Q4_K-Kachel einmal in int8-Blöcke zu 32 auspacken,
dann `InferQ8BlockDots4` und Skalen/Minima in Pascal. Bitgleich, aber langsamer (205 statt
270 Tokens/s im A/B-Lauf): Das Auspacken der Nibbles kostet im fusionierten Kernel fast
nichts, die Pascal-Skalenrechnung pro Zeile dafür viel.

**Vorgehen:**
1. Neuer Kernel-Typ, z. B. `TInferKSums2` (eine Gewichtszeile gegen zwei Q8_K-Tokenzeilen,
   Ausgabe je Block 2 × (isum, msum)), Pascal-Fassung als Maßstab.
2. NEON: aus `Q4KSumsNeonDot` ableiten — Gewichts-Nibbles und Skalen einmal laden, SDOT
   gegen beide Zeilen. Opcodes mit `aarch64-linux-gnu-as` erzeugen (siehe vorhandene
   Kommentare je `.long`), AAPCS64: v8..v15 nicht anfassen.
3. AVX2: aus `Q4KSumsAvx2Asm` ableiten, nur ymm0..ymm5 (Win64-ABI), Scratch im X-Bereich
   von `TInferAvx2Args`; FPC 3.2.2 kennt `vmovq`/`vpmovzxbw` mit 64-Bit-Speicheroperand
   nicht → über GPR laden.
4. `LinearTileWork`: Tokenzeilen paarweise, Rest einzeln; Skalierung exakt wie `DotK`
   (Reihenfolge der Double-Operationen!), damit `Exact` bitgleich bleibt.
5. Danach dasselbe für Q6_K (Gruppen zu 16, `scales` als int8).

**Erledigt (2026-10-16): Schritte 1, 2 und 4 für Q4_K.**
- *Kernel:* `TInferKSums2`, `InferQ4KSums2Pascal`, `Q4KSums2NeonDot` (Opcodes aus
  `q4k2.s` mit GNU as 2.42, Quelltext als Kommentar je `.long`).
- *Rechnung:* `DotQ4K2` und die Paarung in `LinearTileWork`.
- *Test:* `LinearKMatchesReference` prüft Kernel = Pascal, auch über 513 Blöcke in zwei
  Zeilen. Die Gegenprobe (vertauschte Ausgaben) lässt 16 Assertions scheitern.
- *Messung:* Q4_K_M-Prefill bei 200 Tokens auf dem Snapdragon 275 → 299 Tokens/s, auf dem
  M2 Pro 271 → 294.
- *Schritt 5 (2026-10-17):* `InferQ6KSums2Pascal`, `Q6KSums2NeonDot` (aus `q6k2.s`),
  `DotQ4K2` zu `DotK2` für beide K-Typen verallgemeinert. Gegenprobe (vertauschte
  Ausgaben) lässt 16 Assertions scheitern. Q4_K_M-Prefill bei 200 Tokens: Snapdragon
  291 → 311 Tokens/s, M2 Pro 291 → 305.
- *Schritt 3 (2026-10-17):* `Q4KSums2Avx2Asm`, `Q6KSums2Avx2Asm` (erzeugt, ymm0..ymm7,
  xmm6/7 unter Win64 im X-Bereich gesichert, Skalen-Zwischenwerte in `A[5..6]`). Grün
  unter FPC 3.2.2 und Delphi 13 auf dem i9-14900K; Gegenprobe (Zeilen vertauscht) lässt
  32 Assertions scheitern. Q4_K_M-Prefill bei 200 Tokens: 462 → 579 Tokens/s.
- *SysV-Einsprung (Linux x86_64):* gelaufen 2026-10-17 in der WSL des RTX-Rechners
  (FPC 3.2.2, Inferenz-Suite grün, siehe B5).

**Fertig, wenn:** `TTestInferFast.LinearKMatchesReference` (bitgleich, auch 513 Blöcke)
mit einer Form „gerade/ungerade Zeilenzahl" grün; neuer Test Kernel = Pascal; Gegenproben
schlagen an; `demos/infer/llm-bench vendor/models/qwen3-0.6b-q4_k_m.gguf 200 32` im
A/B-Lauf (alte und neue Exe abwechselnd, das Notebook schwankt um bis zu 25 %) schneller.

---

## B2 — Q4_0-Kernel und ARM-Repacking

**Ziel:** Q4_0 beim Dekodieren mindestens so schnell wie Q8_0 (heute halb so schnell:
31 gegen 63 Tokens/s, Snapdragon).

**Ausgangslage:** Q4_0 ist korrekt (byte-gleich ggml, `InferQuantizeRowQ4_0`) aber
experimentell: `LinearTileWork` entpackt jede Kachel per `InferQ4ToQ8` nach Q8_0 und
nutzt dann die Q8-Kernel. Qualität auf Qwen3-0.6B schwach (cos 0,76–0,96), sinnvoll eher
für 4B/8B-Modelle.

**Vorgehen:** fusionierter Q4_0×Q8_0-Kernel wie ggml (`ggml_vec_dot_q4_0_q8_0`, MIT):
NEON — Nibbles maskieren, −8, SDOT; AVX2 — `maddubs` mit Vorzeichentrick. Optional
ARM-Repacking (block_q4_0 4×4/8×4 verschränkt, GEMV über 4/8 Zeilen).

**Fertig, wenn:** `TTestInferFast.LinearQ4MatchesReference` grün (exakt bitgleich),
Gegenproben, Decode-Tempo im A/B-Lauf ≥ Q8_0.

**Erledigt auf ARM (2026-10-17), ohne fusionierten Kernel:** Der Engpass war das
byteweise Entpacken nach Q8_0.
- *Pascal:* `InferQ4ToQ8` entpackt acht Nibbles je Int64 bitweise (`Q4Minus8`) auf x86,
  x64 und aarch64; andere CPUs bleiben byteweise (unausgerichtete Int64 bei Offset 2).
- *NEON:* `Q4UnpackNeon` hinter `InferQ4Unpack` (jedes aarch64).
- *Test:* `LinearQ4MatchesReference` prüft `InferQ4ToQ8` über alle 256 Bytewerte gegen
  die Definition (die Referenz teilt die Funktion, fast = ref allein fände dort nichts)
  und den Kernel = Pascal. Gegenproben: ohne `or (t shl 4)` scheitert der 256-Byte-Test,
  ein `nop` statt des zweiten NEON-`sub` lässt 13 Assertions scheitern.
- *Messung (Decode, Qwen3-0.6B Q4_0, 32 Tokens):* Snapdragon X 30 → 70–78 Tokens/s
  (Q8_0 58–66), M2 Pro 28,5 → 84.
- *x64:* Schon das Pascal-Entpacken erfüllt das Ziel (i9-14900K: 64 → 79 Tokens/s, Q8_0
  71–73); ein AVX2-`InferQ4Unpack` ist zurückgestellt (höchstens etwa 1 ms je Token).
  Fusionierter Kernel und Repacking nur, wenn größere Modelle es verlangen.

---

## B3 — CUDA-Backend (zweites GPU-Backend nach Vulkan) — erledigt 2026-10-17

**Ziel:** NVIDIA-Karten über die Driver API (`nvcuda.dll` / `libcuda.so`, zur Laufzeit
geladen, **keine** mitgelieferte DLL), Kernel als eingebettetes PTX.

**Voraussetzung:** Vulkan-Backend (Phase 4) steht; dessen Backend-Schnittstelle und
Differenztests werden wiederverwendet. Bau-Rechner braucht `nvcc` (pip-Paket, siehe Stand).

**Vorlagen:** calm (zeux, MIT), llama.cpp CUDA (MIT).

**Fertig, wenn:** dieselben Referenztests wie die CPU grün auf dem RTX-Rechner (RTX 5090).

**Stand:** C1 (Fundament) erledigt 2026-10-17, siehe `native-inference.md` „CUDA“. `nvcc`
kommt aus NVIDIAs pip-Paket (13.4.92) und läuft auf dem Laptop; auf dem RTX-Rechner musste
nichts installiert werden. C2 (`Linear` F32/Q8_0/Q4_K/Q6_K) erledigt 2026-10-17, auch
unter Linux (WSL des RTX-Rechners). C3 (die übrigen Operationen als Kernel) erledigt
2026-10-17. C4 (`TQwen3Cuda`: der ganze Forward-Pass auf der Karte, auch hinter
`TNativeLlmClient`) erledigt 2026-10-17. C5 (Tempo) läuft: Messung gegen Vulkan und die
Attention ohne Scores-Puffer (C5a), die Matrixprodukte `linear8`/`gemm8` (C5b) und der
Decode-Schritt als CUDA-Graph (C5c) erledigt 2026-10-17: Decode auf der RTX 5090 jetzt
schneller als Vulkan (Q8_0 404 gegen 358 Token/s), Prefill bei rund 90 % von Vulkan.

---

## B4 — Weitere Modelle: Phase D (Plan 2026-10-17)

Ziel: weitere Decoder-Familien so anbinden, dass jeder neue Baustein mehrere Modelle
freischaltet. Geplant mit Codex-Abgleich; Modelldaten aus den Modellkarten und `config.json`
der Hersteller (Stand Oktober 2026), vor jedem Slice gegen die gepinnte Revision prüfen.

**Entscheidungen (2026-10-17):** gpt-oss-20b ist das erste neue produktive Modell, vor
Gemma 4. Kolibri ist auf dem RTX-Rechner machbar (96 GB RAM, die Experten liegen dort im
Arbeitsspeicher), kommt aber nach DeltaNet.

### Kandidaten

| Modell | Lizenz | Aufbau | neue Bausteine |
|---|---|---|---|
| gpt-oss-20b (120b) | Apache 2.0 | 20b: MoE 32 Experten top-4, 24 Layer (120b: 128 Experten, 36 Layer), abwechselnd Fenster 128 und voll | MoE, Sliding Window, Attention-Sinks, YaRN, Bias, begrenzte SwiGLU, MXFP4, o200k-Tokenizer, Harmony-Chatformat |
| Gemma 4 26B-A4B, 31B (E2B, E4B) | Apache 2.0 | 26B: 30 Layer, 5 lokal (Fenster 1024) : 1 global, 128 Experten top-8 + dichter Parallelzweig; global 2 KV-Köpfe mit Dimension 512; 31B dicht | Kopfdimension 512 auf der GPU, GELU-tanh, Logit-Softcap, teilweise RoPE, K=V, eigener Tokenizer; nur E2B/E4B zusätzlich Per-Layer Embeddings und geteilter KV |
| Qwen3.8-27B (`model_type` `qwen3_5`, wie Qwen3.5) | Apache 2.0 | 64 Layer = 16 × (3 Gated DeltaNet + 1 volle Attention mit Ausgangsgate) | Gated DeltaNet mit rekurrentem Zustand und Convolution-Puffer; Bildteil zunächst nicht |
| Kolibri-1 (Aleph Alpha) | Apache 2.0, nur Gewichte und Konfiguration | MoE 78B / 3,46B aktiv, 50 Layer, 4 Fenster (513) : 1 voll, NoPE in den vollen Layern, 384 Experten top-6 mit Sigmoid-Router + 1 geteilter | FP8 E4M3 in 128×128-Blöcken, safetensors in Teilen, Router mit Bias-Korrektur; GGUF nur von Dritten (Q4_K_M 47,5 GB) |
| Nemotron 3.5 Bolt | OpenMDW-1.1 | 74B-A9B, Mamba-2/Transformer-Hybrid, LatentMoE | Mamba-2, LatentMoE, IQ2_XXS (Experten) und NVFP4 (Mamba-Projektionen) |

Die Architektur bauen wir nach der Implementierung in `transformers` (Apache 2.0) nach,
nicht nach Hersteller-Code; das gilt auch für Kolibri, dessen Lizenz Code und Architektur
ausdrücklich ausnimmt.

### Was der Code heute mitbringt und was fehlt

- Gewichtsformate auf der GPU: F32, Q8_0, Q4_K, Q6_K. Neu wären MXFP4, FP8-Blöcke, IQ2_XXS, NVFP4.
- Tokenizer: Byte-BPE nur mit den Pre-Tokenizern von Qwen2 und o200k (gpt-oss); Unigram fest auf BGE-M3 zugeschnitten.
- `TQwen3Model.Forward` setzt gleich große Layer, QK-Norm, RoPE und SwiGLU voraus.
- Die GPU-Attention kann Kopfdimensionen bis 256 (Gemma 4 global braucht 512).
- `Truncate` des Kontexts setzt nur die Länge zurück. Mit Ringpuffer-KV oder rekurrentem
  Zustand geht das nicht mehr: Es braucht gültige Rücksetzpunkte, sonst einen Cache-Miss.
- safetensors nur als eine Datei `model.safetensors`, keine Teildateien mit Index.
- Kein eigenes Metal-Backend nötig: Apple läuft über Vulkan/MoltenVK.

### Testrechner

- RTX-Rechner: RTX 5090 mit 32 GB, 96 GB RAM. Echte große Modelle laufen hier.
- Mac M2 Pro: 16 GB RAM. Nur Tiny-Modelle und kleine echte Modelle (Qwen3.5-0.8B).
- Snapdragon X (Adreno): nur Tiny- und kleine Modelle.

### Slices

Jeder Slice wie bisher: FPC zuerst, Gegenproben, Review (Claude-Finder + Codex), Upstream-
Revision gepinnt, Tiny-Modell aus `transformers` als Referenz, echtes Modell hinter
`LAI_REQUIRE_MODEL`.

**D1 — Gemeinsame Decoder-Typen.** Reiner Umbau: `TDecoderModel`, Kontext und Gerät statt
`TQwen3…`, nicht-generisch. Eine Layer-Liste beschreibt je Layer getrennt den Mixer (volle
Attention, Fenster-Attention, DeltaNet, Mamba), den Zustand (KV, Fenster-KV, geteilter KV,
rekurrent mit Convolution-Puffer), das FFN (dicht oder MoE: Aktivierung, Router, Top-k-
Normierung, geteilter Zweig) sowie Dimensionen, Norm-/Residual-Anordnung, Bias und
Positionsverfahren (RoPE, teilweise RoPE, YaRN, NoPE). Tokenizer, Chat-Template und
Antwort-Parser hinter kleinen Schnittstellen; Backends melden vor dem Laden, was sie können.
Gebaut wird nur, was Qwen3 braucht. Gate: alle Qwen3-Tests (CPU, Vulkan, CUDA, Prompt-Cache,
Client) unverändert grün auf allen Compilern bis Delphi 7.
*Erledigt 2026-10-17:* `mormot.ai.infer.decoder` (`TDecoderModel`, `TDecoderContext`,
`TDecoderDevice`, `Spec` je Layer), `TInferTokenizer`, `TChatTemplate`,
`TNativeOutputParser`, `RegisterDecoderModel` und `RegisterNativeFamily`; wie man eine
Familie andockt: `native-inference.md`, Phase D. Die Layer-Liste hat erst die Arten, die
Qwen3 braucht; Zustand, Positionsverfahren und Bias kommen mit D2 bis D5. Dabei anzupassen
(Review D1): `TDecoderLayer` ist ein flacher Attention-plus-SwiGLU-Satz, und `CreateFromGguf`
nimmt den Gewichtstyp aus `fLayer[0].Q`; `Grow` legt für jeden Layer KV derselben Breite an;
`Rotary` ist eine Tabelle über die ganze Kopfdimension; `CheckConfig` verlangt
`Intermediate > 0`, `RopeTheta > 1` und prüft von `Spec` nur die Länge; die
Puffergrößen in `Forward`/`Grow` kennen nur `Hidden`, `Intermediate` und die Kopfbreite.

**D2 — MoE auf der CPU.** Router (Softmax und Sigmoid, Top-k, optional normiert), gewichtete
Summe der Experten, Experten als Sichten auf die Gewichte; bestehende Quant-Kernel weiter
nutzen. Decode rechnet nur die gewählten Experten, Prefill gruppiert die Tokens je Experte.
Erstes Modell **Qwen3-30B-A3B** (nur Zwischenschritt für MoE, kein Kandidat): reines MoE ohne neuen Tokenizer, ohne Sliding Window, ohne
neues Quant-Format. Gate: Tiny-Qwen3-MoE gegen `transformers` (Experten-IDs, Gewichte,
Logits), Einzeltoken und Prefill gleich; Qwen3-30B-A3B Q4_K_M auf dem RTX-Rechner.
*Erledigt 2026-10-17* (`native-inference.md`, Phase D): `dfMoe`, `TQwen3MoeModel` aus
safetensors und GGUF; Router nur Softmax (Sigmoid kommt mit Kolibri, D8). Qwen3-30B-A3B auf
der CPU des RTX-Rechners: Decode 16, Prefill 85 Tokens/s, antwortet richtig. Prefill
verliert Zeit in vielen kleinen Produkten (128 Experten mit je wenigen Tokens).

**D3 — MoE auf CUDA und Vulkan.** Expertenauswahl bleibt auf dem Gerät; Decode mit kleinen
Produkten der gewählten Experten, Prefill gruppiert. Erst korrekt ohne CUDA-Graph, danach ein
Graph mit wechselnden Experten-IDs. Gate: CPU und GPU gleich bei wechselnder Auswahl und
leeren Gruppen; Prefill/Decode und Speicherspitze gemessen auf RTX und M2 Pro (Tiny).
*D3a erledigt 2026-10-17: CUDA* (`native-inference.md`, Phase D) - erst korrekt, die Wahl
noch auf dem Host: Qwen3-30B-A3B auf der RTX 5090 Decode 33 bis 35, Prefill 384 bis 494
Tokens/s.
*D3b erledigt 2026-10-19: Vulkan* (`native-inference.md`, Phase D) - Bindungen mit Versatz
(`BatchDispatchAt`), `fill0`/`gatherrows`/`scatteradd.comp`, ein Warten je MoE-Layer, Batch
geteilt, wenn die Deskriptor-Sets knapp werden: Qwen3-30B-A3B auf der RTX 5090 Decode 43 bis
44, Prefill 489 bis 642 Tokens/s (schneller als CUDA mit 41 und 383).
Die Expertenwahl im Vergleichstest ist vorgegeben (`TDecoderRouteTape`, umgesetzt wie unten
beschrieben; unter Delphi 13 bleibt eine Abweichung aus der CPU-Rechnung, B13).
Auf dem M2 Pro (MoltenVK) grün, 2026-10-19: volle Infer-Suite, 2942 Assertions, darin das
Tiny-MoE auf der GPU. Die Wahl auf dem Gerät: beim Decode erledigt, siehe D3c.
*Mit D3b (entschieden 2026-10-17): Expertenwahl im Vergleichstest vorgeben.* Heute misst
`Qwen3MoeGgufMatchesCpu` Rechengenauigkeit und Routing in einem Kosinus: unter Delphi 13 kippt
bei einer knappen Router-Entscheidung ein Experte gegenüber der CPU, deshalb 0,98 und 14 von
16 statt 0,99 und 16 von 16 (FPC erreicht 0,992). Künftig getrennt prüfen:
1. Ein Test-Hook im Decoder gibt der GPU die Wahl der CPU vor (Experten-Ids und Gewichte je
   Token und Layer), so wie die Tokens heute schon vorgegeben werden. Damit gilt wieder die
   scharfe Schwelle (Ziel: Kosinus 0,99 und Argmax 16 von 16 auf beiden Compilern - erst
   messen, dann festlegen).
2. Ein zweiter Lauf ohne Vorgabe zählt die abweichenden Wahlen; erlaubt nur dort, wo der
   Abstand der Router-Logits unter einer kleinen Grenze liegt (Grenze aus der Messung).
Für CUDA und Vulkan gleich; die Gegenprobe muss zeigen, dass ein falsches Routing auf dem Gerät
(z. B. Router-Logits der falschen Zeile) in Prüfung 2 scheitert, obwohl Prüfung 1 es nicht
sehen kann.

**D3c — Expertenwahl auf dem Gerät (aufgenommen 2026-10-19).** Anlass: Decode von
Qwen3-30B-A3B bleibt auf der RTX 5090 bei 41 (CUDA) und 43 bis 44 Tokens/s (Vulkan), weil der
Host in jedem MoE-Layer die Router-Logits abholt, die Experten wählt und Listen hochlädt
(CUDA drei Wartepunkte je Layer, Vulkan einer) und für jeden Experten mit Tokens sechs
Kernel startet - die Rechenleistung ist nicht der Engpass.
*Bau:*
1. Route-Kernel/-Shader: Softmax in double-genauer Reihenfolge wie `TDecoderModel.Route`
   (oder float mit begründeter Toleranz), die k höchsten Logits, bei Gleichstand die niedrigere
   Id, Normierung nach `NormTopK` - Ids und Gewichte in Gerätepuffer.
2. Die Token-Zeilen je Experte auf dem Gerät (Zählen, Präfixsumme, Einsortieren in
   Id-Reihenfolge): dieselbe Reihenfolge wie heute, damit Summen gleich bleiben.
3. Decode (eine Zeile): ein Matvec-Aufruf über die k gewählten Experten, die Experten-Ids aus
   dem Gerätepuffer (Gewichte eines Layers ein Puffer, Versatz je Id) statt einer Schleife
   über alle Experten; Prefill weiter gruppiert je Experte, die Gruppen ohne Host-Wissen
   (Dispatch über die Höchstzahl, leere Gruppen tun nichts, oder indirekter Dispatch).
4. CUDA: Decode wieder als CUDA-Graph (`UseGraphs`); Vulkan: ein Token wieder ein Batch ohne
   Zwischen-Submit.
*Gate:* die heutigen Tests unverändert grün (`TinyMoeMatchesCpu`, `Qwen3MoeGgufMatchesCpu`);
das Routing-Band vergleicht dann die Wahl des Geräts direkt mit der CPU (`TDecoderRouteTape`
braucht dafür eine Rückmeldung der Wahl aus dem Gerätepuffer, oder ein Test liest die Ids
zurück); Gegenproben (falscher Tie-Break, Gewicht nicht normiert, Gruppe eines Experten
leer, Graph mit veralteten Ids). Decode vorher und nachher auf RTX 5090 (CUDA, Vulkan) und M2
Pro (Tiny) - übernommen nur, wenn messbar schneller.
*Erledigt 2026-10-19 für den Decode* (`native-inference.md`, Phase D): Route-Kernel und
-Shader, `linear8ids` / `linear8.comp -DIDS`, `moecombine`; Qwen3-30B-A3B auf der RTX 5090 von
38,9 auf 97,3 Tokens/s (CUDA, wieder ein Graph) und von 42,9 auf 90,4 (Vulkan, ein Batch je
Token). *Offen:* Prefill ohne Host-Wissen (Punkt 3: Gruppen über die Höchstzahl oder
indirekter Dispatch) - nur wenn ein Prefill-Profil zeigt, dass die Wartepunkte dort zählen;
heute 383 / 490 Tokens/s.

**D1b — Qwen2.5 als Familie (aufgenommen 2026-10-19).** Anlass: neural-api zeigt einen Chat mit
Qwen2.5-0.5B-Instruct in wenigen Zeilen (nur die Funktion als Anregung, kein Code - LGPL, keine
Vorlage). Qwen2 ist Qwen3 ohne `q_norm`/`k_norm` je Kopf, dafür mit Bias auf Q, K und V (O ohne
Bias); RoPE, SwiGLU, RMSNorm, GQA wie gehabt. Tokenizer (`TBpeTokenizer`, Qwen2-Regex) und
Chat-Template (ChatML, Qwen2.5-Tool-Calling) können es schon; zu prüfen ist das
Antwortformat ohne `<think>` und der System-Prompt, den das Template von Qwen2.5 ohne
System-Nachricht selbst einsetzt. Nutzen: Qwen2.5 0.5B bis 72B, Qwen2.5-Coder und viele
Fine-Tunes - die meisten kleinen, gut laufenden Modelle für CPU-Rechner.
*Bau:* `TQwen2Model` (`qwen2` / GGUF `qwen2`) auf `TDecoderModel`; Bias als Eigenschaft im
Layer-Spec (D1 hat ihn ausdrücklich offen gelassen) - auf CPU, Vulkan und CUDA, und so, dass
gpt-oss (D5, Attention mit Bias) ihn wiederverwendet; Norm je Kopf abschaltbar.
`RegisterNativeFamily` für den Client. *Gate:* Tiny-Qwen2 aus `make_reference_qwen3.py`
(BF16, Bias ungleich null, `head_dim` ungleich Hidden/Köpfe) gegen transformers auf allen
drei Backends; Qwen2.5-0.5B-Instruct (safetensors und GGUF Q8_0) Greedy-Tokens gleich
transformers, ohne Modell überspringen; Client-Test mit Tool-Aufruf; Gegenproben (Bias
vergessen, Bias auf O, Norm je Kopf doch an). Revision des Modells pinnen.
*Einordnung:* vor D4 oder parallel dazu - klein, unabhängig von MoE.
*Erledigt 2026-10-19* (`native-inference.md`, Phase D): `TQwen2Model` (`qwen2`, safetensors und
GGUF), Bias auf Q/K/V in `TDecoderLayer` (CPU, Vulkan, CUDA; `RecordLinear`/`EnqueueLinear`
mit Bias, `qkrope` mit Schalter `norm`), `TQwen2ChatTemplate`, `TQwen2OutputParser`, Familie
registriert. Qwen2.5-0.5B-Instruct in FP32 Greedy identisch mit transformers, Client mit
Tool-Aufruf; Qwens Q4_K_M-GGUF läuft seit B16 (Q5_0).

**D4 — Sliding Window und Cache-Semantik.** Erst voller KV mit Fenstermaske als Referenz,
dann Ringpuffer mit absoluten Positionen (spart den Speicher). Chunk-Prefill darf benötigte
Schlüssel nicht vorher überschreiben. NoPE als Layer-Eigenschaft. Prompt-Cache mit gültigen
Rücksetzpunkten statt bloßem Kürzen. Gate: Fenstergrenze, mehrfacher Umlauf, verschiedene
Chunk-Größen und verzweigte Prompts geben dieselben Logits wie eine Neuberechnung.
*Erledigt 2026-10-19* (`native-inference.md`, Phase D) auf der CPU: Fenster und NoPE als
Layer-Eigenschaft, schmaler Cache, der statt eines Ringpuffers die gebrauchten Zeilen an den
Anfang schiebt (zusammenhängend, so liest die Attention ohne Umlauf), `Restorable` /
`Usable` als Rücksetzpunkte, Konfiguration von Qwen3, Qwen2, Qwen3-MoE und neu SmolLM3 wie
transformers. Gate bitgleich auf beiden CPU-Backends.

**D4b — Fenster und NoPE auf Vulkan und CUDA (aufgenommen 2026-10-19).** Heute lehnen beide
Geräte solche Modelle ab. Nötig: das Fenster in den Attention-Kerneln (erster Schlüssel je
Zeile), der schmale Cache auf dem Gerät (Verschieben als Kopie im Batch bzw. Graph), NoPE
als Schalter in `qkrope`. Erst mit einem Modell, das es braucht (gpt-oss-20b, D5; Gemma,
D6) - dann gegen die CPU wie die übrigen Gerätetests. Dazu offen aus D4: SmolLM3 als
Chat-Familie (Template mit `/think` und `/no_think`, Parser) und aus GGUF (llama.cpp legt die
NoPE-Layer fest, das wäre gegen die Referenz zu prüfen).
*Erledigt 2026-10-19* (`native-inference.md`, Phase D), vorgezogen vor D5: Fenster und NoPE in
den Attention-Shadern und -Kerneln (der Schlüsselbereich jeder Zeile beginnt am Fenster) und
in `qkrope`; der Cache auf dem Gerät hält alle Zeilen, so kann ein Gerätekontext auf jede
Länge zurück. Gegen die CPU mit `tiny-qwen3-sliding` und `tiny-smollm3`. *Offen:* der schmale
Cache auf dem Gerät (nur Speicher, erst wenn ein Modell es lohnt - gpt-oss-20b), SmolLM3 als
Chat-Familie und aus GGUF.

**D5 — gpt-oss-20b.** MXFP4 (erst skalar entpacken, dann direkt im Linear rechnen, nie das
ganze Modell nach F32), Attention-Sinks (gelernter Zusatzterm im Softmax-Nenner je Kopf, keine
festgehaltenen Anfangstokens), YaRN, Bias in Attention und Experten, die begrenzte
SwiGLU-Variante; safetensors in Teildateien mit Index (Referenzgewichte); o200k-Tokenizer, Harmony-Chatformat und Parser (Kanäle, Tool-Aufrufe). Erst
CPU, dann Vulkan und CUDA. Gate: Tiny-gpt-oss gegen `transformers`, Token-IDs gegen den
Referenz-Tokenizer, Harmony-Rendering und -Parsing; gpt-oss-20b auf dem RTX-Rechner.
In Scheiben:
- *D5a erledigt 2026-10-19* (`native-inference.md`, Phase D): die Rechnung auf der CPU -
  Sinks, YaRN, alle Biases, die begrenzte SwiGLU, die Experten im Layout von transformers 5;
  `tiny-gpt-oss` gegen transformers. Die Geräte lehnen gpt-oss ab (`TDecoderModel.Extras`).
- *D5b erledigt 2026-10-20* (`native-inference.md`, Phase D): MXFP4 (`idtMXFP4`, Blöcke wie
  ggml `block_mxfp4`) auf der CPU - als Q8_0-Blöcke entpackt (die verdoppelten E2M1-Werte,
  2^(e-128) als FP16-Skala: exakt für e 104..143, gpt-oss-20b nutzt 115..136), also über
  die vorhandenen Q8_0-Kernel; die `*_blocks`/`*_scales` der veröffentlichten safetensors
  beim Laden umgepackt; safetensors in Teildateien mit `model.safetensors.index.json`;
  gpt-oss-20b auf der CPU des RTX-Rechners gegen transformers.
- *D5b2 erledigt 2026-10-20* (`native-inference.md`, Phase D): GGUF `gpt-oss` - Tiny-GGUF
  bitgleich zum safetensors-Ordner, `ggml-org/gpt-oss-20b-GGUF` mit Experten Byte für Byte
  gleich unserem Umpacken. *Offen:* ein fusionierter MXFP4-Kernel (Skalarprodukt ohne den
  Umweg über Q8_0-Blöcke) nur, wenn eine Messung das Entpacken noch als Bremse zeigt.
- *D5c1 erledigt 2026-10-20* (`native-inference.md`, Phase D): der o200k-Tokenizer aus
  `tokenizer.json` und aus GGUF (`gpt-4o`) - kein Normalizer, der o200k-Ausdruck von Hand
  mit einer eigenen gemessenen Klassentabelle, `ignore_merges`; 3095 Fälle stück- und
  idgleich mit der tokenizers-Bibliothek (ohne Added Tokens: Text bleibt Text).
- *D5c2 erledigt 2026-10-20* (`native-inference.md`, Phase D): gpt-oss hinter `ILlmClient` -
  das Harmony-Template des Checkpoints (437 Fälle text- und idgleich mit transformers, die
  TypeScript-Umsetzung der Tool-Schemas samt ihrer Eigenheiten), der Parser (Kanäle,
  Tool-Aufrufe), `reasoning_effort`; gezogen werden nur Ids des Tokenizers. gpt-oss-20b
  beantwortet greedy eine Rechenfrage und ruft ein Tool auf.
- *D5c3 erledigt 2026-10-20* (`native-inference.md`, Phase D): das Reasoning früherer Züge
  geht ans Template zurück - `TLlmMessage.Reasoning`, `TLlmChatResponse.Reasoning`, die
  Agentenschleife füllt es; gpt-oss als Analyse vor den Aufrufen, Qwen3 als `<think>`
  (`reasoning_content`); das OpenAI-Wire liest es und schickt es zurück
  (`reasoning_content`, DeepSeek verlangt das mit Tools; Ollama: `reasoning`).
- *D5d1 erledigt 2026-10-20* (`native-inference.md`, Phase D): gpt-oss auf CUDA - Sinks,
  alle Biases, die begrenzte SwiGLU, MXFP4 als Typ der Linear-Kernel (auch `linear8ids`);
  gpt-oss-20b auf der RTX 5090 gegen transformers (näher als die CPU) und hinter
  `ILlmClient`, Decode 124 Tokens/s.
- *D5d2 erledigt 2026-10-20* (`native-inference.md`, Phase D): dasselbe auf Vulkan - Sinks
  in allen Attention-Shadern, Biases (Experten-Zeile als Index im Push), `swiglu.comp` mit
  alpha und Grenze, MXFP4 in `linear8`/`gemm8`; gpt-oss-20b Decode 127 Tokens/s.
  *Offen, nur bei Bedarf:* der schmale Cache auf dem Gerät (gpt-oss-20b hält 24 Layer, die
  Hälfte mit Fenster 128: bei langem Kontext spart er rund die Hälfte des KV-Speichers).

**D6 — Gemma 4 26B-A4B und 31B.** GELU-tanh, Norm-/Residual-Pfade, Router-Skalierung,
Logit-Softcap, K=V, der dichte Parallelzweig neben den Experten (26B), lokale und globale
Kopfdimension verschieden (global 512: Attention-Kernel auf Vulkan und CUDA erweitern),
teilweise RoPE, eigener Tokenizer exakt nach der Referenz.
E2B/E4B (Per-Layer Embeddings, geteilter KV) danach als eigener Schritt. Gate: Tiny-Gemma mit
globalen Köpfen, echtes Modell quantisiert auf dem RTX-Rechner.

**D7 — Gated DeltaNet.** Rekurrenz, Convolution-Zustand, Gates, Normierung, Ausgangsgate der
vollen Attention. Erst seriell auf der CPU, dann Chunk-Prefill, dann GPU. Prompt-Cache mit
Zustands-Checkpoints. Erst **Qwen3.5-0.8B** (läuft auf allen Testrechnern), dann
**Qwen3.8-27B** (volle Attention mit Kopfdimension 256). Gate: seriell und in Chunks gleich,
Checkpoint-Restore und verzweigte Prompts gleich wie Neuberechnung.

**D8 — Kolibri-1.** FP8 E4M3 mit 128×128-Skalen (oder eine geprüfte Umwandlung nach Q4_K/Q8_0
beim Laden; ein GGUF von Dritten nur nach Prüfung gegen die Referenz), Sigmoid-Router mit Bias-Korrektur nur für die Auswahl,
geteilter Experte, NoPE in den vollen Layern. Auf dem RTX-Rechner: Experten im RAM (CPU),
Attention und dichte Teile auf der GPU. Gate: Tiny-Kolibri mit Router-Bias und 513er-Fenster
gegen `transformers`; echtes Modell auf dem RTX-Rechner mit gemessenem Speicherbedarf.

**D9 — Mamba-2, LatentMoE, IQ2_XXS, NVFP4 (Nemotron 3.5 Bolt).** Nur bei konkretem Bedarf.
Gate: Tiny-Mamba-2 und LatentMoE, IQ2_XXS und NVFP4 einzeln, dann echtes Modell mit eigener
Speichermessung (NVIDIA nennt rund 22 GB Gewichte, 23,8 GB mit Laufzeitzustand und Drafter,
ohne Bildteil).

### Außerhalb von D

Reranker (BGE-Reranker), Qwen3-Embedding, BGE-M3 Sparse-/ColBERT-Ausgabe, Llama 3, Mistral:
je ein eigener Eintrag, sobald gebraucht. Llama/Mistral brauchen nach D1 vor allem den
llama-bpe- bzw. SentencePiece-Tokenizer. Bild- und Videoeingang (Qwen3.8, Gemma 4, Bolt) ist
nicht Teil von D.

---

## B5 — Plattform-Läufe nachziehen

Die README-Tabelle „Where it runs" hat veraltete Zellen:

- Windows x64 FPC 3.2.2 und Linux x86_64 (WSL auf dem RTX-Rechner): Inferenz von 821/828
  — vor den Qwen3-Tests (damalige Fußnote ⁵). Rezept: `docs/native-inference.md` „Plattformen und
  Tests" bzw. Notiz zum RTX-Rechner (WSL-Distribution gehört dort Konto `sh`).
- Windows ARM64 FPC 3.3.1: Inferenz „not yet run" (braucht `sqlite3-64.dll` für ARM64
  neben dem Binary).
- LLM-Spalte macOS/Windows ARM64 vor den Load-Governor-Tests (damalige Fußnote ⁴).

**Fertig, wenn:** die Zellen aktuelle Zahlen tragen und die Fußnoten entfallen.

**Erledigt (2026-10-17):** alle Zeilen außer Linux aarch64 und Delphi 7 (beide vom
2026-10-14 bis 16) neu gelaufen, Fußnoten zu alten Ständen entfallen.
Windows ARM64 (FPC 3.3.1, Entwicklungs-Compiler) erstmals mit Inferenz: zuerst 2542,
davon 5 rot; nach der SDOT-Erkennung für Windows ARM (`IsProcessorFeaturePresent`) 2644,
davon 4 rot (B10). Gegenprobe mit den x64-Builds (FPC 3.2.2, Delphi 13) unter Prism auf derselben
Adreno-X1-85-GPU: dieselben 4 Vulkan-Fehler (Q8_0, eine Zeile, 256 Eingänge) — nicht der
Compiler, siehe B10. Der Qwen3-Cosinus 0,979 trat nur im Pascal-Pfad unter FPC 3.3.1 auf
(ohne SDOT-Kernel) und ist mit ihnen weg; als Compiler-Vorbehalt vermerkt.

---

## B6 — Delphi-Build-Skript für die Demos

`demos/infer/llm-bench.lpr` und `embed-bench.lpr` werden nur mit FPC gebaut. Ein
`scripts/build-delphi-demos.ps1` (Delphi 13 Win64, analog `run-delphi-tests.ps1`) fehlt.

**Erledigt (2026-10-17):** `scripts/build-delphi-demos.ps1` baut alle 18 Demos (dieselbe
Liste wie `build-demo.sh`, `-Demo` für eine). Gefunden und behoben: `embed-reference`
nutzte das FPC-only `varQWord` (jetzt mORMots `varWord64`), `vector-bench` mischte
`GetTickCount64` (unter Delphi vorzeichenlos) mit Int64 (W1073); dcc findet Units neben
dem Programm nicht von selbst, das Skript nimmt das Demo-Verzeichnis in den Suchpfad.
- *Delphi 7 (2026-10-17):* `-Delphi7` (mit `-FastMM4`, wie `run-delphi7-tests.ps1`), alle 18
  Demos. Nachgezogen: `TAddTool`/`TAskClaudeTool` ohne Generics (`TMcpRecordTool` bzw.
  `IMcpTool` von Hand - ein Record nur aus Integern hat unter Delphi 7 keine RTTI), kein
  `exit(wert)` mehr in den Demos, `vendor/mormot.ext.os` mit `TStartupInfoW` und
  `CREATE_BREAKAWAY_FROM_JOB` für Delphi 7 (lokale Änderung, im Unit-Kopf vermerkt), und
  `NoRedirect` statt `nil` für `TOnRedirect` - Delphi 7 löst ein nacktes `nil` für einen
  Methodenzeiger unter den `RunRedirect`-Überladungen nicht auf.
- *Fund dabei, behoben:* Die Delphi-13-`llm-bench` war auf dem i9-14900K viel langsamer
  als die FPC-Exe (Q4_K_M, 200 Tokens: Prefill 216 statt 579, Decode 14 statt 78
  Tokens/s), Attention 80 % der Decode-Zeit, mit 1 Thread aber gleichauf. Ursache:
  `CausalRows` legte bei jedem Aufruf (je Kopf und Token, aus allen Workern zugleich)
  dynamische Arrays an; FPC hat einen Heap je Thread, Delphis Speichermanager
  serialisiert. Jetzt Stack-Puffer bis 4096 Schlüssel (Heap darüber, Tests an beiden
  Grenzen): Delphi 468 / 68 Tokens/s, FPC unverändert.

---

## B7 — Kontextlänge für Embeddings (entschieden 2026-10-17: 512 bleibt)

512 Tokens wie im Vertrag v1 oder mehr? Bestimmt den Speicherbedarf der Attention
(`native-inference.md`, „Offene Entscheidungen" Nr. 3).

**Entschieden (2026-10-17):** 512 Tokens bleiben. RAG zerlegt Dokumente ohnehin in Stücke
(`ChunkText`), kurze Stücke treffen bei der Suche genauer, die Attention bleibt klein, und die
Verträglichkeit mit gespeicherten lembed-Vektoren (Vertrag v1) bleibt erhalten. Ein Text
über 512 Tokens wird weiter abgewiesen, nicht abgeschnitten. Eine größere Grenze erst, wenn
ein Anwendungsfall ganze lange Texte als einen Vektor braucht.

---

## B8 — Vulkan auf macOS: Leistung unter MoltenVK

**Ziel:** Auf dem M2 Pro soll die GPU schneller sein als der NEON-Pfad der CPU, auch bei
kurzem Kontext. Ein natives Metal-Backend folgt nur, wenn MoltenVK selbst bremst
(Roadmap: „Apple-GPU“ in `docs/native-inference.md`).

**Ausgangslage:** Seit 2026-10-16 läuft das Vulkan-Backend auf dem Mac über MoltenVK.
- **Korrektheit:** Die Suite ist grün mit `LAI_REQUIRE_VULKAN=1`.
- **Prefill:** Qwen3-0.6B Q8_0 schafft auf der GPU 305 Tokens/s bei 200 Tokens, die CPU 563.
  Bei 4000 Tokens ist die GPU mit 194 gegen 145 vorn.
- **Decode:** 63 Tokens/s auf der GPU gegen 73 auf der CPU bei 200 Tokens, 34 gegen 21 bei
  4000 Tokens.
- **Profil:** `Profile` misst unter MoltenVK nichts, weil alle Zeitstempel eines Batches
  gleich sind.

**Erledigt (2026-10-16):**
- Eine Wanduhr-Sonde misst je Kernel.
- MoltenVK läuft ohne Metal-Argumentpuffer. Barrieren kosten dann 5 statt 25 µs, der Decode
  gewinnt 5,6 %.
- Die Sonde zeigt die Kernel als Engpass, nicht MoltenVK:
  - Decode-Linears erreichen 40 bis 57 GB/s von rund 200 GB/s.
  - Der Prefill-Linear erreicht rund 400 GFLOPS von 6,8 TFLOPS.

**Vorgehen:**
1. **Decode-Matvec, erledigt 2026-10-16:** `matvec8.comp` auf Apple-GPUs, für Q8_0, Q4_K und
   Q6_K, je Typ ein eigener Build. Der Decode bei 200 Tokens steigt bei Q8_0 von 65,8 auf
   98,3 Tokens/s, bei Q4_K_M von 56 auf 83,6. Die CPU schafft 73 bzw. 75.
2. **Prefill-GEMM, erledigt 2026-10-16:** `gemm8.comp` auf allen Geräten. Der Prefill steigt
   bei 200 Tokens auf dem M2 Pro von 303 auf 1125 Tokens/s, auf der RTX 5090 von 10 000
   auf 13 600.
3. **Prefill-Attention, erledigt 2026-10-16:** `flash.comp` auf Apple-GPUs. Der Prefill bei 4000
   Tokens steigt von 381 auf 708 Tokens/s; auf der RTX ist der Kernel langsamer.
4. **Offen:** die Decode-Attention bei langem Kontext. Der Decode fällt auf dem M2 Pro von
   90 Tokens/s (200 Tokens Kontext) auf 40 (4000). `attention.comp` erreicht dort 63 GFLOPS.
   *Erster Schritt, 2026-10-17:* KV-Cache in Halbgenauigkeit (`TQwen3Vulkan.KvF16` und
   `TQwen3Cuda.KvF16`, Vorgabe an; `docs/native-inference.md`): auf der RTX 5090 Decode bei
   8000 Tokens +8 % unter Vulkan, +24 % unter CUDA. Auf dem M2 Pro dagegen +0 bis 2 %
   (4000 Tokens 43,4 statt 42,9): Dort begrenzt nicht die Bandbreite, sondern
   `attention.comp` selbst.
   *Erledigt 2026-10-17:* `attndec.comp` (eine Subgroup je Arbeitsgruppe, ohne Shared Memory
   und Barrieren), auf Apple-GPUs ab 512 Schlüsseln: Decode auf dem M2 Pro bei 1000 Tokens
   101 statt 79, bei 4000 74 statt 43, bei 8000 57 statt 28 Tokens/s.
5. Erst danach über ein Metal-Backend entscheiden.

Mac-Lauf: Homebrew `molten-vk` und `vulkan-loader` sind auf `macos-tests` installiert;
Vorgehen wie in `docs/native-inference.md` (Plattformtabelle).

**Fertig, wenn:** Prefill und Decode bei 200 Tokens auf der GPU schneller sind als auf
der CPU, oder belegt ist, dass MoltenVK die Grenze setzt.

---

## B9 — Vulkan-Attention: Test für Kopfdimensionen über 128 (erledigt 2026-10-16)

Der Zweig für die Dimensionen `t + 128` in `tools/vk/attention.comp` (`acc1`, Kopfdimension
129 bis 256) läuft jetzt in `AttentionRangesMatchCpu`, mit `tests/data/infer-ref/tiny-qwen3-hd256`:
- Das Modell ist ein Tiny-Qwen3 mit 2 Köpfen zu 256 über 1 KV-Kopf.
- Embedding, Normen und MLP stammen aus `tiny-qwen3`.
- Die Attention-Gewichte sind geseedet. Sie kommen aus `tools/infer-ref/make_tiny_headdim.py`,
  das nur numpy braucht, kein torch: Der Test vergleicht das Gerät mit dem CPU-Backend.

Prompt und Decode sind mit und ohne Splits grün. Die Gegenprobe (`acc1` nicht neu skaliert)
lässt 6 Assertions scheitern. `flash.comp` nimmt nur Kopfdimensionen bis 128; hier läuft
deshalb `attention.comp`.

---

## Nicht mehr verfolgen

- SQLite-Extensions `vec0` / `lembed`: fliegen langfristig raus (nativer Store und native
  Engine ersetzen sie) — kein Ausbau mehr.

---

## B10 — `matvec8.comp` Q8_0 rechnet auf der Adreno X1-85 falsch (erledigt 2026-10-17)

**Befund (2026-10-17):** `TTestInferVulkan.LinearMatchesReference` scheitert auf der
Adreno-GPU des Snapdragon-Laptops an Q8_0 mit einer Tokenzeile, egal mit welchem Compiler
(FPC 3.3.1 nativ, FPC 3.2.2 und Delphi 13 unter Prism, identische Fehlerwerte). Eine
Probe über Eingänge 32 bis 1024, 1 bis 3 Zeilen, 1 bis 33 Ausgänge, mit und ohne Bias
grenzt ein: falsch ist nur `matvec8` für Q8_0 (mit und ohne Subgroups); `linear8` und
`matvec8` für Q4_K/Q6_K sind richtig. Grün bleibt nur, wo jede Gewichtszeile an einer durch
4 teilbaren Byte-Adresse beginnt (32 Eingänge, 1 Ausgang) — Verdacht: der Adreno-Compiler
übersetzt das branchfreie `word2`/`half16` an 2-Byte-versetzten Adressen falsch.

**Wirkung:** keine im Produktivbetrieb — `matvec8` ist nur auf Apple-GPUs an (`UseMatvec`,
Vendor $106B); rot ist der Test, der `matvec8` auf jeder GPU erzwingt.

**Eingrenzung (2026-10-17, Probe `bin/winarm/adreno.lpr`, nicht im Repo):** 64 Eingänge,
1 Ausgang: GPU −8,568, Referenz −8,706. Einheit 0 (Byte 0) stimmt, Einheit 1 (Byte 34, also
2 mod 4) liefert einen um etwa 2,4 % zu kleinen Beitrag, als wären die unteren Bits ihrer
Skala oder einzelne Bytes falsch gelesen. Ohne Wirkung blieben: `word2` als `OpSelect`
statt Doppel-Shift, `dot4s` per Shift statt `bitfieldExtract`, die Skala per `word2` statt
`half16`. Rätselhaft bleibt, dass Q6_K (210-Byte-Blöcke, ebenfalls 2 mod 4) mit denselben
Helfern richtig rechnet.

**Vorgehen, wenn es gebraucht wird (die Adreno nutzt `matvec8` nicht):** einzelne Wörter
aus dem Shader in einen Debug-Puffer schreiben (gelesene Rohwörter je Einheit), mit den
Gewichtsbytes vergleichen; danach Treiber-Workaround oder `matvec8` im Test auf Apple
begrenzen. Jede Shader-Änderung im A/B-Lauf auf dem Mac (B8: Q8_0 98 Tokens/s).

**Erledigt (2026-10-17):** Die Debug-Variante schrieb die gelesenen Wörter einmal mit festen
Adressen und einmal aus der Schleife über die Einheiten in den Ausgabepuffer. Mit festen
Adressen stimmten alle, in der Schleife las Einheit 1 jedes Quant-Wort ein Wort zu früh (Wort
k−1 statt k). Der Adreno-Compiler rechnet `(a + 2 + 4k) >> 2` offenbar als `(a >> 2) + k` und
verliert den Übertrag, wenn `a` 2 mod 4 ist; Einheit 0 (`a` = 0) trifft das nicht, Q6_K auch
nicht (dort kommt zu einer Adresse 2 mod 4 nie eine 2 hinzu, die einen Übertrag auslöst).
Darum blieben alle drei Shader-Umbauten von `word2` wirkungslos: falsch war der Wortindex,
nicht das Zusammensetzen. Q8_0 rechnet den Index jetzt ohne diesen Übertrag
(`(a >> 2) + ((a >> 1) & 1)`, Bitlage `16 - (a & 2) * 8`, Helfer `wordAt`). Adreno: die
Probe über Eingänge, Zeilen, Ausgänge und Bias komplett grün, `LinearMatchesReference`
292/292; RTX 5090: Vulkan-Suite 360/360. Der SPIR-V-Code für Q4_K und Q6_K ist bis auf
die Nummerierung gleich, der für Q8_0 hat zwei Befehle weniger. A/B auf dem M2 Pro (Q8_0, 200
Tokens, drei Läufe im Wechsel): Decode 98,1 vorher, 107,5 Tokens/s nachher - der neue
Wortindex ist dort auch schneller.

## B11 — Qwen3 auf GPUs mit kleiner Puffergrenze (erledigt 2026-10-17)

**Befund:** Die Adreno X1-85 bindet höchstens 128 MB je Speicherpuffer
(`maxStorageBufferRange`). Der Kopf von Qwen3-0.6B Q8_0 (die Einbettung, 151936 x 1024) hat
165 MB, `TQwen3Vulkan.Create` brach ab: Auf Windows ARM scheiterten `Qwen3GgufMatchesCpu` und
`NativeClientOnDevice` mit einer Exception, in allen Läufen seit dem ersten.

**Lösung:** `TInferVulkanBackend.RecordLinear` rechnet eine Matrix, die nicht in eine Bindung
passt, in Teilen nach Ausgabe-Merkmalen (`LinearPart`: so viele Zeilen, wie in eine Bindung
passen). Jeder Teil schreibt in einen Zwischenpuffer, dessen Zeilen `TInferVulkan.BatchCopyRows`
mit einem einzigen Kopierbefehl an ihre Spalten im Ergebnis setzt; die Shader bleiben
unverändert. Den Zwischenpuffer stellt der Aufrufer (der Kontext, angelegt in `Prepare`), damit
nie ein Puffer entsteht oder freigegeben wird, während ein Batch aufgezeichnet ist;
`UploadLinear` lädt die Teile vorab in den Gewichts-Cache. Geteilt wird nur der Kopf, eine zu
große Schicht-Matrix wird wie bisher abgewiesen. `MaxLinearBytes` senkt die Grenze für Tests.

**Tests:** `LinearInParts` (F32, Q8_0, Q6_K; 1, 3 und 20 Zeilen, also `matvec8`, `linear8`,
`gemm8`; Teile 10, 10, 10, 7) gegen die Referenz, ohne Zwischenpuffer eine Exception;
`Qwen3GgufMatchesCpu` rechnet Q8_0 auf jeder GPU mit dem Kopf in Teilen von höchstens 48 MB.
Adreno: Q8_0 Kosinus 0,9991, Argmax 16 von 16 (Kopf in zwei Teilen, natürliche Grenze),
`NativeClientOnDevice` grün; RTX 5090, lavapipe, Delphi 13 und 7 grün. Gegenprobe:
falscher Zeilenabstand beim Kopieren (Ziel und Quelle), alle Teile in Spalte 0 oder aus den
ersten Gewichten, `MaxLinearBytes` übergangen, Kopf am Stück geladen, kein Zwischenpuffer, der
letzte Teil zu groß - alle acht scheitern.

**Tempo (offen, kein Auftrag):** Qwen3-0.6B Q8_0, 200 Tokens, auf dem Snapdragon X: Adreno
Prefill 368 und Decode 30 Tokens/s, die CPU desselben Rechners (8 Threads, NEON-Dotprod) 639
und 68. Die Adreno rechnet richtig, lohnt sich aber nicht; die Shader sind nicht auf sie
abgestimmt (`matvec8` dort aus, `gemm8` für RTX und M2 Pro gemessen).

---

## B12 — Telemetrie: Auslastung, Takt, Temperatur, Leistung (aufgenommen 2026-10-17)

Einordnung: nach D3 (MoE auf der GPU), dann sind die GPU-Werte der großen Modelle
aussagekräftig. Anlass: die Frage, was ein Lauf von Qwen3-30B-A3B auf der CPU an Last und
Wärme erzeugt - die Engine kann es heute nicht sagen, nur ein Skript von außen
(Windows-Leistungsindikatoren und `nvidia-smi`).

**Ziel:** eine Unit `mormot.ai.infer.telemetry`, die einen Messwert-Satz liefert, so weit die
Plattform ihn hergibt, und sagt, was fehlt (kein stilles 0):

| Wert | Windows | Linux | macOS |
|---|---|---|---|
| CPU-Auslastung | `GetSystemTimes` | `/proc/stat` | `host_processor_info` |
| effektiver CPU-Takt | `CallNtPowerInformation` (`ProcessorInformation`: aktuelle und maximale MHz je Kern) | `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq` | offen (vermutlich nicht ohne private Schnittstellen) |
| CPU-Temperatur | nicht ohne Sensortreiber (ACPI-Zonen meist leer) | hwmon (`coretemp`, `k10temp`; schon in `mormot.ai.governor`) | nein |
| GPU NVIDIA: Leistung, Temperatur, Auslastung, Takt | NVML (`nvml.dll` kommt mit dem Treiber; zur Laufzeit laden wie `libcuda`, nichts mitliefern) | NVML (`libnvidia-ml.so.1`) | - |
| GPU AMD | offen | hwmon von `amdgpu` | - |
| GPU Apple | - | - | keine öffentliche Schnittstelle |

**Verwendung:**

1. `mormot.ai.governor`: drosseln nach echten Signalen - sinkender Takt bei voller Last
   (thermische Drosselung), GPU-Temperatur oder -Leistung über einer Schwelle - statt nur
   nach der Durchsatz-Heuristik.
2. `llm-bench` und `TNativeLlmClient`: je Lauf Mittel und Spitze von Auslastung, Takt,
   Leistung und Temperatur; daraus Tokens je Joule für den Dauerbetrieb.
3. Verteilte Ausführung (`docs/distributed-execution.md`): ein Rechner meldet, ob er frei und
   kühl genug für weitere Jobs ist.

**Regeln:** keine DLL mitliefern, Bibliotheken des Treibers nur zur Laufzeit laden und ihr
Fehlen melden; Abfragen billig genug für einmal je Sekunde neben einem Lauf; Delphi 7 bis 13
und FPC wie überall.

**Fertig, wenn:** die Werte auf dem RTX-Rechner (Windows, NVIDIA), unter WSL bzw. Linux und
auf dem M2 Pro gegen eine Gegenmessung stimmen (Windows-Leistungsindikatoren, `nvidia-smi`,
`sensors`), der Governor mit einem Takt- oder Temperatursignal drosselt (Test mit
eingespeisten Werten) und `llm-bench` die Werte ausgibt.

**Ausgangswert (2026-10-17, RTX-Rechner, i9-14900K + RTX 5090, über SSH gemessen;
Auslastung per `GetSystemTimes`, GPU per `nvidia-smi`, je Sekunde):**

| Lauf | CPU | GPU-Leistung | GPU-Temperatur | GPU-Last, Takt |
|---|---|---|---|---|
| Leerlauf | 4 % | 20 W | 41 °C | - |
| Qwen3-30B-A3B Q4_K_M, CPU, 22 Threads: Prefill 2000 Tokens (67 Tokens/s), 256 Tokens Decode (11 Tokens/s), 52 s | 69 % (22 von 32 Threads voll) | 19 W | 40 °C | 0 % |
| Qwen3-0.6B Q8_0, Vulkan: Prefill 8000 (6786/s), 2048 Decode (254/s), 10 s | 6 % | Mittel 318 W, Spitze 477 W | 51 bis 56 °C | 85 %, 2,76 GHz |
| dasselbe, CUDA: Prefill 7500/s, Decode 304/s | 7 % | Mittel 341 W, Spitze 495 W | 50 bis 53 °C | 90 %, 2,75 GHz |

Die GPU-Läufe sind zu kurz für einen Wärme-Gleichgewichtszustand. Selbst das kleine Modell
treibt die 5090 an ihre Leistungsgrenze (rund 0,8 Decode-Tokens je Joule bei Vulkan).

**Befunde für den Bau:**

- In einer SSH-Sitzung (Dienstanmeldung) verweigert Windows Leistungsindikatoren
  (`Get-Counter`: c0000bc8) und WMI (`Win32_PerfFormattedData_*`, `MSAcpi_ThermalZone`:
  Zugriff verweigert). Die Telemetrie muss die Win32-API direkt nehmen; `GetSystemTimes`
  geht.
- `CallNtPowerInformation(ProcessorInformation)` meldet als `CurrentMhz` durchgehend 3200,
  den Basistakt, auch unter Volllast - für den effektiven Takt unbrauchbar. Offen bleibt,
  was ohne Treiber den echten Takt liefert (Leistungsindikator `% Processor Performance`
  nur mit Rechten; APERF/MPERF nur im Kernel). Eine Drosselung wäre dann nur über den
  Durchsatz zu erkennen, wie heute im Governor.
- CPU-Temperatur und -Leistung: unter Windows ohne Sensortreiber nicht zu bekommen,
  bestätigt.

## B13 — CPU-Rechnung unter Delphi und FPC angleichen (aufgenommen 2026-10-17, erledigt 2026-10-19)

Anlass: Qwen3-30B-A3B gibt auf der CPU unter Delphi 13 andere Logits als unter FPC, genug, dass
bei knappen Router-Entscheidungen ein anderer Experte gewählt wird (Kosinus gegen CUDA 0,988
statt 0,992, Argmax 15 statt 16 von 16; B4, D3a).

**Befund 2026-10-19 (D3b):** mit vorgegebener Expertenwahl (`TDecoderRouteTape`) bleibt unter
Delphi 13 Kosinus 0,9975 und Argmax 15 von 16 gegen die GPU, unter FPC 0,9995 und 16 von 16 -
die Abweichung steckt also in der Rechnung selbst, nicht im Routing. Damit ist ein Vergleich je
Layer (Vorgehen 1) mit festem Routing möglich.

**Stand der Ursache (nicht gemessen):** die x64-Kernel sind es vermutlich nicht -
`InitKernels` setzt unter FPC und Delphi 11+ dieselben AVX2-Routinen (gemeinsamer Assembler,
`INFERAVX2`). Verdächtig ist der Pascal-Code drumherum (Normen, Softmax, RoPE, SwiGLU,
Akkumulationen in `TDecoderModel`): Delphi Win64 rechnet `single`-Ausdrücke ohne
`{$EXCESSPRECISION OFF}` in doppelter Genauigkeit, FPC in einfacher. Auch Unterschiede in
`Exp`/`Ln` der RTL oder in der Reihenfolge von Summen sind möglich.

**Vorgehen:**
1. Eingrenzen: dieselbe Eingabe, Zwischenergebnisse je Layer und Schritt (Norm, Attention,
   Router-Logits, FFN) unter beiden Compilern bitgenau vergleichen; der erste abweichende
   Schritt zeigt die Stelle.
2. Angleichen, wo es billig ist (z. B. `{$EXCESSPRECISION OFF}` in den Infer-Units, eigene
   `Exp` statt der RTL), und messen, ob Delphi dabei schneller wird - Rechnen in `double` und
   Umwandeln kostet in heißen Schleifen.
3. Delphi 7 (Win32, x87) bleibt außen vor: dort ist Gleichheit nicht erreichbar und nicht nötig.

**Fertig, wenn:** die Ursache benannt ist, die CPU-Logits unter Delphi 13 und FPC für das
Tiny-Modell und Qwen3-30B-A3B bitgleich oder nur noch in den letzten Bits verschieden sind und
der Decode unter Delphi nicht langsamer ist (`llm-bench`). Danach die Delphi-Ausnahme in den
Vergleichstests entfernen, falls die Vorgabe der Expertenwahl (B4, D3) sie nicht schon
überflüssig gemacht hat.

*Erledigt 2026-10-19* (`native-inference.md`, Phase D): Ursache war `{$EXCESSPRECISION}` - die
Gewichtung der Experten im MoE-Block rechnete Delphi Win64 in `double` mit einer Rundung,
FPC in `single` mit zwei. Alle Infer-Units setzen jetzt `{$EXCESSPRECISION OFF}` (Delphi
x64); die CPU-Ergebnisse sind unter beiden Compilern bitgleich, bis zu den Logits von
Qwen3-30B-A3B, und Delphi dekodiert 3-14 % schneller. Die Delphi-Ausnahme im Vergleichstest
ist entfernt.

## B14 — Rückmeldungen aus dem TTS-Projekt von flydev (aufgenommen 2026-10-18)

Anlass: flydev (flydev-fr/mormot.ai) baut mit Teilen von mormot2-ai ein TTS-Modell nativ in
Delphi 13 (Win64, Vulkan auf einer AMD RX 9060 XT): safetensors-Loader, Tokenizer, CPU- und
Vulkan-Backend und Sampler aus der Bibliothek, Talker, Code-Predictor und 12-Hz-Audio-Decoder
selbst gebaut; rund 5,9-fache Echtzeit, erster Ton nach rund 95 ms. Seine Stolpersteine im
Synopse-Forum (Thread „mormot2-ai“, Beitrag #31,
<https://synopse.info/forum/viewtopic.php?pid=45876#p45876>). Gegen den Code geprüft am
2026-10-18 (landrixai `a9ab1a7`):

1. **`InferRowToF32` kann kein BF16 (und kein F16).** Stimmt: `mormot.ai.infer.tensor.pas`
   kennt F32, Q8_0, Q4_K und Q6_K; BF16 wird im Decoder gesondert verbreitert
   (`TDecoderModel.WidenTensor`). Er hat die Zeilen selbst verbreitert.
   *Vorhaben:* BF16 und F16 in `InferRowToF32`; die Decoder-Verbreiterung darauf umstellen,
   wenn es nichts kostet. *Test:* Zeilen aus BF16 und F16 gegen die bekannten Werte
   (auch Unterlauf, Unendlich, NaN), Gegenprobe mit vertauschten Bytes.

2. **`TBpeTokenizer` lädt nur `tokenizer.json`.** Stimmt: es gibt `Create(tokenizer.json)`
   und `CreateFromGguf`. Sein Modell bringt `vocab.json` und `merges.txt` (das ältere
   GPT-2-Format); er hat abgeleitet und das geschützte `Build` gerufen.
   *Vorhaben:* ein Konstruktor für das Paar (`vocab.json` + `merges.txt`, dazu
   `added_tokens.json` und `special_tokens_map.json`, wenn vorhanden), der in dasselbe `Build`
   mündet; die Pre-Tokenizer-Regel muss dann ausdrücklich gewählt werden (GPT-2 als Vorgabe),
   weil die beiden Dateien sie nicht mitbringen. *Test:* dieselben Texte mit beiden
   Formaten geben dieselben Ids (Referenz aus `transformers`, Version vermerken).

3. **Kein Einstieg mit fertigen Embeddings.** Stimmt: `TQwen3Vulkan` (und ebenso CPU und
   CUDA) ruft `fModel.Embed(aIds, ...)` fest im Ablauf; er hat die Push-Constant-Records
   kopiert, um denselben Durchlauf selbst aufzubauen. TTS, Vision und Audio brauchen genau
   das. *Vorhaben:* in `TDecoderModel`/`TDecoderDevice` ein Forward, das statt Ids eine
   Matrix aus Embeddings (Zeilen x Hidden, F32) nimmt, gemischt mit Ids pro Chunk möglich
   (Text-Tokens plus eingeschobene Bild-/Audio-Embeddings), für CPU, Vulkan und CUDA gleich;
   Positionen und KV-Cache wie bei Ids. Passt zu Phase D (D1 hat den Decoder schon
   generisch gemacht); vor D6 Gemma 4 klären, ob dessen multimodale Teile dieselbe
   Schnittstelle brauchen. *Test:* Embeddings, die aus denselben Ids gewonnen wurden, geben
   dieselben Logits wie die Ids, auf allen drei Backends; gemischter Chunk gegen zwei
   getrennte.

4. **F16-KV-Cache ungenauer als F32** (bei ihm 2,3e-2 relativ gegen 2,6e-5). Kein Fehler:
   der Halbfloat-Cache ist die Vorgabe (`KvF16`), abschaltbar, und für langen Kontext gedacht.
   *Vorhaben:* in `native-inference.md` und am Property deutlicher sagen, was er kostet und
   wann man ihn abschaltet (kurzer Kontext, Talker-/Predictor-Schleifen, in denen Fehler
   über Schritte anwachsen); prüfen, ob die Vorgabe vom Kontext abhängen soll (z. B. F32 bis
   zu einer Länge). Entscheidung beim Nutzer.

5. **Heap-Allokation in einem Job des `TInferThreadPool` serialisiert am Memory Manager**
   (seine Umwandlung nach Q8 mit 22 Threads: 3,5 s statt 0,6 s mit einem Puffer auf dem Stack).
   Unsere eigenen Jobs in `mormot.ai.infer.backend.cpu.pas` allokieren nicht (die `SetLength`
   stehen vor dem Verteilen); der Fall war sein Job. Die Lücke ist die Schnittstelle:
   `TOnInferItem = procedure(aItem: integer)` kennt den Worker nicht, ein Job kann also keinen
   eigenen Scratch-Puffer je Thread nehmen. *Vorhaben:* die Regel am Pool dokumentieren (im
   Job nicht allokieren) und einen Weg zu Puffern je Worker anbieten - entweder der
   Worker-Index im Aufruf (neuer Job-Typ neben dem alten) oder ein Scratch-Speicher je Worker,
   den der Pool hält. *Test:* der Index liegt immer in 0 bis Workers und ist innerhalb eines
   Laufs je Thread eindeutig; Messung vorher/nachher mit einem allokierenden Job.

6. **Gewichts-Cache der GPU-Backends liefert nach Freigabe veraltete Daten** (gemeldet von
   flydev im Forum, Beitrag #34). `TInferVulkanBackend.Weight` und ebenso
   `TInferCudaBackend.Weight` suchen nur nach Adresse, Typ und Größe; werden Gewichte
   freigegeben und neue an derselben Adresse angelegt, kommt der alte Gerätepuffer zurück.
   Der Unit-Kopf nennt das als Vertrag (Gewichte nicht freigeben, solange das Backend lebt,
   sonst `ClearWeights`), sein TTS-Projekt hat genau diesen Umweg dokumentiert. Das ist eine
   Falle, sobald mehrere Modelle oder wechselnde Gewichte (TTS: Talker, Predictor, Decoder)
   ein Backend teilen. *Vorhaben:* ein Weg, gezielt zu vergessen, ohne alles zu leeren -
   z. B. `ForgetWeights(aFrom, aTo: pointer)` für einen Adressbereich, gerufen vom
   Besitzer der Gewichte beim Freigeben (`TDecoderModel`, Loader-Puffer), und
   `ForgetWeight(aData)`; dazu prüfen, ob ein Modell das Backend selbst benachrichtigen kann
   (Modell kennt sein Gerät über `TDecoderDevice`). Für Vulkan und CUDA gleich. *Test:*
   Gewicht hochladen, Host-Speicher freigeben, neu an derselben Adresse anlegen (erzwingen:
   derselbe Puffer mit neuem Inhalt), nach `ForgetWeights` die neuen Werte; Gegenprobe ohne
   den Aufruf liefert die alten.

**Referenzdaten für Punkt 3 (übernommen 2026-10-19):** flydev hat sie geliefert:
`tests/data/infer-ref/qwen3-tts-lm` (Embeddings rein, Logits raus, für Talker und
Code-Predictor von Qwen3-TTS-12Hz-0.6B-CustomVoice, PyTorch float32), Erzeuger unverändert
unter `tools/infer-ref/qwen3-tts/`, Herkunft in `UPSTREAM_BASE` und `NOTICE`. Der Test für
den Einstieg mit Embeddings lädt das Modell (Revision im Manifest, Hashes von
`config.json` und `model.safetensors` prüfen; ohne Modell überspringen wie die 30B-Tests),
schneidet die Stapel nach den Präfixen des Manifests heraus und vergleicht Zustände je
Layer und Logits. Beide Stapel sind gewöhnliche Qwen3-Decoder; die multimodale RoPE des
Talkers fällt bei gleichen Positionen auf die normale zurück. Sein Code ist öffentlich:
<https://github.com/flydev-fr/mormot2-qwen-tts> (TTS-Engine auf mormot2-ai als Submodul) -
dort zeigt `src/tts/mormot.ai.tts.qwen.lm.vulkan.pas`, welche Teile von `TQwen3Vulkan` er
nachbauen musste; nach Punkt 3 sollte das entfallen.

**Reihenfolge:** 3 zuerst (größter Nutzen, berührt die Phase-D-Schnittstellen), zusammen mit 6
(derselbe Anwendungsfall: mehrere Stapel auf einem Gerät), dann 1, 2 und 5 (klein,
unabhängig), 4 nur Doku und Entscheidung. Rückmeldung an flydev im Forum, wenn etwas davon
gepusht ist.

*Punkte 3 und 6 erledigt 2026-10-19* (`native-inference.md`, Phase D):
`TDecoderModel.ForwardEmbeds` auf CPU, Vulkan und CUDA, `TDecoderModel.CreateStack` (ein Stapel
eines größeren Checkpoints, z. B. Talker und Code-Predictor von Qwen3-TTS), `ForgetWeights`
an Backend und Modell; flydevs Referenz `qwen3-tts-lm` läuft als Test (Talker-Zustände bis
1,8e-4 gegen PyTorch, Predictor-Codes gleich). Neue Regel: ein Modell vor seinem Backend
freigeben.

*Punkte 1, 2, 4 und 5 erledigt 2026-10-19* (`native-inference.md`, Phase D): BF16 und F16 in
`InferRowToF32` (der Decoder verbreitert jetzt darüber); `TBpeTokenizer.CreateFromVocabMerges`
für `vocab.json` + `merges.txt` - statt GPT-2 als Vorgabe die Qwen2-Pipeline, geprüft an der
`tokenizer_config.json` daneben, weil die Klasse nur diese kann; angenommen wird nur, was
transformers 5.19 mit denselben Ids lädt; Qwen3-TTS gibt damit die Referenz-Ids. Zu 4
(Nutzerentscheidung): F16 bleibt Vorgabe, keine Umschaltung nach Länge; gemessen bringt F16
beim Decode 2-3 % bei 200 Tokens Kontext, 6-8 % bei 4000, 16-32 % bei 16000; `KvF16` und
`ForwardEmbeds` sagen jetzt, wann man es abschaltet (zurückgespeiste floats wie beim Talker).
Zu 5: `TInferThreadPool.RunWorkers` mit Worker-Index für Puffer je Thread; das Allokieren im
Job kostet nur unter Delphi (32 Threads: 67 gegen 12 ms), FPCs Heap arbeitet je Thread.
*Rückmeldung an flydev* im Forum gegeben (2026-10-20): sein Submodul-Stand `5b30ded` liegt vor B14, mit dem Hinweis auf die neuen Einstiege.

## B15 — Schneller Einstieg: Gewichtsvorgabe und Quickstart-Demo (aufgenommen 2026-10-19, erledigt 2026-10-19)

Anlass: neural-api (`neuralchat`) lädt ein Modell mit `LoadModel(dir)` und antwortet mit
`ChatReply(history)`, int8-Gewichte als Vorgabe (nur die Funktion als Anregung, kein Code).
Bei uns ist der Weg ähnlich kurz (`TNativeLlmClient.Create(folder)` hinter `ILlmClient`),
aber die Vorgabe ist `idtF32`: Token für Token gleich mit transformers (Qwen3-0.6B greedy),
dafür viermal so viel Speicher und rund halbes Decode-Tempo wie `idtQ8_0`. Q8_0 kann eine
knappe Entscheidung kippen (Qwen3-0.6B beantwortet die Wetterfrage der Referenz dann selbst,
statt das Werkzeug zu rufen) - deshalb steht F32 bisher als Vorgabe.

**Vorhaben:**
1. Entscheiden (Nutzer): Vorgabe bleibt F32, oder eine benannte Wahl - etwa eine Konstante
   bzw. ein Profil „genau“ (F32, für Tests und Vergleiche) gegen „schnell“ (Q8_0 beim Laden)
   - und welches davon `Create` ohne Angabe nimmt. Ein Wechsel der Vorgabe ändert die
   Client-Tests (die dann F32 ausdrücklich wählen).
2. Demo `demos/llm/llm-quickstart.lpr`: Ordner (git clone von Hugging Face) oder GGUF-Datei
   laden, eine Frage, Antwort gestreamt - so kurz wie möglich, mit dem Schnipsel im README
   der Bibliothek. Zeigt nebenbei, dass die Familie aus dem Ordner erkannt wird
   (`DecoderModelClassOfFolder`).
3. Prüfen, wie die Kontextlänge gedeckelt wird (neural-api: `CtxLen`, „hält den KV-Cache
   klein“): ob `TNativeLlmClient` eine Obergrenze hat und der KV-Cache nur so weit wächst,
   wie er gebraucht wird; falls nicht, eine Eigenschaft dafür.

**Fertig, wenn:** die Entscheidung zu 1 umgesetzt ist, das Demo unter FPC und Delphi 13 baut
und läuft (CPU, und mit Vulkan/CUDA, wenn vorhanden), und das README den Einstieg in
wenigen Zeilen zeigt. Weitere Ideen ohne Termin: [ideas.md](ideas.md).

*Erledigt 2026-10-19* (`native-inference.md`, Phase D): Q8_0 ist die Vorgabe von
`TNativeLlmClient.Create` (Nutzerentscheidung; `idtF32` für Tests und Vergleiche);
`MaxContext` deckelt ein Gespräch und damit den KV-Cache, der schon vorher nur nach Bedarf
wuchs; `demos/llm/llm-quickstart.lpr` lädt Ordner oder GGUF und streamt die Antwort, auf
CPU, Vulkan und CUDA unter FPC, Delphi 13 und Delphi 7 gelaufen; das README zeigt den
Einstieg in zehn Zeilen.

## B16 — Q5_0 (und Q5_1) für die K-Quant-Dateien kleiner Modelle (aufgenommen 2026-10-19, Q5_0 erledigt 2026-10-19)

Anlass: Qwens `qwen2.5-0.5b-instruct-q4_k_m.gguf` hat 133 Q5_0-Tensoren, nur 12 Q4_K und 12
Q6_K - llama.cpp nimmt Q5_0, wo eine Zeile kein Vielfaches von 256 ist (Hidden 896), auch für
`token_embd`. Der Loader lehnt die Datei ab (`unsupported ggml type 6 (Q5_0)`), Q8_0 läuft.
Dasselbe betrifft andere Modelle mit Breiten wie 896 oder 1536 (Qwen2.5-0.5B und -1.5B, viele
Fine-Tunes) und Dateien, die Q5_0/Q5_1 ausdrücklich wählen.

**Vorhaben:** `idtQ5_0` (Block 32: FP16-Skala, 4 Byte hohe Bits, 16 Byte Nibbles) und
`idtQ5_1` (zusätzlich FP16-Minimum) in Tensor-Typen, GGUF-Loader, `InferRowToF32`; CPU-Kernel
(Pascal, dann AVX2/NEON wie Q4_0: entpacken nach Q8_0 je Zeile oder eigene Dot-Produkte),
Vulkan- und CUDA-Linear (`dtype` 5/6 in den Shadern). *Gate:* Tiny-Modell als Q5_0/Q5_1-GGUF
aus ggml's Referenz-Quantisierer (`make_reference.py`, Ggml), Bits wie ggml; Qwen2.5-0.5B
Q4_K_M gegen FP32 (Kosinus, Greedy bis zur knappen Stelle) auf CPU, Vulkan, CUDA;
Gegenproben (hohes Bit vertauscht, Minimum vergessen). *Nutzen messen:* Q5_0 gegen Q8_0 bei
Qwen2.5-0.5B (Speicher, Decode) - übernehmen nur, wenn der Nutzen trägt.

*Erledigt 2026-10-19 für Q5_0* (`native-inference.md`, Phase 3): Typ, Loader, CPU (verlustfrei
nach Q8_0 ausgepackt), Vulkan und CUDA (`dtype` 5); Qwens Q4_K_M von Qwen2.5-0.5B läuft auf
CPU, Vulkan und CUDA, Speicher-Peak 542 statt 718 MB (Q8_0), Decode etwa gleich (CPU 101 /
98, CUDA 447 / 462, Vulkan 386 / 418 Tokens/s). Die Adreno X1-85 rechnete das um j
verschobene `qh`-Wort falsch; gelesen wird das eine Byte. *Offen, nur bei Bedarf:*

- **Q5_1** (Block 24: zusätzlich FP16-Minimum): llama.cpp nimmt es als Ersatz für Q5_K -
  eine Q5_K_M-Datei braucht also zuerst Q5_K. Auf der CPU nicht verlustfrei nach Q8_0
  auszupacken (Wert d · q + m): der Summand m · Summe(x) je Block käme dazu.
- **`matvec8.comp` für Q5_0**: Vulkan dekodiert Q5_0 mit `linear8`, 8 % langsamer als Q8_0
  mit `matvec8`.
- **Auspacken mit NEON/AVX2** wie `InferQ4Unpack`: Prefill auf der CPU 4 % langsamer als
  Q8_0 (891 / 931 Tokens/s).
