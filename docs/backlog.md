# Backlog — native Inferenz und Umfeld

Offene Themen, jedes so beschrieben, dass ein Agent ohne Vorwissen aus dieser Session
loslegen kann. Reihenfolge = grobe Priorität. Stand 2026-11-13.

Erledigte Einträge stehen mit ihrer Nummer (B…, D…) in [`CHANGELOG.md`](../CHANGELOG.md); die
ausführlichen Berichte in [`native-inference.md`](native-inference.md). Ein erledigter Eintrag
wird nicht abgehakt, sondern dorthin verschoben. B2 und B16 stehen in beiden Dateien: Der
CHANGELOG nennt den erledigten Teil, hier steht der offene Rest unter derselben Nummer.

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

## B20 — Decode großer Modelle unter der Bandbreite der GPU (aufgenommen 2026-11-13, Teil 1 erledigt)

Anlass: B19. Erster Teil erledigt 2026-11-13 (`CHANGELOG.md`, `native-inference.md`): Die
Normen einer Zeile laufen beim Decode mit Gruppen von 1024 (CUDA, Vulkan wo möglich);
Qwen3.8-27B auf der RTX 5090 von 36 bis 39 auf 41 bis 45 Tokens/s.

Offen, nach dem Vulkan-Profil von Qwen3.8-27B (nach Teil 1 rund 22,4 ms je Token, davon `linear` 18,8 ms):
- **Q4_K bleibt hinter der Bandbreite:** Die Q4_K_M-Datei von ggml-org lädt 18,6 GB (viele
  Matrizen in Q8_0), die Q8_0-Datei 27,4 GB; beide brauchen für die Matrixprodukte gleich
  lange. Q8_0 erreicht so rund 1,46 TB/s (80 % der 1,8 TB/s), Q4_K_M nur rund 0,98 TB/s.
  Verdächtig: `linear8` holt je Warp und Schritt nur einen Q4_K-Block (144 Bytes), also wenige
  Ladevorgänge gleichzeitig; mehr Blöcke je Schritt oder je Invocation probieren.
- **Weitere Reduktionen über eine Zeile mit 128:** die Norm je Kopf (`qkrope`), `gatednorm`
  der DeltaNet-Layer, `gatherrows`/`scatteradd` - nur, wenn das Profil sie zeigt.
- **CUDA:** keine Zeitmessung je Kernel (Events um die Kernel des Graphen, oder ein Lauf ohne
  Graph); CUDA liegt bei beiden Formaten rund 2 ms je Token hinter Vulkan.

Übernahme nur mit Messung.

---

## B4 — Weitere Modelle: Phase D (Plan 2026-10-17)

Ziel: weitere Decoder-Familien so anbinden, dass jeder neue Baustein mehrere Modelle
freischaltet. Modelldaten aus den Modellkarten und `config.json` der Hersteller, vor jedem
Slice gegen die gepinnte Revision prüfen. Die Architektur bauen wir nach der Implementierung
in `transformers` (Apache 2.0) nach, nicht nach Hersteller-Code; das gilt auch für Kolibri,
dessen Lizenz Code und Architektur ausdrücklich ausnimmt.

Erledigt (siehe `CHANGELOG.md`): D1, D1b, D2, D3 (a-c, Decode), D4, D4b, D5 (gpt-oss-20b),
D6a-D6e2 (Gemma 4 auf CPU, CUDA und Vulkan, aus safetensors und GGUF), D7 (Gated DeltaNet,
Qwen3.5 und Qwen3.8).

### Kandidaten

| Modell | Lizenz | Aufbau | neue Bausteine | Stand |
|---|---|---|---|---|
| gpt-oss-20b (120b) | Apache 2.0 | 20b: MoE 32 Experten top-4, 24 Layer (120b: 128 Experten, 36 Layer), abwechselnd Fenster 128 und voll | MoE, Sliding Window, Attention-Sinks, YaRN, Bias, begrenzte SwiGLU, MXFP4, o200k-Tokenizer, Harmony-Chatformat | erledigt (D5) |
| Gemma 4 26B-A4B, 31B, 12B (E2B, E4B) | Apache 2.0 | 26B: 30 Layer, 5 lokal (Fenster 1024) : 1 global, 128 Experten top-8 + dichter Parallelzweig; global 2 KV-Köpfe mit Dimension 512; 31B dicht | Kopfdimension 512 auf der GPU, GELU-tanh, Logit-Softcap, teilweise RoPE, K=V, eigener Tokenizer; nur E2B/E4B zusätzlich Per-Layer Embeddings und geteilter KV | erledigt bis auf E2B/E4B (D6) |
| Qwen3.8-27B (`model_type` `qwen3_5`, wie Qwen3.5) | Apache 2.0 | 64 Layer = 16 × (3 Gated DeltaNet + 1 volle Attention mit Ausgangsgate) | Gated DeltaNet mit rekurrentem Zustand und Convolution-Puffer; Bildteil zunächst nicht | erledigt (D7) |
| Kolibri-1 (Aleph Alpha) | Apache 2.0, nur Gewichte und Konfiguration | MoE 78B / 3,46B aktiv, 50 Layer, 4 Fenster (513) : 1 voll, NoPE in den vollen Layern, 384 Experten top-6 mit Sigmoid-Router + 1 geteilter | FP8 E4M3 in 128×128-Blöcken, safetensors in Teilen, Router mit Bias-Korrektur; GGUF nur von Dritten (Q4_K_M 47,5 GB) | offen (D8) |
| Nemotron 3.5 Bolt | OpenMDW-1.1 | 74B-A9B, Mamba-2/Transformer-Hybrid, LatentMoE | Mamba-2, LatentMoE, IQ2_XXS (Experten) und NVFP4 (Mamba-Projektionen) | gestrichen (D9) |

**Reihenfolge (Nutzerentscheidung 2026-10-20):** D8 (Kolibri) erst nach einem Vergleich der
deutschen Antworten mit Gemma 4 und Qwen3.5 - fällt weg, wenn es dort nichts voraus hat. D9
(Nemotron 3.5 Bolt) gestrichen: vier neue Bausteine für ein Modell, eigene Lizenz.

**Testrechner:** RTX-Rechner (RTX 5090 mit 32 GB, 96 GB RAM; echte große Modelle), Mac M2 Pro
(16 GB RAM; Tiny-Modelle und kleine echte Modelle), Snapdragon X mit Adreno (nur Tiny- und
kleine Modelle).

Jeder Slice wie bisher: FPC zuerst, Gegenproben, Review (Claude-Finder + Codex), Upstream-
Revision gepinnt, Tiny-Modell aus `transformers` als Referenz, echtes Modell hinter
`LAI_REQUIRE_MODEL`.

**D8 — Kolibri-1** (Aufbau siehe Tabelle). FP8 E4M3 mit 128×128-Skalen (oder eine geprüfte
Umwandlung nach Q4_K/Q8_0 beim Laden; ein GGUF von Dritten, Q4_K_M 47,5 GB, nur nach Prüfung
gegen die Referenz), Sigmoid-Router mit Bias-Korrektur nur für die Auswahl, geteilter Experte,
safetensors in Teilen. Auf dem RTX-Rechner: Experten im RAM (CPU), Attention und dichte Teile
auf der GPU. Gate: Tiny-Kolibri mit Router-Bias und 513er-Fenster gegen `transformers`;
echtes Modell auf dem RTX-Rechner mit gemessenem Speicherbedarf.

**Gemma 4 E2B/E4B:** Einbettungen je Layer (Per-Layer Embeddings) und geteilter KV-Cache.
Gate wie D6: Tiny-Modelle aus transformers, dann die echten Modelle auf CPU und GPU.

**Gemma 4, kleinere Reste:** `suppress_tokens` des 12B (`generation_config.json`, im GGUF
`tokenizer.ggml.suppress_tokens`: `<image|>`, `<audio|>`) wendet der Client nicht an -
transformers verbietet sie bei `generate`. Q4_0 auf den GPUs: siehe B2.

**Offen aus den Reviews von D6e1 und D6e2 (Codex Runde 2, 2026-11-13):** gezielte Tests für
Randfälle, die kein heutiges Modell erzeugt - eine Scheibe von Gate/Up, die keine ganzen
Wörter ist (Rückfall auf einen Puffer je Experte), ein Modell nur aus DeltaNet-Layern
(Attention-Puffer von einem Wort je Zeile), `dfDenseMoe` ohne Sandwich-Norms, ein
NoRope-Layer mit eigener Kopfbreite (`rot` 0) und ein reiner Experten-Block mit
Sandwich-Norms (keine `FfnNorm`, wie `MoeBlock`), auf CUDA und Vulkan. Nächster Schritt:
Tiny-Konfigurationen dafür in `make_reference_d6*.py`, wenn eine solche Familie kommt.

**Nur bei Bedarf (aus erledigten Slices):**
- *D3c:* der Prefill von MoE ohne Host-Wissen (Gruppen über die Höchstzahl oder indirekter
  Dispatch) - nur wenn ein Prefill-Profil zeigt, dass die Wartepunkte zählen; Qwen3-30B-A3B
  heute 383 (CUDA) / 490 (Vulkan) Tokens/s.
- *D4b/D5d2:* der schmale Cache auf dem Gerät (die Geräte halten heute alle Zeilen; bei
  gpt-oss-20b mit der Hälfte der Layer im Fenster 128 spart er bei langem Kontext rund die
  Hälfte des KV-Speichers).
- *D4b:* SmolLM3 als Chat-Familie (Template mit `/think` und `/no_think`, Parser) und aus
  GGUF (llama.cpp legt die NoPE-Layer fest, das wäre gegen die Referenz zu prüfen).
- *D5b2:* ein fusionierter MXFP4-Kernel (Skalarprodukt ohne den Umweg über Q8_0-Blöcke), nur
  wenn eine Messung das Entpacken noch als Bremse zeigt.
- *D7b:* ein schnelleres `exp` für silu - nur mit Messung und Toleranz gegen die Referenz.

**D9 — Mamba-2, LatentMoE, IQ2_XXS, NVFP4 (Nemotron 3.5 Bolt, OpenMDW-1.1):** gestrichen
(siehe oben). Nur bei konkretem Bedarf; NVIDIA nennt rund 22 GB Gewichte, 23,8 GB mit
Laufzeitzustand und Drafter, ohne Bildteil.

**Außerhalb von D:** Reranker (BGE-Reranker), Qwen3-Embedding, BGE-M3 Sparse-/ColBERT-Ausgabe,
Llama 3, Mistral: je ein eigener Eintrag, sobald gebraucht. Llama/Mistral brauchen vor allem
den llama-bpe- bzw. SentencePiece-Tokenizer. Bild- und Videoeingang (Qwen3.8, Gemma 4, Bolt)
ist nicht Teil von D.

---

## B2 — Q4_0: GPU-Kernel und fusionierter CPU-Kernel

Q4_0 läuft auf der CPU (seit 2026-10-17 entpackt nach Q8_0, Decode auf ARM und x64 mindestens
so schnell wie Q8_0), aber **nicht auf den GPUs**: CUDA und Vulkan haben keinen Q4_0-Kernel
(„no kernel for a Q4_0 matrix"). Die QAT-Dateien von Gemma 4 sind Q4_0; das 12B Q4_0 läuft
deshalb nur auf der CPU.

*Vorhaben GPU:* Q4_0 als `dtype` in den Linear-Kerneln von CUDA (`linear8`, `gemm8`,
`linear8ids`) und Vulkan (`linear8.comp`, `gemm8.comp`, `matvec8.comp`), wie Q5_0 in B16.
*Gate:* Linear gegen die Referenz (eine Zeile, mehrere Zeilen, Experten), Gemma 4 12B Q4_0 auf
CUDA und Vulkan gegen die CPU, Gegenproben (Nibble-Reihenfolge, −8 vergessen).

*Nur bei Bedarf (CPU):* ein fusionierter Q4_0×Q8_0-Kernel wie ggml (`ggml_vec_dot_q4_0_q8_0`,
MIT; NEON: Nibbles maskieren, −8, SDOT; AVX2: `maddubs` mit Vorzeichentrick), ARM-Repacking
(block_q4_0 verschränkt) oder ein AVX2-`InferQ4Unpack` (höchstens etwa 1 ms je Token) - nur,
wenn größere Modelle es verlangen.

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

---

## B16 — Q5_1 und Feinschliff an Q5_0 (Q5_0 erledigt 2026-10-19)

Nur bei Bedarf:

- **Q5_1** (Block 24: zusätzlich FP16-Minimum): llama.cpp nimmt es als Ersatz für Q5_K -
  eine Q5_K_M-Datei braucht also zuerst Q5_K. Auf der CPU nicht verlustfrei nach Q8_0
  auszupacken (Wert d · q + m): der Summand m · Summe(x) je Block käme dazu.
- **`matvec8.comp` für Q5_0**: Vulkan dekodiert Q5_0 mit `linear8`, 8 % langsamer als Q8_0
  mit `matvec8`.
- **Auspacken mit NEON/AVX2** wie `InferQ4Unpack`: Prefill auf der CPU 4 % langsamer als
  Q8_0 (891 / 931 Tokens/s).

---

## Weitere Punkte nur bei Bedarf

- **Metal-Backend (aus B8):** Unter MoltenVK ist die GPU des M2 Pro bei 200 Tokens im Prefill
  und Decode schneller als die CPU, die Kernel waren der Engpass, nicht MoltenVK. Ein natives
  Metal-Backend nur, wenn MoltenVK selbst bremst (Roadmap „Apple-GPU" in
  `native-inference.md`).
- **Adreno-Tempo (aus B11):** Qwen3-0.6B Q8_0 bei 200 Tokens auf dem Snapdragon X: Adreno
  Prefill 368 und Decode 30 Tokens/s, die CPU (8 Threads, NEON-Dotprod) 639 und 68. Die
  Shader sind nicht auf die Adreno abgestimmt (`matvec8` dort aus); kein Auftrag.
- **gpt-oss als Stapel (aus dem Review von B18, Codex Runde 2):** Der Loader von gpt-oss
  vergleicht die verstandenen Tensoren mit allen Tensoren der Datei, nicht mit denen unter dem
  Präfix wie Qwen3 und Gemma 4. Als Stapel eines größeren Checkpoints (`CreateStack`) lädt
  gpt-oss deshalb nicht, schon mit einem Kopf. Erst angehen, wenn ein Checkpoint gpt-oss als
  Stapel enthält.
- Weitere Ideen ohne Termin: [ideas.md](ideas.md).

---

## Nicht mehr verfolgen

- SQLite-Extensions `vec0` / `lembed`: fliegen langfristig raus (nativer Store und native
  Engine ersetzen sie) — kein Ausbau mehr.
