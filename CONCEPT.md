# LandrixAI — Konzept (`mormot.ai.*`)

> **Zweck dieses Dokuments.** Architektur-Leitplanke für das LandrixAI-Paket:
> *was* gehört in welche Schicht und *warum*. Single Source of Truth für die
> Schichtgrenze zwischen **generischem mORMot-Code**, **LandrixAI-Paket** und
> **Landrix-Backend** — sowie für den anstehenden **Merge** mit
> [flydevs `mormot.ai`](https://github.com/flydev-fr/mormot.ai) (§4), die Wahl des
> **Inferenz-Backends** (§5) und den **MCP-Sprung auf 2026-07-28** (§6).
>
> Die ursprüngliche Brainstorming-Fassung (ChatGPT-Dialog) ist nach
> [docs/Feature-LandrixAI-Agent.md](../../../docs/Feature-LandrixAI-Agent.md)
> verdichtet — dort steht der Agent als Backend-Feature.

## 1. Was LandrixAI ist (und was nicht)

LandrixAI ist eine **mORMot-basierte AI-/LLM-Toolbox** im Namespace `mormot.ai.*`,
**FPC-only** (Linux/Windows, x64/aarch64), veröffentlicht als **eigenständige
Extension in einem eigenen Repository**.

- **Eigene Extension statt mORMot-PR** (Entscheidung 2026-07-29). Ein Pull Request
  an Synopse mORMot ist **nicht** mehr das Ziel. Grund: Ein Upstream-PR würde uns auf
  mORMots **Dual-Compiler-Anspruch (FPC + Delphi)** festlegen — und damit die
  stärkste verfügbare Inferenz-Option ausschließen, denn **neural-api ist FPC-only**
  (§5). Als eigenständige Extension können wir sie **voll** nutzen. mORMot2 bleibt
  die **Grundlage** (Basis-Units, HTTP, JSON, SQLite, Crypto) — wir liefern nur
  nicht dorthin ab.
- **FPC-only, bewusst.** Kein Delphi-Zweig, keine `{$ifdef}`-Doppelpflege. Deckt sich
  mit der plattformweiten Landrix-Entscheidung (FPC ist einziges Lauf-/Build-Ziel).
- **Die eiserne Regel bleibt**: **kein Landrix-Domänenwissen im `mormot.ai.*`-Code.**
  Sie begründet sich jetzt aus der **Repo-Grenze** (die Extension muss ohne Landrix
  brauchbar sein), nicht mehr aus der Upstream-Fähigkeit.
- **Es ist nicht „OpenClaw in Pascal".** Wir bauen einen schlanken, sicheren
  **Agent-Kernel** + Provider-neutrale Naht (LLM/Embeddings/VectorStore/RAG/MCP).
  Alles Landrix-Spezifische lebt im Backend (Schicht B unten).

> **Namespace `mormot.ai.*` bleibt** — trotz fehlender Upstream-Absicht **und trotz
> der Kollision mit flydev** (der denselben Namespace nutzt): er ist in der
> mORMot-Extension-Szene für AI etabliert und steht überall in unserem Code. Da wir
> von flydev **nur Code übernehmen** und kein gemeinsames Repo betreiben (§4), ist die
> Kollision folgenlos. **Lizenz bleibt MPL/GPL/LGPL**
> (wie mORMot): die adoptierten MCP-Units verlangen es, und es ist mit MIT
> (flydev-pgvector) wie mit LGPL-2.1+Linking-Exception (neural-api) verträglich.

## 2. Schichtenmodell (die Kernregel)

**Zwei Schichten** — die Grenze ist jetzt die **Repo-Grenze**. (Früher waren es drei:
„M — mORMot-Upstream-Kandidat" und „A — Inkubation" unterschieden sich nur durch die
Frage *„reif genug für einen Synopse-PR?"*. Ohne Upstream-Ziel (§1) fällt diese
Unterscheidung weg — beide wandern in **dasselbe** Extension-Repo. **A** und **B**
bleiben als Bezeichner; „Schicht M" gibt es nicht mehr.)

> **Nennt die Unit eine konkrete Landrix-Tabelle, einen Guard, eine Rolle oder
> ein Fach-Konzept?** → Schicht B (Backend). **Sonst** → Schicht A (Extension).

| Schicht | Namespace | Repo | Inhalt | Landrix-abhängig? |
|---|---|---|---|---|
| **A — Extension** | `mormot.ai.*`, `mormot.db.*` | **eigenes Repo** (§7), FPC-only | Alles Generische: LLM-Wire-Treiber, Embeddings, VectorStore + pgvector-Binding, RAG-Engine, MCP, SSE, Structured Output, Agent-Loop, Skill-/Memory-/Session-**Interfaces**, CLI | **nein** |
| **B — Backend** | `landrix.server.ai.*` | landrix-platform | Landrix-Bindungen: Memory/Sessions als `TOrm` in `TLandrixDatabase`, guard-/scope-geprüfte Tools, Audit-Anbindung, Domänen-Skills, REST-Endpunkte | **ja** |

**Konsequenz für Interfaces vs. Implementierungen — „Interface tief, Bindung hoch":**

- **Interfaces** (`ILlmClient`, `IEmbedder`, `IVectorStore`, künftig `IMemoryStore`,
  `ISessionStore`, `IToolbox`) werden in **A** definiert.
- **Landrix-Implementierungen** (ORM-Memory, guard-geprüfte Tools) leben in **B**
  und werden zur Laufzeit injiziert. So bleibt die Extension **ohne Landrix
  brauchbar** (Voraussetzung für ein eigenes Repo), und das Backend erbt nur die
  Naht, nicht die Engine.

## 3. Persistenz- & Vektor-Abstraktion (sqlite-vec ↔ pgvector)

**Status heute ✓ (Interface/Impl-Split umgesetzt):** Die Abstraktion existiert nicht
nur, sie ist jetzt auch **sauber getrennt**. `IVectorStore` und `IEmbedder` liegen in
reinen Interface-Units ohne Backend-Abhängigkeiten; die Engine (`TLlmRag`,
`TRagSearchTool`) hängt **nur** an ihnen und zieht damit **keine** statische SQLite
mehr. Ein zweites Backend (pgvector) ist eine **neue Implementierung**, kein Umbau.

**Vorher das Problem (behoben):** Interface *und* sqlite-vec-Implementierung steckten
in **einer** Unit, die `mormot.db.raw.sqlite3.static` zog — wer pgvector (oder gar
kein SQLite) wollte, schleppte die statische SQLite-Abhängigkeit mit.

**Ist-Struktur (umgesetzt):**

```
mormot.ai.vectorstore           ✓ NUR IVectorStore + TRagHit + Vektor-Helfer (SQLite-frei)
mormot.ai.vectorstore.sqlitevec ✓ TVec0Store (sqlite-vec) + Extension-Loader [static SQLite hier isoliert]
mormot.ai.vectorstore.pgvector  ○ TPgVectorStore (IVectorStore) auf mormot.db.pgvector.* (Merge §4)
mormot.ai.embeddings            ✓ NUR IEmbedder (zieht nicht mehr den LLM-Client)
mormot.ai.embed.provider        ✓ TProviderEmbedder (OpenAI-Wire)
mormot.ai.embed.ollama          ○ TOllamaEmbedder (aus Merge §4)
mormot.ai.embed.lembed          ✓ TLembedEmbedder (sqlite-lembed, lokal; eigene :memory:-Connection) + SharedLembedEmbedder()
```
(✓ = gebaut, Demos grün · ○ = offen, kommt mit dem Merge §4)

**Embedder = Prozess-Service, nicht pro DB.** Embedding-Erzeugung (`text → vector`)
ist orthogonal zur Vektor-**Speicherung** (`IVectorStore`, pro DB). Ein lokales
GGUF-Modell (lembed) wird deshalb **einmal pro Prozess** geladen und von *N* Stores/
DBs geteilt — **nie** an eine Store-Connection gekoppelt (sonst lädt jede DB ihr
eigenes Modell in den RAM). `TLembedEmbedder` besitzt dazu eine **eigene, dedizierte
`:memory:`-Connection** (hostet nur lembed0 + das Modell); `SharedLembedEmbedder()`
gibt den **prozessweiten Singleton je logischem Modellnamen** zurück. Der
lembed-Context ist nicht re-entrant → Embed/EmbedBatch serialisieren unter einem
`TOSLightLock`; wenn Nebenläufigkeit dominiert, skaliert man über einen zweiten
`IEmbedder`-Backend (Ollama, Merge §4), nicht über mehrere Modell-Loads.

Regel (gilt jetzt durchgängig): **Die RAG-/Agent-Engine kennt nur die Interfaces.**
Welches Backend (sqlite-vec im Single-Binary-Edge-Fall, pgvector im
Server-/Multi-Tenant-Fall) zum Einsatz kommt, entscheidet die Komposition in
Schicht A/B per DI — nie ein `{$ifdef}` in der Engine.

`mormot.db.pgvector.*` (Binding/ORM/Core) bleibt bewusst unter `mormot.db.*` statt
unter `mormot.ai.*`: es ist eine **pure DB-Schicht ohne AI-Bezug** und für sich
allein nutzbar. Die Namensgebung folgt mORMots `mormot.db.*`-Familie — auch ohne
Upstream-Absicht (§1) ist das die richtige Einordnung im eigenen Repo.

## 4. Merge mit flydevs `mormot.ai` (seit 2026-07-28 öffentlich)

Das „parallele Repo" ist identifiziert und **öffentlich**:
[flydev-fr/mormot.ai](https://github.com/flydev-fr/mormot.ai), Doku
[mormot-ai.fsb.dev](https://mormot-ai.fsb.dev/). Die Unit-Liste deckt sich exakt mit
dem, was hier bisher als „Parallel-Repo" geführt wurde.

**Ziel: Vollintegration** („das große Ganze") — nicht nur Rosinen. Es sind aber
**zwei getrennte Libraries mit unterschiedlicher Lizenz**, und genau daran hängt der
Zeitplan:

| Library | Inhalt | Lizenz | Status für uns |
|---|---|---|---|
| `mormot2-pgvector` | `mormot.ai.chunk`, `mormot.ai.embed.{base,ollama}`, `mormot.ai.llm.{base,ollama}`, `mormot.ai.rag.{history,service,stream}`, `mormot.db.pgvector.{binding,orm,base}` | **MIT** | ✅ sofort mergefähig |
| `mormot-ai-sdk` | LiteLLM-Port: `mormot.ai.litellm.{client,router,cost,proxy,streaming,handler,types,callbacks}` + `provider.{openai,anthropic,gemini}` (Router/Fallback, Cost-Tracking) | **keine** (Repo-Metadaten `license: null`, keine LICENSE im SDK-Teil) | ⛔ *all rights reserved* — kein Nutzungs-/Ableitungsrecht; **Lizenz ist angefragt** |

> **Rechtliche Leitplanke:** Ohne Lizenz ist Code **nicht** frei verwendbar — auch
> nicht bei öffentlichem Repo. Der SDK-Teil bleibt **gesperrt**, bis flydev eine
> Lizenz setzt. Der MIT-Teil ist mit Attribution mergefähig (MIT ist mit unserem
> MPL/GPL/LGPL-Ziel verträglich; Copyright-Notice erhalten, siehe [NOTICE](NOTICE)).

**Namens-/Konzept-Kollisionen** (jetzt schärfer als zuvor — teils **gleicher
Unit-Name bei anderem Inhalt**):

| Thema | Hier (LandrixAI) | flydev | Entscheidung |
|---|---|---|---|
| Embeddings | `mormot.ai.embeddings` (Interface) | `mormot.ai.embed.base` + `.ollama` | Interface in `mormot.ai.embeddings`, Treiber als `mormot.ai.embed.*` |
| SSE | `mormot.ai.llm.sse` | `mormot.ai.http.sse` | **`mormot.ai.http.sse`** — SSE ist generisch HTTP |
| LLM-Naht | `mormot.ai.llm` clean-room (OpenAI-Wire = Lingua franca) + nativer `.anthropic` | **zwei** konkurrierende: `mormot.ai.llm.base`/`.ollama` (pgvector-Lib) **und** der LiteLLM-Port (SDK) | **Unsere Naht bleibt der Kern** (clean-room, lizenzrein, Anthropic-nativ, MCP-Bridge, Structured). flydevs Ollama-Treiber als weiteren `ILlmClient` einhängen; LiteLLM-Router/Cost-Tracking erst **nach** Lizenzklärung als optionale Zusatzschicht |
| RAG | monolithisch `mormot.ai.rag` | `rag` + `.service` + `.stream` + `.history` | **Zerlegung übernehmen** |
| Chunking | `ChunkText` in `mormot.ai.rag` | eigenes `mormot.ai.chunk` | **Extrahieren** nach `mormot.ai.chunk` |
| Vektor-Store | sqlite-vec (`TVec0Store`) | `mormot.db.pgvector.*` | **Beide** als `IVectorStore`-Backends (§3) |
| Hybrid-Suche | — | cosine + `tsvector` (Postgres) | Konzeptionell übernehmen; unser SQLite-Pendant ist FTS5 + vec (RRF, s. Feature-Adressverwaltung) |
| MCP | `mormot.ai.mcp.*` (Server+Client) | — | bleibt **unser** Beitrag (§6) |

**Merge-Reihenfolge:** (1) MIT-Teil integrieren (pgvector-Backend, Ollama-Embedder,
RAG-Zerlegung, chunk) — blockiert nichts; (2) SDK-Teil erst nach Lizenzzusage.

### Modus: Code-Übernahme, kein gemeinsames Repo (entschieden)

**Wir übernehmen nur den Code** — kein Fork, kein geteiltes Repository, keine
Upstream-PRs. Unser **Namespace `mormot.ai.*` bleibt unverändert**, auch dort, wo er
mit flydevs kollidiert; adoptierte Units werden in unseren Baum eingepasst/umbenannt.
Das hält uns unabhängig (FPC-only, neural-api, MCP 2026-07-28) — Punkte, die flydev
nicht mitträgt.

**Preis dieser Freiheit: manuelles Nachziehen.** Damit spätere flydev-Änderungen
einarbeitbar bleiben, wird der **übernommene Commit-Stand gepinnt** — in
[UPSTREAM_BASE](UPSTREAM_BASE), im selben Muster wie schon für die adoptierten
MCP-Units. Baseline steht dort bereits (`9ab26e88…`, `main`, 2026-07-28).

```bash
# Update-Runde: was hat sich seit unserem Stand getan?
gh api repos/flydev-fr/mormot.ai/commits/HEAD --jq .sha
git -C <klon> diff <gepinnter-sha>..<neuer-sha> -- mormot2-pgvector
# Relevantes portieren, dann Commit/Date in UPSTREAM_BASE im selben Commit bumpen.
```

**Regel:** Wer aus flydev portiert, **bumpt UPSTREAM_BASE im selben Commit** — sonst
ist die Diff-Basis verloren und die nächste Runde muss raten.

## 5. Inferenz-Backends: lembed ↔ Ollama ↔ neural-api

Die zweite Backend-Achse neben der Persistenz (§3): **wo läuft das Modell?** Auch
hier gilt „Interface tief, Bindung hoch" — die Engine kennt nur `IEmbedder` /
`ILlmClient`, das Backend ist eine Kompositions-Entscheidung.

| Backend | Wie | Fremdabhängigkeit | Status |
|---|---|---|---|
| **lembed** (heute) | GGUF-Modell in SQLite via `lembed0` | `lembed0` + `libllama`/`ggml` **je Arch** (`.so`/`.dll`), `load_extension` | ✓ gebaut, prozessweiter Singleton |
| **Provider** | OpenAI-Wire `/embeddings` | Netz + API-Key | ✓ gebaut |
| **Ollama** | HTTP an lokalen Ollama-Server | **externer Prozess** | ○ kommt mit §4 |
| **neural-api** | **native Pascal-Inferenz im Prozess** | **keine** (statisch gelinkt) | ○ evaluieren |

### neural-api (CAI NEURAL API) — die interessante Option

[joaopauloschuler/neural-api](https://github.com/joaopauloschuler/neural-api) —
LGPL-2.1 **mit Linking-Exception**, 432★, sehr aktiv (Push 2026-07-28), AVX/AVX2/
AVX512 + OpenCL. Kann laut README **echte LLM-Inferenz nativ in Pascal**: Qwen2.5
(0.5B–32B), Qwen3, Llama/TinyLlama/SmolLM2, Mistral 7B, Phi-3-mini, OLMoE (MoE),
RWKV, xLSTM (`ChatTerminal`); dazu Retrieval-Beispiele für die **E5/BGE/GTE**-Familie
(`EmbeddingSearch`, `SemanticSearch`, `ColBERTSearch`, `DebertaReranker`).

**Der Gewinn, wenn es trägt** — genau die Linie „Single Binary, kein externer
Dienst": **alle nativen Fremdbibliotheken entfallen**. Kein `lembed0`/`vec0`-`.so`,
kein `libllama`/`ggml`, kein `load_extension`, **kein Arch-Bundle-Download + SHA256**
im Docker-Entrypoint, kein Ollama-Server. Ein statisch gelinktes FPC-Binary macht
Embeddings **und** Chat.

**Die Haken — ehrlich, vor jeder Entscheidung zu klären:**

1. **Kein GGUF.** Eigenes `.nn`-Format; Import aus HuggingFace läuft über eine
   **Python-Toolchain** (torch/transformers/safetensors: `slice_llama.py`,
   `make_tiny_llama.py`). Relativierung: das ist ein **Bereitstellungs-Schritt**, kein
   Laufzeit-Schritt — einmal konvertieren, `.nn` ausliefern (wie heute `.gguf`). **Kein
   Python auf dem Server.**
2. ~~**FPC/Lazarus only** (master; Delphi nur v2.0.0)~~ — **erledigt**: seit der
   Entscheidung „eigene Extension statt mORMot-PR, FPC-only" (§1) ist das **kein
   Hindernis mehr, sondern der Grund dafür**. neural-api darf **Kern-Abhängigkeit**
   werden.
3. **Embedding-Tauglichkeit unverifiziert.** Das `EmbeddingSearch`-Beispiel fährt ein
   Pico-Test-Fixture (`tiny_e5`, 2 Layer, hidden 8), kein Produktionsmodell. Ob
   **bge-m3** (XLM-RoBERTa, 1024 dim) sauber konvertiert und rechnet, ist offen.
4. **Performance unbelegt** gegenüber llama.cpp (hochoptimierte C++-Kernels).

**Entscheidung: neural-api ist das strategische Ziel-Backend** (§1) — eingebunden
über dieselbe Naht: neue Units `mormot.ai.embed.neuralapi` / `mormot.ai.llm.neuralapi`
hinter `IEmbedder`/`ILlmClient`. Die Naht bleibt **trotzdem** bestehen — nicht mehr
wegen Delphi, sondern weil Provider-/Ollama-/lembed-Backends weiter sinnvoll sind
(Cloud-Modelle, bestehende Ollama-Deployments, Fallback).

**Reihenfolge (Risiko zuerst):** ein **Spike** mit harten Kriterien, bevor umgestellt
wird: (a) ein echtes E5/BGE-Modell konvertieren, (b) Embedding-**Qualität** gegen
lembed vergleichen (gleiche Testsätze, Cosine-Ranking), (c) **Durchsatz** messen,
(d) RAM-Bedarf. Fällt der Spike gut aus, wird neural-api der **Default**;
lembed/Ollama/Provider bleiben Fallback — umgeschaltet per Config
(`AI_EMBED_PROVIDER`), nie per `{$ifdef}`.

## 6. MCP-Protokoll: Sprung auf 2026-07-28 (Greenfield, kein Legacy)

Die Spec **2026-07-28** ist die größte Änderung seit MCP-Launch
([Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)).
**Projektentscheidung: wir sprechen ausschließlich 2026-07-28.** Was die neue Spec
entfernt, wird bei uns **gelöscht** — keine Multi-Version-Negotiation, keine
Legacy-Pfade, kein Deprecation-Ballast (Greenfield-Regel des Projekts).

**RAUS (ersatzlos löschen):**
- `initialize` / `notifications/initialized` **und die gesamte Versions-Negotiation**
  (2024-11-05 / 2025-03-26 / 2025-06-18 / 2025-11-25) — Protokoll ist stateless.
- **Protokoll-Sessions + `Mcp-Session-Id`** im Streamable HTTP → damit auch unsere
  Session-Registry, Ablauf-Logik und das FSafe-Re-Resolve.
- **SSE-Resumability**: `Last-Event-ID`, SSE-Event-IDs, `NextSessionEventId`.
- **HTTP-GET-Endpunkt** + `resources/subscribe`/`unsubscribe`.
- `ping`, `logging/setLevel`, `notifications/roots/list_changed`.
- **Legacy-HTTP+SSE-Transport** (`TMcpSseTransport`) — jetzt offiziell *Deprecated*;
  die offene Frage „härten oder entfernen" ist damit **entschieden: entfernen**.
- **Roots / Sampling / Logging** (deprecated) — gar nicht erst bauen.

**REIN:**
- **`server/discover`** — Pflicht-RPC (unterstützte Versionen, Capabilities, Identität).
- **`_meta` pro Request**: `io.modelcontextprotocol/protocolVersion`,
  `clientCapabilities`, `clientInfo`, `logLevel`; `serverInfo` in jedem Result.
- **`subscriptions/listen`** — ein langlebiger POST-Response-Stream, Opt-in je Typ
  (`toolsListChanged`, …), Notifications getaggt mit `subscriptionId`.
  Request-scoped (`notifications/progress`, `…/message`) bleiben auf dem
  Response-Stream ihres Requests.
- **`resultType`** auf allen Results (`complete` | `input_required`).
- **MRTR** statt server-initiierter Requests: `InputRequiredResult` mit
  `inputRequests`; Client antwortet per Retry mit `inputResponses` (+ `requestState`).
- **`CacheableResult`**: `ttlMs` + `cacheScope` auf `tools/list`, `prompts/list`,
  `resources/list`, `resources/read`, `resources/templates/list`; Tools in
  **deterministischer** Reihenfolge.
- **Header** `Mcp-Method` / `Mcp-Name` auf POST; `x-mcp-header` für Tool-Parameter.
- **Extensions-Framework** (`extensions` in Client/ServerCapabilities); **Tasks** nur
  noch als Extension `io.modelcontextprotocol/tasks` (Polling `tasks/get`,
  `tasks/update`, kein `tasks/list`).
- **Error-Codes**: `-32020` HeaderMismatch, `-32021` MissingRequiredClientCapability,
  `-32022` UnsupportedProtocolVersion; resource-not-found `-32002` → **`-32602`**.
- **Auth-Härtung**: `iss` prüfen (RFC 9207), `application_type` bei DCR, Credentials
  an den Issuer gebunden; **Client ID Metadata Documents** statt DCR.

> **Cross-Call-State**: „stateless" heißt nicht zustandslos für Tools — wer Zustand
> über Aufrufe braucht, nutzt **server-minted Handles als normale Tool-Argumente**
> (SEP-2567). Für unsere Agent-/RAG-Tools ist das der vorgesehene Weg.

**Aufwandseinschätzung: netto Vereinfachung.** Der größte Teil des Umbaus ist
**Löschen** — und er erledigt zwei offene Roadmap-Punkte gleich mit (Parallel-
Stresstest der Session-Logik, Härtung des Legacy-SSE-Transports), weil die
zugehörige Mechanik im Zielprotokoll nicht mehr existiert. **Bestehen bleibt** der
offene Punkt **Auth-Resolver** — durch die Auth-Härtung der neuen Spec eher größer.

## 7. Auslösung in ein eigenes Repository — **vollzogen (2026-07-29)**

Aus §1 folgt: der `mormot.ai.*`-Code verlässt das Landrix-Monorepo. Landrix ist
seither **Konsument** der Extension — genau wie schon bei mORMot2 selbst.

> **Stand:** Repo [`landrix/mormot2-ai`](https://github.com/landrix/mormot2-ai)
> (public), erzeugt per `git subtree split --prefix=shared/delphi/landrixai`
> **mit vollständiger Historie** (33 Commits). Landrix bindet es als Submodul unter
> `shared/delphi/landrixai` ein — der Pfad bleibt gleich, deshalb ändern sich die
> `-Fu`-Build-Pfade der Konsumenten **nicht**. `codenav/` wurde vorher aus dem
> Prefix nach `shared/delphi/codenav/` gehoben (s. u.).
>
> Der Repo-Name weicht bewusst von flydevs `mormot.ai` ab; der **Unit**-Namespace
> `mormot.ai.*` bleibt (§4, Modus „Code-Übernahme").

### Was geht, was bleibt

| Geht ins Extension-Repo | Bleibt in landrix-platform |
|---|---|
| `src/` (alle `mormot.ai.*`-Units) | `backend/src/landrix.server.ai.*` (Schicht B) |
| `tests/`, `demos/`, `scripts/` | `docs/Feature-LandrixAI-Agent.md` (Backend-Feature) |
| `CONCEPT.md`, `DESIGN.md`, `README.md`, `LICENSE`, `NOTICE`, `UPSTREAM_BASE` | `AI_EMBED_*`-Config, Docker-Verdrahtung, Modell-/Extension-Bereitstellung |
| `vendor/`-**Struktur** (Binaries bleiben gitignored) | `deploy/docker-native/*` (Entrypoint, `pack-ai-dist.sh`) |
| — | **`codenav/`** (Landrix-Werkzeug, s. u.) |

**`codenav/` bleibt außen vor** (entschieden): Es ist ein **Landrix-Entwickler-
Werkzeug**, kein Bestandteil der Extension — es wandert **nicht** mit. Konsequenz:
Es nutzt die `mormot.ai.mcp.*`-Units dann über das **Submodul**, wie jeder andere
Konsument auch (Build-Pfade entsprechend anpassen).

### Einbindung: Submodul (etabliertes Muster)

Landrix bindet die Extension als **git submodule** ein — dasselbe Muster wie
`shared/delphi/libs/_git_Synopse2`. Kein Paketmanager, kein Vendoring-Copy.

### Extraktion **mit** Historie

Die Historie ist wertvoll (Review-Härtungen, Design-Entscheidungen) — **nicht**
per Copy-Paste in ein leeres Repo:

So ist es gelaufen (zur Reproduktion bei weiteren Auslösungen):

```bash
# 1. codenav VOR dem Split aus dem Prefix heben - sonst landet das
#    Landrix-Werkzeug in der Extension und kollidiert mit dem Submodul-Mount
git mv shared/delphi/landrixai/codenav shared/delphi/codenav

# 2. Historie herausschneiden und in das leere Repo pushen
git subtree split --prefix=shared/delphi/landrixai -b split/mormot2-ai
git push https://github.com/landrix/mormot2-ai.git split/mormot2-ai:main

# 3. im Monorepo den Pfad durch das Submodul ersetzen
git rm -r shared/delphi/landrixai
git submodule add https://github.com/landrix/mormot2-ai.git shared/delphi/landrixai
```

**Fallstrick beim Ersetzen**: Unter `shared/delphi/landrixai` liegen ~2 GB
**gitignorierte** lokale Daten (`vendor/models` mit GGUF, `vendor/sqlite-ext`,
`_eval/`, `bin/`, `_cleanroom/`). `git rm -r` fasst sie nicht an, aber
`git submodule add` verweigert ein nicht-leeres Zielverzeichnis — also die
ignorierten Ordner wegschieben, Submodul klonen, zurückschieben. Neu heruntergeladen
werden müssten sonst mehrere GB.

### Nachzuziehen (Checkliste — hier hängt der Build dran)

1. **Build-Pfade** `-Fu` auf `shared/delphi/landrixai/src` in **drei** Skripten:
   `backend/scripts/run-fpc-tests.sh`, `build-server-linux.sh`, `build-server-win.ps1`.
2. **Docker/CI — der kritische Punkt**: `deploy/docker-native/Dockerfile` kopiert
   `shared/delphi/landrixai/src`. Als Submodul muss der CI-Checkout **`submodules:
   recursive`** setzen, sonst ist das Verzeichnis im Build-Kontext **leer** und der
   FPC-Build bricht ab. (Dasselbe Muster hat schon bei den Synopse2-Static-Libs
   zugeschlagen → `deploy/docker-native/static/`.)
3. **codenav-Indizierung** + CLAUDE.md-Pfade (`shared/delphi/{landrixai,client,dto}`).
4. `.gitignore`-Einträge (`vendor/models`, `vendor/sqlite-ext`, `bin/`) wandern mit.
5. `deploy/docker-native/pack-ai-dist.sh` (greift auf `landrixai/vendor/…`).
6. Skills mit landrixai-Bezug (`landrix-mormot2`).

### Zeitpunkt: **früh, vor den großen Umbauten**

Empfehlung: **jetzt splitten**, nicht nach Merge/MCP-Umbau. Begründung:
- Die Kopplung ans Backend ist **bereits dünn** (Schicht B nutzt nur `IEmbedder` +
  `GetSharedEmbedder`) — der Schnitt ist heute billig und wird mit jedem weiteren
  Feature teurer.
- Die drei großen anstehenden Arbeiten (flydev-Merge §4, MCP-Neubau §6,
  neural-api §5) finden **vollständig innerhalb** der Extension statt. Sie im
  Zielrepo zu machen erspart einen zweiten Umzug.
- Der flydev-Merge bedeutet ohnehin ein neues Repo-Layout (zwei Libraries
  zusammenführen) — das gehört nicht ins Landrix-Monorepo.

**Repo-Modell (entschieden):** **eigenes, unabhängiges Repo** — kein gemeinsames mit
flydev, kein Fork. Von dort wird **nur Code übernommen** (§4, Modus „Code-Übernahme"),
der Namespace `mormot.ai.*` bleibt trotz Kollision. Der Repo-**Name** ist damit frei
wählbar (`mormot.ai` ist bei flydev belegt) — z. B. `landrix/mormot-ai-ext`.

## 8. Ist-Stand-Inventar (Schicht A, gebaut)

Alles unten **gebaut + review-gehärtet + grün** (FPC 3.2.2 aarch64-linux): MCP-Suite
**202 Assertions**, LLM-Suite **272 Assertions**; Streaming/Tool-Loop/RAG/Vision auch
**live** verifiziert (OpenAI/Ollama/Anthropic). Aufbau-Historie: [DESIGN.md](DESIGN.md).

- **LLM-Client**: `mormot.ai.llm` (OpenAI-Wire = Lingua franca, deckt OpenAI/LiteLLM/
  Ollama) + nativer `mormot.ai.llm.anthropic` (Messages-API hinter derselben
  `ILlmClient`-Naht); Typen `mormot.ai.llm.types`; SSE-Streaming
  (`mormot.ai.llm.sse`, Basis + OpenAI-/Anthropic-Subklassen); Vision/Multimodal
  (`TLlmMessage.Images`, beide Wires).
- **Agent-Loop**: `TLlmAgent` + `ILlmToolbox` ([mormot.ai.agent.pas](src/mormot.ai.agent.pas)).
- **MCP**: Server+Client, stdio/HTTP/SSE/Streamable, RTTI-Tool-Schema
  (`mormot.ai.mcp.*`), MCP-Bridge in den Agenten (`mormot.ai.agent.mcp`).
  ⚠️ Spricht **2025-11-25** mit Versions-Negotiation — das ist der **Alt-Stand**;
  Ziel ist ausschließlich **2026-07-28** (§6), inkl. Löschen von Sessions/
  `initialize`/Legacy-SSE.
- **Structured Output**: `mormot.ai.llm.structured` (`ChatStructured` provider-agnostisch);
  OpenAI `response_format` ↔ Anthropic `output_config.format` (Adapter übersetzt + injiziert
  `additionalProperties:false`), beide live verifiziert.
- **Embeddings/RAG** (Interface/Impl getrennt, §3): Interfaces `IEmbedder`
  (`mormot.ai.embeddings`) + `IVectorStore` (`mormot.ai.vectorstore`, SQLite-frei);
  Impls `TProviderEmbedder` (`mormot.ai.embed.provider`), `TLembedEmbedder`
  (`mormot.ai.embed.lembed`), `TVec0Store` (`mormot.ai.vectorstore.sqlitevec`);
  Engine `TLlmRag` (`mormot.ai.rag`) + **agentic RAG** als `search_docs`-MCP-Tool
  (`mormot.ai.rag.tool`, dasselbe Tool extern wie in-process).
- **codenav**: MCP-Tools für Code-Navigation (`codenav/`).

## 9. Offene Punkte / Roadmap

Die Engine-Funktionsfläche (Phase D) steht, die **Repo-Auslösung** (§7) ist erledigt;
offen sind der **flydev-Merge** (§4), der **MCP-Sprung auf 2026-07-28** (§6) und die
**Evaluation von neural-api** (§5). Diese Liste ist die konsolidierte Roadmap —
DESIGN.md verweist hierher.

0. **Repo-Auslösung (§7)** ✓ **erledigt (2026-07-29)** — `landrix/mormot2-ai`, Historie
   erhalten, Landrix zieht per Submodul. Alle weiteren Punkte finden **hier** statt.

1. **Refactor §3** ✓ **umgesetzt** (Interface/Impl-Units getrennt: `vectorstore` +
   `vectorstore.sqlitevec`, `embeddings` + `embed.provider` + `embed.lembed`; Engine
   SQLite-frei, Demos grün). Offen bleibt nur das **neue** pgvector-Backend (Punkt 2).
2. **Merge §4** durchziehen — **Ziel Vollintegration**, gestaffelt nach Lizenz:
   - **(a) jetzt**: `mormot2-pgvector` (**MIT**) — pgvector als zweites
     `IVectorStore`-Backend `mormot.ai.vectorstore.pgvector` (die Naht steht),
     Ollama-Embedder als `mormot.ai.embed.ollama`, RAG-Zerlegung
     (`rag.service`/`.stream`/`.history`), Chunking → `mormot.ai.chunk`,
     SSE → `mormot.ai.http.sse`. Attribution/NOTICE mitführen.
   - **(b) blockiert**: `mormot-ai-sdk` (LiteLLM-Port) — **keine Lizenz**,
     Anfrage läuft. Erst nach Zusage anfassen; unsere clean-room-LLM-Naht bleibt
     bis dahin (und voraussichtlich dauerhaft) der Kern.
3. **Memory-/Session-Interfaces** (Schicht A) definieren — Implementierung im
   Backend (Schicht B), Details in
   [docs/Feature-LandrixAI-Agent.md](../../../docs/Feature-LandrixAI-Agent.md).
4. **x64-linux-`.so`** für sqlite-vec/lembed ✓ **vorhanden + verifiziert** — alle vier
   Arch-Builds da (aarch64/x86_64 × linux/win64); neues vec0 `v0.1.10-alpha.4` +
   lembed0 laufzeit-geprüft auf **beiden** Deploy-Arches: aarch64 nativ (rag-spike/
   chat/agent) und x86_64 gegen mORMots **statisches** SQLite im amd64-Container
   (`scripts/x64-ext-verify.sh`). Offen bleibt nur die Verdrahtung ins Docker-Image.
5. **Anthropic-Restfläche** ✓ **erledigt**: Structured Output über `output_config.format`
   (Adapter übersetzt das neutrale `ResponseFormat` und injiziert das von Anthropic
   geforderte `additionalProperties:false`) — live gegen `claude-opus-4-8` (typisierter
   Record extrahiert). Anthropic-**Streaming** live verifiziert (SSE-Parser inkl.
   message_start/delta-usage end-to-end). Beide Demos (`llm-structured`, `llm-chat`) jetzt
   provider-agnostisch via `LLM_PROVIDER`.
6. **MCP-Sprung auf 2026-07-28** (§6) — ersetzt den früheren Punkt
   „Transport-Produktionsreife". Der Umbau ist überwiegend **Löschen**; mehrere
   Alt-Befunde erledigen sich dadurch:
   - ~~Parallel-Stresstest der Streamable-Sessions~~ **entfällt** — Protokoll-Sessions
     (und damit Registry, Ablauf, FSafe-Re-Resolve, `NextSessionEventId`) existieren
     im Zielprotokoll nicht mehr.
   - ~~Legacy-SSE-Transport härten oder entfernen~~ → **entfernen** (offiziell
     *Deprecated*), Entscheidung getroffen.
   - **Auth: die Mechanik steht, der Resolver fehlt.** Der Server ist jetzt
     OAuth-2.1-**Resource-Server** (RFC-9728-Metadata, Token-Prüfung vor dem Dispatch,
     Audience-Bindung, 401/403-Challenges — Details in [DESIGN.md](DESIGN.md)). Die
     Naht ist `IMcpTokenVerifier`; **offen** ist deren Landrix-Implementierung im
     Backend (`backend/src/landrix.server.ai.*`): JWT-Prüfung gegen die bestehende
     Auth, Rollen aus `GetRolesForUser`, Scope→Permission-Abbildung.
     **Korrektur zur früheren Fassung**: `iss`/RFC 9207, Client ID Metadata Documents
     und Issuer-Bindung sind **Client-/Authorization-Server-Pflichten**, nicht die des
     MCP-Servers — sie stehen hier nicht mehr an.
   - **Gebaut** (Details in [DESIGN.md](DESIGN.md), Abschnitt „Protokoll"):
     `server/discover`, `_meta`-Transport der Protokollversion, `subscriptions/listen`,
     `resultType`, **MRTR**, `ttlMs`/`cacheScope`, neue Header + Error-Codes.
   - **Neu zu bauen**: `x-mcp-header`, Extensions-Framework, JSON Schema 2020-12 im
     `inputSchema`, OTel-`_meta`-Keys, deterministische `tools/list`-Reihenfolge,
     Prompts/Completion/Pagination (Details §6).
   - **Demo-Härtung** ✓ (bleibt gültig): Streamable-Demo bindet **Loopback**,
     `ask_claude` standardmäßig **deaktiviert** (`MCP_ENABLE_ASK_CLAUDE=1` als Opt-in).
7. **neural-api integrieren** (§5) — Units `mormot.ai.embed.neuralapi` /
   `mormot.ai.llm.neuralapi` hinter der bestehenden Naht; **Spike zuerst** (echtes
   E5/BGE-Modell konvertieren, Qualität gegen lembed, Durchsatz, RAM). Ziel: alle
   nativen Fremd-`.so` + der Docker-Bundle-Download entfallen. Seit §1 (FPC-only,
   eigene Extension) **darf** es Kern-Abhängigkeit werden.
8. **Tests**: TVec0Store hat einen **env-gated Real-vec0-Test** (Keyed-Ops
   `Upsert`/`Delete`/`Search`-Key, `test.llm.rag`, läuft bei gesetztem `SQLITE_EXT_DIR`,
   sonst Skip). Offen: TLembedEmbedder-Realtest + RAG-Atomar-Rollback (brauchen die
   lembed-Runtime, s. Punkt 4).

**In diesem Review-Pass bereits behoben** (Build + Tests grün: MCP 202, LLM 256
Assertions): JSON-RPC-Envelope-Validierung (`jsonrpc:"2.0"`, Params-Typ) + breiter
Exception→JSON-RPC-Error-Fang mit korrekten Codes (−32600/−32601/−32603);
fail-closed-Auth; Streamable-UAF via FSafe; SSE-Map thread-safe + Leak-Fix;
RAG-Ingestion **strikt** (exakte Vektoranzahl) **+ atomar** (`IVectorStore.AddBatch`,
eine Transaktion); `build-demo.sh` baut wieder **alle** Demos (vorher nur 5 MCP-Demos
trotz „ALL DEMOS OK"); Demo-Loopback + `ask_claude`-Gate.
