# LandrixAI — Konzept (`mormot.ai.*`)

> **Zweck dieses Dokuments.** Architektur-Leitplanke für das LandrixAI-Paket:
> *was* gehört in welche Schicht und *warum*. Single Source of Truth für die
> Schichtgrenze zwischen **generischem mORMot-Code**, **LandrixAI-Paket** und
> **Landrix-Backend** — sowie für den anstehenden **Merge** mit dem parallelen
> `mormot.ai.*`-Repo (Ollama/pgvector/RAG-Service).
>
> Die ursprüngliche Brainstorming-Fassung (ChatGPT-Dialog) ist nach
> [docs/Feature-LandrixAI-Agent.md](../../../docs/Feature-LandrixAI-Agent.md)
> verdichtet — dort steht der Agent als Backend-Feature.

## 1. Was LandrixAI ist (und was nicht)

LandrixAI ist eine **mORMot-native AI-/LLM-Toolbox** im Namespace `mormot.ai.*`,
dual für **FPC *und* Delphi**, cross-platform (Linux/Windows, x64/aarch64).

- **Es ist ein Inkubator.** Erklärtes Ziel (siehe [README](README.md)): die
  generischen Bausteine später an **Synopse mORMot** zu contributen. Deshalb gilt
  die eiserne Regel: **kein Landrix-Domänenwissen im `mormot.ai.*`-Code.**
- **Es ist nicht „OpenClaw in Pascal".** Wir bauen einen schlanken, sicheren
  **Agent-Kernel** + Provider-neutrale Naht (LLM/Embeddings/VectorStore/RAG/MCP).
  Alles Landrix-Spezifische lebt im Backend (Schicht B unten).

## 2. Schichtenmodell (die Kernregel)

Drei Schichten. Die Zuordnung entscheidet sich am **Namespace** und an einer
einzigen Frage:

> **Nennt die Unit eine konkrete Landrix-Tabelle, einen Guard, eine Rolle oder
> ein Fach-Konzept?** → Schicht B (Backend). **Sonst** → Schicht M oder A.

| Schicht | Namespace | Inhalt | Abhängig von Landrix? | Upstream-Ziel |
|---|---|---|---|---|
| **M — mORMot-Kandidat** | `mormot.db.*`, `mormot.ai.*` | DB-/AI-Primitive: pgvector-Binding, LLM-Wire-Treiber, Embeddings, VectorStore-Interface, RAG-Engine, MCP, SSE, Structured Output, Agent-Loop-Kernel | **nein** | **Synopse mORMot** (Pull Request) |
| **A — LandrixAI-Paket** | `mormot.ai.*` (Inkubation) | Komposition/Orchestrierung, die noch zu jung/landrix-getrieben für Upstream ist: Agent-Orchestrator, Skill-/Memory-/Session-**Interfaces**, RAG-Service, CLI | **nein** (nur „von uns getrieben") | wandert nach M, sobald stabil |
| **B — Backend** | `landrix.server.ai.*` | Landrix-Bindungen: Memory/Sessions als `TOrm` in `TLandrixDatabase`, guard-/scope-geprüfte Tools, Audit-Anbindung, Domänen-Skills, REST-Endpunkte | **ja** | bleibt im Backend |

**Konsequenz für Interfaces vs. Implementierungen — „Interface tief, Bindung hoch":**

- **Interfaces** (`ILlmClient`, `IEmbedder`, `IVectorStore`, künftig `IMemoryStore`,
  `ISessionStore`, `IToolbox`) werden in **M/A** definiert.
- **Landrix-Implementierungen** (ORM-Memory, guard-geprüfte Tools) leben in **B**
  und werden zur Laufzeit injiziert. So bekommt „jeder etwas davon" (M ist
  upstream-fähig), und das Backend erbt nur die Naht, nicht die Engine.

## 3. Persistenz- & Vektor-Abstraktion (sqlite-vec ↔ pgvector)

**Status heute:** Die Abstraktion existiert bereits — `IVectorStore` (Add/Search/
Count) und `IEmbedder` (Embed/EmbedBatch/Model) sind Interfaces, und `TLlmRag`
hängt **nur** an ihnen ([mormot.ai.rag.pas](src/mormot.ai.rag.pas),
[mormot.ai.vectorstore.pas](src/mormot.ai.vectorstore.pas)). Ein zweites Backend
(pgvector) ist damit eine **neue Implementierung**, kein Umbau der RAG-Engine.

**Problem:** Interface *und* sqlite-vec-Implementierung stecken in **einer** Unit,
die `mormot.db.raw.sqlite3.static` zieht
([mormot.ai.vectorstore.pas:21](src/mormot.ai.vectorstore.pas#L21)). Wer pgvector
(oder gar kein SQLite) will, schleppt trotzdem die statische SQLite-Abhängigkeit
mit. Das verhindert sauberes Upstreamen *und* den Merge.

**Soll-Struktur (Refactor):**

```
mormot.ai.vectorstore           # NUR Interface IVectorStore + TRagHit + Vektor-Helfer
mormot.ai.vectorstore.sqlitevec # TVec0Store (sqlite-vec) + sqlite-vec/lembed-Loader
mormot.ai.vectorstore.pgvector  # TPgVectorStore (IVectorStore) auf mormot.db.pgvector.*
mormot.ai.embeddings            # NUR Interface IEmbedder
mormot.ai.embed.provider        # TProviderEmbedder (OpenAI-Wire)
mormot.ai.embed.ollama          # TOllamaEmbedder (aus Merge)
mormot.ai.embed.lembed          # TLembedEmbedder (sqlite-lembed, lokal)
```

Regel: **Die RAG-/Agent-Engine kennt nur die Interfaces.** Welches Backend
(sqlite-vec im Single-Binary-Edge-Fall, pgvector im Server-/Multi-Tenant-Fall)
zum Einsatz kommt, entscheidet die Komposition in Schicht A/B per DI — nie ein
`{$ifdef}` in der Engine.

`mormot.db.pgvector.*` (Binding/ORM/Core) ist der **stärkste reine Upstream-
Kandidat**: pure DB-Schicht, passt in mORMots `mormot.db.*`-Familie, kein
AI-Bezug. Bei der Aufnahme dort verorten, nicht unter `mormot.ai.*`.

## 4. Merge mit dem parallelen `mormot.ai.*`-Repo

Das zweite Repo entwickelt denselben Namespace in eine ergänzende Richtung
(Ollama-Treiber, pgvector, feinere RAG-Zerlegung, CLI). Ziel ist **eine** runde
`mormot.ai.*`-Toolbox. Namenskonflikte vor dem Merge auflösen:

| Thema | Hier (LandrixAI) | Parallel-Repo | Entscheidung (Vorschlag) |
|---|---|---|---|
| Embeddings | `mormot.ai.embeddings` | `mormot.ai.embed[.ollama]` | Interface in `mormot.ai.embeddings`, Treiber als `mormot.ai.embed.*` (beider Stärken) |
| SSE | `mormot.ai.llm.sse` | `mormot.ai.http.sse` | **`mormot.ai.http.sse`** — SSE ist generisch HTTP, nicht LLM-spezifisch |
| Ollama-LLM | über OpenAI-Wire `TLlmClient` | dedizierte `mormot.ai.llm.ollama` | Default OpenAI-Wire; **dedizierter Treiber nur bei echter Wire-Abweichung** (Regel wie beim Anthropic-Treiber) |
| RAG | monolithisch `mormot.ai.rag` | `rag` + `rag.service` + `rag.stream` + `rag.history` | **Zerlegung übernehmen**: Engine (chunk/ingest/query) vs. Session-Service vs. Streaming vs. History |
| Chunking | `ChunkText` in `mormot.ai.rag` | eigenes `mormot.ai.chunk` | **Extrahieren** nach `mormot.ai.chunk` |
| Vektor-Store | sqlite-vec (`TVec0Store`) | `mormot.db.pgvector.*` | **Beide** als `IVectorStore`-Backends (siehe §3) |
| CLI | — | `mormot.ai.cli` | Übernehmen (Test-/Demo-Harness) |
| MCP | `mormot.ai.mcp.*` (Server+Client) | — | bleibt, ist unser Vorsprung |

**Vor dem Merge** Lizenz-/Herkunfts-Header klären (beide adoptieren ggf. von
flydev) — siehe [NOTICE](NOTICE)/[LICENSE](LICENSE).

## 5. Ist-Stand-Inventar (Schicht M/A, gebaut)

- **LLM-Client**: `mormot.ai.llm` (OpenAI-Wire = Lingua franca) + nativer
  `mormot.ai.llm.anthropic`; Typen `mormot.ai.llm.types`; SSE-Streaming;
  Vision/Multimodal (`TLlmMessage.Images`).
- **Agent-Loop**: `TLlmAgent` + `ILlmToolbox` ([mormot.ai.agent.pas](src/mormot.ai.agent.pas)).
- **MCP**: Server+Client, stdio/HTTP/SSE/Streamable, RTTI-Tool-Schema
  (`mormot.ai.mcp.*`), MCP-Bridge in den Agenten (`mormot.ai.agent.mcp`).
- **Structured Output**: `mormot.ai.llm.structured`.
- **Embeddings/RAG**: `IEmbedder` (Provider + lokal lembed), `IVectorStore`
  (sqlite-vec), `TLlmRag`, RAG-as-Tool (`mormot.ai.rag.tool`).
- **codenav**: MCP-Tools für Code-Navigation (`codenav/`).

## 6. Offene Punkte

1. **Refactor §3** (Interface/Impl-Units trennen) — Voraussetzung für pgvector +
   Upstream.
2. **Merge §4** durchziehen (Namespaces angleichen, RAG zerlegen, pgvector rein).
3. **Memory-/Session-Interfaces** (Schicht A) definieren — Implementierung im
   Backend (Schicht B), Details in
   [docs/Feature-LandrixAI-Agent.md](../../../docs/Feature-LandrixAI-Agent.md).
4. **x64-linux-`.so`** für sqlite-vec/lembed im Docker-Image (bekannt offen).
