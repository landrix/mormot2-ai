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
mormot.ai.embed.lembed          ✓ TLembedEmbedder (sqlite-lembed, lokal; lädt lembed0 via sqlitevec-Loader)
```
(✓ = gebaut, Demos grün · ○ = offen, kommt mit dem Merge §4)

Regel (gilt jetzt durchgängig): **Die RAG-/Agent-Engine kennt nur die Interfaces.**
Welches Backend (sqlite-vec im Single-Binary-Edge-Fall, pgvector im
Server-/Multi-Tenant-Fall) zum Einsatz kommt, entscheidet die Komposition in
Schicht A/B per DI — nie ein `{$ifdef}` in der Engine.

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

Alles unten **gebaut + review-gehärtet + grün** (FPC 3.2.2 aarch64-linux): MCP-Suite
**202 Assertions**, LLM-Suite **269 Assertions**; Streaming/Tool-Loop/RAG/Vision auch
**live** verifiziert (OpenAI/Ollama/Anthropic). Aufbau-Historie: [DESIGN.md](DESIGN.md).

- **LLM-Client**: `mormot.ai.llm` (OpenAI-Wire = Lingua franca, deckt OpenAI/LiteLLM/
  Ollama) + nativer `mormot.ai.llm.anthropic` (Messages-API hinter derselben
  `ILlmClient`-Naht); Typen `mormot.ai.llm.types`; SSE-Streaming
  (`mormot.ai.llm.sse`, Basis + OpenAI-/Anthropic-Subklassen); Vision/Multimodal
  (`TLlmMessage.Images`, beide Wires).
- **Agent-Loop**: `TLlmAgent` + `ILlmToolbox` ([mormot.ai.agent.pas](src/mormot.ai.agent.pas)).
- **MCP**: Server+Client, stdio/HTTP/SSE/Streamable, RTTI-Tool-Schema
  (`mormot.ai.mcp.*`), MCP-Bridge in den Agenten (`mormot.ai.agent.mcp`).
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

## 6. Offene Punkte / Roadmap

Die Engine-Funktionsfläche (Phase D) steht; offen ist die **Schichtung/der Merge/die
Backend-Bindung** (Phase E) **und die Produktionsreife des MCP-Transports**. Diese
Liste ist die konsolidierte Roadmap — DESIGN.md verweist hierher.

1. **Refactor §3** ✓ **umgesetzt** (Interface/Impl-Units getrennt: `vectorstore` +
   `vectorstore.sqlitevec`, `embeddings` + `embed.provider` + `embed.lembed`; Engine
   SQLite-frei, Demos grün). Offen bleibt nur das **neue** pgvector-Backend (Punkt 2).
2. **Merge §4** durchziehen (Namespaces angleichen: SSE → `mormot.ai.http.sse`,
   Chunking → `mormot.ai.chunk`; RAG zerlegen; pgvector als zweites
   `IVectorStore`-Backend `mormot.ai.vectorstore.pgvector` rein — die Naht steht
   jetzt; Ollama-Embedder als `mormot.ai.embed.ollama`).
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
6. **MCP-Transport-Produktionsreife** (Befunde aus dem kritischen Review, teils
   behoben, teils offen):
   - **Echte Auth** (Phase B): Der Core setzt `IsAuthenticated` jetzt **fail-closed**
     (eine Session-ID ist keine Identität, [mormot.ai.mcp.pas](src/mormot.ai.mcp.pas)).
     Offen: ein verpflichtender Auth-Resolver, der Identität/Rollen aus der
     Backend-Auth befüllt, bevor ein Tool Identität gewährt.
   - **Streamable-Sessions**: Use-after-free (POST/Stream gg. paralleles DELETE) ist
     durch FSafe-serialisiertes **Re-Resolve** geschlossen (kein Objekt-Pointer mehr
     über Lock-Grenzen, `NextSessionEventId`). Offen: ein echter **Parallel-Stresstest**
     (POST↔DELETE, Session-Ablauf während eines laufenden Streams) — die Härtung ist
     bisher nur durch Code-Review, nicht durch einen Concurrency-Test belegt.
   - **Legacy-SSE-Transport** (`TMcpSseTransport`): Map jetzt thread-safe + kein
     Session-Leak mehr, aber Objekt-Lebensdauer über GET/POST-Handler noch ungelockt.
     **Nicht produktionsreif** — vor Exposition härten oder zugunsten von
     **Streamable HTTP** entfernen.
   - **Demo-Härtung** ✓: Die Streamable-Demo bindet jetzt **Loopback** und das
     `ask_claude`-Tool (lokale Claude-CLI) ist standardmäßig **deaktiviert**
     (`MCP_ENABLE_ASK_CLAUDE=1` als Opt-in). Neue `BindAddress`-Property am Transport.
   - **Tests offen**: TVec0Store/TLembedEmbedder-Realtests + RAG-Atomar-Rollback
     (brauchen die sqlite-vec/lembed-Runtime, s. Punkt 4) sowie der o. g. Parallel-Test.

**In diesem Review-Pass bereits behoben** (Build + Tests grün: MCP 202, LLM 256
Assertions): JSON-RPC-Envelope-Validierung (`jsonrpc:"2.0"`, Params-Typ) + breiter
Exception→JSON-RPC-Error-Fang mit korrekten Codes (−32600/−32601/−32603);
fail-closed-Auth; Streamable-UAF via FSafe; SSE-Map thread-safe + Leak-Fix;
RAG-Ingestion **strikt** (exakte Vektoranzahl) **+ atomar** (`IVectorStore.AddBatch`,
eine Transaktion); `build-demo.sh` baut wieder **alle** Demos (vorher nur 5 MCP-Demos
trotz „ALL DEMOS OK"); Demo-Loopback + `ask_claude`-Gate.
