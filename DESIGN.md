# DESIGN — `mormot.ai.*`

## Ziel

Eine mORMot-native AI-Erweiterung. Erster Use-Case: **landrix als MCP-Server** —
es stellt Tools/Ressourcen bereit, die ein externer Agent (z. B. Claude Desktop)
über das Model Context Protocol aufruft. Langfristig Synopse-Contribution.

## Aktueller Stand (Phasen A·C·D abgeschlossen)

Der MCP-Server ist **adoptiert** (mORMot-lizenziert) statt selbst gebaut — Basis:
flydev-fr/mormot2-extensions, auf `mormot.ai.*` umbenannt (Commit-Pin: siehe
`UPSTREAM_BASE`). Darauf aufgesetzt: Spec-Upgrade (Phase C, MCP 2025-11-25) und der
clean-room LLM-Client (Phase D: OpenAI-Wire + Anthropic-Treiber, Agent-/Tool-Loop,
Embeddings/RAG, agentic RAG, Vision). Build **+ alle Tests + alle Demos grün**
(aarch64-linux/FPC 3.2.2): **202 Assertions** MCP-Suite + **256 Assertions** LLM-Suite.
Offen ist die Schichtung/der Merge/die Backend-Bindung (Phase E, siehe
[CONCEPT.md](CONCEPT.md)).

## Architektur (adoptiert)

```
mormot.ai.mcp         Core: Typen, Auth-Context, IMcpTool/IMcpResource,
                      RTTI-Schema-Generierung, JSON-RPC-2.0-Prozessor,
                      TMcpToolBase<T: record>, TMcpServer (Registry + Dispatch:
                      initialize, tools/list, tools/call, resources/*)
mormot.ai.mcp.server  Transporte: HTTP (THttpAsyncServer), SSE (Session-Mgmt),
                      Streamable HTTP
mormot.ai.mcp.stdio   stdio-Transport (line-based JSON-RPC, Worker-Thread)
mormot.ai.mcp.tools   Beispiel-Tool-Parameter-Records
```

Kern-API: `TMcpServer.RegisterTool(IMcpTool)` / `RegisterResource`,
`Start`/`Stop`/`IsActive`, `ExecuteRequest(json, sessionId)`. Tools implementieren
`IMcpTool` oder erben `TMcpToolBase<T: record>` — das **Input-Schema wird via RTTI
aus dem typisierten Record generiert** (kein handgeschriebenes JSON-Schema).

## Transporte

stdio · HTTP · SSE · **Streamable HTTP** · in-process (`ExecuteRequest` direkt).
SSE ist Spec-`legacy`, bleibt aber als adoptierter Transport vorhanden. Der für
landrix relevante Produktions-Transport (hinter Caddy) ist **Streamable HTTP**.

## Clean-Room-Politik (verbindlich)

Gilt für die **künftigen, selbst gebauten** Teile (v. a. `mormot.ai.llm`) — nicht
für den adoptierten, mORMot-lizenzierten MCP-Server.

- Selbst gebaute Teile werden **clean-room** gegen die **offiziellen Quellen**
  implementiert:
  - MCP-Spec — https://modelcontextprotocol.io
  - JSON-RPC 2.0 — https://www.jsonrpc.org/specification
  - die jeweiligen Provider-API-Dokumentationen
- Es wird **kein fremder Quellcode übernommen**. Die interne Prozess-/Provenance-
  Dokumentation (Trennwand) wird separat und **nicht eingecheckt** geführt.
- Grund: rechtlich sauber und aufnahmefähig als mORMot-Contribution.

## Bauen & Testen

```bash
# in WSL (nativ aarch64), aus dem Repo-Root:
bash shared/delphi/landrixai/scripts/run-fpc-tests.sh   # Lib + Tests
bash shared/delphi/landrixai/scripts/build-demo.sh      # alle Demos
```
Verdrahtung wie das Backend: mORMot-Unit-/Static-Pfade aus
`shared/delphi/libs/_git_Synopse2`; Build-Output unter `bin/` (gitignored).

## Test-Politik

Die Tests nutzen mORMots **`TSynTests`** (nicht FPCUnit) — passend zum
Contribution-Ziel. Zwei Runner:
- `tests/mcp.tests.dpr` — MCP-Suite (Core + Transporte + Streamable),
  **202 Assertions** grün; `scripts/run-fpc-tests.sh`.
- `tests/llm.tests.dpr` — LLM-Suite (SSE, Client, Agent, Agent-MCP, **Anthropic**,
  Structured, RAG, RAG-Tool, Vision), **256 Assertions** grün;
  `scripts/run-fpc-llm-tests.sh`.

Neue Tests im passenden Runner ergänzen.

## Roadmap

- **Phase A** ✓ — flydev-MCP-Server adoptiert, `mormot.ai.*`, Build/Tests/Demos grün.
- **Phase B** — landrix-spezifische MCP-Tools über `TMcpServer.RegisterTool`
  andocken; Auth über die vorhandene mORMot-Auth des Backends.
- **Phase C** ✓ — MCP-Spec auf **2025-11-25** mit Versions-Negotiation
  (`initialize` echot unterstützte Client-Versionen, sonst Fallback = neueste);
  der transportabhängige Patch-Hack wurde entfernt (einheitliches Verhalten).
- **Phase D** ✓ — Clean-Room LLM-Client (`mormot.ai.llm.*`): Provider-Treiber +
  Agent-/Tool-Calling-Loop + Embeddings/RAG + zweiter Provider (Anthropic) +
  Vision. Komplett gebaut, review-gehärtet, **256 Assertions** grün
  (`llm.tests.dpr`); Streaming/Tool-Loop/RAG/Vision live verifiziert. Details unten.
- **Phase E** (offen) — Schichtung/Merge/Backend-Bindung, siehe
  **[CONCEPT.md](CONCEPT.md)** (Single Source of Truth für die offenen Punkte) und
  „Offene Punkte" weiter unten.

### Phase D im Detail — Architektur-Entscheidungen:
  - **Kanonisches Wire = OpenAI Chat Completions** (Lingua franca): ein
    OpenAI-Wire-Client + Provider-Config (Base-URL/Auth/Model) deckt OpenAI,
    LiteLLM und Ollama-Compat ab; native Adapter nur, wo das Wire echt abweicht.
  - **JSON hybrid**: typisierte Records (`mormot.ai.llm.types`) als API-Fläche,
    Wire-Mapping/Parsing über `TDocVariantData` (provider-Toleranz).
  - **Streaming-first**: das einzige echte Risiko zuerst geklärt. Verifiziert per
    Quelltext: `THttpSocket.GetBody(DestStream)` schreibt **jeden Transfer-Chunk
    live** in den übergebenen `OutStream` (mormot.net.http.pas) — ein eigener
    `TStream` (`TLlmSseStream` in `mormot.ai.llm.sse`) parst die `data:`-Events
    inkrementell. Caveat: für den Stream-Request **kein** Content-Encoding (gzip)
    anbieten, sonst wirft GetBody.
  - **Callback = Methoden-Pointer (`of object`)**, NICHT `reference to`/Closures —
    FPC 3.2.2 kennt die Modeswitches `functionreferences`/`anonymousfunctions`
    nicht (erst 3.3.1).
  - **Stand**: vollständig gebaut + review-gehärtet, **256 Assertions** grün
    (`llm.tests`, `scripts/run-fpc-llm-tests.sh`). Bausteine:
    - `mormot.ai.llm.types` — provider-neutrale Records.
    - `mormot.ai.llm.sse` — `TLlmSseStream` (Streaming-Parser; whole/1-byte/tool-call).
    - `mormot.ai.llm` — `ILlmClient`/`TLlmClient` (`ChatComplete` + `ChatStream`)
      über `THttpClientSocket`; freie Funktionen `OpenAIChatRequestJson` /
      `ParseOpenAIChatResponse` (hermetisch getestet).
    - `mormot.ai.llm.openai` — Provider-Configs `OpenAIConfig`/`OllamaConfig`/
      `LiteLLMConfig`.
    - `mormot.ai.agent` — `ILlmToolbox` (toolquellen-agnostische Naht) +
      `TLlmCallbackToolbox` + `TLlmAgent` (Tool-Calling-Loop: Tools anbieten →
      `tool_calls` ausführen → `role:tool` zurückspeisen → bis Antwort, mit
      `MaxIterations`-Guard).
    - `mormot.ai.agent.mcp` — `TLlmMcpToolbox`: `ILlmToolbox` über einen
      `TMcpServer` (JSON-RPC `tools/list`/`tools/call` in-process). Damit treibt
      ein Agent direkt die Tools, die ein MCP-Server exponiert — derselbe
      RTTI-Schema-Generator an beiden Enden (der Suite-Schluss).
    - `mormot.ai.llm.structured` — getypte Ausgabe: `ChatStructured(client, req,
      TypeInfo(TRec), out rec)` generiert das JSON-Schema aus dem Record
      (`TMcpSchemaGenerator`, dieselbe Maschine wie MCP-Tool-Input), setzt
      `response_format: json_schema` und lädt die Antwort via `RecordLoadJson` in
      den Record. `ResponseFormat`-Feld in `TLlmChatRequest` + Demo
      `demos/llm/llm-structured.dpr`. **Live**: deutsche Belegextraktion gegen
      Ollama (`german-text-3.1`) → typisierter Record.
    - **DoS-Cap**: optionales `MaxResponseBytes` in `TLlmProviderConfig` umwickelt
      den Stream-`OutStream` mit `TLimitedStreamWriter` (mORMot-nativ, cappt die
      *kumulative* Antwort — Ergänzung zum per-Chunk-`MaxHttpChunkSize`).
    - Demos `demos/llm/llm-chat.dpr` (Streaming) + `llm-agent.dpr` (Tool-Loop).
    - **Live verifiziert gegen Ollama (172.16.122.3)**: (a) Token-Streaming
      end-to-end (`gemma3:12b`): `TLlmClient → THttpClientSocket → GetBody (live
      chunks) → TLlmSseStream → OnDelta`; (b) Tool-Calling-Loop
      (`german-text-3.1`): Modell ruft `get_weather`, Agent führt aus, Modell
      antwortet aus dem Tool-Ergebnis.
    - **Review-gehärtet** (landrix-code-review, 8 Findings behoben + Tests):
      `AddOrUpdateFrom` statt duplizierendem `AddFrom`; SSE-`Flush` für ein letztes
      Event ohne Trailing-`\n`; `RawBody` für lesbare Nicht-SSE-Fehlerbodies;
      leak-/double-free-sicheres `ChatStream` (TLimitedStreamWriter kann bei
      Position=Size=0 nicht werfen) + Connect-Guard; MCP `isError`-Durchreichung;
      `IsValidJson`-Guard für Tool-Args; `content` weggelassen statt `""` bei
      Assistant-Tool-only. **95 Assertions** grün (inkl. Structured-Output).
    - `mormot.ai.embeddings` — `IEmbedder` (backend-neutral) + `TProviderEmbedder`
      (OpenAI-Wire `/embeddings`). Client um `Embeddings(model, input[])` +
      `ParseOpenAIEmbeddings` erweitert (Connect endpoint-parametrisiert). Demo
      `demos/llm/llm-embed.dpr`. **Live**: text-embedding-3-small (1536d),
      cos(Hund,Katze)=0.59 vs cos(Hund,Auto)=0.25. **104 Assertions** grün.
  - **RAG (in Arbeit, Backend FPC Linux+Windows)**: Machbarkeit bestätigt — mORMots
    **static-SQLite exportiert `sqlite3_load_extension`** (mormot.db.raw.sqlite3.static.pas:1047)
    + `enable_load_extension`, also kann mORMots eigenes SQLite die Extensions
    `vec0`/`lembed0` laden (FPC, Linux+Windows). Plan: `IEmbedder` (Provider ✓ /
    lokal `TLembedEmbedder` via `select lembed(...)` offen) + `IVectorStore`
    (sqlite-vec `vec0` KNN) + Chunking + `TLlmRag` (Ingest/Query→Grounding-Prompt→
    ChatComplete). Basiert konzeptionell auf [landrix/sqlite-vec-for-Delphi]
    (sqlite-vec + sqlite-lembed + sqlite-vector, GGUF z. B. BGE-M3).
    **Fundament bewiesen** (Spike `demos/rag/rag-spike.dpr`, FPC/WSL aarch64 grün):
    vec0+lembed0 in mORMots SQLite geladen (`db_config(SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION)`
    + `load_extension`), GGUF-Modell registriert, deutscher Text lokal geembedded,
    vec0-KNN semantisch korrekt — offline. Binaries/Modelle in `vendor/sqlite-ext/`
    + `vendor/models/` (gitignored; aarch64-linux + win64 + all-MiniLM/bge-m3 da).
    - **Units gebaut + review-gehärtet**: `mormot.ai.vectorstore` (Loader
      `Enable/LoadSqliteExtension`, `IVectorStore`/`TVec0Store`, `TLembedEmbedder`
      als lokaler `IEmbedder`, Blob-Helfer) + `mormot.ai.rag` (`ChunkText`
      wortausgerichtet/UTF-8-sicher, `TLlmRag` Ingest/Query→Grounding→ChatComplete
      mit Zitaten). Provider-`Embeddings` im Client (`/embeddings` +
      `ParseOpenAIEmbeddings`) + `mormot.ai.embeddings.TProviderEmbedder`. Demo
      `demos/rag/rag-chat.dpr` **live**: lokales lembed-Retrieval + OpenAI-Antwort,
      geerdet + zitiert. Chunking/Blob hermetisch.
      Review-Fixes: Embedder pinnt Store-Lifetime; dim/topK-Guards; Add
      transaktional; Embed wirft bei Leerergebnis; Injection-Delimiter; empty-store
      Early-Return.
  - **Agentic RAG** ✓ (`mormot.ai.rag.tool`): `TRagSearchTool` =
    `TMcpToolBase<TRagSearchParams>` exponiert Retrieval als **`search_docs`**-MCP-Tool
    — dasselbe Tool für externe MCP-Clients **und** für den In-Process-Agenten (über
    den vorhandenen `TLlmMcpToolbox`-Bridge, keine neue Toolbox-Verdrahtung). Der Agent
    entscheidet selbst, **wann** er retrievt, und antwortet zitiert nur aus den
    Treffern. Demo `demos/rag/rag-agent.dpr` **live** (verfeinerte Query → Retrieval).
  - **Anthropic-Treiber** ✓ (`mormot.ai.llm.anthropic`, `TAnthropicClient`): nativer
    Messages-API-Adapter hinter derselben `ILlmClient`-Naht — `system` top-level,
    `max_tokens` Pflicht, Tools via `input_schema`/`tool_use`/`tool_result`,
    `x-api-key`+`anthropic-version`, event-getypte SSE (`TAnthropicSseStream`:
    message_start/content_block_delta/message_delta/message_stop/**error**). Parallele
    Tool-Resultate werden zu **einer** User-Message mit mehreren `tool_result`-Blöcken
    gebündelt; `input_tokens` aus `message_start` füllt `TotalTokens`. Test hermetisch
    (kein Key/Netz). **Live** (`claude-opus-4-8`): Tool-Loop `get_weather` → geerdete
    Antwort.
  - **Vision/multimodal** ✓: `TLlmMessage.Images` (`TLlmImage` Source/MediaType/Data;
    `LlmImageBase64`/`LlmImageUrl`/`LlmImageMessage`) — beide Wires serialisieren
    (OpenAI `image_url` inkl. base64-`data:`-URI, Anthropic `image`-Block mit
    typisiertem `source`); leerer `MediaType` → Default `image/png`. Demo
    `demos/llm/llm-vision.dpr` provider-agnostisch (`VISION_PROVIDER=openai|anthropic`).
    **Live** gegen `gpt-4o-mini` **und** `claude-opus-4-8` (beide erkennen denselben
    inline-base64-Kreis).
  - **Offene Punkte** (Roadmap → siehe [CONCEPT.md §6](CONCEPT.md), dort konsolidiert):
    1. **x86_64-linux-`.so`** von `vec0`/`lembed0` ✓ **vorhanden + laufzeit-verifiziert**
       (alle 4 Arch-Builds da; vec0 `v0.1.10-alpha.4` + lembed0 gegen mORMots statisches
       SQLite geprüft — aarch64 nativ + x86_64 im amd64-Container via
       `scripts/x64-ext-verify.sh`). Rest: Verdrahtung ins Docker-Image.
    2. **Interface/Impl-Split** der VectorStore-/Embedder-Units (CONCEPT §3) ✓
       **umgesetzt** — `vectorstore`(+`.sqlitevec`) / `embeddings`(+`embed.provider`/
       `embed.lembed`); Engine ist SQLite-frei, Demos grün. Rest = pgvector (Punkt 3).
    3. **Merge** mit dem parallelen `mormot.ai.*`-Repo (CONCEPT §4): Namespaces
       angleichen (SSE → `mormot.ai.http.sse`, Chunking → `mormot.ai.chunk`), RAG
       zerlegen, pgvector als zweites `IVectorStore`-Backend.
    4. **Memory-/Session-Interfaces** (Schicht A) definieren, Backend-Bindung
       (Schicht B) implementieren — siehe
       [docs/Feature-LandrixAI-Agent.md](../../../docs/Feature-LandrixAI-Agent.md).
    5. **Anthropic-Restfläche**: Structured Output (`output_config.format`, abweichend
       vom OpenAI `response_format`) und Live-Verifikation des Anthropic-Streamings
       (SSE-Parser inkl. error-Event bisher nur hermetisch getestet).
    6. **MCP-Transport-Produktionsreife** (kritischer Review, CONCEPT §6): echter
       Auth-Resolver (Phase B; Core ist jetzt fail-closed), Parallel-Stresstest
       POST↔DELETE/Session-Ablauf für die Streamable-Sessions (UAF per FSafe-Re-Resolve
       geschlossen, aber nur review-belegt), Legacy-SSE-Transport härten oder entfernen,
       TVec0Store/lembed-Realtests + RAG-Atomar-Rollback (brauchen die sqlite-vec-Runtime).

### Review-Härtung (kritischer Review, behoben — Build + Tests grün)

Befunde aus einem kritischen Review abgearbeitet (verifiziert: MCP **202** + LLM **256**
Assertions, alle 15 Demos bauen):
- **JSON-RPC robust**: `ParseRequest` validiert `jsonrpc:"2.0"` + Params-Typ;
  `ExecuteRequest` fängt **jede** Exception (nicht nur `ESynException`) und mappt auf
  korrekte Codes (−32600 Invalid Request / −32601 Method Not Found / −32603 Internal) —
  eine Tool-Exception kann den HTTP-Worker nicht mehr abreißen.
- **Auth ehrlich**: `IsAuthenticated` ist **fail-closed** (Session-ID ≠ Identität).
- **Streamable-UAF geschlossen**: kein Session-**Objekt**-Pointer mehr über Lock-Grenzen;
  DELETE/POST/Stream serialisieren über `FSafe`, Event-IDs via `NextSessionEventId`
  (Re-Resolve unter Lock).
- **Legacy-SSE**: Session-Map thread-safe + Leak in `ClearSessions` behoben; Transport
  klar als nicht produktionsreif markiert.
- **RAG-Ingestion strikt + atomar**: exakte Vektoranzahl erzwungen (sonst Fehler);
  `IVectorStore.AddBatch` committet ein Dokument in **einer** Transaktion (kein
  Teil-Index).
- **build-demo.sh** baut wieder **alle** Demos (vorher nur 5 MCP-Demos trotz „ALL DEMOS
  OK"); **Demo-Härtung**: Streamable-Demo bindet Loopback, `ask_claude` nur per
  `MCP_ENABLE_ASK_CLAUDE=1` (neue `BindAddress`-Property am Transport).

## Lizenz / Contribution

Ziellizenz: mORMot-Drei-Lizenz (MPL 1.1 / GPL 2.0 / LGPL 2.1). Namespace
`mormot.ai.*` und CLA/Coding-Style **vorab mit Synopse (Arnaud Bouchez) sowie mit
flydev abstimmen** (Namespace `mormot.ai.*` vs. flydevs `mormot.ext.mcp`).
