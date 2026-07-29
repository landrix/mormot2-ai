# DESIGN — `mormot.ai.*`

## Ziel

Eine mORMot-native AI-Erweiterung. Erster Use-Case: **landrix als MCP-Server** —
es stellt Tools/Ressourcen bereit, die ein externer Agent (z. B. Claude Desktop)
über das Model Context Protocol aufruft. Langfristig Synopse-Contribution.

## Aktueller Stand (Phasen A·C·D abgeschlossen)

Der MCP-Server ist **adoptiert** (mORMot-lizenziert) statt selbst gebaut — Basis:
flydev-fr/mormot2-extensions, auf `mormot.ai.*` umbenannt (Commit-Pin: siehe
`UPSTREAM_BASE`). Darauf aufgesetzt: das Spec-Upgrade auf **MCP 2026-07-28**
(Phase C, stateless — siehe unten) und der clean-room LLM-Client (Phase D:
OpenAI-Wire + Anthropic-Treiber, Agent-/Tool-Loop, Embeddings/RAG, agentic RAG,
Vision). Build **+ alle Tests + alle Demos grün** (aarch64-linux/FPC 3.2.2):
**402 Assertions** MCP-Suite + **272 Assertions** LLM-Suite. Offen ist die
Schichtung/der Merge/die Backend-Bindung (Phase E, siehe [CONCEPT.md](CONCEPT.md)).

## Architektur (adoptiert)

```
mormot.ai.mcp         Core: Typen, Auth-Context, IMcpTool/IMcpResource,
                      RTTI-Schema-Generierung, JSON-RPC-2.0-Prozessor,
                      TMcpToolBase<T: record>, TMcpServer (Registry + Dispatch:
                      server/discover, tools/list, tools/call, resources/*)
mormot.ai.mcp.server  Transporte: HTTP (THttpAsyncServer), Streamable HTTP
mormot.ai.mcp.stdio   stdio-Transport (line-based JSON-RPC, Worker-Thread)
mormot.ai.mcp.tools   Beispiel-Tool-Parameter-Records
```

Kern-API: `TMcpServer.RegisterTool(IMcpTool)` / `RegisterResource`,
`Start`/`Stop`/`IsActive`, `ExecuteRequest(json)` und `PreflightRequest` (validiert
ohne zu dispatchen und liefert den HTTP-Status, den ein HTTP-Transport senden
muss). Tools implementieren `IMcpTool` oder erben `TMcpToolBase<T: record>` — das
**Input-Schema wird via RTTI aus dem typisierten Record generiert** (kein
handgeschriebenes JSON-Schema).

## Protokoll: MCP 2026-07-28 (stateless), nur diese Revision

Die Revision hat den `initialize`-Handshake und die Protokoll-Sessions **entfernt**.
Konsequenzen, die die ganze Implementierung prägen:

- **Jede** Anfrage trägt ihre eigene Protokollversion und Client-Capabilities in
  `params._meta` — es gibt keinen Verbindungszustand, aus dem sie folgen könnten.
  Fehlen sie, ist die Anfrage ungültig (`-32602`); eine fremde Version ergibt
  `-32022` samt Liste der unterstützten Versionen.
- `server/discover` ersetzt `initialize` als Einstiegs-RPC (MUSS implementiert sein).
- Jedes Result trägt `resultType` und `_meta.serverInfo` (`FinalizeResult`).
- **Caching-Hints sind Pflicht** auf `server/discover`, `tools/list`, `resources/list`
  und `resources/read`: `ttlMs` (≥ 0) und `cacheScope`. Konfigurierbar am
  `TMcpServer` — **getrennt für Listen und Read** (`ListCacheTtlMs`/`ListCacheScope`
  vs. `ReadCacheTtlMs`/`ReadCacheScope`), damit eine cachebare Tool-Liste nicht
  zwingt, auch Ressourcen-**Inhalte** für geteilte Proxies freizugeben. Der Scope
  ist ein **Enum** (`TMcpCacheScope`), kein String: ein Tippfehler kann so keine
  erfolgreiche, aber spec-ungültige Antwort erzeugen, und ein zur Laufzeit
  umkonfigurierter Server kann kein refcountetes Feld unter parallelen Requests
  zerlegen.
  **Defaults sind bewusst konservativ**: `ttlMs = 0` (sofort veraltet) und
  `cacheScope = private`. Grund: Die Registry kann sich jederzeit über
  `RegisterTool` ändern und es gibt bis `subscriptions/listen` kein
  Invalidierungssignal; und `public` erlaubt geteilten Proxies laut Spec
  ausdrücklich, eine Antwort **über Autorisierungs-Kontexte hinweg** an andere
  Aufrufer auszuliefern. `cacheScope` ist ein Cache-Hinweis, **nie** eine
  Zugriffskontrolle.
- **`subscriptions/listen`** ersetzt den GET-Stream und `resources/subscribe`: ein
  langlebiger POST-Response-Stream, der nur die Notification-Typen liefert, die der
  Client im Filter angefordert hat (Whitelist — der Server darf nichts anderes
  senden). Erste Nachricht ist die Acknowledgement, jede Nachricht trägt
  `_meta.subscriptionId` (= id des listen-Requests), Abschluss ist das leere Result
  auf denselben Request. `RegisterTool`/`RegisterResource` lösen die
  list_changed-Notifications selbst aus, deshalb meldet `server/discover`
  `listChanged`/`subscribe` — die Ankündigung deckt sich mit dem Verhalten.
  **Nur der Streamable-HTTP-Transport** kann das; stdio und der einfache
  HTTP-Transport antworten mit `-32601` statt einer leeren Erfolgsantwort.
  Zwei Grenzen sind bewusst gesetzt: `MaxSubscriptions` (Default 8), weil jeder
  offene Stream einen Worker-Thread hält, und `MCP_SUBSCRIPTION_MAX_PENDING`
  (256) pro Stream — ein Client, der nicht mitkommt, wird abgeworfen statt den
  Speicher wachsen zu lassen. `Stop` bricht alle Streams ab, **bevor** es den
  HTTP-Server herunterfährt: dessen Shutdown wartet nur begrenzt auf Worker und
  räumt danach zwangsweise ab.
- **Validierung entscheidet den HTTP-Status, und zwar bevor gestreamt wird**:
  `PreflightRequest` prüft Envelope, `_meta` und Methodenexistenz. Abgelehnte
  Anfragen gehen als gepuffertes JSON mit **400** (bzw. **404** für `-32601`) raus,
  niemals als SSE-Stream — dessen Kopf steht vor dem Ergebnis fest und wäre immer
  `200`. Ein `OnStreamCall`-Hook sieht deshalb nur bereits validierte Anfragen.
- Streamable HTTP: **nur POST** (GET/DELETE → 405), kein Batching, keine
  Resumability (`Last-Event-ID`), keine `Mcp-Session-Id`; die Standard-Header
  `MCP-Protocol-Version`/`Mcp-Method`/`Mcp-Name` sind Pflicht und werden gegen den
  Body geprüft (`-32020` + 400).

Ältere, handshake-basierte Revisionen werden **nicht** bedient (Greenfield: was die
Spec entfernt, ist hier gelöscht, nicht deaktiviert).

## Transporte

stdio · HTTP · **Streamable HTTP** · in-process (`ExecuteRequest` direkt).
Der Legacy-HTTP+SSE-Transport wurde mit 2026-07-28 **gelöscht** (die Spec führt ihn
als *Deprecated*). Der für landrix relevante Produktions-Transport (hinter Caddy)
ist **Streamable HTTP**.

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
- `tests/mcp.tests.lpr` — MCP-Suite (Core + Transporte + Streamable),
  **402 Assertions** grün; `scripts/run-fpc-tests.sh`.
- `tests/llm.tests.lpr` — LLM-Suite (SSE, Client, Agent, Agent-MCP, **Anthropic**,
  Structured, RAG, RAG-Tool, Vision), **272 Assertions** grün;
  `scripts/run-fpc-llm-tests.sh`.

Neue Tests im passenden Runner ergänzen.

## Roadmap

- **Phase A** ✓ — flydev-MCP-Server adoptiert, `mormot.ai.*`, Build/Tests/Demos grün.
- **Phase B** — landrix-spezifische MCP-Tools über `TMcpServer.RegisterTool`
  andocken; Auth über die vorhandene mORMot-Auth des Backends.
- **Phase C** ✓ — MCP-Spec auf **2026-07-28** (stateless): `initialize`, Sessions,
  Batching, SSE-Resumability und der HTTP+SSE-Transport sind **gelöscht**;
  `server/discover` + per-Request-`_meta` treten an ihre Stelle. Siehe „Protokoll"
  oben. (Zwischenstand 2025-11-25 mit Versions-Negotiation ist damit überholt.)
- **Phase D** ✓ — Clean-Room LLM-Client (`mormot.ai.llm.*`): Provider-Treiber +
  Agent-/Tool-Calling-Loop + Embeddings/RAG + zweiter Provider (Anthropic) +
  Vision. Komplett gebaut, review-gehärtet, **272 Assertions** grün
  (`llm.tests.lpr`); Streaming/Tool-Loop/RAG/Vision live verifiziert. Details unten.
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
  - **Stand**: vollständig gebaut + review-gehärtet, **272 Assertions** grün
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
      `demos/llm/llm-structured.lpr`. **Live**: deutsche Belegextraktion gegen
      Ollama (`german-text-3.1`) → typisierter Record.
    - **DoS-Cap**: optionales `MaxResponseBytes` in `TLlmProviderConfig` umwickelt
      den Stream-`OutStream` mit `TLimitedStreamWriter` (mORMot-nativ, cappt die
      *kumulative* Antwort — Ergänzung zum per-Chunk-`MaxHttpChunkSize`).
    - Demos `demos/llm/llm-chat.lpr` (Streaming) + `llm-agent.lpr` (Tool-Loop).
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
      `demos/llm/llm-embed.lpr`. **Live**: text-embedding-3-small (1536d),
      cos(Hund,Katze)=0.59 vs cos(Hund,Auto)=0.25. **104 Assertions** grün.
  - **RAG (in Arbeit, Backend FPC Linux+Windows)**: Machbarkeit bestätigt — mORMots
    **static-SQLite exportiert `sqlite3_load_extension`** (mormot.db.raw.sqlite3.static.pas:1047)
    + `enable_load_extension`, also kann mORMots eigenes SQLite die Extensions
    `vec0`/`lembed0` laden (FPC, Linux+Windows). Plan: `IEmbedder` (Provider ✓ /
    lokal `TLembedEmbedder` via `select lembed(...)` offen) + `IVectorStore`
    (sqlite-vec `vec0` KNN) + Chunking + `TLlmRag` (Ingest/Query→Grounding-Prompt→
    ChatComplete). Basiert konzeptionell auf [landrix/sqlite-vec-for-Delphi]
    (sqlite-vec + sqlite-lembed + sqlite-vector, GGUF z. B. BGE-M3).
    **Fundament bewiesen** (Spike `demos/rag/rag-spike.lpr`, FPC/WSL aarch64 grün):
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
      `demos/rag/rag-chat.lpr` **live**: lokales lembed-Retrieval + OpenAI-Antwort,
      geerdet + zitiert. Chunking/Blob hermetisch.
      Review-Fixes: Embedder pinnt Store-Lifetime; dim/topK-Guards; Add
      transaktional; Embed wirft bei Leerergebnis; Injection-Delimiter; empty-store
      Early-Return.
  - **Agentic RAG** ✓ (`mormot.ai.rag.tool`): `TRagSearchTool` =
    `TMcpToolBase<TRagSearchParams>` exponiert Retrieval als **`search_docs`**-MCP-Tool
    — dasselbe Tool für externe MCP-Clients **und** für den In-Process-Agenten (über
    den vorhandenen `TLlmMcpToolbox`-Bridge, keine neue Toolbox-Verdrahtung). Der Agent
    entscheidet selbst, **wann** er retrievt, und antwortet zitiert nur aus den
    Treffern. Demo `demos/rag/rag-agent.lpr` **live** (verfeinerte Query → Retrieval).
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
    `demos/llm/llm-vision.lpr` provider-agnostisch (`VISION_PROVIDER=openai|anthropic`).
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
    5. **Anthropic-Restfläche** ✓ **erledigt**: Structured Output via
       `output_config.format` (Adapter übersetzt das neutrale `ResponseFormat` +
       injiziert `additionalProperties:false`) und Anthropic-**Streaming** — beide
       **live** gegen `claude-opus-4-8` verifiziert; `llm-structured`/`llm-chat` jetzt
       provider-agnostisch (`LLM_PROVIDER`).
    6. **MCP-Transport-Produktionsreife** (kritischer Review, CONCEPT §6): echter
       Auth-Resolver (Phase B; Core ist jetzt fail-closed) — der einzige verbliebene
       Punkt. Sessions/Legacy-SSE sind mit 2026-07-28 **gelöscht** (damit entfallen
       Session-Ablauf, UAF-Fläche und Härtung des Legacy-Transports); Nebenläufigkeit
       deckt jetzt `ConcurrentPosts` ab (4 parallele Clients + Keep-Alive-Reuse).
       Offen bleiben TVec0Store/lembed-Realtests + RAG-Atomar-Rollback (brauchen die
       sqlite-vec-Runtime).
    7. **MCP-Features nach Phase 1+2**: `CacheableResult`
       (`ttlMs`/`cacheScope`), `x-mcp-header`, MRTR (`InputRequiredResult`,
       `resultType:input_required`), Extensions-Framework, JSON Schema 2020-12 im
       `inputSchema`, OTel-`_meta`-Keys. `MCP_ERROR_MISSING_CLIENT_CAPABILITY`
       (`-32021`) ist dafür bereits vorgesehen, wird aber erst mit MRTR ausgelöst —
       heute verlangt kein Pfad eine Client-Capability.

### Review-Härtung (kritischer Review, behoben — Build + Tests grün)

Befunde aus einem kritischen Review abgearbeitet (verifiziert: MCP **202** + LLM **256**
Assertions, alle 15 Demos bauen):
- **JSON-RPC robust**: `ParseRequest` validiert `jsonrpc:"2.0"` + Params-Typ;
  `ExecuteRequest` fängt **jede** Exception (nicht nur `ESynException`) und mappt auf
  korrekte Codes (−32600 Invalid Request / −32601 Method Not Found / −32603 Internal) —
  eine Tool-Exception kann den HTTP-Worker nicht mehr abreißen.
- **Auth ehrlich**: `IsAuthenticated` ist **fail-closed** (Identität kommt nur aus einem
  echten Auth-Resolver, nie aus dem Transport).
  > Die früheren Session-Befunde (Streamable-UAF über Lock-Grenzen, Thread-Safety und
  > Leak der Legacy-SSE-Session-Map) sind mit dem Umstieg auf 2026-07-28 **gegenstandslos**:
  > Protokoll-Sessions und der SSE-Transport wurden gelöscht, nicht repariert.
- **RAG-Ingestion strikt + atomar**: exakte Vektoranzahl erzwungen (sonst Fehler);
  `IVectorStore.AddBatch` committet ein Dokument in **einer** Transaktion (kein
  Teil-Index).
- **build-demo.sh** baut wieder **alle** Demos (vorher nur 5 MCP-Demos trotz „ALL DEMOS
  OK"); **Demo-Härtung**: Streamable-Demo bindet Loopback, `ask_claude` nur per
  `MCP_ENABLE_ASK_CLAUDE=1` (neue `BindAddress`-Property am Transport).

### Review-Härtung Runde 2 (nach dem 2026-07-28-Umbau, MCP **402** Assertions grün)

Multi-Angle-Review des Umbaus (5 Claude-Angles + Codex als modellfremder Zweitleser):

- **HTTP-Status kam immer als 200**: der Deferral schreibt den SSE-Kopf, bevor das
  Ergebnis existiert. Neu entscheidet `TMcpServer.PreflightRequest` **vor** dem Streamen
  und der Transport antwortet gepuffert mit 400/404 (Spec: MUSS).
- **`OnStreamCall` umging die gesamte Validierung**: ein Hook, der `true` lieferte, bekam
  ungeprüfte Anfragen (kein `_meta`, keine Version) und antwortete ohne `resultType`/
  `serverInfo`. Jetzt läuft der Preflight davor und `FinalizeResponseJson` danach.
- **`McpRequestParams` hängte ein zweites `_meta` an**, wenn der Aufrufer schon eines
  hatte (`TDocVariantData` erlaubt Duplikate, Lookups sehen nur das erste) → der Server
  wies die eigene Client-Anfrage als unvollständig ab. Jetzt echter Merge.
- **`-32602` statt `-32603`** für unbekanntes Tool / unbekannte Ressource / fehlenden
  Namen (`EMcpInvalidParams`); `-32002` gibt es seit dieser Revision nicht mehr.
- **Sentinel `=?base64?…?=` case-sensitiv** (Spec: MUSS exakt lowercase) — sonst würde
  ein legitimer Name, der wie das Sentinel aussieht, still umgeschrieben.
- **`FinalizeResult` verwarf Nicht-Objekt-Payloads still** → leere Erfolgsantwort statt
  Fehler; jetzt Exception. Doppel-Keys via `AddOrUpdateValue` ausgeschlossen.
- **Bootstrapping**: jede `_meta`-Ablehnung nennt jetzt die unterstützte Version — ein
  Erstkontakt-Client kann sie sonst nirgends erfahren (`server/discover` braucht selbst
  gültige `_meta`).
- **Leerer 400 vermieden**: auch Array-Body/fehlendes `method` liefern einen JSON-RPC-
  Fehler, sonst liest die Kompatibilitäts-Probe der Spec „Legacy-Server" und fällt auf
  den entfernten Handshake zurück.
- **Testlücken**: `GetAndDeleteReturn405` sendete nie DELETE; neu `HeaderValidationFailures`
  (9 `-32020`-Fälle inkl. Sentinel-Groß-/Kleinschreibung), `ProtocolErrorsUseHttpStatus`
  (404/400/Array-Body/kein Stream bei Ablehnung), `ConcurrentPosts` (Nebenläufigkeit +
  Keep-Alive nach einem Stream), `RequestParamsMergeExistingMeta`, `ResultMustBeAnObject`,
  `PreflightDecidesHttpStatus`. Test-Helfer nutzen jetzt `McpRequestParams` statt einer
  zweiten, divergierenden Kopie der `_meta`-Regeln.
- **Demos**: `tools/call`/`resources/read` trugen kein `_meta` (liefen also in `-32602`,
  meldeten aber „SUCCESS"); curl-Beispielen fehlten die Pflicht-Header; das DELETE-
  Beispiel und der Banner „2025-03-26" waren veraltet. Die `jsonrpc`-Demo fehlte in
  `build-demo.sh` und wurde nie gebaut.
- **Delphi-Artefakte entfernt**: `.dproj`/`.groupproj` gelöscht (die Projektgruppe
  referenzierte noch die gelöschte SSE-Demo), `.dpr` → `.lpr`. Das Repo ist FPC-only.

## Lizenz / Veröffentlichung

> **Überholt (2026-07-29):** Ein Synopse-PR ist **nicht** mehr das Ziel — LandrixAI
> wird eine **eigenständige FPC-only-Extension** in eigenem Repo. Begründung und Plan:
> [CONCEPT.md §1 / §7](CONCEPT.md). Dieses Dokument ist ein historisches Bau-Log;
> maßgeblich ist CONCEPT.md.

Lizenz bleibt die mORMot-Drei-Lizenz (MPL 1.1 / GPL 2.0 / LGPL 2.1) — die adoptierten
MCP-Units verlangen es. Namespace `mormot.ai.*` bleibt, ist aber bei
[flydev](https://github.com/flydev-fr/mormot.ai) ebenfalls belegt → im Zuge der
Vollintegration (CONCEPT §4) mit ihm abstimmen.
