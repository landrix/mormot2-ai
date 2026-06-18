# DESIGN — `mormot.ai.*`

## Ziel

Eine mORMot-native AI-Erweiterung. Erster Use-Case: **landrix als MCP-Server** —
es stellt Tools/Ressourcen bereit, die ein externer Agent (z. B. Claude Desktop)
über das Model Context Protocol aufruft. Langfristig Synopse-Contribution.

## Aktueller Stand (Phase A abgeschlossen)

Der MCP-Server ist **adoptiert** (mORMot-lizenziert) statt selbst gebaut — Basis:
flydev-fr/mormot2-extensions, auf `mormot.ai.*` umbenannt (Commit-Pin: siehe
`UPSTREAM_BASE`). Build **+ alle Tests + alle Demos grün** (aarch64-linux/FPC 3.2.2).

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

Die adoptierten Tests nutzen mORMots **`TSynTests`** (nicht FPCUnit) — passend zum
Contribution-Ziel; Runner `tests/mcp.tests.dpr`. Aktuell **44 Tests / 187
Assertions** grün (Core + Transporte + Streamable). Neue Tests dort ergänzen.

## Roadmap

- **Phase A** ✓ — flydev-MCP-Server adoptiert, `mormot.ai.*`, Build/Tests/Demos grün.
- **Phase B** — landrix-spezifische MCP-Tools über `TMcpServer.RegisterTool`
  andocken; Auth über die vorhandene mORMot-Auth des Backends.
- **Phase C** ✓ — MCP-Spec auf **2025-11-25** mit Versions-Negotiation
  (`initialize` echot unterstützte Client-Versionen, sonst Fallback = neueste);
  der transportabhängige Patch-Hack wurde entfernt (einheitliches Verhalten).
- **Phase D** — Clean-Room LLM-Client (`mormot.ai.llm.*`): Provider-Treiber +
  Agent-/Tool-Calling-Loop gegen die offiziellen Specs. Architektur-Entscheidungen:
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
  - **Stand**: gebaut + Tests grün (`llm.tests`, `scripts/run-fpc-llm-tests.sh`:
    **51 Assertions**):
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
    Offen: saubere Units + x86_64-linux-`.so` für Docker.
  - **Offen**: lokaler lembed/sqlite-vec-Pfad (s. o.); weitere Provider (Anthropic =
    eigenes Wire); Vision/multimodale Messages (Content-Parts → OCR-Modelle).

## Lizenz / Contribution

Ziellizenz: mORMot-Drei-Lizenz (MPL 1.1 / GPL 2.0 / LGPL 2.1). Namespace
`mormot.ai.*` und CLA/Coding-Style **vorab mit Synopse (Arnaud Bouchez) sowie mit
flydev abstimmen** (Namespace `mormot.ai.*` vs. flydevs `mormot.ext.mcp`).
