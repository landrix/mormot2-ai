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
  - **Stand**: `mormot.ai.llm.types` + `mormot.ai.llm.sse` gebaut, Tests grün
    (`llm.tests`, `scripts/run-fpc-llm-tests.sh`: 18 Assertions — whole/1-byte/
    tool-call). Offen: HTTP-Client (`mormot.ai.llm`: ILlmClient, ChatComplete +
    ChatStream) + Provider-Configs (`mormot.ai.llm.openai`: OpenAI/Ollama/LiteLLM)
    + Agent-/Tool-Calling-Loop (verdrahtet MCP-Tools ↔ Modell).

## Lizenz / Contribution

Ziellizenz: mORMot-Drei-Lizenz (MPL 1.1 / GPL 2.0 / LGPL 2.1). Namespace
`mormot.ai.*` und CLA/Coding-Style **vorab mit Synopse (Arnaud Bouchez) sowie mit
flydev abstimmen** (Namespace `mormot.ai.*` vs. flydevs `mormot.ext.mcp`).
