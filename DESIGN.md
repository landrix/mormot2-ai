# DESIGN — `mormot.ai.*` (MCP-Server)

## Ziel

Eine mORMot-native AI-Erweiterung. Erster Use-Case: **landrix als MCP-Server** —
es stellt Tools/Ressourcen bereit, die ein externer Agent (z. B. Claude Desktop)
über das Model Context Protocol aufruft.

## Clean-Room-Politik (verbindlich)

- MakerAI ist **nur konzeptuelle Vorlage** (welche Features existieren, wie ist
  eine Tool-Registry geformt). Es wird **kein MakerAI-Code übernommen**.
- Implementiert wird gegen die **offiziellen Quellen**:
  - MCP-Spec, Revision **2025-11-25** — https://modelcontextprotocol.io
  - JSON-RPC 2.0 — https://www.jsonrpc.org/specification
- Grund: rechtlich sauber (keine MIT-Bindung an MakerAI) und aufnahmefähig als
  mORMot-Contribution (Synopse nimmt keine fremd-lizenzierten Schnipsel).

## Architektur — transport-neutral

Protokoll-Logik strikt vom Transport getrennt (wie `ILandrixContext` im Backend):

```
mormot.ai.mcp.types            JSON-RPC-/MCP-Typen, Envelope-Builder/-Parser
mormot.ai.mcp.server           Engine: Tool-/Resource-Registry + JSON-RPC-
                               Dispatch (initialize, tools/list, tools/call).
                               JSON rein -> JSON raus. KEIN Netz, KEIN stdio.
mormot.ai.mcp.transport.stdio  stdin/stdout-Loop (Test-Vehikel, lokal)
mormot.ai.mcp.transport.http   Streamable HTTP auf mormot.net.server (Produktion)
```

Tool-Vertrag: Interface `IMcpTool` (`GetName`/`GetDescription`/`GetInputSchema`/
`Execute`).

## Transport-Reihenfolge

1. **stdio** zuerst — einfachstes E2E, perfekt per FPCUnit testbar ohne Netz.
2. **Streamable HTTP** direkt danach — der echte landrix-Produktions-Transport
   (hinter Caddy), auf `mormot.net.server` (+ `mormot.net.ws` für Streaming).
3. SSE-only (alt) wird **nicht** gebaut (Spec-`legacy`).

## Phasen (jede mit grünen FPCUnit-Tests)

- **Phase 0** — Skelett + `mormot.ai.mcp.types` + Tests. *(dieser Stand)*
- **Phase 1** — Engine (Registry + Dispatch + initialize/tools.list/tools.call)
  + Demo-Tool + Engine-Tests (JSON rein/raus, Fehlercodes -32601/-32602).
- **Phase 2** — `transport.stdio` + E2E-Test.
- **Phase 3** — `transport.http` (Streamable HTTP).
- **Phase 4** — resources/*, RTTI-Auto-Schema (via `mormot.core.rtti`), Auth über
  die vorhandene mORMot-Auth des Backends, progress-Notifications.

## Test-Politik

FPCUnit wie im Backend (`{$mode delphi}`, `RegisterTest`, `consoletestrunner`).
Da die Engine transport-neutral ist, wird sie direkt getestet — die Tests sind
die **Spezifikation**, gegen die wir später Spec-Updates abgleichen.

## Lizenz / Contribution

Ziellizenz: mORMot-Drei-Lizenz (MPL 1.1 / GPL 2.0 / LGPL 2.1). Namespace
`mormot.ai.*` und CLA/Coding-Style **vorab mit Synopse (Arnaud Bouchez)
abstimmen** — siehe `docs/forum-post-mormot-ai-extension.md`.
