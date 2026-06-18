# LandrixAI — `mormot.ai.*` Extension (MCP)

Eine **mORMot-native** AI-/LLM-Erweiterung. Erster Use-Case: ein robuster
**MCP-Server** (Model Context Protocol). Die Server-Implementierung wurde aus
[flydev-fr/mormot2-extensions](https://github.com/flydev-fr/mormot2-extensions)
**adoptiert** (mORMot-lizenziert) und auf den `mormot.ai.*`-Namespace umbenannt.

> Status: **Phase A + Spec-Upgrade** — adoptiert, MCP **2025-11-25** mit Versions-
> Negotiation, Build **+ alle Tests grün** (193 Assertions, aarch64-linux/FPC 3.2.2).

## Warum mORMot-nativ

- **Eine Codebasis für FPC *und* Delphi** — mORMot ist dual + cross-platform.
- **Kein Diff-Nachbau-Ballast** — wir folgen den Specs (MCP, JSON-RPC).
- **Robuster MCP-Stack** — `THttpAsyncServer`, WebSockets, RTTI-Schema-Generierung.
- Langfristig **Contribution** an Synopse mORMot.

## Struktur

```
landrixai/
  src/    mormot.ai.mcp[.server|.stdio|.tools]  — Engine, Transporte, Beispiel-Tools
  tests/  test.mcp.core / test.mcp.transports + mcp.tests.dpr (mORMot TSynTests)
  demos/  stdio · http · jsonrpc · sse · streamable (inkl. Claude) + mcp.examples
          demos/_deps/  adoptierte mormot.ext.os (flydev) — Demo-Dependency
  scripts/run-fpc-tests.sh   — Build + Test (WSL/FPC)
  scripts/build-demo.sh      — Demos bauen (alle oder eine)
  _eval/      (gitignored) flydev-Repo-Klon — Referenz + Demos
  DESIGN.md  LICENSE  NOTICE  UPSTREAM_BASE
```

## Bauen & Testen

```bash
# in WSL (nativ aarch64), aus dem Repo-Root:
bash shared/delphi/landrixai/scripts/run-fpc-tests.sh
# VERBOSE=1 für vollen Compiler-/Testlog
```
Verdrahtung wie das Backend: mORMot-Unit-/Static-Pfade aus
`shared/delphi/libs/_git_Synopse2`. Logs unter `bin/fpc/`.

## MCP-Stand

Server spricht MCP **2025-11-25** mit **Versions-Negotiation**: `initialize` echot
die vom Client angefragte Version, wenn unterstützt (2024-11-05 / 2025-03-26 /
2025-06-18 / 2025-11-25), sonst Fallback auf die neueste. Transporte: stdio, HTTP,
SSE, Streamable HTTP, in-process. Tools: `TMcpServer.RegisterTool(IMcpTool)`,
Input-Schema automatisch via RTTI aus typisiertem Record (`TMcpToolBase<T: record>`).

## Herkunft & Lizenz

MCP-Server-Units adoptiert von flydev-fr/mormot2-extensions (MPL/GPL/LGPL); die
Original-Unit-Header bleiben erhalten. Details: [NOTICE](NOTICE),
[LICENSE](LICENSE), [DESIGN.md](DESIGN.md). Vor einem Synopse-Contribution-Push
mit flydev abstimmen (Namespace `mormot.ai.*` vs. flydevs `mormot.ext.mcp`).

## Nächste Schritte

1. landrix-spezifische MCP-Tools andocken (über die `TMcpServer`-Registry).
2. Clean-Room LLM-Client (`mormot.ai.llm`) — gegen die offiziellen Provider-/MCP-
   Specs implementiert (interne Clean-Room-Prozessdoku separat, nicht eingecheckt).

✓ Demos übernommen (stdio/http/jsonrpc/sse/streamable inkl. Claude) — alle bauen grün.
