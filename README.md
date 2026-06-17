# LandrixAI — `mormot.ai.*` Extension (MCP)

Eine **mORMot-native** AI-/LLM-Erweiterung, von Grund auf gebaut — **kein Port**
von MakerAI. Erstes Ziel: ein robuster **MCP-Server** (Model Context Protocol)
auf Basis von `mormot.net.server`/`mormot.net.ws` und `mormot.core.json`.

> Status: **Phase 0** (Skelett). Lauffähig nach Build-Wireup (siehe unten).

## Warum mORMot-nativ statt Port

- **Eine Codebasis für FPC *und* Delphi** — mORMot ist von Haus aus dual und
  cross-platform (Win/Linux, x64/ARM). Kein `{$IFDEF}`-Wust, keine Indy-Altlast.
- **Kein Diff-Nachbau-Ballast** — wir folgen den **Specs** (MCP, JSON-RPC),
  nicht der Entwicklung einer Fremd-Lib.
- **Bessere Qualität bei MCP** — mORMots Server/WebSocket-Stack ist robuster als
  MakerAIs experimenteller Indy-SSE-Transport.
- Langfristig als **Contribution** an Synopse mORMot gedacht (Namespace
  `mormot.ai.*`).

## Struktur

```
landrixai/
  src/      Lib-Units (mormot.ai.*)
  tests/    FPCUnit-Suite + Runner (LandrixAiTestRunner.lpr)
  docs/     Forum-Beitrag, Design-Notizen
  _upstream/  (gitignored) read-only MakerAI-Klon, nur Konzept-Nachschlag
  DESIGN.md  Architektur, Clean-Room-Politik, Phasenplan
  LICENSE  NOTICE  UPSTREAM_BASE
```

## Bauen & Testen (Wireup ausstehend)

Phase 0 liefert Quellcode + Tests. Der FPC-Build braucht noch die
mORMot-Include-Pfade (`shared/delphi/libs/_git_Synopse2/src/**`). Geplant analog
zum Backend (`backend/scripts/run-fpc-tests.*`): ein `.lpi` für
`tests/LandrixAiTestRunner.lpr` mit den mORMot-Unit-Pfaden, dann

```
LandrixAiTestRunner -a --format=plain
```

## MCP-Stand

Implementiert gegen MCP-Revision **2025-11-25** (JSON-RPC 2.0). Erste Methoden:
`initialize`, `tools/list`, `tools/call`.
