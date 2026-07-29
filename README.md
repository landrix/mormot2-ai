# LandrixAI — `mormot.ai.*` Extension (MCP)

Eine **mORMot-native** AI-/LLM-Erweiterung. Erster Use-Case: ein robuster
**MCP-Server** (Model Context Protocol). Die Server-Implementierung wurde aus
[flydev-fr/mormot2-extensions](https://github.com/flydev-fr/mormot2-extensions)
**adoptiert** (mORMot-lizenziert) und auf den `mormot.ai.*`-Namespace umbenannt.

> Status: **Phase A + Spec-Upgrade** — adoptiert, MCP **2025-11-25** mit Versions-
> Negotiation, Build **+ alle Tests grün** (193 Assertions, aarch64-linux/FPC 3.2.2).

## Warum mORMot-basiert — und warum eigenständig

- **Eigene Extension, kein mORMot-PR** (Entscheidung 2026-07-29): ein Upstream-PR
  würde uns auf mORMots Dual-Compiler-Anspruch (FPC + Delphi) festlegen und damit
  **neural-api** (FPC-only) als Inferenz-Backend ausschließen. mORMot2 bleibt die
  **Grundlage**, nicht das Abgabeziel. Begründung: [CONCEPT.md §1](CONCEPT.md).
- **FPC-only** — kein Delphi-Zweig, keine `{$ifdef}`-Doppelpflege.
- **Kein Diff-Nachbau-Ballast** — wir folgen den Specs (MCP, JSON-RPC).
- **Robuster MCP-Stack** — `THttpAsyncServer`, WebSockets, RTTI-Schema-Generierung.
- **Lizenz bleibt MPL/GPL/LGPL** (die adoptierten MCP-Units verlangen es; verträglich
  mit MIT und LGPL+Linking-Exception).

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

> **Geplant: Sprung auf 2026-07-28 (Greenfield).** Die neue Spec macht MCP stateless
> und entfernt `initialize`, Protokoll-Sessions/`Mcp-Session-Id`, SSE-Resumability und
> den HTTP+SSE-Transport. Projektentscheidung: **nur 2026-07-28**, alles Entfernte
> wird gelöscht (keine Multi-Version-Negotiation). Details + Bauliste:
> [CONCEPT.md §6](CONCEPT.md).

**Heute (Alt-Stand):** Server spricht MCP **2025-11-25** mit **Versions-Negotiation**:
`initialize` echot die vom Client angefragte Version, wenn unterstützt (2024-11-05 /
2025-03-26 / 2025-06-18 / 2025-11-25), sonst Fallback auf die neueste. Transporte:
stdio, HTTP, SSE, Streamable HTTP, in-process. Tools:
`TMcpServer.RegisterTool(IMcpTool)`, Input-Schema automatisch via RTTI aus
typisiertem Record (`TMcpToolBase<T: record>`).

## LLM-Client (`mormot.ai.llm`) & Provider-Treiber

Clean-Room-LLM-Client mit provider-neutralen Records (`mormot.ai.llm.types`). Das
**OpenAI Chat Completions-Wire ist die Lingua franca** (`TLlmClient` deckt OpenAI /
LiteLLM / Ollama ab); ein Provider mit echt abweichendem Wire bekommt einen
**nativen Treiber** hinter derselben `ILlmClient`-Naht:

- **Anthropic** (`mormot.ai.llm.anthropic`, `TAnthropicClient`): native Messages-API
  (`system` top-level, `max_tokens` Pflicht, `input_schema`/`tool_use`/`tool_result`,
  `x-api-key`+`anthropic-version`, event-getypte SSE). Agent-Loop, RAG und Structured-
  Output bleiben dadurch provider-agnostisch — nur die `ILlmClient`-Instanz wechselt.

**Vision/multimodal:** `TLlmMessage.Images` (`LlmImageMessage`/`LlmImageBase64`/
`LlmImageUrl`) trägt Bild-Anhänge; beide Wires serialisieren sie (OpenAI `image_url`
inkl. base64-`data:`-URI, Anthropic `image`-Block mit typisiertem `source`; leerer
`MediaType` fällt auf `image/png` zurück). Demo `demos/llm/llm-vision.dpr` ist
provider-agnostisch (`VISION_PROVIDER=openai|anthropic`, Bild via `VISION_IMAGE_B64`/
`_MEDIA` oder `VISION_IMAGE_URL`) — **live** gegen `gpt-4o-mini` **und** `claude-opus-4-8`
geprüft (beide erkennen denselben inline-base64-Kreis).

### Tests (LLM-Suite, inkl. Anthropic)

Eigener Runner — baut + fährt `llm.tests.dpr` (alle LLM-Suiten: SSE, Client, Agent,
Agent-MCP, **Anthropic**, Structured, RAG, RAG-Tool):

```bash
# in WSL, aus dem Repo-Root:
bash shared/delphi/landrixai/scripts/run-fpc-llm-tests.sh
# VERBOSE=1 für vollen Compiler-/Testlog
```

Der Anthropic-Treiber-Test (`tests/test.llm.anthropic.pas`) ist **hermetisch** —
keine Netzwerkverbindung, kein API-Key nötig: er prüft Request-Bau (System-Hoisting,
`input_schema`, Tool-Round-Trip), Response-Parsing (Text/Tool/Usage, stop_reason-
Mapping) und das SSE-Decoding (Text- + Tool-Stream) gegen canned Payloads. Er läuft
automatisch als Teil der Suite mit.

### Demos live fahren

Provider-Config kommt aus der Umgebung (`demos/.env`, kopiert aus
[`demos/.env.sample`](demos/.env.sample)) — **API-Key NUR aus Env**, nie eingecheckt
oder geloggt:

```bash
bash shared/delphi/landrixai/scripts/build-demo.sh llm/llm-anthropic.dpr
set -a; . shared/delphi/landrixai/demos/.env; set +a   # ANTHROPIC_API_KEY[/_MODEL]
shared/delphi/landrixai/bin/fpc/demos/llm-anthropic
```

`llm-anthropic` ist der OpenAI-Tool-Loop-Demo (`llm-agent`) 1:1 nachgebaut — gleicher
`TLlmAgent` + Toolbox, nur `TAnthropicClient` statt `TLlmClient` (beweist die Lingua-
franca-Naht). OpenAI-/Ollama-Demos nutzen weiter die `LLM_*`-Vars derselben `.env`.

## Herkunft & Lizenz

MCP-Server-Units adoptiert von flydev-fr/mormot2-extensions (MPL/GPL/LGPL); die
Original-Unit-Header bleiben erhalten. Details: [NOTICE](NOTICE),
[LICENSE](LICENSE), [DESIGN.md](DESIGN.md).

## Veröffentlichung als eigenständige Extension

**Kein Synopse-PR** (Begründung: [CONCEPT.md §1](CONCEPT.md)). Stattdessen wird der
`mormot.ai.*`-Code aus dem Landrix-Monorepo in ein **eigenes Repository** ausgelöst
(`git subtree split` mit Historie) und von Landrix als **Submodul** eingebunden —
dasselbe Muster wie `_git_Synopse2`. Vollständiger Plan inkl. Checkliste:
[CONCEPT.md §7](CONCEPT.md).

Weiter zu beachten:

- **Verhältnis zu flydev — Code-Übernahme, kein gemeinsames Repo**: Von
  [flydev-fr/mormot.ai](https://github.com/flydev-fr/mormot.ai) wird **nur Code**
  übernommen (kein Fork, keine PRs). Der Namespace `mormot.ai.*` **bleibt**, obwohl
  flydev denselben nutzt — folgenlos, weil die Repos unabhängig sind. Damit spätere
  flydev-Änderungen per Diff einarbeitbar bleiben, ist ihr **Commit-Stand gepinnt**:
  [UPSTREAM_BASE](UPSTREAM_BASE). **Regel: wer portiert, bumpt UPSTREAM_BASE im selben
  Commit.**
- **`codenav/` gehört nicht zur Extension** — es ist ein Landrix-Entwicklerwerkzeug
  und bleibt im Monorepo (CONCEPT §7).
- **Native-Libs defensiv behandeln**: `sqlite-vec`/`lembed` sind externe `.so`/`.dll`.
  Die Testsuite muss den Dynamic-Load **defensiv** abfangen — fehlt die Lib, sauber
  **skippen**, nie abstürzen (so gelöst im env-gated `VectorStoreKeyedOps`-Test).
  Mittelfristig entfällt das Problem, wenn **neural-api** (native Pascal-Inferenz,
  CONCEPT §5) die Fremd-`.so` ersetzt.

## Nächste Schritte

1. landrix-spezifische MCP-Tools andocken (über die `TMcpServer`-Registry).
2. Clean-Room LLM-Client (`mormot.ai.llm`) — gegen die offiziellen Provider-/MCP-
   Specs implementiert (interne Clean-Room-Prozessdoku separat, nicht eingecheckt).

✓ Demos übernommen (stdio/http/jsonrpc/sse/streamable inkl. Claude) — alle bauen grün.
