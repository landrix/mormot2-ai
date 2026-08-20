# mormot2-ai — the `mormot.ai.*` extension for mORMot2 (FPC)

A **mORMot-native** AI/LLM extension for **Free Pascal**: an MCP server, LLM clients,
embeddings, vector stores and RAG. Built on mORMot2 — its HTTP stack, JSON, RTTI and
crypto are the foundation, not a dependency we tolerate.

> **Status:** phases A · B · C · D complete. MCP speaks revision **2026-07-28**
> (stateless) and only that one. Build, tests and demos green on aarch64-linux/FPC 3.2.2:
> **892 assertions** MCP suite, **273 assertions** LLM suite. Open: phase E
> (layering / merge / backend binding) — see [CONCEPT.md](CONCEPT.md).

## Why a standalone extension

- **Not an upstream mORMot PR** (decided 2026-07-29). A PR would commit us to mORMot's
  dual-compiler promise (FPC *and* Delphi) and thereby rule out **neural-api** (FPC-only)
  as an inference backend. mORMot2 stays the **foundation**, not the delivery target.
  Rationale: [CONCEPT.md §1](CONCEPT.md).
- **FPC-only** — no Delphi branch, no `{$ifdef}` double maintenance.
- **Spec-driven, not diff-driven** — we follow the published specs (MCP, JSON-RPC,
  provider wires) rather than chasing another library's changes.
- **License stays MPL/GPL/LGPL** — the adopted MCP units require it, and it composes with
  MIT and LGPL-with-linking-exception.

## Layout

```
mormot2-ai/
  src/
    mormot.ai.mcp[.server|.stdio|.tools]     MCP: core, transports, sample tools
    mormot.ai.llm[.types|.openai|.anthropic]  LLM clients (+ .sse, .structured)
    mormot.ai.agent[.mcp]                     agent / tool-calling loop
    mormot.ai.embeddings, .embed.{provider,lembed}
    mormot.ai.vectorstore[.sqlitevec]
    mormot.ai.rag[.tool]
  tests/    mcp.tests.lpr · llm.tests.lpr (mORMot TSynTests)
  demos/    stdio · http · jsonrpc · streamable · llm/* · rag/*
  docs/specs/mcp-2026-07-28/   verbatim copy of the protocol spec (see below)
  vendor/   mormot.ext.os (adopted) · models, sqlite-ext (gitignored, not in git)
  scripts/  run-fpc-tests.sh · run-fpc-llm-tests.sh · build-demo.sh · x64-ext-verify.sh
  CONCEPT.md  DESIGN.md  LICENSE  NOTICE  SECURITY.md  UPSTREAM_BASE
```

## A server in thirty lines

A tool is a class over a typed record. The input schema is generated from the record's
RTTI — there is no hand-written JSON Schema, so it cannot drift from the code:

```pascal
uses
  mormot.core.base, mormot.core.rtti, mormot.core.variants,
  mormot.ai.mcp, mormot.ai.mcp.stdio;

type
  TAddParams = record
    a: integer;
    b: integer;
  end;

  TAddTool = class(TMcpToolBase<TAddParams>)
  protected
    function ExecuteTyped(const aParams: TAddParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  end;

function TAddTool.ExecuteTyped(const aParams: TAddParams;
  const aAuthCtx: TMcpAuthContext): variant;
begin
  result := _ObjFast(['sum', aParams.a + aParams.b]);
end;

var
  server: TMcpServer;
  transport: TMcpStdioTransport;
  tool: IMcpTool;
begin
  // FPC has no RTTI for plain records — declare the fields once, up front.
  if not RecordHasFields(TypeInfo(TAddParams)) then
    Rtti.RegisterFromText(TypeInfo(TAddParams), 'a,b:integer');

  server := TMcpServer.Create('DemoServer', '1.0');
  try
    tool := TAddTool.Create('add', 'Add two numbers');
    server.RegisterTool(tool);
    server.Start;

    transport := TMcpStdioTransport.Create(server);
    try
      transport.Start;
      while transport.IsActive do
        Sleep(100);
    finally
      transport.Free;
    end;
  finally
    server.Free;
  end;
end.
```

That `Rtti.RegisterFromText` line is the one thing that catches everyone: under FPC a plain
record carries no field RTTI, so the schema generator has nothing to read and the tool
shows up with an empty input schema. Register the record once at startup and it works.

Swap `TMcpStdioTransport` for the HTTP or Streamable transport and nothing else changes —
the server and its tools do not know which transport they are behind. Full examples in
[`demos/`](demos/).

## Build & test

```bash
# from this repo, on Linux/WSL:
bash scripts/run-fpc-tests.sh        # MCP suite
bash scripts/run-fpc-llm-tests.sh    # LLM suite
bash scripts/build-demo.sh           # all demos (or pass one, e.g. llm/llm-agent.lpr)
# VERBOSE=1 for the full compiler/test log; logs land in bin/fpc/
```

**Requirements:** FPC 3.2.2 or newer and a mORMot2 source tree. Verified on aarch64-linux
and x86_64-linux; Windows builds from the same sources (mORMot2 handles the platform
split), but the build scripts are bash — use WSL or call the compiler directly. Lazarus is
optional; `.lpi` files exist for the test projects, the scripts do not need the IDE.

The two native extensions used by RAG (`sqlite-vec`, `lembed`) are **optional**: without
them the vector-store tests skip and everything else runs.

## Using it in your own project

There is no package file to install — add two source directories and mORMot2's, and use
the units. This is what the build scripts do; the flags below are lifted from
[`scripts/run-fpc-tests.sh`](scripts/run-fpc-tests.sh):

```bash
MORMOT2=/path/to/mORMot2
LAI=/path/to/mormot2-ai

fpc -MDelphi -Sci -Ci -O2 \
  -Fi"$MORMOT2/src;$MORMOT2/src/core;$MORMOT2/src/net" \
  -Fu"$LAI/src;$MORMOT2/src/app;$MORMOT2/src/core;$MORMOT2/src/crypt;$MORMOT2/src/db;$MORMOT2/src/lib;$MORMOT2/src/net;$MORMOT2/src/orm;$MORMOT2/src/rest;$MORMOT2/src/soa" \
  -Fl"$MORMOT2/static/$(fpc -iTP)-linux" \
  your-program.lpr
```

`-MDelphi` is not optional — the sources use Delphi-mode syntax, as mORMot2 itself does.
`-Fl` points at mORMot2's static libraries (the bundled SQLite among them); drop it only if
you use no unit that needs them.

In Lazarus, add `$(LAI)/src` to the project's *Other unit files* and mORMot2 per its own
instructions. Consuming this repository as a **git submodule** is the arrangement the
default paths assume — see "Locating mORMot2" above.

## Maturity

Not everything here carries the same weight. "Live" means it was run against a real
provider or runtime, not only against tests.

| Area | State | Notes |
|---|---|---|
| MCP core, JSON-RPC, tool/resource registry | **stable** | 892 assertions, hardened over five review rounds |
| Transports: stdio, HTTP, Streamable HTTP, in-process | **stable** | concurrency covered (parallel clients, keep-alive reuse) |
| MCP authorization (resource server) | **stable** | opt-in per spec (`OPTIONAL`), closed once a verifier is set; the verifier itself is supplied by the consumer |
| LLM client, OpenAI wire | **stable, live** | also covers LiteLLM and Ollama-compatible endpoints |
| Anthropic driver | **stable, live** | streaming and tool loop verified against a real model |
| Agent / tool-calling loop, structured output, vision | **stable, live** | |
| Embeddings via provider API | **stable** | |
| Embeddings via `lembed` (local GGUF) | **works, needs a native library** | process-wide singleton per model; not re-entrant, calls are serialised |
| Vector store `sqlite-vec` | **works, needs a native library** | real-extension test is env-gated (`SQLITE_EXT_DIR`) |
| RAG (ingestion, retrieval, RAG-as-tool) | **works** | ingestion is strict and atomic; the rollback test still needs the runtime |
| pgvector backend | **not built** | phase E — the `IVectorStore` seam exists, the implementation does not |
| Ollama embedder | **not built** | phase E |
| neural-api (native Pascal inference) | **not evaluated** | the strategic goal: it would remove every foreign `.so` — see [CONCEPT.md §5](CONCEPT.md) |

Two things follow from the "needs a native library" rows: those features are **optional at
runtime** (missing library ⇒ skip, never crash), and the libraries are **not in git** —
`vendor/models` and `vendor/sqlite-ext` are ignored and have to be provided locally.

**Locating mORMot2:** the scripts expect a mORMot2 source tree (`src/`, `static/`). With
no configuration they assume the submodule case — `<consumer>/shared/delphi/libs/_git_Synopse2`,
three levels above this repo. For a **standalone clone**, point at it explicitly:

```bash
MORMOT2_ROOT=/path/to/mORMot2 bash scripts/run-fpc-tests.sh
```

## MCP — revision 2026-07-28, and only that one

The 2026-07-28 revision is the largest change since MCP launched, and it is **stateless**:
`initialize`, protocol sessions (`Mcp-Session-Id`), request batching, SSE resumability and
the legacy HTTP+SSE transport are **gone**. Every request carries its own protocol version
and client capabilities in `params._meta`; `server/discover` replaces the handshake.

We deleted the removed machinery rather than keeping it behind a negotiation layer. There
is **no multi-version support** — a greenfield project has nothing to be compatible with,
and a second code path is a second thing to get wrong.

What that buys, beyond simplicity: session expiry, the use-after-free surface around the
streamable transport and the legacy SSE session map all stopped being problems, because
the state they guarded no longer exists.

Implemented, each built against the spec text: `server/discover`, `_meta` transport of the
protocol version, tools/resources/prompts, resource templates and completion, pagination
with a deterministic `tools/list` order, subscriptions, MRTR, cacheable results
(`ttlMs`/`cacheScope`), progress notifications with server-enforced monotonicity,
extensions, OTel `_meta` keys, `x-mcp-header`, JSON Schema 2020-12 in `inputSchema`.

**Deliberately not implemented:** `notifications/message` (logging) is deprecated in
2026-07-28 — "new implementations SHOULD NOT adopt it".

Transports: stdio, HTTP, Streamable HTTP, in-process. Tools register via
`TMcpServer.RegisterTool(IMcpTool)`; the input schema is generated **from RTTI** out of a
typed record (`TMcpToolBase<T: record>`), so there is no hand-written JSON Schema to drift.

**Authorization is opt-in, and closed once it is on.** The core is an OAuth 2.1 resource
server (RFC 9728 metadata, token validation before dispatch, audience binding, 401/403
challenges); the seam is `IMcpTokenVerifier`. Authorization is `OPTIONAL` in the
specification and the core follows that: **with no verifier set the server is open**, so any
HTTP deployment has to set one (stdio SHOULD NOT use it — credentials come from the
environment there). What the core does guarantee is that authorization can never be half on:
`Start` refuses a verifier without an `AuthResource` or without `AuthorizationServers`, and a
request a configured verifier does not accept is refused before dispatch, with a zeroed
context (`IsAuthenticated=false`, no scopes) so a tool that gates on identity fails closed.
Found a path around a configured verifier? [SECURITY.md](SECURITY.md).

### The spec is vendored

A verbatim copy of the specification lives under
[`docs/specs/mcp-2026-07-28/`](docs/specs/mcp-2026-07-28/) — 31 files including the full
schema reference. It is there so reviews and audits can **cite the normative text instead
of recalling it**. In protocol work the most common defect is not in the code, it is the
reviewer inventing a requirement, or remembering an older revision — and both sound
plausible. See the [README](docs/specs/mcp-2026-07-28/README.md) there for provenance,
license and how to update.

## LLM client and provider drivers

A clean-room client over provider-neutral records (`mormot.ai.llm.types`). The **OpenAI
Chat Completions wire is the lingua franca**: one `TLlmClient` covers OpenAI, LiteLLM and
Ollama-compatible endpoints. A provider whose wire genuinely differs gets a **native
driver** behind the same `ILlmClient` seam:

- **Anthropic** (`mormot.ai.llm.anthropic`, `TAnthropicClient`): the native Messages API —
  top-level `system`, mandatory `max_tokens`, `input_schema`/`tool_use`/`tool_result`,
  `x-api-key` + `anthropic-version`, event-typed SSE. The agent loop, RAG and structured
  output stay provider-agnostic; only the `ILlmClient` instance changes.

**Vision:** `TLlmMessage.Images` carries attachments (`LlmImageMessage`/`LlmImageBase64`/
`LlmImageUrl`); both wires serialise them (OpenAI `image_url` including base64 `data:` URIs,
Anthropic an `image` block with a typed `source`; an empty `MediaType` falls back to
`image/png`).

**Structured output** and **embeddings** run through the same seam; RAG combines
`IEmbedder` with `IVectorStore` (sqlite-vec today, pgvector planned in phase E).

### Tests

The Anthropic driver test (`tests/test.llm.anthropic.pas`) is **hermetic** — no network, no
API key. It checks request construction (system hoisting, `input_schema`, tool round-trip),
response parsing (text/tool/usage, stop-reason mapping) and SSE decoding against canned
payloads, and runs as part of the suite.

**Native libraries are handled defensively.** `sqlite-vec` and `lembed` are external
`.so`/`.dll` files; the suite must catch the dynamic load and **skip cleanly** when they are
missing, never crash (see the env-gated `VectorStoreKeyedOps` test). That constraint goes
away if **neural-api** (native Pascal inference, [CONCEPT.md §5](CONCEPT.md)) replaces the
foreign libraries.

### Running the demos live

Provider configuration comes from the environment (`demos/.env`, copied from
[`demos/.env.sample`](demos/.env.sample)) — **API keys only from the environment**, never
committed, never logged:

```bash
bash scripts/build-demo.sh llm/llm-anthropic.lpr
set -a; . demos/.env; set +a      # ANTHROPIC_API_KEY[/_MODEL]
bin/fpc/demos/llm-anthropic
```

`llm-anthropic` is a line-for-line rebuild of the OpenAI tool-loop demo (`llm-agent`): the
same `TLlmAgent` and toolbox, only `TAnthropicClient` instead of `TLlmClient` — which is
what makes the lingua-franca seam a claim you can check rather than one you have to believe.

## Origin and license

The MCP server units were **adopted** from
[flydev-fr/mormot2-extensions](https://github.com/flydev-fr/mormot2-extensions)
(mORMot-licensed) and renamed into the `mormot.ai.*` namespace; the original per-unit
headers are retained. Details: [NOTICE](NOTICE), [LICENSE](LICENSE), [DESIGN.md](DESIGN.md).

The vendored MCP specification under `docs/specs/` is **not** part of this work and not
covered by the three-license above — it keeps its own upstream license alongside it.

### Relationship to flydev — code adoption, not a fork

From [flydev-fr/mormot.ai](https://github.com/flydev-fr/mormot.ai) we adopt **code only**:
no fork, no shared repository, no upstream PRs. Our `mormot.ai.*` namespace **stays** even
where it collides with theirs — harmless, because the repositories are independent.

So that their later changes remain mergeable by diff, their commit is **pinned** in
[UPSTREAM_BASE](UPSTREAM_BASE).

> **Rule: whoever ports from upstream bumps `UPSTREAM_BASE` in the same commit.** Without
> it the diff base is lost and the next round has to guess.

## Roadmap

Only **phase E** is open — layering, the flydev merge and backend binding. See
[CONCEPT.md](CONCEPT.md), the single source of truth for open points.

- ✓ **Phase A** — MCP server adopted, namespace aligned, build/tests/demos green.
- ✓ **Phase B** — a real consumer on the registry: Landrix registers its tools and supplies
  the `IMcpTokenVerifier`. The binding lives outside this repo.
- ✓ **Phase C** — protocol moved to 2026-07-28 (stateless); removed machinery deleted.
- ✓ **Phase D** — clean-room LLM client: OpenAI wire + Anthropic, agent/tool loop,
  embeddings/RAG, vision.

## A note on the documentation language

This README is English. The two design documents behind it — [CONCEPT.md](CONCEPT.md)
(decisions, open points, licensing strategy) and [DESIGN.md](DESIGN.md) (architecture,
protocol details, review history) — are **written in German**, as are most in-code
comments.

That is a working-language artefact, not a statement about who the code is for. Both files
are dense and worth translating; until that happens, machine translation handles them well
because they are structured prose with code identifiers left in English. If you hit a
passage that matters and does not survive translation, open an issue and quote it — that is
a faster path to a fix than a full rewrite nobody has scheduled.

New documentation in this repository should be written in English.
