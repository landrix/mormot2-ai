# LandrixAI — Concept (`mormot.ai.*`)

> **Purpose of this document.** Architecture guardrail for the LandrixAI package:
> *what* belongs in which layer and *why*. Single source of truth for the
> layer boundary between **generic mORMot code**, the **LandrixAI package** and the
> **Landrix backend** — and for the upcoming **merge** with
> [flydev's `mormot.ai`](https://github.com/flydev-fr/mormot.ai) (§4), the choice of the
> **inference backend** (§5) and the **MCP jump to 2026-07-28** (§6).
>
> The original brainstorming version (a ChatGPT dialogue) is condensed into
> [docs/Feature-LandrixAI-Agent.md](../../../docs/Feature-LandrixAI-Agent.md)
> — there the agent is described as a backend feature.

## 1. What LandrixAI is (and what it is not)

LandrixAI is a **mORMot-based AI/LLM toolbox** in the namespace `mormot.ai.*`,
**FPC first** (Linux/Windows/macOS, x64/aarch64) and **Delphi 7 to 13** (Win32/Win64),
published as a **standalone extension in its own repository**.

- **Own repository instead of a mORMot PR** (decision 2026-07-29). The extension does not
  go to Synopse as a pull request. mORMot2 remains the **foundation** (base units, HTTP,
  JSON, SQLite, crypto); we just do not deliver into it. Reasons: our own pace, our own
  platform goals (CUDA, engine) and MCP 2026-07-28 without regard for older protocol
  revisions. The original reason, being able to use neural-api as an FPC-only backend,
  fell away with our own engine (§5).
- **Compilers: FPC and Delphi 7 to 13**, the same range as mORMot2.
  - **FPC** is the reference and the gate: `scripts/run-fpc-*.sh`, fixed to FPC 3.2.4
    (since 2026-09-30, before that 3.2.2; the scripts abort on any other version, see
    `scripts/fpc-env.sh`). FPC 3.2.2 must build as well (Windows and Docker of the
    consumer).
  - **Delphi** is built and tested at both ends: Delphi 13 (Win64,
    `scripts/run-delphi-tests.ps1`) and Delphi 7 (Win32, `scripts/run-delphi7-tests.ps1`).
    mORMot2 supports the versions in between; we do not build them separately. A break
    there counts as a bug, but no run guards against it.
  - What only works under FPC (the POSIX process start in `vendor/mormot.ext.os`) stays
    fenced off with `{$ifdef FPC}` and is missing under Delphi. Do not force a duplicate
    implementation.
  - History: FPC-only (2026-07-29), then Delphi 13 (2026-09-10), then Delphi 7
    (2026-10-14, rework in [docs/native-inference.md](docs/native-inference.md#delphi-7)).
  - The Delphi range applies only to the extension. At the consumer (the Landrix platform)
    the backend and tools stay FPC-only; its shared client builds on its own with
    FPC and Delphi 13.
- **Code in the style of mORMot2** (binding, 2026-10-14): the extension should read and
  build like a mORMot2 unit, so that it does not look foreign next to mORMot and supports
  the same compilers. Two parts:

  **Like mORMot2** (preferred patterns, as mORMot itself mostly uses them):
  - **Unit header**: `/// Title`, then `// - …` lines with the license line
    `// - licensed under a MPL/GPL/LGPL three license - see LICENSE`, then
    `{$I mormot.defines.inc}`. Longer units are divided into `{ ************ Section }`
    and name the sections in an overview block below the `unit` line.
  - **mORMot types and building blocks** before their RTL counterparts:
    - `RawUtf8` instead of `string`, `PtrInt` for indexes;
    - JSON via `TDocVariantData`/`_ObjFast`;
    - text with `FormatUtf8`;
    - `TSynDictionary`, `TSynThread`, `TSynTempBuffer`, `TLightLock`.
  - **Exceptions**: own classes derived from `ESynException` (project standard), with
    `CreateUtf8`/`RaiseUtf8`, or `CreateU` for fixed messages.
  - **Naming**: fields `f…`, classes `T…`, pointers `P…`, exceptions `E…`;
    `result`/`exit` in lower case, two-space indent, `begin` on its own line.
  - **Docs**: `///` before public symbols, details as a `// - …` list,
    comments in English.

  **Documentation language** (decision 2026-10-19): it follows its readers. Everything
  users and contributors read is English: `README.md`, `SECURITY.md`, `CONCEPT.md`, the
  READMEs under `tests/data` and `tools`. The working documents for the maintainers and
  their agents stay German: `docs/native-inference.md`, `docs/backlog.md`,
  `docs/native-embeddings.md`, `docs/distributed-execution.md`, `docs/ideas.md` and
  `DESIGN.md` (mostly a log of review rounds). Code comments are English.

  **Stricter than mORMot2** (landrixai decisions, because of Delphi 7 and readability):
  - **Non-generic core**: no generics in the library, except as an optional
    typed wrapper under `HASGENERICS` over a non-generic core (example:
    `TMcpToolBase<T>` over `TMcpRecordTool`).
  - No anonymous methods, no inline variables, no array constructors.
  - No `exit(value)` (Delphi 7). mORMot uses it occasionally.
  - No `with` (readability: every identifier shows what it belongs to). mORMot itself uses
    `with` often, more than 200 times in trunk.
  - Parameters always with `a…`. Exempt are signatures given from outside
    (mORMot's `Ctxt` and server constructors, `TStream` overrides, SQLite callbacks)
    and small helpers in the `implementation` part with short names (`c`, `v`, `q`).
    mORMot mostly writes parameters without a prefix (`Data`, `Len`); only about one in ten
    carries it; the only fixed rule there is `f` for fields. We keep it throughout, so that
    parameters, fields and local variables can be told apart.
  - Records with methods as `{$ifdef USERECORDWITHMETHODS}record{$else}object{$endif}`.
  - Source text without BOM, only ASCII in string literals.
  - **Performance**: no allocation per element in hot loops; buffers that grow in an
    amortized way (as in `TTopK`) are allowed.

  See the skill `landrix-mormot2` and the mORMot2 sources themselves.

  **Status 2026-10-14**:
  - Both parts are met: Delphi 7 builds, every unit has a header, license line and
    overview block, and `TTopK` indexes with `PtrInt`.
  - Line endings in the repo are uniformly LF (mORMot stores CRLF). This is deliberate,
    because the extension is built primarily under Linux.
- **The iron rule remains**: **no Landrix domain knowledge in the `mormot.ai.*` code.**
  It is now justified by the **repo boundary** (the extension must be usable without
  Landrix), no longer by upstream suitability.
- **It is not "OpenClaw in Pascal".** We build a lean, secure
  **agent kernel** + a provider-neutral seam (LLM/embeddings/vector store/RAG/MCP).
  Everything Landrix-specific lives in the backend (layer B below).

> **The namespace `mormot.ai.*` stays** — despite the lack of upstream intent **and despite
> the collision with flydev** (who uses the same namespace): it is established in the
> mORMot extension scene for AI and appears everywhere in our code. Since we
> **only adopt code** from flydev and do not run a shared repo (§4), the
> collision has no consequences. **The license stays MPL/GPL/LGPL**
> (like mORMot): the adopted MCP units require it, and it is compatible with MIT
> (flydev pgvector) as well as with LGPL-2.1 + linking exception. **Decided
> (2026-10-14):** the triple license applies for good to the whole repo, including the
> native inference engine; `LICENSE` is worded accordingly. Ported third-party code
> comes only from MIT sources ([docs/native-inference.md](docs/native-inference.md#vorlagen-und-lizenzen)).

## 2. Layer model (the core rule)

**Two layers** — the boundary is now the **repo boundary**. (There used to be three:
"M — mORMot upstream candidate" and "A — incubation" differed only by the
question *"mature enough for a Synopse PR?"*. Without an upstream goal (§1) this
distinction falls away — both move into the **same** extension repo. **A** and **B**
remain as labels; "layer M" no longer exists.)

> **Does the unit name a concrete Landrix table, a guard, a role or
> a domain concept?** → layer B (backend). **Otherwise** → layer A (extension).

| Layer | Namespace | Repo | Content | Depends on Landrix? |
|---|---|---|---|---|
| **A — Extension** | `mormot.ai.*`, `mormot.db.*` | **own repo** (§7), FPC and Delphi 7 to 13 | Everything generic: LLM wire drivers, embeddings, vector store + pgvector binding, RAG engine, MCP, SSE, structured output, agent loop, skill/memory/session **interfaces**, CLI | **no** |
| **B — Backend** | `landrix.server.ai.*` | landrix-platform | Landrix bindings: memory/sessions as `TOrm` in `TLandrixDatabase`, guard-/scope-checked tools, audit connection, domain skills, REST endpoints | **yes** |

**Consequence for interfaces vs. implementations — "interfaces below, bindings above":**

- **Interfaces** (`ILlmClient`, `IEmbedder`, `IVectorStore`, later `IMemoryStore`,
  `ISessionStore`, `IToolbox`) are defined in **A**.
- **Landrix implementations** (ORM memory, guard-checked tools) live in **B**
  and are injected at run time. This keeps the extension **usable without
  Landrix** (a precondition for its own repo), and the backend inherits only the
  seam, not the engine.

## 3. Persistence & vector abstraction (sqlite-vec ↔ pgvector)

**Status today ✓ (interface/impl split done):** the abstraction does not just
exist, it is now also **cleanly separated**. `IVectorStore` and `IEmbedder` live in
pure interface units without backend dependencies; the engine (`TLlmRag`,
`TRagSearchTool`) depends **only** on them and thus no longer pulls in **any** static
SQLite. A second backend (pgvector) is a **new implementation**, not a rework.

**The problem before (fixed):** interface *and* sqlite-vec implementation sat
in **one** unit that pulled in `mormot.db.raw.sqlite3.static` — whoever wanted pgvector (or
no SQLite at all) dragged the static SQLite dependency along.

**Current structure (done):**

```
mormot.ai.vectorstore           ✓ ONLY IVectorStore/IFilteredVectorStore + TRagHit + vector helpers (SQLite-free)
mormot.ai.sqlite                ✓ RollbackIfOpen + extension loader (shared by vec0, lembed, native)
mormot.ai.vectorstore.sqlitevec ✓ TVec0Store (sqlite-vec) [static SQLite isolated here]
mormot.ai.vectorstore.native    ✓ TNativeVectorStore: BLOB per vector space + Pascal search, filterable, without extension — exact (2a) + 1-bit RAM index with exact rescoring (2b); IVF only when needed (§5)
mormot.ai.vectorstore.pgvector  ○ TPgVectorStore (IVectorStore) on mormot.db.pgvector.* (merge §4)
mormot.ai.embeddings            ✓ ONLY IEmbedder (no longer pulls in the LLM client)
mormot.ai.embed.provider        ✓ TProviderEmbedder (OpenAI wire)
mormot.ai.embed.ollama          ○ TOllamaEmbedder (from merge §4)
mormot.ai.embed.lembed          ✓ TLembedEmbedder (sqlite-lembed, local; own :memory: connection) + SharedLembedEmbedder()
mormot.ai.embed.native          ○ TNativeEmbedder (own Pascal encoder on the GGUF, option D in §5)
```
(✓ = built, demos green · ○ = open — pgvector/Ollama come with the merge §4, the
native units with §5)

**Embedder = process service, not per DB.** Generating embeddings (`text → vector`)
is orthogonal to vector **storage** (`IVectorStore`, per DB). A local
GGUF model (lembed) is therefore loaded **once per process** and shared by *N* stores/
DBs — **never** coupled to a store connection (otherwise every DB loads its
own model into RAM). For this, `TLembedEmbedder` owns its **own, dedicated
`:memory:` connection** (it hosts only lembed0 + the model); `SharedLembedEmbedder()`
returns the **process-wide singleton per logical model name**. The
lembed context is not re-entrant → Embed/EmbedBatch serialize under a
`TOSLightLock`; when concurrency dominates, you scale through a second
`IEmbedder` backend (provider/IONOS, Ollama), not through several model loads.

Rule (now applied throughout): **the RAG/agent engine knows only the interfaces.**
Which backend is used (sqlite-vec in the single-binary edge case, pgvector in the
server/multi-tenant case) is decided by the composition in
layer A/B via DI — never by an `{$ifdef}` in the engine.

`mormot.db.pgvector.*` (binding/ORM/core) deliberately stays under `mormot.db.*` instead of
under `mormot.ai.*`: it is a **pure DB layer without any AI relation** and usable
on its own. The naming follows mORMot's `mormot.db.*` family — even without
upstream intent (§1) this is the right placement in our own repo.

## 4. Merge with flydev's `mormot.ai` (public since 2026-07-28)

The "parallel repo" is identified and **public**:
[flydev-fr/mormot.ai](https://github.com/flydev-fr/mormot.ai), docs
[mormot-ai.fsb.dev](https://mormot-ai.fsb.dev/). The unit list matches exactly
what was tracked here as the "parallel repo" so far.

**Goal: full integration** ("the big picture") — not just cherry-picking. *(Revised
2026-10-14: now only selective, pgvector only when needed — see §9 item 10.)* But these are
**two separate libraries with different licenses**, and the
schedule depends on exactly that:

| Library | Content | License | Status for us |
|---|---|---|---|
| `mormot2-pgvector` | `mormot.ai.chunk`, `mormot.ai.embed.{base,ollama}`, `mormot.ai.llm.{base,ollama}`, `mormot.ai.rag.{history,service,stream}`, `mormot.db.pgvector.{binding,orm,base}` | **MIT** | ✅ can be merged right away |
| `mormot-ai-sdk` | LiteLLM port: `mormot.ai.litellm.{client,router,cost,proxy,streaming,handler,types,callbacks}` + `provider.{openai,anthropic,gemini}` (router/fallback, cost tracking) | **none** (repo metadata `license: null`, no LICENSE in the SDK part) | ⛔ *all rights reserved* — no right to use or derive; **license has been requested** |

> **Legal guardrail:** without a license, code is **not** free to use — not even
> in a public repo. The SDK part stays **blocked** until flydev sets a
> license. The MIT part can be merged with attribution (MIT is compatible with our
> MPL/GPL/LGPL goal; keep the copyright notice, see [NOTICE](NOTICE)).

**Name/concept collisions** (now sharper than before — partly the **same
unit name with different content**):

| Topic | Here (LandrixAI) | flydev | Decision |
|---|---|---|---|
| Embeddings | `mormot.ai.embeddings` (interface) | `mormot.ai.embed.base` + `.ollama` | interface in `mormot.ai.embeddings`, drivers as `mormot.ai.embed.*` |
| SSE | `mormot.ai.llm.sse` | `mormot.ai.http.sse` | **`mormot.ai.http.sse`** — SSE is generic HTTP |
| LLM seam | `mormot.ai.llm` clean-room (OpenAI wire = lingua franca) + native `.anthropic` | **two** competing ones: `mormot.ai.llm.base`/`.ollama` (pgvector lib) **and** the LiteLLM port (SDK) | **Our seam stays the core** (clean-room, license-clean, Anthropic-native, MCP bridge, structured). Plug in flydev's Ollama driver as another `ILlmClient`; LiteLLM router/cost tracking only **after** the license is clarified, as an optional extra layer |
| RAG | monolithic `mormot.ai.rag` | `rag` + `.service` + `.stream` + `.history` | **adopt the split** |
| Chunking | `ChunkText` in `mormot.ai.rag` | its own `mormot.ai.chunk` | **extract** into `mormot.ai.chunk` |
| Vector store | sqlite-vec (`TVec0Store`) | `mormot.db.pgvector.*` | **both** as `IVectorStore` backends (§3) |
| Hybrid search | — | cosine + `tsvector` (Postgres) | adopt the concept; our SQLite counterpart is FTS5 + vec (RRF, see Feature-Adressverwaltung) |
| MCP | `mormot.ai.mcp.*` (server+client) | — | stays **our** contribution (§6) |

**Merge order:** (1) integrate the MIT part (pgvector backend, Ollama embedder,
RAG split, chunk) — blocks nothing; (2) the SDK part only after the license is granted.

### Mode: code adoption, no shared repo (decided)

**We only adopt the code** — no fork, no shared repository, no
upstream PRs. Our **namespace `mormot.ai.*` stays unchanged**, even where it
collides with flydev's; adopted units are fitted into / renamed within our tree.
This keeps us independent (own compiler strategy, own inference engine, MCP 2026-07-28) — points that flydev
does not share.

**The price of this freedom: manual follow-up.** So that later flydev changes
can still be worked in, the **adopted commit is pinned** — in
[UPSTREAM_BASE](UPSTREAM_BASE), in the same pattern as for the adopted
MCP units. The baseline is already recorded there (`9ab26e88…`, `main`, 2026-07-28).

```bash
# Update round: what has changed since our state?
gh api repos/flydev-fr/mormot.ai/commits/HEAD --jq .sha
git -C <clone> diff <pinned-sha>..<new-sha> -- mormot2-pgvector
# Port what is relevant, then bump commit/date in UPSTREAM_BASE in the same commit.
```

**Rule:** whoever ports from flydev **bumps UPSTREAM_BASE in the same commit** — otherwise
the diff base is lost and the next round has to guess.

## 5. Inference backends: lembed ↔ provider ↔ our own native engine

The second backend axis next to persistence (§3): **where does the model run?** Here
too, "interfaces below, bindings above" applies — the engine knows only `IEmbedder` /
`ILlmClient`; the backend is a composition decision.

| Backend | How | Third-party dependency | Status |
|---|---|---|---|
| **lembed** (today) | GGUF model in SQLite via `lembed0` | `lembed0` + `libllama`/`ggml` **per arch** (`.so`/`.dll`), `load_extension` | ✓ built, process-wide singleton |
| **Provider** | OpenAI wire `/embeddings` | network + API key | ✓ built |
| **Ollama** | HTTP to a local Ollama server | **external process** | ○ comes with §4 |
| **native** | own Pascal engine (`src/infer/`): embedder **and** `ILlmClient` | **none** | ✓ embedder (2026-10-14), ✓ `ILlmClient` with Qwen3 (2026-10-15, still slow), [docs/native-inference.md](docs/native-inference.md) |
| ~~neural-api~~ | native Pascal inference (LGPL) | – | ✗ **not pursued further** (2026-10-14) |

> **Status 2026-10-13:** replacing `vec0` **and** `lembed0` with native Pascal code
> is analyzed in [docs/native-embeddings.md](docs/native-embeddings.md) (draft for
> discussion, checked against the code of neural-api `b172061`). Key results below; the
> details, the plan with checkpoints and the open questions are there.

### Own engine instead of neural-api (decision 2026-10-14)

The native inference becomes an **own engine in plain Object Pascal**: first BGE-M3
as a replacement for lembed, then decoder LLMs (Qwen3) behind `ILlmClient`, later CUDA.
It runs on all mORMot2 compilers (FPC and Delphi 7 to 13; Delphi 7 as a pure
correctness tier without a performance goal) and uses only MIT-licensed sources as code (llama2.c, ggml, calm);
Apache-2.0 sources (transformers, tokenizers) serve only as reference. Plan, phases,
gates and platform matrix: [docs/native-inference.md](docs/native-inference.md).

**neural-api** was the strategic target backend up to this point. It is no longer
pursued:
- For bge-m3 it lacked charsmap normalization, BERT GGUF, NEON and reentrancy (analysis in
  [docs/native-embeddings.md](docs/native-embeddings.md), state `b172061`).
- It is LGPL and FPC-only.
- The contribution offer [#225](https://github.com/joaopauloschuler/neural-api/issues/225)
  is closed ("not planned", 2026-10-14).

The seam stays unchanged: switching happens through config (`AI_EMBED_PROVIDER`), never through
`{$ifdef}`, and the provider/Ollama/lembed backends remain useful (cloud models,
existing deployments, fallback, oracle).

**Vector search belongs to it:** `vec0` is replaced independently by a native store
(`mormot.ai.vectorstore.native`: float32 `embedding BLOB` per vector space (`<space>_items`) as the source of truth, 1-bit index in RAM + exact rescoring, IVF only when needed — designed for 2 million+ vectors per DB,
mORMot binds `create_function_v2`/`create_module_v2` statically too). vec0 computed with
**L2** (no metric in the schema); the native store computes **cosine** on unit vectors —
the ranking is identical (reference contract v1 in docs/native-embeddings.md).

**Order:**
1. Reference contract and German test corpus ✓
2. Native store ✓ (replaces vec0)
3. Delphi 7 port of the existing code (§9 item 9) ✓ (2026-10-14);
   then engine phase 0 ✓ (2026-10-14), Python reference:
   - fixed FP32 reference per layer (`tests/data/infer-ref/`, `tools/infer-ref/`);
   - tiny fixtures;
   - suite `infer.tests` green under FPC 3.2.4/3.2.2, Delphi 13 and Delphi 7.
4. Phase 1a, BGE-M3 FP32 ✓ (2026-10-14)
5. Phase 1b, the existing Q8 GGUF ✓ (2026-10-14):
   - token IDs identical to lembed (literal special tokens checked separately against
     `parse_special = false`);
   - nDCG@10 0.8989 ≥ 0.8863.
6. Phase 1c, measurement and switchover ✓ (2026-10-14: aarch64 and x86_64, Linux x86_64
   and soak test green):
   - "native" is the recommended provider in the backend (`AI_EMBED_PROVIDER=native`);
   - lembed stays fallback and oracle.
7. Phase 2, decoder Qwen3 ✓ (2-0 to 2c 2026-10-14: tokenizer, chat template, forward,
   KV cache; greedy in FP32 token-identical to transformers. 2d 2026-10-15:
   `mormot.ai.llm.native` behind `ILlmClient`, agent loop tested with a tool)
8. Phase 3, quantization and SIMD (3a ✓ 2026-10-15: Q8_0 weights, F32 and attention
   with SIMD and threads, decoding 15 times (F32) or 22 times (Q8_0) as fast; 3b ✓
   2026-10-15: sampler; 3c ✓ 2026-10-15: prompt cache; 3d ✓ 2026-10-15: norms/RoPE/SwiGLU
   in parallel; 3e-1 ✓ Q4_0 experimental; open: GGUF loading, K-quants, Q4_0 kernels)
9. Phase 4, CUDA

Details are in [docs/native-inference.md](docs/native-inference.md).

## 6. MCP protocol: jump to 2026-07-28 (greenfield, no legacy)

The spec **2026-07-28** is the largest change since the MCP launch
([changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)).
**Project decision: we speak only 2026-07-28.** What the new spec
removes is **deleted** here — no multi-version negotiation, no
legacy paths, no deprecation ballast (the project's greenfield rule).

> **Exception since 2026-09-10 — legacy bridge for stdio, scheduled for removal**
> (marker `LEGACY-ERA` in the code). `src/mormot.ai.mcp.legacy.pas` additionally answers the
> `initialize` handshake on **stdio** (lifecycle 2025-11-25, plus `ping` and
> `notifications/initialized`) and then forwards requests without `_meta`, with `_meta`
> filled in, to the **unchanged** 2026-07-28 core. The spec allows exactly that
> (`docs/specs/mcp-2026-07-28/basic/versioning.mdx`: a server **MAY** serve both eras,
> `initialize` selects the legacy semantics for the stdio process).
> **Why:** Claude Code (2.1.267) opens stdio with `initialize` by default; even with
> negotiation enabled (`MCP_PROTOCOL_NEGOTIATION=auto` + `MCP_SDK_GENERATION=v2`) its
> `server/discover` probe gives up after 3 s, while a server started via `wsl` needs about
> 7 s from cold — without the bridge, codenav for example does not connect reliably.
> **Limits:** stdio only (HTTP stays modern-only); no `logging/setLevel`, no
> server-initiated requests, no `resources/subscribe`; an `input_required` result
> becomes a JSON-RPC error. Requests **with** `_meta` keep running purely modern.
> **Removal, as soon as** the target clients open stdio via `server/discover` by default
> **and** their probe survives a slow process start. Then: delete the unit, remove the lines
> marked `LEGACY-ERA` in `mormot.ai.mcp.stdio.pas` and the `mcp.tests` tests
> (`StdioTransportLegacyInitialize`, `LegacyBridgeTranslation`), and strike this paragraph
> and the README section "One temporary exception … stdio legacy bridge".
> Can be switched off beforehand with `TMcpStdioTransport.AcceptLegacyInitialize := false`.

**OUT (delete without replacement):**
- `initialize` / `notifications/initialized` **and the entire version negotiation**
  (2024-11-05 / 2025-03-26 / 2025-06-18 / 2025-11-25) — the protocol is stateless.
  *(Only exception: the legacy bridge for stdio above.)*
- **Protocol sessions + `Mcp-Session-Id`** in Streamable HTTP → and with them our
  session registry, expiry logic and the FSafe re-resolve.
- **SSE resumability**: `Last-Event-ID`, SSE event IDs, `NextSessionEventId`.
- **HTTP GET endpoint** + `resources/subscribe`/`unsubscribe`.
- `ping`, `logging/setLevel`, `notifications/roots/list_changed`.
- **Legacy HTTP+SSE transport** (`TMcpSseTransport`) — now officially *deprecated*;
  the open question "harden or remove" is thus **decided: remove**.
- **Roots / sampling / logging** (deprecated) — do not build them at all.

**IN:**
- **`server/discover`** — mandatory RPC (supported versions, capabilities, identity).
- **`_meta` per request**: `io.modelcontextprotocol/protocolVersion`,
  `clientCapabilities`, `clientInfo`, `logLevel`; `serverInfo` in every result.
- **`subscriptions/listen`** — a long-lived POST response stream, opt-in per type
  (`toolsListChanged`, …), notifications tagged with `subscriptionId`.
  Request-scoped ones (`notifications/progress`, `…/message`) stay on the
  response stream of their request.
- **`resultType`** on all results (`complete` | `input_required`).
- **MRTR** instead of server-initiated requests: `InputRequiredResult` with
  `inputRequests`; the client answers by retrying with `inputResponses` (+ `requestState`).
- **`CacheableResult`**: `ttlMs` + `cacheScope` on `tools/list`, `prompts/list`,
  `resources/list`, `resources/read`, `resources/templates/list`; tools in
  **deterministic** order.
- **Headers** `Mcp-Method` / `Mcp-Name` on POST; `x-mcp-header` for tool parameters.
- **Extensions framework** (`extensions` in client/server capabilities); **tasks** only
  as the extension `io.modelcontextprotocol/tasks` (polling `tasks/get`,
  `tasks/update`, no `tasks/list`).
- **Error codes**: `-32020` HeaderMismatch, `-32021` MissingRequiredClientCapability,
  `-32022` UnsupportedProtocolVersion; resource-not-found `-32002` → **`-32602`**.
- **Auth hardening**: check `iss` (RFC 9207), `application_type` in DCR, credentials
  bound to the issuer; **Client ID Metadata Documents** instead of DCR.

> **Cross-call state**: "stateless" does not mean stateless for tools — whoever needs state
> across calls uses **server-minted handles as normal tool arguments**
> (SEP-2567). For our agent/RAG tools that is the intended way.

**Effort estimate: a net simplification.** Most of the rework is
**deleting** — and it settles two open roadmap items along the way (parallel
stress test of the session logic, hardening of the legacy SSE transport), because the
mechanics behind them no longer exist in the target protocol. **What remains** is the
open item **auth resolver** — if anything larger because of the new spec's auth hardening.

## 7. Split into its own repository — **done (2026-07-29)**

From §1 it follows: the `mormot.ai.*` code leaves the Landrix monorepo. Landrix has
since been a **consumer** of the extension — exactly as with mORMot2 itself.

> **Status:** repo [`landrix/mormot2-ai`](https://github.com/landrix/mormot2-ai)
> (public), created with `git subtree split --prefix=shared/delphi/landrixai`
> **with full history** (33 commits). Landrix includes it as a submodule under
> `shared/delphi/landrixai` — the path stays the same, so the consumers'
> `-Fu` build paths do **not** change. `codenav/` was lifted out of the
> prefix to `shared/delphi/codenav/` beforehand (see below).
>
> The repo name deliberately differs from flydev's `mormot.ai`; the **unit** namespace
> `mormot.ai.*` stays (§4, mode "code adoption").

### What goes, what stays

| Goes into the extension repo | Stays in landrix-platform |
|---|---|
| `src/` (all `mormot.ai.*` units) | `backend/src/landrix.server.ai.*` (layer B) |
| `tests/`, `demos/`, `scripts/` | `docs/Feature-LandrixAI-Agent.md` (backend feature) |
| `CONCEPT.md`, `DESIGN.md`, `README.md`, `LICENSE`, `NOTICE`, `UPSTREAM_BASE` | `AI_EMBED_*` config, Docker wiring, provisioning of models/extensions |
| `vendor/` **structure** (binaries stay gitignored) | `deploy/docker-native/*` (entrypoint, `pack-ai-dist.sh`) |
| — | **`codenav/`** (Landrix tool, see below) |

**`codenav/` stays outside** (decided): it is a **Landrix developer
tool**, not part of the extension — it does **not** move along. Consequence:
it then uses the `mormot.ai.mcp.*` units through the **submodule**, like any other
consumer (adjust the build paths accordingly).

### Inclusion: submodule (established pattern)

Landrix includes the extension as a **git submodule** — the same pattern as
`shared/delphi/libs/_git_Synopse2`. No package manager, no vendoring copy.

### Extraction **with** history

The history is valuable (review hardenings, design decisions) — **not**
by copy-paste into an empty repo:

This is how it went (for reproduction in further splits):

```bash
# 1. lift codenav out of the prefix BEFORE the split - otherwise the
#    Landrix tool ends up in the extension and collides with the submodule mount
git mv shared/delphi/landrixai/codenav shared/delphi/codenav

# 2. cut out the history and push it into the empty repo
git subtree split --prefix=shared/delphi/landrixai -b split/mormot2-ai
git push https://github.com/landrix/mormot2-ai.git split/mormot2-ai:main

# 3. in the monorepo, replace the path with the submodule
git rm -r shared/delphi/landrixai
git submodule add https://github.com/landrix/mormot2-ai.git shared/delphi/landrixai
```

**Pitfall when replacing**: under `shared/delphi/landrixai` there are ~2 GB of
**gitignored** local data (`vendor/models` with GGUF, `vendor/sqlite-ext`,
`_eval/`, `bin/`, `_cleanroom/`). `git rm -r` does not touch them, but
`git submodule add` refuses a non-empty target directory — so move the
ignored folders away, clone the submodule, move them back. Otherwise several GB
would have to be downloaded again.

### To follow up (checklist — the build depends on this)

1. **Build paths** `-Fu` to `shared/delphi/landrixai/src` in **three** scripts:
   `backend/scripts/run-fpc-tests.sh`, `build-server-linux.sh`, `build-server-win.ps1`.
2. **Docker/CI — the critical point**: `deploy/docker-native/Dockerfile` copies
   `shared/delphi/landrixai/src`. As a submodule, the CI checkout must set **`submodules:
   recursive`**, otherwise the directory is **empty** in the build context and the
   FPC build aborts. (The same pattern already struck with the Synopse2 static libs
   → `deploy/docker-native/static/`.)
3. **codenav indexing** + CLAUDE.md paths (`shared/delphi/{landrixai,client,dto}`).
4. `.gitignore` entries (`vendor/models`, `vendor/sqlite-ext`, `bin/`) move along.
5. `deploy/docker-native/pack-ai-dist.sh` (reaches into `landrixai/vendor/…`).
6. Skills related to landrixai (`landrix-mormot2`).

### Timing: **early, before the big reworks**

Recommendation: **split now**, not after the merge/MCP rework. Reasons:
- The coupling to the backend is **already thin** (layer B uses only `IEmbedder` +
  `GetSharedEmbedder`) — the cut is cheap today and gets more expensive with every further
  feature.
- The three big upcoming pieces of work (flydev merge §4, MCP rebuild §6,
  native inference §5) take place **entirely within** the extension. Doing them in the
  target repo saves a second move.
- The flydev merge means a new repo layout anyway (merging two libraries)
  — that does not belong in the Landrix monorepo.

**Repo model (decided):** **own, independent repo** — not a shared one with
flydev, no fork. From there **only code is adopted** (§4, mode "code adoption"),
the namespace `mormot.ai.*` stays despite the collision. The repo **name** is thus free
to choose (`mormot.ai` is taken by flydev) — e.g. `landrix/mormot-ai-ext`.

## 8. Inventory of the current state (layer A, built)

Everything below is **built + review-hardened + green**; current numbers per platform (as of
2026-10-14: MCP 1012; LLM 647–659 depending on platform and vec0 test) in
[docs/native-inference.md](docs/native-inference.md#plattformen-und-tests). Streaming/tool loop/RAG/vision also
verified **live** (OpenAI/Ollama/Anthropic). Build-up history: [DESIGN.md](DESIGN.md).

- **LLM client**: `mormot.ai.llm` (OpenAI wire = lingua franca, covers OpenAI/LiteLLM/
  Ollama) + native `mormot.ai.llm.anthropic` (Messages API behind the same
  `ILlmClient` seam); types `mormot.ai.llm.types`; SSE streaming
  (`mormot.ai.llm.sse`, base + OpenAI/Anthropic subclasses); vision/multimodal
  (`TLlmMessage.Images`, both wires).
- **Agent loop**: `TLlmAgent` + `ILlmToolbox` ([mormot.ai.agent.pas](src/mormot.ai.agent.pas)).
- **MCP**: server+client, stdio/HTTP/Streamable HTTP/in-process, RTTI tool schema
  (`mormot.ai.mcp.*`), MCP bridge into the agents (`mormot.ai.agent.mcp`). Speaks
  only **2026-07-28** (§6, stateless); legacy SSE is removed, only a
  stdio bridge for the old `initialize` handshake (`mormot.ai.mcp.legacy`) is still there
  and scheduled for removal.
- **Structured output**: `mormot.ai.llm.structured` (`ChatStructured` provider-agnostic);
  OpenAI `response_format` ↔ Anthropic `output_config.format` (the adapter translates + injects
  `additionalProperties:false`), both verified live.
- **Embeddings/RAG** (interface/impl separated, §3): interfaces `IEmbedder`
  (`mormot.ai.embeddings`) + `IVectorStore` (`mormot.ai.vectorstore`, SQLite-free);
  impls `TProviderEmbedder` (`mormot.ai.embed.provider`), `TLembedEmbedder`
  (`mormot.ai.embed.lembed`), `TVec0Store` (`mormot.ai.vectorstore.sqlitevec`) and
  `TNativeVectorStore` (`mormot.ai.vectorstore.native`: without a C extension, contract,
  domain filter `SearchWhere`, 1-bit index + exact rescoring, measured at 2 million × 1024);
  engine `TLlmRag` (`mormot.ai.rag`) + **agentic RAG** as the `search_docs` MCP tool
  (`mormot.ai.rag.tool`, the same tool externally as in-process).
- **codenav**: MCP tools for code navigation (`codenav/`).

## 9. Open items / roadmap

The engine feature surface (phase D) is in place, the **repo split** (§7) is done,
the **MCP jump to 2026-07-28** (§6) is carried through and, with the
consumer binding (phase B), also proven in practice; open are the **flydev merge**
(§4) and the **native inference** (§5, own engine). This list is the consolidated
roadmap — DESIGN.md points here.

0. **Repo split (§7)** ✓ **done (2026-07-29)** — `landrix/mormot2-ai`, history
   preserved, Landrix pulls it in as a submodule. All further items take place **here**.

1. **Refactor §3** ✓ **done** (interface/impl units separated: `vectorstore` +
   `vectorstore.sqlitevec`, `embeddings` + `embed.provider` + `embed.lembed`; engine
   SQLite-free, demos green). Only the **new** pgvector backend remains open (item 2).
2. **Merge §4** — *since 2026-10-14 selective instead of full integration (item 10)*;
   originally staged by license:
   - **(a) now**: `mormot2-pgvector` (**MIT**) — pgvector as a second
     `IVectorStore` backend `mormot.ai.vectorstore.pgvector` (the seam is in place),
     Ollama embedder as `mormot.ai.embed.ollama`, RAG split
     (`rag.service`/`.stream`/`.history`), chunking → `mormot.ai.chunk`,
     SSE → `mormot.ai.http.sse`. Carry attribution/NOTICE along.
   - **(b) blocked**: `mormot-ai-sdk` (LiteLLM port) — **no license**,
     request pending. Do not touch until it is granted; our clean-room LLM seam stays
     the core until then (and probably for good).
3. Define **memory/session interfaces** (layer A) — implementation in the
   backend (layer B), details in
   [docs/Feature-LandrixAI-Agent.md](../../../docs/Feature-LandrixAI-Agent.md).
4. **x64-linux `.so`** for sqlite-vec/lembed ✓ **available + verified** — all four
   arch builds present (aarch64/x86_64 × linux/win64); new vec0 `v0.1.10-alpha.4` +
   lembed0 checked at run time on **both** deploy arches: aarch64 natively (rag-spike/
   chat/agent) and x86_64 against mORMot's **static** SQLite in the amd64 container
   (`scripts/x64-ext-verify.sh`). Only the wiring into the Docker image remains open.
5. **Remaining Anthropic surface** ✓ **done**: structured output via `output_config.format`
   (the adapter translates the neutral `ResponseFormat` and injects the
   `additionalProperties:false` Anthropic requires) — live against `claude-opus-4-8` (typed
   record extracted). Anthropic **streaming** verified live (SSE parser incl.
   message_start/delta usage end to end). Both demos (`llm-structured`, `llm-chat`) now
   provider-agnostic via `LLM_PROVIDER`.
6. **MCP jump to 2026-07-28** (§6) ✓ **done** — replaces the former item
   "transport production readiness". The rework is mostly **deleting**; several
   old findings settle themselves that way. Status: branch `feat/mcp-2026-07-28` merged into
   `main`, remaining protocol gaps closed on 2026-08-01, **829 assertions** green
   (evidence in detail in [DESIGN.md](DESIGN.md), sections "Protokoll" and
   "Protokoll-Restlücken geschlossen"):
   - ~~Parallel stress test of the Streamable sessions~~ **dropped** — protocol sessions
     (and with them registry, expiry, FSafe re-resolve, `NextSessionEventId`) no longer
     exist in the target protocol.
   - ~~Harden or remove the legacy SSE transport~~ → **remove** (officially
     *deprecated*), decision made.
   - **Auth** ✓ **done** (mechanics **and** resolver). The server is an
     OAuth 2.1 **resource server** (RFC 9728 metadata, token check before dispatch,
     audience binding, 401/403 challenges — details in [DESIGN.md](DESIGN.md)). The
     seam is `IMcpTokenVerifier`; its implementation has lived in the
     consumer since 2026-08-01 (`backend/src/landrix.server.ai.McpAuth.pas`), together with
     the scope→permission mapping, plus first tools through the registry — that is
     **phase B** and thus the proof that registry, auth context and RTTI schema
     hold up in third-party code. The consumer has meanwhile **also** become an authorization
     server (self-registration, PKCE, refresh rotation, revocation) — that is
     explicitly **not** the business of this repo, see
     [Feature-MCP-Auth.md](../../../docs/Feature-MCP-Auth.md).
     **Correction to the earlier version**: `iss`/RFC 9207, Client ID Metadata Documents
     and issuer binding are **client/authorization server duties**, not those of the
     MCP server — they are no longer pending here.
   - **Built** (details in [DESIGN.md](DESIGN.md), section "Protokoll"):
     `server/discover`, `_meta` transport of the protocol version, `subscriptions/listen`,
     `resultType`, **MRTR**, `ttlMs`/`cacheScope`, new headers + error codes.
   - ~~**To be built**: `x-mcp-header`, extensions framework, JSON Schema 2020-12 in the
     `inputSchema`, OTel `_meta` keys, deterministic `tools/list` order,
     prompts/completion/pagination~~ — **all built** (2026-08-01), plus
     progress notifications and the template resolver. **Deliberately not built**:
     `notifications/message` (logging) is *deprecated* in 2026-07-28 — already decided
     here under §6; listing it earlier as a gap was a documentation error.
   - **Demo hardening** ✓ (still valid): the Streamable demo binds **loopback**,
     `ask_claude` **disabled** by default (`MCP_ENABLE_ASK_CLAUDE=1` as opt-in).
7. **Native embeddings + vector search + inference** (§5):
   - Store ✓, 2026-10-13: `mormot.ai.vectorstore.native` replaces `vec0`. Analysis and
     gate 5 measurement in `docs/native-embeddings.md`; 2c IVF only when needed.
   - Engine ○: own Pascal inference according to
     [docs/native-inference.md](docs/native-inference.md). First BGE-M3 (replaces lembed),
     then Qwen3 behind `ILlmClient`, then CUDA. Phase 0 (reference data, test frame) ✓
     and phase 1a (BGE-M3 FP32 from text to vector, matches PyTorch) ✓ 2026-10-14,
     phase 1b (the same Q8 GGUF as lembed, matches the oracle, nDCG@10 0.8989) ✓
     2026-10-14; 1c ✓ 2026-10-14: NEON and AVX2, on x86_64 1.6 to 2 times as fast
     as lembed, Linux x86_64 green, soak test over 90 minutes passed, the
     recommended provider in the backend; plus the load governor (item 10.7). Phase 2 ✓: up to 2c
     2026-10-14 (Qwen3-0.6B in FP32, greedy token-identical to transformers), 2d
     2026-10-15 (`mormot.ai.llm.native` behind `ILlmClient`).
   - Goal: all native third-party `.so` files and the Docker bundle download go away. vec0 and
     lembed go **independently** of each other, each after its own gate. lembed stays
     fallback and oracle until then.
8. **Tests**: TVec0Store has an **env-gated real vec0 test** (keyed ops
   `Upsert`/`Delete`/`Search` key, `test.llm.rag`, runs when `SQLITE_EXT_DIR` is set,
   otherwise skipped). Open: a TLembedEmbedder real test + RAG atomic rollback (they need the
   lembed runtime, see item 4).
9. **Platform range like mORMot2** (2026-10-14, findings in
   [docs/native-inference.md](docs/native-inference.md#plattformen-und-tests)):
   - **Delphi 7** (Win32) ✓ **done 2026-10-14**: all 22 units, MCP and LLM suite
     green (`scripts/run-delphi7-tests.ps1`). MCP registries on `TSynDictionary`,
     typed tools on `TMcpRecordTool` (`TMcpToolBase<T>` stays as a wrapper under
     `HASGENERICS`, the consumer unchanged), `exit(x)` replaced, `object` instead of records
     with methods. Found and fixed: under Delphi 7, stdio would have encoded UTF-8 twice
     (test `StdioReadsUtf8Unchanged`).
   - **Windows ARM** (FPC 3.3.1): the suites are green as long as an ARM64 `sqlite3-64.dll`
     sits next to them; mORMot2 has no static SQLite for `aarch64-win64`. Missing are a
     test script and a way to obtain the DLL.
   - **macOS M2** (FPC 3.2.4-rc2): green. A repeatable script is missing.
   - **RTX machine** (Windows 11 x64, i9-14900K, RTX 5090): the suites are green natively under
     FPC 3.2.2. Role: AVX2, native x64 performance, CUDA. The CUDA toolkit is still
     missing.
10. **Further gaps**, ordered by immediate benefit (Codex comparison 2026-10-14):
    1. Complete the embedder selection in the consumer. The backend factory knows only
       `none`/`lembed` and rejects `ollama`. The OpenAI-compatible
       `TProviderEmbedder` (e.g. IONOS) is enough as a second path.
    2. Agent loop:
       - cancellation, deadlines, context budget, step events;
       - then session persistence (item 3). Repeated tool calls with
         side effects must be ruled out there.
    3. Ingestion:
       - token-based chunking (`ChunkText` counts bytes);
       - stable document and chunk identity;
       - replace, delete and reindex as a workflow.
    4. flydev adoptions (item 2) only selectively; pgvector only once a deployment
       needs it.
    5. Unify the stale assertion counts in README/DESIGN.
    6. A build script for the demos under Delphi (13 and 7) ✓
       (`scripts/build-delphi-demos.ps1`, `-Delphi7` for Delphi 7, 2026-10-17, all 18 demos).
    7. **Load governor** for bulk embedding ✓ built (2026-10-14, `mormot.ai.governor`,
       in the backend `GetBulkEmbedder` with `AI_BULK_*`); open only its first caller, an
       indexing job, and a run on real sensor hardware (Linux, not WSL2):
       - the native embedder limits threads itself (`AI_EMBED_THREADS`); the governor
         sets a duty cycle per worker (pauses outside the lock, `Abort` on
         shutdown);
       - temperature threshold where a sensor is readable (Linux: `/sys/class/hwmon`,
         in the Docker container without root); under WSL2 and Windows there is no
         sensor without a driver;
       - as long as no threshold is active, a heuristic on throughput against a
         slowly aging reference: a lasting drop lowers the duty cycle, a
         recovery raises it again. It is a heuristic - other load or a different
         text mix also lowers throughput (this is how the Windows throttling was noticed, see
         docs/native-inference.md).
       Fans and power limits (PL1/PL2) stay with firmware and BIOS; from outside,
       `--cpus` limits it in Docker operation.
11. **Distributed execution** (added 2026-10-15, plan:
    [docs/distributed-execution.md](docs/distributed-execution.md), working title
    "Rechenknecht"):
    - distribute large inference jobs by pull to idle machines on the network, with lease
      and watchdog;
    - tasks are distributed, not the model.
    A separate track next to the engine roadmap, whose order stays unchanged.
    - **Phase 1:** embeddings in the LAN. It can start right away; engine phase 1 is
      finished.
    - **Later:** robustness, GPU machines and LLM jobs. LLM jobs require
      engine phase 3 (2d is done).
    Building blocks are available:
    - `mormot.ai.embed.native`;
    - the load governor (item 10.7) as the executor's brake;
    - the exact mode, bit-identical to the reference of the same build. Across platforms
      this still has to be measured.
    To be built:
    - queue, lease and REST API;
    - idle detection per platform;
    - the executor binary.
    Five decisions are open, among them whether the server runs in the extension or in the
    Landrix backend (see the plan).

**Already fixed in this review pass** (build + tests green: MCP 202, LLM 256
assertions): JSON-RPC envelope validation (`jsonrpc:"2.0"`, params type) + broad
exception→JSON-RPC error catch with correct codes (−32600/−32601/−32603);
fail-closed `IsAuthenticated` (identity only from a real resolver, never
from the transport — NOT "without a verifier everything is rejected", see DESIGN
"Review-Härtung Runde 5"); Streamable UAF via FSafe; SSE map thread-safe + leak fix;
RAG ingestion **strict** (exact vector count) **+ atomic** (`IVectorStore.AddBatch`,
one transaction); `build-demo.sh` builds **all** demos again (before, only 5 MCP demos
despite "ALL DEMOS OK"); demo loopback + `ask_claude` gate.
