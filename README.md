# mormot2-ai — the `mormot.ai.*` extension for mORMot2 (FPC · Delphi 7 … 13)

A **mORMot-native** AI extension for **Free Pascal** and **Delphi 7 to 13** — the same
compiler range as mORMot2. It covers the Model Context Protocol, LLM clients, embeddings,
vector search and RAG, and it brings **its own inference engine in plain Object Pascal**:
no Python, no llama.cpp, no foreign `.so`/`.dll` at run time. mORMot2's HTTP stack, JSON,
RTTI, SQLite and crypto are the foundation, not a dependency we tolerate.

> **Status, 2026-10-19:** MCP, LLM clients, agent loop, RAG and the native vector store
> are done and in use. The native inference engine runs BGE-M3 embeddings from the same
> GGUF file as `lembed`, with the same quality and **1.6 to 2 times its speed on x86_64** (Linux / Windows);
> it passes all suites, its quality gate and a 90-minute soak test on Linux x86_64 too,
> and it is the recommended provider of the consuming backend. Three decoder families
> run behind `ILlmClient` - **Qwen3**, **Qwen3-MoE** (mixture of experts, Qwen3-30B-A3B)
> and **Qwen2.5** - with greedy output in FP32 token-identical to `transformers` (for MoE
> on tiny reference models; the 30B model is checked against the CPU path):
> the agent loop calls tools with a model on this machine, greedy or sampled, with a
> prompt cache for the next turn. Weights come from safetensors or llama.cpp GGUF files
> (Q8_0, Q4_0, Q4_K_M, with the Q5_0 matrices llama.cpp writes for widths like
> Qwen2.5's 896). The whole forward pass runs on the CPU (NEON, AVX2), on Vulkan
> (tested on an RTX 5090, an Adreno X1-85 and an M2 Pro through MoltenVK) and on CUDA. An
> RTX 5090 decodes Qwen3-0.6B at about 400 tokens/s (CUDA) and Qwen3-30B-A3B at about 97
> (CUDA; Vulkan 90), choosing its experts on the GPU. Sliding-window attention and layers
> without rotary positions run on the CPU, Vulkan and CUDA (SmolLM3, so far without a chat
> template). gpt-oss-20b runs on the CPU from its published MXFP4 checkpoint or GGUF file
> (attention sinks, YaRN, biases, the clamped SwiGLU) and chats through `ILlmClient` in its Harmony format, tool calls included; the GPU follows. A decoder also takes embedding rows
> instead of token ids and loads as one stack of a larger checkpoint - the talker and code
> predictor of Qwen3-TTS match PyTorch on the CPU, Vulkan and CUDA. Three test suites - about 7200 assertions on Windows x64, fewer
> where a feature is missing - green on every platform in the table below.

## What it does

| Area | Units | In short |
|---|---|---|
| **MCP server** | `mormot.ai.mcp[.server/.stdio/.tools]` | protocol revision **2026-07-28** (stateless); tools from typed records, input schema generated from RTTI; stdio, HTTP, Streamable HTTP, in-process; OAuth 2.1 resource server (opt-in) |
| **LLM clients** | `mormot.ai.llm[.openai/.anthropic/.native/.sse/.structured]` | OpenAI wire as lingua franca (OpenAI, LiteLLM, Ollama-compatible), native Anthropic driver, local Qwen3, Qwen3-MoE and Qwen2.5 on the native engine; streaming, tool calls, structured output, vision |
| **Agent** | `mormot.ai.agent[.mcp]` | tool-calling loop; MCP tools usable by the agent |
| **Embeddings** | `mormot.ai.embeddings`, `.embed.{native,provider,lembed}` | one `IEmbedder` seam: the native engine, any OpenAI-compatible API, or sqlite-lembed |
| **Vector search** | `mormot.ai.vectorstore[.native/.sqlitevec]` | native store in mORMot's SQLite: float32 rows as the truth, 1-bit RAM index with exact rescoring, SQL key filters — measured at 2 million × 1024 |
| **RAG** | `mormot.ai.rag[.tool]` | strict, atomic ingestion; retrieval; RAG as an MCP/agent tool |
| **Load governor** | `mormot.ai.governor` | paces bulk jobs: duty cycle, CPU temperature threshold (Linux hwmon), throughput heuristic; `TGovernedEmbedder` wraps any `IEmbedder` |
| **Native inference** | `src/infer/mormot.ai.infer.*` | GGUF and safetensors loaders (one file or several with an index); Q8_0, Q4_0, Q5_0, Q4_K, Q6_K, MXFP4 weights; SentencePiece/Unigram and byte-level BPE tokenizers (`tokenizer.json`, `vocab.json` + `merges.txt` or GGUF); the XLM-RoBERTa encoder (BGE-M3) and a decoder with dense and mixture-of-experts layers, sliding-window attention and layers without rotary positions (Qwen3, Qwen3-MoE, Qwen2.5, SmolLM3, gpt-oss on the CPU), embedding rows instead of ids and decoder stacks of larger checkpoints (`ForwardEmbeds`, `CreateStack`), chat templates, sampler, prompt cache; CPU backend with NEON and AVX2 kernels and its own thread pool (jobs may take the index of their thread, for scratch buffers per thread); Vulkan (embedded SPIR-V) and CUDA (embedded PTX, driver API) without shipped libraries |

## Where it runs

Number of assertions per suite, every one passing unless noted; each row with numbers ran
between 2026-10-14 and 17, the inference suite of the three i9-14900K rows (RTX 5090,
Qwen3-30B-A3B and Qwen2.5 included) and of the M2 Pro again on 2026-10-19:

| Platform | Compiler | MCP | LLM | Inference |
|---|---|---|---|---|
| Linux aarch64 (WSL2, Snapdragon X) | FPC 3.2.4 and 3.2.2 | 1257 | 727 | 2364 ⁶ |
| Windows x64 (i9-14900K) | FPC 3.2.2 | 1257 | 723 | 5182 |
| Windows x64 (i9-14900K) | Delphi 13 | 1257 | 726 | 5182 |
| Windows x86 | **Delphi 7** | 1253 | 714 | 3407 ⁴ |
| macOS aarch64 (M2 Pro) | FPC 3.2.4 | 1257 | 716 | 4257 ⁵ |
| Windows ARM64 (Snapdragon X) | FPC 3.3.1 ¹ | 1257 | 714 ² | 2728 ¹ |
| Linux x86_64 (WSL2, i9-14900K) | FPC 3.2.2 | 1257 | 725 | 2486 ³ |

¹ FPC 3.3.1 is a development compiler, the only one for this target (FPC 3.2 and Delphi
have none), so this row does not count as a release platform. The Vulkan suite runs on the
Adreno X1-85 GPU, Qwen3-0.6B included: its 165 MB head exceeds the 128 MB the GPU binds per
buffer and runs in parts (backlog B11); four Q8_0 failures in `matvec8.comp` were a
miscompiled word index on that GPU (B10). A first run without the
dot-product kernels (their CPU detection did not cover Windows ARM) also missed a Qwen3
logits check on the Pascal path (cosine 0.979).
² Needs an ARM64 `sqlite3-64.dll` next to the binary — mORMot2 ships no static SQLite for
that target.
³ The production target (Docker uses the same FPC 3.2.2), AVX2 kernels on the System V
entry. Vulkan there is llvmpipe (a CPU implementation), so the Qwen3 tests on the device
skip. The BGE-M3 corpus gate (`LAI_INFER_GATE=1`) passed in an earlier run: nDCG@10 0.8994
on the AVX2 path.
⁴ Natively on an i9-14900K, the Vulkan suite included. The Qwen3-0.6B and Qwen3-30B-A3B
tests skip in a 32-bit process (2.4 GB of FP32 weights, 18.6 GB of GGUF); the tiny
models, the tokenizers and the chat templates run.
⁵ With all models except Qwen3-30B-A3B (the tiny MoE model runs), NEON dot-product
kernels, and the Vulkan suite on the M2 Pro GPU through MoltenVK (`LAI_REQUIRE_VULKAN=1`). The BGE-M3
corpus gate passed there in an earlier run (nDCG@10 0.9004, the same as on the
Snapdragon); it is not part of this count.
⁶ FPC 3.2.4 with all models; under FPC 3.2.2 the native client tests ran on their own
(without the model).

Counts differ because some tests only exist where a feature does: the `vec0` extension
has no 32-bit or macOS build (its tests skip), Delphi runs an extra branch for its UTF-16
strings in the chunking test, the load governor reads other sensors per OS, every SIMD
kernel of a CPU is checked against Pascal (NEON has more of them than AVX2), and Qwen3
on the GPU skips on a CPU implementation of Vulkan.

## Native inference

The engine replaces `lembed` (llama.cpp inside SQLite) and, later, every other foreign
library. It is written from published specs and MIT-licensed templates (llama2.c, ggml,
calm); Apache-2.0 sources (transformers, tokenizers) are only read, never copied. It
targets mORMot2's compiler range — Delphi 7 as a correctness tier without a speed goal;
the tested configurations are listed under [Where it runs](#where-it-runs).

```pascal
program embed;

{$I mormot.defines.inc}

uses
  {$I mormot.uses.inc}
  mormot.ai.llm.types, mormot.ai.embeddings, mormot.ai.embed.native;

var
  emb: IEmbedder;
  v: TLlmEmbedding;
begin
  // one shared embedder per logical model name: the first call decides the
  // model path and the thread count
  emb := SharedNativeEmbedder('vendor/models/bge-m3-q8_0.gguf', 'bge-m3');
  v := emb.Embed('Abdichtung eines Flachdachs');
  // v: 1024 floats, L2-normalised - ready for TNativeVectorStore
end.
```

A chat model is as short - Qwen3, Qwen3-MoE, Qwen2.5 or gpt-oss from a Hugging Face folder or one
GGUF file, behind the same `ILlmClient` as the HTTP providers:

```pascal
program chat;

{$I mormot.defines.inc}

uses
  {$I mormot.uses.inc}
  mormot.core.os, mormot.ai.llm.types, mormot.ai.llm, mormot.ai.llm.native;

var
  llm: ILlmClient;
  msgs: TLlmMessageDynArray;
begin
  // a Hugging Face folder (quantized to Q8_0 at load), or one GGUF file with
  // TNativeLlmClient.CreateFromGguf; the family comes from the files
  llm := TNativeLlmClient.Create('vendor/models/qwen3-0.6b');
  SetLength(msgs, 1);
  msgs[0] := LlmMessage(lrUser, 'Why is the sky blue?');
  ConsoleWrite(llm.ChatComplete(LlmChatRequest('', msgs)).Content);
end.
```

Streamed, and on Vulkan or CUDA: [`demos/llm/llm-quickstart.lpr`](demos/llm/llm-quickstart.lpr)
(`llm-quickstart <folder | file.gguf> [question] [cpu | vulkan | cuda]`).

**How it is checked.** Every layer is compared against a fixed Python reference
(`tools/infer-ref/`, data under `tests/data/infer-ref/`); token ids match lembed on all
1318 cases without special-token literals (those are deliberately treated as text and
checked against SentencePiece instead); on a German retrieval corpus the embeddings reach
nDCG@10 **0.9004** on the NEON path and **0.8994** on the AVX2 path (lembed 0.8963,
gate 0.8863), with a cosine of at least 0.9991 against lembed's vectors. The backend has two arithmetic contracts: `Exact = true` is bit-identical to the
scalar double reference on all cores; the default computes in float lanes like ggml and
is tested against the reference with stated tolerances.

### Measured throughput

BGE-M3 (`bge-m3-q8_0.gguf`, the file lembed uses) on the embed-de corpus: 320 texts,
16268 tokens, mean 51 tokens per text, one text at a time. `demos/infer/embed-bench`
runs both engines over the same loop (`embed-bench [gguf] all [threads|lembed]`); runs
alternate between the engines so that heat and background load hit both alike.

**x86_64, AVX2** — Intel Core i9-14900K (8 P-cores with HT + 16 E-cores, 32 threads),
Windows 11, FPC 3.2.2 win64, 2026-10-14:

| | lembed | native | native / lembed |
|---|---|---|---|
| default threads, run 1 / 2 | 524 / 525 tokens/s | 1039 / 1027 tokens/s (22 threads) | 1.98x / 1.96x |
| native with 8 / 16 / 24 / 32 threads | | 739 / 954 / 1040 / 959 tokens/s | |
| peak working set | 514 MB | 377 MB | 0.73x |

Started over SSH, both engines ran at about half that speed on this machine (lembed 187,
native 417 tokens/s): Windows power throttling slowed the windowless process down.
Switching off execution-speed throttling for the process
(`SetProcessInformation(ProcessPowerThrottling)`) restored the numbers above;
`embed-bench` does so unless `LAI_THROTTLE` is set. A Windows service doing bulk
embedding may want the same.

**x86_64 Linux, AVX2** — the same i9-14900K, Ubuntu 24.04 under WSL2, FPC 3.2.2,
2026-10-14:

| | lembed | native | native / lembed |
|---|---|---|---|
| default threads, run 1 / 2 | 544 / 545 tokens/s | 861 / 860 tokens/s (22 threads) | 1.58x / 1.58x |
| native with 8 / 16 / 24 / 32 threads | | 640 / 755 / 877 / 709 tokens/s | |
| peak RSS | 755 MB | 439 MB | 0.58x |

**aarch64, NEON dot product** — Snapdragon X (12 cores), WSL2, FPC 3.2.4, model on the
Linux file system, 2026-10-14:

| | lembed | native | native / lembed |
|---|---|---|---|
| 4 cores (taskset, 4 threads each), run 1 / 2 | 359 / 297 tokens/s | 361 / 341 tokens/s | 1.00x / 1.15x |
| whole machine, default threads, run 1 / 2 | 336 / 316 tokens/s | 385 / 315 tokens/s (8 threads) | 1.15x / 0.99x |
| peak RSS | 758 MB | 617 MB | 0.81x |

Peak RSS includes the pages of the mapped model file and moves with the page cache
(480 to 617 MB for native over several runs). The laptop throttles under sustained load;
the spread between runs is that, not noise in the engines. The default thread count is
two thirds of the logical cores.

**Qwen3-0.6B decoder** — Snapdragon X, WSL2, FPC 3.2.4, 8 threads, a 200-token prompt
then 32 new tokens (`demos/infer/llm-bench`), 2026-10-15:

| | prefill | decode | peak RSS |
|---|---|---|---|
| FP32, before phase 3 | 2.7 tokens/s | 1.6 tokens/s | 3.8 GB |
| FP32 | 203 tokens/s | 33 tokens/s | 3.8 GB |
| Q8_0 | 498 tokens/s | 64 tokens/s | 2.1 GB |
| FP32, 1000-token prompt | 138 tokens/s | 24 tokens/s | 4.1 GB |
| Q8_0, 1000-token prompt | 240 tokens/s | 40 tokens/s | 2.4 GB |
| Q8_0 GGUF file | 545 tokens/s | 65 tokens/s | 0.7 GB |
| Q4_K_M GGUF file | 290 tokens/s | 73 tokens/s | 0.5 GB |

The same on an Apple M2 Pro (macOS, FPC 3.2.4, 7 threads; two runs, which differ by
up to 12 %):

| | prefill | decode | peak RSS |
|---|---|---|---|
| FP32 | 162-181 tokens/s | 42 tokens/s | 3.8 GB |
| Q8_0 | 515-562 tokens/s | 70-75 tokens/s | 2.1 GB |
| FP32, 1000-token prompt | 147-160 tokens/s | 31-32 tokens/s | 4.1 GB |
| Q8_0, 1000-token prompt | 371-408 tokens/s | 44-47 tokens/s | 2.4 GB |

And on an Intel i9-14900K (Windows 11, FPC 3.2.2, 22 threads, AVX2), 2026-10-16:

| | prefill | decode | peak RSS |
|---|---|---|---|
| Q8_0 (from safetensors) | 746 tokens/s | 70 tokens/s | 2.1 GB |
| Q8_0 GGUF file | 757 tokens/s | 72 tokens/s | 0.7 GB |
| Q4_K_M GGUF file | 455 tokens/s | 78 tokens/s | 0.5 GB |
| Q8_0 GGUF, 1000-token prompt | 540 tokens/s | 44 tokens/s | 1.0 GB |
| Q4_K_M GGUF, 1000-token prompt | 369 tokens/s | 47 tokens/s | 0.8 GB |

**Qwen3-30B-A3B** (mixture of experts, 128 experts with 8 active per token,
`Qwen3-30B-A3B-Q4_K_M.gguf`, 18.6 GB) on the same machine, a 200-token prompt,
2026-10-17 to 19:

| | prefill | decode |
|---|---|---|
| CPU (22 threads, AVX2) | 85 tokens/s | 16 tokens/s |
| RTX 5090, CUDA | 383 tokens/s | 97 tokens/s |
| RTX 5090, Vulkan | 489 tokens/s (642 at 2000 tokens) | 90 tokens/s |

A decode step chooses the experts on the device (a route kernel, the k chosen experts in
one matrix product per projection): one CUDA graph or one Vulkan batch per token. With the
choice on the host, as before 2026-10-19, the same build decodes 39 (CUDA) and 43 (Vulkan)
tokens/s. A prefill still chooses on the host.

FP32 keeps greedy output token-identical to `transformers`; Q8_0 passes its own gate but
can flip a close decision - with it, Qwen3-0.6B answers the weather question of the
reference prompt itself instead of calling the tool. `TNativeLlmClient` still loads a folder
as Q8_0 by default (a quarter of the memory, about twice the decoding speed); pass
`idtF32` for tests and comparisons. `MaxContext` caps a conversation (and with it the KV
cache, which grows by doubling as the conversation does).

Plan, phases, gates and the full record: [docs/native-inference.md](docs/native-inference.md)
(German).

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
  // FPC 3.2 and Delphi 7 have no RTTI for plain records — declare the fields once.
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

Without generics (Delphi 7, or by choice) a tool derives from `TMcpRecordTool`, names its
record once and casts it back — `TMcpToolBase<T>` does exactly this for you. On Delphi 7 a
record only has `TypeInfo` if it holds a managed field (a string, say), so this one does:

```pascal
type
  TGreetParams = record
    name: RawUtf8;
    times: integer;
  end;

  TGreetTool = class(TMcpRecordTool)
  protected
    function ExecuteRecord(const aParams;
      const aAuthCtx: TMcpAuthContext): variant; override;
  public
    constructor Create(const aName, aDescription: RawUtf8);
  end;

constructor TGreetTool.Create(const aName, aDescription: RawUtf8);
begin
  inherited CreateTyped(TypeInfo(TGreetParams), aName, aDescription);
end;

function TGreetTool.ExecuteRecord(const aParams;
  const aAuthCtx: TMcpAuthContext): variant;
var
  p: ^TGreetParams;
begin
  p := @aParams; // the parsed record, of the type given to CreateTyped
  result := _ObjFast(['greeting', 'Hello ' + p^.name, 'times', p^.times]);
end;

// once, before the first tool is created - FPC 3.2 and Delphi 7 have no field RTTI:
//   if not RecordHasFields(TypeInfo(TGreetParams)) then
//     Rtti.RegisterFromText(TypeInfo(TGreetParams), 'name:RawUtf8 times:integer');
```

That `Rtti.RegisterFromText` line is the one thing that catches everyone: under FPC a plain
record carries no field RTTI, so the schema generator has nothing to read and the tool
shows up with an empty input schema. Register the record once at startup and it works.

Swap `TMcpStdioTransport` for the HTTP or Streamable transport and nothing else changes —
the server and its tools do not know which transport they are behind. Full examples in
[`demos/`](demos/).

## MCP — revision 2026-07-28, and only that one

The 2026-07-28 revision is the largest change since MCP launched, and it is **stateless**:
`initialize`, protocol sessions (`Mcp-Session-Id`), request batching, SSE resumability and
the legacy HTTP+SSE transport are **gone**. Every request carries its own protocol version
and client capabilities in `params._meta`; `server/discover` replaces the handshake.

We deleted the removed machinery rather than keeping it behind a negotiation layer. There
is **no multi-version support** — a greenfield project has nothing to be compatible with,
and a second code path is a second thing to get wrong.

**One temporary exception, marked for removal: the stdio legacy bridge**
(`mormot.ai.mcp.legacy`, code marker `LEGACY-ERA`). Clients still open stdio servers with
the old `initialize` handshake — Claude Code does so by default, and its `server/discover`
probe gives up after 3 s, which a slowly starting process misses. On stdio only, the bridge
answers `initialize` (2025-11-25 lifecycle), `notifications/initialized` and `ping`, then
passes requests without `_meta` to the unchanged 2026-07-28 core with the metadata filled
in. The spec allows a server to serve both eras. Requests that carry `_meta` stay purely
modern; `TMcpStdioTransport.AcceptLegacyInitialize := false` switches the bridge off. It
goes away once clients probe by default and survive a slow start — see
[CONCEPT.md §6](CONCEPT.md).

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
- **Native** (`mormot.ai.llm.native`, `TNativeLlmClient`): a model from a folder
  (safetensors) or one GGUF file on this machine, run by the native engine - Qwen3,
  Qwen3-MoE, Qwen2.5 or gpt-oss (on the CPU), chosen by `model_type` or the GGUF architecture
  (`RegisterDecoderModel`, `RegisterNativeFamily`). The prompt is the family's chat
  template; the answer is parsed token by token - a leading `<think>` block (gpt-oss: the
  analysis channel) becomes `TLlmChatResponse.Reasoning`, also as `reasoning_content` in
  `Raw` (as vLLM returns it), each `<tool_call>` a `TLlmToolCall`. The agent loop hands the
  reasoning of a turn with tool calls back (`TLlmMessage.Reasoning`), and the template puts
  it where the model expects it (gpt-oss: the analysis before the calls, Qwen3: the
  `<think>` block after the last user query); the OpenAI wire reads `reasoning_content` or
  `reasoning` and sends it back in `TLlmProviderConfig.ReasoningField` - `reasoning_content`
  by default (DeepSeek requires it in requests with tools), `reasoning` for Ollama; the
  Anthropic wire neither reads nor sends it. `ChatComplete` and
  `ChatStream` share one decoding path. Thinking is switched per request through
  `Extra.chat_template_kwargs.enable_thinking`, as with vLLM. Sampling follows the model's
  `generation_config.json` (as vLLM does) unless a request sets `Temperature` (0 = greedy)
  or `Extra` `top_p`, `top_k`, `min_p`, `seed`; a seed gives the same random stream on
  every compiler and CPU. Images and `ResponseFormat` are refused rather than ignored.
  On a GPU: `client.Device := TQwen3Vulkan.Create(client.Model, gpu)` (Vulkan, with a
  `TInferVulkanBackend` as the client's backend) or `TQwen3Cuda` - the same answers, the KV
  cache on the card (half floats by default: 16-32 % faster decoding at 16000 tokens of
  context on an RTX 5090; set `KvF16` false where floats are fed back step by step, as in a
  speech model's talker, since the rounding then adds up).

**Vision:** `TLlmMessage.Images` carries attachments (`LlmImageMessage`/`LlmImageBase64`/
`LlmImageUrl`); both wires serialise them (OpenAI `image_url` including base64 `data:` URIs,
Anthropic an `image` block with a typed `source`; an empty `MediaType` falls back to
`image/png`).

**Structured output** and **embeddings** run through the same seam; RAG combines
`IEmbedder` with `IVectorStore` (native store or sqlite-vec today, pgvector only when a
deployment needs it).

### Tests

The Anthropic driver test (`tests/test.llm.anthropic.pas`) is **hermetic** — no network, no
API key. It checks request construction (system hoisting, `input_schema`, tool round-trip),
response parsing (text/tool/usage, stop-reason mapping) and SSE decoding against canned
payloads, and runs as part of the suite.

**Native libraries are handled defensively.** `sqlite-vec` and `lembed` are external
`.so`/`.dll` files; the suite must catch the dynamic load and **skip cleanly** when they are
missing, never crash (see the env-gated `VectorStoreKeyedOps` test). That constraint goes
away once the **native engine** ([docs/native-inference.md](docs/native-inference.md))
and the native vector store replace the foreign libraries.

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

## Build & test

```bash
# from this repo, on Linux/WSL:
bash scripts/run-fpc-tests.sh        # MCP suite
bash scripts/run-fpc-llm-tests.sh    # LLM suite
bash scripts/run-fpc-infer-tests.sh  # native inference suite (reference data, no model needed)
LAI_INFER_GATE=1 bash scripts/run-fpc-infer-tests.sh  # + BGE-M3 Q8 corpus gate (nDCG@10, ~1 h scalar)
bash scripts/build-demo.sh           # all demos (or pass one, e.g. llm/llm-agent.lpr)
# VERBOSE=1 for the full compiler/test log; logs land in bin/fpc/
```

```powershell
# Delphi 13, on Windows:
.\scripts\run-delphi-tests.ps1               # MCP, LLM and inference suites (-Suite mcp|llm|infer, -ShowAll)
.\scripts\build-delphi-demos.ps1             # all demos (or -Demo infer\llm-bench.lpr)
.\scripts\build-delphi-demos.ps1 -Delphi7 -FastMM4 <dir>   # the same with Delphi 7
# logs and binaries land in bin\delphi\
# Delphi 7 (Win32):
.\scripts\run-delphi7-tests.ps1 -FastMM4 <dir>   # same suites; bin\delphi7\
```

**Requirements:** a mORMot2 source tree, plus
- **FPC 3.2.4** (currently the release candidate) for the bash scripts. They pin it
  (`scripts/fpc-env.sh`: a `fpc324` wrapper if one is on the PATH, else `fpc`) and stop on
  any other version, because a machine with several FPCs otherwise builds with whichever one
  the shell finds first. Override deliberately with `FPC=<compiler>`; `FPC_EXPECT=<version>`
  still insists on a version. Unit caches live under `lib/<fpc version>`, so builds with
  different compilers never share `.ppu` files.
- **Delphi 13** (`dcc64`, compiler 37.0) for `run-delphi-tests.ps1`; override with
  `-Dcc <path>`. On ARM64 Windows the x64 test programs run under emulation.

- **Delphi 7** (`dcc32`) for `run-delphi7-tests.ps1`: `-Delphi7 <install dir>` (or
  `DELPHI7`) and `-FastMM4 <dir with FastMM4.pas>` (or `FASTMM4`) — mORMot2 uses FastMM4
  as memory manager on Delphi 7, and it is a separate library. The `vec0` test is skipped
  there (no 32-bit build of the extension); the script says so.

Which compiler ran which suite on which platform: [Where it runs](#where-it-runs). Lazarus
is optional; `.lpi` files exist for the test projects, the scripts do not need the IDE.

The two native extensions (`sqlite-vec`, `lembed`) are **optional**: the tests that need
them skip without them; everything else, the native vector store and engine included,
runs.

**Locating mORMot2:** the scripts expect a mORMot2 source tree (`src/`, `static/`). With
no configuration they assume the submodule case — `<consumer>/shared/delphi/libs/_git_Synopse2`,
three levels above this repo. For a **standalone clone**, point at it explicitly:

```bash
MORMOT2_ROOT=/path/to/mORMot2 bash scripts/run-fpc-tests.sh
```

## Using it in your own project

There is no package file to install — add this repository's `src` (and `src/infer` for
native inference) plus mORMot2's source directories, and use the units. This is what the build scripts do; the flags below are lifted from
[`scripts/run-fpc-tests.sh`](scripts/run-fpc-tests.sh):

```bash
MORMOT2=/path/to/mORMot2
LAI=/path/to/mormot2-ai

fpc -MDelphi -Sci -Ci -O2 \
  -Fi"$MORMOT2/src;$MORMOT2/src/core;$MORMOT2/src/net" \
  -Fu"$LAI/src;$LAI/src/infer;$MORMOT2/src/app;$MORMOT2/src/core;$MORMOT2/src/crypt;$MORMOT2/src/db;$MORMOT2/src/lib;$MORMOT2/src/net;$MORMOT2/src/orm;$MORMOT2/src/rest;$MORMOT2/src/soa" \
  -Fl"$MORMOT2/static/$(fpc -iTP)-linux" \
  your-program.lpr
```

`-MDelphi` is not optional — the sources use Delphi-mode syntax, as mORMot2 itself does.
Start the program as mORMot2 programs do, with `{$I mormot.defines.inc}` and
`uses {$I mormot.uses.inc} ...`: on POSIX that brings in `cthreads`, which the MCP
transports and the native engine's thread pool need.
`-Fl` points at mORMot2's static libraries (the bundled SQLite among them); drop it only if
you use no unit that needs them.

In Lazarus, add `$(LAI)/src` and `$(LAI)/src/infer` to the project's *Other unit files* and mORMot2 per its own
instructions. Consuming this repository as a **git submodule** is the arrangement the
default paths assume — see "Locating mORMot2" under Build & test.

## Layout

```
mormot2-ai/
  src/
    mormot.ai.mcp[.server|.stdio|.tools|.legacy]  MCP: core, transports, sample tools
    mormot.ai.llm[.types|.openai|.anthropic]      LLM clients (+ .sse, .structured)
    mormot.ai.agent[.mcp]                         agent / tool-calling loop
    mormot.ai.embeddings, .embed.{native,provider,lembed}
    mormot.ai.vectorstore[.native|.sqlitevec]
    mormot.ai.sqlite                              SQLite helpers, extension loader
    mormot.ai.rag[.tool]
    mormot.ai.governor                            load governor for bulk jobs
    infer/mormot.ai.infer.*                       the native inference engine:
      tensor, memory, loader.{gguf,safetensors}, tokenizer[.ugm|.bpe], model.xlmr,
      decoder, model.qwen3[.vulkan|.cuda], chat, sampler, pipeline.embed,
      backend[.cpu|.vulkan|.cuda], kernels (NEON, AVX2), threads, vulkan, cuda
  tests/    mcp.tests · llm.tests · infer.tests (mORMot TSynTests), data/ (reference data)
  demos/    stdio · http · jsonrpc · streamable · llm/* (llm-quickstart: a local model in
            a few lines) · rag/* · infer/{embed,llm}-bench
  tools/    infer-ref (Python reference for the engine), vk and cuda (shader and kernel
            sources; the compiled SPIR-V / PTX is in git) - none needed to build
  docs/     native-inference.md · native-embeddings.md · specs/mcp-2026-07-28/
  vendor/   mormot.ext.os (adopted) · models, sqlite-ext (gitignored, not in git)
  scripts/  run-fpc-{tests,llm-tests,infer-tests}.sh · build-demo.sh · fpc-env.sh
            run-delphi-tests.ps1 · run-delphi7-tests.ps1 · build-delphi-demos.ps1
            x64-ext-verify.sh
  CONCEPT.md  DESIGN.md  LICENSE  NOTICE  SECURITY.md  UPSTREAM_BASE
```

## Maturity

Not everything here carries the same weight. "Live" means it was run against a real
provider or runtime, not only against tests.

| Area | State | Notes |
|---|---|---|
| MCP core, JSON-RPC, tool/resource registry | **stable** | hardened over five review rounds; a real consumer registers its tools and supplies the token verifier |
| Transports: stdio, HTTP, Streamable HTTP, in-process | **stable** | concurrency covered (parallel clients, keep-alive reuse) |
| MCP authorization (resource server) | **stable** | opt-in per spec (`OPTIONAL`), closed once a verifier is set; the verifier itself is supplied by the consumer |
| LLM client, OpenAI wire | **stable, live** | also covers LiteLLM and Ollama-compatible endpoints |
| Anthropic driver | **stable, live** | streaming, tool loop and structured output verified against a real model |
| Agent / tool-calling loop, structured output, vision | **stable, live** | |
| Embeddings via provider API | **stable** | |
| Native vector store (`mormot.ai.vectorstore.native`) | **stable** | no C extension; persisted model contract, SQL key filters, 1-bit RAM index with exact rescoring — measured at 2M × 1024 ([docs/native-embeddings.md](docs/native-embeddings.md)) |
| RAG (ingestion, retrieval, RAG-as-tool) | **works** | ingestion is strict and atomic; the rollback test still needs the runtime |
| Native embeddings (`mormot.ai.embed.native`) | **stable** | BGE-M3 from GGUF Q8_0; passes its quality gates on aarch64 and x86_64; AVX2 and NEON kernels; 90-minute soak test passed (79,628 texts, every vector bit-identical to the first pass); the recommended provider of the consuming backend |
| Decoder (`mormot.ai.infer.decoder`, `.model.qwen3`, `.tokenizer.bpe`, `.chat`) and its client (`mormot.ai.llm.native`) | **works** | Qwen3, Qwen3-MoE and Qwen2.5, SmolLM3 without a chat template; sliding windows with a cache that keeps twice the window and restore points the prompt cache respects, layers without rotary positions (NoPE), each checked bit for bit against a full cache; tokenizer: 2416 cases identical to Hugging Face, fuzzed against its pre-tokenizer, from `tokenizer.json`, from `vocab.json` + `merges.txt` (Qwen3-TTS ships no `tokenizer.json`) or from GGUF; chat templates identical to `transformers` (26 conversations each for Qwen3 and Qwen2.5, reasoning given back included); greedy output token-identical in FP32 (Qwen3-0.6B 7 prompts, Qwen2.5-0.5B 6, tiny MoE models); behind `ILlmClient` with streaming, tool calls and reasoning, agent loop tested; FP32, Q8_0, Q4_0, Q4_K_M (with Q5_0, as Qwen's own Q4_K_M file of Qwen2.5-0.5B has it); greedy or sampled (temperature, top-k, top-p, min-p, seed); prompt cache for the next turn. Embedding rows instead of ids (`ForwardEmbeds`, bit for bit the logits of the ids) and stacks of larger checkpoints (`CreateStack`): the talker and code predictor of Qwen3-TTS against PyTorch (states within 1.8e-4); a model must be freed before its backend (it forgets its GPU copies, `ForgetWeights`). Sliding windows and NoPE also on Vulkan and CUDA (the device cache keeps every row, so a context goes back to any length). gpt-oss on the CPU: attention sinks, YaRN, biases on the output, router and experts, the clamped SwiGLU, against `transformers` with a tiny model (every state, greedy); gpt-oss-20b from its MXFP4 checkpoint (three files with an index) on the CPU against `transformers` in float32 (logits at cosine 0.99999 for plain text, 0.993 for a chat turn - the error of Q8_0 activations in the experts, as in ggml; greedy identical), decoding at 5-8 tokens/s on an i9; also from llama.cpp's GGUF (`ggml-org/gpt-oss-20b-GGUF`: every expert byte for byte as repacked from safetensors, loaded in 5 s); its o200k tokenizer from `tokenizer.json` or GGUF (3095 cases identical to the tokenizers library, piece by piece and id by id; text stays text); its Harmony chat template identical to `transformers` (444 conversations and tool schemas, text and ids, the reasoning before a tool call given back) and its output parser (reasoning, content, tool calls) - gpt-oss-20b behind `ILlmClient` answers with reasoning and calls tools; the GPU is still to come. Not yet: Q5_1 and Q5_K (Q5_K_M files) |
| GPU backends (`mormot.ai.infer.vulkan`, `.cuda`, `.model.qwen3.vulkan`, `.model.qwen3.cuda`) | **works** | the whole forward pass, experts included, checked against the CPU on every run (logits cosine, argmax, for MoE the expert choice separately); Vulkan on an RTX 5090, an Adreno X1-85 and an M2 Pro (MoltenVK), CUDA on the RTX 5090 - mixture of experts with Qwen3-30B-A3B on the RTX 5090, on the other GPUs with the tiny MoE model; flash attention, F16 KV cache, the experts of a decode step chosen on the device, a decode step as one CUDA graph |
| Load governor (`mormot.ai.governor`) | **works** | tested on a fake clock and a fake sysfs tree; the temperature path needs a Linux CPU sensor, which WSL2 does not expose - not yet seen on real sensor hardware |
| Embeddings via `lembed` (local GGUF) | **works, needs a native library** | shared per model name; calls are serialised (its context is not re-entrant); kept as fallback and as the oracle of the native engine |
| Vector store `sqlite-vec` | **works, needs a native library** | superseded by the native store; real-extension test is env-gated (`SQLITE_EXT_DIR`) |
| pgvector backend, Ollama embedder | **not built** | the `IVectorStore` / `IEmbedder` seams exist; built when a deployment needs them |

The "needs a native library" rows are **optional dependencies**: an application that
selects one of these adapters has to provide its library (or handle the error its
constructor raises without it); the tests that need one are skipped when it is missing.
The libraries are **not in git** — `vendor/models` and `vendor/sqlite-ext` are ignored
and have to be provided locally.

## Roadmap

Done:
- ✓ **MCP** — server adopted and renamed into `mormot.ai.*`, moved to revision 2026-07-28
  (stateless) with the removed machinery deleted, used by a real consumer.
- ✓ **LLM** — clean-room client: OpenAI wire and Anthropic, agent/tool loop, structured
  output, vision, embeddings, RAG.
- ✓ **Compiler range of mORMot2** — FPC and Delphi 7 to 13, non-generic core.
- ✓ **Native vector store** — replaces sqlite-vec.
- ✓ **Native inference, phases 0 to 1b and the speed work of 1c** — reference data,
  BGE-M3 in FP32 and from the Q8_0 GGUF, NEON and AVX2 kernels, measured against lembed.
- ✓ **Qwen3 decoder up to its gate** — byte-level BPE tokenizer and chat template
  identical to Hugging Face on every reference case; Qwen3-0.6B in FP32 with KV cache,
  greedy output token-identical to `transformers` for all seven reference prompts.
- ✓ **Qwen3 behind `ILlmClient`** — `mormot.ai.llm.native`: streaming, stop tokens, tool
  calls, reasoning; the agent loop runs a tool with the local model.
- ✓ **Qwen3-MoE and Qwen2.5** — mixture of experts (Qwen3-30B-A3B) and the Qwen2.5
  family on the CPU, Vulkan and CUDA, both behind `ILlmClient` (see item 3 below).

Next, in this order:
1. **Quantisation and speed** — done: Q8_0 weights, SIMD and threads for the decoder
   (decoding 15x faster in FP32, 22x in Q8_0). Sampling is done too (temperature, top-k, top-p, min-p,
   reproducible per seed), and so is a prompt cache (the next turn of a conversation
   only computes what is new), and norms, RoPE and SwiGLU run on all cores (prefill
   up to 1.8x faster). Q4_0 works (byte-identical to ggml) but is experimental: too
   coarse for a 0.6B model; it now decodes faster than Q8_0. Qwen3
   loads straight from llama.cpp GGUF files (Q8_0, Q4_0, F16), tokenizer included, so
   one .gguf file is a complete native `ILlmClient`, and Q4_K_M files load too (Q4_K
   and Q6_K, checked against ggml itself, with fused NEON / AVX2 kernels: 73 tokens/s
   decoding in 475 MB). Open items: [docs/backlog.md](docs/backlog.md).
2. **GPU** — the whole Qwen3 forward pass runs on Vulkan: an RTX 5090 prefills Qwen3-0.6B
   at about 13600 and decodes at about 365 tokens/s (Q8_0; the i9 CPU: 757 and 72), with
   flash attention still about 330 tokens/s at 4000 tokens of context; `TNativeLlmClient`
   decodes there with `client.Device := TQwen3Vulkan.Create(client.Model, gpu)`. On a Mac
   the same backend runs through MoltenVK (Homebrew `molten-vk` and `vulkan-loader`); on
   the M2 Pro it beats the NEON CPU path (prefill 1254 against 563, decode 98 against
   73 tokens/s at 200 tokens of context). Vulkan first (the driver's loader at run time, shaders as embedded SPIR-V,
   no shipped DLLs), CUDA through the driver API as the second backend - done too:
   `TQwen3Cuda` (kernels as embedded PTX, a decode step as one CUDA graph) prefills about
   12400 and decodes about 404 tokens/s on the RTX 5090 (Q8_0; Vulkan 13700 and 358).
3. **More decoder families (phase D)** — under way. Done: shared decoder types and two
   registries, so a family is one unit (D1); mixture of experts on the CPU, CUDA and
   Vulkan with Qwen3-30B-A3B (D2, D3a, D3b), and a routing tape that lets the GPU
   comparison check computation and expert choice separately; Qwen2 / Qwen2.5 with QKV
   bias and its own chat template (D1b); sliding-window attention with a compact cache,
   valid restore points for the prompt cache and layers without rotary positions, SmolLM3
   as the reference family (D4, CPU); the expert choice of a decode step on the GPU (D3c,
   Qwen3-30B-A3B from 39 to 97 tokens/s on CUDA); Q5_0 weights, so Qwen's own Q4_K_M files
   of small models load (B16); Q8_0 as the native client's default, `MaxContext` and a
   quickstart demo (B15); CPU results bit-identical under Delphi 13 and FPC, Delphi
   decoding 3-14 % faster with it (B13); embedding rows instead of ids and decoder stacks,
   for speech, vision and audio models (B14: Qwen3-TTS's talker and code predictor, its
   `vocab.json` + `merges.txt` tokenizer, BF16 and F16 rows in `InferRowToF32`,
   thread-pool jobs with scratch buffers per thread); sliding windows and NoPE on Vulkan
   and CUDA (D4b); gpt-oss on the CPU - attention sinks, YaRN, biases, the clamped SwiGLU,
   against transformers with a tiny model (D5a); MXFP4 weights and safetensors split into
   several files, gpt-oss-20b on the CPU against transformers (D5b), also from GGUF
   (D5b2), its o200k tokenizer (D5c1) and its Harmony chat format behind `ILlmClient` (D5c2).
   Next: the GPU (D5d); then Gemma. Plan and order: [docs/backlog.md](docs/backlog.md)
   (German).
4. **After that, on demand** — a reranker, Qwen3-Embedding, BGE-M3 sparse/ColBERT output.

Alongside, as a separate track: **distributed execution** - big inference jobs pulled
by idle machines on the network (task parallelism with leases, not model parallelism;
embeddings in the LAN first, LLM batch jobs later), see
[docs/distributed-execution.md](docs/distributed-execution.md) (German). Also alongside:
the first real consumer of the load governor (a bulk indexing job in the backend), agent loop limits (cancellation, deadlines, context budget),
token-based chunking. The German [CONCEPT.md](CONCEPT.md) §9 is
the single source of truth for open points.

## Why a standalone extension

- **Not an upstream mORMot PR** (decided 2026-07-29). mORMot2 stays the **foundation**,
  not the delivery target: an own repository keeps our own pace and platform goals
  (native inference engine, CUDA) and lets MCP follow the 2026-07-28 spec alone.
  Rationale: [CONCEPT.md §1](CONCEPT.md).
- **FPC and Delphi 7 to 13** — the same compiler range as mORMot2. FPC is the reference
  compiler and the gate; Delphi 13 (Win64) and Delphi 7 (Win32), the two ends of the
  range, build the library and run all test suites.
  Parts that only exist under FPC — the POSIX process launcher in `vendor/` — stay behind
  `{$ifdef FPC}` instead of being duplicated. The library needs no generics: the MCP
  registries are `TSynDictionary`, typed tools build on `TMcpRecordTool`, and the generic
  `TMcpToolBase<T>` is only a shorthand where the compiler has generics.
- **Written like mORMot2** — unit headers, `{ ****** section }` layout, `RawUtf8`,
  `TDocVariantData`, `ESynException`, `f` field prefix, `///` docs. Stricter than mORMot2
  where Delphi 7 or readability needs it: a non-generic core (generics only as an optional
  typed shell under `HASGENERICS`), no anonymous methods, no inline variables, no
  `exit(value)`, no `with`, and an `a` prefix on our own parameters. The full rule set
  is in [CONCEPT.md §1](CONCEPT.md).
- **Spec-driven, not diff-driven** — we follow the published specs (MCP, JSON-RPC,
  provider wires) rather than chasing another library's changes.
- **License MPL/GPL/LGPL** (the mORMot three-license, confirmed 2026-10-14 — also for the
  native inference engine) — the adopted MCP units require it, and it composes with
  MIT-licensed ported code and LGPL-with-linking-exception.

## Origin and license

The MCP server units were **adopted** from
[flydev-fr/mormot2-extensions](https://github.com/flydev-fr/mormot2-extensions)
(mORMot-licensed) and renamed into the `mormot.ai.*` namespace; the original per-unit
headers are retained. Details: [NOTICE](NOTICE), [LICENSE](LICENSE), [DESIGN.md](DESIGN.md).

The vendored MCP specification under `docs/specs/` is **not** part of this work and not
covered by the three-license above — it keeps its own upstream license alongside it.

### Relationship to flydev — code adoption, not a fork

Our adoption from flydev so far comes from `mormot2-extensions`: the MCP units, their
tests and demos, and `mormot.ext.os`. From
[flydev-fr/mormot.ai](https://github.com/flydev-fr/mormot.ai) we may adopt **code only**,
selectively (pgvector, if a deployment needs it): no fork, no shared repository, no
upstream PRs. Its evaluation baseline is **pinned** in [UPSTREAM_BASE](UPSTREAM_BASE) so
that a later adoption can work from a diff; nothing from it is in this repository yet.
Both projects use the `mormot.ai.*` namespace — fine while they are used separately; a
program combining both has to mind the overlapping unit names.

> **Rule: whoever ports from upstream bumps `UPSTREAM_BASE` in the same commit.** Without
> it the diff base is lost and the next round has to guess. The same file records the
> templates of the native inference engine, the model revision and file hashes behind its
> reference data, the Python environment that produced them, and the mORMot2 commit the
> suites last ran against.

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
