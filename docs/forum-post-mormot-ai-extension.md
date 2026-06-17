# Forum post — proposing a `mormot.ai.*` extension (MCP server, LLM clients)

> Target board: **synopse.info/forum** (mORMot 2). English. Adjust the intro to
> taste before posting.

---

**Subject:** Proposing a `mormot.ai.*` extension — Model Context Protocol (MCP) server, on top of mORMot's net/json stack

Hi Arnaud, hi all,

We run mORMot 2 in production (FPC on Linux, native, no Indy/Wine) as the backend
of a SaaS for the trades. We now need an **MCP server** so our backend can expose
tools/resources to AI agents (e.g. Claude Desktop and our own agent), and ideally
LLM client drivers later on.

Before writing much code I'd like to check interest and get your guidance, because
I think this belongs **in the mORMot world**, not as yet another standalone lib.

## The gap

As far as I can see, mORMot has no AI/LLM/MCP module today. Meanwhile mORMot
already ships everything such a module needs and would do it better than the
Delphi-only options out there (which lean on Indy and `System.JSON`/`System.Rtti`):

- `mormot.net.server` / `mormot.net.async` — HTTP server for the *Streamable HTTP*
  MCP transport
- `mormot.net.ws.*` — WebSockets for streaming
- `mormot.core.json` / `mormot.core.variants` — JSON-RPC 2.0 envelopes with
  `TDocVariant`
- `mormot.core.rtti` — generating JSON-Schema for tool input from typed records
- cross-compiler (FPC + Delphi), cross-platform, no extra dependencies

## What I'm proposing

A small, **clean-room** extension under a new `mormot.ai.*` namespace, built
against the official **MCP specification (revision 2025-11-25)** and JSON-RPC 2.0
— not derived from any third-party MCP code. Initial, transport-neutral layout:

```
mormot.ai.mcp.types            JSON-RPC/MCP types + envelope build/parse
mormot.ai.mcp.server           engine: tool/resource registry + JSON-RPC dispatch
                               (initialize, tools/list, tools/call) — JSON in/out
mormot.ai.mcp.transport.stdio  stdin/stdout (local subprocess transport)
mormot.ai.mcp.transport.http   Streamable HTTP on mormot.net.server
```

The engine takes JSON in and returns JSON out, so it is fully unit-testable
without a socket; transports are thin adapters. Tools implement a small
`IMcpTool` interface (`GetName`/`GetDescription`/`GetInputSchema`/`Execute`).

I already have the first unit (`mormot.ai.mcp.types`) and FPCUnit tests for the
JSON-RPC envelopes; the rest is staged behind it.

## Questions

1. **Interest & home** — would you welcome such an extension under the
   `mormot.ai.*` namespace (eventual upstream into mORMot), or would you prefer
   it lives as an external companion package? Either is fine for us; I'd just
   like to build it the way that has the best chance of being useful to others.
2. **Namespace & conventions** — if upstream is on the table, is `mormot.ai.*`
   the naming you'd want (e.g. `mormot.ai.mcp.server`)? Any coding-style or CLA
   requirements I should follow from the start?
3. **Transport** — for Streamable HTTP, what's your recommended building block:
   plain `THttpServer`/`THttpAsyncServer` with manual chunked writes, or is there
   a pattern you'd point me at for long-lived streaming responses + an optional
   WebSocket upgrade?
4. **JSON-RPC API shape** — for a framework-grade API, would you prefer the
   envelopes/dispatch built on `TDocVariant` (what I have now) or on typed records
   with `mormot.core.rtti` serialization? Happy to follow your taste here.

I'm willing to maintain this and contribute it back. Thanks for mORMot — it has
been a joy to build a native FPC backend on it.

Best regards,
Sven
