# Security Policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Use GitHub's private vulnerability reporting on this repository
(*Security → Report a vulnerability*), or write to **info@landrix.de** with
`mormot2-ai` in the subject line.

Useful in a report, roughly in order of value:

1. What an attacker gains — read access, forged authorization, denial of service, remote
   code execution.
2. The smallest reproduction you have: a request, a spec-violating payload, a unit test.
3. Which version — commit hash, or the `UPSTREAM_BASE` pin if you are on adopted code.
4. Whether the issue is in **this** extension or in mORMot2 underneath it (see below).

You will get an acknowledgement. This is a small project without a staffed security team,
so please do not expect an SLA — but a report that lands will be looked at, and you will
hear what came of it.

## Scope

This repository is a library, not a deployed service. What is in scope is code that
processes **untrusted input**:

- **`mormot.ai.mcp*`** — the primary attack surface. It accepts JSON-RPC from clients over
  stdio, HTTP and Streamable HTTP. Anything that lets a request bypass authorization, read
  a resource it should not, corrupt another request's state, or crash the process belongs
  here.
- **`mormot.ai.llm*` / `.agent*`** — outbound, but it parses provider responses. A
  malformed or hostile response that leads to a crash, an injected tool call, or a leaked
  API key is in scope.
- **`mormot.ai.rag*` / `.vectorstore*` / `.embed*`** — path handling, extension loading and
  ingestion of untrusted documents.

Out of scope, with reasons:

- **mORMot2 itself.** Report those to [synopse/mORMot2](https://github.com/synopse/mORMot2)
  — we are a consumer, and a fix here would be a workaround at best.
- **The vendored MCP specification** under `docs/specs/`. It is a verbatim copy of an
  upstream document. A problem with the *specification* goes to the MCP project; a problem
  with our *implementation of it* is in scope and welcome.
- **Demos** (`demos/`). They are illustrative and deliberately permissive — one binds
  loopback only and one is disabled unless explicitly opted in, but they are not written to
  be exposed. Do not deploy them.
- **Missing hardening without a path to exploitation.** Useful, but an issue, not an
  advisory.

## Two things worth knowing before you test

**Authorization is fail-closed by design.** With no `IMcpTokenVerifier` set, the core
refuses every request. If you find a path that reaches a tool without passing the verifier,
that is a serious finding — it inverts the intended default.

**The protocol revision is 2026-07-28, and only that one.** There is no negotiation and no
older code path. A report resting on `initialize`, protocol sessions, request batching or
the legacy HTTP+SSE transport is describing machinery that was deleted, not a gap. The
normative text is vendored under
[`docs/specs/mcp-2026-07-28/`](docs/specs/mcp-2026-07-28/) — citing it makes a report much
easier to act on.

## Disclosure

We would rather fix an issue before it is public. Tell us what timeline you plan to work
to; if we cannot meet it, we will say so rather than go quiet. Credit in the release note
if you want it.
