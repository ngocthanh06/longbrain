# Strategy: from Memory Store to Context OS

This is the product/commercial roadmap — distinct from [ROADMAP.md](ROADMAP.md)
(the engineering "what's live, what's next" tracker). This file captures the
positioning and phased plan agreed on 2026-07-28; it does not change any
current engineering priority.

## Positioning

Don't sell LongBrain as "a vector database" or "a RAG framework" — that's
the engine, not the product. Sell it as the layer that decides **what
matters and what doesn't**, not the layer that stores more.

The durability test: if Claude/GPT/Gemini ship a 10M-token context window
tomorrow, does LongBrain still matter?

- If the answer is "it lets you stuff in more data" — no, and it gets
  commoditized by the model vendors.
- If the answer is "it knows which data is authoritative, which is stale,
  which is irrelevant to this task, and assembles the right context for
  it" — yes, and that value survives context-window growth.

That second answer is the moat. It has nothing to do with multi-user or
cloud — it holds even for a single local user with one project.

## Where this does NOT go (reaffirmed non-goals)

Same as [ROADMAP.md § Deliberate non-goals](ROADMAP.md#deliberate-non-goals):
single-user, local-first, no forced multi-tenant sync. The reason isn't a
technical limitation — it's a deliberate operating decision (nobody has to
host, auth, or support anyone else's data). A later "Enterprise" layer
(cloud sync, RBAC, audit, hosted workspace) is possible without a rewrite
*if* the connector/permission interfaces are designed cleanly — but
whether to build it is a **business decision** (do we want to run a
hosted multi-tenant service?), not something a clean architecture
resolves on its own. Not in scope until that decision is made separately.

## Roadmap

### Phase 1 — Memory Reliability (current, in flight)

Not speculative — these are live issues in the codebase today
(`documents.py`, `migrate_document_versions.py`, `mcp_server.py` currently
being touched):

- Document versioning
- Admission policy (what gets let into context, what doesn't)
- L2 isolation (chat-history recall not anchoring/polluting other layers)
- Context budgeting (how much gets injected per call)
- Traceability (why did this fact/chunk get surfaced)
- Behavioral evaluation

No new abstractions here — these are fixes to the existing 4-layer memory
system, not a new subsystem.

### Phase 2 — First real connector

Pick the single highest-value source (candidate: Google Docs/Drive).
Build it as a plain module (`connectors/google_drive/`), no generic SDK,
no base class. It should target the document shape that already exists in
`documents.py` (a `llama_index.core.Document` + the existing metadata
convention: `stored_path`, `project_id`, `user_id`, content-addressed
storage) rather than inventing a new one — that shape is effectively
already the "SourceDocument" model; reuse it, don't wait for it to
"emerge" later.

### Phase 3 — Second connector, deliberately different shape

Not another document source. Pick something structurally unlike Phase 2
— e.g. GitHub (repo/issue/PR, event-based) or Slack (message stream,
webhook-driven, complex permissions) rather than Notion (same
"document connector" shape as Docs). The point is to force the design
through two genuinely different data/auth/sync shapes, not just add a
second integration.

### Phase 4 — Extract abstraction from real friction (provisional)

Only after Phase 2 and 3 exist, compare them and ask: what's essentially
shared vs. coincidentally similar? Expect the answer to be a few small
protocols (e.g. `ResourceFetcher`, `ChangeSource`, `ResourceNormalizer`)
or just a shared output model — not necessarily one `BaseConnector`.

Treat whatever comes out of this as **provisional**, not final: two
examples can't reliably distinguish essential similarity from
coincidence (no third point to triangulate against). Don't lock the
interface until a third connector confirms it holds.

The "Connector Lifecycle" (register → authenticate → discover → initial
sync → incremental sync → version → chunk → embed → index → health check
→ remove) stays a **design checklist** through this phase, not a coded
base class or state machine — auth alone varies too much (OAuth vs PAT vs
webhook signing) to force a shared implementation before real connectors
expose the actual variance. Describe it as **capabilities**
(supported/unsupported per connector: auth, discovery, initial sync,
incremental polling, webhook ingestion, deletion detection, health
reporting), not a mandatory linear workflow — some connectors skip steps
entirely (a filesystem watcher has no "authenticate").

### Phase 5 — SDK, router, lifecycle runtime

Only once Phase 4's abstraction has survived a third connector: formalize
Connector SDK, source router/priority, sync scheduler, lifecycle-as-code,
retry policy.

### Phase 6 — Commercial layer (separate decision, not an engineering phase)

Cloud sync / RBAC / audit / hosted workspace / connector marketplace.
Revisit only when there's an explicit decision to operate a hosted
multi-tenant service — not before, and not as a side effect of clean
connector architecture existing.

## Guardrails carried over from this discussion

- No abstraction before 2+ real, deliberately-divergent instances exist;
  no "final" abstraction before a 3rd confirms it.
- Reuse the existing `Document`/metadata shape in `documents.py` for the
  first connector instead of inventing a new one.
- Lifecycle/capability model is a checklist until Phase 5, never a base
  class or enforced state machine before that.
- "Can be layered in later" (cloud/enterprise) is a business decision
  gate, not a code-readiness gate.
