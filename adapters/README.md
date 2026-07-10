# Writing an adapter

The memory core is agent-agnostic: Qdrant + the memory service at
`http://localhost:8800`, four layers (L1 working buffer, L2 raw history,
L3 distilled facts, L4 documents). An **adapter** is the thin glue that
plugs one AI agent (Claude Code, Hermes Desktop, Cursor, Codex, ...) into
that shared brain. Every record carries a `source_agent` label, so any
number of agents can run in parallel on the same memory.

Two reference implementations ship in this repo:

- `hooks/claude/` — Claude Code (4 lifecycle hooks, stdlib-only python)
- `hooks/` — Hermes Desktop (`pre_llm_call.py`, `post_llm_call.py`,
  `on_session_end.py`, `on_session_start.py`)
- `hooks/codex/` — Codex turn-ended notify sync (write-only lifecycle)

`python_minimal/adapter.py` in this folder is a single-file skeleton to
copy from.

## Support tiers

Not every agent exposes lifecycle hooks, so support comes in tiers. The
core stays agent-agnostic either way — a tier describes how much of the
lifecycle an agent's adapter can automate:

- **Full adapter** — all 4 lifecycle moments are automatic: recall before
  every prompt, append after every response, consolidate on session end,
  catch-up on session start.
- **Write adapter** — completed turns are recorded automatically, and MCP
  tools are available, but recall/consolidation still require model/tool
  initiative because the agent has no pre-prompt/session lifecycle hook.
- **MCP-only** — the model can call the memory tools on its own initiative
  (`memory_recall`, `save_memories`, `search_history`, …) but the agent has
  no hooks, so turns are **not** recorded automatically.
- **Generic MCP client** — anything speaking MCP can connect manually;
  no setup automation.

| Agent | Recall auto | Write chat auto | Consolidate auto | Setup |
|---|---|---|---|---|
| Claude Code | ✅ | ✅ | ✅ | `./setup.sh` |
| Hermes Desktop | ✅ | ✅ | ✅ | `./setup.sh` |
| Codex | tools only | ✅ via `notify` | ❌ | `./setup.sh` (write adapter) |
| Any MCP client | tools only | ❌ | ❌ | point it at `http://localhost:8800/mcp` |

`./setup.sh` detects installed agents and wires each at the highest tier it
supports; `python3 scripts/doctor.py` verifies the wiring afterwards. When
an agent grows lifecycle hooks (or a wrapper becomes worth building), its
tier upgrades by adding an adapter — the core doesn't change.

## The contract: 4 lifecycle moments

An adapter is complete when it covers these four moments. All calls are
best-effort HTTP to `localhost:8800` — a down memory stack must NEVER
break the agent's own conversation.

| # | Moment | Call | Purpose |
|---|--------|------|---------|
| 1 | **before prompt** | `POST /memory/recall` | fetch relevant memory, inject `context_block` into the turn's context |
| 2 | **after response** | `POST /memory/append` | write the finished user/assistant turn into L2 |
| 3 | **session end** | `POST /memory/consolidate` | distill the session into L3 facts + a session summary |
| 4 | **session start** | `POST /memory/consolidate-pending` | catch-up sweep for sessions that missed (3) (crash, force-quit) |

### 1. before prompt — recall

```
POST /memory/recall
{
  "query":        "<the user's prompt, first 2000 chars>",
  "session_id":   "<your session id>",
  "project":      "<project slug — see resolver below>",
  "recent_turns": 0
}
```

- `recent_turns: 0` always — your agent already carries its own live
  conversation; re-injecting it doubles token cost for nothing.
- Inject `context_block` from the response, prefixed with something like
  `"Long-term memory (auto-recalled):"`. Cap it (`LONGBRAIN_MEMORY_MAX_CONTEXT`,
  default 6000 chars) and skip the call entirely for prompts shorter than
  ~15 chars ("ok", "continue") — they carry nothing to search for.
- The service routes channels by itself (facts + related history always;
  document chunks when the prompt mentions specs/docs) and returns its
  decision in `routing`.

### 2. after response — append

```
POST /memory/append
{
  "session_id":         "...",
  "user_message":       "<the user message>",
  "assistant_response": "<the final response text>",
  "project_id":         "<slug>",
  "project_source":     "folder",
  "source_agent":       "<your-agent-name>"
}
```

One call per finished turn. Idempotent — point ids are deterministic,
retries are safe. `project_source: "folder"` marks the slug as a genuine
workspace signal so the server's session-stickiness override applies.

### 3. session end — consolidate

```
POST /memory/consolidate   {"session_id": "...", "background": true}
```

If the service has its own LLM it distills alone. With `LLM_PROVIDER=none`
(the default, no API key anywhere) distillation runs through **your agent's
model** instead: expose the MCP server (below) and the agent calls the
`consolidate_session` tool, follows the returned instructions, then hands
facts + a session summary back via `save_memories`.

### 4. session start — catch-up

```
POST /memory/consolidate-pending   {}
```

Debounced server-side; calling it on every session start is fine. Bonus
points: probe `GET /health` first and surface a loud warning in the agent's
context when the service is unreachable — silent memory loss is the worst
failure mode this stack has.

## Project resolver

Memory is project-scoped (`project` slug). Resolve it consistently or the
same folder gets different memories from different agents:

1. match the working directory against Hermes' `~/.hermes/projects.db`
   (authoritative when Hermes Desktop is installed);
2. else: the git-root folder name, slugified (`[a-z0-9_-]`, max 64);
3. else: `"default"`.

When you resolve via (2), also record the folder in
`~/.hermes/discovered_projects.json` (see `hooks/project_catalog.py`) so the
host-side docs/ ingest watcher can find the project without Hermes.
`hooks/claude/common.py:resolve_project` implements the full chain.

## MCP (active memory access for the model)

Beyond the passive hooks, register the MCP server so the model can search
and save on its own initiative:

- transport: Streamable HTTP, `http://localhost:8800/mcp`
  (clients must send `Accept: application/json, text/event-stream`)
- key tools: `memory_recall`, `search_history`, `search_knowledge_base`,
  `save_memories`, `consolidate_session`, `forget_about`

## Token discipline (learned the expensive way)

- never inject the session's own recent turns back into it
- skip recall for sub-15-char prompts
- cap every injected block
- log nothing by default (`LONGBRAIN_DEBUG_HOOKS=1` to opt in, truncated) —
  hook payloads contain full prompts

All of these are env-tunable; keep the same env names
(`LONGBRAIN_MEMORY_MAX_CONTEXT`, `LONGBRAIN_RECALL_MIN_PROMPT_CHARS`,
`LONGBRAIN_MEMORY_URL`, ...) so one configuration governs every adapter.
