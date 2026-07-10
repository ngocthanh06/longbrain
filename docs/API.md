# API Reference

The memory service listens on `http://localhost:8800` (host port → container
port 8000). Two surfaces share one engine:

- **REST** — what the adapter hooks and scripts call.
- **MCP tools** (`http://localhost:8800/mcp`, Streamable HTTP) — what the
  agent's model calls actively during a chat.

Conventions used below:

- All request/response bodies are JSON unless noted.
- `project` / `project_id` is a project slug: lowercase `[a-z0-9_-]`, max 64
  chars. Empty means the default project.
- Point ids are deterministic, so retrying a write is always safe.

---

## REST

### Health

#### `GET /health`

Service status plus "is memory actually being written?" — `last_written_at`
must advance after every chat turn. First thing to check when memory seems
silent.

```bash
curl localhost:8800/health
```

```json
{
  "status": "ok",
  "embed_provider": "fastembed",
  "embed_model": "paraphrase-multilingual-MiniLM-L12-v2",
  "embed_dim": 384,
  "llm_provider": "none",
  "llm_model": null,
  "schema_version": 2,
  "last_written_at": 1752130000.0,
  "collections": {
    "longbrain_documents": 450,
    "longbrain_chat_history": 385,
    "longbrain_memories": 308
  }
}
```

### Knowledge base (L4)

#### `POST /ingest/text`

Add a piece of text to the document knowledge base.

```bash
curl -X POST localhost:8800/ingest/text -H 'Content-Type: application/json' \
  -d '{"text": "Content...", "metadata": {"source": "faq.md"}, "project_id": "erp"}'
```

```json
{"status": "ok", "total_chunks_indexed": 451}
```

#### `POST /ingest/file`

Upload a document (`.pdf .md .txt .docx`). The original file is stored under
`/data/documents`; re-sending an unchanged file is skipped
(`"status": "skipped_duplicate"`). This is what the `docs/` folder watcher
calls — you rarely need it by hand.

```bash
curl -X POST localhost:8800/ingest/file \
  -F "file=@document.pdf" -F "project_id=erp"
```

```json
{"status": "ok", "total_chunks_indexed": 462}
```

#### `POST /query`

Semantic search over ingested documents. `project` is a **hard filter**
(documents never leak across projects).

```bash
curl -X POST localhost:8800/query -H 'Content-Type: application/json' \
  -d '{"query": "upload size limit", "top_k": 5, "project": "erp"}'
```

```json
{"results": ["...chunk text...", "..."]}
```

### Memory lifecycle (L2 + L3)

#### `POST /memory/append`

Record one completed chat turn into episodic memory. Called by the
`post_llm_call` / Stop hooks after every turn — no LLM involved, idempotent.
`project_source` tells the server how the hook resolved the project
(`folder` / `active` / `default`) so session stickiness can decide whether a
change of project is intentional.

```bash
curl -X POST localhost:8800/memory/append -H 'Content-Type: application/json' \
  -d '{"session_id": "s1", "user_message": "...", "assistant_response": "...",
       "project_id": "erp", "project_source": "folder", "source_agent": "claude-code"}'
```

```json
{"status": "ok", "appended": 2, "project": "erp", "last_written_at": 1752130000.0}
```

#### `POST /memory/recall`

The heart of the system: merge distilled facts (L3), related past
conversations (L2) and the current session's recent turns (L1) into one
ready-to-inject context block. Called by the pre-prompt hook before every
turn. Empty `context_block` means "nothing relevant — inject nothing".

```bash
curl -X POST localhost:8800/memory/recall -H 'Content-Type: application/json' \
  -d '{"query": "which vector db does the project use?", "session_id": "s1", "project": "erp"}'
```

```json
{
  "project": "erp",
  "context_block": "[Long-term memories]\n- (fact, 2026-07-10) ...",
  "memories": [...],
  "related_history": [...],
  "session_summaries": [...],
  "recent_turns": [...],
  "documents": [],
  "routing": {"docs": false, "history_hint": false},
  "last_written_at": 1752130000.0
}
```

`routing` reports the rule-based router's decision: `docs` — whether the
document channel fired (trigger words like "spec", "tài liệu");
`history_hint` — observability only.

Optional tuning fields: `top_k_memories`, `top_k_history`, `recent_turns`.

#### `POST /memory/facts`

Save distilled facts directly (the REST counterpart of the `save_memories`
MCP tool). Near-duplicates are superseded automatically. If `session_id` is
given and `project_id` is not, facts inherit the session's project.

```bash
curl -X POST localhost:8800/memory/facts -H 'Content-Type: application/json' \
  -d '{"facts": [{"text": "ERP uses PostgreSQL 16", "type": "fact", "importance": 0.7}],
       "session_id": "s1"}'
```

```json
{"status": "ok", "project": "erp",
 "results": [{"status": "created", "text": "ERP uses PostgreSQL 16", "id": "..."}]}
```

`type` is one of `fact | preference | decision | task`; `importance` is 0–1.

#### `POST /memory/search`

Raw semantic search over stored facts (no context assembly). Useful for
debugging what recall would see.

```bash
curl -X POST localhost:8800/memory/search -H 'Content-Type: application/json' \
  -d '{"query": "database choice", "top_k": 5}'
```

```json
{"results": [{"id": "...", "text": "...", "type": "fact", "score": 0.62, ...}]}
```

#### `POST /memory/consolidate`

Distill one session's un-consolidated turns into facts using the
**service-side LLM** (`LLM_PROVIDER != none`, otherwise `409`). With
`"background": true` it never blocks or errors — hook-friendly
fire-and-forget.

```bash
curl -X POST localhost:8800/memory/consolidate -H 'Content-Type: application/json' \
  -d '{"session_id": "s1", "background": true}'
```

```json
{"status": "scheduled", "session_id": "s1"}
```

#### `GET /memory/pending-consolidation`

List sessions with enough idle, un-distilled turns to be worth
consolidating. This is how an agent-side model (no service LLM) finds work.

#### `POST /memory/consolidate-pending`

Debounced catch-up sweep over every pending session; returns immediately
(`{"status": "scheduled"}`). Called by the session-start hook so sessions
that ended with a crash still get consolidated.

### Browsing & curating memory

#### `GET /memory/facts`

List stored facts, newest first. Filters: `project`, `type`,
`include_superseded`, `limit`.

```bash
curl "localhost:8800/memory/facts?project=erp&type=decision"
```

#### `GET /memory/graph`

Facts as a similarity graph (nodes + edges) — this feeds the `/ui` page.
Parameters: `include_superseded`, `top_edges` (default 4), `min_similarity`
(default 0.35).

```bash
curl "localhost:8800/memory/graph?min_similarity=0.35"
```

#### `PATCH /memory/facts/{id}/type`

Re-classify a memory (`fact | preference | decision | task`) — for when the
consolidation model got it wrong.

```bash
curl -X PATCH localhost:8800/memory/facts/<id>/type \
  -H 'Content-Type: application/json' -d '{"type": "decision"}'
```

```json
{"status": "retyped", "id": "...", "type": "decision"}
```

#### `DELETE /memory/facts/{id}`

Hard-delete one fact. Permanent — unlike supersede, it keeps no trace.

```bash
curl -X DELETE localhost:8800/memory/facts/<id>
```

#### `DELETE /memory/all`

Full memory reset. Requires the exact confirmation string:

```bash
curl -X DELETE "localhost:8800/memory/all?confirm=DELETE%20ALL"
```

```json
{"status": "wiped", "messages_deleted": 385, "facts_deleted": 308}
```

### Re-tagging (project corrections)

All four take `{"project_id": "<slug>"}` and back the correction features in
`/ui`.

```bash
# move one fact
curl -X PATCH localhost:8800/memory/facts/<id> \
  -H 'Content-Type: application/json' -d '{"project_id": "erp"}'

# bulk move (multi-select in /ui)
curl -X PATCH localhost:8800/memory/facts \
  -H 'Content-Type: application/json' -d '{"ids": ["<id1>", "<id2>"], "project_id": "erp"}'

# move a whole session: its turns + the facts distilled from it;
# future turns follow via session stickiness
curl -X PATCH localhost:8800/sessions/<session_id>/project \
  -H 'Content-Type: application/json' -d '{"project_id": "erp"}'

# rename a project everywhere (history + facts + documents)
curl -X PATCH localhost:8800/memory/projects/<old-slug> \
  -H 'Content-Type: application/json' -d '{"project_id": "new-name"}'
```

### Sessions & projects

```bash
curl localhost:8800/projects                  # projects with session/message counts
curl localhost:8800/sessions                  # all stored sessions
curl localhost:8800/sessions/s1/history       # one session's turns (role + content)
curl -X DELETE localhost:8800/sessions/s1     # delete one session's stored history
```

Deleting a session removes its turns only — facts already distilled from it
stay (delete those via `DELETE /memory/facts/{id}`).

### Transfer (device / embedding-model migration)

#### `GET /memory/export`

Download the whole memory as one JSON bundle — payload text only, **no
vectors**, so it survives a change of embedding model.

```bash
curl -o bundle.json localhost:8800/memory/export
```

#### `POST /memory/import`

Re-embed and upsert a bundle from `/memory/export`. Keeps original
timestamps, supersede links and consolidated flags; skips records that
already exist — running it twice is safe.

```bash
curl -X POST localhost:8800/memory/import -H 'Content-Type: application/json' \
  --data-binary @bundle.json
```

Prefer the wrapper: `./scripts/memory_transfer.sh export|import`.

### Chat (optional)

#### `POST /chat`

A built-in RAG chat over the knowledge base + memory. Requires
`LLM_PROVIDER != none` (`503` otherwise). Most installs never use it — the
agents chat with their own models and only call `/memory/*`.

```bash
curl -X POST localhost:8800/chat -H 'Content-Type: application/json' \
  -d '{"session_id": "s1", "message": "..."}'
```

### UI

#### `GET /ui`

The self-contained memory browser page (see the
[User Guide](USER_GUIDE.md#memory-browser-ui)). Open it in a browser, not
curl.

---

## MCP tools

Registered at `http://localhost:8800/mcp` (Streamable HTTP). All tools
return plain text designed to be read by the model. `project` is always an
optional project slug that scopes/boosts the search.

### Recall & record

| Tool | Use it when |
|---|---|
| `memory_recall(query, session_id?, project?)` | Starting a task, or the user refers to something from before. Returns a ready-to-inject context block combining facts + related past chats + recent turns. |
| `memory_append(session_id, user_message?, assistant_response?)` | Recording a turn manually (the hooks normally do this). Idempotent. |
| `search_history(query, top_k?, project?)` | Finding a specific past conversation turn across all sessions ("the review from last week"). |
| `search_knowledge_base(query, top_k?, project?)` | The question is about a project document/spec (docs/ folders are auto-ingested). |

### Save & consolidate

| Tool | Use it when |
|---|---|
| `save_memories(facts, session_id?, project?, session_summary?)` | Saving distilled facts (each: `text`, `type`: fact/preference/decision/task, `importance` 0–1). Dedup/supersede is automatic. Also closes a `consolidate_session` handout — pass the same `session_id` and the 2–4 sentence `session_summary`. |
| `consolidate_session(session_id)` | Distilling a finished session. With a service-side LLM it saves facts directly; without one it returns the transcript + extraction instructions — follow them, then call `save_memories`. |
| `add_to_knowledge_base(text, source?, project?)` | Storing reference material (not conversational facts) for later retrieval. |

### Inspect

| Tool | Use it when |
|---|---|
| `list_memories(project?, limit?)` | Showing the user what is stored; returns ids needed by `forget_memory`. |
| `list_projects()` | Discovering which project slugs exist (with counts). |
| `list_sessions()` | Listing stored sessions with message counts. |

### Forget (destructive — confirm with the user first)

| Tool | Use it when |
|---|---|
| `forget_about(query)` | The user asks to forget something. Returns **candidates with ids** — show them, get confirmation, then delete each with `forget_memory`. Never deletes by itself. |
| `forget_memory(memory_id, confirm=false)` | Deleting one fact by id. Refuses unless `confirm=true` — set it only after the user explicitly confirmed that specific memory. |
| `forget_session(session_id)` | Deleting one session's stored turns (distilled facts are not touched). |
| `forget_everything(confirm="")` | Full reset. Requires the exact string `confirm="DELETE ALL"`; anything else refuses. |

---

## Common errors

| Status | Where | Meaning |
|---|---|---|
| `400` | most POST/PATCH | Empty `query`/`text`, invalid slug (must match `[a-z0-9][a-z0-9_-]{0,63}`), invalid `type`, malformed metadata JSON, invalid transfer bundle, or missing `confirm=DELETE ALL`. |
| `404` | fact/session endpoints | No record with that id. |
| `409` | `POST /memory/consolidate` (foreground) | Consolidation could not run (e.g. no LLM configured) — use `background: true` or the MCP tool flow instead. |
| `503` | `POST /chat` | `LLM_PROVIDER=none` — /chat is disabled by design; agents should use `/memory/*` + `/query`. |
| connection refused | everything | Service not running — `docker compose up -d`, then check `GET /health`. |
