# Hermes Memory Stack Architecture

Detailed technical documentation. For quick install/usage instructions see [README.md](README.md).

## 1. Overview

A Docker-packaged memory backend that gives Hermes Desktop long-term memory
following a **single-user, local-first** model: each user runs an independent
stack, all data lives entirely on their machine — no sync, no sharing.

```
┌─ USER'S MACHINE ─────────────────────────────────────────────────────────┐
│                                                                          │
│  ┌─ Hermes Desktop (native app — the chat LLM runs here) ─────────────┐  │
│  │                                                                     │  │
│  │   hooks.post_llm_call ──────────────┐        MCP client ─────────┐ │  │
│  └──────────────────────────────────────┼───────────────────────────┼─┘  │
│                                         │ (after every chat turn)   │    │
│                              hooks/post_llm_call.py                 │    │
│                                         │ POST /memory/append       │    │
│                                         ▼                           ▼    │
│  ┌─ Docker Compose ────────────────────────────────────────────────────┐ │
│  │                                                                      │ │
│  │  ┌─ llamaindex-service (FastAPI, host :8800 → container :8000) ───┐  │ │
│  │  │                                                                 │  │ │
│  │  │   REST API           MCP Streamable HTTP (/mcp)                 │  │ │
│  │  │      │                        │                                 │  │ │
│  │  │      └────────┬───────────────┘                                 │  │ │
│  │  │               ▼                                                 │  │ │
│  │  │   ┌─ Memory Engine ────────────────────────────────┐            │  │ │
│  │  │   │ L1 Working   ChatMemoryBuffer (per session)    │            │  │ │
│  │  │   │ L2 Episodic  memory_store.py                   │            │  │ │
│  │  │   │ L3 Semantic  memories.py + consolidation.py    │            │  │ │
│  │  │   │ L4 Knowledge documents.py                      │            │  │ │
│  │  │   └────────────────────────────────────────────────┘            │  │ │
│  │  │               │                                                 │  │ │
│  │  │   Embedding: fastembed ONNX (baked into image, local, no key)   │  │ │
│  │  │   LLM (optional): none|anthropic|openai|nvidia|ollama           │  │ │
│  │  └──────────────┬──────────────────────────────────────────────────┘  │ │
│  │                 │ HTTP :6333                                          │ │
│  │  ┌─ Qdrant ─────▼─────────────────────────────────────┐               │ │
│  │  │ hermes_chat_history │ hermes_memories │             │               │ │
│  │  │ hermes_documents    │ hermes_meta     │             │               │ │
│  │  └─── volume: qdrant_data ───────────────┘             │               │ │
│  │                                                        │               │ │
│  │  [ollama] — optional profile (--profile ollama)        │               │ │
│  └────────────────────────────────────────────────────────┘               │ │
│         volume: hermes_data (/data — original document files)             │ │
└──────────────────────────────────────────────────────────────────────────┘
```

## 2. The four memory layers

| Layer | Role | Stored in | Module |
|---|---|---|---|
| **L1 Working** | Current-session context — a `ChatMemoryBuffer` (~3000-token cap) rebuilt from L2 every turn | RAM (rebuilt per request) | `main.py` |
| **L2 Episodic** | Every raw conversation turn (user + assistant), embedded → semantically searchable across sessions | `hermes_chat_history` | `memory_store.py` |
| **L3 Semantic** | Facts distilled from L2: decisions, preferences, project info, tasks — the permanently memorable stuff | `hermes_memories` | `memories.py`, `consolidation.py` |
| **L4 Knowledge** | Ingested documents (PDF/text/markdown) — classic RAG | `hermes_documents` + original files in `/data/documents` | `documents.py` |

## 3. Data flows

### 3a. Write flow (every chat turn, automatic)

```
User chats in Hermes Desktop
  → Hermes fires the post_llm_call event, piping JSON into the hook's stdin:
      {session_id, extra: {user_message, assistant_response, ...}}
  → hooks/post_llm_call.py  (best-effort, never breaks the chat turn)
  → POST /memory/append
  → memory_store.add_message():
      - embed the content (fastembed, inside the container)
      - point ID = uuid5(user_id : session_id : role : sha256(content))
        → IDEMPOTENT: retries/duplicate writes never create new records
      - upsert into hermes_chat_history
      - update last_written_at in hermes_meta (to detect a dead hook)
```

### 3b. Distillation flow (consolidation, L2 → L3)

```
On session end / on demand:
  MCP tool consolidate_session(session_id)
    │
    ├─ Service HAS an LLM (LLM_PROVIDER != none):
    │    fetch unprocessed turns → LLM extracts facts (JSON) → save_facts()
    │    → mark turns consolidated=true
    │
    └─ Service has NO LLM (default):
         return the transcript + extraction instructions to Hermes' OWN model
         → Hermes distills it itself → calls the MCP tool save_memories(facts)
         (turns are only marked consolidated when save_memories comes back —
          a model that never answers just leaves the session for a later pass;
          likewise, unparseable LLM output raises instead of counting as "no facts")

save_facts() deduplicates on two levels:
  1. Exact hash (point ID from sha256 of normalized text) → skip if duplicate
  2. Similarity ≥ 0.92 with a currently active fact → the old fact is marked
     superseded_by=<new id> (kept for traceability, excluded from recall)
```

### 3c. Read flow (recall)

```
POST /memory/recall {query, session_id?}   (or the MCP tool memory_recall)
  │
  ├─ L3: search hermes_memories (filter out superseded)
  │      score = similarity × 0.5^(age/90 days) × (0.5 + 0.5×importance)
  ├─ L2: search hermes_chat_history across sessions (excluding the current one)
  │      score = similarity × 0.5^(age/30 days)
  └─ L1: the N most recent turns of the current session (chronological)
  │
  └→ context_block ready to inject into the system prompt:
       [Long-term memories] … [Related past conversations] … [Most recent turns] …
```

## 4. Qdrant schema

All vector collections use Cosine distance; dimension follows the embedding
model (default 384 — `paraphrase-multilingual-MiniLM-L12-v2`).

### `hermes_chat_history` (L2)
```jsonc
// point ID: uuid5("msg:{user_id}:{session_id}:{role}:{sha256(content)}")
{
  "user_id": "local",        // payload index — ready for future multi-user
  "session_id": "…",         // payload index
  "project_id": "erp",       // payload index — sidebar project (see section 11)
  "role": "user|assistant",  // payload index
  "content": "…",
  "timestamp": 1783229012.9, // payload index (float)
  "consolidated": false      // payload index — distilled yet or not
}
```

### `hermes_memories` (L3)
```jsonc
// point ID: uuid5("fact:{user_id}:{sha256(normalized_text)}")
{
  "user_id": "local",
  "session_id": "…",              // source session
  "project_id": "erp",            // payload index — inherited from the source session
  "type": "fact|preference|decision|task",
  "text": "…",
  "importance": 0.8,              // 0..1
  "created_at": 1783229012.9,
  "superseded_by": "<point-id>"   // only present when replaced by a newer fact
}
```

### `hermes_documents` (L4)
Managed by LlamaIndex's `QdrantVectorStore` (chunks + metadata `user_id`,
`source`, `stored_path` → original file in `/data/documents/<sha12>_<name>`).

### `hermes_meta` (guard)
A single fixed point, vector dim=1: `{schema_version, embed_provider,
embed_model, embed_dim, last_written_at}`. At startup the service compares the
current config against the meta — **an embedding mismatch refuses to boot**
(protects the vector space).

## 5. Providers (configured via .env)

| | Embedding | LLM |
|---|---|---|
| Role | Determines the vector space — **pick once** | Only used for consolidation + `/chat` — **swap freely** |
| Default | `fastembed` (local ONNX, baked into the image, no key, no GPU) | `none` (Hermes' own model distills via MCP) |
| Options | `ollama`, `openai`, `nvidia` | `anthropic`, `openai`, `nvidia`, `ollama` |
| How to change | Backup → `docker compose down -v` → edit .env → re-ingest (the meta guard blocks silent changes) | Edit `.env` + API key → `docker compose up -d` |

`fastembed` is **hard-pinned** in requirements.txt — this library has changed
its pooling behavior between minor versions before; without the pin, vectors
between two image builds can silently diverge.

## 6. API surface

### REST (`http://localhost:8800`)
| Endpoint | Purpose |
|---|---|
| `GET /health` | Status + `last_written_at` + point counts per collection |
| `POST /memory/append` | Record one conversation turn (called by the hook) — idempotent |
| `POST /memory/recall` | Combined recall of L1+L2+L3 → context_block |
| `POST /memory/facts` | Save distilled facts |
| `POST /memory/search` | Search L3 |
| `POST /memory/consolidate` | Distill (requires LLM; `background: true` for hooks) |
| `GET /memory/pending-consolidation` | Sessions awaiting distillation |
| `GET /memory/facts` · `DELETE /memory/facts/{id}` | List / forget facts |
| `GET /memory/graph` | Facts as nodes + similarity edges (feeds `/ui`) |
| `GET /ui` | Memory browser (graph + list, filters, semantic search, re-tagging) |
| `PATCH /memory/facts/{id}` | Move one fact to another project |
| `PATCH /memory/facts` | Bulk move: `{ids: [...], project_id}` (multi-select in `/ui`) |
| `PATCH /sessions/{id}/project` | Re-tag a whole session (turns + its facts; future turns follow) |
| `PATCH /memory/projects/{slug}` | Rename a project across chat history, facts and documents |
| `DELETE /sessions/{id}` | Delete an entire session |
| `POST /ingest/text` `/ingest/file` | Ingest documents into L4 |
| `POST /query` | Search L4 |
| `POST /chat` | Chat directly with the service (requires LLM; 503 if `none`) |
| `GET /sessions` `/sessions/{id}/history` | List sessions / session contents |

### MCP tools (`http://localhost:8800/mcp`, Streamable HTTP)
`memory_recall` · `memory_append` · `consolidate_session` · `save_memories` ·
`list_memories` · `forget_about` · `forget_memory` · `search_history` ·
`list_sessions` · `list_projects` · `search_knowledge_base` ·
`add_to_knowledge_base` — all search/write tools accept an optional `project` param.

## 7. Hermes integration — 3 touch points

`setup.sh` → `scripts/configure_hermes.py` automates everything (idempotent):

1. **`~/.hermes/config.yaml`**: 3 hooks + `hooks_auto_accept: true` + MCP server:
   - `post_llm_call` → record every chat turn into memory (with the project from cwd)
   - `pre_llm_call` → auto-inject memory: recall based on user_message, return
     `{"context": ...}` (Hermes' official contract) — the model no longer has
     to remember to call a tool
   - `on_session_end` → trigger background consolidation for the session that just ended
   Additionally: borrows an API key from `~/.hermes/.env` (NVIDIA → Gemini) for
   auto-consolidation, installs the 2:00 AM launchd backup (7-copy retention), and
   adds a **memory-routing block to `~/.hermes/SOUL.md`** — Hermes has a built-in
   memory tool (a ~2200-char text file) that competes with the MCP tools when the
   user gives explicit "remember/forget" commands; this block routes long-term
   facts to MCP (Qdrant), while the built-in keeps only core identity.
2. **`~/.hermes/shell-hooks-allowlist.json`**: hook consent (Hermes requires
   per-command approval; Desktop has no TTY to ask → must be pre-recorded).
   ⚠️ Consent is tied to the script's mtime — **editing a hook file means re-running setup.sh**.
3. **Hermes bug patch** (`hermes_cli/main.py`): the `serve` command (the Desktop
   backend) is missing from `_AGENT_COMMANDS`, so shell hooks are never registered
   for Desktop chats (CLI works, Desktop silently doesn't). The patch adds `"serve"`.
   ⚠️ **A Hermes update overwrites the patch — re-run `./setup.sh` after every update.**

## 8. Operations

- **Liveness check**: `curl localhost:8800/health` — `last_written_at` must
  advance after every chat turn. Visual view: `http://localhost:6333/dashboard`.
- **Hook diagnostics**: `hermes hooks doctor` · raw payloads of every hook
  invocation live in `logs/hook-debug.jsonl`.
- **Backup**: `./scripts/backup.sh` — snapshots every collection into `./backups/`.
- **Data**: named volumes `qdrant_data` (vectors) + `hermes_data` (original files).
  `docker compose down -v` = wipe everything and start over.
- **Ports**: 8800 (service — keeps 8000 free for Hermes' hindsight server),
  6333/6334 (Qdrant), 11434 (Ollama if the profile is enabled).

## 9. Source layout

```
hermes-agent/
├── setup.sh                     # one-command install: Docker + automatic Hermes wiring
├── docker-compose.yml           # qdrant + llamaindex (+ optional ollama profile)
├── .env.example                 # provider/collection configuration
├── ARCHITECTURE.md              # this document
├── hooks/
│   ├── post_llm_call.py         # record each turn (project resolution: folder → active sidebar → default)
│   ├── pre_llm_call.py          # auto-inject recalled memory into every turn
│   ├── on_session_end.py        # trigger consolidation when a session finishes
│   └── on_session_start.py      # debounced catch-up sweep when Desktop opens
├── scripts/
│   ├── configure_hermes.py      # auto-patch config.yaml + consent + serve bug + app restart
│   └── backup.sh                # Qdrant snapshots (nightly via launchd)
├── logs/
│   └── hook-debug.jsonl         # raw payload of every hook run (diagnostics)
└── llamaindex-service/
    ├── Dockerfile               # fastembed model baked in
    ├── requirements.txt         # fastembed hard-pinned ==0.7.4
    ├── requirements-dev.txt     # pytest (see "Testing" in README)
    ├── tests/                   # pytest suite (runs in the container)
    └── app/
        ├── main.py              # FastAPI endpoints + lifespan + MCP mount + /ui route
        ├── config.py            # env parsing, constants
        ├── providers.py         # embedding + LLM factory per provider
        ├── qdrant_setup.py      # auto-init collections/indexes + schema guard
        ├── memory_store.py      # L2: write/read/search conversations (deterministic IDs, stickiness)
        ├── memories.py          # L3: save_facts (dedup/supersede) + recall + decay + graph data
        ├── consolidation.py     # L2→L3 distillation (prompt + parsing + handouts)
        ├── documents.py         # L4: document ingest + original file retention
        ├── mcp_server.py        # MCP tools (Streamable HTTP at /mcp)
        ├── scheduler.py         # event-driven consolidation sweeps
        ├── runtime.py           # state shared between REST and MCP
        └── ui.html              # self-contained memory browser (served at /ui)
```

## 10. Per-project memory (project partitioning)

Memory is partitioned by **Hermes Desktop sidebar project** — not with separate
collections, but with a `project_id` field (payload-indexed) across all 3 data
layers, following Qdrant's multitenancy recommendation.

**The source of truth is Hermes itself.** Hermes' sidebar projects live in
`~/.hermes/projects.db` (tables `projects` + `project_folders`; each project is
anchored to one or more folders). There is no separate mapping file to maintain.

**Project-tagging flow (100% automatic):**
```
Chat in Hermes (anywhere)
  → the hook payload already contains "cwd" (the chat session's working directory)
  → hooks/post_llm_call.py resolves, in priority order:
      1. longest-prefix match of cwd against each project's folders
         (child project beats parent, e.g. /work/erp beats /work; two projects
         sharing a folder → the most recently created one wins, logged as a
         conflict warning in hook-debug.jsonl)
      2. the project currently SELECTED in the sidebar (project_meta.active_id)
         — chat-only projects need no folder at all
      3. "default"
  → project_id rides along with /memory/append and is stored in the payload
  → SESSION STICKINESS (server-side): a session that already has stored turns
    keeps its founding project — resuming an old chat while the sidebar
    points elsewhere does not re-tag its new turns.
    Exception: a FOLDER-sourced tag (the session's cwd really moved — Hermes'
    project_switch tool is the intentional way to move a session between
    projects) overrides stickiness; ambient signals (sidebar selection) never do.
```

**Recall flow:**
- `memory_recall(query, session_id)` — the service infers the project from the
  session's own stored messages (nobody has to declare it). For a brand-new
  session with no messages yet, pass `project` explicitly (Hermes' model can
  call `list_projects` to discover slugs).
- **Soft boost instead of a hard filter** for memories/conversations:
  same-project scores × `RECALL_PROJECT_BOOST` (default 1.5). Memories from
  other projects can still surface when genuinely relevant — exactly the value
  of cross-project memory.
- **Documents (L4) are hard-filtered** when a project is specified — project A's
  documents answering project B's questions is usually noise.

Old data without a `project_id` is treated as `"default"` — no migration
needed. Creating a new project in the sidebar is picked up automatically, with
zero extra configuration.

## 11. Extension directions (cost already prepaid)

- Every payload carries a `user_id` (default `"local"`) and deterministic point
  IDs → moving to a shared multi-user server later is just adding auth +
  changing the value, not migrating data.
- Changing the embedding model: create new collections → re-embed from the
  original files (L4) and payload content (L2/L3) → flip an alias. The meta
  guard guarantees two vector spaces are never mixed.
- Deliberately not built yet (for simplicity): TTL/automatic forgetting, hybrid
  search (dense+sparse), auth — enable when there is a real need.
