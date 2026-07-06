# Hermes Agent — Long-term Memory Stack (LlamaIndex + Qdrant)

> 🇻🇳 Bản tiếng Việt: [README.vi.md](README.vi.md)

A Docker-packaged "long brain" for AI agents: each user runs their own
independent stack — all data and memory stay private on their machine. By
default it needs **no API key, no Ollama, and no Python on the host**.
Two agent adapters ship today — **Hermes Desktop** and **Claude Code** —
and they can run **in parallel against the same memory** (what you teach
one agent, the other recalls; records carry a `source_agent` tag).

## Architecture

```
Hermes Desktop (host)                Claude Code (host)
   │ pre_llm_call   ► auto-inject      │ UserPromptSubmit ► auto-inject
   │ post_llm_call  ► auto-record      │ Stop             ► auto-record
   │ on_session_end ► auto-distill     │ SessionEnd       ► auto-distill
   │ on_session_start ► catch-up       │ SessionStart     ► catch-up
   │  MCP (Streamable HTTP) ──► http://localhost:8800/mcp ◄── MCP
   ▼                                   ▼
LlamaIndex service (Docker, 127.0.0.1:8800)
   ├── L1 Working memory   — ChatMemoryBuffer rebuilt per session
   ├── L2 Episodic memory  — hermes_chat_history (every turn, searchable
   │                          semantically and per session)
   ├── L3 Semantic memory  — hermes_memories (facts/preferences/decisions/
   │                          tasks distilled by consolidation, with
   │                          dedup/supersede)
   ├── L4 Knowledge base   — hermes_documents (document RAG)
   ├── Embedding: fastembed (local ONNX, baked into the image)
   └── LLM (for consolidation): none | anthropic | openai | nvidia | gemini | ollama
   ▼
Qdrant (Docker, 127.0.0.1:6333) — named volume `qdrant_data`
```

The memory lifecycle is **fully automatic**: record → auto-recall →
consolidate → controlled forgetting (`forget_about` tool) → nightly backup
(2:00 AM, 7 kept).

## Install (3 steps)

1. Install [Docker Desktop](https://docs.docker.com/get-docker/).
2. Install Hermes Desktop and/or Claude Code — whichever agents you use.
3. In this directory run:

```bash
./setup.sh
```

**No manual steps remain.** The script does everything: creates `.env` →
builds & starts containers → waits for health → wires every installed agent
(each is skipped gracefully when absent):

- **Hermes Desktop**: registers all 4 hooks + consent into `~/.hermes/` →
  patches Hermes' `serve` bug (Desktop never registers shell hooks without
  it) → borrows an available API key (NVIDIA/Gemini) for auto-consolidation
  → adds memory-routing guidance to `~/.hermes/SOUL.md` (explicit
  "remember/forget" commands go to this stack, not Hermes' small built-in
  store) → restarts Hermes Desktop.
- **Claude Code**: registers the 4 hooks in `~/.claude/settings.json` and
  the `hermes-memory` MCP server (user scope) via `scripts/configure_claude.py`.
  **Needs no API key anywhere**: Claude Code runs on your Claude login, and
  consolidation uses the service-side LLM or the `consolidate_session` MCP
  tool. Restart open sessions to pick the hooks up.

It also installs the nightly backup job. Safe to re-run any number of times
(idempotent).

Verify after a few chats: `curl localhost:8800/health` — `last_written_at`
must advance after every turn.

### Running Hermes and Claude Code in parallel

Nothing to switch: both agents' hooks stay registered and write to the same
service. Project slugs stay coherent because the Claude Code adapter first
resolves the cwd against Hermes' own `projects.db` (same slug as the
sidebar), then falls back to the git-root folder name. Every record carries
`source_agent` (`hermes` / `claude-code`) — visible in the `/ui` detail
panel. Note: this stack complements Claude Code's own CLAUDE.md/auto-memory
(static per-repo instructions) with semantic, cross-session, cross-project
recall; memory injection is bounded (`HERMES_MEMORY_MAX_CONTEXT`, default
6000 chars) since it spends your subscription tokens each turn.

## Memory browser (`http://localhost:8800/ui`)

A self-contained page for exploring and curating stored memory — no extra
container, no external assets, light/dark themed:

- **Graph view**: memories as a force-directed graph over a camera-aligned
  dot grid. Edges are real vector similarity (darker = closer in meaning);
  color = project, node **shape = type** (● fact, ▲ preference, ◆ decision,
  ■ task — the `?` button shows the legend), size = importance, dashed
  outline = superseded. Hover highlights a node's neighborhood; click opens
  the detail panel with a selection ripple. Drag, pan, smooth wheel zoom,
  `Fit`, and click the title to reset the view.
- **Spotlight search (⌘K)**: live semantic search with recent queries —
  matches highlight on the graph and the camera glides to the best hit.
- **Filters**: project and type chips (click to solo/toggle), a superseded
  switch, plus link controls — a minimum-similarity slider and a
  "same project only" toggle for the edges.
- **Detail panel**: full text, metadata, related memories (by similarity,
  click to jump), and the source-session transcript rendered as markdown.
- **Corrections** (project picker modal — no manual typing): move one memory,
  re-tag a whole session (turns + facts move, future turns follow via
  stickiness), bulk re-tag a ⇧click multi-selection ("Select linked" expands
  it to the whole connected cluster), or rename a project everywhere
  (the ✎ on a project chip).
- **List view** (same filters, as a table) and **PNG export** of the graph.
- **Transfer** (`⇩ Export` / `⇪ Import` in the header): download the whole
  memory as a JSON bundle, or import one from another machine — with a
  confirmation showing the bundle's counts before anything is written
  (see "Moving to another machine" below).

Deleting is deliberately NOT offered here — forget through Hermes
(`forget_about`) or the REST API, which have confirmation guards.

## Provider configuration (.env)

| Variable | Default | Meaning |
|---|---|---|
| `EMBED_PROVIDER` | `fastembed` | `fastembed` \| `ollama` \| `openai` \| `nvidia` |
| `EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model (multilingual, CPU) |
| `LLM_PROVIDER` | `none`* | `none` \| `anthropic` \| `openai` \| `nvidia` \| `gemini` \| `ollama` |
| `LLM_MODEL` | per provider | e.g. `models/gemini-2.5-flash`, `claude-sonnet-5` |
| `*_API_KEY` | — | `ANTHROPIC` / `OPENAI` / `NVIDIA` / `GOOGLE` — setup.sh borrows an existing key from `~/.hermes/.env` when the provider is `none` |
| `HERMES_USER_ID` | `local` | Stamped into every payload (future multi-user server = no migration) |

- **The LLM is freely swappable** — it is only used for consolidation and
  `/chat`. With `none`, Hermes' own model handles consolidation through the
  `consolidate_session` MCP tool.
- **The embedding is a one-time choice** — changing it changes the vector
  space. The service records model + dimension in a meta collection and
  **refuses to boot** on a mismatch. To really switch: backup →
  `docker compose down -v` → edit `.env` → re-ingest, or re-embed into new
  collections.
- Local Ollama (optional): `docker compose --profile ollama up -d`, then set
  `LLM_PROVIDER=ollama` and `OLLAMA_BASE_URL=http://ollama:11434`.

## MCP tools (registered at `http://localhost:8800/mcp`)

| Tool | Purpose |
|---|---|
| `memory_recall(query, session_id?, project?)` | Combine relevant memory (facts + related past chats + recent turns) into one context block |
| `memory_append(session_id, user_message, assistant_response)` | Record one turn (idempotent) |
| `consolidate_session(session_id)` | Distill a session into facts (server-side with an LLM, otherwise returns transcript + instructions for Hermes' model) |
| `save_memories(facts, session_id?, project?)` | Save distilled facts (auto dedup/supersede) |
| `search_history(query, top_k?, project?)` | Semantic search across all past conversations |
| `list_memories(project?)` | List stored facts (with ids) |
| `forget_about(query)` → `forget_memory(id, confirm=true)` | Controlled forgetting: list candidates first, delete by id — refuses without `confirm=true` (set only after the user agreed) |
| `forget_session(session_id)` | Delete one session's entire stored history |
| `forget_everything(confirm="DELETE ALL")` | Full memory reset — requires the exact confirmation string |
| `list_sessions()` / `list_projects()` | List stored sessions / projects |
| `search_knowledge_base(query, top_k?, project?)` | Search ingested documents |
| `add_to_knowledge_base(text, source?, project?)` | Add text to the knowledge base |

Tools accept `project` (a Hermes sidebar project slug) to scope searches.

## REST API

```bash
# Status + is memory being written? (last_written_at)
curl localhost:8800/health

# Ingest documents
curl -X POST localhost:8800/ingest/text -H 'Content-Type: application/json' \
  -d '{"text": "Content...", "metadata": {"source": "faq.md"}}'
curl -X POST localhost:8800/ingest/file -F "file=@document.pdf"

# Query the knowledge base
curl -X POST localhost:8800/query -H 'Content-Type: application/json' \
  -d '{"query": "..."}'

# Memory
curl -X POST localhost:8800/memory/append -H 'Content-Type: application/json' \
  -d '{"session_id": "s1", "user_message": "...", "assistant_response": "..."}'
curl -X POST localhost:8800/memory/recall -H 'Content-Type: application/json' \
  -d '{"query": "which vector db does the project use?", "session_id": "s1"}'
curl -X POST localhost:8800/memory/consolidate -H 'Content-Type: application/json' \
  -d '{"session_id": "s1"}'          # needs LLM_PROVIDER != none
curl -X POST localhost:8800/memory/search -H 'Content-Type: application/json' \
  -d '{"query": "..."}'
curl "localhost:8800/memory/facts?project=erp"      # list facts
curl -X DELETE localhost:8800/memory/facts/<id>     # forget one fact
curl -X DELETE "localhost:8800/memory/all?confirm=DELETE%20ALL"  # full reset

# Memory graph (nodes + similarity edges, feeds the /ui page)
curl "localhost:8800/memory/graph?include_superseded=false&min_similarity=0.35"

# Transfer (device migration — see the dedicated section below)
curl -o bundle.json localhost:8800/memory/export
curl -X POST localhost:8800/memory/import -H 'Content-Type: application/json' \
  --data-binary @bundle.json

# Re-tagging (corrections; slugs are lowercase [a-z0-9_-])
curl -X PATCH localhost:8800/memory/facts/<id> -H 'Content-Type: application/json' \
  -d '{"project_id": "erp"}'                       # move one fact
curl -X PATCH localhost:8800/memory/facts -H 'Content-Type: application/json' \
  -d '{"ids": ["<id1>", "<id2>"], "project_id": "erp"}'   # bulk move
curl -X PATCH localhost:8800/sessions/<id>/project -H 'Content-Type: application/json' \
  -d '{"project_id": "erp"}'    # whole session: turns + facts; future turns follow
curl -X PATCH localhost:8800/memory/projects/<slug> -H 'Content-Type: application/json' \
  -d '{"project_id": "new-name"}'                  # rename everywhere

# Sessions & projects
curl localhost:8800/sessions
curl localhost:8800/sessions/s1/history
curl -X DELETE localhost:8800/sessions/s1
curl localhost:8800/projects
```

## Per-project memory

Memory is automatically partitioned by **Hermes Desktop sidebar project**:
the hook reads the chat's working directory (`cwd`) → looks it up in
`~/.hermes/projects.db` → stamps `project_id` on every record. When the cwd
matches no project folder, the project currently **selected in the sidebar**
is used — so chat-only projects work with no folder at all. Recall boosts
same-project memories (×1.5) while cross-project knowledge can still surface
when genuinely relevant; documents are hard-filtered by project. Creating a
new project in the sidebar just works — zero configuration.
Details: [ARCHITECTURE.md](ARCHITECTURE.md).

## Backup

Runs automatically at **2:00 AM daily** (launchd; also once at every
boot/login via `RunAtLoad` in case the machine was powered off; 7 newest kept;
log at `logs/backup.log`) — installed by setup.sh. Manual run:

```bash
./scripts/backup.sh    # snapshots every hermes_* collection into ./backups/
```

## Moving to another machine (export / import)

Nightly snapshots are **binary and tied to the embedding model** — they
restore the same machine, but they can't move memory to a new install that
may run a different model. For that, use the text-level transfer bundle:

```bash
# old machine
./scripts/memory_transfer.sh export            # -> backups/memory-export-<stamp>.json

# new machine (after setup.sh, service running)
./scripts/memory_transfer.sh import memory-export-<stamp>.json
```

The bundle contains payload text only (facts, chat turns, document chunks) —
no vectors. Import **re-embeds everything with the current model**, keeps the
original timestamps / supersede links / consolidated flags (so recall decay
and provenance keep working, and imported sessions are not re-distilled), and
skips records that already exist — running it twice is safe.

## Repository layout

```
hermes-agent/
├── setup.sh                 # one-command install (Docker + automatic Hermes wiring)
├── docker-compose.yml       # qdrant + llamaindex (+ optional ollama profile)
├── .env.example
├── ARCHITECTURE.md          # detailed architecture
├── UPGRADE_PLAN.md          # roadmap + progress
├── hooks/
│   ├── post_llm_call.py     # Hermes: record each turn (tagged with sidebar project)
│   ├── pre_llm_call.py      # Hermes: auto-inject memory into every turn
│   ├── on_session_end.py    # Hermes: trigger consolidation when a session ends
│   ├── on_session_start.py  # Hermes: catch-up sweep when Desktop opens
│   └── claude/              # Claude Code adapter (same lifecycle, 4 hooks)
├── scripts/
│   ├── configure_hermes.py  # auto-wire Hermes (hooks + consent + serve patch + key + backup)
│   ├── configure_claude.py  # auto-wire Claude Code (settings.json hooks + MCP)
│   ├── backup.sh            # Qdrant snapshots (called nightly by launchd)
│   └── memory_transfer.sh   # text-level export/import for device migration
└── llamaindex-service/      # memory service (FastAPI + LlamaIndex + MCP)
    └── tests/               # pytest suite (runs in the container, see below)
```

## Testing

```bash
docker compose run --rm --no-deps --entrypoint sh \
  -v "$PWD:/repo" llamaindex \
  -c "pip install -q 'pytest>=8,<9' && cd /repo/llamaindex-service && python -m pytest tests -q"
```

Covers: point-id idempotency, fact dedup/supersede, the recall min-score
filter (against an in-process Qdrant), LLM output parsing (parse failure vs
deliberate `[]`), transcript truncation, and the hooks' payload extraction +
cwd→project resolution (longest prefix, archived projects, symlinks).

## Operational notes

- **After every Hermes update: re-run `./setup.sh`** — updates overwrite the
  `serve` patch (without it the Desktop backend never registers hooks).
- **After editing any file in `hooks/`: re-run `./setup.sh`** — hook consent
  is tied to the script's mtime.
- **To delete memory, tell Hermes ("forget about X") or use the API** —
  avoid deleting in the Qdrant dashboard: it has unconfirmed full write
  access and makes it look like the system "lost data" on its own.
