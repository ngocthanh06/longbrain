# Longbrain — Shared Long-term Memory for AI Agents (LlamaIndex + Qdrant)

> 🇻🇳 Bản tiếng Việt: [README.vi.md](README.vi.md)

Longbrain is a Docker-packaged long-term memory system for AI agents. It
doesn't depend on any specific agent, so multiple agents can share the
same memory. Each user runs their own independent stack — data is always
stored locally and stays fully private.

By default it needs **no API key, no Ollama, and no Python on the host**.

Two agent adapters ship today: **Hermes Desktop** and **Claude Code**. Both
access the same memory store, so what you teach one agent, the other
recalls (records carry a `source_agent` tag). Adding a new agent only
means writing a new adapter — the system's architecture doesn't change.

## Architecture

> Diagram uses [Mermaid](https://mermaid.js.org/) — renders natively on
> GitHub and in VS Code/Cursor (built-in markdown preview, or the
> "Markdown Preview Mermaid Support" extension). If your viewer shows raw
> text instead of a picture, that's a viewer limitation, not a broken
> file — open it on GitHub or in VS Code's preview to see it rendered.

```mermaid
flowchart TB
    subgraph Agents["User's machine — chat agents"]
        HD["Hermes Desktop<br/>hooks/*.py"]
        CC["Claude Code<br/>hooks/claude/*.py"]
    end

    HD -- "POST /memory/append<br/>POST /memory/recall<br/>MCP /mcp" --> SVC
    CC -- "POST /memory/append<br/>POST /memory/recall<br/>MCP /mcp" --> SVC

    subgraph SVC["llamaindex-service — FastAPI (host :8800 → container :8000)"]
        REST["REST API"]
        MCP["MCP Streamable HTTP (/mcp)"]
        ENGINE["Memory Engine<br/>L1 Working · L2 Episodic · L3 Semantic · L4 Knowledge"]
        REST --> ENGINE
        MCP --> ENGINE
    end

    ENGINE -- "HTTP :6333" --> QD[("Qdrant<br/>longbrain_chat_history · longbrain_memories<br/>longbrain_documents · longbrain_meta")]
    QD -.-> VOL[("volume: qdrant_data")]
    ENGINE -.-> DVOL[("volume: longbrain_data<br/>/data — original document files")]

    OLLAMA["ollama (optional profile)"] -. "--profile ollama" .-> SVC
```

- **L1 Working memory** — `ChatMemoryBuffer` rebuilt per session
- **L2 Episodic memory** — `longbrain_chat_history` (every turn, searchable semantically and per session)
- **L3 Semantic memory** — `longbrain_memories` (facts/preferences/decisions/tasks distilled by consolidation, with dedup/supersede)
- **L4 Knowledge base** — `longbrain_documents` (document RAG)
- **Embedding**: fastembed (local ONNX, baked into the image)
- **LLM (for consolidation)**: `none | anthropic | openai | nvidia | gemini | ollama`

Full details (write/consolidate/recall sequence diagrams, Qdrant schema,
multi-agent provenance...): see [ARCHITECTURE.md](ARCHITECTURE.md).

The entire memory lifecycle runs **automatically**: record → recall →
consolidate → controlled forgetting (`forget_about` tool) → backup. By
default the system backs up at 2:00 AM daily and keeps the 7 most recent.

## How context gets built (and why the cost stays flat)

Before every chat turn, a hook calls `POST /memory/recall`, which merges
three sources into one context block that gets injected into the prompt
(full sequence diagram: [ARCHITECTURE.md §3c](ARCHITECTURE.md#3-data-flows)):

- **L3** — distilled facts relevant to what you just asked (semantic search over `longbrain_memories`)
- **L2** — related past conversations from *other* sessions (semantic search over `longbrain_chat_history`)
- **L1** — the current session's own recent turns

Search is **hybrid**: dense semantic cosine plus a BM25 keyword channel
that only fires when the question contains identifier-like tokens
(`ERR_UPLOAD_413`, `10MB`, `snake_case`, quoted strings, …) and can only
*rescue* exact matches the semantic model under-ranks — it never demotes a
semantic hit. Measured on the real document corpus, exact-token hits in
top-2 went from 1/12 to 11/12; prompts without such tokens return
byte-identical results. Kill switch: `HYBRID_BM25=false`.

Two things keep this cheap instead of turning into another growing
`CLAUDE.md`:

1. **Nothing relevant → nothing injected.** Results below `RECALL_MIN_SCORE`
   (default `0.25`) are dropped entirely; if both L2 and L3 come back
   empty, the context block is an empty string and the hook prints
   nothing — zero extra tokens for that turn.
2. **What does get injected is capped**, regardless of how much memory has
   accumulated: `LONGBRAIN_MEMORY_MAX_CONTEXT` (default `6000` characters,
   ≈1500-2000 tokens) truncates the Claude Code hook's injection. A
   `CLAUDE.md` you hand-maintain gets loaded **in full, every turn**, and
   grows as you add to it — cost per turn climbs over time. Here, cost per
   turn stays roughly flat no matter how large the memory store gets,
   because only the top-scoring, size-capped slice is ever injected.

Tune both via `.env` if you want to trade recall breadth for a smaller
footprint (raise `RECALL_MIN_SCORE`) or shrink the injection further
(lower `LONGBRAIN_MEMORY_MAX_CONTEXT`).

## Why use this instead of the alternatives?

An honest comparison — strengths and trade-offs both — so you can judge
whether it fits how you work.

**Strengths:**

- **Real long-term memory, not a stopgap.** A plain chat loses all context
  the moment you close the window. A hand-maintained `CLAUDE.md` needs you
  to update and clean it up yourself. Longbrain records, distills, recalls,
  and drops outdated information automatically — no manual intervention
  required.
- **Shared across multiple AI agents, not locked to one.** Any chat agent
  with an adapter (today: Hermes Desktop, Claude Code) runs against the
  same memory in parallel — teach something in one, the others already
  know it, no re-explaining every time you switch tools.
- **Doesn't have to cost extra money.** Runs entirely on a subscription
  you already pay for (Claude Code) or on a model running locally on your
  machine (Ollama) — a paid API key is never required.
- **Cost per turn stays roughly flat over time.** Unlike `CLAUDE.md`, where
  the entire file gets loaded on every chat turn, Longbrain only recalls
  what's actually relevant to the current question and caps the size of
  what gets injected into the prompt. As memory grows over time, the extra
  tokens spent per turn stay about the same.
- **Fully private, fully yours.** Runs on your machine, nothing syncs
  anywhere, nightly backups, complete control.
- **The memory is visible, not a black box.** The `/ui` page shows
  everything the system currently "remembers" about you as an interactive
  graph — wrong entries get corrected, stale ones get deleted, instead of
  guessing what the AI thinks it knows.
- **Cleans up after itself.** Duplicate/reworded restatements and
  outdated information (superseded by something newer) get detected
  automatically — not an ever-growing, disorganized pile of notes after a
  few months of use.
- **Moving machines doesn't mean losing data.** A built-in export/import
  bundle carries the whole memory across — a new machine, or even a
  different embedding model, and nothing is lost.

**Worth considering:**

- Needs Docker and runs 1-2 background containers — a bit more RAM/CPU
  than running nothing at all.
- Requires one initial setup step (`./setup.sh`). Most of it is automated,
  but it's still an extra step beyond just installing the AI agent.
- The quality of what gets "remembered" depends on the model doing the
  distillation — a weaker model (e.g. Ollama on modest hardware) extracts
  facts less reliably than a stronger one.
- This is **personal, single-machine** data by design — it does not sync
  across machines or share between users (a deliberate design choice, see
  [ARCHITECTURE.md](ARCHITECTURE.md), not a temporary technical limit).

## Install (3 steps)

1. Install [Docker Desktop](https://docs.docker.com/get-docker/).
2. Install Hermes Desktop and/or Claude Code — whichever agents you use.
3. In this directory run:

```bash
./setup.sh
```

**No manual steps remain.** The script automatically: creates `.env` →
builds & starts containers → waits for health → wires every installed agent
(each is skipped gracefully when absent):

- **Hermes Desktop**: registers all 4 hooks + consent into `~/.hermes/` →
  patches Hermes' `serve` bug (Desktop never registers shell hooks without
  it) → borrows an available API key (NVIDIA/Gemini) for auto-consolidation
  → adds memory-routing guidance to `~/.hermes/SOUL.md` (explicit
  "remember/forget" commands go to this stack, not Hermes' small built-in
  store) → restarts Hermes Desktop.
- **Claude Code**: registers the 4 hooks in `~/.claude/settings.json` and
  the `longbrain` MCP server (user scope) via `scripts/configure_claude.py`.
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
recall; memory injection is bounded (`LONGBRAIN_MEMORY_MAX_CONTEXT`, default
6000 chars) since it spends your subscription tokens each turn.

## Memory browser (`http://localhost:8800/ui`)

A self-contained page for exploring and curating stored memory — no extra
container, no external assets, light/dark themed:

- **Graph view — one galaxy per project**: each project gets its own
  gravity well, clustering its memories into a visually separate "galaxy"
  (named by a floating label at its center) instead of one undifferentiated
  mass; cross-project semantic links still render as faint bridges between
  galaxies without dragging them together. **Shape + color together encode
  type** (validated categorical palette, dataviz skill): ● fact — planet,
  ☄ preference — comet (tail points away from its galaxy's core), ✦ decision
  — bright star (glow halo), ◌ task — satellite (orbit ring); the `?` button
  shows the legend. Size = importance, dashed outline = superseded, a dotted
  ring = other memories from the same source session as the current
  selection. Hover highlights a node's neighborhood; click opens the detail
  panel with a selection ripple. Drag, pan, smooth wheel zoom, `Fit`, and
  click the title to reset the view.
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

- **Corrections in the detail panel**: **Change type…** re-classifies a
  memory (fact/preference/decision/task — the consolidation model doesn't
  always get it right) and **Delete this memory** hard-deletes it, both
  gated by a confirmation dialog. Deleting is permanent — unlike supersede,
  it keeps no trace; forgetting through Hermes (`forget_about`) or the
  REST API works the same way.

## Provider configuration (.env)

| Variable | Default | Meaning |
|---|---|---|
| `EMBED_PROVIDER` | `fastembed` | `fastembed` \| `ollama` \| `openai` \| `nvidia` |
| `EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model (multilingual, CPU) |
| `LLM_PROVIDER` | `none`* | `none` \| `anthropic` \| `openai` \| `nvidia` \| `gemini` \| `ollama` |
| `LLM_MODEL` | per provider | e.g. `models/gemini-2.5-flash`, `claude-sonnet-5` |
| `*_API_KEY` | — | `ANTHROPIC` / `OPENAI` / `NVIDIA` / `GOOGLE` — setup.sh borrows an existing key from `~/.hermes/.env` when the provider is `none` |
| `LONGBRAIN_USER_ID` | `local` | Stamped into every payload (future multi-user server = no migration) |

> Every `LONGBRAIN_*` variable also accepts its pre-rename `HERMES_*` name
> as a legacy alias, so an existing install keeps working untouched.

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
curl -X PATCH localhost:8800/memory/facts/<id>/type -H 'Content-Type: application/json' \
  -d '{"type": "decision"}'                         # reclassify (fact/preference/decision/task)
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

Memory is automatically partitioned **by project**, regardless of which
agent is chatting: the hook reads the chat's working directory (`cwd`) and
resolves it to a project slug, stamping `project_id` on every record. Two
folder→slug sources are merged so this works whether or not Hermes Desktop
is even installed: Hermes' own `~/.hermes/projects.db` (sidebar projects,
when present) and `~/.hermes/discovered_projects.json` (a fallback catalog
the Claude Code adapter fills in on its own — see "Auto-ingest documents"
below). When neither has a match, the project currently **selected in the
Hermes sidebar** is used if Hermes is running — so chat-only projects work
with no folder at all. Recall boosts same-project memories (×1.5) while
cross-project knowledge can still surface when genuinely relevant;
documents are hard-filtered by project. Creating a new project just works —
zero configuration. Details: [ARCHITECTURE.md](ARCHITECTURE.md).

## Backup

Runs automatically at **2:00 AM daily** (launchd; also once at every
boot/login via `RunAtLoad` in case the machine was powered off; 7 newest kept;
log at `logs/backup.log`) — installed by setup.sh. Manual run:

```bash
./scripts/backup.sh    # snapshots every longbrain_* collection into ./backups/
```

## Auto-ingest documents (docs/ watcher)

Manually `curl`-ing every file into the knowledge base gets old fast. Instead,
drop files into a **`docs/` subfolder inside a project's own folder** —
`scripts/ingest_watcher.py` picks them up automatically:

- Installed by `setup.sh`/`configure_hermes.py` as a launchd agent, one poll
  pass every 60s (`com.longbrain.memory-ingest.plist.template`; not a long-lived
  daemon, no `watchdog`/inotify dependency — "poll" means comparing
  (mtime, size) against the last run).
- **Works with or without Hermes Desktop.** Project folders come from two
  merged sources: Hermes' `~/.hermes/projects.db` when it's installed, and
  `~/.hermes/discovered_projects.json` — a fallback catalog the Claude Code
  adapter fills in on its own the first time you chat inside a project folder
  (`hooks/project_catalog.py`). A Claude-Code-only machine with no Hermes
  Desktop at all still gets every project discovered this way; nothing to
  configure.
- **Opt-in per project**: a project is only watched if `<project folder>/docs/`
  exists — a code repo is never ingested wholesale by accident.
- Supported: `.pdf .md .txt .docx`. New/changed files are sent to
  `/ingest/file` tagged with that project's `project_id`; unchanged files are
  skipped both locally (a state file at `~/.hermes/ingest_watcher_state.json`)
  and server-side (`documents.already_ingested` — a safety net if that state
  file is ever lost, so re-running the watcher never piles up duplicate chunks).
- Log: `logs/ingest_watcher.log`. Manual single pass: `python3 scripts/ingest_watcher.py`.

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
longbrain/
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
│   ├── project_catalog.py   # agent-agnostic project→folder fallback (no Hermes needed)
│   └── claude/              # Claude Code adapter (same lifecycle, 4 hooks)
├── scripts/
│   ├── configure_hermes.py  # auto-wire Hermes (hooks + consent + serve patch + key + backup + ingest)
│   ├── configure_claude.py  # auto-wire Claude Code (settings.json hooks + MCP)
│   ├── backup.sh            # Qdrant snapshots (called nightly by launchd)
│   ├── ingest_watcher.py    # auto-ingest each project's docs/ folder (called every 60s by launchd)
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

- **After every Hermes Desktop update, re-run `./setup.sh`** — updates
  overwrite the `serve` patch (without it the Desktop backend never
  registers hooks).
- **After editing any file in `hooks/`: re-run `./setup.sh`** — hook consent
  is tied to the script's mtime.
- **To delete memory, tell Hermes ("forget about X") or use the API** —
  avoid deleting in the Qdrant dashboard: it has unconfirmed full write
  access and makes it look like the system "lost data" on its own.
