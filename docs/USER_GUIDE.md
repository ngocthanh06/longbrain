# User Guide

Everything about living with Longbrain day to day. For install, see the
[README](../README.md); for endpoint details, the [API reference](API.md);
for how it works inside, the [architecture](ARCHITECTURE.md).

## How your memory works

You don't operate Longbrain — you just chat. The whole lifecycle runs
automatically: **record → recall → consolidate → controlled forgetting →
backup.**

- Every completed turn is recorded into episodic memory (L2).
- When a session ends (or has sat idle), it is **consolidated**: distilled
  into a handful of durable facts, preferences, decisions and tasks (L3).
  Restating something supersedes the old version instead of piling up.
- Before every turn, a hook recalls what's relevant and injects it into the
  prompt — so an agent "just knows" what you taught another agent last week.

### How context gets built (and why the cost stays flat)

Before every chat turn, the adapter hook calls `POST /memory/recall`, which
merges three sources into one context block (full sequence diagram:
[ARCHITECTURE.md §3c](ARCHITECTURE.md#3-data-flows)):

- **L3** — distilled facts relevant to what you just asked
- **L2** — related past conversations from *other* sessions
- **L1** — the current session's own recent turns

Search is **hybrid**: dense semantic cosine plus a BM25 keyword channel that
only fires when the question contains identifier-like tokens
(`ERR_UPLOAD_413`, `10MB`, `snake_case`, quoted strings, …) and can only
*rescue* exact matches the semantic model under-ranks — it never demotes a
semantic hit. Kill switch: `HYBRID_BM25=false`.

Two things keep this cheap instead of turning into another growing
`CLAUDE.md`:

1. **Nothing relevant → nothing injected.** Results below `RECALL_MIN_SCORE`
   (default `0.25`) are dropped entirely; if nothing survives, the hook
   prints nothing — zero extra tokens for that turn.
2. **What does get injected is capped** by `LONGBRAIN_MEMORY_MAX_CONTEXT`
   (default `6000` characters ≈ 1500–2000 tokens), no matter how large the
   store grows. A hand-maintained `CLAUDE.md` loads **in full, every turn**
   and grows over time; here, cost per turn stays roughly flat.

Tune both via `.env`: raise `RECALL_MIN_SCORE` for a stricter filter, lower
`LONGBRAIN_MEMORY_MAX_CONTEXT` for a smaller injection.

### Per-project memory

Memory is automatically partitioned **by project**, regardless of which
agent is chatting: the hook reads the chat's working directory (`cwd`) and
resolves it to a project slug, stamping `project_id` on every record. Two
folder→slug sources are merged, so this works whether or not Hermes Desktop
is installed: Hermes' own `~/.hermes/projects.db` (sidebar projects) and
`~/.hermes/discovered_projects.json` (a catalog the Claude Code adapter
fills in on its own). When neither matches, the project currently selected
in the Hermes sidebar is used if Hermes is running — chat-only projects work
with no folder at all.

Recall boosts same-project memories (×1.5) while cross-project knowledge can
still surface when genuinely relevant; documents are hard-filtered by
project. Creating a new project just works — zero configuration.

### Running multiple agents in parallel

Nothing to switch: every installed agent's lifecycle hooks stay registered
and write to the same service. What you teach one agent, the others recall.
Every record carries `source_agent` (`hermes` / `claude-code` / `codex`) —
visible in the `/ui` detail panel.

## Memory browser (`/ui`)

Open `http://localhost:8800/ui` — a self-contained page for exploring and
curating stored memory (no extra container, light/dark themed):

- **Graph view — one galaxy per project**: each project clusters its
  memories into a visually separate "galaxy"; cross-project semantic links
  render as faint bridges. **Shape + color encode type**: ● fact — planet,
  ☄ preference — comet, ✦ decision — bright star, ◌ task — satellite (the
  `?` button shows the legend). Size = importance, dashed outline =
  superseded, a dotted ring = memories from the same source session as the
  selection. Hover highlights a neighborhood; click opens the detail panel.
  Drag, pan, smooth wheel zoom, `Fit`, click the title to reset.
- **Spotlight search (⌘K)**: live semantic search — matches highlight on the
  graph and the camera glides to the best hit.
- **Filters**: project and type chips, a superseded switch, plus link
  controls (minimum-similarity slider, "same project only" toggle).
- **Detail panel**: full text, metadata, related memories, and the
  source-session transcript rendered as markdown.
- **List view** (same filters, as a table) and **PNG export** of the graph.

## Fixing what it remembers

Memory you can't correct isn't memory you can trust. Every correction is
available in `/ui` (and as [REST calls](API.md#re-tagging-project-corrections)):

- **Wrong type?** Detail panel → **Change type…** (fact / preference /
  decision / task — the consolidation model doesn't always get it right).
- **Wrong project?** Move one memory, re-tag a whole session (turns + facts
  move, future turns follow), bulk re-tag a ⇧click multi-selection
  ("Select linked" expands it to the whole connected cluster), or rename a
  project everywhere (the ✎ on a project chip).
- **Wrong fact?** Detail panel → **Delete this memory**, or just tell your
  agent *"forget about X"* — it lists candidates and asks before deleting
  (`forget_about` → `forget_memory`).
- **Everything?** `forget_everything` / `DELETE /memory/all` — both demand
  the literal confirmation `DELETE ALL`.

Deleting is permanent (unlike supersede, it keeps no trace). Avoid deleting
in the Qdrant dashboard directly — it bypasses every guard.

## Auto-ingest documents (the `docs/` folder)

Drop files into a **`docs/` subfolder inside a project's folder** and they
flow into the knowledge base automatically — no curl needed:

- `scripts/ingest_watcher.py` runs every 60s (launchd agent installed by
  `setup.sh`; a single poll pass, no daemon).
- **Works with or without Hermes Desktop** — project folders come from
  `~/.hermes/projects.db` and the adapter-maintained
  `~/.hermes/discovered_projects.json`, merged.
- **Opt-in per project**: only watched if `<project folder>/docs/` exists —
  a code repo is never ingested wholesale by accident.
- Supported: `.pdf .md .txt .docx`. New/changed files go to `/ingest/file`
  tagged with the project's slug; unchanged files are skipped locally
  (state file) and server-side (duplicate guard), so re-runs never pile up
  duplicate chunks.
- Log: `logs/ingest_watcher.log`. Manual single pass:
  `python3 scripts/ingest_watcher.py`.

Once ingested, ask your agent about the document, or search explicitly with
the `search_knowledge_base` tool.

## Backup

Runs automatically at **2:00 AM daily** (launchd; also once at every
boot/login in case the machine was off at 2 AM). The 7 newest snapshots are
kept; log at `logs/backup.log`. Manual run:

```bash
./scripts/backup.sh    # snapshots every longbrain_* collection into ./backups/
```

Restoring a snapshot is an operations task — see
[OPERATIONS.md](OPERATIONS.md#restoring-a-backup).

## Moving to another machine (export / import)

Nightly snapshots are binary and tied to the embedding model — they restore
the same machine, but can't move memory to a new install that may run a
different model. For that, use the text-level transfer bundle:

```bash
# old machine
./scripts/memory_transfer.sh export            # -> backups/memory-export-<stamp>.json

# new machine (after setup.sh, service running)
./scripts/memory_transfer.sh import memory-export-<stamp>.json
```

The bundle contains payload text only (facts, chat turns, document chunks) —
no vectors. Import **re-embeds everything with the current model**, keeps
original timestamps / supersede links / consolidated flags, and skips
records that already exist — running it twice is safe. The same transfer is
available in `/ui` (`⇩ Export` / `⇪ Import` in the header, with a
confirmation showing the bundle's counts before anything is written).
