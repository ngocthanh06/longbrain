# Roadmap

What's live, what's next. Historical design details of completed phases are
preserved in git history (`UPGRADE_PLAN.md`, removed 2026-07-10).

## Completed milestones

| When | Milestone |
|---|---|
| 2026-07-05 | **v2 core + project partitioning** — four memory layers, deterministic ids, dedup/supersede, per-project scoping. |
| 2026-07-05 | **Phase A — automatic lifecycle**: consolidation on session end + 30-min idle sweep; recall auto-injected into every turn (`pre_llm_call`); memory management (list/forget REST + MCP). |
| 2026-07-05 | **Phase B — automated backup**: launchd nightly at 2:00 AM, 7-copy retention. |
| 2026-07-07 | **Phase D — docs/ auto-ingest watcher**: stdlib-only 60s poll over each project's `docs/` folder, duplicate-guarded; works on Claude-Code-only machines via the adapter-maintained project catalog. |
| 2026-07-09 | **Recall eval harness** (`scripts/recall_eval.py` + baseline) — recall quality is now measured, not guessed; rule-based memory router (docs channel on trigger words); session summaries in consolidation; adapter SDK docs + minimal example. |
| 2026-07-10 | **Phase C — recall quality, closed**: C1 embedding swap **rejected by measurement** (two blind LLM-judged benchmarks vs `multilingual-e5-large` below the 1.3× decision bar; MiniLM stays). C2 **hybrid BM25 shipped**: sparse exact-token rescue channel on all three collections, gated to identifier-like queries; exact-token hit@top-2 went 1/12 → 11/12 with byte-identical results for prompts without such tokens. Kill switch `HYBRID_BM25=false`. |
| 2026-07-10 | **Identity cleanup**: hermes → longbrain rename across MCP server, containers, image, env (legacy `HERMES_*` aliases kept), Qdrant collections (migrated 1144/1144 points); `/ui` named-vector fix; preference boost in recall (`RECALL_PREFERENCE_BOOST`, trap-tested in the eval set). |
| 2026-07-11 | **v0.1.0 (beta / public preview)**: CI on GitHub Actions with the recall eval as a required gate; README states eval-backed numbers and explicit known limitations; `llms.txt`; Codex write adapter shipped. |
| 2026-07-11 | **Codex lifecycle adapter**: official `SessionStart`, `UserPromptSubmit`, and `Stop` hooks add automatic recall injection, turn recording, and consolidation catch-up; rollout-scanning `notify` retained as a compatibility fallback. |
| 2026-07-11 | **KG-lite — triple-based supersession + entity graph**: facts may carry a `(subject, relation, object)` triple; a new fact sharing subject+relation with an active one retires it, catching plain contradictions (pnpm → bun) that cosine similarity and the LLM dedup band both miss. Kill switch `TRIPLE_SUPERSEDE=false`. Adds `/memory/graph?mode=entities` + an Entities view in `/ui`, and `scripts/backfill_triples.py` (dry-run writes a plan file; `--apply` writes exactly that plan, zero new LLM calls). |
| 2026-07-10 | **Agent support tiers + doctor**: formal tier model (full adapter / write adapter / MCP-only / generic MCP) documented in `adapters/README.md`; Codex wired with MCP + turn-ended notify recording (`scripts/configure_codex.py`, `hooks/codex/turn_ended.py`, detected by setup.sh); `scripts/doctor.py` — one-shot read-only wiring + health check across service, launchd jobs and all agents, with `--fix` re-running setup. |
| 2026-07-11 | **Memory quality-of-life batch**: `last_seen` refreshes a fact's decay clock on recall (`LAST_SEEN_REFRESH`); task-type facts get an open/done `status` (`HIDE_DONE_TASKS`); a contradiction detector flags conflicting facts via `conflicts_with` instead of silently keeping both (`CONTRADICTION_DETECTION`); a memory health dashboard (`GET /memory/stats` + `/ui` panel: counts, superseded ratio, 24h/7d growth, consolidation backlog); `/ui` graph gets topic sub-clustering via connected-components over existing similarity data (`GRAPH_TOPIC_CLUSTERING`, zero new LLM calls). |
| 2026-07-11 | **KG-lite follow-up audit**: closed the 2 open items from the original KG analysis — added a genuine multi-hop eval case (`vn-multihop-project-deploy`, confirmed failing on purpose — the entity graph is `/ui`-visualization-only, not wired into recall) and deleted the 2 specific meta-about-system facts the earlier data audit had flagged but never removed. Separately, root-caused and fixed Codex Desktop's per-chat scratch cwd (`~/Documents/Codex/<date>/<title-slug>`) being persisted as a junk project — `resolve_project` now recognizes and skips that layout. |

## Next (in order)

Nothing queued — the previous "Next" batch (items 1-5 above) shipped this
session. Candidates below are unprioritized; pick one or say what's next.

## Further out

- **Codex session-end completion** — automatic recall and recording now use
  official lifecycle hooks. Move consolidation from next-session catch-up to
  chat-close when Codex exposes a `SessionEnd` event. A Cursor adapter would
  prove the contract on another agent nobody tuned for.
- **"New machine in 10 minutes"** — a packaged restore experience around the
  transfer bundle: install, import, keep working.

## Deliberate non-goals

Decided and not up for re-litigation without new evidence:

- **No multi-user / sync** — single-user, local-first by design
  (privacy-first positioning).
- **No LLM at document-ingest time** — L4 stays pure chunk-RAG: no API key
  required, no doc digests, no `list_documents` tool (weighed 2026-07-09).
- **No embedding model swap** — measured twice, inconclusive both times;
  revisit only with a materially better multilingual model in fastembed.
- **No adapter-registry abstraction, no generic wrapper/proxy** — one
  `configure_<agent>.py` per agent called from setup.sh *is* the registry;
  merging them (or building a universal CLI wrapper for hook-less agents)
  waits for real demand, not aesthetics (decided 2026-07-10).
