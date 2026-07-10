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
| 2026-07-10 | **Agent support tiers + doctor**: formal tier model (full adapter / write adapter / MCP-only / generic MCP) documented in `adapters/README.md`; Codex wired with MCP + turn-ended notify recording (`scripts/configure_codex.py`, `hooks/codex/turn_ended.py`, detected by setup.sh); `scripts/doctor.py` — one-shot read-only wiring + health check across service, launchd jobs and all agents, with `--fix` re-running setup. |

## Next (in order)

1. **`last_seen` for facts** — recalling a fact should refresh it against
   time decay, so long-standing truths don't fade just because they're old.
2. **Memory health dashboard in `/ui`** — counts, growth, consolidation
   backlog, superseded ratio: is the memory healthy at a glance.
3. **`status` field for tasks** — task-type memories should be closable
   (open/done) instead of deleted or superseded.
4. **Contradiction detector** — flag when a new fact conflicts with a stored
   one instead of silently keeping both (needs an LLM on the save path).
5. **Graph semantic layers** — group the `/ui` galaxy by topic within a
   project, not only by similarity edges.

## Further out

- **Codex full adapter** — Codex can record completed turns via `notify`
  today, but recall is still tools-only because there is no pre-prompt
  injection hook. Upgrade to full adapter if/when Codex exposes the missing
  lifecycle moments. A Cursor adapter would prove the contract on another
  agent nobody tuned for.
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
