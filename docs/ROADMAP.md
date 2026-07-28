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
| 2026-07-17 | **Document search upgrade** ([SEARCH_SPEC](SEARCH_SPEC.md)): Sprint 0 eval set (38 vague-VN → JA/EN cases from the real corpus) gated the decision — BGE-M3 doc-level hit@3 0.816 vs MiniLM 0.447 (1.83×, bar 1.3×). Shipped: separate doc embedder (`DOC_EMBED_*`, documents collection ONLY — memories/history stay on MiniLM, distillation thresholds untouched), `huggingface` provider path (fastembed has no bge-m3), doc-space meta guard, 384→1024d migration (`scripts/migrate_doc_embed.py`), typed 503 `doc_embedder_unavailable` error contract + `/health` readiness, cross-encoder reranker behind `DOC_RERANK`, and the Ollama-gated Optional tier (AI summary chunks via `DOC_LLM_*`/`DOC_ENRICH`, on-demand `/query/explain` with coarse labels — no fake confidence %). **Gate passed ≠ KPI met**: hit@3 is 0.816 offline / 0.842 service-path vs the spec KPI ≥ 0.90 — Sprints 0–2 accepted, KPI still open (planned lever: re-run the eval with Sprint 3 enrichment active). |
| 2026-07-28 | **Phase 1 — Memory Reliability, closed** ([STRATEGY.md](STRATEGY.md)): stable `document_key` (relative path, not basename) drives supersede-on-reingest for L4 (686→164 legacy pre-key chunks repaired, 0 left active/visible); recency decay for document search (`DOC_HALF_LIFE_DAYS`); item-level context budgeting (`CONTEXT_BUDGET_CHARS`, never cuts an item/header mid-line, replacing a blind character slice); a `trace` field on every recall hit (similarity/decay/boost breakdown for debugging ranking, not shown to agents); `hooks/admission_gate.py` replacing 3 copy-pasted prompt-length/context-cap gates across Hermes/Claude Code/Codex. **Verified live** (full rebuild + restart, not just unit tests): 0 data loss across restart, re-sync produces 0 duplicates, a live ingest-v1-then-v2 smoke test confirmed correct supersession, project isolation holds under an adversarial same-query-different-project test, 0% stale-chunk leakage (queried verbatim text from a superseded chunk — only the active version came back). **Baseline**: recall hit rate 12/13 (92%, fixture eval, unchanged — known multihop gap only); context size ~880 chars/case (eval avg) and ~3.2–3.5k chars on broad live queries, always under the 6000-char budget; 0 duplicate/version-conflict groups across 98 keyed documents; recall latency ~275ms warm (SEARCH_SPEC's <500ms budget). **Known residual**: 164 chunks predating `document_key` still lack the tag (0 are active/visible — harmless) but `scripts/migrate_document_versions.py` now hard-blocks (exit 2) until they're repaired; fixing them needs manual per-group review, not automation (basename-guessing across ambiguous groups is what caused the earlier gakken/rz-server rollback). 213 tests passing. |

## Next (in order)

1. **Repair the 164 pre-key legacy chunks** — manual per-group review (listed
   in the Phase 1 closing session), not automatable safely. Unblocks
   `migrate_document_versions.py`.
2. **Phase 2 — first connector** (see [STRATEGY.md](STRATEGY.md)): reuse the
   existing `documents.py` ingest shape (no new `NormalizedDocument`
   abstraction yet — STRATEGY.md's own guardrail is no shared shape before
   2+ divergent connectors exist). Cheapest-first order agreed: Folder
   (already ~done via `ingest_watcher.py`, just needs to emit the shape
   explicitly) then Markdown URL (no auth, first real local-vs-remote
   divergence), before Google Docs/Apidog (real auth complexity).

## Further out

- **Codex session-end completion** — automatic recall and recording now use
  official lifecycle hooks. Move consolidation from next-session catch-up to
  chat-close when Codex exposes a `SessionEnd` event. A Cursor adapter would
  prove the contract on another agent nobody tuned for.
- **"New machine in 10 minutes"** — a packaged restore experience around the
  transfer bundle: install, import, keep working.
- **Identifier-gated BM25 on the document search path** — `/query` and MCP
  `search_knowledge_base` are dense-only by design today; the BM25 rescue
  channel exists only in the recall router (SEARCH_SPEC constraint 5,
  clarified 2026-07-17). Wiring the same rescue into those paths would help
  identifier queries (e.g. a ticket ID) but needs an identifier-query eval
  first — same "measured, not asserted" gate as any search change.
- **ONNX doc embedder** — replace the sentence-transformers/torch stack with
  an ONNX bge-m3 (fastembed custom model or optimum export): would shrink
  the doc-search image by ~5-6GB and likely speed up CPU inference 2-4×.
  Requires verifying the exported model reproduces the accepted eval numbers
  (same gate as any vector-space change).

## Deliberate non-goals

Decided and not up for re-litigation without new evidence:

- **No multi-user / sync** — single-user, local-first by design
  (privacy-first positioning).
- **No CLOUD LLM at document-ingest time** — narrowed 2026-07-17: optional
  ai-summary enrichment now exists, but strictly via a local LLM
  (`DOC_LLM_*`, Ollama) and strictly optional — no API key ever required,
  ingest works identically without it (SEARCH_SPEC Sprint 3).
- **No embedding model swap for MEMORY collections** — measured twice,
  inconclusive both times; MiniLM stays for facts/history (its dedup/graph
  thresholds are calibrated to it). The DOCUMENTS collection moved to
  BGE-M3 on 2026-07-17 after clearing the 1.3× bar on its own eval set —
  exactly the "new evidence" this section demands.
- **No adapter-registry abstraction, no generic wrapper/proxy** — one
  `configure_<agent>.py` per agent called from setup.sh *is* the registry;
  merging them (or building a universal CLI wrapper for hook-less agents)
  waits for real demand, not aesthetics (decided 2026-07-10).
