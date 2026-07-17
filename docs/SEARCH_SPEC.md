# Document Search Upgrade — Spec

Status: **approved 2026-07-17** (discussion: Claude Code + Codex sessions, 2026-07-16 → 17).
Scope: LongBrain's generic document search path (`longbrain_documents`, L4),
available to every supported agent and API client.
Out of scope: memory distillation (facts/history/consolidation) — see the
[design constraints](#design-constraints) below, which exist precisely to keep it out of scope.

## Problem

LongBrain document search must find documents from vague natural-language
queries across languages. For example, a vague Vietnamese description must
find the corresponding Japanese document even when the wording differs.
Today only dense MiniLM participates in that query — the BM25 channel is
identifier-gated by design (measured
2026-07-10: full natural-language BM25 has no VI/JA stopwords and drowns rare
tokens, 0/12 rescues) — and MiniLM's cross-lingual VI→JA matching is weak.

## KPI

- Vague Vietnamese query → correct Japanese/English document in **Top 3 ≥ 90%**
  on the Sprint 0 eval set (built from real ingested documents).
- Meaning-search latency **< 500 ms** (excluding on-demand verification).
- KPI is measured, not asserted: no sprint is accepted without re-running the
  eval set.

**Status (measured 2026-07-17):** hit@3 0.816 offline / 0.842 through the
service path, ~190 ms warm. The Sprint 0 gate passed (1.83× vs MiniLM 0.447,
bar 1.3×) so Sprints 0–2 are accepted — but the **≥ 0.90 KPI is not yet
met**, so the upgrade as a whole is not signed off. The planned lever is
Sprint 3 enrichment (re-run this eval set with enrichment active).

## Two tiers

| Tier | Contents | Extra install for the user |
|---|---|---|
| **Core** (Sprint 0–2) | Eval set, BGE-M3 dense for `longbrain_documents`, reranker | **None to manage by hand** — runs inside the service, activated by one command (see below) |
| **Optional** (Sprint 3–4) | AI enrichment (summary/keywords), Ask AI, on-demand verification | Ollama + a local model (e.g. Qwen3) |

**Packaging (decided 2026-07-17)**: memory-only LongBrain installs must not
carry the doc-search weight (torch + sentence-transformers ≈ 6GB of image +
2.3GB model). The default
image is lean (`INSTALL_DOC_SEARCH=false`); doc search is activated on an
existing install by `scripts/enable_doc_search.sh` (sets `.env`, rebuilds,
re-embeds existing documents, restarts — idempotent). API clients detect the
state via `GET /health` (`doc_embedder_ready`). A lean image with
`DOC_EMBED_*` configured does not
crash: document endpoints answer 503 `doc_embedder_unavailable`, memory is
unaffected.

Without Ollama the product remains fully functional for search; only
summaries/explanations are absent. Optional features detect Ollama at call
time and degrade gracefully — never break Core search.

## Design constraints

1. **Per-collection embedder.** BGE-M3 applies **only** to
   `longbrain_documents`, via a separate config
   (`DOC_EMBED_PROVIDER`/`DOC_EMBED_MODEL`, defaulting to the global
   `EMBED_PROVIDER`/`EMBED_MODEL` so existing deployments change nothing).
   It is **not** applied to memories, chat history, or session summaries.
   All distillation thresholds calibrated on MiniLM
   (`SUPERSEDE_SIMILARITY=0.92`, the 0.69–0.73 LLM-check band, graph
   0.55/0.35) stay untouched. Migrating the memory collections is a separate
   future project with its own threshold recalibration.
2. **Ollama is optional.** Core must not require it. fastembed 0.7.4 (pinned)
   has no BGE-M3 (verified 2026-07-10 and re-verified 2026-07-17), so Sprint 1
   adds a new in-service embedding provider path (sentence-transformers/ONNX)
   for the doc embedder — no user-facing install.
3. **Model download/cache responsibility is LongBrain's.** The service is
   responsible for downloading/caching the doc embedder when needed (cache in
   a persistent volume so container recreation does not re-download). If the
   model is absent and cannot be fetched, document search returns an explicit
   error — HTTP 503, code `doc_embedder_unavailable` — and `/health` exposes
   doc-embedder readiness. The service must **never** silently fall back to a
   different embedder for documents: vectors from another model are a
   different vector space and produce garbage rankings, which is worse than an
   honest error. Memory/fact search is unaffected in all cases (MiniLM is
   baked into the image via fastembed).
4. **Client contract.** Clients display `doc_embedder_unavailable` as
   "document search temporarily unavailable", clearly distinct from
   "no matching documents"; no silent empty results, no unbounded retries.
   `longbrain_documents` is shared by all agents, so clients should pass a
   `project` filter when they need hard project isolation; without it search
   is global by design. Client-owned document stores remain outside this
   collection and are not managed by this feature.
5. **Sparse strategy.** Keep BM25 identifier-gated exactly as today — and
   scoped where it lives today: the **recall router** (`documents.search_chunks`).
   `/query` and the MCP `search_knowledge_base` are dense (+ optional
   reranker) by design; extending the identifier-gated rescue to those paths
   is a ROADMAP candidate that must bring its own identifier-query eval
   first. BGE-M3 learned-sparse is **deferred**: adopt only if eval numbers
   later prove the added index cost. Cross-lingual vague queries are carried
   by dense + reranker + enrichment, not by keyword search.
   *(Clarified 2026-07-17: an earlier Sprint 2 draft said "+ BM25 rescues"
   on the reranked search path, contradicting this constraint — this
   constraint is normative.)*
6. **Reranker is Core, behind a flag.** `DOC_RERANK=true|false` (same kill-
   switch pattern as `HYBRID_BM25`), lazy-loaded, so latency can be
   benchmarked with/without.
7. **Verification is on-demand only** (Ask AI / "explain this match"). Never
   in the Meaning-search hot path (violates the <500 ms budget). No
   percentage confidence in the UI — LLM self-reported confidence is
   uncalibrated; show coarse labels (High match / Possibly related) derived
   from reranker score. Raw scores (`retrieval_score`, `rerank_score`) are
   kept internally for debugging.

## Sprints

Sprint order is the normative order (it supersedes any earlier "Phase"
numbering, which had enrichment before the local LLM it depends on).

### Sprint 0 — Eval set + baseline (the decision gate)

- Build a local, gitignored `llamaindex-service/evals/search_eval.local.json`:
  ~30–50 pairs of
  (vague Vietnamese query → expected document `source`), authored from the
  a real ingested corpus, weighted toward cross-lingual Japanese documents,
  with difficulty tags. The repository contains only
  `search_eval.example.json`; corpus-derived filenames, ticket IDs, queries,
  and generated result files must never be committed.
- Benchmark script (extends the `scripts/benchmark_embeddings.py` approach):
  doc-level hit@3 / hit@10 / MRR for the current MiniLM baseline, then BGE-M3
  on the same set.
- **Gate**: BGE-M3 must clearly beat MiniLM (project convention: ≥ 1.3× on
  the headline metric, or reaching the KPI when MiniLM does not). If it does
  not, Sprint 1 is cancelled and the result reported — no migration on vibes.

### Sprint 1 — BGE-M3 dense for documents (Core)

- New provider path for the doc embedder (sentence-transformers/ONNX inside
  the service container; constraint 2).
- `DOC_EMBED_PROVIDER`/`DOC_EMBED_MODEL` config, default = global embedder
  (constraint 1 — zero behavior change until explicitly configured).
- Meta guard (`longbrain_meta`) extended to also record the doc embedder
  (model/provider/dim); boot-time mismatch refusal now covers both spaces.
  Null-safe: meta written before this field existed skips the comparison.
- Migration script: re-embed **only** `longbrain_documents` from the
  originals kept under `/data/documents` (384d → 1024d requires recreating
  that collection; memories/history collections untouched).
- Error contract per constraint 3 (503 `doc_embedder_unavailable`, `/health`
  field).
- Accept: eval hit@3 reproduces the Sprint 0 BGE-M3 numbers through the real
  service path; the recall router's BM25 channel still works on identifier
  queries.

### Sprint 2 — Reranker (Core)

- `bge-reranker-v2-m3` on the document search path: retrieve top ~20 dense,
  rerank, return top k (BM25 stays in the recall router only — constraint 5;
  it does not feed the reranker). Lazy-loaded, `DOC_RERANK` flag
  (constraint 6).
- Accept: eval hit@3 with reranker ≥ without; measured latency budget still
  < 500 ms on the reference machine; flag off ⇒ byte-identical to Sprint 1
  behavior.
- **Measured 2026-07-17 (container CPU)**: 20 candidates ≈ 10 s, 5 ≈ 2.6 s —
  the hot path cannot afford it, so `DOC_RERANK` defaults to **false**
  (dense-only already reached hit@3 0.842 through the service path,
  ~190 ms warm). The reranker still serves `/query/explain` on demand
  (single pair ≈ 0.5 s, loaded lazily even with the flag off). Revisit the
  default together with the ONNX roadmap item.

### Sprint 3 — Local LLM + enrichment (Optional, Ollama-gated)

- Ollama detection + a doc-enrichment LLM config **separate from the global
  `LLM_PROVIDER`** used by distillation (constraint: switching enrichment to
  a local model must not silently change fact-extraction quality).
- Enrichment pipeline: per-document summary/keywords (Vietnamese/English)
  written into `longbrain_documents` payload — this turns VI→JA matching into
  same-language matching and gives BM25 a same-language target. Documents are
  never sent to any cloud LLM (standing privacy decision).
- Accept: with Ollama absent, ingest and search behave exactly as Sprint 2.

### Sprint 4 — Ask AI + on-demand verification (Optional)

- "Explain why this matched" per result + Ask AI over documents, local LLM
  only, per constraint 7 (labels, no fake percentages).
- Accept: verification is reachable only on demand; Meaning-search latency
  unchanged.

## Responsibility split (LongBrain ↔ API clients)

| | LongBrain (service) | API client |
|---|---|---|
| Models | Download/cache/readiness of doc embedder + reranker (+ talks to Ollama for Optional tier) | Nothing |
| Doc embedder missing | 503 `doc_embedder_unavailable`; readiness in `/health`; no silent fallback | Show "search unavailable", distinct from "no matches" |
| Offline | Cached models ⇒ full function; uncached ⇒ only document search errors, memory search unaffected | No unbounded retries |
