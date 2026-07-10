# Memory Stack Upgrade Plan (improvements 1–7)

> Baseline status: v2 + project partitioning running stably (2026-07-05).
> The plan is split into 4 phases by dependency order; each phase deploys independently.
>
> **PROGRESS (2026-07-05 afternoon):**
> - ✅ **Phase A complete** — A1 auto-consolidation (hook `on_session_end` +
>   30-min sweep, LLM = NVIDIA deepseek-v4-pro via OpenAILike because the
>   `llama-index-llms-nvidia` adapter doesn't accept new models; added a
>   fallback `gemini` provider), A2 auto-inject (`pre_llm_call` returns
>   `{"context": ...}`, recent_turns=0 to avoid duplicating history), A3 memory
>   management (list/forget REST + MCP, DELETE session). NVIDIA key auto-synced
>   from ~/.hermes/.env.
> - ✅ **Phase B complete** — launchd backup at 2:00 AM, 7-copy retention,
>   log at logs/backup.log (BSD head fixed with awk).
> - ✅ **Phase C complete (2026-07-10) — C1 (model swap) CLOSED without migrating;
>   C2 (hybrid BM25) SHIPPED.** C2: `app/hybrid.py` adds a named `bm25` sparse
>   vector (fastembed `Qdrant/bm25`, IDF on the Qdrant side) to all three
>   collections; recall fuses client-side as
>   `max(dense_cosine, RECALL_BM25_WEIGHT * bm25_ratio)` — NOT server RRF,
>   whose rank-scores would break every cosine-calibrated mechanism
>   (min-score, decay, importance, boost). Query side only searches
>   identifier-like terms (digits, snake_case, CamelCase, acronyms, quoted
>   spans): full natural-language questions let common Vietnamese words
>   outscore the rare token (measured 0/12 rescued on the real corpus;
>   fastembed has no VN/JA stopwords). With the gate: exact-token hit@top-2
>   on the real 450-chunk corpus went 1/12 (dense) -> 11/12 (hybrid), and
>   prompts without such tokens return byte-identical dense results (eval
>   10/10, 0 violations). Qdrant can't add a sparse vector to an existing
>   collection, so `scripts/migrate_hybrid_bm25.py` recreated the
>   collections carrying dense vectors over (no re-embed; gzip dump in
>   backups/hybrid_migration_* first). Migrated + deployed 2026-07-10
>   (365+283+450 points, 100% with sparse). Kill switch: `HYBRID_BM25=false`.
>   `BAAI/bge-m3` (the model this section originally proposed) turned out to
>   not exist in fastembed at all (checked 0.7.4 and 0.8.0) — substituted
>   `intfloat/multilingual-e5-large`, the strongest multilingual model
>   fastembed does support. `scripts/benchmark_embeddings.py` ran a blind
>   LLM-judged comparison on 15 real Vietnamese chat queries against the
>   real `longbrain_memories` corpus: candidate won 6, current (MiniLM) won 4,
>   5 ties. Too thin a margin on too small a sample to justify a full
>   re-embed migration — staying on MiniLM.
>   **Re-run 2026-07-10** with the larger sample that note asked for (40 real
>   queries, corpus = 256 facts + 200 L4 chunks, e5 given its proper
>   `query:` prefix via `get_query_embedding`): candidate 22, current 17,
>   tie 1 — still below the 1.3× decision bar, second inconclusive result.
>   `recall_eval.py` under `EMBED_MODEL=e5-large`: same 9/9 hits but
>   2 violations and +63% injected chars (recall/dedup thresholds are tuned
>   to MiniLM's cosine distribution; migrating would also mean a threshold
>   retune pass). **C1 closed for good — MiniLM stays.** The misses that
>   motivated Phase C (exact tokens like "10MB" not ranking) are keyword
>   matching, which C2 hybrid BM25 addresses without touching the dense model.
> - ✅ **Phase D complete (2026-07-07)** — `scripts/ingest_watcher.py` (stdlib-only,
>   polls each project's `docs/` subfolder from `~/.hermes/projects.db`, one
>   pass per invocation) + `com.longbrain.memory-ingest.plist.template` (launchd,
>   `StartInterval=60`, installed by `configure_hermes.py`). Along the way,
>   fixed a pre-existing bug: `/ingest/file` 500'd on every call because
>   `llama-index-readers-file` was missing from requirements.txt despite
>   `pypdf` being pinned for it — added. Also added a real dedup guard
>   (`documents.already_ingested`, filtered on the `stored_path` payload
>   field) since `/ingest/file` had none — re-ingesting an unchanged file
>   used to silently pile up duplicate chunks. Verified end-to-end on real
>   data (42 real project doc files → 300 chunks, second pass 0 sent/42
>   unchanged). **Follow-up same day**: the original design only read
>   `~/.hermes/projects.db`, so a Claude-Code-only machine (no Hermes
>   Desktop installed) had no project folders to watch at all — added
>   `hooks/project_catalog.py`, an agent-agnostic fallback the Claude Code
>   adapter fills in on its own (no Hermes required); the watcher now merges
>   both sources. Verified with a from-scratch simulated no-Hermes machine.

## Foundational findings (verified in the Hermes source)

Hermes supports more hook events than the `post_llm_call` currently in use:

| Event | Used for | Mechanism |
|---|---|---|
| `pre_llm_call` | **#2 auto-inject memory** | stdout `{"context": "..."}` is officially injected by Hermes into the chat turn |
| `on_session_end` | **#1 auto-consolidation** | extra contains `completed`, `model`, `platform` — fires exactly when a session ends |
| `on_session_start` | optional: warm-up recall | extra contains `model`, `platform` |

---

## Phase A — Complete the memory lifecycle (do first, highest value)

### A1. Automatic consolidation (#1)

**Problem:** `longbrain_memories` only gets data when someone actively calls the tool — in practice nobody will.

**Design — 2 trigger tiers:**
1. **`on_session_end` hook** (new: `hooks/on_session_end.py`): session ends
   (`completed=true`) → `POST /memory/consolidate {session_id}` (fire-and-forget,
   short timeout, best-effort like the existing hook).
2. **Periodic sweep in the service** (new: `app/scheduler.py`, an asyncio task in
   the lifespan): every `CONSOLIDATION_INTERVAL` (default 30 minutes) scan for
   sessions with ≥ `CONSOLIDATION_MIN_TURNS` (4) undistilled turns AND idle for
   more than `CONSOLIDATION_IDLE_SECONDS` (15 minutes) → consolidate. Catches
   sessions the hook missed (crash, machine shut down mid-session).

**Prerequisite:** the service needs an LLM (`LLM_PROVIDER != none`). Recommended:
`anthropic` + `claude-haiku` (cheap, sufficient for extraction) or the `ollama`
profile. With `none`: the scheduler only logs a warning + exposes
`GET /memory/pending-consolidation` for the Hermes side to handle manually.

**Work:** `app/scheduler.py` (new, ~80 lines) · `hooks/on_session_end.py` (new, ~40 lines)
· `config.py` +4 env vars · `main.py` lifespan +5 lines · `configure_hermes.py` registers
the new hook + consent · test: create a fake session → wait for the sweep → the fact appears.

**Risk:** poor LLM extraction quality with too-small a model → the prompt already
exists; test with 2-3 models before finalizing the recommendation. API cost:
~1 call/session, negligible.

### A2. Automatic memory injection into chat turns (#2)

**Problem:** recall depends on Hermes' model remembering to call the tool — unreliable.

**Design:** new hook `hooks/pre_llm_call.py`:
```
stdin: {session_id, cwd, extra: {user_message?, ...}}
  → POST /memory/recall {query: user_message, session_id, project: resolve(cwd)}
  → stdout: {"context": context_block}   ← officially injected by Hermes
```
- **Step 0 (discovery):** the real `pre_llm_call` payload is unverified
  (the docs don't list the extra keys) → enable a 1-day debug dump like we did
  for post_llm_call, pin down the schema before coding the parsing.
- **Latency control** (the hook runs synchronously before every LLM call):
  3s timeout, on failure → return empty; only inject when recall has real
  results (context_block non-empty); option `RECALL_INJECT_EVERY_N_TURNS`
  (default: every turn; measure in practice, then tune).
- Estimated cost per turn: 1 embed + 3 searches ≈ 100–300ms on an M-series machine.

**Work:** `hooks/pre_llm_call.py` (new, ~70 lines, reuses `resolve_project`)
· `configure_hermes.py` registration + consent · measure real latency · A/B test:
ask again about info from an old session without calling any tool — Hermes must
already know.

### A3. Memory management tools (#3)

**Problem:** when memory remembers something wrong, there is no way to fix it
short of digging through the Qdrant dashboard.

**Design:**
- REST: `GET /memory/facts?project=&type=&limit=` (list, with ids)
  · `DELETE /memory/facts/{id}` · `DELETE /sessions/{session_id}` (delete a whole session)
- MCP: `list_memories(project?)` · `forget_memory(id)` ·
  `forget_about(query)` — search top-5, return a list with ids so the model
  picks what to delete (2 steps, avoids similarity-based accidental deletion).
- Deleting a fact = hard-deleting the point (unlike supersede — supersede is a
  natural replacement, forget is a user command).

**Work:** `memories.py` +3 functions · `main.py` +3 endpoints · `mcp_server.py` +3 tools
· test: save → forget_about → recall no longer sees it.

**Phase A total:** ~1 working session. No migration, no schema change.

---

## Phase B — Automated operations (#7, can be done alongside Phase A)

### B1. Automatic backup

**Design:** a launchd agent on macOS (only the host can reach both volumes via the API):
- `scripts/backup.sh` upgraded: add retention (keep the `BACKUP_KEEP=7` newest),
  log to `logs/backup.log`, snapshot all 3 collections + copy `.env`.
- `scripts/com.longbrain.memory-backup.plist` — runs daily at 2:00 AM.
- `configure_hermes.py` (or setup.sh) installs the plist into `~/Library/LaunchAgents`
  + `launchctl load`. Idempotent.

**Work:** ~30 minutes. Test: manual `launchctl start` → check `backups/`.

---

## Phase C — Recall quality (#4 + #5, DO TOGETHER in a single migration)

> The two improvements are merged because both require re-creating collections
> + re-embedding. Doing them separately = paying the migration cost twice.

### C1. Upgrade embedding to BAAI/bge-m3 (#5)

- Change the default to `EMBED_MODEL=BAAI/bge-m3` (1024-dim, strong
  multilingual — a clear improvement for Vietnamese). The image grows ~2.2GB —
  acceptable, still baked in.
- Weaker machines can keep MiniLM via `.env` — all the guards already exist.

### C2. Hybrid search: dense + sparse BM25 (#4)

- **L4 documents:** `QdrantVectorStore(enable_hybrid=True, fastembed_sparse_model="Qdrant/bm25")`
  — LlamaIndex handles everything (named vectors + RRF fusion). Requires a new collection.
- **L2/L3 (self-managed collections):** add a named sparse vector on write
  (`fastembed SparseTextEmbedding`), switch search to the Query API
  (`prefetch dense + sparse → RRF fusion`). This is the largest code change of
  the whole plan (~150 changed lines in `memory_store.py`/`memories.py`).

### C3. Migration script `scripts/reembed.py`

```
1. Automatic backup before running
2. Create *_new collections with the new config (dense 1024 + sparse)
3. Re-embed: L2/L3 from payload text (100% coverage), L4 from original files in /data/documents
4. Verify point counts match → swap names (delete old, rename new) → update longbrain_meta
5. Rollback: restore from the step-1 snapshot
```

**Phase C total:** ~1-2 sessions. Main risk: `llama-index-vector-stores-qdrant`
0.4.x compatibility with hybrid — check early; if blocked, do a controlled
minor-version bump. Decision point before starting: measure bge-m3 vs MiniLM
quality on your own Vietnamese data (a script comparing recall on 10-20 sample
questions — half a session, avoids a pointless migration).

---

## Phase D — Auto-ingest documents per project (#6) ✅ done (2026-07-07)

**Problem:** getting documents into the knowledge base requires manual curl.

**Design:** a watcher running on the **host** (the container can't see the user's folders):
- `scripts/ingest_watcher.py`: read the project folder list from
  `~/.hermes/projects.db` (reusing the resolver) → watch each project's `docs/`
  subfolder (opt-in per subfolder, avoids accidentally ingesting a whole code
  repo) → new/changed files (`.pdf .md .txt .docx`) → `POST /ingest/file` with
  the right `project_id`.
- Dedup: the service already content-addresses original files (sha) — the
  watcher just sends; the service skips duplicates (needs an added sha check
  before re-indexing on the service side).
- Run via a launchd agent (like the backup), 60s poll — no `watchdog` lib
  needed, fewer dependencies.

**Work:** `scripts/ingest_watcher.py` (~120 lines) · sha check in `documents.py`
· plist + installation in setup.sh · README documenting the `docs/` folder
convention. ~1 session. Decision to settle first: which folder gets watched
(proposal: `<project_folder>/docs/`).

---

## Execution order summary

| Step | Content | Effort | Depends on |
|---|---|---|---|
| A1 | Auto-consolidation (session_end hook + sweep) | half a session | LLM provider decision |
| A2 | Auto-inject recall (pre_llm_call) | half a session | payload discovery first |
| A3 | Memory management (list/forget) | 2-3 hours | — |
| B1 | Automatic backup (launchd) | 30 minutes | — |
| C1-3 | bge-m3 + hybrid + migration | 1-2 sessions | benchmark before migrating |
| D | Auto-ingest watcher | ✅ done | settle the docs/ folder convention |

**Questions to settle before coding:**
1. A1: which LLM for consolidation? (proposal: `anthropic` + Haiku if a key is
   available, otherwise the `ollama` profile + qwen3)
2. D: agree on watching `<project folder>/docs/`?
