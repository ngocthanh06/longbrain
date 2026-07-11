# Operations

Setup internals, configuration, health checks, backup restore and
troubleshooting. Daily usage lives in the [User Guide](USER_GUIDE.md).

## What setup.sh actually does

`./setup.sh` is idempotent — safe to re-run any number of times. It:
creates `.env` → builds & starts containers → waits for health → wires every
installed agent (each skipped gracefully when absent):

- **Hermes Desktop** (`scripts/configure_hermes.py`): registers all 4 hooks
  + consent into `~/.hermes/` → patches Hermes' `serve` bug (Desktop never
  registers shell hooks without it) → borrows an available API key
  (NVIDIA/Gemini) for auto-consolidation → adds memory-routing guidance to
  `~/.hermes/SOUL.md` → restarts Hermes Desktop.
- **Claude Code** (`scripts/configure_claude.py`): registers the 4 hooks in
  `~/.claude/settings.json` and the `longbrain` MCP server (user scope).
  **Needs no API key**: Claude Code runs on your Claude login, and
  consolidation uses the service-side LLM or the `consolidate_session` MCP
  tool. Restart open sessions to pick the hooks up.
- **Codex** (`scripts/configure_codex.py`): registers the `longbrain` MCP
  server plus `SessionStart`, `UserPromptSubmit`, and `Stop` hooks in
  `~/.codex/hooks.json`. These provide consolidation catch-up, automatic
  recall injection, and automatic turn recording. The top-level `notify`
  wrapper remains as a write fallback for older Codex versions. After setup,
  restart Codex and review/trust the Longbrain definitions once with `/hooks`.

It also installs two launchd agents: the nightly backup and the `docs/`
ingest watcher.

## Provider configuration (.env)

| Variable | Default | Meaning |
|---|---|---|
| `EMBED_PROVIDER` | `fastembed` | `fastembed` \| `ollama` \| `openai` \| `nvidia` |
| `EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model (multilingual, CPU) |
| `LLM_PROVIDER` | `none` | `none` \| `anthropic` \| `openai` \| `nvidia` \| `gemini` \| `ollama` |
| `LLM_MODEL` | per provider | e.g. `models/gemini-2.5-flash`, `claude-sonnet-5` |
| `*_API_KEY` | — | `ANTHROPIC` / `OPENAI` / `NVIDIA` / `GOOGLE` — setup.sh borrows an existing key from `~/.hermes/.env` when the provider is `none` |
| `LONGBRAIN_USER_ID` | `local` | Stamped into every payload (future multi-user server = no migration) |

> Every `LONGBRAIN_*` variable also accepts its pre-rename `HERMES_*` name
> as a legacy alias, so an existing install keeps working untouched.

- **The LLM is freely swappable** — it is only used for consolidation and
  `/chat`. With `none`, the agent's own model handles consolidation through
  the `consolidate_session` MCP tool.
- **The embedding is a one-time choice** — changing it changes the vector
  space. The service records model + dimension in a meta collection and
  **refuses to boot** on a mismatch. To really switch: export a transfer
  bundle → `docker compose down -v` → edit `.env` → import (the bundle
  re-embeds with the new model — see
  [USER_GUIDE.md](USER_GUIDE.md#moving-to-another-machine-export--import)).
- Local Ollama (optional): `docker compose --profile ollama up -d`, then set
  `LLM_PROVIDER=ollama` and `OLLAMA_BASE_URL=http://ollama:11434`.

Recall tuning knobs (`RECALL_MIN_SCORE`, `LONGBRAIN_MEMORY_MAX_CONTEXT`,
`HYBRID_BM25`, boost factors) are documented in `.env.example` and
[ARCHITECTURE.md](ARCHITECTURE.md).

## Health checks

```bash
python3 scripts/doctor.py    # one-shot check: service, launchd jobs, every agent's wiring
curl localhost:8800/health
```

`doctor.py` is read-only and skips agents that aren't installed;
`doctor.py --fix` re-runs `./setup.sh` when problems are found.

- `status` must be `ok` and the three `longbrain_*` collection counts
  non-null.
- **`last_written_at` must advance after every chat turn** — if it doesn't,
  the hooks aren't firing (see troubleshooting below).
- Container-level: `docker compose ps` — `qdrant` and `llamaindex` should be
  `Up (healthy)`.

## Restoring a backup

Nightly snapshots live in `./backups/` (7 newest kept, log at
`logs/backup.log`). They are **Qdrant binary snapshots tied to the embedding
model** — restore on the same machine/model only:

1. `docker compose stop llamaindex`
2. Restore the snapshot into Qdrant (per collection, via Qdrant's snapshot
   API or dashboard).
3. `docker compose start llamaindex`, then check `GET /health` counts.

For anything cross-machine or cross-model, use the transfer bundle instead
(text-level, re-embedded on import — see the
[User Guide](USER_GUIDE.md#moving-to-another-machine-export--import)).

## Testing

```bash
docker compose run --rm --no-deps --entrypoint sh \
  -v "$PWD:/repo" llamaindex \
  -c "pip install -q 'pytest>=8,<9' && cd /repo/llamaindex-service && python -m pytest tests -q"
```

Covers: point-id idempotency, fact dedup/supersede, the recall min-score
filter (against an in-process Qdrant), LLM output parsing, transcript
truncation, and the hooks' payload extraction + cwd→project resolution.

Recall quality is guarded separately by an eval set —
`scripts/recall_eval.py` runs the real pipeline against a seeded throwaway
corpus and compares to `llamaindex-service/evals/recall_baseline.json`.

## Troubleshooting

- **Memory not being written** (`last_written_at` frozen): hooks not firing.
  Re-run `./setup.sh`, then restart the agent (Claude Code sessions pick up
  hooks on restart; Hermes Desktop needs the `serve` patch re-applied after
  every update — setup.sh does it).
- **After every Hermes Desktop update, re-run `./setup.sh`** — updates
  overwrite the `serve` patch (without it the Desktop backend never
  registers shell hooks).
- **After editing any file in `hooks/`: re-run `./setup.sh`** — hook consent
  is tied to the script's mtime.
- **Service won't boot, log says embedding mismatch**: `.env`'s
  `EMBED_MODEL` no longer matches what the collections were built with.
  Either restore the original value or do a real model switch (see provider
  configuration above).
- **`/ingest/file` or the watcher seems to do nothing**: duplicates are
  skipped by design — check `logs/ingest_watcher.log` (`0 sent / N
  unchanged` is healthy).
- **To delete memory, tell your agent ("forget about X") or use the API** —
  avoid deleting in the Qdrant dashboard: it bypasses every confirmation
  guard and makes it look like the system "lost data" on its own.
- **Port 8800 already taken**: another service owns it — change the host
  port mapping in `docker-compose.yml`.
