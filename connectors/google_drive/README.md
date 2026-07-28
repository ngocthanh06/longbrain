# Google Drive connector

Longbrain's first connector (see [docs/STRATEGY.md](../../docs/STRATEGY.md)
Phase 2). Pulls a Google Doc (or a plain `.md`/`.txt` file already sitting in
Drive) into the L4 knowledge base, over the exact same `/ingest/file` path
every other source uses — no new Document Engine, no generic connector
framework.

```
Google Drive  --(auth.py + fetch.py)-->  fetch content + metadata
                                          |
                                normalize.py: write a temp file,
                                document_key = Drive file ID (not the title —
                                a rename must not fork a new document)
                                          |
                                sync.py: POST /ingest/file  ------>  documents.py
                                (same file bridge as ingest_watcher.py)         |
                                                                    chunk, embed,
                                                                    supersede old
                                                                    versions, index
```

Runs **host-side**, like `scripts/ingest_watcher.py` and the lifecycle
hooks — never inside the `llamaindex-service` container: the container
never needs filesystem access beyond `/data`, never holds an OAuth token,
and a connector bug can't take down the memory engine.

Scope for v1: Google Docs and plain text/markdown files. **Google Sheets is
explicitly out of scope** — tabular data needs its own normalization shape
(`fetch_file` raises a clear `ValueError` if you point it at one instead of
silently mangling rows into prose).

## Setup

### 1. Install dependencies (host Python, not the container)

```bash
python3 -m venv .venv          # anywhere convenient; a repo-root venv works
source .venv/bin/activate
pip install -r connectors/google_drive/requirements.txt
```

### 2. Create a Google Cloud OAuth client (one-time, in your own account)

This step cannot be automated — it requires your own Google Cloud project:

1. Go to <https://console.cloud.google.com/> and create (or pick) a project.
2. **APIs & Services → Library** → enable the **Google Drive API**.
3. **APIs & Services → OAuth consent screen** → configure it (External is
   fine for personal use; add yourself as a test user if it stays in
   Testing mode).
4. **APIs & Services → Credentials → Create Credentials → OAuth client ID**
   → Application type **Desktop app**.
5. Download the resulting JSON and save it at exactly:
   ```
   ~/.longbrain/google_oauth_client.json
   ```

The read-only scope (`drive.readonly`) means this connector can never write
to or delete anything in your Drive.

### 3. Register a document

```bash
python connectors/google_drive/sync.py add <google-doc-url-or-file-id> <project-slug>
# e.g.
python connectors/google_drive/sync.py add \
  https://docs.google.com/document/d/1AbC.../edit erp
```

This just appends to `~/.longbrain/google_drive_sources.json` — edit that
file directly if you prefer.

### 4. Sync

```bash
python connectors/google_drive/sync.py
```

First run opens a browser for the Google OAuth consent screen (one-time;
the token is cached at `~/.longbrain/google_drive_token.json`, mode `0600`,
and silently refreshed afterward). Only changed documents (Drive's own
`modifiedTime` vs. the last-synced value in
`~/.longbrain/google_drive_sync_state.json`) are re-fetched and re-ingested
— add `--force` to resync everything regardless.

```bash
python connectors/google_drive/sync.py list   # see what's registered
```

Not built yet: periodic automation (a launchd/cron job calling `sync.py`
on an interval, the way `com.longbrain.memory-ingest.plist` already does
for `ingest_watcher.py`). Run it manually, or wire your own scheduler, until
that's added.

## Design notes

- **`document_key` is the Drive file ID, never the title.** A Drive file
  can be renamed without its ID changing; keying on the title would
  silently fork a new `document_key` group on every rename instead of
  updating the existing document — the exact class of bug the
  `document_key` hardening (relative path, not basename) fixed for the
  folder connector.
- **Export is deterministic.** Google Docs are exported as `text/plain` —
  content only, no per-request timestamp or formatting noise — so
  re-syncing an *unchanged* document re-produces byte-identical content,
  which keeps `documents.store_original()`'s content-addressed dedup
  working instead of manufacturing a fake "new version" on every poll.
- **No shared abstraction yet.** `sync.py` duplicates the small
  multipart-POST helper `ingest_watcher.py` already has, rather than
  importing it — per `docs/STRATEGY.md`'s own guardrail: no
  `BaseConnector`/SDK/registry until a second, deliberately different
  connector (e.g. a webhook-driven source) shows what's actually common.

## Tests

Kept out of `llamaindex-service/tests` on purpose: this connector's
dependencies are heavier than the rest of the stdlib-only host scripts, and
CI's default job must not be forced to install them just to collect tests.

```bash
python3 -m venv /tmp/gdrive_venv
/tmp/gdrive_venv/bin/pip install -r connectors/google_drive/requirements.txt pytest
/tmp/gdrive_venv/bin/python -m pytest connectors/google_drive/tests -q
```

All of `auth.py`/`fetch.py`/`normalize.py`/`sync.py`'s logic is covered with
mocked Google API responses — no real credentials or network access
needed to run the suite. The one thing that is **not** verified by these
tests (and can't be, without your own Google Cloud OAuth client): an actual
live sync against real Drive content. Run `sync.py` yourself once your
credentials are in place to confirm that end-to-end.
