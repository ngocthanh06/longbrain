#!/usr/bin/env python3
"""Google Drive connector: host-side sync loop.

Sync `google_drive_sources.json` (a docs/link a user has explicitly
registered) into Longbrain's L4 knowledge base via the same `/ingest/file`
file bridge every other source uses (see README.md's architecture note).
Meant to run periodically (cron/launchd), one pass per invocation, like
scripts/ingest_watcher.py.

Change detection: Drive's own `modifiedTime` per file, compared against
the last-synced value cached in `google_drive_sync_state.json` — the same
"compare a cheap freshness signal, skip if unchanged" shape
ingest_watcher.py uses (mtime+size there, modifiedTime here), just backed
by the Drive API's metadata call instead of a local stat().

Usage:
  python sync.py                          # sync every registered source
  python sync.py add <file_id_or_url> <project_id>   # register one source
  python sync.py list                     # show registered sources
"""

import json
import mimetypes
import os
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auth import get_credentials  # noqa: E402
from fetch import fetch_file, get_metadata  # noqa: E402
from normalize import normalize  # noqa: E402

import urllib.request  # noqa: E402

LONGBRAIN_HOME = Path.home() / ".longbrain"
SOURCES_FILE = LONGBRAIN_HOME / "google_drive_sources.json"
STATE_FILE = LONGBRAIN_HOME / "google_drive_sync_state.json"
MEMORY_URL = os.environ.get(
    "LONGBRAIN_MEMORY_URL", os.environ.get("HERMES_MEMORY_URL", "http://localhost:8800")
)
REQUEST_TIMEOUT = 30.0

# Accepts a bare file ID or a full Docs/Drive URL
# (https://docs.google.com/document/d/<id>/edit, .../file/d/<id>/view, ...).
_URL_ID_RE = re.compile(r"/d/([a-zA-Z0-9_-]{10,})")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def extract_file_id(file_id_or_url: str) -> str:
    match = _URL_ID_RE.search(file_id_or_url)
    return match.group(1) if match else file_id_or_url


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_sources() -> list[dict]:
    return load_json(SOURCES_FILE, [])


def add_source(file_id_or_url: str, project_id: str) -> None:
    file_id = extract_file_id(file_id_or_url)
    sources = load_sources()
    if any(s["file_id"] == file_id for s in sources):
        log(f"already registered: {file_id}")
        return
    sources.append({"file_id": file_id, "project_id": project_id})
    save_json(SOURCES_FILE, sources)
    log(f"registered {file_id} -> project '{project_id}'")


def post_ingest_file(tmp_path: Path, project_id: str, metadata: dict) -> bool:
    """POST one file to /ingest/file as multipart/form-data — same wire
    format as scripts/ingest_watcher.py's helper of the same shape; kept as
    its own small copy here rather than a shared import (see docs/
    STRATEGY.md: no shared abstraction before a second, divergent connector
    shows what's actually common)."""
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(tmp_path.name)[0] or "text/plain"
    data = tmp_path.read_bytes()

    def field(name: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
        ).encode()

    body = bytearray()
    body += field("project_id", project_id)
    body += field("metadata", json.dumps(metadata))
    body += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f'filename="{tmp_path.name}"\r\nContent-Type: {content_type}\r\n\r\n'
    ).encode()
    body += data
    body += f"\r\n--{boundary}--\r\n".encode()

    request = urllib.request.Request(
        f"{MEMORY_URL}/ingest/file",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        log(f"ERROR: ingest failed for {tmp_path.name}: {exc}")
        return False


def service_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{MEMORY_URL}/health", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def sync_one(creds, source: dict, state: dict, force: bool) -> str:
    """Returns 'sent' | 'skipped' | 'failed'."""
    file_id, project_id = source["file_id"], source["project_id"]
    try:
        meta = get_metadata(creds, file_id)
    except Exception as exc:
        log(f"ERROR: could not read metadata for {file_id}: {exc}")
        return "failed"

    if not force and state.get(file_id) == meta["modifiedTime"]:
        return "skipped"

    try:
        fetched = fetch_file(creds, file_id)
    except ValueError as exc:  # unsupported shape (e.g. a Sheet) — not a failure to retry
        log(f"SKIP: {exc}")
        return "skipped"
    except Exception as exc:
        log(f"ERROR: could not fetch {file_id}: {exc}")
        return "failed"

    tmp_path, metadata = normalize(fetched)
    try:
        ok = post_ingest_file(tmp_path, project_id, metadata)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not ok:
        return "failed"
    state[file_id] = fetched.modified_time
    log(f"ingested {fetched.name!r} ({file_id}) -> project '{project_id}'")
    return "sent"


def main() -> int:
    args = sys.argv[1:]

    if args[:1] == ["add"]:
        if len(args) != 3:
            print(__doc__)
            return 1
        add_source(args[1], args[2])
        return 0

    if args[:1] == ["list"]:
        for s in load_sources():
            print(f"{s['file_id']} -> project '{s['project_id']}'")
        return 0

    if not service_reachable():
        log("SKIP: memory service not reachable")
        return 0

    sources = load_sources()
    if not sources:
        log("no sources registered — see: python sync.py add <file_id_or_url> <project_id>")
        return 0

    try:
        creds = get_credentials()
    except FileNotFoundError as exc:
        log(f"ERROR: {exc}")
        return 1
    state = load_json(STATE_FILE, {})
    force = "--force" in args
    counts = {"sent": 0, "skipped": 0, "failed": 0}
    for source in sources:
        counts[sync_one(creds, source, state, force)] += 1
    save_json(STATE_FILE, state)
    log(f"pass complete: {counts['sent']} sent, {counts['skipped']} skipped, "
        f"{counts['failed']} failed")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
