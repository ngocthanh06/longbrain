#!/usr/bin/env python3
"""Auto-ingest each project's docs/ folder into the L4 knowledge base.

Meant to run periodically (launchd, StartInterval=60 — see
com.longbrain.memory-ingest.plist.template), not as a long-lived daemon: each
invocation does one poll pass and exits. No `watchdog`/inotify dependency —
"poll" here means comparing (mtime, size) against the last run, not
filesystem events.

Project folders come from two sources, merged: Hermes' own
~/.hermes/projects.db (read-only) — the same source of truth the memory
hooks use — and ~/.hermes/discovered_projects.json, a catalog any hook
(currently the Claude Code adapter) fills in when it resolves a project
from a real folder Hermes doesn't know about (see hooks/project_catalog.py).
The second source is what makes this work for a Claude-Code-only install
with no Hermes Desktop at all — Hermes source wins on a slug collision. A
project is watched only if it has a docs/ subfolder — opt-in, so a code
repo isn't accidentally ingested wholesale.

Dedup is two-layered: this script skips files whose (mtime, size) match the
last successful send (state file below), and the service itself skips a
file whose content-addressed stored_path was already indexed (documents.
already_ingested) — a safety net for a lost/corrupted state file.

Best-effort throughout: a memory-service hiccup here must never be loud.
"""

import json
import mimetypes
import os
import sqlite3
import sys
import time
import urllib.request
import uuid
from pathlib import Path

MEMORY_URL = os.environ.get(
    "LONGBRAIN_MEMORY_URL", os.environ.get("HERMES_MEMORY_URL", "http://localhost:8800")
)
PROJECTS_DB = Path.home() / ".hermes" / "projects.db"
DISCOVERED_PROJECTS_FILE = Path.home() / ".hermes" / "discovered_projects.json"
STATE_FILE = Path.home() / ".hermes" / "ingest_watcher_state.json"
SUPPORTED_EXTS = {".pdf", ".md", ".txt", ".docx"}
REQUEST_TIMEOUT = 30.0


def log(msg: str) -> None:
    # print() only — the launchd job (com.longbrain.memory-ingest.plist)
    # already redirects both stdout and stderr to logs/ingest_watcher.log,
    # so writing to that file here too would duplicate every line.
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def list_hermes_project_folders() -> list[tuple[str, str]]:
    """(slug, folder_path) for every non-archived project that has one —
    same tables post_llm_call.py's resolver reads, but every row instead of
    a single cwd match. Empty (not an error) when Hermes isn't installed."""
    if not PROJECTS_DB.exists():
        return []
    try:
        try:
            conn = sqlite3.connect(f"file:{PROJECTS_DB}?mode=ro", uri=True, timeout=1)
            conn.execute("SELECT 1").fetchone()
        except sqlite3.OperationalError:
            conn = sqlite3.connect(str(PROJECTS_DB), timeout=1)
        rows = conn.execute(
            "SELECT p.slug, f.path FROM project_folders f"
            " JOIN projects p ON p.id = f.project_id WHERE p.archived = 0"
            " UNION"
            " SELECT slug, primary_path FROM projects"
            " WHERE archived = 0 AND primary_path IS NOT NULL"
        ).fetchall()
        conn.close()
        return [(slug, path) for slug, path in rows if slug and path]
    except Exception as exc:
        log(f"WARN: could not read {PROJECTS_DB}: {exc}")
        return []


def list_discovered_project_folders() -> list[tuple[str, str]]:
    """(slug, folder_path) from hooks/project_catalog.py's catalog — the
    fallback source for a Claude-Code-only (no Hermes Desktop) install."""
    try:
        catalog = json.loads(DISCOVERED_PROJECTS_FILE.read_text())
        return [(slug, e["path"]) for slug, e in catalog.items() if e.get("path")]
    except (OSError, json.JSONDecodeError):
        return []


def list_project_folders() -> list[tuple[str, str]]:
    """Merge both sources, Hermes' projects.db winning on a slug collision
    (it's the authoritative sidebar; the discovered catalog only fills gaps)."""
    merged: dict[str, str] = {}
    for slug, path in list_discovered_project_folders():
        merged[slug] = path
    for slug, path in list_hermes_project_folders():
        merged[slug] = path
    return list(merged.items())


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except OSError as exc:
        log(f"WARN: could not write {STATE_FILE}: {exc}")


def service_reachable() -> bool:
    try:
        with urllib.request.urlopen(f"{MEMORY_URL}/health", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def ingest_file(path: Path, project_id: str) -> bool:
    """POST one file to /ingest/file as multipart/form-data (stdlib only —
    no `requests` dependency on the host). Returns True on a 2xx response."""
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = path.read_bytes()

    def field(name: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
        ).encode()

    body = bytearray()
    body += field("project_id", project_id)
    body += field("metadata", json.dumps({"source": path.name}))
    body += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f'filename="{path.name}"\r\nContent-Type: {content_type}\r\n\r\n'
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
        log(f"ERROR: ingest failed for {path}: {exc}")
        return False


def main() -> int:
    if not service_reachable():
        log("SKIP: memory service not reachable")
        return 0

    projects = list_project_folders()
    if not projects:
        log("no projects with a folder found — nothing to watch")
        return 0

    state = load_state()
    sent, skipped, failed = 0, 0, 0

    for slug, folder in projects:
        docs_dir = Path(folder).expanduser() / "docs"
        if not docs_dir.is_dir():
            continue
        for path in docs_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
                continue
            key = str(path.resolve())
            try:
                st = path.stat()
            except OSError:
                continue
            fingerprint = {"mtime": st.st_mtime, "size": st.st_size}
            if state.get(key) == fingerprint:
                skipped += 1
                continue
            if ingest_file(path, slug):
                state[key] = fingerprint
                sent += 1
                log(f"ingested {path} -> project '{slug}'")
            else:
                failed += 1

    save_state(state)
    log(f"pass complete: {sent} sent, {skipped} unchanged, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
