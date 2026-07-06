#!/usr/bin/env python3
"""Hermes post_llm_call hook: mirror every completed turn into memory.

Register in ~/.hermes/config.yaml:
    hooks:
      post_llm_call:
        - command: python3 "/path/to/hermes-agent/hooks/post_llm_call.py"
          timeout: 10
    hooks_auto_accept: true   # REQUIRED — Desktop has no TTY to approve hooks

Hermes pipes a JSON payload on stdin after each turn. Payload shape has
varied between builds, so extraction is defensive: it looks for the user /
assistant texts under `extra`, at the top level, or in a `messages` list.
Every payload is also appended verbatim to logs/hook-debug.jsonl next to
this repo, so a format change is diagnosable instead of silently dropping
turns. Best-effort: failures never break a Hermes chat turn.
"""

import json
import os
import sqlite3
import sys
import urllib.request
from pathlib import Path

MEMORY_URL = os.environ.get("HERMES_MEMORY_URL", "http://localhost:8800") + "/memory/append"
DEBUG_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "hook-debug.jsonl")
PROJECTS_DB = Path.home() / ".hermes" / "projects.db"


def resolve_project(cwd: str) -> str:
    return resolve_project_with_source(cwd)[0]


def resolve_project_with_source(cwd: str) -> tuple:
    """Map a turn to a Hermes sidebar project. Returns (slug, source) where
    source records WHICH signal produced the slug — "folder" | "active" |
    "default". The service uses it: a folder-sourced tag is an intentional
    workspace (project_switch moved the session there) and may re-tag an
    existing session; ambient signals never do. Signals, in priority order:

    1. cwd folder match — longest folder-prefix wins (a session in /work/erp
       beats the broader /work project); equal-length conflicts go to the
       most recently created project instead of undefined SQL row order.
    2. the project currently SELECTED in the sidebar (project_meta.active_id)
       — so chat-only projects with no folder still tag correctly.
    3. "default".

    Reads Hermes' own projects.db read-only — the sidebar is the single
    source of truth, no separate mapping to maintain."""
    if not PROJECTS_DB.exists():
        return "default", "default"
    try:
        try:
            conn = sqlite3.connect(f"file:{PROJECTS_DB}?mode=ro", uri=True, timeout=1)
            conn.execute("SELECT 1").fetchone()
        except sqlite3.OperationalError:
            # DB is in WAL mode: a read-only open can fail because it needs
            # to write -shm. A normal connection is still safe — SELECT only,
            # and WAL allows concurrent reads.
            conn = sqlite3.connect(str(PROJECTS_DB), timeout=1)
        rows = conn.execute(
            "SELECT p.slug, f.path, p.created_at FROM project_folders f"
            " JOIN projects p ON p.id = f.project_id WHERE p.archived = 0"
            " UNION"
            " SELECT slug, primary_path, created_at FROM projects"
            " WHERE archived = 0 AND primary_path IS NOT NULL"
        ).fetchall()
        try:
            active_row = conn.execute(
                "SELECT p.slug FROM project_meta m JOIN projects p ON p.id = m.value"
                " WHERE m.key = 'active_id' AND p.archived = 0"
            ).fetchone()
        except sqlite3.Error:  # older Hermes without project_meta
            active_row = None
        conn.close()
    except Exception:
        return "default", "default"

    best, best_len, best_created = "", -1, -1
    if cwd:
        # realpath (not normpath): a cwd reached through a symlink must still
        # match the project folder it points to.
        cwd_norm = os.path.realpath(cwd)
        for slug, path, created in rows:
            if not path:
                continue
            p = os.path.realpath(path)
            if cwd_norm == p or cwd_norm.startswith(p + os.sep):
                if len(p) > best_len or (len(p) == best_len and (created or 0) > best_created):
                    if len(p) == best_len and best:
                        _debug_dump(json.dumps({
                            "warn": "project folder conflict",
                            "path": p, "losing": best, "winning": slug,
                        }))
                    best, best_len, best_created = slug, len(p), (created or 0)
    if best:
        return best, "folder"
    if active_row and active_row[0]:
        return active_row[0], "active"
    return "default", "default"


def _debug_dump(raw: str) -> None:
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG), exist_ok=True)
        with open(DEBUG_LOG, "a") as f:
            f.write(raw.strip().replace("\n", " ") + "\n")
    except Exception:
        pass


def _extract(payload: dict) -> tuple[str, str, str]:
    session_id = (
        payload.get("session_id")
        or payload.get("session")
        or payload.get("conversation_id")
        or "unknown"
    )

    sources = [payload.get("extra") or {}, payload]
    user_message = ""
    assistant_response = ""
    for src in sources:
        user_message = user_message or src.get("user_message") or src.get("prompt") or ""
        assistant_response = (
            assistant_response
            or src.get("assistant_response")
            or src.get("response")
            or src.get("completion")
            or ""
        )

    # Fallback: an OpenAI-style messages list
    if not (user_message and assistant_response):
        messages = payload.get("messages") or (payload.get("extra") or {}).get("messages") or []
        for m in reversed(messages):
            role, content = m.get("role"), m.get("content")
            if not isinstance(content, str):
                continue
            if role == "assistant" and not assistant_response:
                assistant_response = content
            elif role == "user" and not user_message:
                user_message = content
            if user_message and assistant_response:
                break

    return str(session_id), user_message, assistant_response


def main():
    raw = sys.stdin.read()
    _debug_dump(raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return

    session_id, user_message, assistant_response = _extract(payload)
    if not user_message and not assistant_response:
        return

    project_id, project_source = resolve_project_with_source(payload.get("cwd") or "")
    body = json.dumps(
        {
            "session_id": session_id,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "project_id": project_id,
            "project_source": project_source,
            "source_agent": "hermes",
        }
    ).encode()

    request = urllib.request.Request(
        MEMORY_URL, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        urllib.request.urlopen(request, timeout=5)
    except Exception:
        pass  # best-effort; don't let a down Docker stack break Hermes


if __name__ == "__main__":
    main()
    print("{}")
