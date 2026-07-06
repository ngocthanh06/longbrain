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
    """Map the turn's cwd to a Hermes sidebar project (longest folder-prefix
    wins, so a session in /work/erp beats the broader /work project). Reads
    Hermes' own projects.db read-only — sidebar is the single source of truth,
    no separate mapping to maintain. Falls back to "default"."""
    if not cwd or not PROJECTS_DB.exists():
        return "default"
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
            "SELECT p.slug, f.path FROM project_folders f"
            " JOIN projects p ON p.id = f.project_id WHERE p.archived = 0"
            " UNION"
            " SELECT slug, primary_path FROM projects"
            " WHERE archived = 0 AND primary_path IS NOT NULL"
        ).fetchall()
        conn.close()
    except Exception:
        return "default"

    # realpath (not normpath): a cwd reached through a symlink must still
    # match the project folder it points to.
    cwd_norm = os.path.realpath(cwd)
    best, best_len = "default", -1
    for slug, path in rows:
        if not path:
            continue
        p = os.path.realpath(path)
        if (cwd_norm == p or cwd_norm.startswith(p + os.sep)) and len(p) > best_len:
            best, best_len = slug, len(p)
    return best


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

    body = json.dumps(
        {
            "session_id": session_id,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "project_id": resolve_project(payload.get("cwd") or ""),
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
