#!/usr/bin/env python3
"""Shared plumbing for Codex lifecycle hooks."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
CLAUDE_HOOKS = REPO / "hooks" / "claude"
sys.path.insert(0, str(CLAUDE_HOOKS))

from common import (  # noqa: E402,F401
    MEMORY_BASE,
    env_get,
    env_int,
    get_json,
    post_json,
    read_payload,
    resolve_project,
)

CODEX_HOME = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
PENDING_DIR = CODEX_HOME / "longbrain_pending_turns"
STATE_FILE = CODEX_HOME / "longbrain_codex_hooks_state.json"


def _turn_key(session_id: str, turn_id: str) -> str:
    return hashlib.sha256(f"{session_id}\0{turn_id}".encode()).hexdigest()


def pending_path(session_id: str, turn_id: str) -> Path:
    return PENDING_DIR / f"{_turn_key(session_id, turn_id)}.json"


def save_pending_prompt(payload: dict, prompt: str) -> None:
    session_id = str(payload.get("session_id") or "")
    turn_id = str(payload.get("turn_id") or "")
    if not session_id or not turn_id or not prompt.strip():
        return
    try:
        path = pending_path(session_id, turn_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(json.dumps({
            "session_id": session_id,
            "turn_id": turn_id,
            "prompt": prompt,
            "cwd": payload.get("cwd") or "",
            "created_at": time.time(),
        }) + "\n")
        os.replace(temp, path)
    except OSError:
        pass


def load_pending_prompt(session_id: str, turn_id: str) -> dict:
    try:
        value = json.loads(pending_path(session_id, turn_id).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def remove_pending_prompt(session_id: str, turn_id: str) -> None:
    try:
        pending_path(session_id, turn_id).unlink()
    except OSError:
        pass


def update_state(**values) -> None:
    """Best-effort diagnostics for doctor; hooks must never fail Codex."""
    try:
        state = json.loads(STATE_FILE.read_text())
        if not isinstance(state, dict):
            state = {}
    except (OSError, json.JSONDecodeError):
        state = {}
    state.update(values)
    state["updated_at"] = time.time()
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp = STATE_FILE.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(temp, STATE_FILE)
    except OSError:
        pass
