#!/usr/bin/env python3
"""Codex turn-ended notify adapter: mirror completed turns into Longbrain.

Codex does not expose the same lifecycle hook payload as Claude Code, but it
does write rollout JSONL files under $CODEX_HOME/sessions and supports a
top-level `notify` command. This script is installed as that notify command by
scripts/configure_codex.py. On every turn-ended notification it scans recent
rollouts, extracts real user prompts plus final assistant answers, and POSTs
new pairs to /memory/append.

Best-effort throughout: a down memory stack or a changed rollout format must
never break Codex itself. Unknown JSONL entries are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
HOOKS = REPO / "hooks"
sys.path.insert(0, str(HOOKS))
sys.path.insert(0, str(HOOKS / "claude"))

from common import resolve_project  # noqa: E402

MEMORY_BASE = os.environ.get(
    "LONGBRAIN_MEMORY_URL", os.environ.get("HERMES_MEMORY_URL", "http://localhost:8800")
)
CODEX_HOME = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
STATE_FILE = CODEX_HOME / "longbrain_codex_notify_state.json"
SOURCE_AGENT = "codex"


def _text_of(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"input_text", "output_text", "text"}:
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _is_real_user_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    context_prefixes = (
        "<environment_context>",
        "<recommended_plugins>",
        "<permissions instructions>",
        "<app-context>",
        "<collaboration_mode>",
        "<skills_instructions>",
        "<plugins_instructions>",
    )
    return not stripped.startswith(context_prefixes)


def _turn_id(payload: dict) -> str:
    meta = payload.get("internal_chat_message_metadata_passthrough") or {}
    return str(meta.get("turn_id") or "")


def _latest_rollouts(limit: int = 10) -> list[Path]:
    sessions = CODEX_HOME / "sessions"
    if not sessions.exists():
        return []
    files = []
    for path in sessions.rglob("rollout-*.jsonl"):
        try:
            if path.is_file():
                files.append((path.stat().st_mtime, path))
        except OSError:
            continue
    files.sort(key=lambda item: item[0], reverse=True)
    return [path for _mtime, path in files[:limit]]


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"processed": []}


def _save_state(state: dict) -> None:
    processed = list(dict.fromkeys(state.get("processed") or []))[-5000:]
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = {**state, "processed": processed}
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def _fingerprint(session_id: str, turn_id: str, user: str, assistant: str) -> str:
    digest = hashlib.sha256(f"{session_id}\0{turn_id}\0{user}\0{assistant}".encode()).hexdigest()
    return digest


def extract_completed_turns(path: str | Path) -> list[dict]:
    """Extract completed user/final-assistant turns from a Codex rollout."""
    path = Path(path)
    session_id = path.stem
    cwd = ""
    users: dict[str, str] = {}
    completed: list[dict] = []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = entry.get("payload") or {}
        if entry.get("type") == "session_meta":
            if payload.get("thread_source") == "subagent":
                # Guardian/subagent rollouts (Codex's internal safety
                # risk-assessment turns — synthetic "approval assessment"
                # prompts answered with a risk_level/outcome JSON blob) are
                # not real user conversation. The whole file is one such
                # thread, so skip it entirely rather than recording noise.
                return []
            session_id = payload.get("session_id") or payload.get("id") or session_id
            cwd = payload.get("cwd") or cwd
            continue
        if entry.get("type") != "response_item":
            continue
        if payload.get("type") != "message":
            continue

        role = payload.get("role")
        turn_id = _turn_id(payload)
        if not turn_id:
            continue
        text = _text_of(payload.get("content"))
        if role == "user":
            if _is_real_user_text(text):
                users[turn_id] = text
        elif role == "assistant" and payload.get("phase") == "final_answer":
            user_text = users.get(turn_id, "")
            if user_text and text.strip():
                completed.append({
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "cwd": cwd,
                    "user_message": user_text,
                    "assistant_response": text,
                })
    return completed


def post_json(path: str, body: dict, timeout: float = 5.0) -> bool:
    request = urllib.request.Request(
        MEMORY_BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except Exception:
        return False


def sync_recent_rollouts() -> int:
    state = _load_state()
    processed_order = list(dict.fromkeys(state.get("processed") or []))
    processed = set(processed_order)
    wrote = 0
    rollouts = _latest_rollouts()
    extracted = 0
    for rollout in reversed(rollouts):
        turns = extract_completed_turns(rollout)
        extracted += len(turns)
        for turn in turns:
            fp = _fingerprint(
                turn["session_id"], turn["turn_id"],
                turn["user_message"], turn["assistant_response"],
            )
            if fp in processed:
                continue
            project_id, project_source = resolve_project(turn.get("cwd") or "")
            ok = post_json("/memory/append", {
                "session_id": turn["session_id"],
                "user_message": turn["user_message"],
                "assistant_response": turn["assistant_response"],
                "project_id": project_id,
                "project_source": project_source,
                "source_agent": SOURCE_AGENT,
            })
            if ok:
                processed.add(fp)
                processed_order.append(fp)
                wrote += 1
    state.update({
        "processed": processed_order,
        "last_scan_at": time.time(),
        "last_scan_rollouts": len(rollouts),
        "last_scan_extracted": extracted,
    })
    if wrote:
        state["last_successful_write_at"] = time.time()
    _save_state(state)
    return wrote


def _run_chain(chain_json: str, forwarded_args: list[str]) -> None:
    if not chain_json:
        return
    try:
        command = json.loads(chain_json)
    except json.JSONDecodeError:
        return
    if not isinstance(command, list) or not command:
        return
    try:
        subprocess.Popen(command + forwarded_args)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--chain-json", default="")
    args, rest = parser.parse_known_args()
    try:
        sync_recent_rollouts()
    except Exception:
        pass
    finally:
        _run_chain(args.chain_json, rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
