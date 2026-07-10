#!/usr/bin/env python3
"""Minimal single-file adapter skeleton — copy this to wire a new AI agent
into the shared memory stack. Stdlib only, best-effort throughout: a down
memory service must never break the agent's own conversation.

Wire each subcommand into the matching lifecycle moment of your agent:

    adapter.py before-prompt   <session_id> <cwd>   # stdin: the user prompt
                                                    # stdout: context to inject
    adapter.py after-response  <session_id> <cwd>   # stdin: JSON {"user": ..., "assistant": ...}
    adapter.py session-end     <session_id>
    adapter.py session-start

See adapters/README.md for the full contract and hooks/claude/ for a
production-grade reference implementation.
"""

import json
import os
import re
import subprocess
import sys
import urllib.request

MEMORY_URL = os.environ.get("LONGBRAIN_MEMORY_URL", "http://localhost:8800")
AGENT_NAME = "python-minimal"  # <- your agent's name; stamps every record
MAX_CONTEXT_CHARS = int(os.environ.get("LONGBRAIN_MEMORY_MAX_CONTEXT", "6000"))
MIN_PROMPT_CHARS = int(os.environ.get("LONGBRAIN_RECALL_MIN_PROMPT_CHARS", "15"))


def post(path: str, body: dict, timeout: float = 5.0):
    req = urllib.request.Request(
        MEMORY_URL + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None  # best-effort — never break the agent


def resolve_project(cwd: str) -> str:
    """Simplified resolver: git-root folder name, slugified. A production
    adapter should first consult ~/.hermes/projects.db and record discovered
    folders — see hooks/claude/common.py:resolve_project."""
    try:
        out = subprocess.run(
            ["git", "-C", cwd or ".", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=2,
        )
        root = out.stdout.strip() if out.returncode == 0 else (cwd or "")
    except Exception:
        root = cwd or ""
    name = os.path.basename(os.path.realpath(root)) if root else ""
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-_")[:64]
    return slug or "default"


def before_prompt(session_id: str, cwd: str) -> None:
    prompt = sys.stdin.read().strip()
    if len(prompt) < MIN_PROMPT_CHARS:
        return  # "ok" / "continue" — nothing to search for
    result = post("/memory/recall", {
        "query": prompt[:2000],
        "session_id": session_id,
        "project": resolve_project(cwd),
        "recent_turns": 0,  # the agent already carries its own history
    }, timeout=3.0)
    block = ((result or {}).get("context_block") or "").strip()
    if block:
        print("Long-term memory (auto-recalled):\n" + block[:MAX_CONTEXT_CHARS])


def after_response(session_id: str, cwd: str) -> None:
    try:
        turn = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        return
    post("/memory/append", {
        "session_id": session_id,
        "user_message": (turn.get("user") or "").strip(),
        "assistant_response": (turn.get("assistant") or "").strip(),
        "project_id": resolve_project(cwd),
        "project_source": "folder",  # cwd is a genuine workspace signal
        "source_agent": AGENT_NAME,
    })


def session_end(session_id: str) -> None:
    post("/memory/consolidate", {"session_id": session_id, "background": True})


def session_start() -> None:
    post("/memory/consolidate-pending", {})


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    arg = lambda i: sys.argv[i] if len(sys.argv) > i else ""  # noqa: E731
    if cmd == "before-prompt":
        before_prompt(arg(2), arg(3))
    elif cmd == "after-response":
        after_response(arg(2), arg(3))
    elif cmd == "session-end":
        session_end(arg(2))
    elif cmd == "session-start":
        session_start()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
