#!/usr/bin/env python3
"""Claude Code SessionEnd hook: trigger memory consolidation for the session.

Registered in ~/.claude/settings.json by scripts/configure_claude.py. When a
session terminates, ask the memory service to distill its turns into
long-term facts (background mode — returns instantly, the LLM work happens
inside the service). Best-effort: missed sessions are caught by the
periodic sweep, exactly like the Hermes on_session_end path.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import post_json, read_payload  # noqa: E402


def main():
    payload = read_payload()
    session_id = payload.get("session_id") or ""
    if not session_id:
        return
    post_json("/memory/consolidate", {"session_id": session_id, "background": True})


if __name__ == "__main__":
    main()
