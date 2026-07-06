#!/usr/bin/env python3
"""Claude Code SessionStart hook: catch-up consolidation sweep.

Registered in ~/.claude/settings.json by scripts/configure_claude.py.
Whenever a session starts, poke the memory service to consolidate any
sessions still pending (missed SessionEnd — crash, force-quit, rate limit).
The service debounces, so opening several sessions in a row costs at most
one sweep per debounce window. Best-effort.

Deliberately prints nothing: SessionStart stdout would be injected as
context, and memory injection is UserPromptSubmit's job (query-relevant,
bounded) — not a session-wide dump.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import post_json, read_payload  # noqa: E402


def main():
    read_payload()  # drain + debug-log; content not needed
    post_json("/memory/consolidate-pending", {})


if __name__ == "__main__":
    main()
