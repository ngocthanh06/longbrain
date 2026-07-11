#!/usr/bin/env python3
"""Codex SessionStart hook: catch up consolidation and report outages."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle_common import (  # noqa: E402
    MEMORY_BASE,
    REPO,
    get_json,
    post_json,
    read_payload,
    update_state,
)


def main() -> None:
    read_payload()
    if post_json("/memory/consolidate-pending", {}) is not None:
        update_state(last_session_start_at=time.time(), last_health_ok=True)
        return
    if get_json("/health") is not None:
        update_state(last_session_start_at=time.time(), last_health_ok=True)
        return
    update_state(last_session_start_at=time.time(), last_health_ok=False)
    print(json.dumps({
        "systemMessage": (
            f"Longbrain is unreachable at {MEMORY_BASE}. Long-term memory is "
            "offline for this session. Tell the user in the first reply. "
            f"Restore it with: cd '{REPO}' && docker compose up -d"
        ),
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "Longbrain is offline; recall and recording are unavailable.",
        },
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
