#!/usr/bin/env python3
"""Codex UserPromptSubmit hook: persist the prompt and auto-recall memory."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle_common import (  # noqa: E402
    env_get,
    env_int,
    post_json,
    read_payload,
    resolve_project,
    save_pending_prompt,
    update_state,
)

TIMEOUT = float(env_get("LONGBRAIN_MEMORY_RECALL_TIMEOUT", "3"))
MAX_CONTEXT_CHARS = env_int("LONGBRAIN_MEMORY_MAX_CONTEXT", 6000)
MIN_PROMPT_CHARS = env_int("LONGBRAIN_RECALL_MIN_PROMPT_CHARS", 15)


def main() -> None:
    payload = read_payload()
    prompt = str(payload.get("prompt") or "").strip()
    save_pending_prompt(payload, prompt)
    if len(prompt) < MIN_PROMPT_CHARS:
        update_state(last_prompt_at=time.time(), last_recall_skipped=True)
        return

    result = post_json("/memory/recall", {
        "query": prompt[:2000],
        "session_id": payload.get("session_id") or "",
        "project": resolve_project(payload.get("cwd") or "")[0],
        "recent_turns": 0,
    }, timeout=TIMEOUT)
    context = (result.get("context_block") or "").strip() if result else ""
    update_state(
        last_prompt_at=time.time(),
        last_recall_at=time.time(),
        last_recall_ok=result is not None,
        last_recall_context_chars=len(context),
        last_recall_skipped=False,
    )
    if not context:
        return
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": (
                "Long-term memory (auto-recalled):\n" + context[:MAX_CONTEXT_CHARS]
            ),
        }
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
