#!/usr/bin/env python3
"""Claude Code UserPromptSubmit hook: auto-inject relevant long-term memory.

Registered in ~/.claude/settings.json by scripts/configure_claude.py. Before
Claude processes the prompt, recall memory relevant to it; plain stdout on
exit 0 is added to the turn's context. The model never needs to remember to
call the recall tool — same contract as the Hermes pre_llm_call hook.

Latency budget: one embed + three vector searches (~100-300ms local), 3s
timeout. Any failure degrades to no injection — never blocks the prompt.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    cap_context,
    MIN_PROMPT_CHARS,
    context_prefix,
    env_get,
    post_json,
    read_payload,
    resolve_project,
)

TIMEOUT = float(env_get("LONGBRAIN_MEMORY_RECALL_TIMEOUT", "3"))


def main():
    payload = read_payload()
    query = (payload.get("user_input") or payload.get("prompt") or "").strip()
    if len(query) < MIN_PROMPT_CHARS:
        return

    result = post_json("/memory/recall", {
        "query": query[:2000],
        "session_id": payload.get("session_id") or "",
        "project": resolve_project(payload.get("cwd") or "")[0],
        "recent_turns": 0,  # Claude Code carries its own session history
    }, timeout=TIMEOUT)
    if not result:
        return

    context = (result.get("context_block") or "").strip()
    if context:
        # Keep injection bounded: it costs the user's subscription tokens
        # on every turn.
        print(context_prefix(query) + "\n" + cap_context(context))


if __name__ == "__main__":
    main()
