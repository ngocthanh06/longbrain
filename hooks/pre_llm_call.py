#!/usr/bin/env python3
"""Hermes pre_llm_call hook: auto-inject relevant long-term memory.

Register in ~/.hermes/config.yaml under hooks.pre_llm_call (setup.sh does
this). Before each LLM call, recall memory relevant to the user's message and
hand it to Hermes via the official contract: printing {"context": "..."} on
stdout injects it into the turn. The model no longer needs to remember to
call the memory_recall tool.

Latency budget: one embed + three vector searches (~100-300ms local). Any
failure or timeout degrades to no injection — never blocks the conversation.
"""

import json
import os
import sys
import urllib.request

# Reuse the sidebar-project resolver from the sibling hook.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from post_llm_call import debug_dump, env_get, env_int, resolve_project  # noqa: E402

MEMORY_URL = env_get("LONGBRAIN_MEMORY_URL", "http://localhost:8800") + "/memory/recall"
TIMEOUT = float(env_get("LONGBRAIN_MEMORY_RECALL_TIMEOUT", "3"))
# Same token gates as the Claude Code adapter (hooks/claude/user_prompt_submit.py):
# skip recall for prompts too short to carry meaning, cap the injected block.
MAX_CONTEXT_CHARS = env_int("LONGBRAIN_MEMORY_MAX_CONTEXT", 6000)
MIN_PROMPT_CHARS = env_int("LONGBRAIN_RECALL_MIN_PROMPT_CHARS", 15)


def _extract_query(payload: dict) -> str:
    extra = payload.get("extra") or {}
    for src in (extra, payload):
        q = src.get("user_message") or src.get("prompt") or src.get("message") or ""
        if q:
            return q
    messages = extra.get("messages") or payload.get("messages") or []
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def main():
    raw = sys.stdin.read()
    debug_dump(raw)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print("{}")
        return

    query = _extract_query(payload)
    if len(query.strip()) < MIN_PROMPT_CHARS:
        print("{}")
        return

    body = json.dumps(
        {
            "query": query[:2000],
            "session_id": payload.get("session_id") or "",
            "project": resolve_project(payload.get("cwd") or ""),
            "recent_turns": 0,  # Hermes already carries its own history
        }
    ).encode()
    request = urllib.request.Request(
        MEMORY_URL, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as resp:
            result = json.loads(resp.read())
    except Exception:
        print("{}")
        return

    context = (result.get("context_block") or "").strip()
    if context:
        print(json.dumps(
            {"context": "Long-term memory (auto-recalled):\n" + context[:MAX_CONTEXT_CHARS]},
            ensure_ascii=False,
        ))
    else:
        print("{}")


if __name__ == "__main__":
    main()
