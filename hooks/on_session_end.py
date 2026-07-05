#!/usr/bin/env python3
"""Hermes on_session_end hook: trigger memory consolidation for the session.

Register in ~/.hermes/config.yaml under hooks.on_session_end (setup.sh does
this). When a session finishes cleanly, ask the memory service to distill its
turns into long-term facts. The request uses background mode so it returns
instantly — the LLM work happens inside the service. Best-effort: a down
stack never breaks Hermes; missed sessions are caught by the periodic sweep.
"""

import json
import os
import sys
import urllib.request

MEMORY_URL = os.environ.get("HERMES_MEMORY_URL", "http://localhost:8800") + "/memory/consolidate"
DEBUG_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "hook-debug.jsonl")


def main():
    raw = sys.stdin.read()
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG), exist_ok=True)
        with open(DEBUG_LOG, "a") as f:
            f.write(raw.strip().replace("\n", " ") + "\n")
    except Exception:
        pass

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return

    session_id = payload.get("session_id")
    extra = payload.get("extra") or {}
    if not session_id or extra.get("completed") is False:
        return  # interrupted sessions: leave for the sweep once they settle

    body = json.dumps({"session_id": session_id, "background": True}).encode()
    request = urllib.request.Request(
        MEMORY_URL, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        urllib.request.urlopen(request, timeout=5)
    except Exception:
        pass  # best-effort


if __name__ == "__main__":
    main()
    print("{}")
