#!/usr/bin/env python3
"""Codex Stop hook: append the completed turn to Longbrain."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle_common import (  # noqa: E402
    load_pending_prompt,
    post_json,
    read_payload,
    remove_pending_prompt,
    resolve_project,
    update_state,
)


def main() -> None:
    payload = read_payload()
    session_id = str(payload.get("session_id") or "")
    turn_id = str(payload.get("turn_id") or "")
    assistant = str(payload.get("last_assistant_message") or "").strip()
    if not session_id or not turn_id or not assistant:
        print(json.dumps({"continue": True}))
        return

    pending = load_pending_prompt(session_id, turn_id)
    prompt = str(pending.get("prompt") or "").strip()
    if not prompt:
        update_state(last_write_at=time.time(), last_write_ok=False,
                     last_write_error="pending prompt missing")
        print(json.dumps({"continue": True}))
        return

    cwd = payload.get("cwd") or pending.get("cwd") or ""
    project_id, project_source = resolve_project(cwd)
    result = post_json("/memory/append", {
        "session_id": session_id,
        "user_message": prompt,
        "assistant_response": assistant,
        "project_id": project_id,
        "project_source": project_source,
        "source_agent": "codex",
    })
    ok = result is not None
    update_state(
        last_write_at=time.time(),
        last_write_ok=ok,
        last_write_error="" if ok else "memory append failed",
        last_write_session_id=session_id,
    )
    if ok:
        remove_pending_prompt(session_id, turn_id)
    print(json.dumps({"continue": True}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"continue": True}))
