#!/usr/bin/env python3
"""Claude Code PreToolUse hook: require a Longbrain recall before editing.

Registered in ~/.claude/settings.json by scripts/configure_claude.py, scoped
to Edit|Write|MultiEdit. UserPromptSubmit already auto-injects a recall
summary every turn (see user_prompt_submit.py) — this hook is the hard
enforcement layer for the deeper, explicit calls
(mcp__longbrain__search_history / mcp__longbrain__memory_recall) that
CLAUDE.md asks the model to make before editing code. If neither tool has
been called since the last real user prompt, the edit is denied with a
reason telling the model to call one first, then retry.

Fails open on any error reading/parsing the transcript (unreadable file,
unexpected format) — a broken gate must never make editing impossible.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_payload  # noqa: E402

GATED_TOOLS = {"Edit", "Write", "MultiEdit"}
RECALL_TOOLS = {"mcp__longbrain__search_history", "mcp__longbrain__memory_recall"}


def _is_tool_result(content) -> bool:
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return ""


def _tool_use_names(content) -> set:
    if not isinstance(content, list):
        return set()
    return {
        b.get("name") for b in content
        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name")
    }


def recalled_this_turn(transcript_path: str) -> bool:
    """True if a Longbrain recall tool was called since the last real user
    prompt in this transcript — or if the transcript can't be read (fail
    open: never block on an infra hiccup)."""
    try:
        entries = []
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("isSidechain") or entry.get("isMeta"):
                    continue
                entries.append(entry)
    except OSError:
        return True

    last_user_index = -1
    for i, entry in enumerate(entries):
        if entry.get("type") != "user":
            continue
        content = (entry.get("message") or {}).get("content")
        if _is_tool_result(content):
            continue
        if _text_of(content).strip():
            last_user_index = i

    if last_user_index == -1:
        return True

    for entry in entries[last_user_index:]:
        if entry.get("type") != "assistant":
            continue
        content = (entry.get("message") or {}).get("content")
        if _tool_use_names(content) & RECALL_TOOLS:
            return True
    return False


def main():
    payload = read_payload()
    if payload.get("tool_name") not in GATED_TOOLS:
        return

    transcript_path = payload.get("transcript_path") or ""
    if not transcript_path or recalled_this_turn(transcript_path):
        return

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "Trước khi sửa file, hãy gọi mcp__longbrain__search_history "
                "hoặc mcp__longbrain__memory_recall để kiểm tra quyết định/"
                "ràng buộc liên quan đã có trước đó, rồi thử lại thao tác này."
            ),
        }
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass  # fail open — never block editing on a bug in this hook
