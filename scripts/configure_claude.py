#!/usr/bin/env python3
"""Wire Claude Code to the memory stack (the Hermes-independent adapter).

Run via setup.sh (or directly). Idempotent — safe to re-run. Steps:

1. settings.json — register the 4 memory hooks in ~/.claude/settings.json
                   (UserPromptSubmit: auto-inject recall, Stop: write turns,
                   SessionEnd: trigger consolidation, SessionStart: catch-up
                   sweep). Other hooks/settings are left untouched.
2. MCP           — register the hermes-memory MCP server user-scoped via
                   `claude mcp add` so the memory tools (recall, save_facts,
                   forget_about, consolidate_session, …) are available in
                   every project.

No API key is needed anywhere in this path: Claude Code runs on its own
login, and consolidation either uses the service-side LLM or the
consolidate_session MCP tool (the agent's own model does the distilling).

Exit code 0 = fully wired; 1 = something needs attention (printed).
Hooks are snapshotted at session start — restart Claude Code sessions to
pick up changes.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SETTINGS = Path.home() / ".claude" / "settings.json"
REPO = Path(__file__).resolve().parent.parent
MCP_NAME = "hermes-memory"
MCP_URL = "http://localhost:8800/mcp"

# Claude Code event -> (script under hooks/claude/, timeout seconds)
HOOKS = {
    "UserPromptSubmit": (REPO / "hooks" / "claude" / "user_prompt_submit.py", 5),
    "Stop": (REPO / "hooks" / "claude" / "stop.py", 15),
    "SessionEnd": (REPO / "hooks" / "claude" / "session_end.py", 10),
    "SessionStart": (REPO / "hooks" / "claude" / "session_start.py", 10),
}

ok_all = True


def note(msg: str) -> None:
    print(f"  {msg}")


def fail(msg: str) -> None:
    global ok_all
    ok_all = False
    print(f"  ✗ {msg}")


def hook_command(script: Path) -> str:
    return f'python3 "{script}"'


# ---------------------------------------------------------------------------
# 1. hooks in ~/.claude/settings.json
# ---------------------------------------------------------------------------
def patch_settings() -> None:
    print("==> ~/.claude/settings.json")
    if not SETTINGS.parent.exists():
        fail(f"{SETTINGS.parent} does not exist — install Claude Code first")
        return

    settings = {}
    if SETTINGS.exists():
        try:
            settings = json.loads(SETTINGS.read_text()) or {}
        except json.JSONDecodeError:
            fail(f"{SETTINGS} is not valid JSON — fix it manually first")
            return

    hooks_cfg = settings.setdefault("hooks", {})
    changed = False
    for event, (script, timeout) in HOOKS.items():
        command = hook_command(script)
        matchers = hooks_cfg.setdefault(event, [])
        already = any(
            h.get("command") == command
            for m in matchers if isinstance(m, dict)
            for h in (m.get("hooks") or []) if isinstance(h, dict)
        )
        if already:
            note(f"hook {event} already present")
            continue
        # Replace any stale entry pointing at an older copy of this script
        # (repo moved), keyed by the script filename.
        for m in matchers:
            if isinstance(m, dict):
                m["hooks"] = [
                    h for h in (m.get("hooks") or [])
                    if not (isinstance(h, dict) and script.name in str(h.get("command", "")))
                ]
        matchers[:] = [m for m in matchers if not isinstance(m, dict) or m.get("hooks")]
        matchers.append({"hooks": [{"type": "command", "command": command, "timeout": timeout}]})
        changed = True
        note(f"registered hook {event}")

    if changed:
        if SETTINGS.exists():
            stamp = time.strftime("%Y%m%d_%H%M%S")
            shutil.copyfile(SETTINGS, SETTINGS.with_name(f"settings.json.bak.{stamp}"))
        SETTINGS.write_text(json.dumps(settings, indent=2) + "\n")
        note("settings.json written (restart Claude Code sessions to pick hooks up)")


# ---------------------------------------------------------------------------
# 2. MCP server (user scope)
# ---------------------------------------------------------------------------
def register_mcp() -> None:
    print("==> MCP server")
    claude = shutil.which("claude")
    if not claude:
        fail("`claude` CLI not found on PATH — register manually: "
             f"claude mcp add --scope user --transport http {MCP_NAME} {MCP_URL}")
        return
    probe = subprocess.run(
        [claude, "mcp", "get", MCP_NAME], capture_output=True, text=True, timeout=30
    )
    if probe.returncode == 0:
        note(f"MCP {MCP_NAME} already registered")
        return
    add = subprocess.run(
        [claude, "mcp", "add", "--scope", "user", "--transport", "http", MCP_NAME, MCP_URL],
        capture_output=True, text=True, timeout=30,
    )
    if add.returncode == 0:
        note(f"registered MCP {MCP_NAME} -> {MCP_URL} (user scope)")
    else:
        fail(f"claude mcp add failed: {(add.stderr or add.stdout).strip()}")


def main() -> int:
    patch_settings()
    register_mcp()
    print("✓ Claude Code wired" if ok_all else "✗ finished with problems (see above)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
