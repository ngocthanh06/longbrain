#!/usr/bin/env python3
"""Wire Claude Code to the memory stack (the Hermes-independent adapter).

Run via setup.sh (or directly). Idempotent — safe to re-run. Steps:

1. settings.json — register the 4 memory hooks in ~/.claude/settings.json
                   (UserPromptSubmit: auto-inject recall, Stop: write turns,
                   SessionEnd: trigger consolidation, SessionStart: catch-up
                   sweep), and default the token-hungry Workflows feature +
                   "ultracode" keyword trigger to off (SETTINGS_DEFAULTS —
                   never overriding a key the user already set). Other
                   hooks/settings are left untouched.
2. MCP           — register the longbrain MCP server user-scoped via
                   `claude mcp add` so the memory tools (recall, save_facts,
                   forget_about, consolidate_session, …) are available in
                   every project. A registration under the legacy name
                   `hermes-memory` is removed so tools don't show up twice.
3. CLAUDE.md     — opt-in only (LONGBRAIN_CONFIGURE_CLAUDE_MD=1, set by
                   setup.sh after asking the user): append/update a marked
                   block in ~/.claude/CLAUDE.md telling Claude to (a) search
                   the shared memory (search_history / memory_recall) before
                   declaring past-session context lost, and (b) prefer the
                   mcp__longbrain__* tools over its own built-in
                   file-based auto-memory when both are available.

No API key is needed anywhere in this path: Claude Code runs on its own
login, and consolidation either uses the service-side LLM or the
consolidate_session MCP tool (the agent's own model does the distilling).

Exit code 0 = fully wired; 1 = something needs attention (printed).
Hooks are snapshotted at session start — restart Claude Code sessions to
pick up changes.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SETTINGS = Path.home() / ".claude" / "settings.json"
CLAUDE_MD = Path.home() / ".claude" / "CLAUDE.md"
REPO = Path(__file__).resolve().parent.parent
MCP_NAME = "longbrain"
MCP_NAME_LEGACY = "hermes-memory"  # pre-rename installs; deregistered on sight
MCP_URL = "http://localhost:8800/mcp"

MARKER_START = "<!-- longbrain:memory-priority:start (auto-managed by longbrain setup, do not edit inside) -->"
MARKER_END = "<!-- longbrain:memory-priority:end -->"
# Markers written by pre-rename installs — found blocks get replaced in place.
MARKER_START_LEGACY = "<!-- hermes-agent:memory-priority:start (auto-managed by hermes-agent setup, do not edit inside) -->"
MARKER_END_LEGACY = "<!-- hermes-agent:memory-priority:end -->"
CLAUDE_MD_BLOCK = f"""{MARKER_START}
## Long-term memory (Longbrain) — shared across agents

**Recall (read):** When the user refers to content from a previous
conversation or session that is not in your current context (e.g. "the
review from earlier", "as we discussed last time", "give me that again"),
do NOT declare the context lost or redo the work from scratch. FIRST call
`mcp__longbrain__search_history` (and `mcp__longbrain__memory_recall`
for distilled facts) to retrieve it. The memory service stores past turns
from ALL connected agents (Claude Code, Hermes Desktop, …), so the answer
may exist even when this session has no trace of it. Likewise, when the
user asks about a project document or spec (anything that could live in a
`docs/` folder), call `mcp__longbrain__search_knowledge_base` before
saying the document is unknown — project `docs/` folders are auto-ingested
into the shared knowledge base. Only fall back to reconstructing from
code/files when the memory search returns nothing relevant — and say that
the search came up empty.

**Save (write):** When you decide on your own to save long-term memory (a
fact, decision, or preference worth keeping), prefer calling the
`mcp__longbrain__*` tools (e.g. `save_memories`,
`add_to_knowledge_base`) if they appear in this session's tool list — that
memory is shared across agents via a local service at
http://localhost:8800.

Only fall back to your own built-in file-based auto-memory
(`~/.claude/projects/.../memory/*.md`) when no `mcp__longbrain__*`
tools are available in this session (the MCP server is not registered or
not reachable).
{MARKER_END}"""

# Claude Code feature defaults merged into ~/.claude/settings.json — only
# when the user has not set the key themselves (their explicit choice wins).
# Workflows / the "ultracode" keyword fan out to multi-agent orchestration,
# which burns subscription tokens fast and adds nothing to the memory-stack
# use case, so installs default them off; users re-enable in settings.json.
SETTINGS_DEFAULTS = {
    "disableWorkflows": True,
    "workflowKeywordTriggerEnabled": False,
}

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

    changed = False
    for key, value in SETTINGS_DEFAULTS.items():
        if key not in settings:
            settings[key] = value
            changed = True
            note(f"set {key} = {json.dumps(value)} (token saver; override in settings.json if wanted)")

    hooks_cfg = settings.setdefault("hooks", {})
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
    # Deregister the pre-rename name first so the tools don't show up twice.
    legacy = subprocess.run(
        [claude, "mcp", "get", MCP_NAME_LEGACY], capture_output=True, text=True, timeout=30
    )
    if legacy.returncode == 0:
        removed = subprocess.run(
            [claude, "mcp", "remove", "--scope", "user", MCP_NAME_LEGACY],
            capture_output=True, text=True, timeout=30,
        )
        if removed.returncode == 0:
            note(f"removed legacy MCP registration {MCP_NAME_LEGACY}")
        else:
            fail(f"could not remove legacy MCP {MCP_NAME_LEGACY}: "
                 f"{(removed.stderr or removed.stdout).strip()}")
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


def patch_claude_md() -> None:
    print("==> ~/.claude/CLAUDE.md (memory priority instruction)")
    # An existing block (either marker generation) is proof of a past opt-in:
    # keep it current without re-asking — a declined re-run must not strand
    # instructions pointing at the deregistered mcp__hermes-memory__* tools.
    # The opt-in prompt only gates ADDING the block to an untouched file.
    text = CLAUDE_MD.read_text() if CLAUDE_MD.exists() else ""
    for start_marker, end_marker in (
        (MARKER_START, MARKER_END),
        (MARKER_START_LEGACY, MARKER_END_LEGACY),
    ):
        if start_marker in text and end_marker in text:
            start = text.index(start_marker)
            end = text.index(end_marker) + len(end_marker)
            new_text = text[:start] + CLAUDE_MD_BLOCK + text[end:]
            if new_text == text:
                note("already present and up to date")
                return
            CLAUDE_MD.write_text(new_text)
            note("updated existing block"
                 + (" (migrated from legacy markers)" if start_marker == MARKER_START_LEGACY else ""))
            return

    opted_in = os.environ.get(
        "LONGBRAIN_CONFIGURE_CLAUDE_MD", os.environ.get("HERMES_CONFIGURE_CLAUDE_MD")
    )
    if opted_in != "1":
        note("skipped (user did not opt in during setup)")
        return

    CLAUDE_MD.parent.mkdir(parents=True, exist_ok=True)
    sep = "" if not text else ("\n\n" if not text.endswith("\n") else "\n")
    CLAUDE_MD.write_text(text + sep + CLAUDE_MD_BLOCK + "\n")
    note("added block")


def main() -> int:
    patch_settings()
    register_mcp()
    patch_claude_md()
    print("✓ Claude Code wired" if ok_all else "✗ finished with problems (see above)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
