"""Shared plumbing for the Claude Code hooks (see sibling scripts).

Claude Code runs hooks with a JSON payload on stdin (documented events:
UserPromptSubmit, Stop, SessionStart, SessionEnd). Everything here is
best-effort: a down memory stack must never break a Claude Code session.

Project resolution keeps parallel operation with Hermes coherent:
1. cwd folder match against Hermes' own projects.db (reusing the Hermes
   hook's resolver) — so a folder anchored to a sidebar project gets the
   SAME slug from both agents and memory doesn't split.
2. Otherwise the git-root (or cwd) folder name, slugified — Claude Code
   has a real per-project cwd, so the folder name is a natural project.
   Home dir and filesystem root fall through: a shell opened there isn't
   a project.
3. "default".
Both 1 and 2 report source="folder": cwd is a genuine workspace signal,
so the server-side intentional-move override applies (see ARCHITECTURE.md).
"""

import json
import os
import re
import subprocess
import sys
import urllib.request

MEMORY_BASE = os.environ.get(
    "LONGBRAIN_MEMORY_URL", os.environ.get("HERMES_MEMORY_URL", "http://localhost:8800")
)
DEBUG_LOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "claude-hook-debug.jsonl"
)

# Reuse the Hermes sidebar-project resolver from the sibling hook package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from post_llm_call import resolve_project_with_source  # noqa: E402
from project_catalog import record_project_folder  # noqa: E402

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def env_get(name: str, default=None):
    """LONGBRAIN_* is the current env prefix; HERMES_* stays a legacy alias
    so existing installs keep working without touching their config."""
    value = os.environ.get(name)
    if value is None and name.startswith("LONGBRAIN_"):
        value = os.environ.get("HERMES_" + name[len("LONGBRAIN_"):])
    return default if value is None else value


def env_int(name: str, default: int) -> int:
    """Env override parsed defensively: a malformed value ("abc") must not
    crash a best-effort hook at import time — fall back to the default."""
    try:
        return int(env_get(name, default))
    except (TypeError, ValueError):
        return default


# Hook payloads carry full prompts/responses, so always-on raw logging is a
# privacy leak plus unbounded disk growth. Opt-in only, truncated.
DEBUG_HOOKS = env_get("LONGBRAIN_DEBUG_HOOKS") == "1"
DEBUG_MAX_CHARS = env_int("LONGBRAIN_DEBUG_HOOKS_MAX_CHARS", 2000)


def debug_dump(raw: str) -> None:
    if not DEBUG_HOOKS:
        return
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG), exist_ok=True)
        with open(DEBUG_LOG, "a") as f:
            f.write(raw.strip().replace("\n", " ")[:DEBUG_MAX_CHARS] + "\n")
    except Exception:
        pass


def read_payload() -> dict:
    raw = sys.stdin.read()
    debug_dump(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def post_json(path: str, body: dict, timeout: float = 5.0):
    """POST to the memory service; parsed JSON dict, or None on any failure."""
    request = urllib.request.Request(
        MEMORY_BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None  # best-effort — never break the session


def get_json(path: str, timeout: float = 3.0):
    """GET from the memory service; parsed JSON, or None on any failure."""
    try:
        with urllib.request.urlopen(MEMORY_BASE + path, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None  # best-effort — never break the session


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("-_")[:64]
    return slug if _SLUG_RE.match(slug) else ""


def _git_root(cwd: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def resolve_project(cwd: str) -> tuple:
    """(slug, source) for a Claude Code session — see module docstring."""
    slug, source = resolve_project_with_source(cwd or "")
    if source == "folder":
        return slug, "folder"
    # The Hermes "active sidebar project" signal is ambient Desktop UI state
    # and says nothing about a Claude Code session — skip it, go by folder.
    if cwd:
        root = _git_root(cwd) or os.path.realpath(cwd)
        if root not in (os.path.expanduser("~"), os.sep):
            folder_slug = _slugify(os.path.basename(root))
            if folder_slug:
                # Hermes doesn't know this folder (no Hermes install, or the
                # folder isn't a sidebar project) — record it so host-side
                # jobs with no cwd of their own (the docs/ ingest watcher)
                # can still find it.
                record_project_folder(folder_slug, root)
                return folder_slug, "folder"
    return "default", "default"
