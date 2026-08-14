#!/usr/bin/env python3
"""Wire Codex to the memory stack.

Codex support has three parts:
1. register the MCP server so the model can actively call memory tools;
2. register official lifecycle hooks for automatic recall, recording, and
   session-start catch-up;
3. wrap Codex's top-level `notify` command so each turn-ended notification
   syncs completed rollout turns into Longbrain.

Run via setup.sh (or directly). Idempotent — safe to re-run.

The file is edited text-level (the macOS system python has no tomllib):
an existing `[mcp_servers.longbrain]` section is updated in place and its
sub-tables (`[mcp_servers.longbrain.tools.*]`, user-set approval modes) are
left untouched; a missing section is appended at end of file, which is
always a fresh valid TOML table.

Exit code 0 = wired (or Codex not installed — nothing to do); 1 = problem.
"""

import ast
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
CODEX_HOME = Path(os.environ.get("CODEX_HOME", "")) if os.environ.get("CODEX_HOME") \
    else Path.home() / ".codex"
CONFIG = CODEX_HOME / "config.toml"
HOOKS_CONFIG = CODEX_HOME / "hooks.json"
GLOBAL_AGENTS = CODEX_HOME / "AGENTS.md"
SECTION = "[mcp_servers.longbrain]"
MCP_URL = "http://localhost:8800/mcp"
URL_LINE = f'url = "{MCP_URL}"'
# Empty (default) = the service's own auth is disabled too; matches it.
# env_http_headers references an ENV VAR NAME (not the secret itself) —
# Codex reads LONGBRAIN_API_KEY from its own process environment at
# connection time, so the key never lands in config.toml.
API_KEY = os.environ.get("LONGBRAIN_API_KEY", "")
ENV_HEADER_LINE = 'env_http_headers = { "X-API-Key" = "LONGBRAIN_API_KEY" }'
HOOK_SCRIPT = REPO / "hooks" / "codex" / "turn_ended.py"
NOTIFY_MARKER = "Longbrain Codex notify"
AGENTS_MARKER_START = "<!-- longbrain:codex-memory-priority:start (managed by setup.sh) -->"
AGENTS_MARKER_END = "<!-- longbrain:codex-memory-priority:end -->"
AGENTS_BLOCK = f"""{AGENTS_MARKER_START}
## Longbrain Memory

Before answering any non-trivial project question, check Longbrain first.
Use the automatically injected "Long-term memory (auto-recalled)" context
when it is present. If it is missing or too thin, call the Longbrain MCP
tool `memory_recall` (shown as `mcp__longbrain.memory_recall` or
`mcp__longbrain__memory_recall`, depending on the Codex surface) before
doing substantive analysis. If the Longbrain MCP tools are not visible,
use tool discovery/search for "longbrain memory_recall" and then call it.

For questions about previous discussions, reviews, decisions, project
history, specs, docs, or "what we did before", recall/search Longbrain
before saying context is missing or reconstructing from files. Treat
Longbrain as the shared long-term project memory across Codex, Claude Code,
Hermes Desktop, and future adapters.
{AGENTS_MARKER_END}"""
LIFECYCLE_HOOKS = {
    "SessionStart": {
        "script": REPO / "hooks" / "codex" / "session_start.py",
        "matcher": "startup|resume|clear|compact",
        "timeout": 10,
        "statusMessage": "Checking Longbrain",
    },
    "UserPromptSubmit": {
        "script": REPO / "hooks" / "codex" / "user_prompt_submit.py",
        "timeout": 10,
        "statusMessage": "Recalling Longbrain memory",
    },
    "Stop": {
        "script": REPO / "hooks" / "codex" / "stop.py",
        "timeout": 10,
        "statusMessage": "Recording turn in Longbrain",
    },
}

ok_all = True


def note(msg: str) -> None:
    print(f"  {msg}")


def fail(msg: str) -> None:
    global ok_all
    ok_all = False
    print(f"  ✗ {msg}")


def detected() -> bool:
    return shutil.which("codex") is not None or CODEX_HOME.is_dir()


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_array(values: list[str]) -> str:
    return "[" + ", ".join(_toml_string(v) for v in values) + "]"


def _parse_string_array(value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value.strip())
    except Exception:
        return []
    if isinstance(parsed, list) and all(isinstance(v, str) for v in parsed):
        return parsed
    return []


def _top_level_key_span(lines: list[str], key: str) -> Optional[tuple[int, int]]:
    """Return the line span for a top-level array key, including continuations."""
    for start, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("["):
            return None
        if stripped.startswith("#") or "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        if lhs.strip() != key:
            continue

        depth = 0
        quote = ""
        escaped = False
        for end in range(start, len(lines)):
            fragment = rhs if end == start else lines[end]
            in_comment = False
            for char in fragment:
                if in_comment:
                    continue
                if escaped:
                    escaped = False
                    continue
                if quote:
                    if char == "\\" and quote == '"':
                        escaped = True
                    elif char == quote:
                        quote = ""
                    continue
                if char in {'"', "'"}:
                    quote = char
                elif char == "#":
                    in_comment = True
                elif char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
            if depth <= 0:
                return start, end
        return start, len(lines) - 1
    return None


def _is_codex_hook_path(value: str) -> bool:
    return Path(value).as_posix().endswith("/hooks/codex/turn_ended.py")


def _is_our_notify(values: list[str]) -> bool:
    return len(values) >= 2 and values[1] == str(HOOK_SCRIPT)


def _unwrap_codex_notify(values: list[str]) -> list[str]:
    """Recover the original notifier from an older Longbrain wrapper."""
    if len(values) < 2 or not _is_codex_hook_path(values[1]):
        return values
    try:
        idx = values.index("--chain-json")
        chained = json.loads(values[idx + 1])
    except (ValueError, IndexError, json.JSONDecodeError):
        return []
    return chained if isinstance(chained, list) else []


def _without_longbrain_previous_notify(values: list[str]) -> list[str]:
    """Remove old nested Longbrain wrappers from another notifier.

    Some Codex Desktop builds wrap notify themselves (for example Computer
    Use's `--previous-notify`). If our previous installer was already nested
    there, installing Longbrain as the top-level notifier would otherwise run
    Longbrain twice. Keep the outer notifier, but strip its stale Longbrain
    previous-notify pointer.
    """
    cleaned = list(values)
    idx = 0
    while idx < len(cleaned) - 1:
        if cleaned[idx] not in {"--previous-notify", "--chain-json"}:
            idx += 1
            continue
        try:
            nested = json.loads(cleaned[idx + 1])
        except json.JSONDecodeError:
            idx += 2
            continue
        if not isinstance(nested, list):
            idx += 2
            continue
        if len(nested) >= 2 and _is_codex_hook_path(str(nested[1])):
            original = _unwrap_codex_notify([str(v) for v in nested])
            base = cleaned[:idx]
            if not original or original == base:
                del cleaned[idx:idx + 2]
                continue
            cleaned[idx + 1] = json.dumps(original)
        idx += 2
    return cleaned


def _notify_command(existing: list[str]) -> list[str]:
    existing = _unwrap_codex_notify(existing)
    existing = _without_longbrain_previous_notify(existing)
    command = ["python3", str(HOOK_SCRIPT)]
    if existing and not _is_our_notify(existing):
        command.extend(["--chain-json", json.dumps(existing)])
    return command


def _patch_notify(lines: list[str]) -> tuple[list[str], bool]:
    span = _top_level_key_span(lines, "notify")
    if span is None:
        return [
            f"# {NOTIFY_MARKER}: sync completed Codex turns into Longbrain",
            "notify = " + _toml_array(_notify_command([])),
            "",
            *lines,
        ], True

    notify_idx, notify_end = span
    value = "\n".join([
        lines[notify_idx].split("=", 1)[1],
        *lines[notify_idx + 1:notify_end + 1],
    ])
    existing = _parse_string_array(value)
    if not existing:
        fail("top-level notify must be an array of strings; config left unchanged")
        return lines, False
    if _is_our_notify(existing):
        cleaned = _notify_command(existing)
        if cleaned != existing:
            new_lines = list(lines)
            new_lines[notify_idx:notify_end + 1] = [
                "notify = " + _toml_array(cleaned)
            ]
            note("cleaned nested Longbrain notify wrapper")
            return new_lines, True
        note("Codex notify hook already registered")
        return lines, False

    new_lines = list(lines)
    comment = f"# {NOTIFY_MARKER}: wraps any previous notify command"
    if notify_idx == 0 or NOTIFY_MARKER not in new_lines[notify_idx - 1]:
        new_lines.insert(notify_idx, comment)
        notify_idx += 1
        notify_end += 1
    new_lines[notify_idx:notify_end + 1] = [
        "notify = " + _toml_array(_notify_command(existing))
    ]
    note("registered Codex turn-ended notify hook")
    return new_lines, True


def _lifecycle_handler(spec: dict) -> dict:
    return {
        "type": "command",
        "command": f'python3 "{spec["script"]}"',
        "timeout": spec["timeout"],
        "statusMessage": spec["statusMessage"],
    }


def _is_lifecycle_command(command: str, script_name: str) -> bool:
    normalized = command.replace("\\", "/")
    return normalized.endswith(f'/hooks/codex/{script_name}"') or normalized.endswith(
        f"/hooks/codex/{script_name}"
    )


def _merge_lifecycle_hooks(config: dict) -> tuple[dict, bool]:
    hooks = config.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        fail(f"{HOOKS_CONFIG} field 'hooks' must be an object")
        return config, False
    changed = False

    for event, spec in LIFECYCLE_HOOKS.items():
        groups = hooks.setdefault(event, [])
        if not isinstance(groups, list):
            fail(f"{HOOKS_CONFIG} hooks.{event} must be an array")
            continue
        expected = _lifecycle_handler(spec)
        found = False
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                continue
            for index, handler in enumerate(group["hooks"]):
                command = handler.get("command", "") if isinstance(handler, dict) else ""
                if _is_lifecycle_command(command, spec["script"].name):
                    if not found:
                        if handler != expected:
                            group["hooks"][index] = expected
                            changed = True
                        if spec.get("matcher") and group.get("matcher") != spec["matcher"]:
                            group["matcher"] = spec["matcher"]
                            changed = True
                        found = True
                    else:
                        group["hooks"][index] = None
                        changed = True
            group["hooks"] = [handler for handler in group["hooks"] if handler is not None]
        if not found:
            group = {"hooks": [expected]}
            if spec.get("matcher"):
                group["matcher"] = spec["matcher"]
            groups.append(group)
            changed = True
            note(f"registered Codex {event} lifecycle hook")
    return config, changed


def register_lifecycle_hooks() -> None:
    try:
        config = json.loads(HOOKS_CONFIG.read_text()) if HOOKS_CONFIG.exists() else {}
    except json.JSONDecodeError:
        fail(f"{HOOKS_CONFIG} is not valid JSON; lifecycle hooks left unchanged")
        return
    if not isinstance(config, dict):
        fail(f"{HOOKS_CONFIG} must contain a JSON object")
        return
    config, changed = _merge_lifecycle_hooks(config)
    if not changed:
        note("Codex lifecycle hooks already registered")
        return
    if HOOKS_CONFIG.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        shutil.copyfile(HOOKS_CONFIG, HOOKS_CONFIG.with_name(f"hooks.json.bak.{stamp}"))
    HOOKS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    HOOKS_CONFIG.write_text(json.dumps(config, indent=2) + "\n")
    note("hooks.json written (review/trust Longbrain hooks with /hooks in Codex)")


def patch_global_agents() -> None:
    """Install a model-visible fallback instruction.

    Lifecycle hooks are the real pre-model path. Codex still requires the user
    to review/trust hooks once via /hooks, so this global AGENTS.md block keeps
    the model behavior correct even before that trust step has happened.
    """
    print(f"==> {GLOBAL_AGENTS} (Longbrain recall fallback)")
    text = GLOBAL_AGENTS.read_text() if GLOBAL_AGENTS.exists() else ""
    if AGENTS_MARKER_START in text and AGENTS_MARKER_END in text:
        start = text.index(AGENTS_MARKER_START)
        end = text.index(AGENTS_MARKER_END) + len(AGENTS_MARKER_END)
        new_text = text[:start] + AGENTS_BLOCK + text[end:]
        if new_text == text:
            note("Longbrain AGENTS.md block already present")
            return
        text = new_text
        note("updated Longbrain AGENTS.md block")
    else:
        text = (text.rstrip() + "\n\n" if text.strip() else "") + AGENTS_BLOCK + "\n"
        note("added Longbrain AGENTS.md block")
    GLOBAL_AGENTS.parent.mkdir(parents=True, exist_ok=True)
    if GLOBAL_AGENTS.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        shutil.copyfile(GLOBAL_AGENTS, GLOBAL_AGENTS.with_name(f"AGENTS.md.bak.{stamp}"))
    GLOBAL_AGENTS.write_text(text)


def register_mcp() -> None:
    print(f"==> {CONFIG} (Codex wiring)")
    lines = CONFIG.read_text().splitlines() if CONFIG.exists() else []
    changed = False

    header_idx = next(
        (i for i, line in enumerate(lines) if line.strip() == SECTION), None
    )
    if header_idx is None:
        block = ["", "# Longbrain shared memory (added by longbrain setup)",
                 SECTION, URL_LINE]
        if API_KEY:
            block.append(ENV_HEADER_LINE)
        new_lines = lines + block
        note(f"registered MCP longbrain -> {MCP_URL}"
             + (", with X-API-Key from $LONGBRAIN_API_KEY" if API_KEY else ""))
        changed = True
    else:
        # Body of the main section only — it ends at the next table header,
        # which may be one of our own sub-tables (tools.* approval modes);
        # those belong to the user and are preserved as-is.
        end = next(
            (i for i in range(header_idx + 1, len(lines))
             if lines[i].lstrip().startswith("[")),
            len(lines),
        )
        url_idx = next(
            (i for i in range(header_idx + 1, end)
             if lines[i].split("=")[0].strip() == "url"),
            None,
        )
        new_lines = list(lines)
        if url_idx is not None and lines[url_idx].split("=", 1)[1].strip().strip('"') == MCP_URL:
            note("MCP longbrain already registered")
        elif url_idx is not None:
            new_lines[url_idx] = URL_LINE
            note(f"updated MCP longbrain url -> {MCP_URL}")
            changed = True
        else:
            new_lines.insert(header_idx + 1, URL_LINE)
            note(f"set MCP longbrain url -> {MCP_URL}")
            changed = True

        # env_http_headers is fully managed by us (only key we ever write
        # here besides url) — add/update/remove to match whether an
        # LONGBRAIN_API_KEY is currently configured.
        end = next(
            (i for i in range(header_idx + 1, len(new_lines))
             if new_lines[i].lstrip().startswith("[")),
            len(new_lines),
        )
        header_line_idx = next(
            (i for i in range(header_idx + 1, end)
             if new_lines[i].split("=")[0].strip() == "env_http_headers"),
            None,
        )
        if API_KEY and header_line_idx is None:
            url_now_idx = next(
                i for i in range(header_idx + 1, len(new_lines))
                if new_lines[i].split("=")[0].strip() == "url"
            )
            new_lines.insert(url_now_idx + 1, ENV_HEADER_LINE)
            note("set MCP longbrain X-API-Key header from $LONGBRAIN_API_KEY")
            changed = True
        elif API_KEY and new_lines[header_line_idx] != ENV_HEADER_LINE:
            new_lines[header_line_idx] = ENV_HEADER_LINE
            note("updated MCP longbrain X-API-Key header")
            changed = True
        elif not API_KEY and header_line_idx is not None:
            del new_lines[header_line_idx]
            note("removed MCP longbrain X-API-Key header (auth disabled)")
            changed = True

    new_lines, notify_changed = _patch_notify(new_lines)
    changed = changed or notify_changed
    if not changed:
        return

    if CONFIG.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        shutil.copyfile(CONFIG, CONFIG.with_name(f"config.toml.bak.{stamp}"))
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text("\n".join(new_lines) + "\n")
    note("config.toml written (restart Codex sessions to pick it up)")


def main() -> int:
    if not detected():
        print("Codex not found (no `codex` on PATH, no ~/.codex) — nothing to do.")
        return 0
    register_mcp()
    register_lifecycle_hooks()
    patch_global_agents()
    if ok_all:
        print("✓ Codex wired (lifecycle hooks + MCP + notify fallback).")
        print("  Restart Codex, run /hooks once to trust the Longbrain hooks, then "
              "finish a turn and verify with scripts/doctor.py.")
    else:
        print("✗ finished with problems (see above)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
