#!/usr/bin/env python3
"""Wire Codex to the memory stack.

Codex support has two parts:
1. register the MCP server so the model can actively call memory tools;
2. wrap Codex's top-level `notify` command so each turn-ended notification
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
SECTION = "[mcp_servers.longbrain]"
MCP_URL = "http://localhost:8800/mcp"
URL_LINE = f'url = "{MCP_URL}"'
HOOK_SCRIPT = REPO / "hooks" / "codex" / "turn_ended.py"
NOTIFY_MARKER = "Longbrain Codex notify"

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


def _notify_command(existing: list[str]) -> list[str]:
    existing = _unwrap_codex_notify(existing)
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
        new_lines = lines + block
        note(f"registered MCP longbrain -> {MCP_URL}")
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
    if ok_all:
        print("✓ Codex wired (MCP tools + turn-ended chat recording).")
        print("  Verify after a Codex turn: /health last_written_at should advance; "
              "restart Codex sessions to pick config changes up.")
    else:
        print("✗ finished with problems (see above)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
