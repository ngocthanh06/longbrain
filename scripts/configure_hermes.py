#!/usr/bin/env python3
"""Fully-automated Hermes Desktop wiring for the memory stack.

Run via setup.sh (or directly). Idempotent — safe to re-run. Steps:

1. config.yaml   — register the 3 memory hooks (post_llm_call: write turns,
                   pre_llm_call: auto-inject recall, on_session_end: trigger
                   consolidation), hooks_auto_accept: true, and the
                   hermes-memory MCP server.
2. allowlist     — record hook consent in ~/.hermes/shell-hooks-allowlist.json
                   (Desktop has no TTY to approve interactively).
3. serve patch   — Hermes bug: `hermes serve` (the Desktop backend) is missing
                   from _AGENT_COMMANDS so shell hooks never fire for Desktop
                   chats. Re-apply after every Hermes update.
4. LLM env sync  — if the stack has no LLM configured, borrow an available
                   API key from ~/.hermes/.env (NVIDIA first, then Gemini) so
                   auto-consolidation works out of the box.
5. backup agent  — install the nightly Qdrant backup launchd job (macOS).
6. restart       — quit Hermes Desktop (verify the backend died) and relaunch.

Exit code 0 = fully wired; 1 = something needs attention (printed).
"""

import datetime
import json
import os
import plistlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERMES_DIR = Path.home() / ".hermes"
SOUL = HERMES_DIR / "SOUL.md"
CONFIG = HERMES_DIR / "config.yaml"
HERMES_ENV = HERMES_DIR / ".env"
ALLOWLIST = HERMES_DIR / "shell-hooks-allowlist.json"
MAIN_PY = HERMES_DIR / "hermes-agent" / "hermes_cli" / "main.py"
APP = HERMES_DIR / "hermes-agent" / "apps" / "desktop" / "release" / "mac-arm64" / "Hermes.app"

REPO = Path(__file__).resolve().parent.parent
REPO_ENV = REPO / ".env"
MCP_URL = "http://localhost:8800/mcp"

# event -> script (all under hooks/)
HOOKS = {
    "post_llm_call": REPO / "hooks" / "post_llm_call.py",
    "pre_llm_call": REPO / "hooks" / "pre_llm_call.py",
    "on_session_end": REPO / "hooks" / "on_session_end.py",
    "on_session_start": REPO / "hooks" / "on_session_start.py",
}


def hook_command(script: Path) -> str:
    return f'python3 "{script}"'


ok_all = True


def note(msg: str) -> None:
    print(f"  {msg}")


def fail(msg: str) -> None:
    global ok_all
    ok_all = False
    print(f"  ✗ {msg}")


def _backup(path: Path) -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    shutil.copyfile(path, path.with_name(path.name + f".bak.{stamp}"))


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# 1. config.yaml
# ---------------------------------------------------------------------------
def patch_config() -> None:
    print("==> config.yaml")
    try:
        import yaml
    except ImportError:
        fail("PyYAML không có trong python này — chạy qua setup.sh (dùng venv của Hermes)")
        return

    if not CONFIG.exists():
        fail(f"{CONFIG} không tồn tại — cài Hermes Desktop trước")
        return

    cfg = yaml.safe_load(CONFIG.read_text()) or {}
    changed = False

    hooks_cfg = cfg.setdefault("hooks", {})
    for event, script in HOOKS.items():
        command = hook_command(script)
        entries = hooks_cfg.setdefault(event, [])
        if not any(isinstance(e, dict) and e.get("command") == command for e in entries):
            entries[:] = [
                e for e in entries
                if not (isinstance(e, dict) and script.name in str(e.get("command", "")))
            ]
            entries.append({"command": command, "timeout": 10})
            changed = True
            note(f"đã đăng ký hook {event}")
        else:
            note(f"hook {event} đã có")

    if cfg.get("hooks_auto_accept") is not True:
        cfg["hooks_auto_accept"] = True
        changed = True
        note("đã bật hooks_auto_accept")

    mcp = cfg.setdefault("mcp_servers", {})
    if not isinstance(mcp, dict):
        mcp = cfg["mcp_servers"] = {}
    entry = mcp.get("hermes-memory")
    if not (isinstance(entry, dict) and entry.get("url") == MCP_URL and entry.get("enabled")):
        mcp["hermes-memory"] = {"url": MCP_URL, "enabled": True}
        changed = True
        note("đã đăng ký MCP hermes-memory")
    else:
        note("MCP hermes-memory đã có")

    if changed:
        _backup(CONFIG)
        CONFIG.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        note("đã ghi config.yaml (backup .bak.* bên cạnh)")


# ---------------------------------------------------------------------------
# 2. hook consent allowlist
# ---------------------------------------------------------------------------
def patch_allowlist() -> None:
    print("==> hook consent (shell-hooks-allowlist.json)")
    data = {"approvals": []}
    if ALLOWLIST.exists():
        try:
            data = json.loads(ALLOWLIST.read_text())
        except json.JSONDecodeError:
            pass
    approvals = data.setdefault("approvals", [])

    for event, script in HOOKS.items():
        command = hook_command(script)
        mtime = datetime.datetime.fromtimestamp(
            script.stat().st_mtime, tz=datetime.timezone.utc
        ).isoformat().replace("+00:00", "Z")
        existing = next(
            (a for a in approvals
             if a.get("event") == event and a.get("command") == command),
            None,
        )
        if existing:
            if existing.get("script_mtime_at_approval") != mtime:
                existing["script_mtime_at_approval"] = mtime
                existing["approved_at"] = _utcnow()
                note(f"cập nhật consent {event} (script đã đổi)")
            else:
                note(f"consent {event} đã có")
        else:
            approvals.append({
                "approved_at": _utcnow(),
                "command": command,
                "event": event,
                "script_mtime_at_approval": mtime,
            })
            note(f"đã ghi consent {event}")
    ALLOWLIST.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# 3. patch Hermes: `serve` must register shell hooks
# ---------------------------------------------------------------------------
OLD_CMDS = '_AGENT_COMMANDS = {None, "chat", "acp", "rl"}'
NEW_CMDS = '_AGENT_COMMANDS = {None, "chat", "acp", "rl", "serve"}  # "serve" patched by hermes-agent setup: Desktop backend must register shell hooks'
OLD_GATE = 'def _command_has_dedicated_mcp_startup(args) -> bool:\n    if args.command == "acp":\n        return True'
NEW_GATE = 'def _command_has_dedicated_mcp_startup(args) -> bool:\n    if args.command == "serve":  # patched by hermes-agent setup: serve owns its MCP startup\n        return True\n    if args.command == "acp":\n        return True'


def patch_serve() -> None:
    print("==> vá Hermes serve (đăng ký hook cho Desktop backend)")
    if not MAIN_PY.exists():
        fail(f"không tìm thấy {MAIN_PY}")
        return
    src = MAIN_PY.read_text()
    if NEW_CMDS.split("  #")[0] in src:
        note("đã vá từ trước")
        return
    if OLD_CMDS not in src or OLD_GATE not in src:
        fail("source Hermes khác phiên bản mong đợi — vá tay _AGENT_COMMANDS trong hermes_cli/main.py (thêm \"serve\")")
        return
    _backup(MAIN_PY)
    src = src.replace(OLD_CMDS, NEW_CMDS).replace(OLD_GATE, NEW_GATE)
    MAIN_PY.write_text(src)
    note("đã vá (backup .bak.* bên cạnh). Lưu ý: update Hermes sẽ ghi đè — chạy lại setup.sh sau khi update")


# ---------------------------------------------------------------------------
# 3b. SOUL.md memory routing — Hermes has a built-in memory tool (a small
# capped markdown blob) that competes with the hermes-memory MCP tools for
# explicit "remember/forget" commands. This guidance routes durable memory
# to the MCP tools; built-in keeps only a short identity summary.
# ---------------------------------------------------------------------------
SOUL_MARKER_START = "<!-- hermes-memory routing (managed by hermes-agent setup.sh) -->"
SOUL_MARKER_END = "<!-- /hermes-memory routing -->"
SOUL_BLOCK = f"""{SOUL_MARKER_START}
## Long-term memory routing
For durable information — user facts, preferences, decisions, project info,
deadlines, commitments — ALWAYS use the hermes-memory MCP tools:
- `save_memories` to remember; `forget_about` then `forget_memory` to forget
- `memory_recall` / `search_history` to look up past conversations
- `forget_session` / `forget_everything` (only with the user's explicit
  confirmation) for bigger resets
When the user says "remember X" or "forget X", that means the hermes-memory
MCP tools — NOT the built-in memory tool. Use the built-in memory tool only
for a short core identity summary (name, machine, language preference).
{SOUL_MARKER_END}"""


def patch_soul() -> None:
    print("==> SOUL.md (định tuyến memory cho model)")
    text = SOUL.read_text() if SOUL.exists() else ""
    if SOUL_MARKER_START in text and SOUL_MARKER_END in text:
        start = text.index(SOUL_MARKER_START)
        end = text.index(SOUL_MARKER_END) + len(SOUL_MARKER_END)
        if text[start:end] == SOUL_BLOCK:
            note("đã có, không đổi")
            return
        text = text[:start] + SOUL_BLOCK + text[end:]
        note("đã cập nhật block định tuyến")
    else:
        text = (text.rstrip() + "\n\n" if text.strip() else "") + SOUL_BLOCK + "\n"
        note("đã thêm block định tuyến")
    if SOUL.exists():
        _backup(SOUL)
    SOUL.write_text(text)


# ---------------------------------------------------------------------------
# 4. LLM env sync — enable auto-consolidation with an existing key
# ---------------------------------------------------------------------------
def _read_env(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def _set_env_lines(path: Path, updates: dict) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    seen = set()
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key in updates:
            lines[i] = f"{key}={updates[key]}"
            seen.add(key)
    for key, value in updates.items():
        if key not in seen:
            lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")


def sync_llm_env() -> None:
    print("==> LLM cho auto-consolidation")
    repo_env = _read_env(REPO_ENV)
    if repo_env.get("LLM_PROVIDER", "none") not in ("", "none"):
        note(f"LLM_PROVIDER={repo_env['LLM_PROVIDER']} đã cấu hình — giữ nguyên")
        return

    hermes_env = _read_env(HERMES_ENV)
    if hermes_env.get("NVIDIA_API_KEY"):
        updates = {
            "LLM_PROVIDER": "nvidia",
            "LLM_MODEL": "deepseek-ai/deepseek-v4-pro",
            "NVIDIA_API_KEY": hermes_env["NVIDIA_API_KEY"],
        }
        note("dùng NVIDIA key từ ~/.hermes/.env (model: deepseek-v4-pro)")
    elif hermes_env.get("GOOGLE_API_KEY"):
        updates = {
            "LLM_PROVIDER": "gemini",
            "LLM_MODEL": "models/gemini-2.5-flash",
            "GOOGLE_API_KEY": hermes_env["GOOGLE_API_KEY"],
        }
        note("dùng Gemini key từ ~/.hermes/.env")
    else:
        note("không tìm thấy API key — consolidation sẽ chờ tool consolidate_session (LLM_PROVIDER=none)")
        return
    _set_env_lines(REPO_ENV, updates)
    note("đã cập nhật .env — cần `docker compose up -d` để nhận cấu hình")


# ---------------------------------------------------------------------------
# 5. nightly backup launchd agent (macOS)
# ---------------------------------------------------------------------------
def install_backup_agent() -> None:
    print("==> backup tự động (launchd, 2:00 sáng)")
    if sys.platform != "darwin":
        note("không phải macOS — tự cài cron cho scripts/backup.sh")
        return
    template = REPO / "scripts" / "com.hermes.memory-backup.plist.template"
    target = Path.home() / "Library" / "LaunchAgents" / "com.hermes.memory-backup.plist"
    rendered = template.read_text().replace("__REPO__", str(REPO))
    if target.exists() and target.read_text() == rendered:
        note("đã cài từ trước")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered)
    plistlib.loads(rendered.encode())  # validate
    subprocess.run(["launchctl", "unload", str(target)], capture_output=True)
    result = subprocess.run(["launchctl", "load", str(target)], capture_output=True, text=True)
    if result.returncode == 0:
        note(f"đã cài + load {target.name}")
    else:
        fail(f"launchctl load lỗi: {result.stderr.strip()}")


# ---------------------------------------------------------------------------
# 6. restart Hermes Desktop
# ---------------------------------------------------------------------------
def restart_desktop() -> None:
    print("==> restart Hermes Desktop")
    if sys.platform != "darwin":
        note("không phải macOS — tự restart Hermes Desktop thủ công")
        return
    subprocess.run(["osascript", "-e", 'quit app "Hermes"'], capture_output=True)
    for _ in range(15):
        probe = subprocess.run(["pgrep", "-f", "hermes_cli.main serve"], capture_output=True)
        if probe.returncode != 0:
            break
        time.sleep(1)
    else:
        subprocess.run(["pkill", "-f", "hermes_cli.main serve"], capture_output=True)
        time.sleep(2)
        note("backend không tự thoát — đã buộc dừng")
    if APP.exists():
        subprocess.run(["open", "-a", str(APP)], capture_output=True)
        note("đã mở lại Hermes Desktop")
    else:
        note("không tìm thấy Hermes.app — mở lại thủ công")


def main() -> int:
    missing = [s for s in HOOKS.values() if not s.exists()]
    if missing:
        print(f"✗ thiếu hook script: {missing}")
        return 1
    patch_config()
    patch_allowlist()
    patch_serve()
    patch_soul()
    sync_llm_env()
    install_backup_agent()
    if "--no-restart" not in sys.argv:
        restart_desktop()
    print("Hoàn tất" if ok_all else "Có mục cần xử lý tay (xem ✗ ở trên)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
