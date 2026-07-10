#!/usr/bin/env python3
"""Longbrain doctor — one-shot, read-only wiring + health check.

Answers "is my memory actually working?" without running the installer:

  python3 scripts/doctor.py          # check everything, exit 0 = all good
  python3 scripts/doctor.py --fix    # on problems, re-run ./setup.sh
                                     # (idempotent — it only repairs what's off)

Checks: the memory service (/health, last_written_at), the launchd
background jobs (nightly backup, docs/ ingest watcher), and every detected
agent's wiring (Claude Code hooks + MCP, Hermes Desktop hooks, Codex notify
sync + MCP).
Agents that aren't installed are skipped, not failed.
"""

import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import configure_claude  # noqa: E402 — reuse HOOKS / SETTINGS / MCP constants
import configure_codex  # noqa: E402 — reuse CONFIG / SECTION / MCP_URL

HEALTH_URL = "http://localhost:8800/health"
LAUNCHD_JOBS = ("com.longbrain.memory-backup", "com.longbrain.memory-ingest")
HERMES_HOME = Path.home() / ".hermes"

problems = 0


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def bad(msg: str) -> None:
    global problems
    problems += 1
    print(f"  ✗ {msg}")


def skip(msg: str) -> None:
    print(f"  – {msg}")


def check_service() -> None:
    print("==> Memory service")
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=5) as resp:
            health = json.load(resp)
    except Exception as exc:
        bad(f"{HEALTH_URL} unreachable ({exc}) — is the stack up? docker compose up -d")
        return
    if health.get("status") != "ok":
        bad(f"health status = {health.get('status')!r}")
        return
    counts = health.get("collections") or {}
    missing = [name for name, n in counts.items() if n is None]
    if missing:
        bad(f"collections unreadable: {', '.join(missing)}")
    else:
        ok("service healthy: " + ", ".join(f"{k.split('_', 1)[-1]}={v}" for k, v in counts.items()))
    last = health.get("last_written_at")
    if last:
        age_h = (time.time() - float(last)) / 3600
        (ok if age_h < 24 else bad)(
            f"last memory write {age_h:.1f}h ago"
            + ("" if age_h < 24 else " — hooks may not be firing (chat once, re-check)")
        )
    else:
        skip("no write recorded yet (fresh install?)")


def check_background_jobs() -> None:
    print("==> Background jobs (launchd)")
    if not shutil.which("launchctl"):
        skip("launchctl not available (not macOS) — jobs unmanaged here")
        return
    listed = subprocess.run(
        ["launchctl", "list"], capture_output=True, text=True, timeout=15
    ).stdout
    for label in LAUNCHD_JOBS:
        if label in listed:
            ok(f"{label} loaded")
        else:
            bad(f"{label} not loaded — re-run ./setup.sh")


def check_claude() -> None:
    print("==> Claude Code (full adapter)")
    settings_path = configure_claude.SETTINGS
    if not shutil.which("claude") and not settings_path.exists():
        skip("not installed")
        return
    try:
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    except json.JSONDecodeError:
        bad(f"{settings_path} is not valid JSON")
        return
    hooks_cfg = settings.get("hooks") or {}
    for event, (script, _timeout) in configure_claude.HOOKS.items():
        command = configure_claude.hook_command(script)
        present = any(
            h.get("command") == command
            for m in (hooks_cfg.get(event) or []) if isinstance(m, dict)
            for h in (m.get("hooks") or []) if isinstance(h, dict)
        )
        if present and script.exists():
            ok(f"hook {event}")
        elif present:
            bad(f"hook {event} points at a missing script: {script}")
        else:
            bad(f"hook {event} not registered — re-run ./setup.sh")
    if shutil.which("claude"):
        probe = subprocess.run(
            ["claude", "mcp", "get", configure_claude.MCP_NAME],
            capture_output=True, text=True, timeout=30,
        )
        if probe.returncode == 0:
            ok(f"MCP {configure_claude.MCP_NAME} registered")
        else:
            bad(f"MCP {configure_claude.MCP_NAME} not registered — re-run ./setup.sh")
    else:
        skip("`claude` CLI not on PATH — MCP registration unverified")


def check_hermes() -> None:
    print("==> Hermes Desktop (full adapter)")
    if not HERMES_HOME.is_dir():
        skip("not installed")
        return
    if shutil.which("hermes"):
        result = subprocess.run(
            ["hermes", "hooks", "doctor"], capture_output=True, text=True, timeout=60
        )
        tail = (result.stdout or result.stderr).strip().splitlines()[-1:] or ["(no output)"]
        if result.returncode == 0 and "healthy" in (result.stdout or "").lower():
            ok(f"hermes hooks doctor: {tail[0].strip()}")
        else:
            bad(f"hermes hooks doctor: {tail[0].strip()} — re-run ./setup.sh")
        return
    config_yaml = HERMES_HOME / "config.yaml"
    if config_yaml.exists() and "post_llm_call" in config_yaml.read_text():
        ok("hooks present in ~/.hermes/config.yaml (`hermes` CLI not on PATH for a deep check)")
    else:
        bad("hooks missing from ~/.hermes/config.yaml — re-run ./setup.sh")


def check_codex() -> None:
    print("==> Codex (notify write + MCP)")
    if not configure_codex.detected():
        skip("not installed")
        return
    config = configure_codex.CONFIG
    text = config.read_text() if config.exists() else ""
    if configure_codex.SECTION in text and configure_codex.MCP_URL in text:
        ok(f"MCP longbrain registered in {config}")
    else:
        bad(f"MCP longbrain missing from {config} — re-run ./setup.sh")
    if str(configure_codex.HOOK_SCRIPT) in text and "notify" in text:
        ok("turn-ended notify hook registered (records completed turns)")
        skip("no pre-prompt hook: Codex recall is tools-only")
        state_path = configure_codex.CODEX_HOME / "longbrain_codex_notify_state.json"
        try:
            state = json.loads(state_path.read_text())
        except FileNotFoundError:
            skip("notify has not run yet (finish one Codex turn, then re-check)")
        except (OSError, json.JSONDecodeError):
            bad(f"Codex notify state is unreadable: {state_path}")
        else:
            scanned = int(state.get("last_scan_rollouts") or 0)
            extracted = int(state.get("last_scan_extracted") or 0)
            recorded = len(state.get("processed") or [])
            has_scan_stats = bool(state.get("last_scan_at"))
            if has_scan_stats and scanned and not extracted:
                bad("Codex rollout scan extracted 0 turns — rollout format may have changed")
            elif has_scan_stats and recorded:
                ok(
                    f"Codex adapter has recorded {recorded} completed turn(s); "
                    f"parser found {extracted} on last scan"
                )
            elif has_scan_stats and extracted:
                bad("Codex rollout parser works, but no turn has been recorded successfully")
            elif recorded:
                ok(f"Codex adapter has recorded {recorded} completed turn(s)")
                skip("parser health stats will appear after the next Codex turn")
            else:
                skip("Codex has no rollout files to verify yet")
    else:
        bad(f"Codex notify hook missing from {config} — re-run ./setup.sh")


def main() -> int:
    for check in (check_service, check_background_jobs, check_claude, check_hermes, check_codex):
        check()
    print()
    if problems == 0:
        print("✓ All checks passed — memory stack fully wired.")
        return 0
    print(f"✗ {problems} problem(s) found.")
    if "--fix" in sys.argv[1:]:
        print("Running ./setup.sh to repair (idempotent)…\n")
        setup = Path(__file__).resolve().parent.parent / "setup.sh"
        return subprocess.call(["bash", str(setup)])
    print("Fix: re-run ./setup.sh (or: python3 scripts/doctor.py --fix)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
