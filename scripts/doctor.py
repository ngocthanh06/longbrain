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

from __future__ import annotations  # `bool | None` below needs this on Python < 3.10

import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hooks"))
import api_auth  # noqa: E402 — same LONGBRAIN_API_KEY reader every hook uses
import configure_claude  # noqa: E402 — reuse HOOKS / SETTINGS / MCP constants
import configure_codex  # noqa: E402 — reuse CONFIG / SECTION / MCP_URL

HEALTH_URL = "http://localhost:8800/health"
AUTH_PROBE_URL = "http://localhost:8800/memory/stats"
LAUNCHD_JOBS = ("com.longbrain.memory-backup", "com.longbrain.memory-ingest")
HERMES_HOME = Path.home() / ".hermes"

problems = 0


def _request_status(url: str, headers: dict) -> int | None:
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except OSError:
        return None


def _probe_auth_key() -> str:
    """Two requests, not one — a single "does the key work" call can't
    tell a real success apart from a server that isn't enforcing auth at
    all (in which case ANY key, or no key, gets 200).
      "no_key"       - .env has no LONGBRAIN_API_KEY; nothing to probe
      "ok"           - unauth request got 401 AND the .env key got 200:
                        auth is enforced and .env matches the server
      "not_enforced" - unauth request did NOT get 401: the server isn't
                        actually applying auth right now, .env or not
      "key_mismatch" - unauth got 401 (auth IS enforced) but the .env key
                        didn't get 200 — stale/wrong key
      "unreachable"  - couldn't complete both requests
    """
    key = api_auth.api_key_header().get("X-API-Key", "")
    if not key:
        return "no_key"
    unauth_status = _request_status(AUTH_PROBE_URL, {})
    if unauth_status is None:
        return "unreachable"
    if unauth_status != 401:
        return "not_enforced"
    auth_status = _request_status(AUTH_PROBE_URL, {"X-API-Key": key})
    if auth_status is None:
        return "unreachable"
    return "ok" if auth_status == 200 else "key_mismatch"


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
    print("==> Codex (lifecycle hooks + MCP)")
    if not configure_codex.detected():
        skip("not installed")
        return
    config = configure_codex.CONFIG
    text = config.read_text() if config.exists() else ""
    if configure_codex.SECTION in text and configure_codex.MCP_URL in text:
        ok(f"MCP longbrain registered in {config}")
    else:
        bad(f"MCP longbrain missing from {config} — re-run ./setup.sh")

    codex_key = api_auth.api_key_header().get("X-API-Key", "")
    has_header = configure_codex.ENV_HEADER_LINE in text
    if codex_key and not has_header:
        # .env has a key now, but this Codex config predates it (or was
        # never re-run after the key was set) — the MCP connection will
        # get 401. Skipping silently here (the old bug) made doctor report
        # "MCP registered" as if everything were fine.
        bad(
            "LONGBRAIN_API_KEY is set in .env, but this Codex config has no "
            "env_http_headers for X-API-Key — its MCP connection will get 401. "
            "Re-run: python3 scripts/configure_codex.py"
        )
    elif has_header and not codex_key:
        skip(
            "Codex config still has env_http_headers, but .env has no "
            "LONGBRAIN_API_KEY now — harmless (Codex sends an empty header), "
            "but re-run scripts/configure_codex.py to clean it up"
        )
    elif has_header:
        # Two SEPARATE questions, deliberately not conflated:
        #   1. Is the key in .env actually the one the running server wants,
        #      AND is the server actually enforcing auth right now? Both are
        #      checked with real requests, not assumed from any single 200.
        #   2. Will the `codex` process have LONGBRAIN_API_KEY in ITS OWN
        #      launch environment (env_http_headers reads Codex's own env at
        #      connect time — not this repo's .env, not whatever env ran
        #      ./setup.sh)? This we CANNOT verify from here, on any shell —
        #      doctor.py's own environment proves nothing about how `codex`
        #      itself will be launched later.
        probe = _probe_auth_key()
        if probe == "ok":
            ok(".env's LONGBRAIN_API_KEY is accepted by the running server (auth enforced)")
        elif probe == "not_enforced":
            bad(
                "Codex is configured to send X-API-Key, but the running server did NOT "
                "reject an unauthenticated request — auth isn't actually enforced yet "
                "(likely a stale container). Run: docker compose up -d"
            )
        elif probe == "key_mismatch":
            bad(
                ".env's LONGBRAIN_API_KEY was REJECTED by the running server (401) even "
                "though it IS enforcing auth — the container is likely running with a "
                "stale/different key. Run: docker compose up -d (to apply the current .env)"
            )
        else:
            skip("could not verify the key against the running server — see the service check above")

        # env_http_headers makes the `codex` process read LONGBRAIN_API_KEY
        # from ITS OWN launch environment at connect time — `launchctl
        # setenv` (synced by configure_codex.py + the codex-env LaunchAgent
        # on every login) is what actually gets it there for GUI-launched
        # apps like ChatGPT.app. Verify that sync actually landed, instead
        # of just noting it can't be checked.
        if not shutil.which("launchctl"):
            skip("launchctl not available (not macOS) — can't verify Codex's launch environment")
        else:
            launch_env = subprocess.run(
                ["launchctl", "getenv", "LONGBRAIN_API_KEY"], capture_output=True, text=True, timeout=15
            ).stdout.strip()
            if launch_env == codex_key:
                ok("launchd env LONGBRAIN_API_KEY matches .env (Codex will pick it up on next launch)")
            else:
                bad(
                    "launchd env LONGBRAIN_API_KEY does not match .env — Codex's MCP "
                    "connection will 401. Re-run: python3 scripts/sync_codex_launch_env.py"
                )
            listed = subprocess.run(
                ["launchctl", "list"], capture_output=True, text=True, timeout=15
            ).stdout
            if "com.longbrain.codex-env" in listed:
                ok("codex-env LaunchAgent loaded (re-syncs the launchd env at every login)")
            else:
                bad("com.longbrain.codex-env LaunchAgent not loaded — re-run ./setup.sh")
    hooks_ok = True
    try:
        hooks_config = json.loads(configure_codex.HOOKS_CONFIG.read_text())
    except FileNotFoundError:
        hooks_config = {}
        hooks_ok = False
        bad(f"Codex lifecycle hooks missing from {configure_codex.HOOKS_CONFIG}")
    except (OSError, json.JSONDecodeError):
        hooks_config = {}
        hooks_ok = False
        bad(f"Codex lifecycle hooks unreadable: {configure_codex.HOOKS_CONFIG}")
    configured = json.dumps(hooks_config)
    for event, spec in configure_codex.LIFECYCLE_HOOKS.items():
        if str(spec["script"]) in configured and spec["script"].exists():
            ok(f"lifecycle hook {event}")
        else:
            hooks_ok = False
            bad(f"Codex {event} hook missing — re-run ./setup.sh")

    lifecycle_state = configure_codex.CODEX_HOME / "longbrain_codex_hooks_state.json"
    if hooks_ok:
        try:
            state = json.loads(lifecycle_state.read_text())
        except FileNotFoundError:
            skip("Codex lifecycle hooks not observed yet — restart Codex, run /hooks, and trust Longbrain hooks")
        except (OSError, json.JSONDecodeError):
            bad(f"Codex lifecycle state is unreadable: {lifecycle_state}")
        else:
            if state.get("last_recall_ok") is True:
                ok("Codex automatic recall observed")
            elif state.get("last_recall_ok") is False:
                bad("Codex recall hook ran but the memory service call failed")
            else:
                skip("Codex recall hook has not handled a searchable prompt yet")
            if state.get("last_write_ok") is True:
                ok("Codex automatic turn recording observed")
            elif state.get("last_write_ok") is False:
                bad(f"Codex write hook failed: {state.get('last_write_error') or 'unknown error'}")
            else:
                skip("Codex write hook has not completed a turn yet")

    agents_text = configure_codex.GLOBAL_AGENTS.read_text() if configure_codex.GLOBAL_AGENTS.exists() else ""
    if configure_codex.AGENTS_MARKER_START in agents_text:
        ok(f"Longbrain fallback instruction present in {configure_codex.GLOBAL_AGENTS}")
    else:
        bad(f"Longbrain fallback instruction missing from {configure_codex.GLOBAL_AGENTS} — re-run ./setup.sh")

    if str(configure_codex.HOOK_SCRIPT) in text and "notify" in text:
        ok("turn-ended notify fallback registered")
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
        skip("notify fallback missing; lifecycle hooks remain the primary adapter")


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
