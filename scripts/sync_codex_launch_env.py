#!/usr/bin/env python3
"""Sync LONGBRAIN_API_KEY into this user's launchd environment for Codex.

Codex's MCP `env_http_headers` (see configure_codex.py) makes the `codex`
process read LONGBRAIN_API_KEY from its OWN launch environment at connect
time — not from this repo's .env, and not from whatever shell happened to
run setup.sh. `launchctl setenv` is the mechanism that makes a var visible
to every app a user launches afterwards (including GUI apps like
ChatGPT.app), but it does not persist across logout/reboot on its own — a
LaunchAgent (installed by configure_host_jobs.install_codex_env_agent)
re-runs this script at every login to keep it in sync.

Run via setup.sh (through configure_codex.py, for immediate effect) or
directly. Idempotent.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hooks"))
from api_auth import _read_key  # noqa: E402


def sync() -> None:
    key = _read_key()
    if key:
        subprocess.run(["launchctl", "setenv", "LONGBRAIN_API_KEY", key], check=False)
    else:
        subprocess.run(["launchctl", "unsetenv", "LONGBRAIN_API_KEY"], check=False)


if __name__ == "__main__":
    sync()
