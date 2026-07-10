#!/usr/bin/env python3
"""Install the host-side background jobs (macOS launchd agents).

Two jobs, both agent-independent — they must exist on EVERY install, with or
without Hermes Desktop (a Claude-Code-only install needs the docs/ watcher
just as much):

- com.longbrain.memory-backup  — nightly Qdrant backup at 2:00 AM
- com.longbrain.memory-ingest  — docs/ auto-ingest poll every 60s (L4)

Jobs installed under the pre-rename com.hermes.* labels are unloaded and
removed before the com.longbrain.* ones go in.

Run via setup.sh (or directly: python3 scripts/configure_host_jobs.py).
Idempotent — an already-installed, identical plist is left untouched.
Also imported by configure_hermes.py so the Hermes path installs the same
jobs without duplicating this logic.
"""

import plistlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _note(msg: str) -> None:
    print(f"  {msg}")


def _remove_legacy_agent(legacy_name: str) -> None:
    """Unload + delete a pre-rename com.hermes.* job so it doesn't run twice."""
    legacy = Path.home() / "Library" / "LaunchAgents" / legacy_name
    if not legacy.exists():
        return
    subprocess.run(["launchctl", "unload", str(legacy)], capture_output=True)
    legacy.unlink()
    _note(f"removed legacy job {legacy_name}")


def _install_launchd_agent(
    template_name: str, target_name: str, legacy_name: str = "", extra_note: str = ""
) -> bool:
    """Render a plist template into ~/Library/LaunchAgents and load it.
    Returns True when the job is installed (or already was)."""
    if sys.platform != "darwin":
        _note(f"not macOS — set up a cron job for the {target_name} equivalent yourself")
        return True
    if legacy_name:
        _remove_legacy_agent(legacy_name)
    template = REPO / "scripts" / template_name
    target = Path.home() / "Library" / "LaunchAgents" / target_name
    rendered = (
        template.read_text().replace("__REPO__", str(REPO)).replace("__PYTHON__", sys.executable)
    )
    if target.exists() and target.read_text() == rendered:
        _note("already installed")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered)
    plistlib.loads(rendered.encode())  # validate
    subprocess.run(["launchctl", "unload", str(target)], capture_output=True)
    result = subprocess.run(["launchctl", "load", str(target)], capture_output=True, text=True)
    if result.returncode == 0:
        _note(f"installed + loaded {target.name}{extra_note}")
        return True
    print(f"  ✗ launchctl load failed: {result.stderr.strip()}")
    return False


def install_backup_agent() -> bool:
    print("==> automatic backup (launchd, 2:00 AM)")
    return _install_launchd_agent(
        "com.longbrain.memory-backup.plist.template", "com.longbrain.memory-backup.plist",
        legacy_name="com.hermes.memory-backup.plist",
    )


def install_ingest_watcher_agent() -> bool:
    print("==> docs/ auto-ingest (launchd, every 60s)")
    return _install_launchd_agent(
        "com.longbrain.memory-ingest.plist.template", "com.longbrain.memory-ingest.plist",
        legacy_name="com.hermes.memory-ingest.plist",
        extra_note=" (watches each project's docs/ subfolder)",
    )


def main() -> int:
    ok = install_backup_agent()
    ok = install_ingest_watcher_agent() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
