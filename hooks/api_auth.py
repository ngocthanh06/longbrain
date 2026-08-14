"""Shared X-API-Key header helper for every Longbrain hook/adapter HTTP call.

LONGBRAIN_API_KEY empty (default) means auth is disabled server-side too —
api_key_header() then returns {}, which merges into a headers dict as a
no-op, so every caller's existing (no-auth) behavior is unchanged.

Hooks are spawned by Claude Code / Codex / Hermes Desktop as subprocesses
that do NOT inherit setup.sh's temporary `. ./.env` sourcing — an env var
set only in the environment of the process that ran setup.sh is invisible
here. So the key is read from the repo's .env file directly (falling back
to the process environment, e.g. for tests), not from process env alone.
"""
import os
import re
from pathlib import Path

_REPO_ENV = Path(__file__).resolve().parent.parent / ".env"
# An inline comment only starts at a '#' preceded by whitespace (dotenv/
# Compose convention) — LONGBRAIN_API_KEY=abc#123 keeps the literal
# 'abc#123', it does NOT truncate at the '#'.
_INLINE_COMMENT_RE = re.compile(r"\s+#.*$")


def _parse_value(raw: str) -> str:
    """Docker Compose's env-file rules, the subset this project needs. A
    value starting with a quote ends at its FIRST matching closing quote —
    anything after that (e.g. a trailing inline comment) is discarded, and
    everything between the quotes is kept verbatim (so a quoted value CAN
    contain a literal # or trailing text like `"secret" # comment`).
    Without the quote handling, LONGBRAIN_API_KEY="secret" would send the
    literal string '"secret"' (quotes included) as the header — a
    permanent, silent 401."""
    raw = raw.strip()
    if raw and raw[0] in "\"'":
        end = raw.find(raw[0], 1)
        return raw[1:end] if end != -1 else raw[1:]
    return _INLINE_COMMENT_RE.sub("", raw).strip()


def _read_dotenv_key() -> tuple[bool, str]:
    """(found, value). found=False means .env has no LONGBRAIN_API_KEY
    line at all (missing file counts as not found) — distinct from an
    explicit `LONGBRAIN_API_KEY=` (found=True, value=""), which means
    .env deliberately disables auth and must not be overridden by a
    stale process-env value."""
    try:
        text = _REPO_ENV.read_text()
    except OSError:
        return False, ""
    found = False
    value = ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, rest = line.partition("=")
        if sep and key.strip() == "LONGBRAIN_API_KEY":
            found = True
            value = _parse_value(rest)  # last assignment wins, matches shell/Compose
    return found, value


def _read_key() -> str:
    # The repo's own .env is authoritative for this checkout — if it
    # defines LONGBRAIN_API_KEY (even as empty, i.e. explicitly disabled),
    # that wins. Process env is only a fallback for when .env doesn't
    # mention the var at all (missing file, or no assignment), e.g. a
    # deployment with no checked-out .env, or deliberate test overrides.
    found, value = _read_dotenv_key()
    if found:
        return value
    return os.environ.get("LONGBRAIN_API_KEY", "")


def api_key_header() -> dict:
    key = _read_key()
    return {"X-API-Key": key} if key else {}


if __name__ == "__main__":
    # CLI escape hatch for bash operator scripts (clean_garbage.sh,
    # memory_transfer.sh) so they read the key through this SAME parser
    # instead of a second, inevitably-drifting `${LONGBRAIN_API_KEY:-}`
    # read straight from process env: `API_KEY="$(python3 hooks/api_auth.py)"`.
    print(_read_key())
