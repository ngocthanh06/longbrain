"""api_key_header() must find LONGBRAIN_API_KEY even when hooks are spawned
by an app (Claude Code / Codex / Hermes Desktop) that never sourced the
repo's .env — see hooks/api_auth.py for why process env alone isn't enough.
"""
import importlib

import pytest


@pytest.fixture
def api_auth(monkeypatch, tmp_path):
    """Reload api_auth with _REPO_ENV pointed at a throwaway .env file."""
    import api_auth as module

    monkeypatch.delenv("LONGBRAIN_API_KEY", raising=False)
    fake_env = tmp_path / ".env"
    monkeypatch.setattr(module, "_REPO_ENV", fake_env)
    yield module, fake_env
    importlib.reload(module)  # restore the real _REPO_ENV for other tests


def test_no_key_anywhere_returns_empty_headers(api_auth):
    module, _fake_env = api_auth
    assert module.api_key_header() == {}


def test_reads_key_from_dotenv_file(api_auth):
    module, fake_env = api_auth
    fake_env.write_text("SOME_OTHER_VAR=1\nLONGBRAIN_API_KEY=from-dotenv\n")
    assert module.api_key_header() == {"X-API-Key": "from-dotenv"}


def test_dotenv_takes_precedence_over_a_stale_process_env(api_auth, monkeypatch):
    # The repo's own .env is authoritative — a process-env value (e.g. a
    # shell profile that still exports an old key) must not win, or
    # rotating the key in .env would silently keep sending the old one.
    module, fake_env = api_auth
    fake_env.write_text("LONGBRAIN_API_KEY=from-dotenv\n")
    monkeypatch.setenv("LONGBRAIN_API_KEY", "stale-process-env")
    assert module.api_key_header() == {"X-API-Key": "from-dotenv"}


def test_dotenv_explicitly_empty_disables_even_with_stale_process_env(api_auth, monkeypatch):
    # .env explicitly defining LONGBRAIN_API_KEY= (empty) means "auth is
    # disabled" and must win too, not just a non-empty dotenv value.
    module, fake_env = api_auth
    fake_env.write_text("LONGBRAIN_API_KEY=\n")
    monkeypatch.setenv("LONGBRAIN_API_KEY", "stale-process-env")
    assert module.api_key_header() == {}


def test_process_env_is_a_fallback_when_dotenv_has_no_such_line(api_auth, monkeypatch):
    module, fake_env = api_auth
    fake_env.write_text("SOME_OTHER_VAR=1\n")  # no LONGBRAIN_API_KEY line at all
    monkeypatch.setenv("LONGBRAIN_API_KEY", "from-process-env")
    assert module.api_key_header() == {"X-API-Key": "from-process-env"}


def test_missing_dotenv_file_falls_back_to_process_env(api_auth, monkeypatch):
    module, _fake_env = api_auth  # fake_env was never written
    monkeypatch.setenv("LONGBRAIN_API_KEY", "from-process-env")
    assert module.api_key_header() == {"X-API-Key": "from-process-env"}


def test_missing_dotenv_file_is_not_an_error(api_auth):
    module, _fake_env = api_auth  # fake_env was never written
    assert module.api_key_header() == {}


@pytest.mark.parametrize(
    "line, expected",
    [
        ('LONGBRAIN_API_KEY="quoted-secret"', "quoted-secret"),
        ("LONGBRAIN_API_KEY='single-quoted'", "single-quoted"),
        ("LONGBRAIN_API_KEY=inline # a comment", "inline"),
        ('LONGBRAIN_API_KEY="has # inside quotes"', "has # inside quotes"),
        ("export LONGBRAIN_API_KEY=exported", "exported"),
        ("  LONGBRAIN_API_KEY=with-leading-whitespace  ", "with-leading-whitespace"),
        # '#' with no preceding whitespace is NOT a comment marker (dotenv/
        # Compose convention) — the literal value keeps it.
        ("LONGBRAIN_API_KEY=abc#123", "abc#123"),
        # A quoted value ends at its closing quote; anything after
        # (a trailing inline comment) is discarded, not appended.
        ('LONGBRAIN_API_KEY="secret" # comment', "secret"),
    ],
)
def test_dotenv_value_forms_docker_compose_would_also_accept(api_auth, line, expected):
    module, fake_env = api_auth
    fake_env.write_text(f"# a comment line\n\n{line}\n")
    assert module.api_key_header() == {"X-API-Key": expected}


def test_duplicate_dotenv_assignment_last_one_wins(api_auth):
    module, fake_env = api_auth
    fake_env.write_text("LONGBRAIN_API_KEY=old\nLONGBRAIN_API_KEY=new\n")
    assert module.api_key_header() == {"X-API-Key": "new"}


def test_commented_out_key_is_ignored(api_auth):
    module, fake_env = api_auth
    fake_env.write_text("#LONGBRAIN_API_KEY=disabled\n")
    assert module.api_key_header() == {}
