"""Unit tests for scripts/doctor.py's auth-verification logic (security
hardening plan): the guards that catch a stale/mismatched key, a server
not actually enforcing auth, or a Codex config that predates the key —
each was previously a silent-401 gap found in review.
"""
import urllib.error

import pytest

import configure_codex
import doctor


class _FakeResponse:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _set_key(monkeypatch, value):
    monkeypatch.setattr(doctor.api_auth, "api_key_header",
                         lambda: ({"X-API-Key": value} if value else {}))


# ---------------------------------------------------------------------------
# _probe_auth_key — must distinguish "server not enforcing auth" from
# "server enforces auth but this key is wrong" from "it actually matches".
# A single request that just checks for a 200 can't tell these apart.
# ---------------------------------------------------------------------------
def test_probe_no_key_when_env_has_none(monkeypatch):
    _set_key(monkeypatch, "")
    assert doctor._probe_auth_key() == "no_key"


def test_probe_ok_when_unauth_401_then_key_200(monkeypatch):
    _set_key(monkeypatch, "the-key")
    calls = {"n": 0}

    def fake_urlopen(request, timeout=5):
        calls["n"] += 1
        if calls["n"] == 1:  # unauthenticated attempt, first
            raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, None)
        return _FakeResponse(200)  # authenticated attempt, second

    monkeypatch.setattr(doctor.urllib.request, "urlopen", fake_urlopen)
    assert doctor._probe_auth_key() == "ok"


def test_probe_not_enforced_when_unauth_request_succeeds(monkeypatch):
    # The exact bug this guards against: an old container running with
    # auth disabled would return 200 to ANY request, key or not — a
    # single "does my key work" check would have wrongly reported "ok".
    _set_key(monkeypatch, "the-key")
    monkeypatch.setattr(doctor.urllib.request, "urlopen", lambda request, timeout=5: _FakeResponse(200))
    assert doctor._probe_auth_key() == "not_enforced"


def test_probe_key_mismatch_when_unauth_401_but_key_also_rejected(monkeypatch):
    _set_key(monkeypatch, "stale-key")

    def fake_urlopen(request, timeout=5):
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr(doctor.urllib.request, "urlopen", fake_urlopen)
    assert doctor._probe_auth_key() == "key_mismatch"


def test_probe_unreachable_when_service_is_down(monkeypatch):
    _set_key(monkeypatch, "the-key")
    monkeypatch.setattr(doctor.urllib.request, "urlopen",
                         lambda request, timeout=5: (_ for _ in ()).throw(OSError("refused")))
    assert doctor._probe_auth_key() == "unreachable"


# ---------------------------------------------------------------------------
# check_codex() — the header-gap check: .env has a key, but the Codex
# config on disk predates it (never re-run after the key was set). This
# used to be silently skipped entirely (gated behind `if ENV_HEADER_LINE in
# text`, with no else), so doctor reported "MCP registered" as if fine.
# ---------------------------------------------------------------------------
@pytest.fixture
def codex_sandbox(tmp_path, monkeypatch):
    """Point every configure_codex path constant doctor.py touches at a
    throwaway directory — never read/write the real ~/.codex."""
    monkeypatch.setattr(configure_codex, "detected", lambda: True)
    monkeypatch.setattr(configure_codex, "CODEX_HOME", tmp_path)
    monkeypatch.setattr(configure_codex, "CONFIG", tmp_path / "config.toml")
    monkeypatch.setattr(configure_codex, "HOOKS_CONFIG", tmp_path / "hooks.json")
    monkeypatch.setattr(configure_codex, "GLOBAL_AGENTS", tmp_path / "AGENTS.md")
    return tmp_path


def _capture_doctor_output(monkeypatch):
    calls = {"ok": [], "bad": [], "skip": []}
    for name in calls:
        monkeypatch.setattr(doctor, name, (lambda n: lambda msg: calls[n].append(msg))(name))
    return calls


def test_reports_problem_when_key_set_but_codex_header_missing(codex_sandbox, monkeypatch):
    (codex_sandbox / "config.toml").write_text(
        f"{configure_codex.SECTION}\n{configure_codex.URL_LINE}\n"  # no ENV_HEADER_LINE
    )
    _set_key(monkeypatch, "the-key")
    calls = _capture_doctor_output(monkeypatch)

    doctor.check_codex()

    assert any("env_http_headers" in msg for msg in calls["bad"]), calls["bad"]


def test_no_problem_reported_when_header_matches_key_state(codex_sandbox, monkeypatch):
    # Auth disabled (.env has no key) and the config has no header either —
    # the matched, expected default state must not raise a false alarm.
    (codex_sandbox / "config.toml").write_text(
        f"{configure_codex.SECTION}\n{configure_codex.URL_LINE}\n"
    )
    _set_key(monkeypatch, "")
    calls = _capture_doctor_output(monkeypatch)

    doctor.check_codex()

    assert not any("env_http_headers" in msg for msg in calls["bad"])
