"""configure_hermes.py's MCP header wiring — Hermes Desktop DOES support a
literal `headers: {name: value}` dict on an mcp_servers.* entry (confirmed
2026-08-14 against the hermes-agent source's connection code), so this
attaches LONGBRAIN_API_KEY the same way configure_claude.py/configure_codex.py
do, instead of refusing to enable auth as an earlier version did.
"""
import pytest

yaml = pytest.importorskip("yaml")

import configure_hermes


@pytest.fixture
def hermes_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"mcp_servers": {}}))
    monkeypatch.setattr(configure_hermes, "CONFIG", config_path)
    monkeypatch.setattr(configure_hermes, "HOOKS", {})  # isolate: only test MCP wiring
    monkeypatch.delenv("LONGBRAIN_API_KEY", raising=False)
    return config_path


def _read(config_path):
    return yaml.safe_load(config_path.read_text())


def test_no_key_registers_entry_without_headers(hermes_config):
    configure_hermes.patch_config()
    entry = _read(hermes_config)["mcp_servers"]["longbrain"]
    assert entry["url"] == configure_hermes.MCP_URL
    assert entry["enabled"] is True
    assert "headers" not in entry


def test_key_set_attaches_x_api_key_header(hermes_config, monkeypatch):
    monkeypatch.setenv("LONGBRAIN_API_KEY", "the-secret")
    configure_hermes.patch_config()
    entry = _read(hermes_config)["mcp_servers"]["longbrain"]
    assert entry["headers"] == {"X-API-Key": "the-secret"}


def test_key_set_restricts_config_file_permissions(hermes_config, monkeypatch):
    monkeypatch.setenv("LONGBRAIN_API_KEY", "the-secret")
    configure_hermes.patch_config()
    mode = hermes_config.stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_existing_unrelated_entry_keys_are_preserved(hermes_config, monkeypatch):
    hermes_config.write_text(yaml.safe_dump({
        "mcp_servers": {
            "longbrain": {
                "url": "http://stale-url/mcp", "enabled": False,
                "tools": {"include": ["memory_recall"]},
            }
        }
    }))
    monkeypatch.setenv("LONGBRAIN_API_KEY", "the-secret")
    configure_hermes.patch_config()
    entry = _read(hermes_config)["mcp_servers"]["longbrain"]
    assert entry["url"] == configure_hermes.MCP_URL
    assert entry["enabled"] is True
    assert entry["headers"] == {"X-API-Key": "the-secret"}
    assert entry["tools"] == {"include": ["memory_recall"]}


def test_removing_the_key_drops_the_stale_header(hermes_config, monkeypatch):
    monkeypatch.setenv("LONGBRAIN_API_KEY", "the-secret")
    configure_hermes.patch_config()
    assert _read(hermes_config)["mcp_servers"]["longbrain"]["headers"] == {"X-API-Key": "the-secret"}

    monkeypatch.delenv("LONGBRAIN_API_KEY", raising=False)
    configure_hermes.patch_config()
    entry = _read(hermes_config)["mcp_servers"]["longbrain"]
    assert "headers" not in entry


def test_rerun_with_same_key_is_a_no_op(hermes_config, monkeypatch):
    monkeypatch.setenv("LONGBRAIN_API_KEY", "the-secret")
    configure_hermes.patch_config()
    first = hermes_config.read_text()
    configure_hermes.patch_config()
    assert hermes_config.read_text() == first
