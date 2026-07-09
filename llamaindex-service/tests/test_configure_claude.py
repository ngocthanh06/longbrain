"""configure_claude.py: the settings.json patcher must both register the
memory hooks and default the token-hungry workflow features off — without
ever overriding a value the user set themselves."""

import json

import configure_claude


def _patch_target(monkeypatch, path):
    monkeypatch.setattr(configure_claude, "SETTINGS", path)
    monkeypatch.setattr(configure_claude, "ok_all", True)


def test_patch_settings_writes_defaults_and_hooks(tmp_path, monkeypatch):
    _patch_target(monkeypatch, tmp_path / "settings.json")
    configure_claude.patch_settings()
    settings = json.loads((tmp_path / "settings.json").read_text())
    assert settings["disableWorkflows"] is True
    assert settings["workflowKeywordTriggerEnabled"] is False
    assert set(settings["hooks"]) == {"UserPromptSubmit", "Stop", "SessionEnd", "SessionStart"}


def test_patch_settings_respects_explicit_user_choice(tmp_path, monkeypatch):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({
        "disableWorkflows": False,  # user deliberately re-enabled workflows
        "workflowKeywordTriggerEnabled": True,
    }))
    _patch_target(monkeypatch, target)
    configure_claude.patch_settings()
    settings = json.loads(target.read_text())
    assert settings["disableWorkflows"] is False
    assert settings["workflowKeywordTriggerEnabled"] is True


def test_patch_settings_is_idempotent(tmp_path, monkeypatch):
    _patch_target(monkeypatch, tmp_path / "settings.json")
    configure_claude.patch_settings()
    first = (tmp_path / "settings.json").read_text()
    configure_claude.patch_settings()
    assert (tmp_path / "settings.json").read_text() == first
    # second run must not leave a backup: nothing changed
    assert not list(tmp_path.glob("settings.json.bak.*"))
