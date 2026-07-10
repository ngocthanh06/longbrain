"""Codex adapter: rollout extraction and config patching."""

import importlib.util
import json
import subprocess
from pathlib import Path

import configure_codex


def _load_turn_ended():
    path = Path(__file__).resolve().parents[2] / "hooks" / "codex" / "turn_ended.py"
    spec = importlib.util.spec_from_file_location("codex_turn_ended", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


turn_ended = _load_turn_ended()


def _write_rollout(path, entries):
    path.write_text("\n".join(json.dumps(e) for e in entries))
    return path


def _msg(role, text, turn_id, phase=None):
    payload = {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}],
        "internal_chat_message_metadata_passthrough": {"turn_id": turn_id},
    }
    if phase:
        payload["phase"] = phase
    return {"type": "response_item", "payload": payload}


def test_extract_completed_turns_skips_context_and_commentary(tmp_path):
    rollout = _write_rollout(tmp_path / "rollout-2026-07-10T00-00-00-session123.jsonl", [
        {"type": "session_meta", "payload": {"session_id": "session123", "cwd": str(tmp_path)}},
        _msg("user", "<environment_context>noise</environment_context>", "t1"),
        _msg("user", "real prompt", "t1"),
        _msg("assistant", "working...", "t1", phase="commentary"),
        _msg("assistant", "final answer", "t1", phase="final_answer"),
    ])

    assert turn_ended.extract_completed_turns(rollout) == [{
        "session_id": "session123",
        "turn_id": "t1",
        "cwd": str(tmp_path),
        "user_message": "real prompt",
        "assistant_response": "final answer",
    }]


def test_extract_completed_turns_requires_final_answer(tmp_path):
    rollout = _write_rollout(tmp_path / "rollout.jsonl", [
        {"type": "session_meta", "payload": {"session_id": "s1"}},
        _msg("user", "real prompt", "t1"),
        _msg("assistant", "commentary only", "t1", phase="commentary"),
    ])

    assert turn_ended.extract_completed_turns(rollout) == []


def test_extract_completed_turns_keeps_html_like_prompt(tmp_path):
    rollout = _write_rollout(tmp_path / "rollout.jsonl", [
        _msg("user", "<div>Build this component</div>", "t1"),
        _msg("assistant", "done", "t1", phase="final_answer"),
    ])

    assert turn_ended.extract_completed_turns(rollout)[0]["user_message"] == (
        "<div>Build this component</div>"
    )


def _patch_target(monkeypatch, path):
    monkeypatch.setattr(configure_codex, "CONFIG", path)
    monkeypatch.setattr(configure_codex, "HOOK_SCRIPT", Path("/repo/hooks/codex/turn_ended.py"))
    monkeypatch.setattr(configure_codex, "ok_all", True)


def test_configure_codex_wraps_existing_notify_and_keeps_mcp_tools(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    target.write_text(
        'notify = ["/old/notifier", "turn-ended"]\n\n'
        "[mcp_servers.longbrain]\n"
        'url = "http://localhost:8800/mcp"\n\n'
        "[mcp_servers.longbrain.tools.memory_recall]\n"
        'approval_mode = "approve"\n'
    )
    _patch_target(monkeypatch, target)

    configure_codex.register_mcp()
    text = target.read_text()

    assert 'notify = ["python3", "/repo/hooks/codex/turn_ended.py", "--chain-json",' in text
    assert '\\"/old/notifier\\"' in text
    assert "[mcp_servers.longbrain.tools.memory_recall]" in text
    assert 'approval_mode = "approve"' in text


def test_configure_codex_is_idempotent(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    _patch_target(monkeypatch, target)

    configure_codex.register_mcp()
    first = target.read_text()
    configure_codex.register_mcp()

    assert target.read_text() == first


def test_configure_codex_ignores_notify_inside_table(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    target.write_text(
        "[features]\n"
        'notify = ["feature-local"]\n'
    )
    _patch_target(monkeypatch, target)

    configure_codex.register_mcp()
    text = target.read_text()

    assert text.startswith("# Longbrain Codex notify")
    assert 'notify = ["feature-local"]' in text


def test_configure_codex_wraps_multiline_notify(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    target.write_text(
        "notify = [\n"
        '  "/old/notifier",\n'
        '  "turn-ended",\n'
        "]\n\n"
        "[features]\n"
        "web_search = true\n"
    )
    _patch_target(monkeypatch, target)

    configure_codex.register_mcp()
    text = target.read_text()

    assert '\\"/old/notifier\\"' in text
    assert '  "turn-ended",' not in text
    subprocess.run(
        ["python3", "-c", "import tomllib; tomllib.load(open(__import__('sys').argv[1], 'rb'))", str(target)],
        check=True,
    )


def test_configure_codex_replaces_moved_wrapper_without_nesting(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    old_chain = json.dumps(["/old/notifier", "turn-ended"])
    target.write_text(
        "notify = " + configure_codex._toml_array([
            "python3", "/old/repo/hooks/codex/turn_ended.py", "--chain-json", old_chain,
        ]) + "\n"
    )
    _patch_target(monkeypatch, target)

    configure_codex.register_mcp()
    text = target.read_text()

    assert "/old/repo/hooks/codex/turn_ended.py" not in text
    assert text.count("--chain-json") == 1
    assert '\\"/old/notifier\\"' in text


def test_main_runs_original_notifier_when_sync_raises(monkeypatch):
    called = []
    monkeypatch.setattr(turn_ended, "sync_recent_rollouts", lambda: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(turn_ended, "_run_chain", lambda chain, rest: called.append((chain, rest)))
    monkeypatch.setattr(turn_ended.sys, "argv", ["turn_ended.py", "--chain-json", '["old"]', "event"])

    assert turn_ended.main() == 0
    assert called == [('["old"]', ["event"])]


def test_sync_state_distinguishes_extracted_from_recorded(tmp_path, monkeypatch):
    rollout = _write_rollout(tmp_path / "rollout.jsonl", [
        _msg("user", "prompt", "t1"),
        _msg("assistant", "answer", "t1", phase="final_answer"),
    ])
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(turn_ended, "STATE_FILE", state_file)
    monkeypatch.setattr(turn_ended, "_latest_rollouts", lambda: [rollout])
    monkeypatch.setattr(turn_ended, "post_json", lambda _path, _body: False)

    assert turn_ended.sync_recent_rollouts() == 0
    state = json.loads(state_file.read_text())
    assert state["last_scan_extracted"] == 1
    assert state["processed"] == []

    monkeypatch.setattr(turn_ended, "post_json", lambda _path, _body: True)
    assert turn_ended.sync_recent_rollouts() == 1
    state = json.loads(state_file.read_text())
    assert len(state["processed"]) == 1
    assert state["last_successful_write_at"] > 0
