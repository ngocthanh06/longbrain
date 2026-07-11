"""Claude Code adapter hooks: transcript extraction and project resolution.
Stdlib-only (the hooks must run with the host python3, which may be 3.9)."""

import json
import shutil
import sqlite3
import subprocess

import pytest

import post_llm_call
import project_catalog
from common import _slugify, resolve_project
from stop import extract_last_turn


# ---------------------------------------------------------------------------
# transcript parsing (format is internal to Claude Code — parse defensively)
# ---------------------------------------------------------------------------
def _write_transcript(path, entries):
    path.write_text("\n".join(json.dumps(e) for e in entries))
    return str(path)


def _user(content, **extra):
    return {"type": "user", "message": {"role": "user", "content": content}, **extra}


def _assistant(content):
    return {"type": "assistant", "message": {"role": "assistant", "content": content}}


def test_extract_simple_turn(tmp_path):
    p = _write_transcript(tmp_path / "t.jsonl", [
        _user("first question"),
        _assistant([{"type": "text", "text": "first answer"}]),
        _user("second question"),
        _assistant([{"type": "text", "text": "second answer"}]),
    ])
    assert extract_last_turn(p) == ("second question", "second answer")


def test_extract_final_text_wins_over_interim(tmp_path):
    # Text emitted between tool calls must not shadow the closing reply,
    # and tool_result user entries must not reset the turn.
    p = _write_transcript(tmp_path / "t.jsonl", [
        _user("do the thing"),
        _assistant([{"type": "text", "text": "let me check"},
                    {"type": "tool_use", "id": "x", "name": "Bash", "input": {}}]),
        _user([{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]),
        _assistant([{"type": "text", "text": "done: the thing works"}]),
    ])
    assert extract_last_turn(p) == ("do the thing", "done: the thing works")


def test_extract_skips_meta_sidechain_and_garbage(tmp_path):
    p = _write_transcript(tmp_path / "t.jsonl", [
        {"type": "summary", "summary": "irrelevant"},
        _user("real prompt"),
        _user("sidechain prompt", isSidechain=True),
        _user("meta line", isMeta=True),
        _assistant([{"type": "text", "text": "real answer"}]),
    ])
    # garbage line appended raw
    with open(p, "a") as f:
        f.write("\nnot json at all")
    assert extract_last_turn(p) == ("real prompt", "real answer")


def test_extract_string_content(tmp_path):
    p = _write_transcript(tmp_path / "t.jsonl", [_user("hi"), _assistant("hello!")])
    assert extract_last_turn(p) == ("hi", "hello!")


def test_extract_missing_file():
    assert extract_last_turn("/nonexistent/t.jsonl") == ("", "")


# ---------------------------------------------------------------------------
# project resolution: projects.db parity with Hermes, then folder fallback
# ---------------------------------------------------------------------------
def test_slugify():
    assert _slugify("Hermes Agent") == "hermes-agent"
    assert _slugify("my_repo-2") == "my_repo-2"
    assert _slugify("---") == ""
    assert _slugify("Ω≈ç√") == ""


@pytest.fixture()
def hermes_db(tmp_path, monkeypatch):
    db = tmp_path / "projects.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE projects (
            id INTEGER PRIMARY KEY, slug TEXT, archived INTEGER DEFAULT 0,
            primary_path TEXT, created_at INTEGER DEFAULT 0
        );
        CREATE TABLE project_folders (project_id INTEGER, path TEXT);
        CREATE TABLE project_meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    erp = tmp_path / "erp-workdir"
    erp.mkdir()
    conn.execute("INSERT INTO projects (id, slug, primary_path) VALUES (1, 'erp', ?)", (str(erp),))
    # a sidebar project is ACTIVE — Hermes' ambient signal, must be ignored here
    conn.execute("INSERT INTO project_meta (key, value) VALUES ('active_id', '1')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(post_llm_call, "PROJECTS_DB", db)
    monkeypatch.setattr(project_catalog, "CATALOG_FILE", tmp_path / "discovered_projects.json")
    return tmp_path


def test_resolve_hermes_anchor_wins(hermes_db):
    # Same folder ⇒ same slug as the Hermes hook — memory must not split.
    assert resolve_project(str(hermes_db / "erp-workdir")) == ("erp", "folder")


def test_resolve_ignores_hermes_active_project(hermes_db):
    # Unanchored cwd: the active sidebar project says nothing about a Claude
    # Code session — fall through to the folder name instead.
    other = hermes_db / "My Tool"
    other.mkdir()
    assert resolve_project(str(other)) == ("my-tool", "folder")
    # Hermes doesn't know this folder — the fallback catalog must, so a
    # Claude-Code-only (no Hermes Desktop) install can still be discovered
    # by the docs/ ingest watcher.
    catalog = json.loads(project_catalog.CATALOG_FILE.read_text())
    assert catalog["my-tool"]["path"] == str(other)


def test_resolve_hermes_anchored_folder_not_recorded_in_catalog(hermes_db):
    # A folder Hermes already knows about doesn't need the fallback catalog.
    resolve_project(str(hermes_db / "erp-workdir"))
    assert not project_catalog.CATALOG_FILE.exists()


def test_resolve_home_and_empty_are_default(hermes_db, monkeypatch):
    import common
    monkeypatch.setattr(common.os.path, "expanduser", lambda p: str(hermes_db / "fakehome"))
    (hermes_db / "fakehome").mkdir()
    assert resolve_project(str(hermes_db / "fakehome")) == ("default", "default")
    assert resolve_project("") == ("default", "default")


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_resolve_git_root_names_the_project(hermes_db):
    repo = hermes_db / "myrepo"
    sub = repo / "src" / "deep"
    sub.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    assert resolve_project(str(sub)) == ("myrepo", "folder")
    catalog = json.loads(project_catalog.CATALOG_FILE.read_text())
    assert catalog["myrepo"]["path"] == str(repo)


def test_resolve_skips_codex_desktop_scratch_dir(hermes_db):
    # Codex Desktop invents ~/Documents/Codex/<date>/<slug-of-title> for
    # ad-hoc chats with no real workspace open — must not become a project.
    scratch = hermes_db / "Documents" / "Codex" / "2026-07-11" / "hi-n-nay-khi-m-s"
    scratch.mkdir(parents=True)
    assert resolve_project(str(scratch)) == ("default", "default")
    assert not project_catalog.CATALOG_FILE.exists()


def test_resolve_real_folder_named_codex_still_works(hermes_db):
    # A genuine project folder that happens to be named "codex" (e.g. a
    # clone of the Codex CLI repo) is a real signal and must still resolve —
    # only the Documents/Codex/<date>/ scratch layout is special-cased.
    real = hermes_db / "codex"
    real.mkdir()
    assert resolve_project(str(real)) == ("codex", "folder")


# ---------------------------------------------------------------------------
# debug logging: opt-in only, truncated, env parsed defensively
# ---------------------------------------------------------------------------
import common  # noqa: E402


def test_debug_dump_writes_nothing_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "DEBUG_LOG", str(tmp_path / "dbg.jsonl"))
    monkeypatch.setattr(common, "DEBUG_HOOKS", False)
    common.debug_dump('{"prompt": "full private prompt"}')
    assert not (tmp_path / "dbg.jsonl").exists()


def test_debug_dump_opt_in_appends_truncated(monkeypatch, tmp_path):
    log = tmp_path / "dbg.jsonl"
    monkeypatch.setattr(common, "DEBUG_LOG", str(log))
    monkeypatch.setattr(common, "DEBUG_HOOKS", True)
    monkeypatch.setattr(common, "DEBUG_MAX_CHARS", 10)
    common.debug_dump("x" * 100)
    assert log.read_text() == "x" * 10 + "\n"


def test_debug_gate_defaults_off_without_env(monkeypatch):
    # Fresh evaluation of the import-time gate: env unset -> disabled.
    monkeypatch.delenv("HERMES_DEBUG_HOOKS", raising=False)
    import importlib
    reloaded = importlib.reload(common)
    assert reloaded.DEBUG_HOOKS is False


def test_env_int_malformed_value_falls_back(monkeypatch):
    monkeypatch.setenv("HERMES_TEST_INT", "abc")
    assert common.env_int("HERMES_TEST_INT", 42) == 42
    monkeypatch.setenv("HERMES_TEST_INT", "7")
    assert common.env_int("HERMES_TEST_INT", 7) == 7
