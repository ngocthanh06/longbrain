"""Claude Code adapter hooks: transcript extraction and project resolution.
Stdlib-only (the hooks must run with the host python3, which may be 3.9)."""

import json
import shutil
import sqlite3
import subprocess

import pytest

import post_llm_call
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
