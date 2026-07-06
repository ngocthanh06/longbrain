"""Host-side hook logic: payload extraction and cwd → project resolution.
Stdlib-only (the hooks must run with the host python3)."""

import os
import sqlite3

import pytest

import post_llm_call
from post_llm_call import _extract


# ---------------------------------------------------------------------------
# payload extraction (shape has varied between Hermes builds)
# ---------------------------------------------------------------------------
def test_extract_from_extra():
    sid, user, assistant = _extract(
        {"session_id": "s1", "extra": {"user_message": "u", "assistant_response": "a"}}
    )
    assert (sid, user, assistant) == ("s1", "u", "a")


def test_extract_top_level_aliases():
    sid, user, assistant = _extract(
        {"conversation_id": "c1", "prompt": "u", "completion": "a"}
    )
    assert (sid, user, assistant) == ("c1", "u", "a")


def test_extract_messages_list_fallback():
    payload = {
        "session": "s2",
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ],
    }
    sid, user, assistant = _extract(payload)
    assert (sid, user, assistant) == ("s2", "u", "a")  # newest of each role


def test_extract_empty_payload():
    sid, user, assistant = _extract({})
    assert sid == "unknown"
    assert user == "" and assistant == ""


# ---------------------------------------------------------------------------
# resolve_project: longest prefix, archived, symlinks
# ---------------------------------------------------------------------------
@pytest.fixture()
def projects_db(tmp_path, monkeypatch):
    db = tmp_path / "projects.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE projects (
            id INTEGER PRIMARY KEY, slug TEXT, archived INTEGER DEFAULT 0,
            primary_path TEXT
        );
        CREATE TABLE project_folders (project_id INTEGER, path TEXT);
        """
    )
    work = tmp_path / "work"
    erp = work / "erp"
    erp.mkdir(parents=True)
    old = tmp_path / "old"
    old.mkdir()
    conn.executemany(
        "INSERT INTO projects (id, slug, archived, primary_path) VALUES (?, ?, ?, ?)",
        [
            (1, "work", 0, None),
            (2, "erp", 0, None),
            (3, "old", 1, str(old)),  # archived: must never match
        ],
    )
    conn.executemany(
        "INSERT INTO project_folders (project_id, path) VALUES (?, ?)",
        [(1, str(work)), (2, str(erp))],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(post_llm_call, "PROJECTS_DB", db)
    return tmp_path


def test_resolve_inside_folder(projects_db):
    assert post_llm_call.resolve_project(str(projects_db / "work" / "docs")) == "work"


def test_resolve_child_project_beats_parent(projects_db):
    assert post_llm_call.resolve_project(str(projects_db / "work" / "erp" / "src")) == "erp"


def test_resolve_unmatched_is_default(projects_db):
    assert post_llm_call.resolve_project(str(projects_db / "elsewhere")) == "default"


def test_resolve_archived_excluded(projects_db):
    assert post_llm_call.resolve_project(str(projects_db / "old")) == "default"


def test_resolve_prefix_is_path_aware_not_string_prefix(projects_db):
    # /work-other must NOT match the /work project.
    other = projects_db / "work-other"
    other.mkdir()
    assert post_llm_call.resolve_project(str(other)) == "default"


def test_resolve_cwd_through_symlink(projects_db):
    link = projects_db / "link-to-erp"
    os.symlink(projects_db / "work" / "erp", link)
    assert post_llm_call.resolve_project(str(link)) == "erp"


def test_resolve_missing_db_or_empty_cwd(monkeypatch, tmp_path):
    monkeypatch.setattr(post_llm_call, "PROJECTS_DB", tmp_path / "missing.db")
    assert post_llm_call.resolve_project("/anywhere") == "default"
    assert post_llm_call.resolve_project("") == "default"
