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
            primary_path TEXT, created_at INTEGER DEFAULT 0
        );
        CREATE TABLE project_folders (project_id INTEGER, path TEXT);
        CREATE TABLE project_meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    work = tmp_path / "work"
    erp = work / "erp"
    erp.mkdir(parents=True)
    old = tmp_path / "old"
    old.mkdir()
    conn.executemany(
        "INSERT INTO projects (id, slug, archived, primary_path, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        [
            (1, "work", 0, None, 100),
            (2, "erp", 0, None, 200),
            (3, "old", 1, str(old), 300),  # archived: must never match
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


def _db(projects_db):
    return sqlite3.connect(projects_db / "projects.db")


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


def test_resolve_shared_folder_newest_project_wins(projects_db):
    # Two projects anchored to the SAME folder: the most recently created
    # one wins deterministically (not SQL row order).
    conn = _db(projects_db)
    conn.execute(
        "INSERT INTO projects (id, slug, archived, primary_path, created_at)"
        " VALUES (4, 'work2', 0, NULL, 999)"
    )
    conn.execute(
        "INSERT INTO project_folders (project_id, path) VALUES (4, ?)",
        (str(projects_db / "work"),),
    )
    conn.commit(); conn.close()
    assert post_llm_call.resolve_project(str(projects_db / "work" / "x")) == "work2"


def test_resolve_falls_back_to_active_sidebar_project(projects_db):
    conn = _db(projects_db)
    conn.execute("INSERT INTO project_meta (key, value) VALUES ('active_id', '2')")
    conn.commit(); conn.close()
    # cwd matches no folder → the project selected in the sidebar wins.
    assert post_llm_call.resolve_project(str(projects_db / "elsewhere")) == "erp"
    # empty cwd also falls back to the active project
    assert post_llm_call.resolve_project("") == "erp"


def test_resolve_folder_match_beats_active_project(projects_db):
    conn = _db(projects_db)
    conn.execute("INSERT INTO project_meta (key, value) VALUES ('active_id', '2')")
    conn.commit(); conn.close()
    # cwd inside `work`: the per-chat folder signal outranks the global selection.
    assert post_llm_call.resolve_project(str(projects_db / "work" / "docs")) == "work"


def test_resolve_archived_active_project_ignored(projects_db):
    conn = _db(projects_db)
    conn.execute("INSERT INTO project_meta (key, value) VALUES ('active_id', '3')")
    conn.commit(); conn.close()
    assert post_llm_call.resolve_project(str(projects_db / "elsewhere")) == "default"


def test_resolve_source_reflects_signal(projects_db):
    slug, src = post_llm_call.resolve_project_with_source(str(projects_db / "work"))
    assert (slug, src) == ("work", "folder")
    slug, src = post_llm_call.resolve_project_with_source(str(projects_db / "elsewhere"))
    assert (slug, src) == ("default", "default")
    conn = _db(projects_db)
    conn.execute("INSERT INTO project_meta (key, value) VALUES ('active_id', '2')")
    conn.commit(); conn.close()
    slug, src = post_llm_call.resolve_project_with_source(str(projects_db / "elsewhere"))
    assert (slug, src) == ("erp", "active")


def test_resolve_without_project_meta_table(projects_db):
    conn = _db(projects_db)
    conn.execute("DROP TABLE project_meta")
    conn.commit(); conn.close()
    # older Hermes without project_meta: folder matching keeps working.
    assert post_llm_call.resolve_project(str(projects_db / "work")) == "work"
    assert post_llm_call.resolve_project(str(projects_db / "elsewhere")) == "default"
