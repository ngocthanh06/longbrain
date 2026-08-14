"""scripts/ingest_watcher.py — project folder discovery, plus a targeted
auth-wiring check for ingest_file (the rest of the file-poll/HTTP-upload
logic needs a live filesystem + service, exercised manually per README's
"docs/ watcher" section, not here)."""

import json
import sqlite3

import api_auth
import ingest_watcher


def _make_hermes_db(path, rows):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE projects (
            id INTEGER PRIMARY KEY, slug TEXT, archived INTEGER DEFAULT 0,
            primary_path TEXT, created_at INTEGER DEFAULT 0
        );
        CREATE TABLE project_folders (project_id INTEGER, path TEXT);
        """
    )
    for i, (slug, folder_path) in enumerate(rows, start=1):
        conn.execute(
            "INSERT INTO projects (id, slug, primary_path) VALUES (?, ?, ?)",
            (i, slug, folder_path),
        )
    conn.commit()
    conn.close()


def test_list_hermes_project_folders(tmp_path, monkeypatch):
    db = tmp_path / "projects.db"
    _make_hermes_db(db, [("erp", "/work/erp")])
    monkeypatch.setattr(ingest_watcher, "PROJECTS_DB", db)
    assert ingest_watcher.list_hermes_project_folders() == [("erp", "/work/erp")]


def test_list_hermes_project_folders_missing_db(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_watcher, "PROJECTS_DB", tmp_path / "nope.db")
    assert ingest_watcher.list_hermes_project_folders() == []


def test_list_discovered_project_folders(tmp_path, monkeypatch):
    catalog = tmp_path / "discovered_projects.json"
    catalog.write_text(json.dumps({"myrepo": {"path": "/work/myrepo", "last_seen": 1.0}}))
    monkeypatch.setattr(ingest_watcher, "DISCOVERED_PROJECTS_FILE", catalog)
    assert ingest_watcher.list_discovered_project_folders() == [("myrepo", "/work/myrepo")]


def test_merge_combines_both_sources(tmp_path, monkeypatch):
    db = tmp_path / "projects.db"
    _make_hermes_db(db, [("erp", "/work/erp")])
    catalog = tmp_path / "discovered_projects.json"
    catalog.write_text(json.dumps({"myrepo": {"path": "/work/myrepo"}}))
    monkeypatch.setattr(ingest_watcher, "PROJECTS_DB", db)
    monkeypatch.setattr(ingest_watcher, "DISCOVERED_PROJECTS_FILE", catalog)
    assert set(ingest_watcher.list_project_folders()) == {
        ("erp", "/work/erp"),
        ("myrepo", "/work/myrepo"),
    }


def test_merge_hermes_wins_on_slug_collision(tmp_path, monkeypatch):
    db = tmp_path / "projects.db"
    _make_hermes_db(db, [("erp", "/work/erp-hermes-anchored")])
    catalog = tmp_path / "discovered_projects.json"
    catalog.write_text(json.dumps({"erp": {"path": "/work/erp-stale-claude-guess"}}))
    monkeypatch.setattr(ingest_watcher, "PROJECTS_DB", db)
    monkeypatch.setattr(ingest_watcher, "DISCOVERED_PROJECTS_FILE", catalog)
    assert ingest_watcher.list_project_folders() == [("erp", "/work/erp-hermes-anchored")]


def test_merge_works_with_no_hermes_at_all(tmp_path, monkeypatch):
    # The exact scenario this fallback exists for: no ~/.hermes/projects.db.
    catalog = tmp_path / "discovered_projects.json"
    catalog.write_text(json.dumps({"myrepo": {"path": "/work/myrepo"}}))
    monkeypatch.setattr(ingest_watcher, "PROJECTS_DB", tmp_path / "nope.db")
    monkeypatch.setattr(ingest_watcher, "DISCOVERED_PROJECTS_FILE", catalog)
    assert ingest_watcher.list_project_folders() == [("myrepo", "/work/myrepo")]


class _FakeUploadResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_ingest_file_request_carries_the_repo_env_api_key(tmp_path, monkeypatch):
    # launchd runs this script with a near-empty environment (see
    # com.longbrain.memory-ingest.plist.template — no EnvironmentVariables),
    # so process env alone would silently send no header once auth is
    # enabled. Simulate exactly that: no process env key, only a repo .env.
    fake_env = tmp_path / ".env"
    fake_env.write_text("LONGBRAIN_API_KEY=from-dotenv\n")
    monkeypatch.setattr(api_auth, "_REPO_ENV", fake_env)
    monkeypatch.delenv("LONGBRAIN_API_KEY", raising=False)

    captured = {}

    def fake_urlopen(request, timeout=None):
        # Request normalizes header names on storage (title-cases them),
        # so look up case-insensitively rather than assume the exact form.
        headers = {k.lower(): v for k, v in request.header_items()}
        captured["header"] = headers.get("x-api-key")
        return _FakeUploadResponse()

    monkeypatch.setattr(ingest_watcher.urllib.request, "urlopen", fake_urlopen)

    doc = tmp_path / "note.md"
    doc.write_text("hello")
    assert ingest_watcher.ingest_file(doc, "erp", "note.md") is True
    assert captured["header"] == "from-dotenv"
