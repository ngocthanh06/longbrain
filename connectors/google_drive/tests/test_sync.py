import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import sync
from fetch import FetchedFile


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Every test gets its own sources/state files — never touch the real
    ~/.longbrain/ directory."""
    monkeypatch.setattr(sync, "SOURCES_FILE", tmp_path / "sources.json")
    monkeypatch.setattr(sync, "STATE_FILE", tmp_path / "state.json")
    return tmp_path


# ---------------------------------------------------------------------------
# extract_file_id
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [
    ("1AbCdEfGhIjKlMnOpQrStUvWxYz", "1AbCdEfGhIjKlMnOpQrStUvWxYz"),
    ("https://docs.google.com/document/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/edit",
     "1AbCdEfGhIjKlMnOpQrStUvWxYz"),
    ("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing",
     "1AbCdEfGhIjKlMnOpQrStUvWxYz"),
])
def test_extract_file_id(value, expected):
    assert sync.extract_file_id(value) == expected


# ---------------------------------------------------------------------------
# add_source / load_sources
# ---------------------------------------------------------------------------
def test_add_source_persists_and_is_idempotent():
    file_id = "doc123456789012345"  # realistic length (regex requires >=10 chars)
    sync.add_source(f"https://docs.google.com/document/d/{file_id}/edit", "erp")
    sync.add_source(file_id, "erp")  # same id, bare form — must not duplicate

    sources = sync.load_sources()
    assert sources == [{"file_id": file_id, "project_id": "erp"}]


def test_add_source_multiple_distinct():
    sync.add_source("doc1", "erp")
    sync.add_source("doc2", "gakken")
    assert {s["file_id"] for s in sync.load_sources()} == {"doc1", "doc2"}


# ---------------------------------------------------------------------------
# sync_one: skip / sent / failed decision logic
# ---------------------------------------------------------------------------
def _fetched(file_id="doc1", text="content", modified="2026-02-01T00:00:00Z"):
    return FetchedFile(
        file_id=file_id, name="doc.md", mime_type="application/vnd.google-apps.document",
        modified_time=modified, text=text,
    )


def test_sync_one_skips_when_modified_time_unchanged(monkeypatch):
    monkeypatch.setattr(sync, "get_metadata", lambda creds, fid: {"modifiedTime": "T1"})
    fetch_mock = MagicMock()
    monkeypatch.setattr(sync, "fetch_file", fetch_mock)
    state = {"doc1": "T1"}

    result = sync.sync_one(None, {"file_id": "doc1", "project_id": "erp"}, state, force=False)

    assert result == "skipped"
    fetch_mock.assert_not_called()


def test_sync_one_force_refetches_even_when_unchanged(monkeypatch):
    monkeypatch.setattr(sync, "get_metadata", lambda creds, fid: {"modifiedTime": "T1"})
    monkeypatch.setattr(sync, "fetch_file", lambda creds, fid: _fetched(modified="T1"))
    monkeypatch.setattr(sync, "post_ingest_file", lambda tmp, proj, meta: True)
    state = {"doc1": "T1"}

    result = sync.sync_one(None, {"file_id": "doc1", "project_id": "erp"}, state, force=True)

    assert result == "sent"


def test_sync_one_sends_and_updates_state_when_changed(monkeypatch):
    monkeypatch.setattr(sync, "get_metadata", lambda creds, fid: {"modifiedTime": "T2"})
    monkeypatch.setattr(sync, "fetch_file", lambda creds, fid: _fetched(modified="T2"))
    posted = {}
    monkeypatch.setattr(
        sync, "post_ingest_file",
        lambda tmp, proj, meta: posted.update(project_id=proj, metadata=meta) or True,
    )
    state = {"doc1": "T1"}

    result = sync.sync_one(None, {"file_id": "doc1", "project_id": "erp"}, state, force=False)

    assert result == "sent"
    assert state["doc1"] == "T2"
    assert posted["project_id"] == "erp"
    assert posted["metadata"]["document_key"] == "doc1"


def test_sync_one_does_not_update_state_on_post_failure(monkeypatch):
    monkeypatch.setattr(sync, "get_metadata", lambda creds, fid: {"modifiedTime": "T2"})
    monkeypatch.setattr(sync, "fetch_file", lambda creds, fid: _fetched(modified="T2"))
    monkeypatch.setattr(sync, "post_ingest_file", lambda tmp, proj, meta: False)
    state = {"doc1": "T1"}

    result = sync.sync_one(None, {"file_id": "doc1", "project_id": "erp"}, state, force=False)

    assert result == "failed"
    assert state["doc1"] == "T1"  # unchanged — safe to retry next pass


def test_sync_one_treats_unsupported_shape_as_skip_not_failure(monkeypatch):
    """A Google Sheet raises ValueError from fetch_file — that is a
    permanent 'not supported', not a transient failure to keep retrying."""
    monkeypatch.setattr(sync, "get_metadata", lambda creds, fid: {"modifiedTime": "T2"})

    def raise_unsupported(creds, fid):
        raise ValueError("is a Google Sheet")

    monkeypatch.setattr(sync, "fetch_file", raise_unsupported)
    state = {}

    result = sync.sync_one(None, {"file_id": "sheet1", "project_id": "erp"}, state, force=False)

    assert result == "skipped"
    assert "sheet1" not in state


def test_sync_one_fails_on_metadata_error(monkeypatch):
    def raise_error(creds, fid):
        raise RuntimeError("network down")

    monkeypatch.setattr(sync, "get_metadata", raise_error)
    result = sync.sync_one(None, {"file_id": "doc1", "project_id": "erp"}, {}, force=False)
    assert result == "failed"


# ---------------------------------------------------------------------------
# post_ingest_file: multipart body shape
# ---------------------------------------------------------------------------
def test_post_ingest_file_sends_project_id_and_document_key(monkeypatch, tmp_path):
    tmp_file = tmp_path / "doc.md"
    tmp_file.write_text("hello")

    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout):
        captured["body"] = request.data
        captured["headers"] = dict(request.header_items())
        return FakeResponse()

    monkeypatch.setattr(sync.urllib.request, "urlopen", fake_urlopen)

    ok = sync.post_ingest_file(tmp_file, "erp", {"document_key": "doc1", "source": "doc.md"})

    assert ok is True
    body = captured["body"].decode()
    assert 'name="project_id"' in body and "erp" in body
    assert '"document_key": "doc1"' in body
    assert "hello" in body


def test_post_ingest_file_returns_false_on_error(monkeypatch, tmp_path):
    tmp_file = tmp_path / "doc.md"
    tmp_file.write_text("hello")

    def fake_urlopen(request, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(sync.urllib.request, "urlopen", fake_urlopen)

    assert sync.post_ingest_file(tmp_file, "erp", {}) is False
