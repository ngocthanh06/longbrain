from unittest.mock import MagicMock

import pytest

import fetch


def _fake_service(get_result=None, export_result=None, media_result=None):
    service = MagicMock()
    service.files.return_value.get.return_value.execute.return_value = get_result
    service.files.return_value.export.return_value.execute.return_value = export_result
    service.files.return_value.get_media.return_value.execute.return_value = media_result
    return service


def test_fetch_file_exports_google_doc_as_text(monkeypatch):
    service = _fake_service(
        get_result={
            "id": "doc1", "name": "Design Notes",
            "mimeType": fetch.GOOGLE_DOC_MIME_TYPE, "modifiedTime": "2026-01-01T00:00:00Z",
        },
        export_result=b"Doc body text",
    )
    monkeypatch.setattr(fetch, "_service", lambda creds: service)

    result = fetch.fetch_file(creds=object(), file_id="doc1")

    assert result.text == "Doc body text"
    assert result.name == "Design Notes"
    assert result.file_id == "doc1"
    service.files.return_value.export.assert_called_once_with(
        fileId="doc1", mimeType="text/plain"
    )
    service.files.return_value.get_media.assert_not_called()


def test_fetch_file_downloads_plain_drive_file_as_is(monkeypatch):
    service = _fake_service(
        get_result={
            "id": "file1", "name": "notes.md",
            "mimeType": "text/plain", "modifiedTime": "2026-01-01T00:00:00Z",
        },
        media_result=b"raw markdown content",
    )
    monkeypatch.setattr(fetch, "_service", lambda creds: service)

    result = fetch.fetch_file(creds=object(), file_id="file1")

    assert result.text == "raw markdown content"
    service.files.return_value.get_media.assert_called_once_with(fileId="file1")
    service.files.return_value.export.assert_not_called()


def test_fetch_file_downloads_text_markdown_mimetype(monkeypatch):
    service = _fake_service(
        get_result={
            "id": "file2", "name": "spec.md",
            "mimeType": "text/markdown", "modifiedTime": "2026-01-01T00:00:00Z",
        },
        media_result=b"# heading",
    )
    monkeypatch.setattr(fetch, "_service", lambda creds: service)

    result = fetch.fetch_file(creds=object(), file_id="file2")

    assert result.text == "# heading"
    service.files.return_value.get_media.assert_called_once_with(fileId="file2")


def test_fetch_file_rejects_google_sheet(monkeypatch):
    service = _fake_service(
        get_result={
            "id": "sheet1", "name": "Budget",
            "mimeType": fetch.GOOGLE_SHEET_MIME_TYPE, "modifiedTime": "2026-01-01T00:00:00Z",
        },
    )
    monkeypatch.setattr(fetch, "_service", lambda creds: service)

    with pytest.raises(ValueError, match="unsupported mimeType"):
        fetch.fetch_file(creds=object(), file_id="sheet1")
    service.files.return_value.get_media.assert_not_called()


def test_fetch_file_rejects_binary_mimetype_without_downloading(monkeypatch):
    """PDF/DOCX/images etc. must be rejected outright, never downloaded and
    blindly UTF-8-decoded (P2 finding: decode would crash on real binary
    content, or worse, silently ingest garbage on input that happens not to
    raise)."""
    service = _fake_service(
        get_result={
            "id": "pdf1", "name": "report.pdf",
            "mimeType": "application/pdf", "modifiedTime": "2026-01-01T00:00:00Z",
        },
    )
    monkeypatch.setattr(fetch, "_service", lambda creds: service)

    with pytest.raises(ValueError, match="unsupported mimeType"):
        fetch.fetch_file(creds=object(), file_id="pdf1")
    service.files.return_value.get_media.assert_not_called()
    service.files.return_value.export.assert_not_called()
