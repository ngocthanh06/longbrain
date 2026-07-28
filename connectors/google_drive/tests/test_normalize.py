from pathlib import Path

from fetch import GOOGLE_DOC_MIME_TYPE, FetchedFile
from normalize import normalize


def test_document_key_is_file_id_not_name():
    """The whole point of keying on file_id: renaming the Drive file must
    not fork a new document_key group (see normalize.py's docstring)."""
    fetched = FetchedFile(
        file_id="abc123", name="My Renamed Doc Title.md",
        mime_type=GOOGLE_DOC_MIME_TYPE, modified_time="2026-01-01T00:00:00Z",
        text="hello world",
    )
    tmp_path, metadata = normalize(fetched)
    try:
        assert metadata["document_key"] == "abc123"
        assert metadata["source"] == "My Renamed Doc Title.md"
    finally:
        tmp_path.unlink(missing_ok=True)


def test_normalize_writes_fetched_text_to_temp_file():
    fetched = FetchedFile(
        file_id="doc1", name="notes.md", mime_type=GOOGLE_DOC_MIME_TYPE,
        modified_time="2026-01-01T00:00:00Z", text="line one\nline two",
    )
    tmp_path, _ = normalize(fetched)
    try:
        assert tmp_path.read_text(encoding="utf-8") == "line one\nline two"
        assert tmp_path.suffix == ".md"
    finally:
        tmp_path.unlink(missing_ok=True)


def test_normalize_preserves_extension_for_plain_drive_files():
    fetched = FetchedFile(
        file_id="file1", name="readme.txt", mime_type="text/plain",
        modified_time="2026-01-01T00:00:00Z", text="plain content",
    )
    tmp_path, metadata = normalize(fetched)
    try:
        assert tmp_path.suffix == ".txt"
        assert metadata["document_key"] == "file1"
    finally:
        tmp_path.unlink(missing_ok=True)
