"""Normalize a fetched Google Drive file into the file-bridge shape that
documents.ingest_file() already understands (see README.md's architecture
note) — no new Document Engine, no NormalizedDocument abstraction, just the
existing `(temp_file, {"source", "document_key"})` contract every ingest
path already speaks.
"""

import tempfile
from pathlib import Path

from fetch import GOOGLE_DOC_MIME_TYPE, FetchedFile


def normalize(fetched: FetchedFile) -> tuple[Path, dict]:
    """(temp_file_path, metadata) ready for store_original() + /ingest/file.

    document_key is the Drive file ID, NOT the display name: a Drive file
    can be renamed without its ID changing, and document_key is what
    documents._supersede_previous_versions() groups versions by. Keying on
    the display name instead would silently fork a new document_key group on
    every rename — the same class of bug the document_key hardening this
    session fixed for the folder connector (basename identity is not
    stable identity).
    """
    suffix = ".md" if fetched.mime_type == GOOGLE_DOC_MIME_TYPE else (
        Path(fetched.name).suffix or ".txt"
    )
    fd, path = tempfile.mkstemp(prefix="google_drive_", suffix=suffix)
    tmp_path = Path(path)
    with open(fd, "w", encoding="utf-8") as f:
        f.write(fetched.text)
    metadata = {
        "document_key": fetched.file_id,
        "source": fetched.name,
    }
    return tmp_path, metadata
