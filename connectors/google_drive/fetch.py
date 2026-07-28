"""Drive API reads for the Google Drive connector: file metadata + content.

Scope for this first connector (see docs/STRATEGY.md Phase 2): Google Docs
and plain text/markdown files already sitting in Drive. Google Sheets is a
structurally different shape (tabular, not a document) and is deliberately
out of scope here — see README.md.
"""

from dataclasses import dataclass

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

GOOGLE_DOC_MIME_TYPE = "application/vnd.google-apps.document"
GOOGLE_SHEET_MIME_TYPE = "application/vnd.google-apps.spreadsheet"
# Plain Drive files (not Google-native) this connector accepts as-is. A
# closed whitelist, not a Sheet-only blocklist: an unlisted mimeType (PDF,
# DOCX, images, any other binary) is rejected outright rather than
# downloaded and blindly UTF-8-decoded, which would crash on real binary
# content or, worse, silently ingest decoded garbage on the rare input that
# happens not to raise.
SUPPORTED_PLAIN_MIME_TYPES = {"text/plain", "text/markdown"}

# Deterministic, boilerplate-free export — no per-request timestamp or
# formatting noise. Content-addressed storage (documents.store_original)
# only avoids "duplicate" version churn on an unchanged doc if unchanged
# content actually re-exports to identical bytes every time.
_EXPORT_MIME_TYPE = "text/plain"


@dataclass
class FetchedFile:
    file_id: str
    name: str
    mime_type: str
    modified_time: str  # RFC3339, from Drive — used for change detection
    text: str


def _service(creds: Credentials):
    return build("drive", "v3", credentials=creds)


def get_metadata(creds: Credentials, file_id: str) -> dict:
    return _service(creds).files().get(
        fileId=file_id, fields="id,name,mimeType,modifiedTime"
    ).execute()


def fetch_file(creds: Credentials, file_id: str) -> FetchedFile:
    """Fetch one file's current content + metadata.

    Raises ValueError for anything outside the supported shape — a Google
    Sheet, a PDF/DOCX, or any other mimeType not in SUPPORTED_PLAIN_MIME_TYPES
    — rather than downloading it and blindly decoding as UTF-8 text."""
    meta = get_metadata(creds, file_id)
    mime_type = meta["mimeType"]

    if mime_type == GOOGLE_DOC_MIME_TYPE:
        raw = _service(creds).files().export(
            fileId=file_id, mimeType=_EXPORT_MIME_TYPE
        ).execute()
    elif mime_type in SUPPORTED_PLAIN_MIME_TYPES:
        raw = _service(creds).files().get_media(fileId=file_id).execute()
    else:
        raise ValueError(
            f"{meta['name']!r} ({file_id}) has unsupported mimeType {mime_type!r} — "
            "this connector only handles Google Docs and "
            f"{sorted(SUPPORTED_PLAIN_MIME_TYPES)} files (see README.md)."
        )

    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    return FetchedFile(
        file_id=meta["id"], name=meta["name"], mime_type=mime_type,
        modified_time=meta["modifiedTime"], text=text,
    )
