"""Knowledge base (L4): document ingestion into the Qdrant-backed index.

Original files are kept under DATA_DIR/documents so the knowledge base can
be re-embedded from source when the embedding model changes.
"""

import hashlib
import shutil
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app import config

# Bookkeeping metadata must stay OUT of the text that gets embedded (and out
# of what the LLM sees): every document shares near-identical values here
# (`/data/documents/...` paths, `user_id: local`, ...), so letting LlamaIndex
# prepend them to each chunk before embedding drowns the actual content and
# makes retrieval ranking near-random — verbatim chunk text stopped matching
# its own chunk. `source` (the original filename, often the document's title)
# is deliberately NOT listed: it is real semantic signal for title queries.
EXCLUDED_METADATA_KEYS = [
    "user_id", "project_id", "stored_path", "document_id",
    "file_path", "file_name", "file_type", "file_size",
    "creation_date", "last_modified_date", "last_accessed_date",
]


def _hide_admin_metadata(document: Document) -> None:
    document.excluded_embed_metadata_keys = list(EXCLUDED_METADATA_KEYS)
    document.excluded_llm_metadata_keys = list(EXCLUDED_METADATA_KEYS)


def point_count(qdrant_client: QdrantClient) -> int:
    info = qdrant_client.get_collection(config.DOCUMENTS_COLLECTION)
    return info.points_count


def already_ingested(qdrant_client: QdrantClient, stored_path: str) -> bool:
    """True if a chunk from this exact stored_path (content-addressed by
    store_original, so identical file content -> identical path) is already
    in the index. LlamaIndex flattens metadata onto the top-level payload
    (alongside the serialized `_node_content` it uses to reconstruct nodes),
    so `stored_path` is a plain, indexed field — a cheap filtered lookup.
    Used to make re-running the docs/ watcher a no-op on unchanged files
    instead of piling up duplicate chunks."""
    try:
        points, _ = qdrant_client.scroll(
            collection_name=config.DOCUMENTS_COLLECTION,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="stored_path", match=qmodels.MatchValue(value=stored_path))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(points)
    except Exception:
        return False  # collection doesn't exist yet -> nothing ingested


def store_original(source_path: Path, filename: str) -> Path:
    """Copy an uploaded file into the persistent documents dir, content-addressed
    so re-uploads of the same file do not pile up."""
    documents_dir = Path(config.DOCUMENTS_DIR)
    documents_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()[:12]
    target = documents_dir / f"{digest}_{Path(filename).name}"
    if not target.exists():
        shutil.copyfile(source_path, target)
    return target


def ingest_text(
    index: VectorStoreIndex,
    qdrant_client: QdrantClient,
    text: str,
    metadata: dict | None = None,
    project_id: str = config.DEFAULT_PROJECT,
) -> int:
    document = Document(
        text=text,
        metadata={
            "user_id": config.USER_ID,
            "project_id": project_id or config.DEFAULT_PROJECT,
            **(metadata or {}),
        },
    )
    _hide_admin_metadata(document)
    index.insert(document)
    return point_count(qdrant_client)


def ingest_file(
    index: VectorStoreIndex,
    qdrant_client: QdrantClient,
    file_path: Path,
    metadata: dict | None = None,
    project_id: str = config.DEFAULT_PROJECT,
) -> int:
    documents = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
    for document in documents:
        document.metadata.update(
            {
                "user_id": config.USER_ID,
                "project_id": project_id or config.DEFAULT_PROJECT,
                **(metadata or {}),
            }
        )
        _hide_admin_metadata(document)
        index.insert(document)
    return point_count(qdrant_client)
