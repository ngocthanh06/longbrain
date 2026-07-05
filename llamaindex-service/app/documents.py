"""Knowledge base (L4): document ingestion into the Qdrant-backed index.

Original files are kept under DATA_DIR/documents so the knowledge base can
be re-embedded from source when the embedding model changes.
"""

import hashlib
import shutil
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from qdrant_client import QdrantClient

from app import config


def _collection_point_count(qdrant_client: QdrantClient) -> int:
    info = qdrant_client.get_collection(config.DOCUMENTS_COLLECTION)
    return info.points_count


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
    index.insert(document)
    return _collection_point_count(qdrant_client)


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
        index.insert(document)
    return _collection_point_count(qdrant_client)
