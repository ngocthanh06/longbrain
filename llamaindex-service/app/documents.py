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

from app import config, hybrid

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
    "enriched",  # bookkeeping flag on AI-summary chunks (app/enrich.py)
]


def _hide_admin_metadata(document: Document) -> None:
    document.excluded_embed_metadata_keys = list(EXCLUDED_METADATA_KEYS)
    document.excluded_llm_metadata_keys = list(EXCLUDED_METADATA_KEYS)


def point_count(qdrant_client: QdrantClient) -> int:
    info = qdrant_client.get_collection(config.DOCUMENTS_COLLECTION)
    return info.points_count


def already_ingested(qdrant_client: QdrantClient, stored_path: str, project_id: str = "") -> bool:
    """True if a chunk from this exact stored_path (content-addressed by
    store_original, so identical file content -> identical path) is already
    in the index. LlamaIndex flattens metadata onto the top-level payload
    (alongside the serialized `_node_content` it uses to reconstruct nodes),
    so `stored_path` is a plain, indexed field — a cheap filtered lookup.
    Used to make re-running the docs/ watcher a no-op on unchanged files
    instead of piling up duplicate chunks.

    `project_id` is required to avoid a cross-project false positive: two
    projects with docs/ folders that happen to contain byte-identical files
    (e.g. both have a README.md with the same boilerplate) share the same
    content-addressed stored_path, so without this filter the second
    project's ingest would be skipped as a "duplicate" of the first
    project's chunks and end up with zero chunks of its own."""
    try:
        must = [qmodels.FieldCondition(key="stored_path", match=qmodels.MatchValue(value=stored_path))]
        if project_id:
            must.append(qmodels.FieldCondition(key="project_id", match=qmodels.MatchValue(value=project_id)))
        points, _ = qdrant_client.scroll(
            collection_name=config.DOCUMENTS_COLLECTION,
            scroll_filter=qmodels.Filter(must=must),
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


def _backfill_sparse(qdrant_client: QdrantClient, doc_ids: list[str]) -> None:
    """LlamaIndex's insert writes only the unnamed dense vector; add the BM25
    sparse vector to the chunks it just created. No-op when the collection
    has no sparse schema (pre-migration) or hybrid is disabled."""
    if not doc_ids or not hybrid.collection_enabled(qdrant_client, config.DOCUMENTS_COLLECTION):
        return
    import json as _json

    flt = qmodels.Filter(must=[
        qmodels.FieldCondition(key="doc_id", match=qmodels.MatchAny(any=list(doc_ids)))
    ])
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=config.DOCUMENTS_COLLECTION,
            scroll_filter=flt,
            limit=128,
            offset=offset,
            with_payload=["_node_content"],
            with_vectors=False,
        )
        updates = []
        for p in points:
            try:
                text = _json.loads((p.payload or {}).get("_node_content") or "{}").get("text", "")
            except (ValueError, TypeError):
                text = ""
            sparse = hybrid.text_vector(text)
            if sparse is not None:
                updates.append(qmodels.PointVectors(
                    id=p.id, vector={config.BM25_VECTOR_NAME: sparse}
                ))
        if updates:
            qdrant_client.update_vectors(
                collection_name=config.DOCUMENTS_COLLECTION, points=updates
            )
        if offset is None:
            break


def ingest_text(
    index: VectorStoreIndex,
    qdrant_client: QdrantClient,
    text: str,
    metadata: dict | None = None,
    project_id: str = config.DEFAULT_PROJECT,
) -> int:
    document = Document(
        text=text,
        # user_id/project_id are the partitioning invariant — they must win
        # over caller-supplied metadata, not be overridable by it.
        metadata={
            **(metadata or {}),
            "user_id": config.USER_ID,
            "project_id": project_id or config.DEFAULT_PROJECT,
        },
    )
    _hide_admin_metadata(document)
    index.insert(document)
    _backfill_sparse(qdrant_client, [document.doc_id])
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
        # user_id/project_id are the partitioning invariant — they must win
        # over caller-supplied metadata, not be overridable by it.
        document.metadata.update(
            {
                **(metadata or {}),
                "user_id": config.USER_ID,
                "project_id": project_id or config.DEFAULT_PROJECT,
            }
        )
        _hide_admin_metadata(document)
        index.insert(document)
    _backfill_sparse(qdrant_client, [d.doc_id for d in documents])
    return point_count(qdrant_client)


def search_chunks(
    client: QdrantClient,
    embed_model,
    query: str,
    project: str | None = None,
    top_k: int = config.RECALL_TOP_K_DOCS,
    min_score: float = config.RECALL_MIN_SCORE,
) -> list[dict]:
    """Lightweight L4 lookup for the recall router: nearest document chunks,
    hard-filtered to the project (documents are project-scoped by design).
    Reads the chunk text out of the serialized `_node_content` directly so
    recall() doesn't need the LlamaIndex index object."""
    import json as _json

    vector = embed_model.get_text_embedding(query)
    must = []
    if project:
        must.append(
            qmodels.FieldCondition(key="project_id", match=qmodels.MatchValue(value=project))
        )
    flt = qmodels.Filter(must=must) if must else None
    try:
        dense_hits = client.search(
            collection_name=config.DOCUMENTS_COLLECTION,
            query_vector=vector,
            query_filter=flt,
            limit=top_k,
            score_threshold=min_score,
            with_payload=["_node_content", "source", "project_id"],
        )
    except Exception:
        return []  # collection missing/empty — recall stays best-effort
    sparse_hits = hybrid.search(
        client, config.DOCUMENTS_COLLECTION, query, flt, top_k,
        with_payload=["_node_content", "source", "project_id"],
    )
    results = []
    for e in hybrid.fuse(dense_hits, sparse_hits):
        if e["similarity"] < min_score:
            continue  # sparse candidates skip the server-side threshold
        raw = e["payload"].get("_node_content")
        if not raw:
            continue
        try:
            text = _json.loads(raw).get("text", "")
        except (ValueError, TypeError):
            continue
        results.append({
            "source": e["payload"].get("source") or "",
            "project_id": e["payload"].get("project_id") or "",
            "text": text,
            "score": e["similarity"],
        })
    return results[:top_k]
