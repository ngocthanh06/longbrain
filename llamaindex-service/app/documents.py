"""Knowledge base (L4): document ingestion into the Qdrant-backed index.

Original files are kept under DATA_DIR/documents so the knowledge base can
be re-embedded from source when the embedding model changes.
"""

import hashlib
import math
import shutil
import time
from pathlib import Path

import requests
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
    "ingested_at",  # bookkeeping timestamp for recency decay in search_chunks
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
            "ingested_at": time.time(),
        },
    )
    _hide_admin_metadata(document)
    index.insert(document)
    _backfill_sparse(qdrant_client, [document.doc_id])
    return point_count(qdrant_client)


def _supersede_previous_versions(
    qdrant_client: QdrantClient,
    project_id: str,
    document_key: str,
    new_stored_path: str,
) -> int:
    """Mark earlier versions of the same logical document as superseded so
    search excludes them by default (mirrors memories.py's superseded_by
    pattern instead of deleting anything).

    Scope is deliberately narrow: only chunks that themselves carry a
    `stored_path` (i.e. produced by ingest_file(), never by ingest_text())
    are eligible. Enrichment summary chunks (`enriched: true`) and manually
    added text (add_to_knowledge_base) have no `stored_path` and no version
    concept, so they are excluded by construction and never touched here."""
    if not document_key:
        return 0
    must = [
        qmodels.FieldCondition(key="project_id", match=qmodels.MatchValue(value=project_id)),
        qmodels.FieldCondition(key="document_key", match=qmodels.MatchValue(value=document_key)),
    ]
    must_not = [
        qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="stored_path")),
        qmodels.FieldCondition(key="stored_path", match=qmodels.MatchValue(value=new_stored_path)),
    ]
    stale_ids: list = []
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=config.DOCUMENTS_COLLECTION,
            scroll_filter=qmodels.Filter(must=must, must_not=must_not),
            limit=128, offset=offset, with_payload=False, with_vectors=False,
        )
        stale_ids.extend(p.id for p in points)
        if offset is None:
            break
    if stale_ids:
        qdrant_client.set_payload(
            collection_name=config.DOCUMENTS_COLLECTION,
            payload={"superseded_by": new_stored_path},
            points=stale_ids,
        )
    return len(stale_ids)


def tag_existing_version(
    qdrant_client: QdrantClient,
    project_id: str,
    stored_path: str,
    document_key: str,
) -> int:
    """Backfill the stable key onto legacy chunks for the same content.

    The first post-fix watcher pass may encounter a content-addressed chunk
    created before ``document_key`` existed. Tagging it lets the normal
    supersession pass compare it safely with the new version without using
    the ambiguous basename/source field.
    """
    if not (project_id and stored_path and document_key):
        return 0
    flt = qmodels.Filter(must=[
        qmodels.FieldCondition(key="project_id", match=qmodels.MatchValue(value=project_id)),
        qmodels.FieldCondition(key="stored_path", match=qmodels.MatchValue(value=stored_path)),
    ])
    points, offset = [], None
    while True:
        batch, offset = qdrant_client.scroll(
            collection_name=config.DOCUMENTS_COLLECTION,
            scroll_filter=flt,
            limit=128,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        points.extend(p.id for p in batch)
        if offset is None:
            break
    if points:
        qdrant_client.set_payload(
            collection_name=config.DOCUMENTS_COLLECTION,
            payload={"document_key": document_key},
            points=points,
        )
    return len(points)


def delete_document(qdrant_client: QdrantClient, project_id: str, document_key: str) -> dict:
    """Hard-delete every chunk (active AND superseded) for exactly this
    (project_id, document_key) pair — unlike _supersede_previous_versions,
    which only marks old chunks hidden, this actually removes them from
    Qdrant. Also removes the original file copy under DOCUMENTS_DIR for any
    stored_path no longer referenced by any remaining chunk (in any project
    or under any other document_key), so a shared/duplicate upload isn't
    deleted out from under a still-live document.

    Never touches the source system the content came from (e.g. Google
    Drive) — this only clears Longbrain's own copy of the data."""
    if not (project_id and document_key):
        return {"chunks_deleted": 0, "files_removed": 0}
    must = [
        qmodels.FieldCondition(key="project_id", match=qmodels.MatchValue(value=project_id)),
        qmodels.FieldCondition(key="document_key", match=qmodels.MatchValue(value=document_key)),
    ]
    points = []
    offset = None
    while True:
        batch, offset = qdrant_client.scroll(
            collection_name=config.DOCUMENTS_COLLECTION,
            scroll_filter=qmodels.Filter(must=must),
            limit=256, offset=offset, with_payload=["stored_path"], with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    if not points:
        return {"chunks_deleted": 0, "files_removed": 0}

    stored_paths = {p.payload.get("stored_path") for p in points if p.payload.get("stored_path")}
    qdrant_client.delete(
        collection_name=config.DOCUMENTS_COLLECTION,
        points_selector=qmodels.PointIdsList(points=[p.id for p in points]),
    )

    files_removed = 0
    for stored_path in stored_paths:
        still_referenced = qdrant_client.count(
            collection_name=config.DOCUMENTS_COLLECTION,
            count_filter=qmodels.Filter(must=[
                qmodels.FieldCondition(key="stored_path", match=qmodels.MatchValue(value=stored_path)),
            ]),
            exact=True,
        ).count
        if still_referenced == 0:
            try:
                Path(stored_path).unlink(missing_ok=True)
                files_removed += 1
            except OSError:
                pass
    return {"chunks_deleted": len(points), "files_removed": files_removed}


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
                "ingested_at": time.time(),
            }
        )
        _hide_admin_metadata(document)
        index.insert(document)
    _backfill_sparse(qdrant_client, [d.doc_id for d in documents])
    document_key = (metadata or {}).get("document_key")
    stored_path = (metadata or {}).get("stored_path")
    if document_key and stored_path:
        tag_existing_version(
            qdrant_client,
            project_id or config.DEFAULT_PROJECT,
            stored_path,
            document_key,
        )
        _supersede_previous_versions(
            qdrant_client, project_id or config.DEFAULT_PROJECT, document_key, stored_path
        )
    return point_count(qdrant_client)


def _decay(age_seconds: float, half_life_days: float) -> float:
    return math.pow(0.5, age_seconds / (half_life_days * 86400.0))


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
    recall() doesn't need the LlamaIndex index object.

    Results are ranked by similarity decayed on `ingested_at` age (see
    config.DOC_HALF_LIFE_DAYS) so a stale chunk doesn't keep outranking a
    newer one at the same similarity; chunks ingested before this field
    existed fall back to age 0 (no penalty)."""
    import json as _json

    vector = embed_model.get_text_embedding(query)
    must = [qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="superseded_by"))]
    if project:
        must.append(
            qmodels.FieldCondition(key="project_id", match=qmodels.MatchValue(value=project))
        )
    flt = qmodels.Filter(must=must)
    payload_fields = [
        "_node_content", "source", "project_id", "document_key",
        "stored_path", "ingested_at",
    ]
    try:
        dense_hits = client.search(
            collection_name=config.DOCUMENTS_COLLECTION,
            query_vector=vector,
            query_filter=flt,
            limit=top_k,
            score_threshold=min_score,
            with_payload=payload_fields,
        )
    except Exception:
        return []  # collection missing/empty — recall stays best-effort
    sparse_hits = hybrid.search(
        client, config.DOCUMENTS_COLLECTION, query, flt, top_k,
        with_payload=payload_fields,
    )
    now = time.time()
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
        age = max(now - (e["payload"].get("ingested_at") or now), 0.0)
        decay_factor = _decay(age, config.DOC_HALF_LIFE_DAYS)
        results.append({
            "source": e["payload"].get("source") or "",
            "project_id": e["payload"].get("project_id") or "",
            "document_key": e["payload"].get("document_key") or "",
            "stored_path": e["payload"].get("stored_path") or "",
            "point_id": str(e["id"]),
            "text": text,
            "score": e["similarity"] * decay_factor,
            # Traceability (why this chunk was surfaced/ranked here):
            # decomposed factors behind `score`, not shown in context_block.
            "trace": {
                "similarity": e["similarity"],
                "decay_factor": decay_factor,
                "point_id": str(e["id"]),
                "document_key": e["payload"].get("document_key") or "",
                "stored_path": e["payload"].get("stored_path") or "",
            },
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def federated_search_chunks(
    query: str,
    project: str | None = None,
    top_k: int = config.RECALL_TOP_K_DOCS,
) -> list[dict]:
    """Document hits from Connector Layer's own backend (see
    config.CONNECTOR_SEARCH_URL) via its /documents/search endpoint, tagged
    origin="connector-layer" so a reader can tell them apart from LongBrain's
    own KB. Disabled (returns []) when CONNECTOR_SEARCH_URL is unset, and on
    ANY failure/timeout — this must never fail or stall recall() (same
    fail-open shape as app/enrich.py's llm_available()). Deliberately an
    HTTP call, not a direct Qdrant query against connector_layer_documents:
    the connector backend owns its own embedder/collection/schema and may
    diverge from LongBrain's own at any point, so LongBrain must go through
    its search API rather than assume a shared vector space."""
    if not config.CONNECTOR_SEARCH_URL:
        return []
    try:
        resp = requests.post(
            f"{config.CONNECTOR_SEARCH_URL}/documents/search",
            json={"query": query, "project": project or "", "top_k": top_k},
            timeout=config.CONNECTOR_SEARCH_TIMEOUT_MS / 1000.0,
        )
        resp.raise_for_status()
        hits = resp.json().get("results", [])
    except Exception:  # noqa: BLE001 — unreachable/slow/malformed = no federated results
        return []
    for hit in hits:
        hit["origin"] = "connector-layer"
    return hits
