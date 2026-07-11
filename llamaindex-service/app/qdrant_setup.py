"""Startup provisioning: collections, payload indexes and the schema guard.

The meta collection stores which embedding model produced the vectors on
disk. On boot we compare it with the current config and refuse to start on a
mismatch — mixing vector spaces silently corrupts recall, so it must be an
explicit migration instead.
"""

import logging
import time

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app import config

logger = logging.getLogger("uvicorn")

META_POINT_ID = "00000000-0000-0000-0000-000000000001"


def _existing_collections(client: QdrantClient) -> set[str]:
    return {c.name for c in client.get_collections().collections}


def _collection_vector_size(client: QdrantClient, name: str) -> int | None:
    info = client.get_collection(name)
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):  # named vectors
        vectors = next(iter(vectors.values()), None)
    return getattr(vectors, "size", None)


def _ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    if name not in _existing_collections(client):
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=dim, distance=qmodels.Distance.COSINE
            ),
            # The dense vector stays unnamed; BM25 rides along as a named
            # sparse vector (see app/hybrid.py). Qdrant cannot add a sparse
            # vector to an existing collection, so it must be born here —
            # pre-existing collections need scripts/migrate_hybrid_bm25.py.
            sparse_vectors_config=(
                {config.BM25_VECTOR_NAME: qmodels.SparseVectorParams(
                    modifier=qmodels.Modifier.IDF)}
                if config.HYBRID_BM25 else None
            ),
        )
    elif config.HYBRID_BM25:
        params = client.get_collection(name).config.params
        if config.BM25_VECTOR_NAME not in (params.sparse_vectors or {}):
            logger.warning(
                "Collection %s has no %r sparse vector — hybrid BM25 recall is "
                "OFF for it. Run scripts/migrate_hybrid_bm25.py to enable.",
                name, config.BM25_VECTOR_NAME,
            )


def _ensure_indexes(client: QdrantClient, name: str, fields: dict[str, qmodels.PayloadSchemaType]) -> None:
    for field, schema in fields.items():
        try:
            client.create_payload_index(
                collection_name=name, field_name=field, field_schema=schema
            )
        except Exception:
            pass  # already exists


def get_meta(client: QdrantClient) -> dict | None:
    if config.META_COLLECTION not in _existing_collections(client):
        return None
    points = client.retrieve(
        collection_name=config.META_COLLECTION, ids=[META_POINT_ID], with_payload=True
    )
    return points[0].payload if points else None


def touch_meta(client: QdrantClient) -> float:
    """Record the last successful memory write, so clients can detect a
    silently-broken write path (e.g. a hook that stopped firing)."""
    now = time.time()
    client.set_payload(
        collection_name=config.META_COLLECTION,
        payload={"last_written_at": now},
        points=[META_POINT_ID],
    )
    return now


def ensure_all(client: QdrantClient, embed_dim: int) -> None:
    meta = get_meta(client)
    if meta is not None:
        stored_model = meta.get("embed_model")
        stored_dim = meta.get("embed_dim")
        if stored_model != config.EMBED_MODEL or stored_dim != embed_dim:
            raise RuntimeError(
                "Embedding mismatch: data on disk was created with "
                f"{stored_model!r} (dim={stored_dim}) but the service is "
                f"configured with {config.EMBED_MODEL!r} (dim={embed_dim}). "
                "Either restore the previous EMBED_PROVIDER/EMBED_MODEL, or "
                "start fresh (docker compose down -v) / re-embed into new "
                "collections before switching."
            )

    keyword = qmodels.PayloadSchemaType.KEYWORD
    ts = qmodels.PayloadSchemaType.FLOAT

    _ensure_collection(client, config.CHAT_HISTORY_COLLECTION, embed_dim)
    _ensure_indexes(
        client,
        config.CHAT_HISTORY_COLLECTION,
        {"user_id": keyword, "session_id": keyword, "role": keyword,
         "project_id": keyword, "timestamp": ts,
         "consolidated": qmodels.PayloadSchemaType.BOOL},
    )

    _ensure_collection(client, config.MEMORIES_COLLECTION, embed_dim)
    _ensure_indexes(
        client,
        config.MEMORIES_COLLECTION,
        {"user_id": keyword, "session_id": keyword, "type": keyword,
         "project_id": keyword, "created_at": ts, "last_seen": ts,
         "superseded_by": keyword, "status": keyword,
         "triple_subject": keyword, "triple_relation": keyword},
    )

    # The documents collection is created here too (LlamaIndex's
    # QdrantVectorStore would otherwise create it on first insert WITHOUT the
    # sparse vector, permanently locking hybrid out of L4). Metadata keys
    # land directly in the Qdrant payload, so the indexes work the same.
    _ensure_collection(client, config.DOCUMENTS_COLLECTION, embed_dim)
    doc_dim = _collection_vector_size(client, config.DOCUMENTS_COLLECTION)
    if doc_dim is not None and doc_dim != embed_dim:
        raise RuntimeError(
            f"Collection {config.DOCUMENTS_COLLECTION!r} holds "
            f"{doc_dim}-dim vectors but the configured embedding produces "
            f"{embed_dim}-dim. See the embedding-migration note in README."
        )
    _ensure_indexes(
        client, config.DOCUMENTS_COLLECTION,
        {"project_id": keyword, "user_id": keyword, "stored_path": keyword},
    )

    if config.META_COLLECTION not in _existing_collections(client):
        client.create_collection(
            collection_name=config.META_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=1, distance=qmodels.Distance.DOT
            ),
        )
    client.upsert(
        collection_name=config.META_COLLECTION,
        points=[
            qmodels.PointStruct(
                id=META_POINT_ID,
                vector=[0.0],
                payload={
                    "schema_version": config.SCHEMA_VERSION,
                    "embed_provider": config.EMBED_PROVIDER,
                    "embed_model": config.EMBED_MODEL,
                    "embed_dim": embed_dim,
                    **({"last_written_at": meta.get("last_written_at")}
                       if meta and meta.get("last_written_at") else {}),
                },
            )
        ],
    )
