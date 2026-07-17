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


def record_doc_space(client: QdrantClient, provider: str, model: str, dim: int) -> None:
    """Stamp the document-embedding space in meta. Only
    scripts/migrate_doc_embed.py calls this, and only after the restored
    point count is verified — see ensure_all(allow_doc_space_change=...)."""
    client.set_payload(
        collection_name=config.META_COLLECTION,
        payload={"doc_embed_provider": provider, "doc_embed_model": model,
                 "doc_embed_dim": dim},
        points=[META_POINT_ID],
    )


def ensure_all(
    client: QdrantClient,
    embed_dim: int,
    doc_embed_dim: int | None = None,
    allow_doc_space_change: bool = False,
) -> None:
    """`doc_embed_dim` is the documents-collection vector size when a separate
    doc embedder is configured (SEARCH_SPEC constraint 1); None = documents
    share the global embedder. Pass None also when the doc embedder failed to
    load — the doc-space checks are then skipped and the rest still boots.

    `allow_doc_space_change` is reserved for scripts/migrate_doc_embed.py:
    changing the doc embedder is exactly what a migration does, so the
    doc-space mismatch refusal must not fire there (a lean install records
    the global embedder as the doc space, which would deadlock the upgrade).
    In that mode the recorded doc_embed_* meta is left untouched — the
    migration stamps the new space via record_doc_space() only after the
    restored point count is verified, so a migration that dies mid-restore
    keeps refusing normal boots instead of serving a partial collection."""
    doc_dim_effective = doc_embed_dim if doc_embed_dim is not None else embed_dim
    doc_model = config.DOC_EMBED_MODEL or config.EMBED_MODEL
    doc_provider = (config.DOC_EMBED_PROVIDER or config.EMBED_PROVIDER) \
        if config.DOC_EMBED_MODEL else config.EMBED_PROVIDER
    meta = get_meta(client)
    if meta is not None:
        stored_model = meta.get("embed_model")
        stored_dim = meta.get("embed_dim")
        stored_provider = meta.get("embed_provider")
        # Same model name + dim from two different providers can still be a
        # different vector space (different pooling/normalization) — compare
        # provider too. Null-safe: meta written before this field existed
        # has no embed_provider, so it can't be compared and is skipped.
        if (
            stored_model != config.EMBED_MODEL
            or stored_dim != embed_dim
            or (stored_provider and stored_provider != config.EMBED_PROVIDER)
        ):
            raise RuntimeError(
                "Embedding mismatch: data on disk was created with "
                f"{stored_provider or 'unknown provider'}/{stored_model!r} "
                f"(dim={stored_dim}) but the service is configured with "
                f"{config.EMBED_PROVIDER!r}/{config.EMBED_MODEL!r} (dim={embed_dim}). "
                "Either restore the previous EMBED_PROVIDER/EMBED_MODEL, or "
                "start fresh (docker compose down -v) / re-embed into new "
                "collections before switching."
            )
        # Doc-space guard — null-safe: meta written before the doc fields
        # existed skips the comparison (the collection-dim check below still
        # catches a hard mismatch).
        stored_doc_model = meta.get("doc_embed_model")
        stored_doc_provider = meta.get("doc_embed_provider")
        if not allow_doc_space_change and doc_embed_dim is not None and stored_doc_model and (
            stored_doc_model != doc_model
            or meta.get("doc_embed_dim") != doc_dim_effective
            or (stored_doc_provider and stored_doc_provider != doc_provider)
        ):
            raise RuntimeError(
                "Document-embedding mismatch: documents on disk were embedded "
                f"with {stored_doc_provider or 'unknown provider'}/"
                f"{stored_doc_model!r} (dim={meta.get('doc_embed_dim')}) but "
                f"the service is configured with {doc_provider!r}/{doc_model!r} "
                f"(dim={doc_dim_effective}). Restore the previous DOC_EMBED_* "
                "or run scripts/migrate_doc_embed.py to re-embed."
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
    # It uses the DOC embedder's dimension; the doc-space checks are skipped
    # entirely when the doc embedder is unavailable (doc search 503s instead).
    if doc_embed_dim is not None or not config.DOC_EMBED_MODEL:
        _ensure_collection(client, config.DOCUMENTS_COLLECTION, doc_dim_effective)
        doc_dim = _collection_vector_size(client, config.DOCUMENTS_COLLECTION)
        if doc_dim is not None and doc_dim != doc_dim_effective:
            raise RuntimeError(
                f"Collection {config.DOCUMENTS_COLLECTION!r} holds "
                f"{doc_dim}-dim vectors but the configured document embedding "
                f"produces {doc_dim_effective}-dim. Run scripts/"
                "migrate_doc_embed.py (see the embedding-migration note in README)."
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
                    # Doc space recorded only when it is actually known; on a
                    # failed doc-embedder load the previous values survive.
                    **({"doc_embed_provider": doc_provider,
                        "doc_embed_model": doc_model,
                        "doc_embed_dim": doc_dim_effective}
                       if ((doc_embed_dim is not None or not config.DOC_EMBED_MODEL)
                           and not allow_doc_space_change)
                       else {k: meta[k] for k in
                             ("doc_embed_provider", "doc_embed_model", "doc_embed_dim")
                             if meta and k in meta}),
                    **({"last_written_at": meta.get("last_written_at")}
                       if meta and meta.get("last_written_at") else {}),
                },
            )
        ],
    )
