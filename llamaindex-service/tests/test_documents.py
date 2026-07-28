"""L4 (documents) dedup guard used by the docs/ auto-ingest watcher, and
version-supersession when a re-ingested file replaces an earlier one."""

import json
import time

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app import config, documents
from tests.conftest import FakeEmbed

DIM = 2


@pytest.fixture()
def client():
    c = QdrantClient(":memory:")
    yield c
    c.close()


def _seed_node(client, stored_path: str) -> None:
    # Real payload shape (see a live /ingest/file response): LlamaIndex
    # flattens metadata onto the top-level payload alongside the serialized
    # `_node_content` blob it uses to reconstruct nodes.
    client.create_collection(
        collection_name=config.DOCUMENTS_COLLECTION,
        vectors_config=qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=config.DOCUMENTS_COLLECTION,
        field_name="stored_path",
        field_schema=qmodels.PayloadSchemaType.KEYWORD,
    )
    client.upsert(
        collection_name=config.DOCUMENTS_COLLECTION,
        points=[
            qmodels.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={"stored_path": stored_path},
            )
        ],
    )


def test_already_ingested_true_for_matching_stored_path(client):
    _seed_node(client, "/data/documents/abc_file.pdf")
    assert documents.already_ingested(client, "/data/documents/abc_file.pdf") is True


def test_already_ingested_false_for_different_stored_path(client):
    _seed_node(client, "/data/documents/abc_file.pdf")
    assert documents.already_ingested(client, "/data/documents/other_file.pdf") is False


def test_already_ingested_false_when_collection_missing(client):
    assert documents.already_ingested(client, "/data/documents/whatever.pdf") is False


def _create_collection(client) -> None:
    client.create_collection(
        collection_name=config.DOCUMENTS_COLLECTION,
        vectors_config=qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=config.DOCUMENTS_COLLECTION,
        field_name="stored_path",
        field_schema=qmodels.PayloadSchemaType.KEYWORD,
    )


def _upsert_point(client, point_id: int, payload: dict, text: str = "chunk text") -> None:
    node_content = json.dumps({"text": text})
    client.upsert(
        collection_name=config.DOCUMENTS_COLLECTION,
        points=[
            qmodels.PointStruct(
                id=point_id,
                vector=[1.0, 0.0],
                payload={**payload, "_node_content": node_content},
            )
        ],
    )


def test_supersede_previous_versions_marks_old_chunk(client):
    _create_collection(client)
    _upsert_point(client, 1, {
        "project_id": "erp", "source": "faq.md", "document_key": "faq.md",
        "stored_path": "/data/documents/aaa_faq.md",
    })
    _upsert_point(client, 2, {
        "project_id": "erp", "source": "faq.md", "document_key": "faq.md",
        "stored_path": "/data/documents/bbb_faq.md",
    })

    count = documents._supersede_previous_versions(
        client, "erp", "faq.md", "/data/documents/bbb_faq.md"
    )

    assert count == 1
    by_id = {p.id: p.payload for p in
              client.retrieve(collection_name=config.DOCUMENTS_COLLECTION, ids=[1, 2])}
    assert by_id[1]["superseded_by"] == "/data/documents/bbb_faq.md"
    assert "superseded_by" not in by_id[2]


def test_supersede_previous_versions_ignores_enrichment_chunk(client):
    """Enrichment summary chunks (enriched=True) carry no stored_path — they
    must never be marked superseded just because the real document got a
    new version, and the original file chunk must still get superseded."""
    _create_collection(client)
    _upsert_point(client, 1, {
        "project_id": "erp", "source": "faq.md", "document_key": "faq.md",
        "stored_path": "/data/documents/aaa_faq.md",
    })
    _upsert_point(client, 2, {
        "project_id": "erp", "source": "faq.md", "enriched": True,
    })

    documents._supersede_previous_versions(
        client, "erp", "faq.md", "/data/documents/bbb_faq.md"
    )

    by_id = {p.id: p.payload for p in
              client.retrieve(collection_name=config.DOCUMENTS_COLLECTION, ids=[1, 2])}
    assert by_id[1]["superseded_by"] == "/data/documents/bbb_faq.md"
    assert "superseded_by" not in by_id[2]


def test_search_chunks_excludes_superseded(client):
    _create_collection(client)
    _upsert_point(
        client, 1,
        {"project_id": "erp", "source": "faq.md", "document_key": "faq.md",
         "stored_path": "/data/documents/aaa_faq.md", "superseded_by": "/data/documents/bbb_faq.md"},
        text="old content",
    )
    _upsert_point(
        client, 2,
        {"project_id": "erp", "source": "faq.md", "document_key": "faq.md",
         "stored_path": "/data/documents/bbb_faq.md"},
        text="new content",
    )

    embed_model = FakeEmbed({"query": [1.0, 0.0]})
    results = documents.search_chunks(client, embed_model, "query", project="erp", min_score=0.0)

    assert [r["text"] for r in results] == ["new content"]


def test_search_chunks_ranks_recent_chunk_above_older_equal_similarity(client):
    """Two chunks that tie on raw similarity must be reordered by recency:
    the one ingested long ago should decay below the freshly ingested one
    (see config.DOC_HALF_LIFE_DAYS)."""
    _create_collection(client)
    now = time.time()
    _upsert_point(
        client, 1,
        {"project_id": "erp", "source": "old.md",
         "ingested_at": now - 400 * 86400},
        text="old content",
    )
    _upsert_point(
        client, 2,
        {"project_id": "erp", "source": "new.md", "ingested_at": now},
        text="new content",
    )

    embed_model = FakeEmbed({"query": [1.0, 0.0]})
    results = documents.search_chunks(client, embed_model, "query", project="erp", min_score=0.0)

    assert [r["text"] for r in results] == ["new content", "old content"]


def test_search_chunks_no_penalty_for_missing_ingested_at(client):
    """Legacy chunks ingested before this field existed must not be penalized
    (fall back to age 0 -> no decay)."""
    _create_collection(client)
    _upsert_point(
        client, 1,
        {"project_id": "erp", "source": "legacy.md"},
        text="legacy content",
    )

    embed_model = FakeEmbed({"query": [1.0, 0.0]})
    results = documents.search_chunks(client, embed_model, "query", project="erp", min_score=0.0)

    assert results[0]["score"] == pytest.approx(1.0)


def test_same_basename_different_document_keys_do_not_supersede(client):
    _create_collection(client)
    _upsert_point(client, 1, {
        "project_id": "erp", "source": "README.md",
        "document_key": "backend/README.md",
        "stored_path": "/data/documents/backend.md",
    })
    _upsert_point(client, 2, {
        "project_id": "erp", "source": "README.md",
        "document_key": "frontend/README.md",
        "stored_path": "/data/documents/frontend.md",
    })

    count = documents._supersede_previous_versions(
        client, "erp", "frontend/README.md", "/data/documents/new-frontend.md"
    )

    assert count == 1
    by_id = {p.id: p.payload for p in
              client.retrieve(collection_name=config.DOCUMENTS_COLLECTION, ids=[1, 2])}
    assert "superseded_by" not in by_id[1]
    assert by_id[2]["superseded_by"] == "/data/documents/new-frontend.md"
