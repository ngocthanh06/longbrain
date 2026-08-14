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


# ---------------------------------------------------------------------------
# federated_search_chunks: Connector Layer federation (config.CONNECTOR_SEARCH_URL)
# ---------------------------------------------------------------------------
def test_federated_search_chunks_disabled_when_url_unset(monkeypatch):
    monkeypatch.setattr(config, "CONNECTOR_SEARCH_URL", "")
    assert documents.federated_search_chunks("query") == []


def test_federated_search_chunks_tags_origin_and_forwards_params(monkeypatch):
    monkeypatch.setattr(config, "CONNECTOR_SEARCH_URL", "http://localhost:8801")
    monkeypatch.setattr(config, "CONNECTOR_SEARCH_TIMEOUT_MS", 250)
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [
                {"text": "doc text", "source": "drive-doc.md", "project_id": "erp",
                 "document_key": "abc123", "score": 0.8, "trace": {}},
            ]}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(documents.requests, "post", fake_post)

    results = documents.federated_search_chunks("query text", project="erp", top_k=3)

    assert captured["url"] == "http://localhost:8801/documents/search"
    assert captured["json"] == {"query": "query text", "project": "erp", "top_k": 3}
    assert captured["timeout"] == pytest.approx(0.25)
    assert results == [
        {"text": "doc text", "source": "drive-doc.md", "project_id": "erp",
         "document_key": "abc123", "score": 0.8, "trace": {}, "origin": "connector-layer"},
    ]


def test_federated_search_chunks_fails_open_on_error(monkeypatch):
    monkeypatch.setattr(config, "CONNECTOR_SEARCH_URL", "http://localhost:8801")

    def raise_error(url, json, timeout):
        raise TimeoutError("connector backend unreachable")

    monkeypatch.setattr(documents.requests, "post", raise_error)

    assert documents.federated_search_chunks("query") == []


def test_delete_document_removes_active_and_superseded_chunks(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCUMENTS_DIR", str(tmp_path))
    _create_collection(client)
    old_file = tmp_path / "aaa_faq.md"
    new_file = tmp_path / "bbb_faq.md"
    old_file.write_text("old")
    new_file.write_text("new")
    _upsert_point(client, 1, {
        "project_id": "erp", "document_key": "faq.md",
        "stored_path": str(old_file), "superseded_by": str(new_file),
    })
    _upsert_point(client, 2, {
        "project_id": "erp", "document_key": "faq.md", "stored_path": str(new_file),
    })

    result = documents.delete_document(client, "erp", "faq.md")

    assert result == {"chunks_deleted": 2, "files_removed": 2}
    remaining, _ = client.scroll(collection_name=config.DOCUMENTS_COLLECTION, limit=10)
    assert remaining == []
    assert not old_file.exists()
    assert not new_file.exists()


def test_delete_document_ignores_other_project_and_document_key(client, tmp_path):
    _create_collection(client)
    keep_file = tmp_path / "other.md"
    keep_file.write_text("keep")
    _upsert_point(client, 1, {
        "project_id": "erp", "document_key": "faq.md",
        "stored_path": str(tmp_path / "target.md"),
    })
    _upsert_point(client, 2, {
        "project_id": "erp", "document_key": "other.md", "stored_path": str(keep_file),
    })
    _upsert_point(client, 3, {
        "project_id": "other_project", "document_key": "faq.md",
        "stored_path": str(tmp_path / "target.md"),
    })

    result = documents.delete_document(client, "erp", "faq.md")

    assert result == {"chunks_deleted": 1, "files_removed": 0}
    remaining, _ = client.scroll(collection_name=config.DOCUMENTS_COLLECTION, limit=10)
    assert {p.id for p in remaining} == {2, 3}
    assert keep_file.exists()


def test_delete_document_keeps_file_still_referenced_by_another_document(client, tmp_path):
    """Two document_keys sharing one content-addressed stored_path (a
    duplicate upload) must not have their shared file deleted while either
    document still references it."""
    _create_collection(client)
    shared_file = tmp_path / "shared.md"
    shared_file.write_text("shared")
    _upsert_point(client, 1, {
        "project_id": "erp", "document_key": "doc-a", "stored_path": str(shared_file),
    })
    _upsert_point(client, 2, {
        "project_id": "erp", "document_key": "doc-b", "stored_path": str(shared_file),
    })

    result = documents.delete_document(client, "erp", "doc-a")

    assert result == {"chunks_deleted": 1, "files_removed": 0}
    assert shared_file.exists()


def test_delete_document_no_match_returns_zero(client):
    _create_collection(client)
    _upsert_point(client, 1, {"project_id": "erp", "document_key": "faq.md", "stored_path": "/x"})

    result = documents.delete_document(client, "erp", "does-not-exist")

    assert result == {"chunks_deleted": 0, "files_removed": 0}


def test_delete_document_requires_project_id_and_document_key(client):
    assert documents.delete_document(client, "", "faq.md") == {"chunks_deleted": 0, "files_removed": 0}
    assert documents.delete_document(client, "erp", "") == {"chunks_deleted": 0, "files_removed": 0}


def test_cleanup_superseded_is_dry_run_by_default_and_preserves_active(client, tmp_path):
    _create_collection(client)
    stale_file = tmp_path / "stale.md"
    active_file = tmp_path / "active.md"
    stale_file.write_text("stale")
    active_file.write_text("active")
    _upsert_point(client, 1, {
        "project_id": "erp", "document_key": "faq.md",
        "stored_path": str(stale_file), "superseded_by": str(active_file),
    })
    _upsert_point(client, 2, {
        "project_id": "erp", "document_key": "faq.md",
        "stored_path": str(active_file),
    })

    result = documents.cleanup_superseded(client)

    assert result["status"] == "planned"
    assert result["chunks_found"] == 1
    assert result["chunks_deleted"] == 0
    assert result["items"][0]["document_key"] == "faq.md"
    assert result["items"][0]["superseded_by"] == str(active_file)
    assert result["items"][0]["file_action"] == "remove"
    assert result["items"][0]["text_preview"] == "chunk text"
    assert stale_file.exists()
    assert client.retrieve(collection_name=config.DOCUMENTS_COLLECTION, ids=[1])


def test_cleanup_superseded_deletes_unreferenced_file_but_keeps_shared_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DOCUMENTS_DIR", str(tmp_path))
    _create_collection(client)
    stale_file = tmp_path / "stale.md"
    shared_file = tmp_path / "shared.md"
    stale_file.write_text("stale")
    shared_file.write_text("shared")
    _upsert_point(client, 1, {
        "project_id": "erp", "document_key": "faq.md",
        "stored_path": str(stale_file), "superseded_by": "/new.md",
    })
    _upsert_point(client, 2, {
        "project_id": "erp", "document_key": "faq.md",
        "stored_path": str(shared_file), "superseded_by": "/new2.md",
    })
    _upsert_point(client, 3, {
        "project_id": "other", "document_key": "keep.md",
        "stored_path": str(shared_file),
    })

    result = documents.cleanup_superseded(client, dry_run=False)

    assert result["chunks_deleted"] == 2
    assert result["files_removed"] == 1
    assert not stale_file.exists()
    assert shared_file.exists()
    remaining, _ = client.scroll(collection_name=config.DOCUMENTS_COLLECTION, limit=10)
    assert {p.id for p in remaining} == {3}


# ---------------------------------------------------------------------------
# Path safety: stored_path comes straight out of a Qdrant payload, which a
# bad ingest or a hand-edited point could set to anything. These confirm
# delete_document_path / delete_document / cleanup_superseded still delete
# the Qdrant chunk(s) but never touch a file outside config.DOCUMENTS_DIR,
# no matter what the payload says — only document_path() decides what's
# safe to unlink.
# ---------------------------------------------------------------------------

def test_delete_document_path_refuses_traversal_outside_documents_dir(client, tmp_path, monkeypatch):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    monkeypatch.setattr(config, "DOCUMENTS_DIR", str(docs_dir))
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("do not delete me")
    traversal_path = str(docs_dir / ".." / "secret.txt")
    _create_collection(client)
    _upsert_point(client, 1, {"project_id": "erp", "stored_path": traversal_path})

    result = documents.delete_document_path(client, "erp", traversal_path)

    assert result["chunks_deleted"] == 1
    assert result["files_removed"] == 0
    assert outside_file.exists()


def test_delete_document_path_refuses_absolute_path_outside_storage(client, tmp_path, monkeypatch):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    monkeypatch.setattr(config, "DOCUMENTS_DIR", str(docs_dir))
    outside_file = tmp_path / "outside.md"
    outside_file.write_text("keep me")
    _create_collection(client)
    _upsert_point(client, 1, {"project_id": "erp", "stored_path": str(outside_file)})

    result = documents.delete_document_path(client, "erp", str(outside_file))

    assert result["chunks_deleted"] == 1
    assert result["files_removed"] == 0
    assert outside_file.exists()


def test_delete_document_path_refuses_symlink_escape(client, tmp_path, monkeypatch):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    monkeypatch.setattr(config, "DOCUMENTS_DIR", str(docs_dir))
    outside_file = tmp_path / "real_secret.txt"
    outside_file.write_text("do not delete me")
    link_path = docs_dir / "link.md"
    link_path.symlink_to(outside_file)
    _create_collection(client)
    _upsert_point(client, 1, {"project_id": "erp", "stored_path": str(link_path)})

    result = documents.delete_document_path(client, "erp", str(link_path))

    assert result["chunks_deleted"] == 1
    assert result["files_removed"] == 0
    assert outside_file.exists()


def test_delete_document_refuses_malicious_stored_path(client, tmp_path, monkeypatch):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    monkeypatch.setattr(config, "DOCUMENTS_DIR", str(docs_dir))
    outside_file = tmp_path / "outside.md"
    outside_file.write_text("keep me")
    _create_collection(client)
    _upsert_point(client, 1, {
        "project_id": "erp", "document_key": "faq.md", "stored_path": str(outside_file),
    })

    result = documents.delete_document(client, "erp", "faq.md")

    assert result["chunks_deleted"] == 1
    assert result["files_removed"] == 0
    assert outside_file.exists()


def test_cleanup_superseded_refuses_malicious_stored_path(client, tmp_path, monkeypatch):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    monkeypatch.setattr(config, "DOCUMENTS_DIR", str(docs_dir))
    outside_file = tmp_path / "outside.md"
    outside_file.write_text("keep me")
    _create_collection(client)
    _upsert_point(client, 1, {
        "project_id": "erp", "document_key": "faq.md",
        "stored_path": str(outside_file), "superseded_by": "/new.md",
    })

    result = documents.cleanup_superseded(client, dry_run=False)

    assert result["chunks_deleted"] == 1
    assert result["files_removed"] == 0
    assert outside_file.exists()
