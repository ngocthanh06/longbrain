"""Enrichment version-awareness: a summary chunk must be tied to the exact
version it was generated from (document_key + stored_path), not just
`source` — otherwise a stale v1 summary survives forever after the real
document changes to v2 (the old already_enriched() kept matching on source
alone, and the summary chunk had no stored_path/document_key for the normal
supersede pass to ever catch)."""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app import config, documents, enrich

DIM = 2


@pytest.fixture()
def client():
    c = QdrantClient(":memory:")
    c.create_collection(
        collection_name=config.DOCUMENTS_COLLECTION,
        vectors_config=qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE),
    )
    yield c
    c.close()


@pytest.fixture(autouse=True)
def llm_enabled(monkeypatch):
    monkeypatch.setattr(config, "DOC_ENRICH", True)
    monkeypatch.setattr(enrich, "llm_available", lambda: True)
    monkeypatch.setattr(enrich, "_complete", lambda prompt: "Tóm tắt giả.")


def _upsert_chunk(client, point_id, payload):
    client.upsert(
        collection_name=config.DOCUMENTS_COLLECTION,
        points=[qmodels.PointStruct(id=point_id, vector=[1.0, 0.0], payload=payload)],
    )


# ---------------------------------------------------------------------------
# already_enriched
# ---------------------------------------------------------------------------
def test_already_enriched_checks_exact_version_not_just_source(client):
    _upsert_chunk(client, 1, {
        "project_id": "erp", "source": "faq.md", "enriched": True,
        "document_key": "faq.md", "stored_path": "/data/documents/v1_faq.md",
    })
    assert enrich.already_enriched(
        client, "erp", "faq.md", "faq.md", "/data/documents/v1_faq.md"
    ) is True
    # v2 of the SAME document (new stored_path) has no summary of its own yet
    assert enrich.already_enriched(
        client, "erp", "faq.md", "faq.md", "/data/documents/v2_faq.md"
    ) is False


def test_already_enriched_falls_back_to_source_when_no_version_identity(client):
    _upsert_chunk(client, 1, {"project_id": "erp", "source": "note", "enriched": True})
    assert enrich.already_enriched(client, "erp", "note") is True


# ---------------------------------------------------------------------------
# _version_is_current — the race guard
# ---------------------------------------------------------------------------
def test_version_is_current_false_after_supersede(client):
    _upsert_chunk(client, 1, {
        "project_id": "erp", "document_key": "faq.md",
        "stored_path": "/data/documents/v1_faq.md",
        "superseded_by": "/data/documents/v2_faq.md",
    })
    assert enrich._version_is_current(client, "erp", "faq.md", "/data/documents/v1_faq.md") is False


def test_version_is_current_true_for_active_version(client):
    _upsert_chunk(client, 1, {
        "project_id": "erp", "document_key": "faq.md",
        "stored_path": "/data/documents/v2_faq.md",
    })
    assert enrich._version_is_current(client, "erp", "faq.md", "/data/documents/v2_faq.md") is True


# ---------------------------------------------------------------------------
# enrich_document — end to end
# ---------------------------------------------------------------------------
def test_enrich_document_skips_when_already_enriched_for_this_version(client, monkeypatch):
    _upsert_chunk(client, 1, {
        "project_id": "erp", "source": "faq.md", "document_key": "faq.md",
        "stored_path": "/data/documents/v1_faq.md",
    })
    _upsert_chunk(client, 2, {
        "project_id": "erp", "source": "faq.md", "enriched": True,
        "document_key": "faq.md", "stored_path": "/data/documents/v1_faq.md",
    })
    calls = []
    monkeypatch.setattr(documents, "ingest_text", lambda *a, **k: calls.append((a, k)))

    result = enrich.enrich_document(
        index=object(), qdrant_client=client, source="faq.md", project_id="erp",
        document_key="faq.md", stored_path="/data/documents/v1_faq.md",
    )

    assert result is False
    assert calls == []


def test_enrich_document_writes_summary_tagged_with_document_key_and_stored_path(client, monkeypatch):
    _upsert_chunk(client, 1, {
        "project_id": "erp", "source": "faq.md", "document_key": "faq.md",
        "stored_path": "/data/documents/v2_faq.md",
        "_node_content": '{"text": "real content of v2"}',
    })
    calls = []
    monkeypatch.setattr(documents, "ingest_text", lambda *a, **k: calls.append((a, k)))

    result = enrich.enrich_document(
        index=object(), qdrant_client=client, source="faq.md", project_id="erp",
        document_key="faq.md", stored_path="/data/documents/v2_faq.md",
    )

    assert result is True
    assert len(calls) == 1
    _, kwargs = calls[0]
    metadata = kwargs["metadata"]
    assert metadata["document_key"] == "faq.md"
    assert metadata["stored_path"] == "/data/documents/v2_faq.md"
    assert metadata["enriched"] is True
    assert kwargs["project_id"] == "erp"


def test_enrich_document_skips_stale_summary_when_version_superseded_mid_flight(client, monkeypatch):
    """Race guard: the LLM call is slow — by the time it finishes, v2 may
    already have superseded v1's real content. Must not write a stale v1
    summary after that point."""
    _upsert_chunk(client, 1, {
        "project_id": "erp", "source": "faq.md", "document_key": "faq.md",
        "stored_path": "/data/documents/v1_faq.md",
        "_node_content": '{"text": "stale v1 content"}',
        "superseded_by": "/data/documents/v2_faq.md",  # v2 already landed
    })
    calls = []
    monkeypatch.setattr(documents, "ingest_text", lambda *a, **k: calls.append((a, k)))

    result = enrich.enrich_document(
        index=object(), qdrant_client=client, source="faq.md", project_id="erp",
        document_key="faq.md", stored_path="/data/documents/v1_faq.md",
    )

    assert result is False
    assert calls == []


def test_enrich_document_legacy_path_without_version_identity_still_works(client, monkeypatch):
    """Manually added text (add_to_knowledge_base) has no document_key/
    stored_path at all — must keep working exactly as before."""
    _upsert_chunk(client, 1, {
        "project_id": "erp", "source": "note",
        "_node_content": '{"text": "manual note content"}',
    })
    calls = []
    monkeypatch.setattr(documents, "ingest_text", lambda *a, **k: calls.append((a, k)))

    result = enrich.enrich_document(
        index=object(), qdrant_client=client, source="note", project_id="erp",
    )

    assert result is True
    metadata = calls[0][1]["metadata"]
    assert "document_key" not in metadata
    assert "stored_path" not in metadata


def test_enrich_document_new_version_chunks_not_mixed_with_old(client, monkeypatch):
    """_doc_text must summarize ONLY the current version's chunks — an old
    (superseded) chunk sharing the same `source` must not leak into the
    text handed to the LLM."""
    _upsert_chunk(client, 1, {
        "project_id": "erp", "source": "faq.md", "document_key": "faq.md",
        "stored_path": "/data/documents/v1_faq.md",
        "_node_content": '{"text": "OLD stale content"}',
        "superseded_by": "/data/documents/v2_faq.md",
    })
    _upsert_chunk(client, 2, {
        "project_id": "erp", "source": "faq.md", "document_key": "faq.md",
        "stored_path": "/data/documents/v2_faq.md",
        "_node_content": '{"text": "NEW current content"}',
    })
    captured_prompt = {}

    def fake_complete(prompt):
        captured_prompt["value"] = prompt
        return "Tóm tắt giả."

    monkeypatch.setattr(enrich, "_complete", fake_complete)
    monkeypatch.setattr(documents, "ingest_text", lambda *a, **k: None)

    enrich.enrich_document(
        index=object(), qdrant_client=client, source="faq.md", project_id="erp",
        document_key="faq.md", stored_path="/data/documents/v2_faq.md",
    )

    assert "NEW current content" in captured_prompt["value"]
    assert "OLD stale content" not in captured_prompt["value"]
