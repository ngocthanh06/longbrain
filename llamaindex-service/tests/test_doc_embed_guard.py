"""Doc-space meta guard vs the migration path — regression for the
enable_doc_search catch-22: a lean install (no DOC_EMBED_*) records the
global embedder as the doc space in meta, so the first BGE-M3 migration was
refused by the very guard that migration exists to satisfy. The migration
must be able to bypass the doc-space guard in a controlled way, and must
stamp the new space only after a verified restore (record_doc_space)."""

import pytest
from qdrant_client import QdrantClient

from app import config, qdrant_setup

DIM = 2
DOC_DIM = 4


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(config, "EMBED_PROVIDER", "fastembed")
    monkeypatch.setattr(config, "EMBED_MODEL", "mini-test")
    monkeypatch.setattr(config, "DOC_EMBED_PROVIDER", "")
    monkeypatch.setattr(config, "DOC_EMBED_MODEL", "")
    c = QdrantClient(":memory:")
    # Lean-install boot: DOC_EMBED_* unset — meta records the global
    # embedder as the doc space too.
    qdrant_setup.ensure_all(c, DIM)
    yield c
    c.close()


def _enable_doc_embedder(monkeypatch):
    monkeypatch.setattr(config, "DOC_EMBED_PROVIDER", "huggingface")
    monkeypatch.setattr(config, "DOC_EMBED_MODEL", "bge-test")


def test_lean_boot_records_global_embedder_as_doc_space(client):
    meta = qdrant_setup.get_meta(client)
    assert meta["doc_embed_provider"] == "fastembed"
    assert meta["doc_embed_model"] == "mini-test"
    assert meta["doc_embed_dim"] == DIM


def test_normal_boot_refuses_doc_space_change(client, monkeypatch):
    _enable_doc_embedder(monkeypatch)
    with pytest.raises(RuntimeError, match="Document-embedding mismatch"):
        qdrant_setup.ensure_all(client, DIM, doc_embed_dim=DOC_DIM)


def test_migration_bypass_recreates_without_stamping_meta(client, monkeypatch):
    _enable_doc_embedder(monkeypatch)
    # What scripts/migrate_doc_embed.py does after its backup dump:
    client.delete_collection(config.DOCUMENTS_COLLECTION)
    qdrant_setup.ensure_all(
        client, DIM, doc_embed_dim=DOC_DIM, allow_doc_space_change=True
    )
    assert (
        qdrant_setup._collection_vector_size(client, config.DOCUMENTS_COLLECTION)
        == DOC_DIM
    )
    # Meta still holds the old space: a migration that dies mid-restore must
    # keep refusing normal boots instead of serving a partial collection.
    assert qdrant_setup.get_meta(client)["doc_embed_model"] == "mini-test"
    with pytest.raises(RuntimeError, match="Document-embedding mismatch"):
        qdrant_setup.ensure_all(client, DIM, doc_embed_dim=DOC_DIM)


def test_record_doc_space_unblocks_normal_boot(client, monkeypatch):
    _enable_doc_embedder(monkeypatch)
    client.delete_collection(config.DOCUMENTS_COLLECTION)
    qdrant_setup.ensure_all(
        client, DIM, doc_embed_dim=DOC_DIM, allow_doc_space_change=True
    )
    qdrant_setup.record_doc_space(client, "huggingface", "bge-test", DOC_DIM)
    qdrant_setup.ensure_all(client, DIM, doc_embed_dim=DOC_DIM)  # must not raise
    meta = qdrant_setup.get_meta(client)
    assert meta["doc_embed_provider"] == "huggingface"
    assert meta["doc_embed_model"] == "bge-test"
    assert meta["doc_embed_dim"] == DOC_DIM
