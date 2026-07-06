"""L2/L3 behaviour against a real (in-process) Qdrant: qdrant-client's local
mode gives the actual filter/search semantics without a server."""

import math

import pytest
from qdrant_client import QdrantClient

from app import config, memories, memory_store, qdrant_setup
from tests.conftest import FakeEmbed

DIM = 2


def _unit(angle_deg: float) -> list[float]:
    rad = math.radians(angle_deg)
    return [math.cos(rad), math.sin(rad)]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(config, "EMBED_MODEL", "fake-test-model")
    c = QdrantClient(":memory:")
    qdrant_setup.ensure_all(c, DIM)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# save_facts: dedup + supersede
# ---------------------------------------------------------------------------
def test_save_facts_exact_duplicate_skipped(client):
    embed = FakeEmbed()
    r1 = memories.save_facts(client, embed, [{"text": "User likes Qdrant"}])
    r2 = memories.save_facts(client, embed, [{"text": "  user   LIKES qdrant "}])
    assert r1[0]["status"] == "new"
    assert r2[0]["status"] == "duplicate"


def test_save_facts_near_duplicate_superseded(client):
    embed = FakeEmbed({
        "User likes Qdrant": _unit(0),
        "User loves Qdrant a lot": _unit(5),  # cos(5°) ≈ 0.996 ≥ 0.92
    })
    memories.save_facts(client, embed, [{"text": "User likes Qdrant"}])
    r = memories.save_facts(client, embed, [{"text": "User loves Qdrant a lot"}])
    assert r[0]["status"] == "supersedes"
    assert r[0]["superseded"] == ["User likes Qdrant"]

    # The superseded fact is excluded from recall; the new one is returned.
    hits = memories.search_memories(client, embed, "User likes Qdrant", top_k=5)
    texts = [h["text"] for h in hits]
    assert "User loves Qdrant a lot" in texts
    assert "User likes Qdrant" not in texts


def test_save_facts_distinct_fact_not_superseded(client):
    embed = FakeEmbed({
        "User likes Qdrant": _unit(0),
        "Deadline is Friday": _unit(80),  # cos(80°) ≈ 0.17 < 0.92
    })
    memories.save_facts(client, embed, [{"text": "User likes Qdrant"}])
    r = memories.save_facts(client, embed, [{"text": "Deadline is Friday"}])
    assert r[0]["status"] == "new"


def test_save_facts_skips_empty_text_and_clamps(client):
    embed = FakeEmbed()
    r = memories.save_facts(
        client, embed,
        [{"text": "   "}, {"text": "real", "type": "bogus", "importance": 7}],
    )
    assert len(r) == 1
    facts = memories.list_facts(client)
    assert facts[0]["type"] == "fact"  # invalid type coerced
    assert facts[0]["importance"] == 1.0  # clamped to [0, 1]


# ---------------------------------------------------------------------------
# search_memories: RECALL_MIN_SCORE filter
# ---------------------------------------------------------------------------
def test_search_applies_min_score(client):
    embed = FakeEmbed({
        "close fact": _unit(0),
        "far fact": _unit(85),  # cos(85°) ≈ 0.09 → scored ≈ 0.07 < 0.25
        "the query": _unit(0),
    })
    memories.save_facts(client, embed, [{"text": "close fact"}, {"text": "far fact"}])
    hits = memories.search_memories(client, embed, "the query", top_k=5)
    texts = [h["text"] for h in hits]
    assert "close fact" in texts
    assert "far fact" not in texts


# ---------------------------------------------------------------------------
# L2: idempotent append + consolidated roundtrip
# ---------------------------------------------------------------------------
def test_add_message_idempotent(client):
    embed = FakeEmbed()
    memory_store.add_message(client, embed, "s1", "user", "hello")
    memory_store.add_message(client, embed, "s1", "user", "hello")  # retry
    count = client.count(config.CHAT_HISTORY_COLLECTION, exact=True).count
    assert count == 1


def test_mark_consolidated_roundtrip(client):
    embed = FakeEmbed()
    memory_store.add_message(client, embed, "s1", "user", "hello")
    memory_store.add_message(client, embed, "s1", "assistant", "hi there")
    points = memory_store.fetch_unconsolidated(client, "s1")
    assert len(points) == 2
    memory_store.mark_consolidated(client, [p.id for p in points])
    assert memory_store.fetch_unconsolidated(client, "s1") == []


def test_delete_fact_hard_deletes(client):
    embed = FakeEmbed()
    memories.save_facts(client, embed, [{"text": "to forget"}])
    fact_id = memories.list_facts(client)[0]["id"]
    assert memories.delete_fact(client, fact_id) is True
    assert memories.delete_fact(client, fact_id) is False  # already gone
    assert memories.list_facts(client, include_superseded=True) == []
