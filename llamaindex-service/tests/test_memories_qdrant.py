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


def test_graph_data_edges_by_similarity(client):
    embed = FakeEmbed({
        # 45° apart: cos ≈ 0.71 → related (≥ 0.35) but NOT superseding (< 0.92)
        "fact A": _unit(0),
        "fact B": _unit(45),
        "fact C": _unit(120),  # cos to A/B: -0.5 / 0.26 → below 0.35, no edge
    })
    memories.save_facts(
        client, embed,
        [{"text": "fact A"}, {"text": "fact B"}, {"text": "fact C"}],
        project_id="gakken",
    )
    g = memories.graph_data(client, min_similarity=0.35)
    assert len(g["nodes"]) == 3
    assert all(n["project_id"] == "gakken" for n in g["nodes"])
    texts = {n["id"]: n["text"] for n in g["nodes"]}
    assert len(g["edges"]) == 1
    edge = g["edges"][0]
    assert {texts[edge["source"]], texts[edge["target"]]} == {"fact A", "fact B"}
    assert 0.69 <= edge["weight"] <= 0.72


def test_graph_data_excludes_superseded_by_default(client):
    embed = FakeEmbed({
        "old version": _unit(0),
        "new version here": _unit(3),  # supersedes old
    })
    memories.save_facts(client, embed, [{"text": "old version"}])
    memories.save_facts(client, embed, [{"text": "new version here"}])
    g = memories.graph_data(client)
    assert [n["text"] for n in g["nodes"]] == ["new version here"]
    g_all = memories.graph_data(client, include_superseded=True)
    assert len(g_all["nodes"]) == 2
    assert sum(1 for n in g_all["nodes"] if n["superseded"]) == 1


def test_session_project_sticky_to_founding_turn(client):
    embed = FakeEmbed()
    memory_store.add_message(client, embed, "s9", "user", "first turn", project_id="erp")
    # A later turn arrives tagged differently (sidebar selection changed):
    # the session's project must stay the founding one, deterministically.
    memory_store.add_message(client, embed, "s9", "assistant", "later turn", project_id="study")
    assert memory_store.get_session_project(client, "s9") == "erp"


def test_append_project_folder_source_overrides_sticky(client):
    embed = FakeEmbed()
    memory_store.add_message(client, embed, "s10", "user", "t1", project_id="study")
    # cwd moved into erp's folder (project_switch): intentional → override
    assert memory_store.resolve_append_project(client, "s10", "erp", "folder") == "erp"


def test_append_project_ambient_sources_stay_sticky(client):
    embed = FakeEmbed()
    memory_store.add_message(client, embed, "s11", "user", "t1", project_id="study")
    # sidebar selection drift / default / legacy hook: founding project wins
    assert memory_store.resolve_append_project(client, "s11", "erp", "active") == "study"
    assert memory_store.resolve_append_project(client, "s11", "erp", "default") == "study"
    assert memory_store.resolve_append_project(client, "s11", "erp", "") == "study"
    # same project via folder: no-op either way
    assert memory_store.resolve_append_project(client, "s11", "study", "folder") == "study"


def test_append_project_new_session_uses_hook_project(client):
    assert memory_store.resolve_append_project(client, "s-new", "erp", "active") == "erp"
    assert memory_store.resolve_append_project(client, "s-new", "", "") == "default"


def test_session_project_fallback_for_new_session(client):
    assert memory_store.get_session_project(client, "brand-new", fallback="abc") == "abc"
    assert memory_store.get_session_project(client, "brand-new") == "default"


def test_retag_fact_and_session(client):
    embed = FakeEmbed()
    memory_store.add_message(client, embed, "s20", "user", "q1", project_id="study")
    memory_store.add_message(client, embed, "s20", "assistant", "a1", project_id="study")
    memories.save_facts(client, embed, [{"text": "fact from s20"}],
                        session_id="s20", project_id="study")
    fact_id = memories.list_facts(client)[0]["id"]

    # single fact re-tag
    assert memories.set_fact_project(client, fact_id, "erp") is True
    assert memories.list_facts(client)[0]["project_id"] == "erp"
    assert memories.set_fact_project(client, "00000000-0000-0000-0000-00000000dead", "erp") is False

    # whole-session re-tag: turns + facts move, stickiness follows
    assert memory_store.set_session_project(client, "s20", "gakken") == 2
    assert memories.set_session_facts_project(client, "s20", "gakken") == 1
    assert memory_store.get_session_project(client, "s20") == "gakken"
    assert memories.list_facts(client)[0]["project_id"] == "gakken"


def test_bulk_retag_facts(client):
    embed = FakeEmbed({"b1": _unit(0), "b2": _unit(45), "b3": _unit(120)})
    memories.save_facts(client, embed,
                        [{"text": "b1"}, {"text": "b2"}, {"text": "b3"}],
                        project_id="study")
    facts = {f["text"]: f["id"] for f in memories.list_facts(client)}
    moved = memories.set_facts_project(
        client, [facts["b1"], facts["b2"], "00000000-0000-0000-0000-00000000dead"], "erp"
    )
    assert moved == 2  # bogus id skipped
    by_text = {f["text"]: f["project_id"] for f in memories.list_facts(client)}
    assert by_text == {"b1": "erp", "b2": "erp", "b3": "study"}


def test_rename_project_across_collections(client):
    embed = FakeEmbed()
    memory_store.add_message(client, embed, "s21", "user", "x", project_id="old-name")
    memories.save_facts(client, embed, [{"text": "y"}], project_id="old-name")
    counts = memory_store.rename_project(client, "old-name", "new-name")
    assert counts[config.CHAT_HISTORY_COLLECTION] == 1
    assert counts[config.MEMORIES_COLLECTION] == 1
    assert memory_store.get_session_project(client, "s21") == "new-name"
    assert memories.list_facts(client)[0]["project_id"] == "new-name"
    # unknown project → all zeros
    assert not any(memory_store.rename_project(client, "ghost", "x").values())


def test_delete_fact_hard_deletes(client):
    embed = FakeEmbed()
    memories.save_facts(client, embed, [{"text": "to forget"}])
    fact_id = memories.list_facts(client)[0]["id"]
    assert memories.delete_fact(client, fact_id) is True
    assert memories.delete_fact(client, fact_id) is False  # already gone
    assert memories.list_facts(client, include_superseded=True) == []


# ---------------------------------------------------------------------------
# source_agent provenance (multi-agent parallel operation)
# ---------------------------------------------------------------------------
def test_source_agent_stamped_and_optional(client):
    embed = FakeEmbed()
    memory_store.add_message(client, embed, "s30", "user", "from claude",
                             source_agent="claude-code")
    memory_store.add_message(client, embed, "s30", "assistant", "no agent given")
    points, _ = client.scroll(collection_name=config.CHAT_HISTORY_COLLECTION,
                              limit=10, with_payload=True)
    by_content = {p.payload["content"]: p.payload for p in points}
    assert by_content["from claude"]["source_agent"] == "claude-code"
    assert "source_agent" not in by_content["no agent given"]  # old shape kept

    memories.save_facts(client, embed, [{"text": "tagged fact"}],
                        source_agent="hermes")
    fact = memories.list_facts(client)[0]
    assert fact["source_agent"] == "hermes"
