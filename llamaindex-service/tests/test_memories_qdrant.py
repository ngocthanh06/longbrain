"""L2/L3 behaviour against a real (in-process) Qdrant: qdrant-client's local
mode gives the actual filter/search semantics without a server."""

import math
import time

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from app import config, memories, memory_store, qdrant_setup
from tests.conftest import FakeEmbed, FakeLLM

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


# ---------------------------------------------------------------------------
# save_facts: LLM-checked dedup band (paraphrases too far apart for raw
# cosine to trust, but not so far they're clearly distinct — see config.py)
# ---------------------------------------------------------------------------
def test_save_facts_llm_confirms_paraphrase_in_dedup_band(client):
    embed = FakeEmbed({
        "User likes Qdrant": _unit(0),
        "User is quite fond of Qdrant indeed": _unit(41),  # cos(41°)≈0.75: in-band
    })
    memories.save_facts(client, embed, [{"text": "User likes Qdrant"}])
    llm = FakeLLM("yes")
    r = memories.save_facts(
        client, embed, [{"text": "User is quite fond of Qdrant indeed"}], llm=llm
    )
    assert r[0]["status"] == "supersedes"
    assert len(llm.calls) == 1


def test_save_facts_llm_rejects_stays_distinct(client):
    embed = FakeEmbed({
        "User likes Qdrant": _unit(0),
        "Something loosely related to Qdrant": _unit(41),
    })
    memories.save_facts(client, embed, [{"text": "User likes Qdrant"}])
    llm = FakeLLM("no")
    r = memories.save_facts(
        client, embed, [{"text": "Something loosely related to Qdrant"}], llm=llm
    )
    assert r[0]["status"] == "new"


def test_save_facts_without_llm_skips_dedup_band(client):
    # Backward compatible: no llm configured (LLM_PROVIDER=none) → the
    # in-band candidate is left alone, only the >=0.92 threshold still fires.
    embed = FakeEmbed({
        "User likes Qdrant": _unit(0),
        "User is quite fond of Qdrant indeed": _unit(41),
    })
    memories.save_facts(client, embed, [{"text": "User likes Qdrant"}])
    r = memories.save_facts(client, embed, [{"text": "User is quite fond of Qdrant indeed"}])
    assert r[0]["status"] == "new"


# ---------------------------------------------------------------------------
# save_facts: contradiction detector — third, weakest-signal tier after
# triple-supersede and cosine-dedup. Flags, never auto-resolves.
# ---------------------------------------------------------------------------
def test_save_facts_llm_flags_contradiction_in_dedup_band(client):
    embed = FakeEmbed({
        "User prefers dark mode": _unit(0),
        "User now prefers light mode": _unit(41),  # in-band, not near-exact
    })
    memories.save_facts(client, embed, [{"text": "User prefers dark mode"}])
    llm = FakeLLM(lambda p: "yes" if "CONTRADICT" in p else "no")
    r = memories.save_facts(
        client, embed, [{"text": "User now prefers light mode"}], llm=llm
    )
    assert r[0]["status"] == "new"
    assert r[0]["conflicts_with"]

    facts = {f["text"]: f for f in memories.list_facts(client)}
    assert facts["User prefers dark mode"]["superseded_by"] is None
    assert (facts["User prefers dark mode"]["conflicts_with"]
            == facts["User now prefers light mode"]["id"])
    assert (facts["User now prefers light mode"]["conflicts_with"]
            == facts["User prefers dark mode"]["id"])


def test_save_facts_no_contradiction_stays_plain(client):
    embed = FakeEmbed({
        "User likes Qdrant": _unit(0),
        "Something loosely related to Qdrant": _unit(41),
    })
    memories.save_facts(client, embed, [{"text": "User likes Qdrant"}])
    llm = FakeLLM("no")
    r = memories.save_facts(
        client, embed, [{"text": "Something loosely related to Qdrant"}], llm=llm
    )
    assert "conflicts_with" not in r[0]
    assert all(f.get("conflicts_with") is None for f in memories.list_facts(client))


def test_save_facts_contradiction_skipped_without_llm(client):
    embed = FakeEmbed({
        "User prefers dark mode": _unit(0),
        "User now prefers light mode": _unit(41),
    })
    memories.save_facts(client, embed, [{"text": "User prefers dark mode"}])
    r = memories.save_facts(client, embed, [{"text": "User now prefers light mode"}])
    assert "conflicts_with" not in r[0]


def test_save_facts_contradiction_disabled_via_config(client, monkeypatch):
    monkeypatch.setattr(config, "CONTRADICTION_DETECTION", False)
    embed = FakeEmbed({
        "User prefers dark mode": _unit(0),
        "User now prefers light mode": _unit(41),
    })
    memories.save_facts(client, embed, [{"text": "User prefers dark mode"}])
    llm = FakeLLM(lambda p: "yes" if "CONTRADICT" in p else "no")
    r = memories.save_facts(
        client, embed, [{"text": "User now prefers light mode"}], llm=llm
    )
    assert "conflicts_with" not in r[0]
    assert not any("CONTRADICT" in c for c in llm.calls)


def test_save_facts_triple_supersede_skips_contradiction_check(client):
    # A has no triple but sits in the cosine dedup band of B; B carries a
    # triple that matches nothing existing (triple-supersede finds no
    # candidate). The contradiction check must still be skipped for B
    # because it has a triple, even though best_relevant (A) exists.
    embed = FakeEmbed({
        "Something in the dedup band": _unit(0),
        "A fresh fact with its own triple": _unit(41),
    })
    memories.save_facts(client, embed, [{"text": "Something in the dedup band"}])
    llm = FakeLLM("no")  # keeps the dedup band from superseding either side
    r = memories.save_facts(client, embed, [{
        "text": "A fresh fact with its own triple",
        "subject": "user", "relation": "favorite_color", "object": "blue",
    }], llm=llm)
    assert r[0]["status"] == "new"
    assert "conflicts_with" not in r[0]
    assert not any("CONTRADICT" in c for c in llm.calls)


def test_recall_surfaces_conflict_warning(client):
    embed = FakeEmbed({
        "User prefers dark mode": _unit(0),
        "User now prefers light mode": _unit(41),
        "what theme does the user like": _unit(0),
    })
    memories.save_facts(client, embed, [{"text": "User prefers dark mode"}])
    llm = FakeLLM(lambda p: "yes" if "CONTRADICT" in p else "no")
    memories.save_facts(
        client, embed, [{"text": "User now prefers light mode"}], llm=llm
    )
    result = memories.recall(client, embed, "what theme does the user like")
    assert "conflicts with another stored fact" in result["context_block"]


# ---------------------------------------------------------------------------
# save_facts: triple-based supersession — contradictions the cosine bands
# miss (same subject+relation, different object). Vectors are kept 60° apart
# (cos ≈ 0.5 < DEDUP_LLM_CHECK_MIN) so the cosine paths provably never fire.
# ---------------------------------------------------------------------------
def test_triple_supersedes_contradiction(client):
    embed = FakeEmbed({
        "User uses pnpm as package manager": _unit(0),
        "User switched the whole stack over to bun": _unit(60),
    })
    memories.save_facts(client, embed, [
        {"text": "User uses pnpm as package manager",
         "subject": "user", "relation": "package_manager", "object": "pnpm"},
    ])
    r = memories.save_facts(client, embed, [
        {"text": "User switched the whole stack over to bun",
         "subject": "user", "relation": "package_manager", "object": "bun"},
    ])
    assert r[0]["status"] == "supersedes"
    assert r[0]["superseded"] == ["User uses pnpm as package manager"]

    hits = memories.search_memories(client, embed, "User uses pnpm as package manager", top_k=5)
    texts = [h["text"] for h in hits]
    assert "User switched the whole stack over to bun" in texts
    assert "User uses pnpm as package manager" not in texts


def test_triple_same_object_supersedes_and_keeps_provenance(client):
    embed = FakeEmbed({
        "Primary language is Go": _unit(0),
        "User confirmed Go stays the primary language": _unit(60),
    })
    memories.save_facts(client, embed, [
        {"text": "Primary language is Go",
         "subject": "user", "relation": "primary_language", "object": "go"},
    ])
    r = memories.save_facts(client, embed, [
        {"text": "User confirmed Go stays the primary language",
         "subject": "user", "relation": "primary_language", "object": "go"},
    ])
    assert r[0]["status"] == "supersedes"
    old_id = memories.fact_point_id(
        config.USER_ID, "Primary language is Go", config.DEFAULT_PROJECT
    )
    old = client.retrieve(
        collection_name=config.MEMORIES_COLLECTION, ids=[old_id], with_payload=True
    )[0]
    assert old.payload["superseded_by"]


def test_triple_garbage_never_fails_the_save(client):
    embed = FakeEmbed({"fact a": _unit(0), "fact b": _unit(60), "fact c": _unit(120)})
    r = memories.save_facts(client, embed, [
        {"text": "fact a", "subject": "user"},  # missing keys
        {"text": "fact b", "subject": 42, "relation": "x", "object": "y"},  # non-string
        {"text": "fact c", "subject": "  ", "relation": "x", "object": "y"},  # empty after norm
    ])
    assert [x["status"] for x in r] == ["new", "new", "new"]
    b = client.retrieve(
        collection_name=config.MEMORIES_COLLECTION,
        ids=[memories.fact_point_id(config.USER_ID, "fact b", config.DEFAULT_PROJECT)],
        with_payload=True,
    )[0]
    assert "triple_subject" not in b.payload


def test_triple_matching_is_normalized(client):
    embed = FakeEmbed({
        "Package manager is pnpm": _unit(0),
        "Moved everything to bun": _unit(60),
    })
    memories.save_facts(client, embed, [
        {"text": "Package manager is pnpm",
         "subject": "user", "relation": "package_manager", "object": "pnpm"},
    ])
    r = memories.save_facts(client, embed, [
        {"text": "Moved everything to bun",
         "subject": " The User ", "relation": "Package Manager", "object": "Bun"},
    ])
    assert r[0]["status"] == "supersedes"


def test_triple_not_superseded_across_plain_projects(client):
    # Non-preference facts in another (non-default) project can never
    # co-appear in recall with the new fact, so they must not be retired.
    embed = FakeEmbed({
        "Alpha uses MySQL": _unit(0),
        "Alpha uses PostgreSQL per beta's notes": _unit(60),
    })
    memories.save_facts(client, embed, [
        {"text": "Alpha uses MySQL",
         "subject": "alpha", "relation": "database", "object": "mysql"},
    ], project_id="alpha")
    r = memories.save_facts(client, embed, [
        {"text": "Alpha uses PostgreSQL per beta's notes",
         "subject": "alpha", "relation": "database", "object": "postgres"},
    ], project_id="beta")
    assert r[0]["status"] == "new"


def test_triple_preference_superseded_globally(client):
    # Preferences are global in recall, so a stale one must be reachable
    # from any project.
    embed = FakeEmbed({
        "Comments in Vietnamese": _unit(0),
        "Comments in English from now on": _unit(60),
    })
    memories.save_facts(client, embed, [
        {"text": "Comments in Vietnamese", "type": "preference",
         "subject": "user", "relation": "comment_language", "object": "vi"},
    ], project_id="beta")
    r = memories.save_facts(client, embed, [
        {"text": "Comments in English from now on", "type": "preference",
         "subject": "user", "relation": "comment_language", "object": "en"},
    ], project_id="alpha")
    assert r[0]["status"] == "supersedes"


def test_triple_default_project_superseded_from_any_project(client):
    embed = FakeEmbed({
        "Editor is vim": _unit(0),
        "Editor is now zed": _unit(60),
    })
    memories.save_facts(client, embed, [
        {"text": "Editor is vim",
         "subject": "user", "relation": "editor", "object": "vim"},
    ])  # default project
    r = memories.save_facts(client, embed, [
        {"text": "Editor is now zed",
         "subject": "user", "relation": "editor", "object": "zed"},
    ], project_id="alpha")
    assert r[0]["status"] == "supersedes"


def test_triple_supersede_disabled_by_flag(client, monkeypatch):
    monkeypatch.setattr(config, "TRIPLE_SUPERSEDE", False)
    embed = FakeEmbed({
        "User uses pnpm as package manager": _unit(0),
        "User switched the whole stack over to bun": _unit(60),
    })
    memories.save_facts(client, embed, [
        {"text": "User uses pnpm as package manager",
         "subject": "user", "relation": "package_manager", "object": "pnpm"},
    ])
    r = memories.save_facts(client, embed, [
        {"text": "User switched the whole stack over to bun",
         "subject": "user", "relation": "package_manager", "object": "bun"},
    ])
    assert r[0]["status"] == "new"


def test_triple_leaves_legacy_facts_alone(client):
    # A legacy fact saved without a triple has no matching key — the new
    # triple-bearing fact must not touch it (cosine fallback still governs).
    embed = FakeEmbed({
        "User uses pnpm as package manager": _unit(0),
        "User switched the whole stack over to bun": _unit(60),
    })
    memories.save_facts(client, embed, [
        {"text": "User uses pnpm as package manager"},
    ])
    r = memories.save_facts(client, embed, [
        {"text": "User switched the whole stack over to bun",
         "subject": "user", "relation": "package_manager", "object": "bun"},
    ])
    assert r[0]["status"] == "new"
    hits = memories.search_memories(client, embed, "User uses pnpm as package manager", top_k=5)
    assert "User uses pnpm as package manager" in [h["text"] for h in hits]


# ---------------------------------------------------------------------------
# save_facts: meta-about-the-assistant filter (belt and braces on the
# consolidation prompt's own exclusion rule)
# ---------------------------------------------------------------------------
def test_save_facts_rejects_meta_about_assistant(client):
    embed = FakeEmbed()
    r = memories.save_facts(
        client, embed,
        [{"text": "User set Sonnet as the default model in Claude Code for new sessions."}],
    )
    assert r[0]["status"] == "rejected_meta"
    assert memories.list_facts(client) == []


# ---------------------------------------------------------------------------
# save_facts: a missing session_id gets a synthetic one so facts from one
# direct save_memories call still group together (provenance for /ui)
# ---------------------------------------------------------------------------
def test_save_facts_synthesizes_session_id_when_missing(client):
    embed = FakeEmbed({"fact one": _unit(0), "fact two": _unit(90)})
    memories.save_facts(client, embed, [{"text": "fact one"}, {"text": "fact two"}])
    points, _ = client.scroll(collection_name=config.MEMORIES_COLLECTION, limit=10, with_payload=True)
    sids = {p.payload["session_id"] for p in points}
    assert len(sids) == 1  # both facts share ONE synthetic session
    assert next(iter(sids)).startswith("direct:")


def test_save_facts_keeps_explicit_session_id(client):
    embed = FakeEmbed()
    memories.save_facts(client, embed, [{"text": "from a real session"}], session_id="s99")
    points, _ = client.scroll(collection_name=config.MEMORIES_COLLECTION, limit=10, with_payload=True)
    assert points[0].payload["session_id"] == "s99"


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
# last_seen: recall refreshes the decay clock (config.LAST_SEEN_REFRESH)
# ---------------------------------------------------------------------------
def test_search_memories_refreshes_last_seen_on_hit(client):
    embed = FakeEmbed({"old fact": _unit(0), "the query": _unit(0)})
    memories.save_facts(client, embed, [{"text": "old fact"}])
    facts = memories.list_facts(client)
    fact_id = facts[0]["id"]
    backdated = facts[0]["created_at"] - 60 * 86400
    client.set_payload(
        collection_name=config.MEMORIES_COLLECTION,
        payload={"created_at": backdated, "last_seen": backdated},
        points=[fact_id],
    )
    memories.search_memories(client, embed, "the query", top_k=5)
    refreshed = memories.list_facts(client)[0]
    assert refreshed["last_seen"] > backdated + 30 * 86400


def test_decay_uses_last_seen_not_created_at(client):
    # Same embedding for both facts (isolates the decay math from
    # similarity); different projects so the save-time supersede check
    # (scoped to one project) doesn't collapse them into one fact.
    embed = FakeEmbed({
        "seen recently": _unit(0),
        "seen long ago": _unit(0),
        "the query": _unit(0),
    })
    memories.save_facts(client, embed, [{"text": "seen recently"}], project_id="p1")
    memories.save_facts(client, embed, [{"text": "seen long ago"}], project_id="p2")
    old_age = 60 * 86400
    for f in memories.list_facts(client):
        if f["text"] == "seen long ago":
            backdated = f["created_at"] - old_age
            client.set_payload(
                collection_name=config.MEMORIES_COLLECTION,
                payload={"created_at": backdated, "last_seen": backdated},
                points=[f["id"]],
            )
    hits = memories.search_memories(client, embed, "the query", top_k=5)
    scores = {h["text"]: h["score"] for h in hits}
    assert scores["seen recently"] > scores["seen long ago"]


def test_last_seen_refresh_disabled_via_config(client, monkeypatch):
    monkeypatch.setattr(config, "LAST_SEEN_REFRESH", False)
    embed = FakeEmbed({"old fact": _unit(0), "the query": _unit(0)})
    memories.save_facts(client, embed, [{"text": "old fact"}])
    facts = memories.list_facts(client)
    fact_id = facts[0]["id"]
    backdated = facts[0]["created_at"] - 200 * 86400
    client.set_payload(
        collection_name=config.MEMORIES_COLLECTION,
        payload={"created_at": backdated, "last_seen": backdated},
        points=[fact_id],
    )
    memories.search_memories(client, embed, "the query", top_k=5)
    refreshed = memories.list_facts(client)[0]
    assert refreshed["last_seen"] == backdated


def test_search_memories_backward_compatible_without_last_seen(client):
    # Simulates a fact saved before this field existed: no last_seen key.
    embed = FakeEmbed({"legacy fact": _unit(0), "the query": _unit(0)})
    vector = embed.get_text_embedding("legacy fact")
    client.upsert(
        collection_name=config.MEMORIES_COLLECTION,
        points=[qmodels.PointStruct(
            id="11111111-1111-1111-1111-111111111111",
            vector=vector,
            payload={
                "user_id": config.USER_ID, "session_id": "s1",
                "project_id": config.DEFAULT_PROJECT, "type": "fact",
                "text": "legacy fact", "importance": 0.5,
                "created_at": time.time() - 10 * 86400,
            },
        )],
    )
    hits = memories.search_memories(client, embed, "the query", top_k=5)
    assert [h["text"] for h in hits] == ["legacy fact"]


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


# ---------------------------------------------------------------------------
# health_stats: /ui dashboard aggregation
# ---------------------------------------------------------------------------
def test_health_stats_counts_by_type_and_project(client):
    embed = FakeEmbed({
        "fact one": _unit(0), "decision one": _unit(30), "fact two": _unit(60),
    })
    memories.save_facts(client, embed, [
        {"text": "fact one", "type": "fact"},
        {"text": "decision one", "type": "decision"},
    ], project_id="p1")
    memories.save_facts(client, embed, [{"text": "fact two", "type": "fact"}], project_id="p2")

    stats = memories.health_stats(client)
    assert stats["by_type"] == {"fact": 2, "decision": 1}
    assert stats["by_project"] == {"p1": 2, "p2": 1}
    assert stats["total_active"] == 3
    assert stats["total_superseded"] == 0
    assert stats["superseded_ratio"] == 0.0


def test_health_stats_superseded_ratio(client):
    embed = FakeEmbed({
        "old version": _unit(0),
        "new version here": _unit(3),  # supersedes old
    })
    memories.save_facts(client, embed, [{"text": "old version"}])
    memories.save_facts(client, embed, [{"text": "new version here"}])
    stats = memories.health_stats(client)
    assert stats["total_active"] == 1
    assert stats["total_superseded"] == 1
    assert stats["superseded_ratio"] == 0.5


def test_health_stats_tracks_task_status_and_conflicts(client):
    embed = FakeEmbed({
        "ship it": _unit(80),
        "prefers dark mode": _unit(0),
        "now prefers light mode": _unit(41),
    })
    memories.save_facts(client, embed, [{"text": "ship it", "type": "task"}])
    fact_id = memories.list_facts(client)[0]["id"]
    memories.set_fact_status(client, fact_id, "done")
    memories.save_facts(client, embed, [{"text": "prefers dark mode"}])
    llm = FakeLLM(lambda p: "yes" if "CONTRADICT" in p else "no")
    memories.save_facts(client, embed, [{"text": "now prefers light mode"}], llm=llm)

    stats = memories.health_stats(client)
    assert stats["open_tasks"] == 0
    assert stats["done_tasks"] == 1
    assert stats["flagged_conflicts"] == 2  # both sides of the flagged pair


def test_health_stats_excludes_session_summaries(client):
    embed = FakeEmbed()
    memories.save_session_summary(client, embed, "s1", "summary text")
    stats = memories.health_stats(client)
    assert stats["total_active"] == 0
    assert stats["by_type"] == {}


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


def test_set_fact_type(client):
    embed = FakeEmbed()
    memories.save_facts(client, embed, [{"text": "reclassify me", "type": "fact"}])
    fact_id = memories.list_facts(client)[0]["id"]

    assert memories.set_fact_type(client, fact_id, "decision") is True
    assert memories.list_facts(client)[0]["type"] == "decision"
    assert memories.set_fact_type(client, fact_id, "not-a-real-type") is False
    assert memories.list_facts(client)[0]["type"] == "decision"  # unchanged
    assert memories.set_fact_type(client, "00000000-0000-0000-0000-00000000dead", "task") is False


# ---------------------------------------------------------------------------
# status: open/done for task-type facts
# ---------------------------------------------------------------------------
def test_set_fact_status_open_to_done_and_back(client):
    embed = FakeEmbed()
    memories.save_facts(client, embed, [{"text": "ship the thing", "type": "task"}])
    fact_id = memories.list_facts(client)[0]["id"]

    assert memories.list_facts(client)[0]["status"] is None  # no field = open
    assert memories.set_fact_status(client, fact_id, "done") is True
    assert memories.list_facts(client, include_done=True)[0]["status"] == "done"
    assert memories.set_fact_status(client, fact_id, "open") is True
    assert memories.list_facts(client)[0]["status"] == "open"


def test_set_fact_status_rejects_invalid_value(client):
    embed = FakeEmbed()
    memories.save_facts(client, embed, [{"text": "ship the thing", "type": "task"}])
    fact_id = memories.list_facts(client)[0]["id"]
    assert memories.set_fact_status(client, fact_id, "archived") is False
    assert memories.list_facts(client)[0]["status"] is None


def test_set_fact_status_unknown_id_returns_false(client):
    assert memories.set_fact_status(client, "00000000-0000-0000-0000-00000000dead", "done") is False


def test_search_memories_hides_done_tasks_by_default(client):
    embed = FakeEmbed({"finish the report": _unit(0), "the query": _unit(0)})
    memories.save_facts(client, embed, [{"text": "finish the report", "type": "task"}])
    fact_id = memories.list_facts(client)[0]["id"]
    memories.set_fact_status(client, fact_id, "done")
    hits = memories.search_memories(client, embed, "the query", top_k=5)
    assert "finish the report" not in [h["text"] for h in hits]


def test_search_memories_shows_done_tasks_when_hide_disabled(client, monkeypatch):
    monkeypatch.setattr(config, "HIDE_DONE_TASKS", False)
    embed = FakeEmbed({"finish the report": _unit(0), "the query": _unit(0)})
    memories.save_facts(client, embed, [{"text": "finish the report", "type": "task"}])
    fact_id = memories.list_facts(client, include_done=True)[0]["id"]
    memories.set_fact_status(client, fact_id, "done")
    hits = memories.search_memories(client, embed, "the query", top_k=5)
    assert "finish the report" in [h["text"] for h in hits]


def test_list_facts_include_done_toggle(client):
    embed = FakeEmbed()
    memories.save_facts(client, embed, [{"text": "closed task", "type": "task"}])
    fact_id = memories.list_facts(client)[0]["id"]
    memories.set_fact_status(client, fact_id, "done")
    assert memories.list_facts(client) == []
    assert len(memories.list_facts(client, include_done=True)) == 1


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


# ---------------------------------------------------------------------------
# project-scoped dedup/supersede (facts in different projects are distinct)
# ---------------------------------------------------------------------------
def test_same_text_in_two_projects_both_kept(client):
    embed = FakeEmbed()
    r1 = memories.save_facts(client, embed, [{"text": "Deploy uses Docker"}],
                             project_id="proj-a")
    r2 = memories.save_facts(client, embed, [{"text": "Deploy uses Docker"}],
                             project_id="proj-b")
    assert r1[0]["status"] == "new"
    assert r2[0]["status"] == "new"  # neither "duplicate" nor "supersedes"
    active = memories.list_facts(client)
    assert sorted(f["project_id"] for f in active) == ["proj-a", "proj-b"]


def test_near_duplicate_supersedes_only_within_project(client):
    embed = FakeEmbed({
        "Deadline is Friday": _unit(0),
        "Deadline is on Friday": _unit(5),  # cos(5°) ≈ 0.996 ≥ 0.92
    })
    memories.save_facts(client, embed, [{"text": "Deadline is Friday"}],
                        project_id="proj-a")
    # Other project: near-identical text must NOT retire proj-a's fact.
    r_other = memories.save_facts(client, embed, [{"text": "Deadline is on Friday"}],
                                  project_id="proj-b")
    assert r_other[0]["status"] == "new"
    # Same project: the usual supersede still fires.
    r_same = memories.save_facts(client, embed, [{"text": "Deadline is on Friday"}],
                                 project_id="proj-a")
    assert r_same[0]["status"] == "supersedes"
    assert r_same[0]["superseded"] == ["Deadline is Friday"]


def test_retagged_fact_same_text_supersedes_not_duplicate(client):
    """Documented residual of project-scoped ids: /ui can retag a fact to a
    new project, but its point id still carries the ORIGINAL project. An
    exact re-save under the new project then lands as a fresh point and
    retires the retagged one via supersede — not as an id-level "duplicate".
    This test locks that behavior."""
    embed = FakeEmbed()
    memories.save_facts(client, embed, [{"text": "Movable fact"}], project_id="proj-a")
    old_id = memories.fact_point_id(config.USER_ID, "Movable fact", "proj-a")
    client.set_payload(collection_name=config.MEMORIES_COLLECTION,
                       payload={"project_id": "proj-b"}, points=[old_id])
    r = memories.save_facts(client, embed, [{"text": "Movable fact"}], project_id="proj-b")
    assert r[0]["status"] == "supersedes"
    assert r[0]["superseded"] == ["Movable fact"]


# ---------------------------------------------------------------------------
# importance parsing (LLM consolidation output is untrusted)
# ---------------------------------------------------------------------------
def test_unparseable_importance_defaults_instead_of_crashing(client):
    # Far-apart vectors: these three must not fall into each other's
    # supersede band — this test is about importance parsing only.
    embed = FakeEmbed({
        "importance is a word": _unit(0),
        "importance is null": _unit(80),
        "importance out of range": _unit(160),
    })
    r = memories.save_facts(client, embed, [
        {"text": "importance is a word", "importance": "high"},
        {"text": "importance is null", "importance": None},
        {"text": "importance out of range", "importance": 7},
    ])
    assert [x["status"] for x in r] == ["new", "new", "new"]
    by_text = {f["text"]: f["importance"] for f in memories.list_facts(client)}
    assert by_text["importance is a word"] == 0.5
    assert by_text["importance is null"] == 0.5
    assert by_text["importance out of range"] == 1.0  # clamped, not defaulted


# ---------------------------------------------------------------------------
# recall context block carries agent provenance
# ---------------------------------------------------------------------------
def test_recall_context_block_shows_source_agent(client):
    # Both facts sit close to the query (0°) but 30° apart from each other:
    # similar enough to be recalled, distinct enough not to supersede.
    embed = FakeEmbed({
        "tagged by claude": _unit(15),
        "untagged legacy fact": _unit(-15),
    })
    memories.save_facts(client, embed, [{"text": "tagged by claude"}],
                        source_agent="claude-code")
    memories.save_facts(client, embed, [{"text": "untagged legacy fact"}])
    memory_store.add_message(client, embed, "s-hist", "user", "hello from hermes",
                             source_agent="hermes")
    block = memories.recall(client, embed, "anything", recent_turns=0)["context_block"]
    assert ", claude-code) tagged by claude" in block
    assert ", hermes) user: hello from hermes" in block
    # a record without the tag renders without a dangling separator
    assert ") untagged legacy fact" in block and ", ) untagged" not in block


# ---------------------------------------------------------------------------
# recall router: docs channel on trigger words only
# ---------------------------------------------------------------------------
def _seed_doc_chunk(client, text, source, project, vector):
    import json as _json
    import uuid as _uuid

    from qdrant_client.http import models as qmodels
    if not client.collection_exists(config.DOCUMENTS_COLLECTION):
        # In production LlamaIndex creates this collection lazily on the
        # first document insert; tests seed raw points, so create it here.
        client.create_collection(
            collection_name=config.DOCUMENTS_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=DIM, distance=qmodels.Distance.COSINE),
        )
    client.upsert(
        collection_name=config.DOCUMENTS_COLLECTION,
        points=[qmodels.PointStruct(
            id=str(_uuid.uuid4()), vector=vector,
            payload={"_node_content": _json.dumps({"text": text}),
                     "source": source, "project_id": project},
        )],
    )


def test_route_query_triggers():
    assert memories.route_query("xem tài liệu spec giúp tôi")["docs"] is True
    assert memories.route_query("エラーの指示書を確認して")["docs"] is True
    assert memories.route_query("sửa bug này đi")["docs"] is False
    assert memories.route_query("lần trước đã bàn gì?")["history_hint"] is True
    assert memories.route_query("build lại service")["history_hint"] is False


def test_recall_includes_docs_only_when_triggered(client):
    embed = FakeEmbed()
    _seed_doc_chunk(client, "File size limit is 10MB per upload.",
                    "upload-spec.md", "proj-a", [1.0, 0.0])
    triggered = memories.recall(client, embed, "tài liệu spec nói gì về upload?",
                                project="proj-a", recent_turns=0)
    assert triggered["routing"]["docs"] is True
    assert "[Project documents]" in triggered["context_block"]
    assert "10MB" in triggered["context_block"]
    assert triggered["documents"][0]["source"] == "upload-spec.md"

    untriggered = memories.recall(client, embed, "upload đang lỗi gì?",
                                  project="proj-a", recent_turns=0)
    assert untriggered["routing"]["docs"] is False
    assert "[Project documents]" not in untriggered["context_block"]
    assert untriggered["documents"] == []


def test_recall_docs_hard_filtered_by_project(client):
    embed = FakeEmbed()
    _seed_doc_chunk(client, "Booking slots every 15 minutes.",
                    "booking-spec.md", "proj-b", [1.0, 0.0])
    r = memories.recall(client, embed, "tài liệu spec về slot?",
                        project="proj-a", recent_turns=0)
    assert r["documents"] == []  # proj-b's doc must not cross into proj-a


# ---------------------------------------------------------------------------
# session summaries: stored per session, replace raw snippets in recall,
# never leak into fact search / listings
# ---------------------------------------------------------------------------
def test_session_summary_upsert_overwrites(client):
    embed = FakeEmbed()
    memories.save_session_summary(client, embed, "s-sum", "first version")
    memories.save_session_summary(client, embed, "s-sum", "second version")
    got = memories.get_session_summaries(client, ["s-sum"])
    assert got["s-sum"]["text"] == "second version"


def test_recall_prefers_summary_over_raw_snippets(client):
    embed = FakeEmbed()
    memory_store.add_message(client, embed, "s-old", "assistant",
                             "raw detail about the deploy fix")
    memory_store.add_message(client, embed, "s-none", "assistant",
                             "another session without summary")
    memories.save_session_summary(
        client, embed, "s-old",
        "Fixed staging deploy: healthcheck start_period raised to 30s.",
        source_agent="claude-code")
    r = memories.recall(client, embed, "deploy fix?", recent_turns=0)
    block = r["context_block"]
    assert "[Session summaries" in block
    assert "start_period raised" in block
    assert ", claude-code)" in block
    # raw snippet of the summarized session is suppressed...
    assert "raw detail about the deploy fix" not in block
    # ...but sessions without a summary still show raw snippets
    assert "another session without summary" in block


def test_summaries_excluded_from_fact_search_and_listing(client):
    embed = FakeEmbed()
    memories.save_session_summary(client, embed, "s-x", "summary text here")
    memories.save_facts(client, embed, [{"text": "a real fact"}])
    hits = memories.search_memories(client, embed, "anything", top_k=10)
    assert all(h["type"] != "session_summary" for h in hits)
    assert all(f["type"] != "session_summary" for f in memories.list_facts(client))
    graph_types = {n["type"] for n in memories.graph_data(client)["nodes"]}
    assert "session_summary" not in graph_types


# ---------------------------------------------------------------------------
# global vs project scope: other projects' facts/history stay out of
# auto-recall; preferences and default-project knowledge are global
# ---------------------------------------------------------------------------
def test_project_query_excludes_other_projects_facts(client):
    embed = FakeEmbed()
    memories.save_facts(client, embed, [{"text": "proj-b decision about imports"}],
                        project_id="proj-b")
    hits = memories.search_memories(client, embed, "imports?", project="proj-a")
    assert hits == []  # hard-scoped out, no longer just down-ranked


def test_preferences_and_default_project_are_global(client):
    embed = FakeEmbed()
    memories.save_facts(client, embed,
                        [{"text": "commit without extra trailers", "type": "preference"}],
                        project_id="proj-b")  # preference saved under ANOTHER project
    memories.save_facts(client, embed, [{"text": "user timezone is UTC+7"}])  # default project
    texts = [h["text"] for h in
             memories.search_memories(client, embed, "anything", project="proj-a", top_k=10)]
    assert "commit without extra trailers" in texts
    assert "user timezone is UTC+7" in texts


def test_history_scoped_to_project_plus_default(client):
    embed = FakeEmbed()
    memory_store.add_message(client, embed, "s-b", "assistant", "proj-b deploy detail",
                             project_id="proj-b")
    memory_store.add_message(client, embed, "s-d", "assistant", "default session detail",
                             project_id="default")
    contents = [h["content"] for h in memory_store.search_history(
        client, embed, "detail?", project="proj-a", top_k=10)]
    assert "proj-b deploy detail" not in contents
    assert "default session detail" in contents


def test_no_project_means_no_scoping(client):
    embed = FakeEmbed()
    memories.save_facts(client, embed, [{"text": "proj-b only fact"}], project_id="proj-b")
    texts = [h["text"] for h in memories.search_memories(client, embed, "fact?", top_k=10)]
    assert "proj-b only fact" in texts
