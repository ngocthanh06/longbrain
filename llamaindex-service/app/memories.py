"""Semantic memory (L3): distilled long-term facts in Qdrant.

Facts are what consolidation extracts from raw conversation turns —
decisions, preferences, project constraints, tasks. Saving is idempotent
(deterministic point id from normalized text) and contradiction-aware: a new
fact that is near-identical in vector space to an existing one supersedes it
instead of piling up duplicates, and superseded facts stay on disk for
provenance but are excluded from recall.
"""

import hashlib
import math
import re
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app import config, documents, hybrid, memory_store

_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

VALID_TYPES = {"fact", "preference", "decision", "task"}

# Shared fact-hygiene filter, applied at every entry point (consolidation's
# LLM extraction, direct REST /memory/facts, direct MCP save_memories):
# the prompt already tells the extracting LLM not to save meta-commentary
# about the assistant/tool itself, but it slips through often enough
# (observed in production: a mid-session model switch saved as a
# "decision") to warrant a code-level net. Deliberately narrow —
# multi-word/contextual phrases, not single generic words — to keep false
# positives on unrelated legitimate facts rare.
_META_ABOUT_ASSISTANT_RE = re.compile(
    r"model mặc định|default model|mô hình mặc định|"
    r"claude code|hermes desktop|ai assistant|trợ lý ai|"
    r"phiên bản (của )?(claude|model|ai)|context window|system prompt",
    re.IGNORECASE,
)


def is_meta_about_assistant(text: str) -> bool:
    return bool(_META_ABOUT_ASSISTANT_RE.search(text))


def fact_point_id(user_id: str, text: str, project_id: str = "") -> str:
    """Deterministic id per (user, project, normalized text). The project is
    part of the identity: the same sentence can be a distinct fact in two
    projects, and exact-text dedup must not swallow it across them. Legacy
    points (pre-project ids) stay valid — a re-save of their text lands as a
    new point and the project-scoped supersede search retires the old one."""
    digest = hashlib.sha256(" ".join(text.split()).lower().encode("utf-8")).hexdigest()
    scope = project_id or config.DEFAULT_PROJECT
    return str(uuid.uuid5(_NAMESPACE, f"fact:{user_id}:{scope}:{digest}"))


def summary_point_id(user_id: str, session_id: str) -> str:
    """One summary per session: re-consolidating overwrites in place."""
    return str(uuid.uuid5(_NAMESPACE, f"summary:{user_id}:{session_id}"))


def _norm_key(s: str, as_relation: bool = False) -> str:
    """Normalize a triple part into a matching key: matching is exact-string,
    so casing/spacing/punctuation noise from the extracting LLM must not
    break it. Relations become snake_case; subjects drop a leading article."""
    s = " ".join(s.split()).casefold().rstrip(".,;:!?")
    if as_relation:
        s = re.sub(r"[\s\-]+", "_", s)
    else:
        s = re.sub(r"^(the|an|a)\s+", "", s)
    return s


def _triple_of(fact: dict) -> tuple[str, str, str] | None:
    """Validated, normalized (subject, relation, object) from a fact dict.
    The extracting LLM may emit anything here — a bad triple must never fail
    the save, so any non-string, empty-after-normalization or oversized part
    drops the whole triple while the fact itself is kept."""
    parts = (fact.get("subject"), fact.get("relation"), fact.get("object"))
    if not all(isinstance(p, str) for p in parts):
        return None
    subject, obj = _norm_key(parts[0]), _norm_key(parts[2])
    relation = _norm_key(parts[1], as_relation=True)
    if not (subject and relation and obj):
        return None
    if any(len(p) > 128 for p in (subject, relation, obj)):
        return None
    return subject, relation, obj


def _active_filter(user_id: str, extra: list | None = None) -> qmodels.Filter:
    must = [
        qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id)),
        qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="superseded_by")),
    ]
    if extra:
        must.extend(extra)
    return qmodels.Filter(must=must)


_DEDUP_PROMPT = """Are Fact A and Fact B stating the same underlying fact \
about the user or their project — one possibly a rewording, translation, \
summary, or more detailed version of the other?

Fact A: {a}
Fact B: {b}

Answer with exactly one word: yes or no."""


def _llm_confirms_duplicate(llm, new_text: str, existing_text: str) -> bool:
    try:
        answer = llm.complete(_DEDUP_PROMPT.format(a=new_text, b=existing_text)).text
    except Exception:
        return False  # best-effort — a flaky LLM call must not block saving
    return answer.strip().lower().startswith("yes")


def save_facts(
    client: QdrantClient,
    embed_model,
    facts: list[dict],
    user_id: str = config.USER_ID,
    session_id: str = "",
    project_id: str = config.DEFAULT_PROJECT,
    source_agent: str = "",
    llm=None,
) -> list[dict]:
    # A direct save (no session behind it — e.g. an agent calling save_memories
    # on its own initiative, not through consolidate_session) must still get a
    # session_id: without one, these facts can't be grouped in /ui (the "same
    # session" ring, "re-tag whole session") and provenance is lost. One
    # synthetic id per CALL, shared by every fact in this batch.
    if not session_id:
        session_id = f"direct:{uuid.uuid4().hex[:12]}"
    project_id = project_id or config.DEFAULT_PROJECT
    results = []
    for fact in facts:
        text = (fact.get("text") or "").strip()
        if not text:
            continue
        if is_meta_about_assistant(text):
            results.append({"text": text, "status": "rejected_meta"})
            continue
        ftype = fact.get("type", "fact")
        if ftype not in VALID_TYPES:
            ftype = "fact"
        try:
            importance = min(max(float(fact.get("importance", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            importance = 0.5  # LLM consolidation may emit "high"/null — don't fail the save

        point_id = fact_point_id(user_id, text, project_id)
        existing = client.retrieve(
            collection_name=config.MEMORIES_COLLECTION, ids=[point_id], with_payload=False
        )
        if existing:
            results.append({"text": text, "status": "duplicate"})
            continue

        vector = embed_model.get_text_embedding(text)

        # Triple-based supersession: same (subject, relation) with a different
        # object is a CONTRADICTION ("uses pnpm" -> "switched to bun") that
        # both cosine bands below miss — the texts score far apart, and the
        # LLM band only detects rewordings. Same object supersedes too: a
        # reconfirmed fact gets a fresh created_at instead of decaying from
        # its original date. Scope mirrors recall co-visibility
        # (search_memories): a fact may only retire what could have
        # co-appeared with it — same project or the default project, plus
        # preferences, which are global in recall.
        superseded = []
        triple = _triple_of(fact) if config.TRIPLE_SUPERSEDE else None
        if triple:
            candidates, _ = client.scroll(
                collection_name=config.MEMORIES_COLLECTION,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="user_id", match=qmodels.MatchValue(value=user_id)
                        ),
                        qmodels.IsEmptyCondition(
                            is_empty=qmodels.PayloadField(key="superseded_by")
                        ),
                        qmodels.FieldCondition(
                            key="triple_subject", match=qmodels.MatchValue(value=triple[0])
                        ),
                        qmodels.FieldCondition(
                            key="triple_relation", match=qmodels.MatchValue(value=triple[1])
                        ),
                    ],
                    should=[
                        qmodels.FieldCondition(
                            key="project_id",
                            match=qmodels.MatchAny(any=[project_id, config.DEFAULT_PROJECT]),
                        ),
                        qmodels.FieldCondition(
                            key="type", match=qmodels.MatchValue(value="preference")
                        ),
                    ],
                ),
                limit=16,
                with_payload=["text"],
            )
            for cand in candidates:
                if str(cand.id) == point_id:
                    continue
                client.set_payload(
                    collection_name=config.MEMORIES_COLLECTION,
                    payload={"superseded_by": point_id},
                    points=[cand.id],
                )
                superseded.append(cand.payload.get("text", ""))

        # A near-identical active fact gets superseded (updated info wins).
        # Two bands: >=SUPERSEDE_SIMILARITY is close enough to auto-supersede
        # (near-exact text); the wider DEDUP_LLM_CHECK_MIN..SUPERSEDE_SIMILARITY
        # band catches reworded/summarized restatements that score too low on
        # raw cosine to trust blindly (see config.py) — an LLM judges those,
        # and only when one is configured (LLM_PROVIDER=none skips the band).
        # Scope the supersede search to this fact's project: near-identical
        # facts in DIFFERENT projects are distinct knowledge, and letting one
        # retire the other silently loses the other project's memory.
        hits = client.search(
            collection_name=config.MEMORIES_COLLECTION,
            query_vector=vector,
            query_filter=_active_filter(user_id, extra=[
                qmodels.FieldCondition(
                    key="project_id", match=qmodels.MatchValue(value=project_id)
                ),
            ]),
            limit=5,
            score_threshold=config.DEDUP_LLM_CHECK_MIN,
            with_payload=["text"],
        )
        for hit in hits:
            is_dup = hit.score >= config.SUPERSEDE_SIMILARITY or (
                llm is not None and _llm_confirms_duplicate(llm, text, hit.payload.get("text", ""))
            )
            if is_dup:
                client.set_payload(
                    collection_name=config.MEMORIES_COLLECTION,
                    payload={"superseded_by": point_id},
                    points=[hit.id],
                )
                superseded.append(hit.payload.get("text", ""))

        now = time.time()
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "project_id": project_id or config.DEFAULT_PROJECT,
            "type": ftype,
            "text": text,
            "importance": importance,
            "created_at": now,
            "last_seen": now,
        }
        if source_agent:
            payload["source_agent"] = source_agent
        if triple:
            payload["triple_subject"] = triple[0]
            payload["triple_relation"] = triple[1]
            payload["triple_object"] = triple[2]
        client.upsert(
            collection_name=config.MEMORIES_COLLECTION,
            points=[qmodels.PointStruct(
                id=point_id,
                vector=hybrid.point_vector(client, config.MEMORIES_COLLECTION, vector, text),
                payload=payload,
            )],
        )
        results.append(
            {"text": text, "status": "supersedes" if superseded else "new",
             **({"superseded": superseded} if superseded else {})}
        )
    return results


def save_session_summary(
    client: QdrantClient,
    embed_model,
    session_id: str,
    text: str,
    user_id: str = config.USER_ID,
    project_id: str = config.DEFAULT_PROJECT,
    source_agent: str = "",
) -> dict:
    """Store the structured summary of a finished session (goal, decisions,
    unresolved). Lives in the memories collection as type=session_summary but
    is NOT a fact: excluded from fact search/listing, surfaced by recall in
    place of that session's raw snippets."""
    text = (text or "").strip()
    if not text or not session_id:
        return {"status": "skipped"}
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "project_id": project_id or config.DEFAULT_PROJECT,
        "type": "session_summary",
        "text": text,
        "created_at": time.time(),
    }
    if source_agent:
        payload["source_agent"] = source_agent
    client.upsert(
        collection_name=config.MEMORIES_COLLECTION,
        points=[qmodels.PointStruct(
            id=summary_point_id(user_id, session_id),
            vector=hybrid.point_vector(
                client, config.MEMORIES_COLLECTION,
                embed_model.get_text_embedding(text), text,
            ),
            payload=payload,
        )],
    )
    return {"status": "ok", "session_id": session_id}


def get_session_summaries(
    client: QdrantClient, session_ids, user_id: str = config.USER_ID
) -> dict:
    """session_id -> summary payload, for the given sessions only."""
    ids = [s for s in set(session_ids) if s]
    if not ids:
        return {}
    points, _ = client.scroll(
        collection_name=config.MEMORIES_COLLECTION,
        scroll_filter=qmodels.Filter(must=[
            qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id)),
            qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="session_summary")),
            qmodels.FieldCondition(key="session_id", match=qmodels.MatchAny(any=ids)),
        ]),
        limit=len(ids),
        with_payload=True,
        with_vectors=False,
    )
    return {
        p.payload["session_id"]: {
            "text": p.payload.get("text", ""),
            "created_at": p.payload.get("created_at"),
            "source_agent": p.payload.get("source_agent") or "",
        }
        for p in points
    }


def _decay(age_seconds: float, half_life_days: float) -> float:
    return math.pow(0.5, age_seconds / (half_life_days * 86400.0))


def route_query(query: str) -> dict:
    """Rule-based recall router (no LLM): which memory channels does this
    prompt actually need? Facts and history stay always-on (cheap, small);
    the documents channel activates only on trigger words ("tài liệu",
    "spec", "指示書", ...) because doc chunks are the biggest injection.
    history_hint is surfaced for observability/tuning, not yet gating."""
    q = query.lower()
    return {
        "docs": any(t in q for t in config.RECALL_DOC_TRIGGERS),
        "history_hint": any(t in q for t in config.RECALL_HISTORY_TRIGGERS),
    }


def search_memories(
    client: QdrantClient,
    embed_model,
    query: str,
    user_id: str = config.USER_ID,
    top_k: int = config.RECALL_TOP_K_MEMORIES,
    project: str | None = None,
) -> list[dict]:
    vector = embed_model.get_text_embedding(query)
    flt = _active_filter(user_id)
    # Session summaries share this collection but are not facts — recall
    # surfaces them per matched session, not in [Long-term memories].
    flt.must_not = [
        qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="session_summary"))
    ]
    if config.HIDE_DONE_TASKS:
        # A closed task is done, not deleted/superseded — stop surfacing it
        # in recall the same way a superseded fact stops surfacing. Absent
        # field never matches "done", so non-task facts are unaffected.
        flt.must_not.append(
            qmodels.FieldCondition(key="status", match=qmodels.MatchValue(value="done"))
        )
    if project:
        # Scope: project-anchored knowledge (facts/decisions/tasks) from
        # OTHER projects stays out of auto-recall — asking about project A
        # must not surface project B's decisions. Preferences are global by
        # nature (commit style, language, workflow) and pass regardless;
        # so does anything stored under the default project. Cross-project
        # lookups stay available through the explicit MCP search tools.
        flt.should = [
            qmodels.FieldCondition(
                key="project_id",
                match=qmodels.MatchAny(any=[project, config.DEFAULT_PROJECT]),
            ),
            qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="preference")),
        ]
    dense_hits = client.search(
        collection_name=config.MEMORIES_COLLECTION,
        query_vector=vector,
        query_filter=flt,
        limit=top_k * 3,  # over-fetch, then rerank by decayed/boosted score
        with_payload=True,
    )
    sparse_hits = hybrid.search(
        client, config.MEMORIES_COLLECTION, query, flt, top_k * 3
    )
    now = time.time()
    scored = []
    for e in hybrid.fuse(dense_hits, sparse_hits):
        payload = e["payload"]
        importance = payload.get("importance", 0.5)
        # Decay runs off last_seen, not created_at: recalling a fact is
        # itself evidence it's still relevant (see config.LAST_SEEN_REFRESH).
        # Facts saved before this field existed fall back to created_at.
        last_seen = payload.get("last_seen") or payload.get("created_at") or now
        age = max(now - last_seen, 0.0)
        final = e["similarity"] * _decay(age, config.MEMORY_HALF_LIFE_DAYS) * (0.5 + 0.5 * importance)
        hit_project = payload.get("project_id") or config.DEFAULT_PROJECT
        if project and hit_project == project:
            final *= config.RECALL_PROJECT_BOOST
        if payload.get("type") == "preference":
            # Standing conventions must survive the top-k race against
            # fresher, project-boosted same-topic facts at the moment of
            # acting on them (see config.RECALL_PREFERENCE_BOOST).
            final *= config.RECALL_PREFERENCE_BOOST
        scored.append(
            {
                "id": str(e["id"]),
                "text": payload["text"],
                "type": payload.get("type", "fact"),
                "project_id": hit_project,
                "importance": importance,
                "created_at": payload.get("created_at"),
                "last_seen": last_seen,
                "source_agent": payload.get("source_agent") or "",
                "similarity": e["similarity"],
                "score": final,
            }
        )
    scored.sort(key=lambda m: m["score"], reverse=True)
    top = [m for m in scored if m["score"] >= config.RECALL_MIN_SCORE][:top_k]
    if config.LAST_SEEN_REFRESH and top:
        client.set_payload(
            collection_name=config.MEMORIES_COLLECTION,
            payload={"last_seen": now},
            points=[m["id"] for m in top],
        )
    return top


def list_facts(
    client: QdrantClient,
    user_id: str = config.USER_ID,
    project: str | None = None,
    ftype: str | None = None,
    include_superseded: bool = False,
    include_done: bool = False,
    limit: int = 200,
) -> list[dict]:
    must = [
        qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))
    ]
    if not include_superseded:
        must.append(qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="superseded_by")))
    if project:
        must.append(qmodels.FieldCondition(key="project_id", match=qmodels.MatchValue(value=project)))
    if ftype:
        must.append(qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value=ftype)))
    flt = qmodels.Filter(must=must)
    flt.must_not = []
    if ftype != "session_summary":
        # Summaries are per-session artifacts, not facts — keep them out of
        # the /ui graph and fact listings unless asked for explicitly.
        flt.must_not.append(
            qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="session_summary"))
        )
    if not include_done:
        flt.must_not.append(
            qmodels.FieldCondition(key="status", match=qmodels.MatchValue(value="done"))
        )
    points, _ = client.scroll(
        collection_name=config.MEMORIES_COLLECTION,
        scroll_filter=flt,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    points.sort(key=lambda p: p.payload.get("created_at", 0), reverse=True)
    return [
        {
            "id": str(p.id),
            "text": p.payload.get("text"),
            "type": p.payload.get("type", "fact"),
            "project_id": p.payload.get("project_id") or config.DEFAULT_PROJECT,
            "importance": p.payload.get("importance", 0.5),
            "created_at": p.payload.get("created_at"),
            "last_seen": p.payload.get("last_seen") or p.payload.get("created_at"),
            "superseded_by": p.payload.get("superseded_by"),
            "status": p.payload.get("status"),
            "source_agent": p.payload.get("source_agent") or "",
        }
        for p in points
    ]


def set_fact_project(client: QdrantClient, fact_id: str, project_id: str) -> bool:
    """Re-tag one fact (user correction from /ui). False if the id is unknown."""
    existing = client.retrieve(
        collection_name=config.MEMORIES_COLLECTION, ids=[fact_id], with_payload=False
    )
    if not existing:
        return False
    client.set_payload(
        collection_name=config.MEMORIES_COLLECTION,
        payload={"project_id": project_id},
        points=[fact_id],
    )
    return True


def set_fact_type(client: QdrantClient, fact_id: str, ftype: str) -> bool:
    """Change a fact's type (user correction from /ui — the consolidation
    LLM's fact/preference/decision/task call isn't always right). False if
    the id is unknown or ftype isn't one of VALID_TYPES."""
    if ftype not in VALID_TYPES:
        return False
    existing = client.retrieve(
        collection_name=config.MEMORIES_COLLECTION, ids=[fact_id], with_payload=False
    )
    if not existing:
        return False
    client.set_payload(
        collection_name=config.MEMORIES_COLLECTION,
        payload={"type": ftype},
        points=[fact_id],
    )
    return True


VALID_STATUSES = {"open", "done"}


def set_fact_status(client: QdrantClient, fact_id: str, status: str) -> bool:
    """Close/reopen a task fact (/ui action). False if the id is unknown or
    status isn't open/done. No status set means "open" (see HIDE_DONE_TASKS)."""
    if status not in VALID_STATUSES:
        return False
    existing = client.retrieve(
        collection_name=config.MEMORIES_COLLECTION, ids=[fact_id], with_payload=False
    )
    if not existing:
        return False
    client.set_payload(
        collection_name=config.MEMORIES_COLLECTION,
        payload={"status": status},
        points=[fact_id],
    )
    return True


def set_facts_project(
    client: QdrantClient, fact_ids: list, project_id: str
) -> int:
    """Bulk re-tag facts by id (multi-select in /ui). Unknown ids are
    silently skipped; returns the number actually re-tagged."""
    existing = client.retrieve(
        collection_name=config.MEMORIES_COLLECTION, ids=list(fact_ids), with_payload=False
    )
    ids = [p.id for p in existing]
    if ids:
        client.set_payload(
            collection_name=config.MEMORIES_COLLECTION,
            payload={"project_id": project_id},
            points=ids,
        )
    return len(ids)


def set_session_facts_project(
    client: QdrantClient, session_id: str, project_id: str,
    user_id: str = config.USER_ID,
) -> int:
    """Re-tag every fact distilled from a session. Returns affected count."""
    flt = qmodels.Filter(must=[
        qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id)),
        qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id)),
    ])
    count = client.count(
        collection_name=config.MEMORIES_COLLECTION, count_filter=flt, exact=True
    ).count
    if count:
        client.set_payload(
            collection_name=config.MEMORIES_COLLECTION,
            payload={"project_id": project_id},
            points=qmodels.FilterSelector(filter=flt),
        )
    return count


def entity_graph_data(client: QdrantClient, user_id: str = config.USER_ID) -> dict:
    """Entities and relations from fact triples, for the /ui graph's entity
    mode. Nodes are the distinct subjects/objects of active triple-bearing
    facts; each fact contributes one labelled edge. Node shape matches
    graph_data so the Canvas renderer needs no new code."""
    from collections import Counter

    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id)),
            qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="superseded_by")),
        ],
        must_not=[
            qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="triple_subject")),
        ],
    )
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=config.MEMORIES_COLLECTION, scroll_filter=flt,
            limit=256, offset=offset, with_payload=True,
        )
        points.extend(batch)
        if offset is None:
            break

    degree: Counter = Counter()
    projects: dict[str, Counter] = {}
    edges = []
    for p in points:
        subject = p.payload.get("triple_subject", "")
        obj = p.payload.get("triple_object", "")
        project = p.payload.get("project_id") or config.DEFAULT_PROJECT
        edges.append({
            "source": f"ent:{subject}",
            "target": f"ent:{obj}",
            "weight": 0.8,
            "label": p.payload.get("triple_relation", ""),
            "fact_id": str(p.id),
        })
        for name in (subject, obj):
            degree[name] += 1
            projects.setdefault(name, Counter())[project] += 1

    nodes = [
        {
            "id": f"ent:{name}",
            "text": name,
            "type": "entity",
            "project_id": projects[name].most_common(1)[0][0],
            "importance": min(1.0, degree[name] / 5),
            "created_at": None,
            "session_id": "",
            "superseded": False,
            "source_agent": "",
        }
        for name in degree
    ]
    return {"nodes": nodes, "edges": edges}


def delete_fact(client: QdrantClient, fact_id: str) -> bool:
    """Hard-delete a fact (user-initiated forget — distinct from supersede,
    which is the natural update path and keeps provenance)."""
    existing = client.retrieve(
        collection_name=config.MEMORIES_COLLECTION, ids=[fact_id], with_payload=False
    )
    if not existing:
        return False
    client.delete(
        collection_name=config.MEMORIES_COLLECTION,
        points_selector=qmodels.PointIdsList(points=[fact_id]),
    )
    return True


def delete_all_facts(client: QdrantClient, user_id: str = config.USER_ID) -> int:
    """Wipe every fact for the user — INCLUDING superseded provenance points
    (a full reset should leave no residue in the dashboard)."""
    flt = qmodels.Filter(must=[
        qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))
    ])
    count = client.count(
        collection_name=config.MEMORIES_COLLECTION, count_filter=flt, exact=True
    ).count
    if count:
        client.delete(
            collection_name=config.MEMORIES_COLLECTION,
            points_selector=qmodels.FilterSelector(filter=flt),
        )
    return count


def graph_data(
    client: QdrantClient,
    user_id: str = config.USER_ID,
    include_superseded: bool = False,
    top_edges: int = 4,
    min_similarity: float = 0.35,
    limit: int = 2000,
) -> dict:
    """Nodes + similarity edges over the fact collection, for the /ui graph.

    Each node keeps its top_edges most similar neighbours above
    min_similarity. Read-only; O(n²) similarity is fine at personal-memory
    scale (hundreds of facts)."""
    import numpy as np

    must = [
        qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))
    ]
    if not include_superseded:
        must.append(qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="superseded_by")))
    flt = qmodels.Filter(must=must, must_not=[
        # summaries are per-session artifacts, not graph nodes
        qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="session_summary")),
    ])
    points = []
    offset = None
    while len(points) < limit:
        batch, offset = client.scroll(
            collection_name=config.MEMORIES_COLLECTION,
            scroll_filter=flt,
            limit=min(256, limit - len(points)),
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(batch)
        if offset is None:
            break

    nodes = [
        {
            "id": str(p.id),
            "text": p.payload.get("text", ""),
            "type": p.payload.get("type", "fact"),
            "project_id": p.payload.get("project_id") or config.DEFAULT_PROJECT,
            "importance": p.payload.get("importance", 0.5),
            "created_at": p.payload.get("created_at"),
            "last_seen": p.payload.get("last_seen") or p.payload.get("created_at"),
            "session_id": p.payload.get("session_id") or "",
            "superseded": bool(p.payload.get("superseded_by")),
            "status": p.payload.get("status"),
            "source_agent": p.payload.get("source_agent") or "",
        }
        for p in points
    ]

    edges: list[dict] = []
    if len(points) > 1:
        # Since the BM25 migration the collection carries named vectors, so
        # p.vector comes back as {"": dense, "bm25": sparse}; un-migrated
        # points still return the bare dense list.
        vectors = np.array(
            [p.vector.get("") if isinstance(p.vector, dict) else p.vector
             for p in points],
            dtype=np.float32,
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sims = (vectors / norms) @ (vectors / norms).T
        seen = set()
        for i in range(len(points)):
            order = np.argsort(sims[i])[::-1]
            kept = 0
            for j in order:
                if j == i or kept >= top_edges:
                    if kept >= top_edges:
                        break
                    continue
                if sims[i][j] < min_similarity:
                    break
                key = (min(i, int(j)), max(i, int(j)))
                if key not in seen:
                    seen.add(key)
                    edges.append(
                        {"source": nodes[i]["id"], "target": nodes[int(j)]["id"],
                         "weight": round(float(sims[i][j]), 4)}
                    )
                kept += 1

    return {"nodes": nodes, "edges": edges}


def _fmt_date(ts: float | None) -> str:
    if not ts:
        return "?"
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def recall(
    client: QdrantClient,
    embed_model,
    query: str,
    user_id: str = config.USER_ID,
    session_id: str = "",
    project: str = "",
    top_k_memories: int = config.RECALL_TOP_K_MEMORIES,
    top_k_history: int = config.RECALL_TOP_K_HISTORY,
    recent_turns: int = config.RECALL_RECENT_TURNS,
) -> dict:
    """Combine the three long-term sources into one recall result:
    distilled facts (L3), semantically related past turns (L2) and the
    current session's recent turns (L1 rebuild).

    Project scoping: explicit `project` wins; otherwise it is derived from
    the current session's own stored messages (the hook stamped them), so
    the caller needs no extra knowledge. Same-project hits get a soft boost.
    """
    if not project and session_id:
        project = memory_store.get_session_project(client, session_id, user_id)
        if project == config.DEFAULT_PROJECT:
            project = ""  # no meaningful project — skip boosting

    mems = search_memories(
        client, embed_model, query, user_id, top_k_memories, project=project or None
    )

    now = time.time()
    history = []
    for h in memory_store.search_history(
        client, embed_model, query, user_id,
        exclude_session=session_id or None, top_k=top_k_history * 2,
        project=project or None,
    ):
        age = max(now - (h["timestamp"] or now), 0.0)
        h["score"] = h.pop("score") * _decay(age, config.HISTORY_HALF_LIFE_DAYS)
        history.append(h)
    history.sort(key=lambda h: h["score"], reverse=True)
    history = [h for h in history if h["score"] >= config.RECALL_MIN_SCORE][:top_k_history]

    recent = (
        memory_store.get_recent_turns(client, session_id, user_id, recent_turns)
        if session_id
        else []
    )

    routing = route_query(query)
    docs = (
        documents.search_chunks(client, embed_model, query, project or None)
        if routing["docs"]
        else []
    )

    # A matched session with a distilled summary is shown AS the summary
    # (coherent, token-cheap); raw 300-char snippets only remain for sessions
    # never consolidated. Details stay reachable via search_history.
    summaries = get_session_summaries(client, (h["session_id"] for h in history), user_id)

    # Tag entries with the agent that produced them ("long brain + N
    # adapters": the reader should know a hit came from another agent).
    def _agent(entry: dict) -> str:
        return f", {entry['source_agent']}" if entry.get("source_agent") else ""

    lines: list[str] = []
    if mems:
        lines.append("[Long-term memories]")
        lines += [
            f"- ({m['type']}, {_fmt_date(m['created_at'])}{_agent(m)}) {m['text']}"
            for m in mems
        ]
    if summaries:
        lines.append("[Session summaries (related past sessions)]")
        lines += [
            f"- ({sid}, {_fmt_date(s['created_at'])}{_agent(s)}) {s['text'][:500]}"
            for sid, s in summaries.items()
        ]
    raw_history = [h for h in history if h["session_id"] not in summaries]
    if raw_history:
        lines.append("[Related past conversations]")
        lines += [
            f"- ({h['session_id']}, {_fmt_date(h['timestamp'])}{_agent(h)}) "
            f"{h['role']}: {h['content'][:300]}"
            for h in raw_history
        ]
    if docs:
        lines.append("[Project documents]")
        lines += [
            f"- ({d['source']}) {d['text'][:config.RECALL_DOC_SNIPPET_CHARS]}"
            for d in docs
        ]
    if recent:
        lines.append("[Most recent turns in this session]")
        lines += [f"- {t['role']}: {t['content'][:300]}" for t in recent]

    return {
        "project": project or config.DEFAULT_PROJECT,
        "memories": mems,
        "related_history": history,
        "session_summaries": summaries,
        "recent_turns": recent,
        "documents": docs,
        "routing": routing,
        "context_block": "\n".join(lines),
    }
