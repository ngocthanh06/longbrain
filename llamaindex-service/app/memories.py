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
import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app import config, memory_store

_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

VALID_TYPES = {"fact", "preference", "decision", "task"}


def fact_point_id(user_id: str, text: str) -> str:
    digest = hashlib.sha256(" ".join(text.split()).lower().encode("utf-8")).hexdigest()
    return str(uuid.uuid5(_NAMESPACE, f"fact:{user_id}:{digest}"))


def _active_filter(user_id: str, extra: list | None = None) -> qmodels.Filter:
    must = [
        qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id)),
        qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="superseded_by")),
    ]
    if extra:
        must.extend(extra)
    return qmodels.Filter(must=must)


def save_facts(
    client: QdrantClient,
    embed_model,
    facts: list[dict],
    user_id: str = config.USER_ID,
    session_id: str = "",
    project_id: str = config.DEFAULT_PROJECT,
    source_agent: str = "",
) -> list[dict]:
    results = []
    for fact in facts:
        text = (fact.get("text") or "").strip()
        if not text:
            continue
        ftype = fact.get("type", "fact")
        if ftype not in VALID_TYPES:
            ftype = "fact"
        importance = min(max(float(fact.get("importance", 0.5)), 0.0), 1.0)

        point_id = fact_point_id(user_id, text)
        existing = client.retrieve(
            collection_name=config.MEMORIES_COLLECTION, ids=[point_id], with_payload=False
        )
        if existing:
            results.append({"text": text, "status": "duplicate"})
            continue

        vector = embed_model.get_text_embedding(text)

        # A near-identical active fact gets superseded (updated info wins).
        superseded = []
        hits = client.search(
            collection_name=config.MEMORIES_COLLECTION,
            query_vector=vector,
            query_filter=_active_filter(user_id),
            limit=3,
            with_payload=["text"],
        )
        for hit in hits:
            if hit.score >= config.SUPERSEDE_SIMILARITY:
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
        }
        if source_agent:
            payload["source_agent"] = source_agent
        client.upsert(
            collection_name=config.MEMORIES_COLLECTION,
            points=[qmodels.PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        results.append(
            {"text": text, "status": "supersedes" if superseded else "new",
             **({"superseded": superseded} if superseded else {})}
        )
    return results


def _decay(age_seconds: float, half_life_days: float) -> float:
    return math.pow(0.5, age_seconds / (half_life_days * 86400.0))


def search_memories(
    client: QdrantClient,
    embed_model,
    query: str,
    user_id: str = config.USER_ID,
    top_k: int = config.RECALL_TOP_K_MEMORIES,
    project: str | None = None,
) -> list[dict]:
    vector = embed_model.get_text_embedding(query)
    hits = client.search(
        collection_name=config.MEMORIES_COLLECTION,
        query_vector=vector,
        query_filter=_active_filter(user_id),
        limit=top_k * 3,  # over-fetch, then rerank by decayed/boosted score
        with_payload=True,
    )
    now = time.time()
    scored = []
    for h in hits:
        importance = h.payload.get("importance", 0.5)
        age = max(now - (h.payload.get("created_at") or now), 0.0)
        final = h.score * _decay(age, config.MEMORY_HALF_LIFE_DAYS) * (0.5 + 0.5 * importance)
        hit_project = h.payload.get("project_id") or config.DEFAULT_PROJECT
        if project and hit_project == project:
            final *= config.RECALL_PROJECT_BOOST
        scored.append(
            {
                "id": str(h.id),
                "text": h.payload["text"],
                "type": h.payload.get("type", "fact"),
                "project_id": hit_project,
                "importance": importance,
                "created_at": h.payload.get("created_at"),
                "similarity": h.score,
                "score": final,
            }
        )
    scored.sort(key=lambda m: m["score"], reverse=True)
    return [m for m in scored if m["score"] >= config.RECALL_MIN_SCORE][:top_k]


def list_facts(
    client: QdrantClient,
    user_id: str = config.USER_ID,
    project: str | None = None,
    ftype: str | None = None,
    include_superseded: bool = False,
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
    points, _ = client.scroll(
        collection_name=config.MEMORIES_COLLECTION,
        scroll_filter=qmodels.Filter(must=must),
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
            "superseded_by": p.payload.get("superseded_by"),
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
    points = []
    offset = None
    while len(points) < limit:
        batch, offset = client.scroll(
            collection_name=config.MEMORIES_COLLECTION,
            scroll_filter=qmodels.Filter(must=must),
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
            "session_id": p.payload.get("session_id") or "",
            "superseded": bool(p.payload.get("superseded_by")),
            "source_agent": p.payload.get("source_agent") or "",
        }
        for p in points
    ]

    edges: list[dict] = []
    if len(points) > 1:
        vectors = np.array([p.vector for p in points], dtype=np.float32)
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

    lines: list[str] = []
    if mems:
        lines.append("[Long-term memories]")
        lines += [f"- ({m['type']}, {_fmt_date(m['created_at'])}) {m['text']}" for m in mems]
    if history:
        lines.append("[Related past conversations]")
        lines += [
            f"- ({h['session_id']}, {_fmt_date(h['timestamp'])}) {h['role']}: {h['content'][:300]}"
            for h in history
        ]
    if recent:
        lines.append("[Most recent turns in this session]")
        lines += [f"- {t['role']}: {t['content'][:300]}" for t in recent]

    return {
        "project": project or config.DEFAULT_PROJECT,
        "memories": mems,
        "related_history": history,
        "recent_turns": recent,
        "context_block": "\n".join(lines),
    }
