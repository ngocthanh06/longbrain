"""Episodic memory (L2): raw conversation turns in Qdrant.

Every user/assistant message is embedded and upserted with a deterministic
point id derived from (user, session, role, content), so hook retries and
replays are idempotent. Turns are retrieved two ways: chronologically per
session (to rebuild the working buffer) and semantically across sessions
(long-term recall of past conversations).
"""

import hashlib
import time
import uuid

from llama_index.core.llms import ChatMessage, MessageRole
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app import config, qdrant_setup

_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def message_point_id(user_id: str, session_id: str, role: str, content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(_NAMESPACE, f"msg:{user_id}:{session_id}:{role}:{digest}"))


def _user_filter(user_id: str, extra: list | None = None) -> qmodels.Filter:
    must = [
        qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))
    ]
    if extra:
        must.extend(extra)
    return qmodels.Filter(must=must)


def add_message(
    client: QdrantClient,
    embed_model,
    session_id: str,
    role: str,
    content: str,
    user_id: str = config.USER_ID,
    project_id: str = config.DEFAULT_PROJECT,
) -> str:
    vector = embed_model.get_text_embedding(content)
    point_id = message_point_id(user_id, session_id, role, content)
    client.upsert(
        collection_name=config.CHAT_HISTORY_COLLECTION,
        points=[
            qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "user_id": user_id,
                    "session_id": session_id,
                    "project_id": project_id or config.DEFAULT_PROJECT,
                    "role": role,
                    "content": content,
                    "timestamp": time.time(),
                    "consolidated": False,
                },
            )
        ],
    )
    qdrant_setup.touch_meta(client)
    return point_id


def get_session_project(
    client: QdrantClient, session_id: str, user_id: str = config.USER_ID
) -> str:
    """Which project does this session belong to? Read from the session's own
    stored messages — the hook stamped them, so recall needs no extra input."""
    points, _ = client.scroll(
        collection_name=config.CHAT_HISTORY_COLLECTION,
        scroll_filter=_user_filter(
            user_id,
            [qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id))],
        ),
        limit=1,
        with_payload=["project_id"],
        with_vectors=False,
    )
    if points:
        return points[0].payload.get("project_id") or config.DEFAULT_PROJECT
    return config.DEFAULT_PROJECT


def _session_points(
    client: QdrantClient, session_id: str, user_id: str, limit: int
) -> list:
    points, _ = client.scroll(
        collection_name=config.CHAT_HISTORY_COLLECTION,
        scroll_filter=_user_filter(
            user_id,
            [qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id))],
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    points.sort(key=lambda p: p.payload.get("timestamp", 0))
    return points


def get_session_history(
    client: QdrantClient, session_id: str, user_id: str = config.USER_ID
) -> list[ChatMessage]:
    role_map = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT}
    return [
        ChatMessage(
            role=role_map.get(p.payload["role"], MessageRole.USER),
            content=p.payload["content"],
        )
        for p in _session_points(client, session_id, user_id, config.CHAT_HISTORY_MAX_MESSAGES)
    ]


def get_recent_turns(
    client: QdrantClient,
    session_id: str,
    user_id: str = config.USER_ID,
    limit: int = config.RECALL_RECENT_TURNS,
) -> list[dict]:
    points = _session_points(client, session_id, user_id, config.CHAT_HISTORY_MAX_MESSAGES)
    return [
        {"role": p.payload["role"], "content": p.payload["content"],
         "timestamp": p.payload.get("timestamp")}
        for p in points[-limit:]
    ]


def search_history(
    client: QdrantClient,
    embed_model,
    query: str,
    user_id: str = config.USER_ID,
    exclude_session: str | None = None,
    top_k: int = config.RECALL_TOP_K_HISTORY,
    project: str | None = None,
) -> list[dict]:
    """Semantic search over all past turns (the vectors have been there all
    along — this is what makes old conversations recallable). When a project
    is given, same-project hits get a soft score boost."""
    vector = embed_model.get_text_embedding(query)
    flt = _user_filter(user_id)
    if exclude_session:
        flt.must_not = [
            qmodels.FieldCondition(
                key="session_id", match=qmodels.MatchValue(value=exclude_session)
            )
        ]
    hits = client.search(
        collection_name=config.CHAT_HISTORY_COLLECTION,
        query_vector=vector,
        query_filter=flt,
        limit=top_k * 3 if project else top_k,
        with_payload=True,
    )
    results = []
    for h in hits:
        hit_project = h.payload.get("project_id") or config.DEFAULT_PROJECT
        score = h.score
        if project and hit_project == project:
            score *= config.RECALL_PROJECT_BOOST
        results.append(
            {
                "content": h.payload["content"],
                "role": h.payload["role"],
                "session_id": h.payload["session_id"],
                "project_id": hit_project,
                "timestamp": h.payload.get("timestamp"),
                "score": score,
            }
        )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k] if project else results


def fetch_unconsolidated(
    client: QdrantClient, session_id: str, user_id: str = config.USER_ID
) -> list:
    points, _ = client.scroll(
        collection_name=config.CHAT_HISTORY_COLLECTION,
        scroll_filter=_user_filter(
            user_id,
            [
                qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id)),
                qmodels.FieldCondition(key="consolidated", match=qmodels.MatchValue(value=False)),
            ],
        ),
        limit=config.CHAT_HISTORY_MAX_MESSAGES,
        with_payload=True,
        with_vectors=False,
    )
    points.sort(key=lambda p: p.payload.get("timestamp", 0))
    return points


def mark_consolidated(client: QdrantClient, point_ids: list[str]) -> None:
    if not point_ids:
        return
    client.set_payload(
        collection_name=config.CHAT_HISTORY_COLLECTION,
        payload={"consolidated": True},
        points=point_ids,
    )


def list_sessions(client: QdrantClient, user_id: str = config.USER_ID) -> list[dict]:
    sessions: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=config.CHAT_HISTORY_COLLECTION,
            scroll_filter=_user_filter(user_id),
            limit=512,
            offset=offset,
            with_payload=["session_id", "project_id", "timestamp"],
            with_vectors=False,
        )
        for p in points:
            sid = p.payload.get("session_id", "?")
            entry = sessions.setdefault(
                sid,
                {"session_id": sid,
                 "project_id": p.payload.get("project_id") or config.DEFAULT_PROJECT,
                 "messages": 0, "last_activity": 0.0},
            )
            entry["messages"] += 1
            entry["last_activity"] = max(entry["last_activity"], p.payload.get("timestamp") or 0.0)
        if offset is None:
            break
    return sorted(sessions.values(), key=lambda s: s["last_activity"], reverse=True)


def delete_session(
    client: QdrantClient, session_id: str, user_id: str = config.USER_ID
) -> int:
    """Hard-delete every stored turn of a session. Returns deleted count."""
    flt = _user_filter(
        user_id,
        [qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id))],
    )
    count = client.count(
        collection_name=config.CHAT_HISTORY_COLLECTION, count_filter=flt, exact=True
    ).count
    if count:
        client.delete(
            collection_name=config.CHAT_HISTORY_COLLECTION,
            points_selector=qmodels.FilterSelector(filter=flt),
        )
    return count


def delete_all_history(client: QdrantClient, user_id: str = config.USER_ID) -> int:
    """Wipe every stored turn for the user. Returns deleted count."""
    flt = _user_filter(user_id)
    count = client.count(
        collection_name=config.CHAT_HISTORY_COLLECTION, count_filter=flt, exact=True
    ).count
    if count:
        client.delete(
            collection_name=config.CHAT_HISTORY_COLLECTION,
            points_selector=qmodels.FilterSelector(filter=flt),
        )
    return count


def list_projects(client: QdrantClient, user_id: str = config.USER_ID) -> list[dict]:
    """Distinct projects present in stored memory, with message counts."""
    projects: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=config.CHAT_HISTORY_COLLECTION,
            scroll_filter=_user_filter(user_id),
            limit=512,
            offset=offset,
            with_payload=["project_id", "session_id", "timestamp"],
            with_vectors=False,
        )
        for p in points:
            pid = p.payload.get("project_id") or config.DEFAULT_PROJECT
            entry = projects.setdefault(
                pid, {"project_id": pid, "messages": 0, "sessions": set(), "last_activity": 0.0}
            )
            entry["messages"] += 1
            entry["sessions"].add(p.payload.get("session_id"))
            entry["last_activity"] = max(entry["last_activity"], p.payload.get("timestamp") or 0.0)
        if offset is None:
            break
    result = []
    for entry in sorted(projects.values(), key=lambda e: e["last_activity"], reverse=True):
        entry["sessions"] = len(entry["sessions"])
        result.append(entry)
    return result
