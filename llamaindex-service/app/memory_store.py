"""Episodic memory (L2): raw conversation turns in Qdrant.

Every user/assistant message is embedded and upserted with a deterministic
point id derived from (user, session, role, content), so hook retries and
replays are idempotent — a repeat of identical content with no OTHER turn
recorded since is treated as a retry of the same turn. A repeat that arrives
after the session has recorded further activity (a genuinely new occurrence,
not a retry) gets a disambiguated id instead of overwriting the earlier
message. Turns are retrieved two ways: chronologically per session (to
rebuild the working buffer) and semantically across sessions (long-term
recall of past conversations).
"""

import hashlib
import time
import uuid

from llama_index.core.llms import ChatMessage, MessageRole
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app import config, hybrid, qdrant_setup, scope_policy

_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def message_point_id(
    user_id: str, session_id: str, role: str, content: str, disambiguator: str = "",
) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    key = f"msg:{user_id}:{session_id}:{role}:{digest}"
    if disambiguator:
        key += f":{disambiguator}"
    return str(uuid.uuid5(_NAMESPACE, key))


def _user_filter(user_id: str, extra: list | None = None) -> qmodels.Filter:
    must = [
        qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))
    ]
    if extra:
        must.extend(extra)
    return qmodels.Filter(must=must)


def _scroll_all(
    client: QdrantClient, collection: str, flt: qmodels.Filter,
    with_payload, page_size: int,
) -> list:
    """Scroll every point matching `flt`, paging on `offset` until exhausted.
    A single scroll() call only returns up to `limit` points in point-ID
    order (unrelated to timestamp for our hash-derived ids) — for a session
    with more turns than that, treating that one page as "the" turns silently
    picks an arbitrary subset instead of the earliest/most-recent ones."""
    points: list = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=page_size,
            offset=offset,
            with_payload=with_payload,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def _recent_points(
    client: QdrantClient, collection: str, flt: qmodels.Filter,
    with_payload, limit: int, ascending: bool = False,
) -> list:
    """Fetch up to `limit` points ordered by `timestamp` directly from
    Qdrant (native order_by, backed by the existing timestamp payload
    index) — the newest `limit` by default, or the oldest when `ascending`.
    Used where only one end of a session's history is needed (the founding
    turn, or the last few turns): cheaper than _scroll_all's full-session
    pagination, which is only necessary when every matching point is
    actually required (e.g. the unconsolidated backlog)."""
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=flt,
        order_by=qmodels.OrderBy(
            key="timestamp",
            direction=qmodels.Direction.ASC if ascending else qmodels.Direction.DESC,
        ),
        limit=limit,
        with_payload=with_payload,
        with_vectors=False,
    )
    return points


def add_message(
    client: QdrantClient,
    embed_model,
    session_id: str,
    role: str,
    content: str,
    user_id: str = config.USER_ID,
    project_id: str = config.DEFAULT_PROJECT,
    source_agent: str = "",
    sibling_content: str = "",
) -> str:
    point_id = message_point_id(user_id, session_id, role, content)
    now = time.time()
    existing = client.retrieve(
        collection_name=config.CHAT_HISTORY_COLLECTION, ids=[point_id], with_payload=["timestamp"]
    )
    if existing:
        prev_ts = existing[0].payload.get("timestamp") or 0
        # A fixed time window can't tell a slow retry (still the SAME turn,
        # arriving late) apart from a genuine repeat (a new occurrence of
        # identical content) — both look identical from content+elapsed time
        # alone. What actually distinguishes them: a retry of this exact turn
        # cannot be followed by any OTHER turn yet (the hook only fires once
        # a turn completes, so nothing new can have been recorded for this
        # session in between two attempts at writing the same one). If the
        # session has moved on since the existing point was written, this
        # match is a genuine new occurrence, not a retry.
        #
        # A caller writes a user+assistant PAIR per call (one /memory/append
        # = one turn), so a retry of that call rewrites BOTH messages, each
        # moments after the other. Without excluding the sibling, checking
        # the user message would see the (just-written-first-time) assistant
        # reply as "newer activity" and wrongly conclude the session moved
        # on — and symmetrically for the assistant message seeing the
        # sibling user retry. Excluding content matching the sibling from
        # the count fixes this: it no longer counts as evidence the session
        # progressed, while a genuine repeat's PRIOR turn still has a
        # different sibling reply, so it still counts correctly.
        must_not = (
            [qmodels.FieldCondition(key="content", match=qmodels.MatchValue(value=sibling_content))]
            if sibling_content else None
        )
        flt = _user_filter(user_id, [
            qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id)),
            qmodels.FieldCondition(key="timestamp", range=qmodels.Range(gt=prev_ts)),
        ])
        if must_not:
            flt.must_not = must_not
        newer_activity = client.count(
            collection_name=config.CHAT_HISTORY_COLLECTION, count_filter=flt, exact=True,
        ).count
        if newer_activity:
            point_id = message_point_id(user_id, session_id, role, content, disambiguator=str(now))

    vector = embed_model.get_text_embedding(content)
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "project_id": project_id or config.DEFAULT_PROJECT,
        "role": role,
        "content": content,
        "timestamp": now,
        "consolidated": False,
    }
    if source_agent:
        payload["source_agent"] = source_agent
    client.upsert(
        collection_name=config.CHAT_HISTORY_COLLECTION,
        points=[qmodels.PointStruct(
            id=point_id,
            vector=hybrid.point_vector(client, config.CHAT_HISTORY_COLLECTION, vector, content),
            payload=payload,
        )],
    )
    qdrant_setup.touch_meta(client)
    return point_id


def get_session_project(
    client: QdrantClient,
    session_id: str,
    user_id: str = config.USER_ID,
    fallback: str = config.DEFAULT_PROJECT,
) -> str:
    """Which project does this session belong to? Read from the session's own
    stored messages — the hook stamped them, so recall needs no extra input.

    The EARLIEST turn wins, deterministically: a resumed session keeps its
    founding project even if later turns were written while the sidebar
    pointed elsewhere. Sessions with no stored turns return `fallback`."""
    points = _recent_points(
        client, config.CHAT_HISTORY_COLLECTION,
        _user_filter(
            user_id,
            [qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id))],
        ),
        with_payload=["project_id"], limit=1, ascending=True,
    )
    if not points:
        return fallback
    return points[0].payload.get("project_id") or fallback


def resolve_append_project(
    client: QdrantClient,
    session_id: str,
    hook_project: str,
    hook_source: str,
    user_id: str = config.USER_ID,
) -> str:
    """Which project should a turn being appended get?

    Sessions keep their founding project (stickiness) EXCEPT when the hook's
    tag comes from a real folder match: a changed workspace is Hermes'
    intentional way to move a session between projects (project_switch), so
    it wins. Ambient signals (sidebar selection, default) never re-tag an
    existing session — that is exactly the drift stickiness exists to stop."""
    existing = get_session_project(client, session_id, user_id, fallback="")
    if not existing:
        return hook_project or config.DEFAULT_PROJECT
    if hook_source == "folder" and hook_project and hook_project != existing:
        return hook_project
    return existing


def set_session_project(
    client: QdrantClient, session_id: str, project_id: str,
    user_id: str = config.USER_ID,
) -> int:
    """Re-tag every stored turn of a session (user-initiated correction from
    the /ui browser). Returns the number of affected turns."""
    flt = _user_filter(
        user_id,
        [qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id))],
    )
    count = client.count(
        collection_name=config.CHAT_HISTORY_COLLECTION, count_filter=flt, exact=True
    ).count
    if count:
        client.set_payload(
            collection_name=config.CHAT_HISTORY_COLLECTION,
            payload={"project_id": project_id},
            points=qmodels.FilterSelector(filter=flt),
        )
    return count


def rename_project(client: QdrantClient, old: str, new: str,
                   user_id: str = config.USER_ID) -> dict:
    """Re-tag every record of a project across chat history, facts and
    documents (project rename from /ui). Returns per-collection counts."""
    counts = {}
    for name, with_user in (
        (config.CHAT_HISTORY_COLLECTION, True),
        (config.MEMORIES_COLLECTION, True),
        (config.DOCUMENTS_COLLECTION, False),  # LlamaIndex-managed payload
    ):
        must = [qmodels.FieldCondition(key="project_id", match=qmodels.MatchValue(value=old))]
        if with_user:
            must.insert(0, qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id)))
        flt = qmodels.Filter(must=must)
        try:
            n = client.count(collection_name=name, count_filter=flt, exact=True).count
            if n:
                client.set_payload(
                    collection_name=name,
                    payload={"project_id": new},
                    points=qmodels.FilterSelector(filter=flt),
                )
        except Exception:
            n = 0  # collection may not exist yet (documents before first ingest)
        counts[name] = n
    return counts


def _session_points(
    client: QdrantClient, session_id: str, user_id: str, limit: int
) -> list:
    """The newest `limit` turns of a session, in chronological order. Fetched
    directly via order_by rather than paginating the whole session — a
    session much longer than `limit` would otherwise cost one Qdrant
    round-trip per page just to read off the tail end."""
    points = _recent_points(
        client, config.CHAT_HISTORY_COLLECTION,
        _user_filter(
            user_id,
            [qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id))],
        ),
        with_payload=True, limit=limit,
    )
    points.reverse()  # _recent_points is newest-first; callers want chronological order
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
    if limit <= 0:
        return []  # points[-0:] would be the WHOLE session, not none
    # Fetches exactly the last `limit` turns directly — called on every
    # recall(), so this stays O(1) in the session's length instead of
    # scanning up to CHAT_HISTORY_MAX_MESSAGES turns just to keep the tail.
    points = _session_points(client, session_id, user_id, limit)
    return [
        {"role": p.payload["role"], "content": p.payload["content"],
         "timestamp": p.payload.get("timestamp")}
        for p in points
    ]


def search_history(
    client: QdrantClient,
    embed_model,
    query: str,
    user_id: str = config.USER_ID,
    exclude_session: str | None = None,
    top_k: int = config.RECALL_TOP_K_HISTORY,
    project: str | None = None,
    project_scope: str = "strict",
) -> list[dict]:
    """Semantic search over all past turns (the vectors have been there all
    along — this is what makes old conversations recallable). Project scope
    follows memory recall: strict (project + default), boost, or global."""
    allowed_projects = scope_policy.filter_projects(
        project, project_scope, config.DEFAULT_PROJECT
    )
    vector = embed_model.get_text_embedding(query)
    flt = _user_filter(user_id)
    if allowed_projects:
        # Same scoping rule as fact recall: another project's conversations
        # stay out of auto-recall (default-project sessions still pass).
        # Cross-project history remains reachable by searching without a
        # project through the MCP tools.
        flt.must.append(qmodels.FieldCondition(
            key="project_id",
            match=qmodels.MatchAny(any=allowed_projects),
        ))
    if exclude_session:
        flt.must_not = [
            qmodels.FieldCondition(
                key="session_id", match=qmodels.MatchValue(value=exclude_session)
            )
        ]
    fetch_k = top_k * 3 if project else top_k
    dense_hits = client.search(
        collection_name=config.CHAT_HISTORY_COLLECTION,
        query_vector=vector,
        query_filter=flt,
        limit=fetch_k,
        with_payload=True,
    )
    sparse_hits = hybrid.search(
        client, config.CHAT_HISTORY_COLLECTION, query, flt, fetch_k
    )
    results = []
    for e in hybrid.fuse(dense_hits, sparse_hits):
        payload = e["payload"]
        hit_project = payload.get("project_id") or config.DEFAULT_PROJECT
        score = e["similarity"]
        if scope_policy.boost_same_project(project, hit_project, project_scope, config.DEFAULT_PROJECT):
            score *= config.RECALL_PROJECT_BOOST
        results.append(
            {
                "content": payload["content"],
                "role": payload["role"],
                "session_id": payload["session_id"],
                "project_id": hit_project,
                "timestamp": payload.get("timestamp"),
                "source_agent": payload.get("source_agent") or "",
                "score": score,
            }
        )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def fetch_unconsolidated(
    client: QdrantClient, session_id: str, user_id: str = config.USER_ID
) -> list:
    points = _scroll_all(
        client, config.CHAT_HISTORY_COLLECTION,
        _user_filter(
            user_id,
            [
                qmodels.FieldCondition(key="session_id", match=qmodels.MatchValue(value=session_id)),
                qmodels.FieldCondition(key="consolidated", match=qmodels.MatchValue(value=False)),
            ],
        ),
        with_payload=True,
        page_size=config.CHAT_HISTORY_MAX_MESSAGES,
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
