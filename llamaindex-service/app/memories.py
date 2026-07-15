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

from app import config, documents, hybrid, memory_store, scope_policy

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


_CONTRADICTION_PROMPT = """Fact A is an existing stored memory. Fact B is a new \
memory being saved right now. Do they CONTRADICT each other — stating \
different, mutually exclusive values for the same real-world thing (e.g. a \
different preference, a different current status, a different answer to \
the same question)? Do not count it as a contradiction if Fact B is simply \
about a different topic, or is consistent with / additional to Fact A.

Fact A (existing): {a}
Fact B (new): {b}

Answer with exactly one word: yes or no."""


def _llm_flags_contradiction(llm, new_text: str, existing_text: str) -> bool:
    try:
        answer = llm.complete(
            _CONTRADICTION_PROMPT.format(a=existing_text, b=new_text)
        ).text
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
        # its original date. A project-specific fact may only retire the same
        # project's facts or legacy global/default facts. Its type does not
        # grant cross-project authority (a project UI preference is not a
        # global user preference). A genuinely global/default correction may
        # still retire matching project records.
        #
        # NOTE (see #10 in review): this intentionally lets a project-scoped
        # save retire a DEFAULT-scoped fact sharing the same subject+relation
        # (tests test_triple_default_preference_superseded_from_project and
        # test_triple_default_project_superseded_from_any_project lock this
        # in) — subject="user" relations like editor/comment_language really
        # are meant to be one global value updatable from any project. The
        # actual gap is that TRIPLE_VOCABULARY (consolidation.py) also lists
        # inherently PROJECT-scoped attributes (package_manager, database,
        # framework) under the same subject="user" convention, so two
        # different projects' independent choices can incorrectly collide
        # and retire each other. Fixing that needs a vocabulary/prompt
        # redesign (e.g. project-scoped relations get a project-qualified
        # subject) plus a backfill of existing triples — not a change to
        # this scope filter, which already does the right thing for the
        # relations it was designed around.
        superseded = []
        superseded_ids = []
        triple = _triple_of(fact) if config.TRIPLE_SUPERSEDE else None
        if triple:
            triple_must = [
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
            ]
            if project_id != config.DEFAULT_PROJECT:
                triple_must.append(qmodels.FieldCondition(
                    key="project_id",
                    match=qmodels.MatchAny(any=[project_id, config.DEFAULT_PROJECT]),
                ))
            candidates, _ = client.scroll(
                collection_name=config.MEMORIES_COLLECTION,
                scroll_filter=qmodels.Filter(must=triple_must),
                limit=16,
                with_payload=["text"],
            )
            for cand in candidates:
                if str(cand.id) == point_id:
                    continue
                superseded_ids.append(cand.id)
                superseded.append(cand.payload.get("text", ""))

        # A near-identical active fact gets superseded (updated info wins).
        # Two bands: >=SUPERSEDE_SIMILARITY is close enough to auto-supersede
        # (near-exact text); the wider DEDUP_LLM_CHECK_MIN..SUPERSEDE_SIMILARITY
        # band catches reworded/summarized restatements that score too low on
        # raw cosine to trust blindly (see config.py) — an LLM judges those,
        # and only when one is configured (LLM_PROVIDER=none skips the band).
        # Scope the supersede search to this fact's project PLUS default:
        # near-identical facts in a DIFFERENT specific project are distinct
        # knowledge (letting one retire the other would silently lose the
        # other project's memory), but a genuinely global fact already saved
        # under "default" must still be found here — otherwise restating the
        # same standing preference while working in a new project creates an
        # orphan project-scoped duplicate instead of being recognized as the
        # same fact (observed: a "reply in Vietnamese" preference duplicated
        # across projects because this search never looked at "default").
        hits = client.search(
            collection_name=config.MEMORIES_COLLECTION,
            query_vector=vector,
            query_filter=_active_filter(user_id, extra=[
                qmodels.FieldCondition(
                    key="project_id",
                    match=(
                        qmodels.MatchValue(value=project_id)
                        if project_id == config.DEFAULT_PROJECT
                        else qmodels.MatchAny(any=[project_id, config.DEFAULT_PROJECT])
                    ),
                ),
            ]),
            limit=5,
            score_threshold=config.DEDUP_LLM_CHECK_MIN,
            with_payload=["text", "project_id"],
        )
        best_relevant = None  # highest-scoring hit NOT judged a duplicate
        duplicate_of_global = False
        for hit in hits:
            is_dup = hit.score >= config.SUPERSEDE_SIMILARITY or (
                llm is not None and _llm_confirms_duplicate(llm, text, hit.payload.get("text", ""))
            )
            if is_dup:
                hit_project = hit.payload.get("project_id") or config.DEFAULT_PROJECT
                if hit_project == config.DEFAULT_PROJECT and project_id != config.DEFAULT_PROJECT:
                    # The match is already global; this project-scoped
                    # restatement must not create a narrower duplicate that
                    # could shadow it — the default fact already covers
                    # every project, so treat this as a no-op confirmation.
                    duplicate_of_global = True
                    break
                superseded_ids.append(hit.id)
                superseded.append(hit.payload.get("text", ""))
            elif best_relevant is None:
                best_relevant = hit

        if duplicate_of_global:
            results.append({"text": text, "status": "duplicate_of_global"})
            continue

        # Contradiction detector (third, weakest-signal tier): triple-
        # supersede and the dedup band above already resolve the fact when
        # they can. What's left — no matching triple, no dedup verdict — may
        # still be a plain contradiction ("prefers dark mode" -> "prefers
        # light mode") that both of those miss. Flag, don't auto-resolve:
        # write a symmetric conflicts_with pointer on both facts and let the
        # LLM/user reconcile. At most one LLM call per save.
        contradicts_id = None
        if (
            config.CONTRADICTION_DETECTION and llm is not None and not triple
            and best_relevant is not None
            and _llm_flags_contradiction(llm, text, best_relevant.payload.get("text", ""))
        ):
            contradicts_id = str(best_relevant.id)

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
        if contradicts_id:
            payload["conflicts_with"] = contradicts_id
        client.upsert(
            collection_name=config.MEMORIES_COLLECTION,
            points=[qmodels.PointStruct(
                id=point_id,
                vector=hybrid.point_vector(client, config.MEMORIES_COLLECTION, vector, text),
                payload=payload,
            )],
        )
        # Old facts are only marked superseded/conflicting AFTER the new fact
        # lands: if the upsert above had failed, doing this first would drop
        # the old facts from recall while the replacement never existed.
        if superseded_ids:
            client.set_payload(
                collection_name=config.MEMORIES_COLLECTION,
                payload={"superseded_by": point_id},
                points=superseded_ids,
            )
        if contradicts_id:
            client.set_payload(
                collection_name=config.MEMORIES_COLLECTION,
                payload={"conflicts_with": point_id},
                points=[contradicts_id],
            )
        results.append(
            {"text": text, "status": "supersedes" if superseded else "new",
             **({"superseded": superseded} if superseded else {}),
             **({"conflicts_with": contradicts_id} if contradicts_id else {})}
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
    covers_through: float = 0.0,
) -> dict:
    """Store the structured summary of a finished session (goal, decisions,
    unresolved). Lives in the memories collection as type=session_summary but
    is NOT a fact: excluded from fact search/listing, surfaced by recall in
    place of that session's raw snippets.

    `covers_through` (optional, epoch seconds) is the timestamp of the newest
    turn this summary actually reflects. A backlog longer than one
    consolidation pass gets processed newest-chunk-first (see
    consolidation._covered_points), so a LATER pass over the same session can
    end up summarizing an OLDER leftover chunk after a fresher one already
    produced a summary — passing `covers_through` lets that later, older-
    content pass be skipped instead of regressing the summary backwards in
    time. Callers that don't track this (e.g. an explicitly provided summary
    via the MCP save_memories tool) simply omit it and always overwrite, as
    before."""
    text = (text or "").strip()
    if not text or not session_id:
        return {"status": "skipped"}
    point_id = summary_point_id(user_id, session_id)
    if covers_through:
        existing = client.retrieve(
            collection_name=config.MEMORIES_COLLECTION, ids=[point_id],
            with_payload=["covers_through"],
        )
        if existing and (existing[0].payload.get("covers_through") or 0) > covers_through:
            return {"status": "skipped_stale", "session_id": session_id}
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "project_id": project_id or config.DEFAULT_PROJECT,
        "type": "session_summary",
        "text": text,
        "created_at": time.time(),
        "covers_through": covers_through,
    }
    if source_agent:
        payload["source_agent"] = source_agent
    client.upsert(
        collection_name=config.MEMORIES_COLLECTION,
        points=[qmodels.PointStruct(
            id=point_id,
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
    project_scope: str = "strict",
) -> list[dict]:
    allowed_projects = scope_policy.filter_projects(
        project, project_scope, config.DEFAULT_PROJECT
    )
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
    if allowed_projects:
        # A fact's type does not make it global. A preference captured in
        # project B may describe B's UI, not a standing user convention.
        # Legacy/default memories are the only global records until scope is
        # represented explicitly in the schema.
        flt.must.append(qmodels.FieldCondition(
            key="project_id",
            match=qmodels.MatchAny(any=allowed_projects),
        ))
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
        if scope_policy.boost_same_project(project, hit_project, project_scope, config.DEFAULT_PROJECT):
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
                "conflicts_with": payload.get("conflicts_with"),
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
            "conflicts_with": p.payload.get("conflicts_with"),
            "source_agent": p.payload.get("source_agent") or "",
        }
        for p in points
    ]


def health_stats(client: QdrantClient, user_id: str = config.USER_ID) -> dict:
    """At-a-glance counts for the /ui health panel: no new storage, no
    time-series — "growth" is approximated from created_at on facts already
    on disk. Consolidation backlog is NOT included here (would need
    importing scheduler, which imports consolidation, which imports this
    module — a cycle); the /memory/stats endpoint merges it in."""
    must = [qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=user_id))]
    flt = qmodels.Filter(must=must, must_not=[
        qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="session_summary")),
    ])
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=config.MEMORIES_COLLECTION,
            scroll_filter=flt,
            limit=256,
            offset=offset,
            with_payload=["type", "superseded_by", "created_at", "project_id",
                          "status", "conflicts_with"],
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break

    now = time.time()
    by_type: dict[str, int] = {}
    by_project: dict[str, int] = {}
    total_active = total_superseded = 0
    new_24h = new_7d = 0
    open_tasks = done_tasks = 0
    flagged_conflicts = 0
    for p in points:
        payload = p.payload
        if payload.get("superseded_by"):
            total_superseded += 1
            continue
        total_active += 1
        ftype = payload.get("type", "fact")
        by_type[ftype] = by_type.get(ftype, 0) + 1
        project = payload.get("project_id") or config.DEFAULT_PROJECT
        by_project[project] = by_project.get(project, 0) + 1
        created_at = payload.get("created_at")
        if created_at:
            age = now - created_at
            if age <= 86400:
                new_24h += 1
            if age <= 7 * 86400:
                new_7d += 1
        if ftype == "task":
            done_tasks += 1 if payload.get("status") == "done" else 0
            open_tasks += 1 if payload.get("status") != "done" else 0
        if payload.get("conflicts_with"):
            flagged_conflicts += 1

    total = total_active + total_superseded
    return {
        "total_active": total_active,
        "total_superseded": total_superseded,
        "superseded_ratio": round(total_superseded / total, 4) if total else 0.0,
        "by_type": by_type,
        "by_project": by_project,
        "new_last_24h": new_24h,
        "new_last_7d": new_7d,
        "open_tasks": open_tasks,
        "done_tasks": done_tasks,
        "flagged_conflicts": flagged_conflicts,
    }


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
            "topic": None,
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


def _cluster_topics(nodes: list[dict], sims, threshold: float) -> None:
    """Connected-components over `sims`, scoped to same-project pairs above
    `threshold`. Writes nodes[i]["topic"] in place; clusters of size < 2
    keep topic=None (nothing to group). Stable ordering (by descending
    cluster size, tie-broken by the cluster's smallest node id) so reloads
    don't reshuffle which cluster is "t0" vs "t1"."""
    n = len(nodes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if nodes[i]["project_id"] == nodes[j]["project_id"] and sims[i][j] >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    clusters = [members for members in groups.values() if len(members) >= 2]
    clusters.sort(key=lambda members: (-len(members), nodes[members[0]]["id"]))
    for rank, members in enumerate(clusters):
        topic_id = f"{nodes[members[0]]['project_id']}::t{rank}"
        for idx in members:
            nodes[idx]["topic"] = topic_id


def graph_data(
    client: QdrantClient,
    user_id: str = config.USER_ID,
    include_superseded: bool = False,
    top_edges: int = 4,
    min_similarity: float = 0.35,
    limit: int = 2000,
    topic_min_similarity: float | None = None,
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
            "conflicts_with": p.payload.get("conflicts_with"),
            "source_agent": p.payload.get("source_agent") or "",
            "topic": None,
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

        if config.GRAPH_TOPIC_CLUSTERING:
            _cluster_topics(
                nodes, sims,
                topic_min_similarity
                if topic_min_similarity is not None
                else config.GRAPH_TOPIC_MIN_SIMILARITY,
            )

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
    project_scope: str = "strict",
    top_k_memories: int = config.RECALL_TOP_K_MEMORIES,
    top_k_history: int = config.RECALL_TOP_K_HISTORY,
    recent_turns: int = config.RECALL_RECENT_TURNS,
) -> dict:
    """Combine the three long-term sources into one recall result:
    distilled facts (L3), semantically related past turns (L2) and the
    current session's recent turns (L1 rebuild).

    Project scoping: explicit `project` wins; otherwise it is derived from
    the current session's own stored messages (the hook stamped them), so
    the caller needs no extra knowledge. `strict` includes only that project
    plus legacy global/default memories; `boost` permits cross-project hits;
    `global` searches everything without a project preference.
    """
    if not project and session_id:
        project = memory_store.get_session_project(client, session_id, user_id)
        # Keep it as DEFAULT_PROJECT (not ""): scope_policy.filter_projects
        # still needs a truthy project to apply the strict-scope filter, and
        # boost_same_project already skips the boost for the default bucket
        # on its own — blanking it here used to silently disable the FILTER
        # too, turning every default-project session's "strict" recall into
        # an unscoped global search.

    mems = search_memories(
        client, embed_model, query, user_id, top_k_memories,
        project=project or None, project_scope=project_scope,
    )

    now = time.time()
    history = []
    for h in memory_store.search_history(
        client, embed_model, query, user_id,
        exclude_session=session_id or None, top_k=top_k_history * 2,
        project=project or None, project_scope=project_scope,
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

    # Flagged by the contradiction detector (save_facts) — surfaced, not
    # auto-resolved, so the reader can ask the user which one is current.
    def _conflict(entry: dict) -> str:
        return (
            " [note: conflicts with another stored fact — verify with the user]"
            if entry.get("conflicts_with") else ""
        )

    lines: list[str] = []
    if mems:
        lines.append("[Long-term memories]")
        lines += [
            f"- ({m['type']}, {_fmt_date(m['created_at'])}{_agent(m)}) {m['text']}{_conflict(m)}"
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
        "project_scope": project_scope,
        "memories": mems,
        "related_history": history,
        "session_summaries": summaries,
        "recent_turns": recent,
        "documents": docs,
        "routing": routing,
        "context_block": "\n".join(lines),
    }
