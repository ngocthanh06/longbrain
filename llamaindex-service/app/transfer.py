"""Memory transfer: text-level export/import for moving memory between
machines (device migration) or embedding models.

The bundle contains payload text only — no vectors — so it is independent
of the embedding model that produced it: import re-embeds every record with
the CURRENT model and upserts it with its original payload, so timestamps,
supersede links and consolidated flags survive the move (recall decay and
provenance keep working, imported sessions are not re-distilled). Point ids
are deterministic functions of the text, which makes import idempotent and
safe to run over existing data.

Import deliberately bypasses save_facts' supersede-similarity check: the
bundle already encodes its supersede state, and re-running the check would
re-stamp created_at and could re-link facts differently on the new machine.
"""

import hashlib
import json
import time

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app import config, documents, memories, memory_store, qdrant_setup

FORMAT = "hermes-memory-export"
VERSION = 1


class InvalidBundle(ValueError):
    pass


def _scroll_all(client: QdrantClient, collection: str, flt=None) -> list:
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def _user_filter() -> qmodels.Filter:
    return qmodels.Filter(must=[
        qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value=config.USER_ID))
    ])


def _existing_ids(client: QdrantClient, collection: str, ids: list[str]) -> set[str]:
    existing: set[str] = set()
    for i in range(0, len(ids), 512):
        for p in client.retrieve(collection_name=collection, ids=ids[i:i + 512], with_payload=False):
            existing.add(str(p.id))
    return existing


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_bundle(client: QdrantClient) -> dict:
    facts = []
    for p in _scroll_all(client, config.MEMORIES_COLLECTION, _user_filter()):
        pl = p.payload or {}
        if not (pl.get("text") or "").strip():
            continue
        fact = {
            "text": pl["text"],
            "type": pl.get("type", "fact"),
            "importance": pl.get("importance", 0.5),
            "project_id": pl.get("project_id") or config.DEFAULT_PROJECT,
            "session_id": pl.get("session_id") or "",
            "created_at": pl.get("created_at"),
        }
        if pl.get("superseded_by"):
            fact["superseded_by"] = str(pl["superseded_by"])
        facts.append(fact)
    facts.sort(key=lambda f: f["created_at"] or 0)

    turns = []
    for p in _scroll_all(client, config.CHAT_HISTORY_COLLECTION, _user_filter()):
        pl = p.payload or {}
        if not (pl.get("content") or "").strip() or not pl.get("session_id"):
            continue
        turns.append({
            "session_id": pl["session_id"],
            "role": pl.get("role", "user"),
            "content": pl["content"],
            "project_id": pl.get("project_id") or config.DEFAULT_PROJECT,
            "timestamp": pl.get("timestamp"),
            "consolidated": bool(pl.get("consolidated", True)),
        })
    turns.sort(key=lambda t: (t["session_id"], t["timestamp"] or 0))

    docs = _export_documents(client)

    return {
        "format": FORMAT,
        "version": VERSION,
        "exported_at": time.time(),
        "user_id": config.USER_ID,
        "embed_provider": config.EMBED_PROVIDER,  # informational only —
        "embed_model": config.EMBED_MODEL,        # import re-embeds anyway
        "counts": {"facts": len(facts), "turns": len(turns), "documents": len(docs)},
        "facts": facts,
        "turns": turns,
        "documents": docs,
    }


def _export_documents(client: QdrantClient) -> list[dict]:
    try:
        points = _scroll_all(client, config.DOCUMENTS_COLLECTION)
    except Exception:
        return []  # collection may not exist yet (no document ever ingested)
    docs = []
    for p in points:
        pl = p.payload or {}
        # LlamaIndex-managed payload: the chunk text + metadata live inside
        # the serialized node, not as plain payload fields.
        text, metadata = "", {}
        try:
            node = json.loads(pl.get("_node_content") or "")
            text = node.get("text") or ""
            metadata = dict(node.get("metadata") or {})
        except (TypeError, ValueError):
            continue
        if not text.strip():
            continue
        metadata = {k: v for k, v in metadata.items() if not k.startswith("_")}
        metadata.pop("user_id", None)
        project_id = metadata.pop("project_id", None) or config.DEFAULT_PROJECT
        docs.append({"text": text, "metadata": metadata, "project_id": project_id})
    return docs


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
def import_bundle(client: QdrantClient, embed_model, index, bundle: dict) -> dict:
    if not isinstance(bundle, dict) or bundle.get("format") != FORMAT:
        raise InvalidBundle(
            'not a Hermes memory export (expected format "hermes-memory-export")'
        )
    if bundle.get("version") != VERSION:
        raise InvalidBundle(
            f"unsupported bundle version {bundle.get('version')!r} (expected {VERSION})"
        )

    facts = _import_facts(client, embed_model, bundle.get("facts") or [])
    turns = _import_turns(client, embed_model, bundle.get("turns") or [])
    docs = _import_documents(client, index, bundle.get("documents") or [])
    if facts["imported"] or turns["imported"] or docs["imported"]:
        qdrant_setup.touch_meta(client)
    return {"status": "ok", "facts": facts, "turns": turns, "documents": docs}


def _import_facts(client: QdrantClient, embed_model, facts: list[dict]) -> dict:
    rows: dict[str, tuple[str, dict]] = {}
    for f in facts:
        text = (f.get("text") or "").strip()
        if not text:
            continue
        rows.setdefault(memories.fact_point_id(config.USER_ID, text), (text, f))
    existing = _existing_ids(client, config.MEMORIES_COLLECTION, list(rows))
    imported = 0
    for point_id, (text, f) in rows.items():
        if point_id in existing:
            continue
        ftype = f.get("type", "fact")
        if ftype not in memories.VALID_TYPES:
            ftype = "fact"
        try:
            importance = min(max(float(f.get("importance", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            importance = 0.5
        payload = {
            "user_id": config.USER_ID,
            "session_id": f.get("session_id") or "",
            "project_id": f.get("project_id") or config.DEFAULT_PROJECT,
            "type": ftype,
            "text": text,
            "importance": importance,
            "created_at": f.get("created_at") or time.time(),
        }
        if f.get("superseded_by"):
            payload["superseded_by"] = str(f["superseded_by"])
        client.upsert(
            collection_name=config.MEMORIES_COLLECTION,
            points=[qmodels.PointStruct(
                id=point_id,
                vector=embed_model.get_text_embedding(text),
                payload=payload,
            )],
        )
        imported += 1
    return {"imported": imported, "skipped_existing": len(rows) - imported}


def _import_turns(client: QdrantClient, embed_model, turns: list[dict]) -> dict:
    rows: dict[str, dict] = {}
    for t in turns:
        content = t.get("content") or ""
        session_id = t.get("session_id") or ""
        role = t.get("role") or "user"
        if not content.strip() or not session_id:
            continue
        # content is hashed verbatim (not stripped) so the id matches what
        # the hook would produce for the same turn.
        pid = memory_store.message_point_id(config.USER_ID, session_id, role, content)
        rows.setdefault(pid, t)
    existing = _existing_ids(client, config.CHAT_HISTORY_COLLECTION, list(rows))
    imported = 0
    for point_id, t in rows.items():
        if point_id in existing:
            continue
        client.upsert(
            collection_name=config.CHAT_HISTORY_COLLECTION,
            points=[qmodels.PointStruct(
                id=point_id,
                vector=embed_model.get_text_embedding(t["content"]),
                payload={
                    "user_id": config.USER_ID,
                    "session_id": t["session_id"],
                    "project_id": t.get("project_id") or config.DEFAULT_PROJECT,
                    "role": t.get("role") or "user",
                    "content": t["content"],
                    "timestamp": t.get("timestamp") or time.time(),
                    # default True: imported history must never retrigger
                    # consolidation sweeps on the new machine
                    "consolidated": bool(t.get("consolidated", True)),
                },
            )],
        )
        imported += 1
    return {"imported": imported, "skipped_existing": len(rows) - imported}


def _import_documents(client: QdrantClient, index, docs: list[dict]) -> dict:
    imported = skipped = 0
    for d in docs:
        text = (d.get("text") or "").strip()
        if not text:
            continue
        # LlamaIndex assigns random node ids, so idempotency comes from a
        # content hash stamped into the metadata instead.
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if _document_hash_exists(client, digest):
            skipped += 1
            continue
        metadata = dict(d.get("metadata") or {})
        metadata["import_hash"] = digest
        documents.ingest_text(
            index, client, text, metadata,
            project_id=d.get("project_id") or config.DEFAULT_PROJECT,
        )
        imported += 1
    return {"imported": imported, "skipped_existing": skipped}


def _document_hash_exists(client: QdrantClient, digest: str) -> bool:
    try:
        hits, _ = client.scroll(
            collection_name=config.DOCUMENTS_COLLECTION,
            scroll_filter=qmodels.Filter(must=[
                qmodels.FieldCondition(key="import_hash", match=qmodels.MatchValue(value=digest))
            ]),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(hits)
    except Exception:
        return False  # collection not created yet
