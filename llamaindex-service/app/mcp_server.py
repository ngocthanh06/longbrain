"""MCP tools served over Streamable HTTP, mounted inside the FastAPI app.

Replaces the old host-side stdio mcp-bridge: Hermes registers
http://localhost:8800/mcp and needs no Python environment on the host.
"""

import logging
import re
from typing import Annotated

from pydantic import BaseModel, Field

from app import config, consolidation, documents, memories, memory_store, ratelimit, scope_policy
from app.runtime import state

logger = logging.getLogger("uvicorn")

_AUDIT_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


def _audit(action: str, allowed: bool, **fields) -> None:
    """Structured log line for a write or destructive action's outcome
    (security hardening plan, Phase 2/3 — every rate-limited action gets
    audit metadata, not just the confirm-gated ones). Only ids, counts and
    the allow/deny decision — never memory/document content. Field values
    are caller-supplied ids (session_id, memory_id, ...); strip control
    characters so one can't inject fake newline-delimited log lines."""
    detail = " ".join(
        f"{k}={_AUDIT_CONTROL_CHARS_RE.sub('_', str(v))}" for k, v in fields.items()
    )
    logger.info("audit action=%s allow=%s %s", action, allowed, detail)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = None


class Fact(BaseModel):
    text: str = Field(
        min_length=1, max_length=config.MAX_FACT_TEXT_CHARS,
        description="Self-contained fact worth remembering long-term",
    )
    type: str = Field(default="fact", description="fact | preference | decision | task")
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    # Optional (subject, relation, object) triple from the extraction
    # instructions. relation must be a single-valued snake_case attribute
    # (e.g. package_manager); a new fact with the same subject+relation
    # supersedes the old value. Omit when no single-valued attribute applies.
    subject: str = Field(default="")
    relation: str = Field(default="")
    object: str = Field(default="")


mcp = FastMCP("longbrain", stateless_http=True) if FastMCP else None


def _register_tools() -> None:
    @mcp.tool()
    def memory_recall(
        query: str, session_id: str = "", project: str = "",
        project_scope: scope_policy.ProjectScope = "strict",
    ) -> str:
        """Recall long-term memory relevant to a query: distilled facts, related
        past conversations and (if session_id given) the current session's
        recent turns. Call this at the start of a task or when the user refers
        to something from before. Returns a ready-to-inject context block.

        project: optional Hermes project slug. project_scope defaults to
        strict (only this project plus global/default memories); boost allows
        cross-project results while preferring this project; global searches
        all projects. Leave project empty to auto-detect from the session."""
        result = memories.recall(
            state["qdrant_client"], state["embed_model"], query,
            session_id=session_id, project=project, project_scope=project_scope,
        )
        return result["context_block"] or "No relevant long-term memory found."

    @mcp.tool()
    def memory_append(
        session_id: str,
        user_message: Annotated[str, Field(max_length=config.MAX_TURN_TEXT_CHARS)] = "",
        assistant_response: Annotated[str, Field(max_length=config.MAX_TURN_TEXT_CHARS)] = "",
        turn_id: str = "",
    ) -> str:
        """Persist a completed conversation turn into episodic memory. Idempotent
        when the caller passes a stable turn_id (a per-turn id from your own
        agent, if you have one) — without it, retries are best-effort only:
        content/session-state alone cannot distinguish every retry from a
        genuine repeat (see memory_store.add_message)."""
        if not ratelimit.allow("memory_append"):
            _audit("memory_append", False, session_id=session_id, reason="rate_limited")
            return "Refused: too many memory_append calls in a short window. Wait a moment and retry."
        client, embed = state["qdrant_client"], state["embed_model"]
        # Same session stickiness as REST /memory/append.
        project_id = memory_store.get_session_project(client, session_id)
        n = 0
        if user_message.strip():
            memory_store.add_message(client, embed, session_id, "user", user_message,
                                     project_id=project_id, sibling_content=assistant_response,
                                     turn_id=turn_id)
            n += 1
        if assistant_response.strip():
            memory_store.add_message(client, embed, session_id, "assistant", assistant_response,
                                     project_id=project_id, sibling_content=user_message,
                                     turn_id=turn_id)
            n += 1
        _audit("memory_append", True, session_id=session_id, messages_stored=n)
        return f"Stored {n} message(s) for session {session_id}."

    @mcp.tool()
    def save_memories(
        facts: Annotated[list[Fact], Field(max_length=config.MAX_FACTS_PER_CALL)],
        session_id: str = "", project: str = "",
        session_summary: Annotated[str, Field(max_length=config.MAX_SESSION_SUMMARY_CHARS)] = "",
    ) -> str:
        """Save distilled long-term facts (decisions, preferences, project info,
        constraints, tasks) into semantic memory. Near-duplicate existing facts
        are superseded by the new version automatically. Facts inherit the
        session's project unless `project` (a Hermes project slug) is given.
        If this follows a consolidate_session handout, the session's turns are
        marked consolidated now — pass the same session_id (an empty facts
        list is fine when nothing was worth keeping) and pass the 2-4 sentence
        `session_summary` the handout instructions asked for (goal, decisions,
        unresolved) so future recall can show it instead of raw snippets."""
        if not ratelimit.allow("save_memories"):
            _audit("save_memories", False, session_id=session_id, reason="rate_limited")
            return "Refused: too many save_memories calls in a short window. Wait a moment and retry."
        client = state["qdrant_client"]
        project_id = project or (
            memory_store.get_session_project(client, session_id) if session_id
            else config.DEFAULT_PROJECT
        )
        results = memories.save_facts(
            client, state["embed_model"],
            [f.model_dump() for f in facts],
            session_id=session_id, project_id=project_id,
            llm=state.get("llm"),
        )
        summary_saved = memories.save_session_summary(
            client, state["embed_model"], session_id, session_summary,
            project_id=project_id,
        )["status"] == "ok"
        # Close the consolidation loop: turns handed out by consolidate_session
        # only count as consolidated once the extraction actually came back.
        handout = consolidation.pop_handout(session_id) if session_id else []
        if handout:
            memory_store.mark_consolidated(client, handout)
        suffix = " Session summary stored." if summary_saved else ""
        _audit("save_memories", True, session_id=session_id, project_id=project_id, facts_saved=len(results))
        if not results:
            return "Nothing to save." + (
                f" Marked {len(handout)} turn(s) consolidated." if handout else ""
            ) + suffix
        return f"(project: {project_id})\n" + "\n".join(
            f"[{r['status']}] {r['text']}" for r in results
        ) + suffix

    @mcp.tool()
    def consolidate_session(session_id: str) -> str:
        """Consolidate a session's un-consolidated turns into long-term facts.
        If the memory service has its own LLM it extracts and saves the facts
        directly. Otherwise it returns the transcript plus extraction
        instructions — follow them, then call save_memories with the result,
        then this tool again is NOT needed (turns are marked on save)."""
        if not ratelimit.allow("consolidate_session"):
            _audit("consolidate_session", False, session_id=session_id, reason="rate_limited")
            return "Refused: too many consolidate_session calls in a short window. Wait a moment and retry."
        client, embed, llm = state["qdrant_client"], state["embed_model"], state.get("llm")
        if llm is not None:
            result = consolidation.consolidate_session(client, embed, llm, session_id)
            saved = result["facts"]
            _audit("consolidate_session", True, session_id=session_id, facts_saved=len(saved))
            return (
                f"Consolidated {result.get('turns_processed', 0)} turns into "
                f"{len(saved)} fact(s):\n" + "\n".join(f"- {f['text']}" for f in saved)
                if saved else "Nothing worth remembering in this session."
            )
        points = memory_store.fetch_unconsolidated(client, session_id)
        if not points:
            return "Nothing to consolidate for this session."
        # Don't mark yet — the turns only count as consolidated when the facts
        # come back via save_memories. If that never happens, the session is
        # offered again next time instead of being silently dropped.
        consolidation.record_handout(session_id, [p.id for p in points])
        transcript = consolidation.transcript_from_points(points)
        instructions = consolidation.EXTRACTION_INSTRUCTIONS.format(
            max_facts=config.CONSOLIDATION_MAX_FACTS
        )
        return (
            f"{instructions}\n\n"
            f"After extracting, call save_memories(facts, session_id={session_id!r}, "
            f"session_summary=<the summary you produced>) — even with an empty "
            f"facts list if nothing qualifies.\n\n"
            f"<transcript>\n{transcript}\n</transcript>"
        )

    @mcp.tool()
    def search_history(
        query: str, top_k: int = 5, project: str = "",
        project_scope: scope_policy.ProjectScope = "strict",
    ) -> str:
        """Semantically search all past conversation turns across every session.
        Optional project slug is strict by default. Use project_scope=boost
        for cross-project pattern search, or global to search everything."""
        hits = memory_store.search_history(
            state["qdrant_client"], state["embed_model"], query, top_k=top_k,
            project=project or None, project_scope=project_scope,
        )
        if not hits:
            return "No matching past conversations."
        return "\n\n".join(
            f"[{h['project_id']}/{h['session_id']}] {h['role']}: {h['content']}"
            for h in hits
        )

    @mcp.tool()
    def list_memories(project: str = "", limit: int = 50) -> str:
        """List stored long-term facts (newest first), optionally filtered by
        Hermes project slug. Returns each fact with its id — needed for
        forget_memory."""
        facts = memories.list_facts(
            state["qdrant_client"], project=project or None, limit=limit
        )
        if not facts:
            return "No memories stored yet."
        return "\n".join(
            f"[{f['id']}] ({f['project_id']}/{f['type']}) {f['text']}" for f in facts
        )

    @mcp.tool()
    def forget_about(query: str) -> str:
        """Find memories matching a query so the user can forget them. Returns
        candidate facts with ids — review them, then call forget_memory(id)
        for each one the user actually wants removed. Never delete without
        the user's explicit confirmation."""
        hits = memories.search_memories(
            state["qdrant_client"], state["embed_model"], query, top_k=5
        )
        if not hits:
            return "No matching memories found."
        return "Candidates (after the user confirms, call forget_memory(id, confirm=true)):\n" + "\n".join(
            f"[{h['id']}] ({h['project_id']}/{h['type']}) {h['text']}" for h in hits
        )

    @mcp.tool()
    def forget_memory(memory_id: str, confirm: bool = False) -> str:
        """Permanently delete one stored fact by id (get ids from
        list_memories or forget_about). Deletion is irreversible: pass
        confirm=true ONLY after the user explicitly confirmed this specific
        memory should be removed."""
        if not confirm:
            _audit("forget_memory", False, memory_id=memory_id, reason="no_confirm")
            return (
                "Refused: deletion needs the user's explicit confirmation. "
                "Show them the memory text, and once they agree call "
                "forget_memory(memory_id, confirm=true)."
            )
        if not ratelimit.allow("forget_memory"):
            _audit("forget_memory", False, memory_id=memory_id, reason="rate_limited")
            return "Refused: too many forget_memory calls in a short window. Wait a moment and retry."
        deleted = memories.delete_fact(state["qdrant_client"], memory_id)
        _audit("forget_memory", deleted, memory_id=memory_id)
        if deleted:
            return f"Deleted memory {memory_id}."
        return f"No memory with id {memory_id}."

    @mcp.tool()
    def forget_session(session_id: str, confirm: bool = False) -> str:
        """Permanently delete one conversation session's stored history
        (all its turns in episodic memory). Facts already distilled from it
        are NOT touched — use forget_about/forget_memory for those.

        Deletion is irreversible: pass confirm=true ONLY after the user
        explicitly confirmed this session's history should be removed."""
        if not confirm:
            _audit("forget_session", False, session_id=session_id, reason="no_confirm")
            return (
                "Refused: deletion needs the user's explicit confirmation. "
                "Show them which session this is, and once they agree call "
                "forget_session(session_id, confirm=true)."
            )
        if not ratelimit.allow("forget_session"):
            _audit("forget_session", False, session_id=session_id, reason="rate_limited")
            return "Refused: too many forget_session calls in a short window. Wait a moment and retry."
        deleted = memory_store.delete_session(state["qdrant_client"], session_id)
        _audit("forget_session", bool(deleted), session_id=session_id, messages_deleted=deleted)
        if not deleted:
            return f"No stored history for session {session_id}."
        return f"Deleted {deleted} stored message(s) of session {session_id}."

    @mcp.tool()
    def cleanup_garbage(
        dry_run: bool = True, confirm: str = "", limit: int = 10000
    ) -> str:
        """Find or remove superseded document chunks and unreferenced local
        document files.  The default is read-only.  Facts and conversation
        history are never included.  Destructive execution requires the
        exact confirmation ``CLEANUP SUPERSEDED``."""
        if not dry_run and confirm != "CLEANUP SUPERSEDED":
            _audit("cleanup_garbage", False, reason="no_confirm")
            return (
                'Refused. First run cleanup_garbage(dry_run=true), review the '
                'result, then call cleanup_garbage(dry_run=false, '
                'confirm="CLEANUP SUPERSEDED").'
            )
        if not dry_run and not ratelimit.allow("cleanup_garbage"):
            _audit("cleanup_garbage", False, reason="rate_limited")
            return "Refused: too many cleanup_garbage runs in a short window. Wait a moment and retry."
        result = documents.cleanup_superseded(
            state["qdrant_client"], dry_run=dry_run, limit=limit
        )
        if not dry_run:
            _audit(
                "cleanup_garbage", True,
                chunks_deleted=result["chunks_deleted"], files_removed=result["files_removed"],
            )
        return (
            f"Cleanup {result['status']}: {result['chunks_found']} stale chunk(s), "
            f"{result['chunks_deleted']} deleted, {result['files_removed']} file(s) removed. "
            f"Truncated={result['truncated']}."
        )

    @mcp.tool()
    def forget_everything(confirm: str = "") -> str:
        """FULL RESET: permanently delete ALL stored memory — every
        conversation turn, every fact (including superseded ones), across all
        sessions and projects. Irreversible (nightly backups aside).

        Call ONLY when the user explicitly asks to wipe/reset all memory, and
        only AFTER they confirmed. Pass confirm="DELETE ALL" (exact string) —
        anything else refuses. Note: the CURRENT conversation will keep being
        recorded from this point on; that is normal behaviour."""
        if confirm != "DELETE ALL":
            _audit("forget_everything", False, reason="no_confirm")
            return (
                'Refused. To wipe all memory, ask the user to confirm, then '
                'call forget_everything(confirm="DELETE ALL").'
            )
        if not ratelimit.allow("forget_everything"):
            _audit("forget_everything", False, reason="rate_limited")
            return "Refused: too many forget_everything calls in a short window. Wait a moment and retry."
        client = state["qdrant_client"]
        turns = memory_store.delete_all_history(client)
        facts = memories.delete_all_facts(client)
        _audit("forget_everything", True, messages_deleted=turns, facts_deleted=facts)
        return (
            f"Memory wiped: {turns} conversation message(s) and {facts} fact(s) "
            "deleted. From now on new turns will be recorded again as usual."
        )

    @mcp.tool()
    def list_projects() -> str:
        """List projects that have stored memory (Hermes sidebar projects,
        auto-detected from where each chat happened) with counts."""
        projects = memory_store.list_projects(state["qdrant_client"])
        if not projects:
            return "No memory stored yet."
        return "\n".join(
            f"{p['project_id']} — {p['sessions']} sessions, {p['messages']} messages"
            for p in projects
        )

    @mcp.tool()
    def list_sessions() -> str:
        """List all stored conversation sessions with their project and
        message counts."""
        sessions = memory_store.list_sessions(state["qdrant_client"])
        if not sessions:
            return "No sessions stored yet."
        return "\n".join(
            f"[{s['project_id']}] {s['session_id']} — {s['messages']} messages"
            for s in sessions
        )

    @mcp.tool()
    def search_knowledge_base(
        query: str, top_k: int = config.RETRIEVAL_TOP_K, project: str = ""
    ) -> str:
        """Search the document knowledge base (ingested files/notes) for
        information relevant to a query. Pass a Hermes project slug to search
        only that project's documents; leave empty to search everything."""
        if state.get("index") is None:
            # SEARCH_SPEC constraint 3: loud, typed unavailability — clearly
            # distinct from "no matching documents".
            return ("Document search unavailable (doc_embedder_unavailable): "
                    + (state.get("doc_embed_error") or "unknown error"))
        from llama_index.core.vector_stores.types import (
            FilterOperator, MetadataFilter, MetadataFilters,
        )

        conditions = [
            MetadataFilter(key="superseded_by", operator=FilterOperator.IS_EMPTY, value=None)
        ]
        if project:
            conditions.append(MetadataFilter(key="project_id", value=project))
        filters = MetadataFilters(filters=conditions)
        from app import rerank as _rerank

        fetch_k = max(top_k, config.DOC_RERANK_CANDIDATES) \
            if config.DOC_RERANK else top_k
        retriever = state["index"].as_retriever(similarity_top_k=fetch_k, filters=filters)
        nodes = retriever.retrieve(query)
        if not nodes:
            return "No relevant documents found in the knowledge base."
        contents = [node.node.get_content() for node in nodes]
        scores = _rerank.rerank(query, contents)
        if scores is not None:
            contents = [c for _, c in
                        sorted(zip(scores, contents), key=lambda p: -p[0])]
        return "\n\n---\n\n".join(contents[:top_k])

    @mcp.tool()
    def add_to_knowledge_base(
        text: Annotated[str, Field(min_length=1, max_length=config.MAX_KB_TEXT_CHARS)],
        source: str = "", project: str = "",
    ) -> str:
        """Add a piece of text to the document knowledge base for future
        retrieval, optionally scoped to a Hermes project slug."""
        if not ratelimit.allow("add_to_knowledge_base"):
            _audit("add_to_knowledge_base", False, reason="rate_limited")
            return "Refused: too many add_to_knowledge_base calls in a short window. Wait a moment and retry."
        if state.get("index") is None:
            return ("Cannot ingest: document embedder unavailable "
                    "(doc_embedder_unavailable): "
                    + (state.get("doc_embed_error") or "unknown error"))
        project_id = project or config.DEFAULT_PROJECT
        total = documents.ingest_text(
            state["index"], state["qdrant_client"], text,
            {"source": source} if source else {},
            project_id=project_id,
        )
        _audit("add_to_knowledge_base", True, project_id=project_id, total_chunks_indexed=total)
        return f"Added to knowledge base. Total chunks now indexed: {total}"


if mcp is not None:
    _register_tools()
