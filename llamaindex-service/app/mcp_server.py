"""MCP tools served over Streamable HTTP, mounted inside the FastAPI app.

Replaces the old host-side stdio mcp-bridge: Hermes registers
http://localhost:8800/mcp and needs no Python environment on the host.
"""

from pydantic import BaseModel, Field

from app import config, consolidation, documents, memories, memory_store
from app.runtime import state

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = None


class Fact(BaseModel):
    text: str = Field(description="Self-contained fact worth remembering long-term")
    type: str = Field(default="fact", description="fact | preference | decision | task")
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


mcp = FastMCP("hermes-memory", stateless_http=True) if FastMCP else None


def _register_tools() -> None:
    @mcp.tool()
    def memory_recall(query: str, session_id: str = "", project: str = "") -> str:
        """Recall long-term memory relevant to a query: distilled facts, related
        past conversations and (if session_id given) the current session's
        recent turns. Call this at the start of a task or when the user refers
        to something from before. Returns a ready-to-inject context block.

        project: optional Hermes project slug — same-project memories are
        boosted. Leave empty to auto-detect from the session, or call
        list_projects to see available slugs."""
        result = memories.recall(
            state["qdrant_client"], state["embed_model"], query,
            session_id=session_id, project=project,
        )
        return result["context_block"] or "No relevant long-term memory found."

    @mcp.tool()
    def memory_append(session_id: str, user_message: str = "", assistant_response: str = "") -> str:
        """Persist a completed conversation turn into episodic memory. Idempotent —
        safe to retry."""
        client, embed = state["qdrant_client"], state["embed_model"]
        # Same session stickiness as REST /memory/append.
        project_id = memory_store.get_session_project(client, session_id)
        n = 0
        if user_message.strip():
            memory_store.add_message(client, embed, session_id, "user", user_message,
                                     project_id=project_id)
            n += 1
        if assistant_response.strip():
            memory_store.add_message(client, embed, session_id, "assistant", assistant_response,
                                     project_id=project_id)
            n += 1
        return f"Stored {n} message(s) for session {session_id}."

    @mcp.tool()
    def save_memories(
        facts: list[Fact], session_id: str = "", project: str = "",
        session_summary: str = "",
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
        client, embed, llm = state["qdrant_client"], state["embed_model"], state.get("llm")
        if llm is not None:
            result = consolidation.consolidate_session(client, embed, llm, session_id)
            saved = result["facts"]
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
    def search_history(query: str, top_k: int = 5, project: str = "") -> str:
        """Semantically search all past conversation turns across every session.
        Optional project slug boosts same-project results."""
        hits = memory_store.search_history(
            state["qdrant_client"], state["embed_model"], query, top_k=top_k,
            project=project or None,
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
            return (
                "Refused: deletion needs the user's explicit confirmation. "
                "Show them the memory text, and once they agree call "
                "forget_memory(memory_id, confirm=true)."
            )
        if memories.delete_fact(state["qdrant_client"], memory_id):
            return f"Deleted memory {memory_id}."
        return f"No memory with id {memory_id}."

    @mcp.tool()
    def forget_session(session_id: str) -> str:
        """Permanently delete one conversation session's stored history
        (all its turns in episodic memory). Facts already distilled from it
        are NOT touched — use forget_about/forget_memory for those."""
        deleted = memory_store.delete_session(state["qdrant_client"], session_id)
        if not deleted:
            return f"No stored history for session {session_id}."
        return f"Deleted {deleted} stored message(s) of session {session_id}."

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
            return (
                'Refused. To wipe all memory, ask the user to confirm, then '
                'call forget_everything(confirm="DELETE ALL").'
            )
        client = state["qdrant_client"]
        turns = memory_store.delete_all_history(client)
        facts = memories.delete_all_facts(client)
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
        filters = None
        if project:
            from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

            filters = MetadataFilters(filters=[MetadataFilter(key="project_id", value=project)])
        retriever = state["index"].as_retriever(similarity_top_k=top_k, filters=filters)
        nodes = retriever.retrieve(query)
        if not nodes:
            return "No relevant documents found in the knowledge base."
        return "\n\n---\n\n".join(node.node.get_content() for node in nodes)

    @mcp.tool()
    def add_to_knowledge_base(text: str, source: str = "", project: str = "") -> str:
        """Add a piece of text to the document knowledge base for future
        retrieval, optionally scoped to a Hermes project slug."""
        total = documents.ingest_text(
            state["index"], state["qdrant_client"], text,
            {"source": source} if source else {},
            project_id=project or config.DEFAULT_PROJECT,
        )
        return f"Added to knowledge base. Total chunks now indexed: {total}"


if mcp is not None:
    _register_tools()
