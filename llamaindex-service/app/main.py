import asyncio
import json
import re
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from app import config, consolidation, documents, memories, memory_store, providers, qdrant_setup, scheduler, transfer
from app.mcp_server import mcp
from app.runtime import state


@asynccontextmanager
async def lifespan(app: FastAPI):
    embed_model = providers.build_embed_model()
    llm = providers.build_llm()
    Settings.embed_model = embed_model
    if llm is not None:
        Settings.llm = llm

    embed_dim = len(embed_model.get_text_embedding("dimension probe"))

    qdrant_client = QdrantClient(url=config.QDRANT_URL)
    qdrant_setup.ensure_all(qdrant_client, embed_dim)
    Path(config.DOCUMENTS_DIR).mkdir(parents=True, exist_ok=True)

    vector_store = QdrantVectorStore(
        client=qdrant_client, collection_name=config.DOCUMENTS_COLLECTION
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, storage_context=storage_context
    )

    state.update(
        qdrant_client=qdrant_client,
        index=index,
        embed_model=embed_model,
        llm=llm,
        embed_dim=embed_dim,
    )

    sweep_task = asyncio.create_task(scheduler.consolidation_loop(state))

    async with mcp.session_manager.run():
        yield

    sweep_task.cancel()
    qdrant_client.close()
    state.clear()


app = FastAPI(title="Hermes Memory Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sources: list[str]


class IngestTextRequest(BaseModel):
    text: str
    metadata: dict = {}
    project_id: str = config.DEFAULT_PROJECT


class IngestResponse(BaseModel):
    status: str
    total_chunks_indexed: int


class QueryRequest(BaseModel):
    query: str
    top_k: int = config.RETRIEVAL_TOP_K
    project: str = ""  # hard filter when set (documents are project-scoped)


class QueryResponse(BaseModel):
    results: list[str]


class MemoryAppendRequest(BaseModel):
    session_id: str
    user_message: str = ""
    assistant_response: str = ""
    project_id: str = config.DEFAULT_PROJECT  # hook resolves from Hermes sidebar via cwd
    project_source: str = ""  # "folder" | "active" | "default" — how the hook resolved it


class RecallRequest(BaseModel):
    query: str
    session_id: str = ""
    project: str = ""  # empty -> derived from the session's stored messages
    top_k_memories: int = config.RECALL_TOP_K_MEMORIES
    top_k_history: int = config.RECALL_TOP_K_HISTORY
    recent_turns: int = config.RECALL_RECENT_TURNS


class FactIn(BaseModel):
    text: str
    type: str = "fact"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class SaveFactsRequest(BaseModel):
    facts: list[FactIn]
    session_id: str = ""
    project_id: str = config.DEFAULT_PROJECT


class ConsolidateRequest(BaseModel):
    session_id: str
    background: bool = False  # true: return immediately (hook-friendly)


class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = config.RECALL_TOP_K_MEMORIES


# ---------------------------------------------------------------------------
# Health / status
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    client: QdrantClient = state["qdrant_client"]
    meta = qdrant_setup.get_meta(client) or {}
    counts = {}
    for name in (
        config.DOCUMENTS_COLLECTION,
        config.CHAT_HISTORY_COLLECTION,
        config.MEMORIES_COLLECTION,
    ):
        try:
            counts[name] = client.get_collection(name).points_count
        except Exception:
            counts[name] = None
    return {
        "status": "ok",
        "embed_provider": config.EMBED_PROVIDER,
        "embed_model": config.EMBED_MODEL,
        "embed_dim": state.get("embed_dim"),
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": config.LLM_MODEL if state.get("llm") else None,
        "schema_version": meta.get("schema_version"),
        "last_written_at": meta.get("last_written_at"),
        "collections": counts,
    }


# ---------------------------------------------------------------------------
# Knowledge base (L4)
# ---------------------------------------------------------------------------
@app.post("/ingest/text", response_model=IngestResponse)
def ingest_text(payload: IngestTextRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    total = documents.ingest_text(
        state["index"], state["qdrant_client"], payload.text, payload.metadata,
        project_id=payload.project_id,
    )
    return IngestResponse(status="ok", total_chunks_indexed=total)


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
    project_id: str = Form(config.DEFAULT_PROJECT),
):
    try:
        parsed_metadata = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / file.filename
        with tmp_path.open("wb") as f:
            f.write(await file.read())
        stored = documents.store_original(tmp_path, file.filename)

    parsed_metadata.setdefault("source", file.filename)
    parsed_metadata["stored_path"] = str(stored)
    total = documents.ingest_file(
        state["index"], state["qdrant_client"], stored, parsed_metadata,
        project_id=project_id,
    )
    return IngestResponse(status="ok", total_chunks_indexed=total)


def _project_filters(project: str):
    if not project:
        return None
    from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

    return MetadataFilters(filters=[MetadataFilter(key="project_id", value=project)])


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    retriever = state["index"].as_retriever(
        similarity_top_k=payload.top_k, filters=_project_filters(payload.project)
    )
    nodes = retriever.retrieve(payload.query)
    return QueryResponse(results=[node.node.get_content() for node in nodes])


# ---------------------------------------------------------------------------
# Memory (L2 + L3)
# ---------------------------------------------------------------------------
@app.post("/memory/append")
def memory_append(payload: MemoryAppendRequest):
    """Persist a completed turn into episodic memory (no LLM call).

    Called from the Hermes `post_llm_call` hook so every conversation lands
    in Qdrant. Point ids are deterministic — retries are idempotent.
    """
    client, embed = state["qdrant_client"], state["embed_model"]
    # Session stickiness with intentional-move override — see
    # memory_store.resolve_append_project for the rules.
    project_id = memory_store.resolve_append_project(
        client, payload.session_id, payload.project_id, payload.project_source
    )
    appended = 0
    if payload.user_message.strip():
        memory_store.add_message(
            client, embed, payload.session_id, "user", payload.user_message,
            project_id=project_id,
        )
        appended += 1
    if payload.assistant_response.strip():
        memory_store.add_message(
            client, embed, payload.session_id, "assistant", payload.assistant_response,
            project_id=project_id,
        )
        appended += 1
    meta = qdrant_setup.get_meta(client) or {}
    return {"status": "ok", "appended": appended, "project": project_id,
            "last_written_at": meta.get("last_written_at")}


@app.post("/memory/recall")
def memory_recall(payload: RecallRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    result = memories.recall(
        state["qdrant_client"], state["embed_model"], payload.query,
        session_id=payload.session_id,
        project=payload.project,
        top_k_memories=payload.top_k_memories,
        top_k_history=payload.top_k_history,
        recent_turns=payload.recent_turns,
    )
    meta = qdrant_setup.get_meta(state["qdrant_client"]) or {}
    result["last_written_at"] = meta.get("last_written_at")
    return result


@app.post("/memory/facts")
def memory_save_facts(payload: SaveFactsRequest):
    project_id = payload.project_id
    if project_id == config.DEFAULT_PROJECT and payload.session_id:
        project_id = memory_store.get_session_project(
            state["qdrant_client"], payload.session_id
        )
    results = memories.save_facts(
        state["qdrant_client"], state["embed_model"],
        [f.model_dump() for f in payload.facts],
        session_id=payload.session_id, project_id=project_id,
    )
    return {"status": "ok", "project": project_id, "results": results}


@app.post("/memory/search")
def memory_search(payload: MemorySearchRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    return {
        "results": memories.search_memories(
            state["qdrant_client"], state["embed_model"], payload.query,
            top_k=payload.top_k,
        )
    }


@app.post("/memory/consolidate")
def memory_consolidate(payload: ConsolidateRequest, background_tasks: BackgroundTasks):
    if payload.background:
        # Hook-friendly: never block or error the caller. No LLM -> the
        # periodic sweep / manual path will pick the session up later.
        if state.get("llm") is None:
            return {"status": "skipped", "reason": "no LLM configured"}
        background_tasks.add_task(
            consolidation.consolidate_session,
            state["qdrant_client"], state["embed_model"], state["llm"],
            payload.session_id,
        )
        return {"status": "scheduled", "session_id": payload.session_id}
    try:
        return consolidation.consolidate_session(
            state["qdrant_client"], state["embed_model"], state.get("llm"),
            payload.session_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/memory/pending-consolidation")
def pending_consolidation():
    return scheduler.pending_sessions(state["qdrant_client"])


@app.post("/memory/consolidate-pending")
def consolidate_pending(background_tasks: BackgroundTasks):
    """Debounced catch-up sweep over every pending session. Called by the
    on_session_start hook when Hermes Desktop opens a chat; returns fast."""
    background_tasks.add_task(scheduler.run_sweep_if_due, state)
    return {"status": "scheduled"}


@app.get("/memory/graph")
def memory_graph(
    include_superseded: bool = False, top_edges: int = 4, min_similarity: float = 0.35
):
    """Facts as a similarity graph (nodes + edges) for the /ui browser."""
    return memories.graph_data(
        state["qdrant_client"],
        include_superseded=include_superseded,
        top_edges=top_edges,
        min_similarity=min_similarity,
    )


@app.get("/memory/facts")
def memory_list_facts(
    project: str = "", type: str = "", include_superseded: bool = False, limit: int = 200
):
    return memories.list_facts(
        state["qdrant_client"], project=project or None, ftype=type or None,
        include_superseded=include_superseded, limit=limit,
    )


@app.delete("/memory/facts/{fact_id}")
def memory_delete_fact(fact_id: str):
    if not memories.delete_fact(state["qdrant_client"], fact_id):
        raise HTTPException(status_code=404, detail="fact not found")
    return {"status": "deleted", "id": fact_id}


# ---------------------------------------------------------------------------
# Re-tagging (user corrections from the /ui browser)
# ---------------------------------------------------------------------------
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class ProjectRetag(BaseModel):
    project_id: str


def _clean_slug(value: str) -> str:
    slug = value.strip().lower()
    if not _SLUG_RE.match(slug):
        raise HTTPException(
            status_code=400,
            detail="project_id must be a slug: lowercase letters/digits/-/_ (max 64)",
        )
    return slug


class BulkRetag(BaseModel):
    ids: list[str]
    project_id: str


@app.patch("/memory/facts")
def memory_retag_facts(payload: BulkRetag):
    """Bulk re-tag: move many facts to a project in one call (multi-select
    in the /ui browser)."""
    slug = _clean_slug(payload.project_id)
    if not payload.ids:
        raise HTTPException(status_code=400, detail="ids must not be empty")
    n = memories.set_facts_project(state["qdrant_client"], payload.ids, slug)
    if not n:
        raise HTTPException(status_code=404, detail="no matching facts")
    return {"status": "retagged", "count": n, "project": slug}


@app.patch("/memory/facts/{fact_id}")
def memory_retag_fact(fact_id: str, payload: ProjectRetag):
    slug = _clean_slug(payload.project_id)
    if not memories.set_fact_project(state["qdrant_client"], fact_id, slug):
        raise HTTPException(status_code=404, detail="fact not found")
    return {"status": "retagged", "id": fact_id, "project": slug}


@app.patch("/sessions/{session_id}/project")
def session_retag(session_id: str, payload: ProjectRetag):
    """Move a whole session (its turns + the facts distilled from it) to
    another project. Future turns follow via session stickiness."""
    slug = _clean_slug(payload.project_id)
    client = state["qdrant_client"]
    turns = memory_store.set_session_project(client, session_id, slug)
    facts = memories.set_session_facts_project(client, session_id, slug)
    if not turns and not facts:
        raise HTTPException(status_code=404, detail="session not found")
    return {"status": "retagged", "session_id": session_id, "project": slug,
            "turns": turns, "facts": facts}


@app.patch("/memory/projects/{slug}")
def project_rename(slug: str, payload: ProjectRetag):
    """Rename a project label across chat history, facts and documents."""
    new = _clean_slug(payload.project_id)
    counts = memory_store.rename_project(state["qdrant_client"], slug, new)
    if not any(counts.values()):
        raise HTTPException(status_code=404, detail="no records under that project")
    return {"status": "renamed", "from": slug, "to": new, "counts": counts}


# ---------------------------------------------------------------------------
# Chat (optional — requires an LLM provider)
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    if state.get("llm") is None:
        raise HTTPException(
            status_code=503,
            detail="No LLM configured (LLM_PROVIDER=none). /chat is disabled; "
                   "Hermes should chat with its own model and use /memory/* + /query.",
        )
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    client = state["qdrant_client"]
    index = state["index"]

    history = memory_store.get_session_history(client, payload.session_id)
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=config.CHAT_MEMORY_TOKEN_LIMIT, chat_history=history
    )

    recall = memories.recall(
        client, state["embed_model"], payload.message, session_id=payload.session_id
    )
    system_prompt = (
        "You are Hermes, a helpful assistant. Use the provided context from "
        "the document index when relevant, and rely on the conversation "
        "history to stay consistent across turns."
    )
    if recall["context_block"]:
        system_prompt += "\n\nLong-term memory:\n" + recall["context_block"]

    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        similarity_top_k=config.RETRIEVAL_TOP_K,
        system_prompt=system_prompt,
    )
    result = chat_engine.chat(payload.message)

    memory_store.add_message(client, state["embed_model"], payload.session_id, "user", payload.message)
    memory_store.add_message(client, state["embed_model"], payload.session_id, "assistant", str(result))

    sources = [node.node.get_content()[:200] for node in result.source_nodes]
    return ChatResponse(session_id=payload.session_id, response=str(result), sources=sources)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
@app.get("/projects")
def projects():
    return memory_store.list_projects(state["qdrant_client"])


@app.get("/sessions")
def sessions():
    return memory_store.list_sessions(state["qdrant_client"])


@app.get("/sessions/{session_id}/history")
def session_history(session_id: str):
    history = memory_store.get_session_history(state["qdrant_client"], session_id)
    return [{"role": m.role.value, "content": m.content} for m in history]


@app.delete("/sessions/{session_id}")
def session_delete(session_id: str):
    deleted = memory_store.delete_session(state["qdrant_client"], session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="session not found")
    return {"status": "deleted", "session_id": session_id, "messages_deleted": deleted}


@app.delete("/memory/all")
def memory_wipe_all(confirm: str = ""):
    """Full reset. Requires ?confirm=DELETE%20ALL — same guard as the
    forget_everything MCP tool."""
    if confirm != "DELETE ALL":
        raise HTTPException(status_code=400, detail='confirm must be "DELETE ALL"')
    client = state["qdrant_client"]
    return {
        "status": "wiped",
        "messages_deleted": memory_store.delete_all_history(client),
        "facts_deleted": memories.delete_all_facts(client),
    }


# ---------------------------------------------------------------------------
# Transfer (device / embedding-model migration) — text-level bundles, see
# transfer.py for why vectors are deliberately excluded.
# ---------------------------------------------------------------------------
@app.get("/memory/export")
def memory_export():
    """Download every fact, chat turn and document chunk as one JSON bundle."""
    bundle = transfer.export_bundle(state["qdrant_client"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return JSONResponse(
        bundle,
        headers={
            "Content-Disposition": f'attachment; filename="hermes-memory-{stamp}.json"'
        },
    )


@app.post("/memory/import")
def memory_import(bundle: dict):
    """Re-embed and upsert a bundle produced by /memory/export. Idempotent:
    records already present (deterministic ids / content hash) are skipped."""
    try:
        return transfer.import_bundle(
            state["qdrant_client"], state["embed_model"], state["index"], bundle
        )
    except transfer.InvalidBundle as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Memory browser UI (read-only, single self-contained page)
# ---------------------------------------------------------------------------
from fastapi.responses import HTMLResponse  # noqa: E402

_UI_PATH = Path(__file__).parent / "ui.html"


@app.get("/ui", response_class=HTMLResponse)
def ui():
    # no-store: the page is tiny and local; stale cached copies after an
    # image rebuild are far more confusing than the re-download is costly.
    return HTMLResponse(_UI_PATH.read_text(), headers={"Cache-Control": "no-store"})


# MCP over Streamable HTTP. Mounted last so FastAPI's own routes win; the MCP
# endpoint ends up at POST /mcp.
app.mount("/", mcp.streamable_http_app())
