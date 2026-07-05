"""Consolidation sweeps — event-driven, not polling.

Triggers:
- service boot (one catch-up after the stack settles)
- POST /memory/consolidate-pending, called by the on_session_start hook when
  Hermes Desktop opens a chat (debounced so several chats = one sweep)
- optionally a periodic loop when CONSOLIDATION_INTERVAL > 0 (off by default)

The on_session_end hook handles the common case (a session consolidates as it
finishes); sweeps only pick up what that missed — crash, force-quit, or an
LLM rate-limit at the time.
"""

import asyncio
import logging
import threading
import time

from app import config, consolidation, memory_store

logger = logging.getLogger("uvicorn.error")

_last_sweep_at = 0.0
_sweep_lock = threading.Lock()


def pending_sessions(client) -> list[dict]:
    """Sessions that are idle long enough and have enough un-consolidated
    turns to be worth distilling."""
    now = time.time()
    pending = []
    for s in memory_store.list_sessions(client):
        if now - s["last_activity"] < config.CONSOLIDATION_IDLE_SECONDS:
            continue
        turns = memory_store.fetch_unconsolidated(client, s["session_id"])
        if len(turns) >= config.CONSOLIDATION_MIN_TURNS:
            pending.append(
                {"session_id": s["session_id"], "project_id": s["project_id"],
                 "unconsolidated_turns": len(turns)}
            )
    return pending


def sweep(client, embed_model, llm) -> list[dict]:
    results = []
    for entry in pending_sessions(client):
        sid = entry["session_id"]
        try:
            r = consolidation.consolidate_session(client, embed_model, llm, sid)
            results.append({"session_id": sid, "facts": len(r.get("facts", []))})
            logger.info(
                "auto-consolidated session %s: %d fact(s)", sid, len(r.get("facts", []))
            )
        except Exception:
            logger.exception("consolidation failed for session %s", sid)
    return results


def run_sweep_if_due(state: dict, force: bool = False) -> dict:
    """Debounced catch-up sweep. Safe to call often (hook fires per chat)."""
    global _last_sweep_at
    llm = state.get("llm")
    if llm is None or not config.CONSOLIDATION_ENABLED:
        return {"status": "skipped", "reason": "no LLM / disabled"}
    with _sweep_lock:
        now = time.time()
        if not force and now - _last_sweep_at < config.CONSOLIDATION_SWEEP_DEBOUNCE:
            return {"status": "debounced",
                    "next_allowed_in": round(config.CONSOLIDATION_SWEEP_DEBOUNCE - (now - _last_sweep_at))}
        _last_sweep_at = now
    results = sweep(state["qdrant_client"], state["embed_model"], llm)
    return {"status": "ok", "sessions_consolidated": len(results), "details": results}


async def consolidation_loop(state: dict) -> None:
    await asyncio.sleep(90)  # boot catch-up, after the stack settles
    try:
        await asyncio.to_thread(run_sweep_if_due, state, True)
    except Exception:
        logger.exception("boot consolidation sweep crashed")

    if config.CONSOLIDATION_INTERVAL <= 0:
        return  # event-driven only (default)
    while True:
        await asyncio.sleep(config.CONSOLIDATION_INTERVAL)
        try:
            await asyncio.to_thread(run_sweep_if_due, state, True)
        except Exception:
            logger.exception("consolidation sweep crashed")
