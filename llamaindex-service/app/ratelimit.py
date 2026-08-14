"""In-process rate limiter for write/destructive tool calls.

Local-only deployments (see config.py) have no external attacker to
throttle — this exists to catch a single runaway/looping agent hammering
the same action, not to enforce per-user quotas (there is only one user)."""

import threading
import time
from collections import deque

from app import config

_lock = threading.Lock()
_calls: dict[str, deque] = {}

# action -> config attribute name for its per-minute budget (security
# hardening plan, Phase 3). A flat limit either throttles normal
# fact-saving or leaves destructive actions too permissive, so writes and
# deletes get separate tiers. Looked up by attribute name (not value) so
# tests can monkeypatch config.* and see it take effect immediately.
# Actions not listed (recall/search/list — read-only) fall back to the flat
# RATE_LIMIT_WINDOW_SECONDS/RATE_LIMIT_MAX_CALLS pair below.
_TIER_ATTR: dict[str, str] = {
    "memory_append": "RATE_LIMIT_APPEND_PER_MIN",
    "save_memories": "RATE_LIMIT_SAVE_PER_MIN",
    "add_to_knowledge_base": "RATE_LIMIT_INGEST_PER_MIN",
    "ingest": "RATE_LIMIT_INGEST_PER_MIN",
    "consolidate_session": "RATE_LIMIT_CONSOLIDATE_PER_MIN",
    "documents_delete": "RATE_LIMIT_DELETE_PER_MIN",
    "cleanup_garbage": "RATE_LIMIT_DELETE_PER_MIN",
    "forget_memory": "RATE_LIMIT_DELETE_PER_MIN",
    "forget_session": "RATE_LIMIT_DELETE_PER_MIN",
    "forget_everything": "RATE_LIMIT_FULL_RESET_PER_MIN",
}
_TIER_WINDOW_SECONDS = 60.0


def _limit_for(action: str) -> tuple[float, int]:
    attr = _TIER_ATTR.get(action)
    if attr is not None:
        return _TIER_WINDOW_SECONDS, getattr(config, attr)
    return config.RATE_LIMIT_WINDOW_SECONDS, config.RATE_LIMIT_MAX_CALLS


def allow(action: str) -> bool:
    """True if `action` is still under its configured budget (see
    _TIER_ATTR, or the flat RATE_LIMIT_* pair for anything not tiered).
    Records this call as taken when it returns True."""
    window, max_calls = _limit_for(action)
    now = time.monotonic()
    with _lock:
        calls = _calls.setdefault(action, deque())
        while calls and now - calls[0] > window:
            calls.popleft()
        if len(calls) >= max_calls:
            return False
        calls.append(now)
        return True
