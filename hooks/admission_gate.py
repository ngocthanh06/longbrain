#!/usr/bin/env python3
"""Shared admission-gate constants for auto-recall lifecycle hooks (Hermes,
Claude Code, Codex): the same two thresholds and the same context-prefix
logic, declared once instead of copied per hook. Independent copies risk
silently drifting apart — an asymmetric gate on one entry path but not
another causes continuation-noise tokens (see the project's Token-Cost Lens
note) exactly the way this file exists to prevent.
"""

import os
import re


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Prompts shorter than this ("ok", "tiếp tục", "continue") carry no
# searchable meaning — recall would only match noise, and every injected
# block costs tokens. The turn is still written to memory elsewhere; only
# the recall lookup is skipped.
MIN_PROMPT_CHARS = env_int("LONGBRAIN_RECALL_MIN_PROMPT_CHARS", 15)
# Outer safety net only: the backend (memories.recall) already budgets
# context_block by whole item (config.CONTEXT_BUDGET_CHARS), so this slice
# should rarely cut anything mid-line in practice.
MAX_CONTEXT_CHARS = env_int("LONGBRAIN_MEMORY_MAX_CONTEXT", 6000)

# Mirrors app.memories.is_vietnamese — this wrapper line is the one piece of
# injected text the hook itself controls (context_block's own headers are
# matched server-side), so it should follow the query's language too instead
# of guaranteeing a dose of English on every single Vietnamese turn.
_VN_CHARS_RE = re.compile(
    r"[ăâàáảãạằắẳẵặầấẩẫậêèéẻẽẹềếểễệìíỉĩịôơòóỏõọồốổỗộờớởỡợ"
    r"ưùúủũụừứửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)


def context_prefix(query: str) -> str:
    return (
        "Bộ nhớ dài hạn (tự động gọi lại):" if _VN_CHARS_RE.search(query)
        else "Long-term memory (auto-recalled):"
    )


def cap_context(context: str, budget: int = MAX_CONTEXT_CHARS) -> str:
    """Cap injected context without cutting an item or a section header.

    The service already performs item-level budgeting, but adapters remain a
    second safety boundary. Splitting on newlines preserves the same contract
    when an older service or a direct MCP response exceeds the adapter cap.
    """
    if budget <= 0:
        return ""
    lines = context.strip().splitlines()
    kept: list[str] = []
    pending_header = ""
    for line in lines:
        if line.startswith("[") and line.endswith("]"):
            pending_header = line
            continue
        candidate = kept + ([pending_header] if pending_header else []) + [line]
        if len("\n".join(candidate)) > budget:
            break
        if pending_header:
            kept.append(pending_header)
            pending_header = ""
        kept.append(line)
    return "\n".join(kept)
