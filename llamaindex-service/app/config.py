import os

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
DATA_DIR = os.getenv("DATA_DIR", "/data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")

# ---------------------------------------------------------------------------
# Embedding provider — determines the vector space. Changing EMBED_PROVIDER /
# EMBED_MODEL after data exists requires a re-embed migration; the service
# refuses to start on a mismatch instead of silently mixing vector spaces.
#   fastembed (default, local ONNX, no API key) | ollama | openai | nvidia
# ---------------------------------------------------------------------------
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "fastembed").lower()
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ---------------------------------------------------------------------------
# LLM provider — stateless, safe to switch at any time. Only used for the
# optional /chat endpoint and server-side consolidation. With "none" the
# service still runs fully; consolidation is then driven by the Hermes-side
# model through the `consolidate_session` MCP tool.
#   none (default) | anthropic | openai | nvidia | ollama
# API keys are read by the provider SDKs from their standard env vars:
#   ANTHROPIC_API_KEY / OPENAI_API_KEY / NVIDIA_API_KEY
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").lower()
_DEFAULT_LLM_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
    "nvidia": "meta/llama-3.1-70b-instruct",
    "gemini": "models/gemini-2.5-flash",
    "ollama": "qwen3:latest",
}
LLM_MODEL = os.getenv("LLM_MODEL", _DEFAULT_LLM_MODELS.get(LLM_PROVIDER, ""))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "180"))
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "180"))

# ---------------------------------------------------------------------------
# Qdrant collections
# ---------------------------------------------------------------------------
DOCUMENTS_COLLECTION = os.getenv("DOCUMENTS_COLLECTION", "hermes_documents")
CHAT_HISTORY_COLLECTION = os.getenv("CHAT_HISTORY_COLLECTION", "hermes_chat_history")
MEMORIES_COLLECTION = os.getenv("MEMORIES_COLLECTION", "hermes_memories")
META_COLLECTION = os.getenv("META_COLLECTION", "hermes_meta")

# Single-user deployment default. Kept in every payload so a future move to a
# shared multi-user server is a data-compatible change, not a migration.
USER_ID = os.getenv("HERMES_USER_ID", "local")

# Per-project memory partitioning. The hook resolves the Hermes sidebar
# project from the turn's cwd; anything unmatched lands in "default".
DEFAULT_PROJECT = "default"

SCHEMA_VERSION = 3

# ---------------------------------------------------------------------------
# Memory behaviour
# ---------------------------------------------------------------------------
CHAT_MEMORY_TOKEN_LIMIT = int(os.getenv("CHAT_MEMORY_TOKEN_LIMIT", "3000"))
CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "200"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

RECALL_TOP_K_MEMORIES = int(os.getenv("RECALL_TOP_K_MEMORIES", "5"))
RECALL_TOP_K_HISTORY = int(os.getenv("RECALL_TOP_K_HISTORY", "3"))
RECALL_RECENT_TURNS = int(os.getenv("RECALL_RECENT_TURNS", "6"))
RECALL_MIN_SCORE = float(os.getenv("RECALL_MIN_SCORE", "0.25"))
# Half-lives (days) for the recency decay applied on top of similarity.
MEMORY_HALF_LIFE_DAYS = float(os.getenv("MEMORY_HALF_LIFE_DAYS", "90"))
HISTORY_HALF_LIFE_DAYS = float(os.getenv("HISTORY_HALF_LIFE_DAYS", "30"))
# Similarity above which a new fact supersedes an existing one.
SUPERSEDE_SIMILARITY = float(os.getenv("SUPERSEDE_SIMILARITY", "0.92"))
# Recall multiplier for hits belonging to the active project. Soft boost, not
# a hard filter — cross-project knowledge can still surface when it's the
# best match; same-project memories just win ties comfortably.
RECALL_PROJECT_BOOST = float(os.getenv("RECALL_PROJECT_BOOST", "1.5"))

# ---------------------------------------------------------------------------
# Auto-consolidation (needs LLM_PROVIDER != none). Event-driven by design:
# - on_session_end hook  -> consolidate that session as it finishes
# - on_session_start hook + service boot -> one debounced catch-up sweep for
#   anything missed (crash, force-quit, rate-limited earlier)
# - CONSOLIDATION_INTERVAL > 0 optionally re-enables a periodic sweep loop
#   (default 0 = off; per user preference, no repeating background LLM calls)
# ---------------------------------------------------------------------------
CONSOLIDATION_ENABLED = os.getenv("CONSOLIDATION_ENABLED", "true").lower() == "true"
CONSOLIDATION_INTERVAL = float(os.getenv("CONSOLIDATION_INTERVAL", "0"))
CONSOLIDATION_IDLE_SECONDS = float(os.getenv("CONSOLIDATION_IDLE_SECONDS", "900"))  # session quiet 15m
CONSOLIDATION_MIN_TURNS = int(os.getenv("CONSOLIDATION_MIN_TURNS", "2"))
# Catch-up sweeps triggered by on_session_start are debounced: at most one
# per this window, so opening several chats doesn't hammer the LLM API.
CONSOLIDATION_SWEEP_DEBOUNCE = float(os.getenv("CONSOLIDATION_SWEEP_DEBOUNCE", "600"))
# Anti-pollution: hard cap per session + importance floor. The prompt already
# forbids generic/tool knowledge; these are the safety nets on top.
CONSOLIDATION_MAX_FACTS = int(os.getenv("CONSOLIDATION_MAX_FACTS", "5"))
CONSOLIDATION_MIN_IMPORTANCE = float(os.getenv("CONSOLIDATION_MIN_IMPORTANCE", "0.5"))
