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
DOCUMENTS_COLLECTION = os.getenv("DOCUMENTS_COLLECTION", "longbrain_documents")
CHAT_HISTORY_COLLECTION = os.getenv("CHAT_HISTORY_COLLECTION", "longbrain_chat_history")
MEMORIES_COLLECTION = os.getenv("MEMORIES_COLLECTION", "longbrain_memories")
META_COLLECTION = os.getenv("META_COLLECTION", "longbrain_meta")

# Single-user deployment default. Kept in every payload so a future move to a
# shared multi-user server is a data-compatible change, not a migration.
USER_ID = os.getenv("LONGBRAIN_USER_ID") or os.getenv("HERMES_USER_ID", "local")

# Per-project memory partitioning. The hook resolves the Hermes sidebar
# project from the turn's cwd; anything unmatched lands in "default".
DEFAULT_PROJECT = "default"

SCHEMA_VERSION = 4

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
# Similarity above which a new fact supersedes an existing one — auto,
# no LLM check needed (near-exact text).
SUPERSEDE_SIMILARITY = float(os.getenv("SUPERSEDE_SIMILARITY", "0.92"))
# Below SUPERSEDE_SIMILARITY but above this, a candidate is "maybe the same
# fact reworded" — measured empirically: real paraphrase/summary pairs found
# in production data scored 0.69-0.73 cosine on this embedding model, well
# under 0.92, so a pure threshold change can't catch them without also
# merging unrelated same-topic facts. Only checked when an LLM is configured
# (LLM_PROVIDER=none skips this band and falls back to threshold-only).
DEDUP_LLM_CHECK_MIN = float(os.getenv("DEDUP_LLM_CHECK_MIN", "0.60"))
# Triple-based supersession: facts may carry a (subject, relation, object)
# triple extracted during consolidation. A new fact with the same
# subject+relation as an active one retires it, catching CONTRADICTIONS
# ("uses pnpm" -> "switched to bun") that both cosine bands miss: the texts
# score far below SUPERSEDE_SIMILARITY, and the LLM band only detects
# rewordings — and never runs at save time under LLM_PROVIDER=none.
TRIPLE_SUPERSEDE = os.getenv("TRIPLE_SUPERSEDE", "true").lower() == "true"
# A fact's decay clock runs off `last_seen`, not `created_at`: recalling a
# fact is itself evidence it's still relevant, so every fact returned by
# search_memories gets last_seen bumped to now in the same call. Facts saved
# before this field existed fall back to created_at (see _decay call site).
LAST_SEEN_REFRESH = os.getenv("LAST_SEEN_REFRESH", "true").lower() == "true"
# Recall multiplier for same-project hits, applied on top of the scope rule:
# when a project is known, auto-recall hard-scopes facts/history to that
# project + the default project (preferences are global and always pass) —
# asking in project A must not surface project B's decisions. The boost then
# ranks same-project hits above default-project ones. Cross-project lookups
# stay available through the explicit MCP search tools (no project given).
RECALL_PROJECT_BOOST = float(os.getenv("RECALL_PROJECT_BOOST", "1.5"))
# Recall multiplier for type=preference hits. Preferences are standing
# conventions (commit style, comment language, workflow) that matter most at
# the moment of ACTING on them, yet that is exactly when they lose the top-k
# race: an action prompt ("commit this for me") matches fresh same-topic
# status facts — newer (less decay), higher importance, project-boosted —
# while the older global preference sits at default importance with no boost.
# Measured failure 2026-07-10: a stored "no Co-Authored-By trailer"
# preference was not recalled on a commit request and the agent followed its
# harness default instead (eval case vn-preference-vs-fresh-facts).
RECALL_PREFERENCE_BOOST = float(os.getenv("RECALL_PREFERENCE_BOOST", "1.5"))

# ---------------------------------------------------------------------------
# Hybrid recall (C2): a BM25 sparse channel next to the dense vectors, for
# exact tokens the dense model under-ranks (error codes, "10MB", ids).
# Sparse can only RESCUE hits — a keyword match enters recall with
# similarity RECALL_BM25_WEIGHT * (its BM25 score / best BM25 score of the
# query); dense-ranked results are never demoted. Collections created before
# the migration (scripts/migrate_hybrid_bm25.py) have no sparse schema and
# everything silently stays dense-only.
# ---------------------------------------------------------------------------
HYBRID_BM25 = os.getenv("HYBRID_BM25", "true").lower() == "true"
BM25_MODEL = os.getenv("BM25_MODEL", "Qdrant/bm25")
BM25_VECTOR_NAME = "bm25"
RECALL_BM25_WEIGHT = float(os.getenv("RECALL_BM25_WEIGHT", "0.6"))


def _csv_env(name: str, default: str) -> tuple:
    return tuple(
        t.strip().lower() for t in os.getenv(name, default).split(",") if t.strip()
    )


# ---------------------------------------------------------------------------
# Recall router (rule-based, no LLM). Facts + history stay always-on; the
# documents channel (L4) is expensive per hit and only useful when the user
# is actually asking about a document, so it activates on trigger words.
# Matching is lowercase-substring; lists are CSV env overrides.
# ---------------------------------------------------------------------------
RECALL_DOC_TRIGGERS = _csv_env(
    "RECALL_DOC_TRIGGERS",
    "tài liệu,tai lieu,spec,docs,document,thiết kế,thiet ke,chỉ thị,chi thi,"
    "readme,仕様,指示書,設計書,ドキュメント",
)
RECALL_HISTORY_TRIGGERS = _csv_env(
    "RECALL_HISTORY_TRIGGERS",
    "lần trước,lan truoc,trước đó,truoc do,đã bàn,da ban,phiên trước,"
    "phien truoc,hôm qua,hom qua,hôm trước,hom truoc,review cũ,review cu,"
    "last time,earlier we,previously,we discussed,前回,以前",
)
RECALL_TOP_K_DOCS = int(os.getenv("RECALL_TOP_K_DOCS", "2"))
RECALL_DOC_SNIPPET_CHARS = int(os.getenv("RECALL_DOC_SNIPPET_CHARS", "600"))

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
