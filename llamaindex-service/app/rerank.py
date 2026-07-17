"""Document reranker (SEARCH_SPEC Sprint 2): a cross-encoder pass over the
dense/BM25 candidates of the document search path.

Retrieval (bi-encoder) gets the right document into the top ~20; the
cross-encoder reads query + chunk TOGETHER and is what pushes it into the
top 3 — that ordering is the KPI. Kill switch DOC_RERANK (same pattern as
HYBRID_BM25): off, or any load failure, degrades to the plain retrieval
order — document search must never fail because of the extra pass.

Only the explicit document-search paths (/query, search_knowledge_base)
rerank. The recall router's docs channel does not: recall runs on every
prompt and must stay cheap.

Raw scores are kept on the result objects for debugging/labels
(SEARCH_SPEC constraint 7) — never surfaced as percentage confidence.
"""

import logging

from app import config

logger = logging.getLogger("uvicorn")

_model = None
_model_failed = False


def _get_model(force: bool = False):
    """`force=True` loads the model even when DOC_RERANK is off — used by the
    on-demand /query/explain path, which can afford the per-pair cost
    (~0.5s CPU) that the search hot path cannot (measured 2026-07-17:
    20 candidates ≈ 10s on container CPU)."""
    global _model, _model_failed
    if (not config.DOC_RERANK and not force) or _model_failed:
        return None
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder

            # max_length caps query+chunk pairs; chunks are ~1024 tokens and
            # truncation is fine — the head of a chunk identifies it.
            _model = CrossEncoder(config.DOC_RERANK_MODEL, max_length=512)
        except Exception as exc:  # noqa: BLE001 — degrade, never break search
            _model_failed = True
            logger.warning(
                "rerank: model %s unavailable (%s) — document search stays "
                "retrieval-ordered", config.DOC_RERANK_MODEL, exc,
            )
            return None
    return _model


def available() -> bool:
    return _get_model() is not None


def rerank(query: str, texts: list, force: bool = False) -> "list | None":
    """Cross-encoder scores for (query, text) pairs, or None when the
    reranker is off/unavailable (caller keeps the retrieval order)."""
    model = _get_model(force=force)
    if model is None or not texts:
        return None
    try:
        return [float(s) for s in model.predict([(query, t) for t in texts])]
    except Exception as exc:  # noqa: BLE001
        logger.warning("rerank: scoring failed (%s) — keeping retrieval order", exc)
        return None
