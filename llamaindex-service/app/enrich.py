"""Optional tier (SEARCH_SPEC Sprints 3-4): AI enrichment and on-demand
match explanation, both strictly local (Ollama) and strictly optional.

Enrichment writes ONE extra synthetic chunk per document — a Vietnamese
summary + bilingual keywords — into the documents collection (same `source`,
flagged `enriched: true`). That chunk is what turns a vague Vietnamese query
against a Japanese document into same-language matching, for both the dense
and the BM25 channel. Documents are never sent to a cloud LLM (standing
privacy decision); the enrichment LLM is configured separately from the
global LLM_PROVIDER so distillation quality never changes as a side effect.

Every entry point degrades gracefully: no Ollama (or DOC_ENRICH=false) means
ingest and search behave exactly as the Core tier — never an error, never a
blocked ingest (enrichment runs as a background task).
"""

import json
import logging
import time

import requests
from qdrant_client.http import models as qmodels

from app import config

logger = logging.getLogger("uvicorn")

_availability: dict = {"ts": 0.0, "ok": False}
AVAILABILITY_TTL = 60.0


def llm_available() -> bool:
    """Is the local LLM backend reachable? Probed cheaply, cached briefly."""
    if config.DOC_LLM_PROVIDER != "ollama":
        return False
    now = time.time()
    if now - _availability["ts"] > AVAILABILITY_TTL:
        try:
            resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2)
            _availability["ok"] = resp.ok
        except Exception:  # noqa: BLE001 — unreachable = unavailable
            _availability["ok"] = False
        _availability["ts"] = now
    return _availability["ok"]


def _complete(prompt: str) -> str:
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={"model": config.DOC_LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=config.OLLAMA_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return (resp.json().get("response") or "").strip()


def _doc_text(qdrant_client, source: str, project_id: str, max_chars: int = 8000) -> str:
    """Concatenated chunk text of one document (works for any ingested file
    type — the parsed text already lives in the chunks)."""
    must = [
        qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=source)),
        qmodels.FieldCondition(key="project_id", match=qmodels.MatchValue(value=project_id)),
    ]
    parts, offset = [], None
    while sum(len(p) for p in parts) < max_chars:
        points, offset = qdrant_client.scroll(
            collection_name=config.DOCUMENTS_COLLECTION,
            scroll_filter=qmodels.Filter(must=must),
            limit=32, offset=offset, with_payload=["_node_content", "enriched"],
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            if payload.get("enriched"):
                continue
            try:
                parts.append(json.loads(payload.get("_node_content") or "{}").get("text", ""))
            except (ValueError, TypeError):
                continue
        if offset is None:
            break
    return "\n".join(parts)[:max_chars]


def already_enriched(qdrant_client, source: str, project_id: str) -> bool:
    points, _ = qdrant_client.scroll(
        collection_name=config.DOCUMENTS_COLLECTION,
        scroll_filter=qmodels.Filter(must=[
            qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=source)),
            qmodels.FieldCondition(key="project_id", match=qmodels.MatchValue(value=project_id)),
            qmodels.FieldCondition(key="enriched", match=qmodels.MatchValue(value=True)),
        ]),
        limit=1, with_payload=False, with_vectors=False,
    )
    return bool(points)


def enrich_document(index, qdrant_client, source: str, project_id: str) -> bool:
    """Generate the summary chunk for one document. Returns True when a
    chunk was written. Safe to call unconditionally: silently a no-op when
    the LLM is unavailable, enrichment is disabled, or the doc is done."""
    if not (config.DOC_ENRICH and llm_available()):
        return False
    if already_enriched(qdrant_client, source, project_id):
        return False
    text = _doc_text(qdrant_client, source, project_id)
    if not text.strip():
        return False
    try:
        summary = _complete(
            "Tài liệu sau đây có thể viết bằng tiếng Nhật, tiếng Anh hoặc tiếng "
            "Việt. Hãy viết bằng TIẾNG VIỆT: (1) một đoạn tóm tắt 3-5 câu nói rõ "
            "tài liệu này về cái gì, dành cho ai cần tìm lại nó sau này; (2) một "
            "dòng 'Từ khóa:' gồm 8-15 từ khóa tiếng Việt và tiếng Anh (giữ "
            "nguyên tên riêng, mã lỗi, tên hàm). Không thêm lời dẫn.\n\n"
            f"Tên file: {source}\n\nNội dung:\n{text}"
        )
    except Exception as exc:  # noqa: BLE001 — enrichment is best-effort
        logger.warning("enrich: LLM failed for %s (%s)", source, exc)
        return False
    if not summary:
        return False
    from app import documents

    documents.ingest_text(
        index, qdrant_client,
        f"[Tóm tắt tài liệu {source}]\n{summary}",
        metadata={"source": source, "enriched": True},
        project_id=project_id,
    )
    logger.info("enrich: summary chunk written for %s (%s)", source, project_id)
    return True


# --- Sprint 4: on-demand match explanation ---------------------------------

# Rough label bands over the reranker score — coarse on purpose
# (SEARCH_SPEC constraint 7: no fake percentage confidence in the UI).
# sentence-transformers' CrossEncoder already applies sigmoid, so the score
# is a [0,1] probability — but bge-reranker-v2-m3 compresses CROSS-LINGUAL
# matches far below same-language ones (measured 2026-07-17 on real pairs:
# true VI→JA/EN matches 0.0075–0.645, unrelated ~0.0001), so the bands are
# log-scale, not the naive 0.7/0.3.
def match_label(rerank_score: float | None) -> str:
    if rerank_score is None:
        return "unknown"
    p = rerank_score
    if not 0.0 <= p <= 1.0:  # raw logit from a non-sigmoid activation
        import math

        p = 1.0 / (1.0 + math.exp(-p))
    if p >= 0.05:
        return "Khớp cao"
    if p >= 0.002:
        return "Có thể liên quan"
    return "Ít liên quan"


def explain_match(query: str, text: str) -> str:
    """One-sentence Vietnamese reason why `text` matches `query`.
    Empty string when the local LLM is unavailable."""
    if not llm_available():
        return ""
    try:
        return _complete(
            "Người dùng tìm kiếm: \"" + query + "\"\n\nĐoạn tài liệu tìm thấy:\n"
            + text[:2000] + "\n\nGiải thích NGẮN GỌN bằng tiếng Việt (1-2 câu) "
            "vì sao đoạn tài liệu này khớp với điều người dùng tìm. Nếu thực sự "
            "không liên quan, nói thẳng là không liên quan. Không thêm lời dẫn."
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("explain: LLM failed (%s)", exc)
        return ""
