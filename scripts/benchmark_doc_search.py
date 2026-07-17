#!/usr/bin/env python3
"""Sprint 0 decision-gate benchmark (docs/SEARCH_SPEC.md): does BAAI/bge-m3
find the right document from a vague Vietnamese query better than the current
paraphrase-multilingual-MiniLM-L12-v2? A one-off analysis tool, not part of
the running service.

Method: pull every chunk of the live `longbrain_documents` collection
(text + `source`), embed corpus and a local eval query file with
each model, rank chunks by cosine, deduplicate to a DOCUMENT ranking (best
chunk per source), then score doc-level hit@3 / hit@10 / MRR per case.
Both models go through the identical ranking code, so the only variable is
the embedding space. Results are written next to the eval set so runs can be
compared across sprints.

Run on the host (NOT in the container — bge-m3 needs torch):
  python3 -m venv .bench-venv && . .bench-venv/bin/activate
  pip install "fastembed==0.7.4" sentence-transformers numpy requests
  cp llamaindex-service/evals/search_eval.example.json \
     llamaindex-service/evals/search_eval.local.json
  python3 scripts/benchmark_doc_search.py --model both

fastembed is pinned to the service's version so the MiniLM baseline is the
same vector space the service actually searches.
"""

import argparse
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parent.parent
EVAL_PATH = REPO / "llamaindex-service" / "evals" / "search_eval.local.json"
RESULTS_DIR = REPO / "llamaindex-service" / "evals"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "longbrain_documents"

MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BGE_M3 = "BAAI/bge-m3"


def fetch_corpus(qdrant_url: str) -> list[dict]:
    """Every chunk of the documents collection as {source, text}."""
    chunks, offset = [], None
    while True:
        resp = requests.post(
            f"{qdrant_url}/collections/{COLLECTION}/points/scroll",
            json={"limit": 256, "offset": offset,
                  "with_payload": ["source", "_node_content"], "with_vector": False},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()["result"]
        for p in result["points"]:
            payload = p.get("payload") or {}
            try:
                text = json.loads(payload.get("_node_content") or "{}").get("text", "")
            except (ValueError, TypeError):
                text = ""
            if text.strip() and payload.get("source"):
                chunks.append({"source": payload["source"], "text": text})
        offset = result.get("next_page_offset")
        if offset is None:
            return chunks


def embed_minilm(texts: list[str]) -> np.ndarray:
    from fastembed import TextEmbedding

    model = TextEmbedding(MINILM)
    return np.array(list(model.embed(texts, batch_size=32)))


def embed_bge_m3(texts: list[str]) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(BGE_M3, device=device)
    return model.encode(texts, batch_size=8, normalize_embeddings=True,
                        show_progress_bar=True)


EMBEDDERS = {"minilm": embed_minilm, "bge-m3": embed_bge_m3}


def doc_ranking(query_vec: np.ndarray, corpus_vecs: np.ndarray, sources: list[str]) -> list[str]:
    """Chunk cosine ranking collapsed to distinct documents (best chunk wins)."""
    sims = corpus_vecs @ query_vec / (
        np.linalg.norm(corpus_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-9
    )
    ranked, seen = [], set()
    for i in np.argsort(-sims):
        src = sources[i]
        if src not in seen:
            seen.add(src)
            ranked.append(src)
    return ranked


def normalized_source(value: str) -> str:
    """Normalize macOS NFD filenames and NFC eval labels before comparison."""
    return unicodedata.normalize("NFC", value)


def evaluate(model_key: str, cases: list[dict], corpus: list[dict]) -> dict:
    texts = [c["text"] for c in corpus]
    sources = [c["source"] for c in corpus]
    embed = EMBEDDERS[model_key]

    print(f"[{model_key}] embedding corpus ({len(texts)} chunks)...")
    t0 = time.time()
    corpus_vecs = embed(texts)
    print(f"[{model_key}] corpus embedded in {time.time() - t0:.0f}s")
    query_vecs = embed([c["query"] for c in cases])

    per_case = []
    for case, qv in zip(cases, query_vecs):
        ranked = doc_ranking(np.asarray(qv), corpus_vecs, sources)
        expected = {normalized_source(s) for s in case["expected_sources"]}
        rank = next(
            (i + 1 for i, source in enumerate(ranked)
             if normalized_source(source) in expected),
            None,
        )
        per_case.append({
            "id": case["id"], "difficulty": case["difficulty"], "rank": rank,
            "hit3": rank is not None and rank <= 3,
            "hit10": rank is not None and rank <= 10,
            "rr": (1.0 / rank) if rank else 0.0,
            "top3": ranked[:3],
        })

    def agg(rows):
        n = len(rows)
        return {"n": n,
                "hit@3": round(sum(r["hit3"] for r in rows) / n, 3),
                "hit@10": round(sum(r["hit10"] for r in rows) / n, 3),
                "mrr": round(sum(r["rr"] for r in rows) / n, 3)}

    by_difficulty = {}
    for d in sorted({r["difficulty"] for r in per_case}):
        by_difficulty[d] = agg([r for r in per_case if r["difficulty"] == d])
    return {"model": model_key, "overall": agg(per_case),
            "by_difficulty": by_difficulty, "cases": per_case}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*EMBEDDERS, "both"], default="both")
    parser.add_argument("--qdrant", default=QDRANT_URL)
    parser.add_argument("--eval", type=Path, default=EVAL_PATH)
    args = parser.parse_args()

    if not args.eval.exists():
        parser.error(
            f"eval file not found: {args.eval}; copy search_eval.example.json "
            "to search_eval.local.json and populate it from your local corpus"
        )
    cases = json.loads(args.eval.read_text())["cases"]
    corpus = fetch_corpus(args.qdrant)
    print(f"corpus: {len(corpus)} chunks, "
          f"{len({c['source'] for c in corpus})} documents | cases: {len(cases)}")

    models = list(EMBEDDERS) if args.model == "both" else [args.model]
    results = {}
    for key in models:
        results[key] = evaluate(key, cases, corpus)
        out = RESULTS_DIR / f"search_eval_results_{key}.json"
        out.write_text(json.dumps(results[key], ensure_ascii=False, indent=1))
        print(f"[{key}] overall: {results[key]['overall']}  -> {out.name}")

    if len(results) == 2:
        a, b = results["minilm"]["overall"], results["bge-m3"]["overall"]
        print("\n=== GATE (SEARCH_SPEC Sprint 0) ===")
        print(f"minilm : {a}\nbge-m3 : {b}")
        for r_a, r_b in zip(results["minilm"]["cases"], results["bge-m3"]["cases"]):
            mark = "=" if r_a["hit3"] == r_b["hit3"] else ("+" if r_b["hit3"] else "-")
            print(f" {mark} {r_a['id']}: rank {r_a['rank']} -> {r_b['rank']}")
        ratio = (b["hit@3"] / a["hit@3"]) if a["hit@3"] else float("inf")
        print(f"hit@3 ratio bge-m3/minilm: {ratio:.2f} "
              f"(project bar: >=1.3, or bge-m3 reaching the 0.9 KPI)")


if __name__ == "__main__":
    main()
