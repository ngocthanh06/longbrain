#!/usr/bin/env python3
"""Recall quality evaluation: measure what the auto-recall pipeline actually
returns against a fixed bilingual eval set, so threshold/router changes are
judged by numbers ("saved N chars, lost M expected hits") instead of feel.
Born out of a bug that survived for days precisely because nothing measured
retrieval: chunks embedded with metadata ranked near-randomly and no one saw.

Seeds a throwaway in-memory Qdrant with the corpus from
llamaindex-service/evals/recall_eval.json (facts + history + documents),
runs every case through memories.recall() with the REAL embedding model,
and scores the produced context_block:

  - include hit  : an expect_include substring appears in the block
  - violation    : an expect_exclude substring appears in the block
  - chars        : block size (proxy for injected tokens)

The committed baseline (evals/recall_baseline.json) records reality — some
cases fail on purpose until the feature they measure lands. The run exits
nonzero only when results are WORSE than baseline (fewer hits, more
violations); --update-baseline rewrites it after an accepted change.

Run inside the service container (needs app deps + the embedding model):

  docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo/llamaindex-service \
      llamaindex python /repo/scripts/recall_eval.py [--update-baseline]
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "llamaindex-service"))

from llama_index.core import Settings, StorageContext, VectorStoreIndex  # noqa: E402
from llama_index.vector_stores.qdrant import QdrantVectorStore  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from app import config, documents, memories, memory_store, qdrant_setup  # noqa: E402
from app.providers import build_embed_model  # noqa: E402

EVAL_FILE = REPO / "llamaindex-service" / "evals" / "recall_eval.json"
BASELINE_FILE = REPO / "llamaindex-service" / "evals" / "recall_baseline.json"


def seed(client: QdrantClient, embed, index, corpus: dict) -> None:
    for f in corpus.get("facts", []):
        memories.save_facts(
            client, embed,
            [{"text": f["text"], "type": f.get("type", "fact"),
              "importance": f.get("importance", 0.5)}],
            project_id=f.get("project", ""), source_agent=f.get("source_agent", ""),
        )
        if f.get("age_days"):
            # Backdate created_at so cases can stage the top-k race between
            # an old standing preference and fresh same-topic facts — the
            # point id is deterministic, so the seeded point is addressable.
            client.set_payload(
                collection_name=config.MEMORIES_COLLECTION,
                payload={"created_at": time.time() - f["age_days"] * 86400},
                points=[memories.fact_point_id(
                    config.USER_ID, f["text"],
                    f.get("project") or config.DEFAULT_PROJECT,
                )],
            )
    for turn in corpus.get("history", []):
        memory_store.add_message(
            client, embed, turn["session"], turn["role"], turn["content"],
            project_id=turn.get("project", config.DEFAULT_PROJECT),
        )
    for doc in corpus.get("documents", []):
        documents.ingest_text(
            index, client, doc["text"], {"source": doc.get("source", "")},
            project_id=doc.get("project", config.DEFAULT_PROJECT),
        )


def run_cases(client: QdrantClient, embed, cases: list) -> list:
    results = []
    for case in cases:
        recall = memories.recall(
            client, embed, case["query"],
            project=case.get("project", ""), recent_turns=0,
        )
        block = recall.get("context_block") or ""
        results.append({
            "name": case["name"],
            "hits": [s for s in case.get("expect_include", []) if s in block],
            "misses": [s for s in case.get("expect_include", []) if s not in block],
            "violations": [s for s in case.get("expect_exclude", []) if s in block],
            "chars": len(block),
            "note": case.get("note", ""),
        })
    return results


def summarize(results: list) -> dict:
    expected = sum(len(r["hits"]) + len(r["misses"]) for r in results)
    return {
        "include_hits": sum(len(r["hits"]) for r in results),
        "include_expected": expected,
        "violations": sum(len(r["violations"]) for r in results),
        "total_chars": sum(r["chars"] for r in results),
    }


def main() -> int:
    spec = json.loads(EVAL_FILE.read_text())

    embed = build_embed_model()
    Settings.embed_model = embed
    dim = len(embed.get_text_embedding("dimension probe"))

    client = QdrantClient(":memory:")
    qdrant_setup.ensure_all(client, dim)
    vector_store = QdrantVectorStore(client=client, collection_name=config.DOCUMENTS_COLLECTION)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
    )

    seed(client, embed, index, spec["corpus"])
    results = run_cases(client, embed, spec["cases"])

    print(f"{'case':34} {'hits':>6} {'viol':>5} {'chars':>6}")
    for r in results:
        total = len(r["hits"]) + len(r["misses"])
        flag = "" if not r["misses"] and not r["violations"] else "  <-"
        print(f"{r['name']:34} {len(r['hits'])}/{total:<4} {len(r['violations']):>5} {r['chars']:>6}{flag}")
        for m in r["misses"]:
            print(f"    miss: {m!r}" + (f"  ({r['note']})" if r["note"] else ""))
        for v in r["violations"]:
            print(f"    VIOLATION: {v!r}")

    summary = summarize(results)
    print(f"\nsummary: {summary['include_hits']}/{summary['include_expected']} expected hits, "
          f"{summary['violations']} violations, {summary['total_chars']} chars injected")

    if "--update-baseline" in sys.argv:
        BASELINE_FILE.write_text(json.dumps({"summary": summary, "cases": results},
                                            ensure_ascii=False, indent=1) + "\n")
        print(f"baseline written: {BASELINE_FILE}")
        return 0

    if not BASELINE_FILE.exists():
        print("no baseline yet — run with --update-baseline to record one")
        return 0

    base = json.loads(BASELINE_FILE.read_text())["summary"]
    regressed = []
    if summary["include_hits"] < base["include_hits"]:
        regressed.append(f"hits {base['include_hits']} -> {summary['include_hits']}")
    if summary["violations"] > base["violations"]:
        regressed.append(f"violations {base['violations']} -> {summary['violations']}")
    if summary["total_chars"] > base["total_chars"] * 1.2:
        regressed.append(f"chars {base['total_chars']} -> {summary['total_chars']} (+20%)")
    if regressed:
        print("\n✗ WORSE than baseline: " + "; ".join(regressed))
        return 1
    print("\n✓ not worse than baseline"
          + (" (better — consider --update-baseline)"
             if summary["include_hits"] > base["include_hits"]
             or summary["violations"] < base["violations"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
