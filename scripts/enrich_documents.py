#!/usr/bin/env python3
"""Backfill enrichment (SEARCH_SPEC Sprint 3) for documents ingested before
the feature existed: one AI summary chunk per document, via the local LLM
(DOC_LLM_*, default Ollama). Idempotent — documents that already carry an
`enriched` chunk are skipped, so re-running is always safe. Exits cleanly
when Ollama is not running (the Optional tier is optional).

Run inside the service container (needs the doc embedder + Qdrant):

  docker compose run --rm -v "$PWD:/repo" -w /repo/llamaindex-service \\
      llamaindex python /repo/scripts/enrich_documents.py
"""

import sys

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

sys.path.insert(0, "/repo/llamaindex-service")
from app import config, enrich, providers  # noqa: E402


def main() -> int:
    if not config.DOC_ENRICH:
        print("DOC_ENRICH=false — enrichment disabled.")
        return 0
    if not enrich.llm_available():
        print(f"Local LLM not reachable at {config.OLLAMA_BASE_URL} — "
              "start Ollama first (the Optional tier needs it).")
        return 1

    embed_model = providers.build_embed_model()
    doc_embed = providers.build_doc_embed_model(embed_model)
    client = QdrantClient(url=config.QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=client, collection_name=config.DOCUMENTS_COLLECTION
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=doc_embed,
    )

    pairs, offset = set(), None
    while True:
        points, offset = client.scroll(
            collection_name=config.DOCUMENTS_COLLECTION, limit=256, offset=offset,
            with_payload=["source", "project_id"], with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            if payload.get("source"):
                pairs.add((payload["source"], payload.get("project_id") or config.DEFAULT_PROJECT))
        if offset is None:
            break

    done = skipped = 0
    for source, project in sorted(pairs):
        if enrich.enrich_document(index, client, source, project):
            done += 1
            print(f"  enriched: {project}/{source}")
        else:
            skipped += 1
    print(f"\n{done} documents enriched, {skipped} skipped (already done or empty).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
