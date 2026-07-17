#!/usr/bin/env python3
"""One-time migration for SEARCH_SPEC Sprint 1: re-embed the documents
collection with the configured DOC_EMBED_* model (e.g. BAAI/bge-m3, 1024d).
ONLY `longbrain_documents` is touched — memories/chat-history stay in the
MiniLM space by design (SEARCH_SPEC constraint 1).

Why recreate: the vector dimension changes (384 -> 1024), and Qdrant cannot
resize a collection in place. Chunking, ids and payloads are preserved
exactly: each point's node is rebuilt from its `_node_content` payload and
re-embedded from the same text LlamaIndex embedded at ingest time
(node.get_content(EMBED) — includes the `source` metadata line, so migrated
vectors match what future ingests will produce). BM25 sparse vectors are
recomputed the same way documents._backfill_sparse does.

Safety: dumps the collection (ids + dense vectors + payloads) to
backups/doc_embed_migration_<timestamp>/ BEFORE deleting anything. An
INCOMPLETE marker file lives next to the dump from before the collection is
deleted until the restored point count is verified, so a run that dies
mid-restore is detected: the next run re-restores from that dump instead of
mistaking the already-1024d (but partial) collection for a finished
migration. Skips only when the dimension matches AND no INCOMPLETE marker
is left behind.

Run with the service STOPPED, after the image is rebuilt with the new
requirements and DOC_EMBED_* is set in .env:

  docker compose stop llamaindex
  docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo/llamaindex-service \\
      llamaindex python /repo/scripts/migrate_doc_embed.py
  docker compose up -d llamaindex
"""

import gzip
import json
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

sys.path.insert(0, "/repo/llamaindex-service")
from app import config, hybrid, providers, qdrant_setup  # noqa: E402

BACKUP_ROOT = Path("/repo/backups")
BATCH = 16


def main() -> int:
    if not config.DOC_EMBED_MODEL:
        print("DOC_EMBED_MODEL is not set — nothing to migrate.")
        return 1

    print(f"loading doc embedder {config.DOC_EMBED_MODEL} "
          f"({config.DOC_EMBED_PROVIDER or config.EMBED_PROVIDER})...")
    doc_embed = providers._build_embed(
        config.DOC_EMBED_PROVIDER or config.EMBED_PROVIDER, config.DOC_EMBED_MODEL
    )
    doc_dim = len(doc_embed.get_text_embedding("dimension probe"))
    # The memory embedder only supplies embed_dim for the untouched collections.
    embed_dim = len(providers.build_embed_model().get_text_embedding("dimension probe"))

    client = QdrantClient(url=config.QDRANT_URL)
    coll = config.DOCUMENTS_COLLECTION
    try:
        info = client.get_collection(coll)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            vectors = next(iter(vectors.values()), None)
        current_dim = getattr(vectors, "size", None)
    except Exception:
        # A previous run died between delete and recreate — the collection is
        # gone; the INCOMPLETE-marked dump below is the only complete copy.
        info, current_dim = None, None

    # An INCOMPLETE marker means an earlier run deleted the collection but
    # never verified the restore: the target dimension alone cannot prove the
    # data is complete, so those runs must be redone from their dump.
    pending = sorted(
        d for d in BACKUP_ROOT.glob("doc_embed_migration_*")
        if (d / "INCOMPLETE").exists()
    )
    if current_dim == doc_dim and not pending:
        print(f"{coll} already holds {doc_dim}-dim vectors — nothing to do.")
        return 0
    if info is None and not pending:
        print(f"{coll} does not exist and no INCOMPLETE backup was found — "
              "nothing to migrate from.")
        return 1

    if pending and (info is None or current_dim == doc_dim):
        # The live collection is missing or only partially restored — the
        # dump of the interrupted run is the authoritative copy. (A dump is
        # written and marked before its run deletes anything, so the newest
        # pending dump is always complete.)
        backup_dir = pending[-1]
        with gzip.open(backup_dir / f"{coll}.json.gz", "rt", encoding="utf-8") as fh:
            rows = json.load(fh)
        print(f"resuming interrupted migration: {len(rows)} points from {backup_dir}")
    else:
        print(f"{coll}: {info.points_count} points, {current_dim}d -> {doc_dim}d")

        # --- dump (backup + work list) -------------------------------------
        rows, offset = [], None
        while True:
            batch, offset = client.scroll(
                collection_name=coll, limit=256, offset=offset,
                with_payload=True, with_vectors=True,
            )
            for p in batch:
                dense = p.vector
                if isinstance(dense, dict):
                    dense = dense.get("")
                rows.append({"id": str(p.id), "vector": dense, "payload": p.payload or {}})
            if offset is None:
                break
        backup_dir = BACKUP_ROOT / f"doc_embed_migration_{time.strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        (backup_dir / "INCOMPLETE").write_text(
            "Migration not verified yet — removed automatically on success.\n"
        )
        with gzip.open(backup_dir / f"{coll}.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False)
        print(f"dumped {len(rows)} points -> {backup_dir / (coll + '.json.gz')}")

    # --- rebuild the exact embed-time text of every chunk -------------------
    from llama_index.core.schema import MetadataMode
    from llama_index.core.vector_stores.utils import metadata_dict_to_node

    embed_texts = []
    for row in rows:
        try:
            node = metadata_dict_to_node(row["payload"])
            embed_texts.append(node.get_content(metadata_mode=MetadataMode.EMBED))
        except Exception:
            # Non-LlamaIndex payload (shouldn't exist) — fall back to raw text.
            try:
                embed_texts.append(
                    json.loads(row["payload"].get("_node_content") or "{}").get("text", "")
                )
            except (ValueError, TypeError):
                embed_texts.append("")

    # --- recreate + restore --------------------------------------------------
    try:
        client.delete_collection(coll)
    except Exception:
        pass  # already gone (previous run died between delete and recreate)
    # allow_doc_space_change: the recorded doc space is the OLD embedder (a
    # lean install stamps the global MiniLM there), which would trip the
    # boot-time mismatch guard mid-migration. Meta keeps the old space until
    # record_doc_space() below confirms the verified restore.
    qdrant_setup.ensure_all(
        client, embed_dim, doc_embed_dim=doc_dim, allow_doc_space_change=True
    )

    t0 = time.time()
    for start in range(0, len(rows), BATCH):
        chunk_rows = rows[start:start + BATCH]
        dense_vecs = doc_embed.get_text_embedding_batch(
            embed_texts[start:start + BATCH]
        )
        points = []
        for row, dense in zip(chunk_rows, dense_vecs):
            try:
                raw_text = json.loads(
                    row["payload"].get("_node_content") or "{}"
                ).get("text", "")
            except (ValueError, TypeError):
                raw_text = ""
            sparse = hybrid.text_vector(raw_text)
            vector = {"": dense, config.BM25_VECTOR_NAME: sparse} \
                if sparse is not None else dense
            points.append(qmodels.PointStruct(
                id=row["id"], vector=vector, payload=row["payload"]
            ))
        client.upsert(collection_name=coll, points=points)
        done = min(start + BATCH, len(rows))
        print(f"  re-embedded {done}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)

    restored = client.count(collection_name=coll, exact=True).count
    if restored != len(rows):
        print(f"!! count mismatch: {restored}/{len(rows)} — restore from {backup_dir}")
        return 1
    qdrant_setup.record_doc_space(
        client, config.DOC_EMBED_PROVIDER or config.EMBED_PROVIDER,
        config.DOC_EMBED_MODEL, doc_dim,
    )
    for d in {*pending, backup_dir}:
        (d / "INCOMPLETE").unlink(missing_ok=True)
    print(f"\nMigration complete: {restored} points in {doc_dim}d. "
          "Start the service (docker compose up -d llamaindex).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
