#!/usr/bin/env python3
"""Rename the Qdrant collections hermes_* -> longbrain_* (identity cleanup).

Qdrant has no rename: each collection is copied point-by-point (dense +
sparse vectors and payload preserved verbatim) into a fresh collection
created with the current schema + payload indexes, verified by exact count,
and only then is the old collection deleted. Every old collection is dumped
to backups/collection_rename_<ts>/*.json.gz first.

The NEW names come from config (env) — run this only after .env /
docker-compose.yml point at the longbrain_* names. The OLD name of each
collection is derived by swapping the prefix back to hermes_. Idempotent:
nothing to do when no hermes_* source exists; refuses to guess when both
old and new exist with data.

Run (from the repo root, service stopped so nothing writes mid-copy):

  docker compose stop llamaindex
  docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo/llamaindex-service \\
      llamaindex python /repo/scripts/migrate_collection_names.py
  docker compose up -d --build llamaindex
"""

import gzip
import json
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

sys.path.insert(0, "/repo/llamaindex-service")
from app import config, qdrant_setup  # noqa: E402

BACKUP_ROOT = Path("/repo/backups")
BATCH = 64
NEW_PREFIX, OLD_PREFIX = "longbrain_", "hermes_"


def _old_name(new: str) -> str | None:
    return OLD_PREFIX + new[len(NEW_PREFIX):] if new.startswith(NEW_PREFIX) else None


def _jsonable_vector(vector):
    """Vectors as stored: bare dense list, or {"": dense, "bm25": sparse}."""
    if isinstance(vector, dict):
        return {
            name: ({"indices": list(v.indices), "values": list(v.values)}
                   if isinstance(v, qmodels.SparseVector) else v)
            for name, v in vector.items()
        }
    return vector


def dump_points(client: QdrantClient, collection: str) -> list:
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=collection, limit=256, offset=offset,
            with_payload=True, with_vectors=True,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def main() -> int:
    client = QdrantClient(url=config.QDRANT_URL)
    existing = {c.name for c in client.get_collections().collections}

    new_names = [
        config.META_COLLECTION,          # first: ensure_all reads meta afterwards
        config.MEMORIES_COLLECTION,
        config.CHAT_HISTORY_COLLECTION,
        config.DOCUMENTS_COLLECTION,
    ]
    todo = []
    for new in new_names:
        old = _old_name(new)
        if old is None:
            print(f"{new}: configured name is not {NEW_PREFIX}* — flip .env/compose first")
            return 1
        if old not in existing:
            print(f"{old}: absent — nothing to migrate for {new}")
            continue
        if new in existing and client.count(collection_name=new, exact=True).count > 0:
            print(f"{new}: already exists WITH data while {old} is still present — "
                  f"resolve manually (delete whichever is stale), aborting")
            return 1
        todo.append((old, new))
    if not todo:
        print("Nothing to migrate.")
        return 0

    # embed_dim must come from the OLD meta collection (the new one is empty).
    old_meta = OLD_PREFIX + config.META_COLLECTION[len(NEW_PREFIX):]
    meta_points = client.retrieve(
        collection_name=old_meta, ids=[qdrant_setup.META_POINT_ID], with_payload=True
    ) if old_meta in existing else []
    embed_dim = int((meta_points[0].payload or {}).get("embed_dim", 0)) if meta_points else 0
    if not embed_dim:
        print(f"No embed_dim in {old_meta} — is this a fresh install? Aborting.")
        return 1

    backup_dir = BACKUP_ROOT / f"collection_rename_{time.strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    dumps = {}
    for old, _ in todo:
        points = dump_points(client, old)
        rows = [{"id": str(p.id), "vector": _jsonable_vector(p.vector),
                 "payload": p.payload or {}} for p in points]
        with gzip.open(backup_dir / f"{old}.json.gz", "wt", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False)
        dumps[old] = points
        print(f"{old}: dumped {len(points)} points -> {backup_dir / (old + '.json.gz')}")

    qdrant_setup.ensure_all(client, embed_dim)  # creates the longbrain_* schema

    for old, new in todo:
        points = dumps[old]
        for start in range(0, len(points), BATCH):
            client.upsert(collection_name=new, points=[
                qmodels.PointStruct(id=str(p.id), vector=p.vector, payload=p.payload or {})
                for p in points[start:start + BATCH]
            ])
        copied = client.count(collection_name=new, exact=True).count
        if copied != len(points):
            print(f"{new}: copied {copied}/{len(points)} [MISMATCH] — {old} kept, "
                  f"restore reference: {backup_dir}")
            return 1
        client.delete_collection(old)
        print(f"{old} -> {new}: {copied}/{len(points)} points [OK], old deleted")

    print("\nRename complete. Start the service: docker compose up -d --build llamaindex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
