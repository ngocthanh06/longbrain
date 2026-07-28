#!/usr/bin/env python3
"""One-time backfill: stamp `ingested_at` on document chunks that predate the
recency-decay feature in documents.search_chunks().

Why this exists: chunks ingested before ingested_at existed have no such
field. search_chunks() already falls back to age=0 (no decay penalty) for
those, so nothing is broken — but it also means old file-backed chunks never
decay until they get re-ingested. This script estimates their true ingestion
time from the original file's mtime on disk (DATA_DIR/documents/<stored_path>
is content-addressed and never modified after ingest, so its mtime is a
reliable proxy for when it was ingested) and backfills it.

Scope is deliberately narrow: only chunks that carry a `stored_path` (i.e.
produced by ingest_file()) and whose file still exists on disk are touched.
Manual text (add_to_knowledge_base) and enrichment summary chunks have no
file to date them from and no `stored_path` — they already get the safe
no-penalty fallback in search_chunks(), so they are intentionally skipped.
A stored_path whose file is missing is skipped too rather than guessed, to
avoid writing a wrong timestamp that would wrongly decay a still-relevant
chunk.

Safety: dumps every affected point (id + payload) to
backups/document_ingested_at_backfill_<timestamp>.json before writing
anything. --dry-run reports the plan without writing.

Run inside the container (qdrant_client + app config live there):

  docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo/llamaindex-service \\
      llamaindex python /repo/scripts/backfill_document_ingested_at.py --dry-run
  docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo/llamaindex-service \\
      llamaindex python /repo/scripts/backfill_document_ingested_at.py
"""

import json
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

sys.path.insert(0, "/repo/llamaindex-service")
from app import config  # noqa: E402


BACKUP_ROOT = Path("/repo/backups")


def fetch_undated_file_chunks(client: QdrantClient) -> list[dict]:
    """Every point with a stored_path but no ingested_at yet."""
    flt = qmodels.Filter(
        must_not=[
            qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="stored_path")),
        ],
        must=[
            qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="ingested_at")),
        ],
    )
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=config.DOCUMENTS_COLLECTION,
            scroll_filter=flt,
            limit=128, offset=offset,
            with_payload=["project_id", "source", "stored_path"],
            with_vectors=False,
        )
        points.extend({"id": p.id, "payload": p.payload or {}} for p in batch)
        if offset is None:
            break
    return points


def plan_backfill(points: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into (datable, missing_file) — the latter is skipped, not guessed."""
    datable, missing_file = [], []
    for p in points:
        stored_path = p["payload"].get("stored_path", "")
        try:
            mtime = Path(stored_path).stat().st_mtime
        except OSError:
            missing_file.append(p)
            continue
        datable.append({**p, "mtime": mtime})
    return datable, missing_file


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    client = QdrantClient(url=config.QDRANT_URL)
    points = fetch_undated_file_chunks(client)

    if not points:
        print("Nothing to do: every file-backed chunk already has ingested_at.")
        return 0

    datable, missing_file = plan_backfill(points)
    print(f"{len(points)} chunk(s) missing ingested_at: "
          f"{len(datable)} datable from disk, {len(missing_file)} skipped (file missing).")
    for p in datable[:20]:
        print(f"  [{p['payload'].get('project_id')}] {p['payload'].get('source')} "
              f"-> {time.strftime('%Y-%m-%d', time.localtime(p['mtime']))}")
    if len(datable) > 20:
        print(f"  ... and {len(datable) - 20} more")
    if missing_file:
        print(f"WARNING: {len(missing_file)} chunk(s) skipped — their source file is gone "
              "from disk, so ingested_at is left unset (search_chunks treats that as "
              "no decay penalty, same as today).")

    if dry_run:
        print("\n--dry-run: no changes written.")
        return 0

    if not datable:
        return 0

    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_ROOT / f"document_ingested_at_backfill_{time.strftime('%Y%m%d_%H%M%S')}.json"
    backup_path.write_text(json.dumps(points, default=str, ensure_ascii=False, indent=2))
    print(f"Backup written to {backup_path}")

    for p in datable:
        client.set_payload(
            collection_name=config.DOCUMENTS_COLLECTION,
            payload={"ingested_at": p["mtime"]},
            points=[p["id"]],
        )
    print(f"Done: {len(datable)} chunk(s) backfilled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
