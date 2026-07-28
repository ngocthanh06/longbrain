#!/usr/bin/env python3
"""One-time repair: supersede orphaned pre-document_key chunks that
migrate_document_versions.py deliberately left untouched.

Why this exists: migrate_document_versions.py only compares chunks that
already carry document_key — chunks ingested before that field existed have
none, and the script correctly refuses to group them by basename (that
guesswork, across projects/folders, is exactly what caused the earlier
gakken/rz-server supersession bug). But WITHIN a single (project_id, source)
pair, project_id and source are literal payload fields already recorded on
each orphan chunk — nothing is inferred — so when exactly one OTHER chunk
in that same project_id+source group is both active (no superseded_by) and
already carries document_key (i.e. the winner from the current docs/ file,
tagged by a --force ingest_watcher pass), every orphan in the group can be
superseded to it safely. A group with zero or more than one such winner is
left alone and reported instead of guessed.

Safety: dumps every affected point (id + payload) to
backups/orphan_document_versions_<timestamp>.json before writing anything.
--dry-run reports the plan without writing.

Run inside the container (qdrant_client + app config live there):

  docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo/llamaindex-service \\
      llamaindex python /repo/scripts/repair_orphan_document_versions.py --dry-run
  docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo/llamaindex-service \\
      llamaindex python /repo/scripts/repair_orphan_document_versions.py
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


def fetch_orphans(client: QdrantClient) -> list[dict]:
    """Active file-backed chunks with no document_key yet."""
    flt = qmodels.Filter(
        must_not=[
            qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="stored_path")),
        ],
        must=[
            qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="document_key")),
            qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="superseded_by")),
        ],
    )
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=config.DOCUMENTS_COLLECTION,
            scroll_filter=flt,
            limit=256, offset=offset,
            with_payload=["project_id", "source", "stored_path"],
            with_vectors=False,
        )
        points.extend({"id": p.id, "payload": p.payload or {}} for p in batch)
        if offset is None:
            break
    return points


def group_by_project_source(points: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for p in points:
        key = (p["payload"].get("project_id", ""), p["payload"].get("source", ""))
        groups.setdefault(key, []).append(p)
    return groups


def find_winner(client: QdrantClient, project_id: str, source: str) -> list[dict]:
    """Active, already-keyed chunks sharing this exact project_id+source."""
    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(key="project_id", match=qmodels.MatchValue(value=project_id)),
            qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=source)),
            qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="superseded_by")),
        ],
        must_not=[
            qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="document_key")),
        ],
    )
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=config.DOCUMENTS_COLLECTION,
            scroll_filter=flt,
            limit=64, offset=offset,
            with_payload=["document_key", "stored_path"],
            with_vectors=False,
        )
        points.extend({"id": p.id, "payload": p.payload or {}} for p in batch)
        if offset is None:
            break
    return points


def plan_repairs(client: QdrantClient, groups: dict) -> tuple[list[dict], list[dict]]:
    """Split into (safe, ambiguous) — the latter is reported, not applied."""
    safe, ambiguous = [], []
    for (project_id, source), orphans in groups.items():
        winners = find_winner(client, project_id, source)
        distinct_paths = {w["payload"].get("stored_path") for w in winners}
        if len(distinct_paths) != 1:
            ambiguous.append({
                "project_id": project_id, "source": source,
                "orphan_count": len(orphans), "winner_count": len(distinct_paths),
            })
            continue
        winner_stored_path = next(iter(distinct_paths))
        safe.append({
            "project_id": project_id, "source": source,
            "winner_stored_path": winner_stored_path,
            "orphan_ids": [o["id"] for o in orphans],
        })
    return safe, ambiguous


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    client = QdrantClient(url=config.QDRANT_URL)
    orphans = fetch_orphans(client)

    if not orphans:
        print("Nothing to do: no orphaned chunk lacks document_key.")
        return 0

    groups = group_by_project_source(orphans)
    safe, ambiguous = plan_repairs(client, groups)

    total_safe = sum(len(item["orphan_ids"]) for item in safe)
    print(f"{len(orphans)} orphan chunk(s) across {len(groups)} (project, source) group(s):")
    print(f"  {len(safe)} group(s) with exactly one active keyed winner -> "
          f"{total_safe} chunk(s) safe to supersede")
    for item in safe:
        print(f"    [{item['project_id']}] {item['source']}: "
              f"{len(item['orphan_ids'])} chunk(s) -> superseded_by={item['winner_stored_path']!r}")
    if ambiguous:
        print(f"  {len(ambiguous)} group(s) SKIPPED (0 or >1 active keyed winner — "
              "not guessed):")
        for item in ambiguous:
            print(f"    [{item['project_id']}] {item['source']}: "
                  f"{item['orphan_count']} orphan(s), {item['winner_count']} winner(s)")

    if dry_run:
        print("\n--dry-run: no changes written.")
        return 0

    if not safe:
        return 0

    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_ROOT / f"orphan_document_versions_{time.strftime('%Y%m%d_%H%M%S')}.json"
    backup_path.write_text(json.dumps(orphans, default=str, ensure_ascii=False, indent=2))
    print(f"Backup written to {backup_path}")

    for item in safe:
        client.set_payload(
            collection_name=config.DOCUMENTS_COLLECTION,
            payload={"superseded_by": item["winner_stored_path"]},
            points=item["orphan_ids"],
        )
    print(f"Done: {total_safe} chunk(s) superseded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
