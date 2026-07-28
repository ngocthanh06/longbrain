#!/usr/bin/env python3
"""One-time cleanup: mark stale document versions already sitting in the L4
knowledge base as superseded, so search excludes them by default.

Why this exists: before documents._supersede_previous_versions() started
running on every ingest_file() call, re-ingesting a changed file (new
content -> new content-addressed stored_path) never touched the old
version's chunks. Both versions have been coexisting in the collection ever
since, and a query like "what changed in this doc" can retrieve a mix of
both. This script backfills the same superseded_by marking for data that
predates the fix; new ingests handle themselves automatically.

Scope is deliberately narrow, matching the ingest-time logic: only chunks
that carry both a `stored_path` and a `document_key` are considered. Legacy
chunks without `document_key` are intentionally skipped: falling back to the
basename/source would recreate the collision that caused the data corruption.
The migration refuses to write while any such legacy file chunks remain; they
must first be repaired by keyed re-ingest or an explicit manifest.
Enrichment summary chunks (`enriched: true`) and manually added text
(add_to_knowledge_base) have no stored_path and no version concept, so they
are excluded from the scroll filter itself and never touched.

Within each (project_id, document_key) group with more than one distinct
stored_path, the version whose file on disk (DATA_DIR/documents/<stored_path>)
has the newest mtime is kept active; every other version in the group gets
superseded_by = <winning stored_path>. A stored_path whose file is missing
from disk sorts as oldest (mtime 0) — a safe default since a re-ingest never
deletes the original file, so a missing file means something else already
removed it.

Safety: dumps every affected point (id + payload) to
backups/document_versions_migration_<timestamp>.json before writing
anything. --dry-run reports the plan without writing.

Run inside the container (qdrant_client + app config live there):

  docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo/llamaindex-service \\
      llamaindex python /repo/scripts/migrate_document_versions.py --dry-run
  docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo/llamaindex-service \\
      llamaindex python /repo/scripts/migrate_document_versions.py
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


def fetch_versioned_points(client: QdrantClient) -> list[dict]:
    """Every point carrying both versioning fields; enrichment/manual-add
    chunks and legacy chunks are excluded by the filter."""
    must_not = [
        qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="stored_path")),
        qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="document_key")),
        qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="superseded_by")),
    ]
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=config.DOCUMENTS_COLLECTION,
            scroll_filter=qmodels.Filter(must_not=must_not),
            limit=128, offset=offset,
            with_payload=["project_id", "source", "document_key", "stored_path", "superseded_by"],
            with_vectors=False,
        )
        points.extend({"id": p.id, "payload": p.payload or {}} for p in batch)
        if offset is None:
            break
    return points


def count_legacy_points(client: QdrantClient) -> int:
    """Count file chunks that still lack document_key. They cannot be
    safely grouped by basename and require a subsequent keyed re-ingest or a
    separate manifest-based repair."""
    flt = qmodels.Filter(
        must=[qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="document_key"))],
        must_not=[qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="stored_path"))],
    )
    return client.count(
        collection_name=config.DOCUMENTS_COLLECTION,
        count_filter=flt,
        exact=True,
    ).count


def group_by_document_key(points: list[dict]) -> dict:
    groups: dict[tuple, dict] = {}
    for p in points:
        payload = p["payload"]
        key = (payload.get("project_id", ""), payload.get("document_key", ""))
        by_path = groups.setdefault(key, {})
        by_path.setdefault(payload.get("stored_path"), []).append(p["id"])
    return groups


def stored_path_mtime(stored_path: str) -> float:
    try:
        return Path(stored_path).stat().st_mtime
    except OSError:
        return 0.0  # file missing on disk -> treated as oldest


def plan_supersessions(groups: dict) -> list[dict]:
    plan = []
    for (project_id, document_key), by_path in groups.items():
        if len(by_path) < 2:
            continue  # only one version -> nothing to supersede
        winner = max(by_path, key=stored_path_mtime)
        supersede = [
            {"stored_path": sp, "ids": ids}
            for sp, ids in by_path.items() if sp != winner
        ]
        if supersede:
            plan.append({
                "project_id": project_id, "document_key": document_key,
                "keep": winner, "supersede": supersede,
            })
    return plan


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    client = QdrantClient(url=config.QDRANT_URL)
    points = fetch_versioned_points(client)
    legacy_count = count_legacy_points(client)
    groups = group_by_document_key(points)
    plan = plan_supersessions(groups)

    if legacy_count:
        print(
            f"BLOCKED: {legacy_count} legacy file chunk(s) lack document_key. "
            "Repair all of them before applying version migration."
        )
        if dry_run:
            print("--dry-run: no changes written.")
        return 2

    if not plan:
        print("Nothing to do: no keyed document has more than one active version.")
        if legacy_count:
            print(
                f"WARNING: {legacy_count} legacy file chunk(s) lack document_key "
                "and were not grouped by basename. Re-ingest or provide a "
                "manifest before attempting further cleanup."
            )
        return 0

    total_stale = sum(len(s["ids"]) for item in plan for s in item["supersede"])
    print(f"{len(plan)} document(s) with stale versions, {total_stale} chunk(s) to mark superseded:")
    for item in plan:
        stale_count = sum(len(s["ids"]) for s in item["supersede"])
        print(f"  [{item['project_id']}] {item['document_key']}: keep {item['keep']!r}, "
              f"supersede {stale_count} chunk(s) from {len(item['supersede'])} old version(s)")

    if dry_run:
        print("\n--dry-run: no changes written.")
        return 0

    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_ROOT / f"document_versions_migration_{time.strftime('%Y%m%d_%H%M%S')}.json"
    backup_path.write_text(json.dumps(points, default=str, ensure_ascii=False, indent=2))
    print(f"Backup written to {backup_path}")

    for item in plan:
        for s in item["supersede"]:
            client.set_payload(
                collection_name=config.DOCUMENTS_COLLECTION,
                payload={"superseded_by": item["keep"]},
                points=s["ids"],
            )
    print(f"Done: {total_stale} chunk(s) marked superseded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
