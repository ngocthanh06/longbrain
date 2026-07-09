#!/usr/bin/env python3
"""Re-embed the L4 document knowledge base after an embedding-content fix.

Why this exists: chunks indexed before documents.EXCLUDED_METADATA_KEYS was
introduced were embedded WITH their bookkeeping metadata prepended
(file_path/stored_path/user_id/...), which wrecked retrieval ranking. The
stored vectors can't be fixed in place — every chunk has to be re-embedded
from its text. This script:

1. scrolls every point out of the documents collection (text + metadata),
2. writes a full backup JSON next to the watcher state (~/.hermes/),
3. reconstructs each original document's text from its chunks via their
   start/end char offsets (chunks overlap, so naive joining would duplicate),
4. drops the collection and restarts the llamaindex container so
   qdrant_setup recreates it empty,
5. re-ingests every document through POST /ingest/text with its original
   metadata (project_id, source, stored_path, ... all preserved — so the
   watcher's stored_path dedup keeps working and nothing gets re-sent twice).

The ingest watcher's state file is left alone on purpose: metadata is
preserved, so from the watcher's point of view nothing changed.

Run on the host: python3 scripts/reembed_documents.py [--dry-run|--check]

--check (used by setup.sh on upgrades): only report whether the collection
still holds chunks embedded the old way — exit 0 when clean (or empty, or
unreachable), exit 2 when a re-embed is needed.
"""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MEMORY_URL = "http://localhost:8800"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "hermes_documents"
BACKUP = Path.home() / ".hermes" / f"{COLLECTION}_reembed_backup_{time.strftime('%Y%m%d_%H%M%S')}.json"
REPO = Path(__file__).resolve().parent.parent


def http_json(url: str, body=None, method: str = None, timeout: float = 120.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if data else "GET"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def fetch_all_points() -> list:
    points, offset = [], None
    while True:
        body = {"limit": 100, "with_payload": True, "with_vector": False}
        if offset:
            body["offset"] = offset
        result = http_json(f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll", body)["result"]
        points.extend(result["points"])
        offset = result.get("next_page_offset")
        if not offset:
            return points


def group_documents(points: list) -> dict:
    """ref_doc_id -> {"chunks": [(start, end, text)], "metadata": {...}}"""
    groups = {}
    for p in points:
        payload = p["payload"]
        node = json.loads(payload["_node_content"])
        ref = payload.get("ref_doc_id") or node.get("id_")
        entry = groups.setdefault(ref, {"chunks": [], "metadata": {}})
        start = node.get("start_char_idx")
        entry["chunks"].append((start if start is not None else 0, node.get("end_char_idx"), node["text"]))
        if not entry["metadata"]:
            entry["metadata"] = {
                k: v for k, v in payload.items()
                if not k.startswith("_") and k not in ("ref_doc_id", "doc_id", "document_id")
            }
    return groups


def reconstruct_text(chunks: list) -> str:
    """Rebuild the original text from overlapping chunks by char offsets.
    Chunks whose offsets are missing are appended at the end as-is."""
    positioned = [c for c in chunks if c[1] is not None]
    loose = [c for c in chunks if c[1] is None]
    buf: list = []
    for start, _end, text in sorted(positioned, key=lambda c: c[0]):
        if len(buf) < start:
            buf.extend(" " * (start - len(buf)))
        for i, ch in enumerate(text):
            pos = start + i
            if pos < len(buf):
                buf[pos] = ch
            else:
                buf.append(ch)
    parts = ["".join(buf)] if buf else []
    parts += [t for _s, _e, t in loose]
    return "\n\n".join(parts)


def check_dirty() -> int:
    """Exit 2 when any chunk was embedded with the old metadata-polluted
    content (its node lacks the admin keys in excluded_embed_metadata_keys),
    0 otherwise. Never blocks setup: unreachable service/collection -> 0."""
    try:
        points = fetch_all_points()
    except Exception as exc:
        print(f"  could not inspect {COLLECTION} ({exc}) — skipping the check")
        return 0
    dirty = 0
    for p in points:
        raw = (p.get("payload") or {}).get("_node_content")
        if not raw:
            continue
        node = json.loads(raw)
        if "user_id" not in (node.get("excluded_embed_metadata_keys") or []):
            dirty += 1
    if dirty:
        print(f"  {dirty}/{len(points)} chunks still use old-style embeddings "
              "(metadata was embedded with the text, which ruins ranking)")
        return 2
    print(f"  all {len(points)} chunks are clean")
    return 0


def main() -> int:
    if "--check" in sys.argv:
        return check_dirty()
    dry_run = "--dry-run" in sys.argv

    print(f"==> fetching all points from {COLLECTION}")
    points = fetch_all_points()
    print(f"  {len(points)} points")
    if not points:
        print("  nothing to migrate")
        return 0

    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    BACKUP.write_text(json.dumps(points, ensure_ascii=False))
    print(f"  backup written: {BACKUP}")

    groups = group_documents(points)
    print(f"  {len(groups)} source documents")
    docs = []
    for ref, entry in groups.items():
        text = reconstruct_text(entry["chunks"])
        docs.append((entry["metadata"], text))
        if not text.strip():
            print(f"  WARN: empty reconstruction for {ref} ({entry['metadata'].get('source')})")
    if dry_run:
        for meta, text in docs:
            print(f"  [dry-run] {meta.get('project_id')}/{meta.get('source', '?')}: {len(text)} chars")
        return 0

    print(f"==> dropping collection {COLLECTION}")
    http_json(f"{QDRANT_URL}/collections/{COLLECTION}", method="DELETE")

    print("==> restarting llamaindex so it recreates the collection")
    subprocess.run(
        ["docker", "compose", "restart", "llamaindex"], cwd=str(REPO), check=True
    )
    for _ in range(60):
        try:
            http_json(f"{MEMORY_URL}/health", timeout=3)
            break
        except Exception:
            time.sleep(2)
    else:
        print("✗ service did not come back — restore from backup and investigate")
        return 1

    print(f"==> re-ingesting {len(docs)} documents")
    failed = 0
    for meta, text in docs:
        meta = dict(meta)
        project_id = meta.pop("project_id", "") or "default"
        meta.pop("user_id", None)  # service re-stamps it
        try:
            http_json(f"{MEMORY_URL}/ingest/text", {
                "text": text, "metadata": meta, "project_id": project_id,
            })
            print(f"  ok: {project_id}/{meta.get('source', '(text note)')}")
        except Exception as exc:
            failed += 1
            print(f"  ✗ FAILED {project_id}/{meta.get('source', '?')}: {exc}")

    total = http_json(f"{QDRANT_URL}/collections/{COLLECTION}")["result"]["points_count"]
    print(f"==> done: {len(docs) - failed}/{len(docs)} documents re-ingested, "
          f"{total} points now in {COLLECTION}")
    if failed:
        print(f"✗ {failed} documents failed — their text is in the backup: {BACKUP}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
