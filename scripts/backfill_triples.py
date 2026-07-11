#!/usr/bin/env python3
"""Backfill supersession triples onto facts saved before TRIPLE_SUPERSEDE.

Facts stored before the triple feature carry no (subject, relation, object)
payload, so a new fact can never retire their stale values. This script
closes that gap in two passes:

  Pass 1 — extraction: every active fact without a triple_subject is sent
  (in batches) through the configured LLM with the same attribute vocabulary
  the consolidation prompt uses; valid triples become part of the plan.
  Pass 2 — supersede sweep: active facts sharing (user, subject, relation)
  within the same recall co-visibility scope (same project or the default
  project; preferences are global) keep only the newest created_at — older
  ones get superseded_by, exactly what save-time supersession would have done.

Review-then-apply contract: the default (plan) run calls the LLM, prints the
full plan AND saves it to backups/backfill_triples_plan.json without writing
anything to Qdrant. --apply reads that file back and writes EXACTLY it — no
new LLM calls, so what was reviewed is what lands (LLM output is not
deterministic; re-extracting on apply could write pairs nobody reviewed).

Only set_payload is used (no collection recreate), so the service may stay
up. Idempotent: applying the same plan twice skips already-written entries,
and a new plan run skips facts that already carry a triple.

The plan run requires an LLM (--apply does not). If the service runs with
LLM_PROVIDER=none, override the env for that one invocation:

  docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo \\
      -e LLM_PROVIDER=anthropic -e LLM_MODEL=claude-haiku-4-5-20251001 \\
      -e ANTHROPIC_API_KEY=... \\
      llamaindex python /repo/scripts/backfill_triples.py
  # review the printed plan, then:
  docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo \\
      llamaindex python /repo/scripts/backfill_triples.py --apply
"""

import argparse
import json
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

sys.path.insert(0, "/repo/llamaindex-service")
from app import config, memories  # noqa: E402
from app.consolidation import TRIPLE_VOCABULARY  # noqa: E402
from app.providers import build_llm  # noqa: E402

BATCH = 20
PLAN_FILE = Path("/repo/backups/backfill_triples_plan.json")

EXTRACT_PROMPT = """\
You extract supersession triples from stored long-term memory facts.
For each numbered fact below, decide whether it states the CURRENT value of
a single-valued attribute. If it does, produce its subject, relation and
object. {vocabulary}

Return ONLY a JSON array (no prose, no code fences), one entry per fact that
HAS a triple, referencing the fact by its number:
[{{"i": 0, "subject": "...", "relation": "...", "object": "..."}}]
Facts without a single-valued attribute are simply omitted from the array.

Facts:
{facts}"""


def _scroll_all(client: QdrantClient, flt: qmodels.Filter) -> list:
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=config.MEMORIES_COLLECTION, scroll_filter=flt,
            limit=256, offset=offset, with_payload=True,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def _active_no_triple_filter() -> qmodels.Filter:
    return qmodels.Filter(
        must=[
            qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="superseded_by")),
            qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="triple_subject")),
        ],
        must_not=[
            qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="session_summary")),
        ],
    )


def _parse_batch_reply(raw: str, batch_size: int) -> dict[int, dict]:
    """index -> raw triple dict; anything unparseable is dropped silently."""
    start = raw.find("[")
    if start == -1:
        return {}
    try:
        entries = json.loads(raw[start:raw.rfind("]") + 1])
    except ValueError:
        return {}
    out = {}
    for e in entries:
        if isinstance(e, dict) and isinstance(e.get("i"), int) and 0 <= e["i"] < batch_size:
            out[e["i"]] = e
    return out


def extract_pass(client: QdrantClient, llm) -> dict[str, tuple]:
    """Plan triples for active facts that have none: {point_id: triple}."""
    points = _scroll_all(client, _active_no_triple_filter())
    print(f"Pass 1: {len(points)} active fact(s) without a triple")
    planned: dict[str, tuple] = {}
    for s in range(0, len(points), BATCH):
        batch = points[s:s + BATCH]
        numbered = "\n".join(
            f"{i}. {p.payload.get('text', '')}" for i, p in enumerate(batch)
        )
        try:
            raw = llm.complete(EXTRACT_PROMPT.format(
                vocabulary=TRIPLE_VOCABULARY, facts=numbered,
            )).text
        except Exception as exc:  # a flaky batch must not kill the run
            print(f"  !! LLM call failed for batch at {s}: {exc}")
            continue
        for i, entry in _parse_batch_reply(raw, len(batch)).items():
            triple = memories._triple_of(entry)
            if not triple:
                continue
            point = batch[i]
            print(f"  ({triple[0]}, {triple[1]}, {triple[2]})  <-  "
                  f"{point.payload.get('text', '')[:80]}")
            planned[str(point.id)] = triple
    print(f"Pass 1: {len(planned)} triple(s) planned")
    return planned


def _covisible_buckets(group: list) -> list[list]:
    """Split a (user, subject, relation) group by recall co-visibility:
    preferences are global — save-time supersession (memories.py's should
    filter: project match OR type=preference) lets them compete with every
    project's facts, so they're folded into each per-project/default bucket
    rather than judged only among themselves."""
    prefs = [p for p in group if p.payload.get("type") == "preference"]
    rest = [p for p in group if p.payload.get("type") != "preference"]
    default = [p for p in rest if p.payload.get("project_id") == config.DEFAULT_PROJECT]
    projects = sorted({p.payload.get("project_id") for p in rest} - {config.DEFAULT_PROJECT})
    if not projects:
        return [default + prefs]
    return [
        [p for p in rest if p.payload.get("project_id") == proj] + default + prefs
        for proj in projects
    ]


def supersede_pass(client: QdrantClient, planned: dict[str, tuple]) -> list[tuple]:
    """Plan supersessions over existing triples merged with the pass-1 plan.
    Returns (old_id, new_id) pairs; writes nothing."""
    points = _scroll_all(client, qmodels.Filter(must=[
        qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="superseded_by")),
    ], must_not=[
        qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key="triple_subject")),
    ]))
    # Pass 1 wrote nothing, so merge its planned triples in memory — the
    # plan must include the pairs they would create.
    no_triple = _scroll_all(client, _active_no_triple_filter())
    for p in no_triple:
        triple = planned.get(str(p.id))
        if triple:
            p.payload["triple_subject"], p.payload["triple_relation"], \
                p.payload["triple_object"] = triple
            points.append(p)
    groups: dict[tuple, list] = {}
    for p in points:
        key = (p.payload.get("user_id"), p.payload.get("triple_subject"),
               p.payload.get("triple_relation"))
        groups.setdefault(key, []).append(p)

    pairs: list[tuple] = []
    seen_old: set[str] = set()
    for key, group in sorted(groups.items()):
        for bucket in _covisible_buckets(group):
            if len(bucket) < 2:
                continue
            bucket.sort(key=lambda p: p.payload.get("created_at") or 0)
            newest = bucket[-1]
            for old in bucket[:-1]:
                if str(old.id) in seen_old:
                    continue  # already planned via another bucket
                seen_old.add(str(old.id))
                print(f"  supersede ({key[1]}, {key[2]}):\n"
                      f"    old: {old.payload.get('text', '')[:80]}\n"
                      f"    new: {newest.payload.get('text', '')[:80]}")
                pairs.append((str(old.id), str(newest.id)))
    print(f"Pass 2: {len(pairs)} stale fact(s) planned")
    return pairs


def apply_plan(client: QdrantClient, plan: dict) -> None:
    """Write a reviewed plan. Skips entries already written or whose points
    disappeared since the plan run, so re-applying is safe."""
    written = 0
    for pid, triple in plan.get("triples", {}).items():
        existing = client.retrieve(
            collection_name=config.MEMORIES_COLLECTION, ids=[pid], with_payload=True
        )
        if not existing or existing[0].payload.get("triple_subject"):
            continue
        client.set_payload(
            collection_name=config.MEMORIES_COLLECTION,
            payload={"triple_subject": triple[0], "triple_relation": triple[1],
                     "triple_object": triple[2]},
            points=[pid],
        )
        written += 1
    print(f"Pass 1: {written} triple(s) written")

    retired = 0
    for old_id, new_id in plan.get("supersedes", []):
        existing = client.retrieve(
            collection_name=config.MEMORIES_COLLECTION, ids=[old_id, new_id], with_payload=True
        )
        by_id = {str(p.id): p for p in existing}
        old_point, new_point = by_id.get(old_id), by_id.get(new_id)
        if not old_point or old_point.payload.get("superseded_by"):
            continue
        # The target may have been deleted or itself superseded between the
        # plan run and --apply — only retire old_id into a still-active target.
        if not new_point or new_point.payload.get("superseded_by"):
            continue
        client.set_payload(
            collection_name=config.MEMORIES_COLLECTION,
            payload={"superseded_by": new_id},
            points=[old_id],
        )
        print(f"  retired: {old_point.payload.get('text', '')[:80]}")
        retired += 1
    print(f"Pass 2: {retired} stale fact(s) retired")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help=f"write the reviewed plan from {PLAN_FILE} "
                             "(default: compute + save the plan, write nothing)")
    args = parser.parse_args()

    client = QdrantClient(url=config.QDRANT_URL)

    if args.apply:
        if not PLAN_FILE.exists():
            print(f"No plan file at {PLAN_FILE} — run without --apply first, "
                  "review its output, then re-run with --apply.")
            return 1
        plan = json.loads(PLAN_FILE.read_text())
        apply_plan(client, plan)
        return 0

    llm = build_llm()
    if llm is None:
        print("LLM_PROVIDER=none — the plan run needs an LLM, e.g.:\n"
              '  docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo \\\n'
              "      -e LLM_PROVIDER=... -e LLM_MODEL=... \\\n"
              "      llamaindex python /repo/scripts/backfill_triples.py")
        return 1

    print("PLAN RUN — nothing will be written. Review, then re-run with --apply.\n")
    planned = extract_pass(client, llm)
    pairs = supersede_pass(client, planned)
    PLAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    PLAN_FILE.write_text(json.dumps({
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "triples": {pid: list(t) for pid, t in planned.items()},
        "supersedes": [list(p) for p in pairs],
    }, ensure_ascii=False, indent=1))
    print(f"\nPlan saved to {PLAN_FILE}. Review the output above, then re-run "
          "with --apply to write exactly this plan (no new LLM calls).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
