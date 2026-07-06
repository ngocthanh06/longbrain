#!/usr/bin/env bash
# Export/import the whole memory stack as a text-level JSON bundle — for
# moving to another machine (or another embedding model). Import re-embeds
# with the current model and skips records that already exist, so it is
# safe to re-run.
#
#   ./scripts/memory_transfer.sh export [file.json]   # default: backups/memory-export-<stamp>.json
#   ./scripts/memory_transfer.sh import <file.json>
set -euo pipefail

SERVICE_URL="${MEMORY_SERVICE_URL:-http://localhost:8800}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() { echo "usage: $0 export [file.json] | import <file.json>" >&2; exit 1; }
[ $# -ge 1 ] || usage

if ! curl -fsS "$SERVICE_URL/health" >/dev/null 2>&1; then
  echo "ERROR: memory service not responding at $SERVICE_URL (is 'docker compose up' running?)" >&2
  exit 1
fi

case "$1" in
  export)
    out="${2:-$ROOT/backups/memory-export-$(date +%Y%m%d-%H%M%S).json}"
    mkdir -p "$(dirname "$out")"
    curl -fsS "$SERVICE_URL/memory/export" -o "$out"
    python3 -c "
import json, sys
c = json.load(open(sys.argv[1]))['counts']
print(f\"exported {c['facts']} facts, {c['turns']} turns, {c['documents']} document chunks\")
" "$out"
    echo "bundle: $out"
    ;;
  import)
    [ $# -ge 2 ] || usage
    [ -f "$2" ] || { echo "ERROR: file not found: $2" >&2; exit 1; }
    curl -fsS -X POST "$SERVICE_URL/memory/import" \
      -H "Content-Type: application/json" --data-binary "@$2" | python3 -m json.tool
    ;;
  *) usage ;;
esac
