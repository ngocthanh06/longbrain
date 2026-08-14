#!/usr/bin/env bash
# Quick, guarded cleanup for superseded document chunks.
# Usage: ./scripts/clean_garbage.sh [--yes]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_URL="${MEMORY_SERVICE_URL:-http://localhost:8800}"
# Read through hooks/api_auth.py, not process env directly: a fresh
# terminal that hasn't sourced .env would otherwise see an empty key even
# when .env has one configured. Empty (default) = auth disabled; matches it.
API_KEY="$(python3 "$ROOT/hooks/api_auth.py")"
AUTH_HEADER=()
[[ -n "$API_KEY" ]] && AUTH_HEADER=(-H "X-API-Key: $API_KEY")

command -v curl >/dev/null || { echo "ERROR: curl is required" >&2; exit 1; }

echo "Scanning superseded document chunks..."
preview="$(curl -fsS -X POST "$SERVICE_URL/maintenance/cleanup" \
  ${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"} \
  -H 'Content-Type: application/json' \
  -d '{"dry_run":true,"limit":10000}')"
echo "$preview" | python3 -m json.tool

if [[ "${1:-}" != "--yes" ]]; then
  printf "Run the destructive cleanup? Type CLEANUP SUPERSEDED: "
  read -r answer
  [[ "$answer" == "CLEANUP SUPERSEDED" ]] || { echo "Cancelled."; exit 0; }
fi

curl -fsS -X POST "$SERVICE_URL/maintenance/cleanup" \
  ${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"} \
  -H 'Content-Type: application/json' \
  -d '{"dry_run":false,"confirm":"CLEANUP SUPERSEDED","limit":10000}' \
  | python3 -m json.tool
