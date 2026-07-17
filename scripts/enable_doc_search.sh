#!/usr/bin/env bash
# Activate the document-search tier (SEARCH_SPEC) on an EXISTING LongBrain
# install. Idempotent.
#
# What it does:
#   1. Sets INSTALL_DOC_SEARCH=true + DOC_EMBED_* in .env
#   2. Rebuilds the service image with the doc-search stack (torch + ST)
#   3. Re-embeds the existing documents collection into the BGE-M3 space
#      (scripts/migrate_doc_embed.py — backs up first, memories untouched)
#   4. Restarts the service and verifies /health doc_embedder_ready
#
# Memory-only users never run this and keep the lean image; document
# endpoints on a lean install answer 503 doc_embedder_unavailable.

set -euo pipefail
cd "$(dirname "$0")/.."

ENV_FILE=".env"
[ -f "$ENV_FILE" ] || { echo "No .env — run ./setup.sh first."; exit 1; }

set_env() { # set_env KEY VALUE — replace or append
  local key="$1" value="$2"
  if grep -q "^#*${key}=" "$ENV_FILE"; then
    sed -i.bak "s|^#*${key}=.*|${key}=${value}|" "$ENV_FILE" && rm -f "$ENV_FILE.bak"
  else
    printf '\n%s=%s\n' "$key" "$value" >> "$ENV_FILE"
  fi
}

echo "==> enabling doc search in .env"
set_env INSTALL_DOC_SEARCH true
set_env DOC_EMBED_PROVIDER huggingface
set_env DOC_EMBED_MODEL BAAI/bge-m3

echo "==> rebuilding service image with the doc-search stack"
docker compose build llamaindex

echo "==> re-embedding the documents collection (service stopped meanwhile)"
docker compose stop llamaindex
docker compose run --rm --no-deps -v "$PWD:/repo" -w /repo/llamaindex-service \
  llamaindex python /repo/scripts/migrate_doc_embed.py

echo "==> starting service"
docker compose up -d llamaindex

echo "==> waiting for health"
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:8800/health 2>/dev/null | grep -q '"doc_embedder_ready":true'; then
    echo "doc search is ACTIVE (doc_embedder_ready: true)"
    exit 0
  fi
  sleep 2
done
echo "Service did not report doc_embedder_ready — check: curl localhost:8800/health"
exit 1
