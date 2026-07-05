#!/usr/bin/env bash
# Snapshot every hermes_* Qdrant collection into ./backups/<timestamp>/,
# keep the newest $BACKUP_KEEP backups, log to logs/backup.log.
# Run manually or via the com.hermes.memory-backup launchd agent (2:00 AM).
set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
KEEP="${BACKUP_KEEP:-7}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_DIR="$ROOT/backups/$STAMP"
LOG="$ROOT/logs/backup.log"
mkdir -p "$BACKUP_DIR" "$ROOT/logs"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if ! curl -fsS "$QDRANT_URL/collections" >/dev/null 2>&1; then
  log "SKIP: Qdrant không phản hồi tại $QDRANT_URL"
  rmdir "$BACKUP_DIR" 2>/dev/null || true
  exit 0
fi

collections=$(curl -fsS "$QDRANT_URL/collections" | python3 -c \
  "import json,sys; print('\n'.join(c['name'] for c in json.load(sys.stdin)['result']['collections'] if c['name'].startswith('hermes_')))")

for col in $collections; do
  name=$(curl -fsS -X POST "$QDRANT_URL/collections/$col/snapshots" | python3 -c \
    "import json,sys; print(json.load(sys.stdin)['result']['name'])")
  curl -fsS "$QDRANT_URL/collections/$col/snapshots/$name" -o "$BACKUP_DIR/$col.snapshot"
  curl -fsS -X DELETE "$QDRANT_URL/collections/$col/snapshots/$name" >/dev/null
  log "snapshot $col -> $BACKUP_DIR/$col.snapshot ($(du -h "$BACKUP_DIR/$col.snapshot" | cut -f1))"
done

cp "$ROOT/.env" "$BACKUP_DIR/env.backup" 2>/dev/null || true

# Retention: keep newest $KEEP (awk — BSD head has no negative -n)
cd "$ROOT/backups"
ls -1d */ 2>/dev/null | sort | awk -v keep="$KEEP" '{a[NR]=$0} END{for(i=1;i<=NR-keep;i++) print a[i]}' | while read -r old; do
  rm -rf "$old"
  log "retention: đã xoá backup cũ $old"
done

log "backup hoàn tất: $BACKUP_DIR"
