#!/usr/bin/env bash
# One-shot setup for the Hermes memory stack — zero manual configuration.
# Usage: ./setup.sh
set -euo pipefail

cd "$(dirname "$0")"

BOLD=$(tput bold 2>/dev/null || true)
RESET=$(tput sgr0 2>/dev/null || true)
step() { echo; echo "${BOLD}==> $*${RESET}"; }

# 1. Docker available?
step "Checking Docker"
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed. Install Docker Desktop first: https://docs.docker.com/get-docker/"
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running. Open Docker Desktop, then run ./setup.sh again"
  exit 1
fi
echo "OK"

# 2. .env
step "Configuring .env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example (default: local fastembed, no API key needed)."
else
  echo ".env already exists, keeping it as-is."
fi

# 3. Build + up
step "Starting containers (first run builds the image, takes a few minutes)"
docker compose up -d --build

# 4. Wait for health
step "Waiting for the memory service to become ready"
for i in $(seq 1 60); do
  if curl -fsS http://localhost:8800/health >/dev/null 2>&1; then
    echo "Service OK: http://localhost:8800"
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "Service did not come up after 5 minutes. Check logs: docker compose logs llamaindex"
    exit 1
  fi
  sleep 5
done

# 5. Wire Hermes Desktop automatically (config + hook consent + serve patch + restart)
HERMES_PY="$HOME/.hermes/hermes-agent/venv/bin/python"
[ -x "$HERMES_PY" ] || HERMES_PY="python3"
if [ -d "$HOME/.hermes" ]; then
  step "Configuring Hermes Desktop automatically"
  chmod +x hooks/post_llm_call.py 2>/dev/null || true
  "$HERMES_PY" scripts/configure_hermes.py
else
  step "Hermes Desktop not found (~/.hermes missing) — skipping its wiring"
fi

# 5b. Wire Claude Code, if installed (hooks + MCP; works on login, no API key)
if command -v claude >/dev/null 2>&1; then
  step "Configuring Claude Code automatically"
  python3 scripts/configure_claude.py
else
  step "Claude Code not found — skipping (install it and re-run ./setup.sh to wire it)"
fi

# 6. Verify
step "Verifying"
if command -v hermes >/dev/null 2>&1; then
  hermes hooks doctor 2>&1 | tail -6 || true
fi
curl -fsS http://localhost:8800/health | "$HERMES_PY" -m json.tool 2>/dev/null || true

echo
echo "${BOLD}Done — chat in Hermes Desktop, then verify with:${RESET}"
echo "  curl http://localhost:8800/health   # last_written_at must update after every chat turn"
echo "  http://localhost:6333/dashboard     # browse the data visually in Qdrant"
