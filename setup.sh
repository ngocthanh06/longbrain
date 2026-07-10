#!/usr/bin/env bash
# One-shot setup for the Longbrain memory stack — zero manual configuration.
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
# `docker info` can hang for minutes when the daemon is half-up — bound it so
# setup fails fast with a clear message instead of appearing frozen.
run_with_timeout() { # <seconds> <cmd...>
  "${@:2}" & local cmd_pid=$!
  ( sleep "$1"; kill -9 "$cmd_pid" 2>/dev/null ) & local watcher_pid=$!
  local rc=0; wait "$cmd_pid" 2>/dev/null || rc=$?
  kill "$watcher_pid" 2>/dev/null; wait "$watcher_pid" 2>/dev/null || true
  return "$rc"
}
if ! run_with_timeout 15 docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running (or not responding). Open Docker Desktop, then run ./setup.sh again"
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

# 4b. Upgrade check: chunks embedded before the metadata fix rank terribly —
# offer a one-time re-embed when any are found (fresh installs: instant no-op).
step "Checking knowledge-base vectors"
if ! python3 scripts/reembed_documents.py --check; then
  if [ -t 0 ]; then
    echo "The document knowledge base needs a one-time re-embed (a backup is"
    echo "written to ~/.hermes/ first; originals are preserved)."
    read -r -p "Re-embed now? [y/N] " reembed_consent
    case "$reembed_consent" in
      [yY]*) python3 scripts/reembed_documents.py ;;
      *) echo "Skipped — run later: python3 scripts/reembed_documents.py" ;;
    esac
  else
    echo "Non-interactive shell — run later: python3 scripts/reembed_documents.py"
  fi
fi

HERMES_PY="$HOME/.hermes/hermes-agent/venv/bin/python"
[ -x "$HERMES_PY" ] || HERMES_PY="python3"

# 4c. Host background jobs (docs/ auto-ingest + nightly backup) — agent-
# independent, so install them here, NOT only inside the Hermes wiring:
# a Claude-Code-only machine needs the docs/ watcher just as much.
step "Installing background jobs (docs/ auto-ingest + nightly backup)"
"$HERMES_PY" scripts/configure_host_jobs.py || true

# 5. Wire Hermes Desktop automatically (config + hook consent + serve patch + restart)
if [ -d "$HOME/.hermes" ]; then
  step "Configuring Hermes Desktop automatically"
  chmod +x hooks/post_llm_call.py 2>/dev/null || true
  "$HERMES_PY" scripts/configure_hermes.py
else
  step "Hermes Desktop not found (~/.hermes missing) — skipping its wiring"
fi

# 5b. Wire Claude Code, if installed (hooks + MCP; works on login, no API key)
if ! command -v claude >/dev/null 2>&1; then
  step "Claude Code not found"
  if [ -t 0 ] && command -v npm >/dev/null 2>&1; then
    echo "Claude Code is the CLI that logs in with your Claude subscription (no API"
    echo "key needed) and is what Longbrain wires this memory stack into."
    read -r -p "Install it now via 'npm install -g @anthropic-ai/claude-code'? [y/N] " claude_install_consent
    case "$claude_install_consent" in
      [yY]*)
        if npm install -g @anthropic-ai/claude-code; then
          echo "Installed. Run 'claude' once to log in — memory wiring continues below."
        else
          echo "Install failed — install manually: https://docs.claude.com/en/docs/claude-code"
        fi
        ;;
      *) echo "Skipped — install manually later and re-run ./setup.sh to wire it." ;;
    esac
  elif [ -t 0 ]; then
    echo "npm is not on PATH, so it can't be auto-installed here."
    echo "Install Node.js/npm, or install Claude Code manually:"
    echo "https://docs.claude.com/en/docs/claude-code — then re-run ./setup.sh."
  else
    echo "Non-interactive shell — skipping the install prompt."
    echo "Install it and re-run ./setup.sh to wire it."
  fi
fi

if command -v claude >/dev/null 2>&1; then
  step "Configuring Claude Code automatically"
  LONGBRAIN_CONFIGURE_CLAUDE_MD=0
  if [ -t 0 ]; then
    echo "Optional: add a block to ~/.claude/CLAUDE.md (applies to EVERY project)"
    echo "teaching Claude Code to actually USE this memory stack, both directions:"
    echo "  - read:  search shared memory (past conversations of ALL agents, and"
    echo "           project docs in the knowledge base) BEFORE declaring old"
    echo "           context lost or a document unknown"
    echo "  - write: prefer the longbrain MCP (localhost:8800) over Claude's"
    echo "           built-in file-based memory when saving long-term facts"
    read -r -p "Add this instruction? [y/N] " claude_md_consent
    case "$claude_md_consent" in
      [yY]*) LONGBRAIN_CONFIGURE_CLAUDE_MD=1 ;;
      *) echo "Skipped — ~/.claude/CLAUDE.md left untouched." ;;
    esac
  else
    echo "Non-interactive shell — skipping the ~/.claude/CLAUDE.md consent prompt."
    echo "Re-run ./setup.sh in a terminal if you want to enable this option."
  fi
  export LONGBRAIN_CONFIGURE_CLAUDE_MD
  python3 scripts/configure_claude.py
else
  echo "Skipping Claude Code wiring — install it and re-run ./setup.sh."
fi

# 5c. Wire Codex, if installed (MCP tools + turn-ended recording via notify)
if command -v codex >/dev/null 2>&1 || [ -d "$HOME/.codex" ]; then
  step "Configuring Codex (MCP + turn-ended memory sync)"
  python3 scripts/configure_codex.py
else
  step "Codex not found — skipping (any MCP client can connect manually: http://localhost:8800/mcp)"
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
echo "  python3 scripts/doctor.py           # re-check service + every agent's wiring any time"
echo "  http://localhost:6333/dashboard     # browse the data visually in Qdrant"
