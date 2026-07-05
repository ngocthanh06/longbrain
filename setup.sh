#!/usr/bin/env bash
# One-shot setup for the Hermes memory stack — zero manual configuration.
# Usage: ./setup.sh
set -euo pipefail

cd "$(dirname "$0")"

BOLD=$(tput bold 2>/dev/null || true)
RESET=$(tput sgr0 2>/dev/null || true)
step() { echo; echo "${BOLD}==> $*${RESET}"; }

# 1. Docker available?
step "Kiểm tra Docker"
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker chưa được cài. Cài Docker Desktop trước: https://docs.docker.com/get-docker/"
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon chưa chạy. Mở Docker Desktop rồi chạy lại ./setup.sh"
  exit 1
fi
echo "OK"

# 2. .env
step "Cấu hình .env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Đã tạo .env từ .env.example (mặc định: fastembed local, không cần API key)."
else
  echo ".env đã tồn tại, giữ nguyên."
fi

# 3. Build + up
step "Khởi động containers (lần đầu sẽ build image, mất vài phút)"
docker compose up -d --build

# 4. Wait for health
step "Chờ memory service sẵn sàng"
for i in $(seq 1 60); do
  if curl -fsS http://localhost:8800/health >/dev/null 2>&1; then
    echo "Service OK: http://localhost:8800"
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "Service không lên sau 5 phút. Xem log: docker compose logs llamaindex"
    exit 1
  fi
  sleep 5
done

# 5. Wire Hermes Desktop automatically (config + hook consent + serve patch + restart)
step "Cấu hình Hermes Desktop tự động"
chmod +x hooks/post_llm_call.py 2>/dev/null || true
HERMES_PY="$HOME/.hermes/hermes-agent/venv/bin/python"
[ -x "$HERMES_PY" ] || HERMES_PY="python3"
"$HERMES_PY" scripts/configure_hermes.py

# 6. Verify
step "Kiểm tra"
if command -v hermes >/dev/null 2>&1; then
  hermes hooks doctor 2>&1 | tail -6 || true
fi
curl -fsS http://localhost:8800/health | "$HERMES_PY" -m json.tool 2>/dev/null || true

echo
echo "${BOLD}Hoàn tất — chat trong Hermes Desktop rồi kiểm tra bằng:${RESET}"
echo "  curl http://localhost:8800/health   # last_written_at phải cập nhật sau mỗi lượt chat"
echo "  http://localhost:6333/dashboard     # xem dữ liệu trực quan trong Qdrant"
