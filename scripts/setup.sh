#!/usr/bin/env bash
# One-time setup on the GPU server: build images, download models, start everything.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { cp .env.example .env; echo "Created .env from .env.example"; }
set -a; source .env; set +a

echo "==> Building images"
docker compose build

echo "==> Downloading FLUX weights into ./models (~17 GB, resumable)"
docker compose run --rm --no-deps -v "$PWD/models:/app/models" \
  -e HF_TOKEN="${HF_TOKEN:-}" api python scripts/download_models.py --models-dir /app/models

echo "==> Pulling LLM ${LLM_MODEL}"
docker compose up -d ollama
until docker compose exec -T ollama ollama list >/dev/null 2>&1; do sleep 2; done
docker compose exec -T ollama ollama pull "${LLM_MODEL}"

echo "==> Starting all services"
docker compose up -d
echo "Done. API: http://localhost:${API_PORT:-8000}/docs"
