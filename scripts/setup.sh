#!/usr/bin/env bash
# One-time setup on the GPU server: build images, download models, start everything.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { cp .env.example .env; echo "Created .env from .env.example"; }
set -a; source .env; set +a

MODELS_DIR="${MODELS_DIR:-./models}"
DATA_DIR="${DATA_DIR:-./data}"
mkdir -p "$MODELS_DIR" "$DATA_DIR"
MODELS_DIR_ABS="$(cd "$MODELS_DIR" && pwd)"

# --- Disk space check -------------------------------------------------------
free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'; }
check_space() {  # path, needed GB, label
  local avail; avail=$(free_gb "$1")
  if [ "$avail" -lt "$2" ]; then
    echo "!! Not enough space for $3: $1 has ${avail} GB free, needs ~$2 GB."
    echo "   See README > Troubleshooting > Disk space."
    exit 1
  fi
  echo "   $3: $1 has ${avail} GB free (needs ~$2 GB) - ok"
}
echo "==> Checking disk space (SKIP_DISK_CHECK=1 to skip)"
if [ "${SKIP_DISK_CHECK:-0}" != "1" ]; then
  DOCKER_ROOT="$(docker info -f '{{.DockerRootDir}}')"
  if docker image inspect article-image-gen-comfyui >/dev/null 2>&1; then
    check_space "$DOCKER_ROOT" 8 "Docker image update"   # rebuild: only changed layers
  else
    check_space "$DOCKER_ROOT" 30 "Docker images + LLM"   # first build
  fi
  if [ ! -f "$MODELS_DIR_ABS/checkpoints/flux1-dev-fp8.safetensors" ]; then
    check_space "$MODELS_DIR_ABS" 18 "FLUX weights"
  fi
fi

# --- Build, download, start -------------------------------------------------
echo "==> Building images"
docker compose build

echo "==> Downloading FLUX weights into $MODELS_DIR_ABS (~17 GB, resumable)"
docker compose run --rm --no-deps -v "$MODELS_DIR_ABS:/app/models" \
  -e HF_TOKEN="${HF_TOKEN:-}" -e HF_HUB_DISABLE_XET=1 \n  api python scripts/download_models.py --models-dir /app/models

echo "==> Pulling LLM ${LLM_MODEL}"
docker compose up -d ollama
until docker compose exec -T ollama ollama list >/dev/null 2>&1; do sleep 2; done
docker compose exec -T ollama ollama pull "${LLM_MODEL}"

echo "==> Starting all services"
docker compose up -d
echo "Done. API: http://localhost:${API_PORT:-8000}/docs  UI: http://localhost:${UI_PORT:-7860}"
