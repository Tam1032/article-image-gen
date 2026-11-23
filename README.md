# Article Image Generator

Generate **original, web-ready illustrations for articles**: send an article title (and
optionally its content) and get back unique images, sized and compressed for the web, with
alt text.

Built on **FLUX.1-dev** running in **ComfyUI**, with a **local LLM (Qwen via Ollama)** that
turns the article into image prompts. Everything runs on a single GPU (tested target: RTX 4090, 24 GB).

## The problem

Content teams need an illustration for every article. Reusing photos from other sites risks
copyright claims and duplicate-content penalties; commissioning or shooting photos is slow and
expensive. This service produces a fresh, on-topic image in seconds.

| Input | Output |
|---|---|
| Article title (required) | 1–4 WebP images (`featured` 1200×630, `inline` 1200×800, `square` 1080×1080) |
| Article content / summary (optional) | Alt text for each image, in the article's language |
| Number of images, size preset, seed (optional) | The prompt and seed used, for reproducibility |

## How it works

```
POST /v1/jobs ──► FastAPI ──► Redis queue ──► worker
                                                │
                     1. plan     Ollama (Qwen):     article ─► N image prompts + alt text
                     2. render   ComfyUI (FLUX.1-dev fp8): prompt ─► image   (progress via WebSocket)
                     3. publish  crop/resize ─► WebP ─► /files/<job>/<slug>-N.webp
                                                │
GET /v1/jobs/{id} ◄── status, progress, image URLs + alt text
```

Design notes:

- **Workflow templates, not hard-coded node ids.** Each ComfyUI workflow is an API-format JSON
  export plus a small YAML manifest mapping friendly parameters (`prompt`, `seed`, `steps`…) to
  node inputs (`workflows/flux_txt2img.yaml`). A workflow can be re-exported from the ComfyUI
  editor without touching Python code.
- **One GPU, two models.** By default the stages time-share the GPU: ComfyUI is asked to unload
  (`POST /free`) before the LLM runs, and Ollama unloads right after answering (`keep_alive: 0`).
  This is required for larger LLMs (Qwen 3.6 27B ~17 GB + FLUX fp8 do not fit in 24 GB together);
  with the default Qwen 3.5 9B you can set `FREE_VRAM_BETWEEN_STAGES=false` to keep both loaded.
  The worker runs one job at a time.
- **Structured LLM output.** The LLM answers against a JSON schema (Ollama structured outputs),
  with validation and a retry.
- **Reproducible environment.** ComfyUI is pinned to a release tag; model weights are declared in
  `models.yaml` and downloaded by a script; everything starts with `docker compose`.

## Quick start (GPU server)

Requirements: NVIDIA GPU with ≥ 24 GB VRAM, NVIDIA driver ≥ 570 (or see below),
Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html),
disk space: ~30 GB where Docker stores images (`docker info -f '{{.DockerRootDir}}'`) and
~18 GB for model weights (`MODELS_DIR`).

```bash
git clone https://github.com/Tam1032/article-image-gen.git
cd article-image-gen
./scripts/setup.sh        # checks disk space, builds images, downloads FLUX (~17 GB) + Qwen (~7 GB), starts services
```

Then open `http://<server>:8000/docs`, or:

```bash
curl -X POST http://localhost:8000/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"title": "Metro số 1 TP.HCM chính thức vận hành thương mại", "num_images": 2}'
# -> {"job_id": "…", "status": "queued", "status_url": "…/v1/jobs/…"}

curl http://localhost:8000/v1/jobs/<job_id>
```

Settings live in `.env` (see `.env.example`). Older driver? Build ComfyUI with
`--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126`.
To watch or edit workflows in the ComfyUI editor, open an SSH tunnel:
`ssh -L 8188:localhost:8188 <server>` and browse `http://localhost:8188`.

### Troubleshooting: disk space

`no space left on device` during the build means the disk holding Docker's data is full.

- Model weights and outputs can live on any disk: set `MODELS_DIR` and `DATA_DIR` in `.env`
  (e.g. `MODELS_DIR=/mnt/bigdisk/article-image-gen/models`).
- Docker images and the Ollama model live in Docker's data root. To move it to another disk,
  stop Docker, copy `/var/lib/docker` (and `/var/lib/containerd` if used) there, set
  `"data-root": "/mnt/bigdisk/docker"` in `/etc/docker/daemon.json` (keep existing keys such as
  the `nvidia` runtime), and start Docker again.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/jobs` | Create a generation job → `202` with `job_id` |
| `GET` | `/v1/jobs/{job_id}` | Status (`queued`, `planning`, `generating`, `completed`, `failed`), progress 0–1, images |
| `GET` | `/files/{job_id}/{file}` | Generated image |
| `GET` | `/health` | Redis / ComfyUI / Ollama reachability |

## Development

```bash
uv sync
uv run pytest            # tests use fake LLM/ComfyUI clients, no GPU needed
```

Project layout:

```
app/
  main.py              FastAPI app
  worker.py            arq worker (one job at a time)
  pipeline.py          plan -> render -> publish
  jobs.py              job state in Redis
  services/
    llm.py             Ollama client + prompt writing
    comfy.py           ComfyUI HTTP/WebSocket client
    workflows.py       workflow templates
    postprocess.py     crop/resize/WebP
workflows/             ComfyUI API-format graphs + YAML manifests
docker/                Dockerfiles (API/worker, ComfyUI)
scripts/               setup, model download, benchmark
```

## Roadmap

- [x] Article → prompts → FLUX images → web-ready files, async job API, Docker
- [ ] Reference images → new original image (FLUX Redux)
- [ ] Originality check: CLIP / perceptual-hash similarity against reference images, auto-retry
- [ ] Relevance score (CLIP image–text) per image
- [ ] Gradio demo UI
- [ ] Benchmarks on RTX 4090 (latency, VRAM, scores)

## License notes

FLUX.1-dev weights are under the FLUX.1 [dev] Non-Commercial License. For commercial use,
switch to FLUX.1-schnell (Apache-2.0, listed as optional in `models.yaml`).
