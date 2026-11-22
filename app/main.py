from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.jobs import JobStore
from app.schemas import Job, JobCreated, JobRequest
from app.services.comfy import ComfyClient
from app.services.llm import OllamaClient

settings = get_settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    app.state.store = JobStore(app.state.redis, settings.job_ttl_s)
    yield
    await app.state.redis.aclose()


app = FastAPI(
    title="Article Image Generator",
    description="Generate original, web-ready illustrations for articles.",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/files", StaticFiles(directory=settings.output_dir), name="files")


@app.post("/v1/jobs", response_model=JobCreated, status_code=status.HTTP_202_ACCEPTED)
async def create_job(body: JobRequest, request: Request) -> JobCreated:
    job = await request.app.state.store.create(body)
    await request.app.state.redis.enqueue_job("generate_images", job.id, _job_id=job.id)
    return JobCreated(
        job_id=job.id,
        status=job.status,
        status_url=str(request.url_for("get_job", job_id=job.id)),
    )


@app.get("/v1/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str, request: Request) -> Job:
    job = await request.app.state.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/health")
async def health(request: Request) -> dict:
    try:
        redis_ok = bool(await request.app.state.redis.ping())
    except Exception:
        redis_ok = False
    comfy_ok = await ComfyClient(settings.comfyui_url).health()
    ollama_ok = await OllamaClient(settings.ollama_url, settings.llm_model).health()
    return {
        "status": "ok" if redis_ok and comfy_ok and ollama_ok else "degraded",
        "redis": redis_ok,
        "comfyui": comfy_ok,
        "ollama": ollama_ok,
    }
