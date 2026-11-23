"""Background worker (arq). Run with: ``arq app.worker.WorkerSettings``."""

import asyncio
import logging

from arq.connections import RedisSettings

from app.config import get_settings
from app.jobs import JobStore
from app.pipeline import OriginalityPolicy, run_job
from app.schemas import JobStatus
from app.services.comfy import ComfyClient
from app.services.llm import OllamaClient
from app.services.scoring import load_scorer
from app.services.workflows import WorkflowRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def startup(ctx: dict) -> None:
    settings = get_settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    ctx["settings"] = settings
    ctx["store"] = JobStore(ctx["redis"], settings.job_ttl_s)
    ctx["planner"] = OllamaClient(
        settings.ollama_url,
        settings.llm_model,
        think=settings.llm_think,
        temperature=settings.llm_temperature,
        timeout_s=settings.llm_timeout_s,
        unload_after_use=settings.free_vram_between_stages,
    )
    ctx["engine"] = ComfyClient(settings.comfyui_url, timeout_s=settings.generation_timeout_s)
    ctx["workflows"] = WorkflowRegistry(settings.workflows_dir)
    ctx["scorer"] = (
        await asyncio.to_thread(load_scorer, settings.clip_model, settings.copy_detection_model) if settings.enable_scoring else None
    )
    ctx["policy"] = OriginalityPolicy(
        max_copy_similarity=settings.originality_max_copy_similarity,
        min_phash_distance=settings.originality_min_phash_distance,
        max_retries=settings.originality_max_retries,
        retry_strength_factor=settings.retry_strength_factor,
    )
    logger.info(
        "Worker ready. Workflows: %s, scoring: %s",
        ctx["workflows"].names(),
        "on" if ctx["scorer"] else "off",
    )


async def generate_images(ctx: dict, job_id: str) -> None:
    settings = ctx["settings"]
    store: JobStore = ctx["store"]
    job = await store.get(job_id)
    if job is None:
        logger.warning("Job %s not found (expired?)", job_id)
        return

    async def report(status: JobStatus, progress: float, message: str) -> None:
        await store.update(job_id, status=status, progress=round(progress, 3), message=message)

    try:
        ref_dir = settings.reference_dir / job_id
        references = [(ref_dir / name).read_bytes() for name in job.reference_files]
        workflows = ctx["workflows"]
        images = await run_job(
            job,
            planner=ctx["planner"],
            engine=ctx["engine"],
            workflow=workflows.get(settings.default_workflow),
            output_dir=settings.output_dir,
            public_base_url=settings.public_base_url,
            report=report,
            webp_quality=settings.webp_quality,
            free_vram_between_stages=settings.free_vram_between_stages,
            references=references,
            reference_workflow=(
                workflows.get(settings.reference_workflow) if references else None
            ),
            scorer=ctx["scorer"],
            policy=ctx["policy"],
        )
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        await store.update(job_id, status=JobStatus.failed, message=None, error=str(exc))
        return

    await store.update(
        job_id, status=JobStatus.completed, progress=1.0, message=None, images=images
    )


class WorkerSettings:
    functions = [generate_images]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 1  # one GPU -> one generation at a time
    # 4 images x (1 + retries) generations + LLM
    job_timeout = int(
        get_settings().generation_timeout_s * 4 * (1 + get_settings().originality_max_retries)
        + get_settings().llm_timeout_s
    )
