"""Article -> prompts (LLM) -> images (ComfyUI/FLUX) -> web-ready files."""

import logging
import random
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

from slugify import slugify

from app.schemas import SIZE_PRESETS, GeneratedImage, ImagePlan, Job, JobStatus
from app.services.comfy import ProgressCallback
from app.services.postprocess import to_web_image
from app.services.workflows import Workflow

logger = logging.getLogger(__name__)

Reporter = Callable[[JobStatus, float, str], Awaitable[None]]


class PromptPlanner(Protocol):
    async def plan_images(
        self, title: str, content: str | None, num_images: int
    ) -> list[ImagePlan]: ...


class ImageEngine(Protocol):
    async def free_memory(self) -> None: ...

    async def run(self, graph: dict, on_progress: ProgressCallback | None = None) -> list[bytes]: ...


async def run_job(
    job: Job,
    *,
    planner: PromptPlanner,
    engine: ImageEngine,
    workflow: Workflow,
    output_dir: Path,
    public_base_url: str,
    report: Reporter,
    webp_quality: int = 85,
    free_vram_between_stages: bool = True,
) -> list[GeneratedImage]:
    req = job.request
    n = req.num_images
    (out_w, out_h), (gen_w, gen_h) = SIZE_PRESETS[req.size]

    # 1. Plan: the LLM writes one prompt + alt text per image.
    await report(JobStatus.planning, 0.02, "Writing image prompts")
    if free_vram_between_stages:
        await engine.free_memory()  # let the LLM use the GPU
    plans = await planner.plan_images(req.title, req.content, n)

    # 2. Generate each image.
    base_seed = req.seed if req.seed is not None else random.randrange(2**32)
    slug = slugify(req.title, max_length=60) or "image"
    job_dir = output_dir / job.id
    job_dir.mkdir(parents=True, exist_ok=True)

    images: list[GeneratedImage] = []
    for i, plan in enumerate(plans):
        seed = (base_seed + i) % 2**32
        start, span = 0.1 + 0.9 * i / n, 0.9 / n
        message = f"Generating image {i + 1}/{n}"
        await report(JobStatus.generating, start, message)

        async def on_progress(value: int, maximum: int, start=start, message=message) -> None:
            await report(JobStatus.generating, start + span * value / max(maximum, 1), message)

        graph = workflow.build(
            {"prompt": plan.prompt, "seed": seed, "width": gen_w, "height": gen_h}
        )
        outputs = await engine.run(graph, on_progress=on_progress)

        # 3. Post-process into a web-ready file.
        filename = f"{slug}-{i + 1}.webp"
        (job_dir / filename).write_bytes(to_web_image(outputs[0], (out_w, out_h), webp_quality))
        images.append(
            GeneratedImage(
                url=f"{public_base_url.rstrip('/')}/files/{job.id}/{filename}",
                filename=filename,
                width=out_w,
                height=out_h,
                seed=seed,
                prompt=plan.prompt,
                alt_text=plan.alt_text,
            )
        )
        logger.info("Job %s: saved %s", job.id, filename)

    return images
