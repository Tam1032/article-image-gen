"""Article -> prompts (LLM) -> images (ComfyUI/FLUX) -> checks -> web-ready files."""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from slugify import slugify

from app.schemas import (
    SIZE_PRESETS,
    GeneratedImage,
    ImagePlan,
    ImageScores,
    Job,
    JobStatus,
    Originality,
)
from app.services.comfy import ProgressCallback
from app.services.postprocess import to_web_image
from app.services.workflows import ModelRef, Workflow

logger = logging.getLogger(__name__)

Reporter = Callable[[JobStatus, float, str], Awaitable[None]]


class PipelineError(RuntimeError):
    pass


class PromptPlanner(Protocol):
    async def plan_images(
        self, title: str, content: str | None, num_images: int
    ) -> list[ImagePlan]: ...


class ImageEngine(Protocol):
    async def free_memory(self) -> None: ...

    async def upload_image(self, data: bytes, filename: str) -> str: ...

    async def missing_models(self, refs: list[ModelRef]) -> list[str]: ...

    async def run(self, graph: dict, on_progress: ProgressCallback | None = None) -> list[bytes]: ...


class ImageScorer(Protocol):
    def clip_score(self, image: bytes, text: str) -> float: ...

    def image_similarity(self, a: bytes, b: bytes) -> float: ...

    def phash_distance(self, a: bytes, b: bytes) -> int: ...


@dataclass(frozen=True)
class OriginalityPolicy:
    max_clip_similarity: float = 0.92
    min_phash_distance: int = 12
    max_retries: int = 2
    retry_strength_factor: float = 0.6


def check_originality(
    scorer: ImageScorer, image: bytes, references: list[bytes], policy: OriginalityPolicy, attempts: int
) -> Originality:
    sims = [scorer.image_similarity(image, ref) for ref in references]
    dists = [scorer.phash_distance(image, ref) for ref in references]
    max_sim, min_dist = max(sims), min(dists)
    passed = max_sim <= policy.max_clip_similarity and min_dist >= policy.min_phash_distance
    return Originality(
        max_clip_similarity=max_sim, min_phash_distance=min_dist, passed=passed, attempts=attempts
    )


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
    references: list[bytes] | None = None,
    reference_workflow: Workflow | None = None,
    scorer: ImageScorer | None = None,
    policy: OriginalityPolicy = OriginalityPolicy(),
) -> list[GeneratedImage]:
    req = job.request
    n = req.num_images
    references = references or []
    (out_w, out_h), (gen_w, gen_h) = SIZE_PRESETS[req.size]

    if references:
        if reference_workflow is None:
            raise PipelineError("Reference images were given but no reference workflow is configured")
        workflow = reference_workflow

    # Never report progress going backwards (retries restart an image).
    last_progress = 0.0

    async def progress(status: JobStatus, value: float, message: str) -> None:
        nonlocal last_progress
        last_progress = max(last_progress, min(value, 1.0))
        await report(status, last_progress, message)

    # 0. Fail early with a clear message if ComfyUI lacks a model file.
    missing = await engine.missing_models(workflow.required_models())
    if missing:
        raise PipelineError(
            f"ComfyUI is missing model files {missing} needed by workflow '{workflow.name}'. "
            "Run scripts/download_models.py (FLUX Redux needs HF_TOKEN, see README)."
        )

    # 1. Plan: the LLM writes one prompt + alt text per image.
    await progress(JobStatus.planning, 0.02, "Writing image prompts")
    if free_vram_between_stages:
        await engine.free_memory()  # let the LLM use the GPU
    plans = await planner.plan_images(req.title, req.content, n)

    uploaded = [
        await engine.upload_image(data, f"{job.id}-ref-{k + 1}.png")
        for k, data in enumerate(references)
    ]

    # 2. Generate each image.
    base_seed = req.seed if req.seed is not None else random.randrange(2**32)
    slug = slugify(req.title, max_length=60) or "image"
    job_dir = output_dir / job.id
    job_dir.mkdir(parents=True, exist_ok=True)

    images: list[GeneratedImage] = []
    for i, plan in enumerate(plans):
        start, span = 0.1 + 0.9 * i / n, 0.9 / n
        ref_index = i % len(references) if references else None
        strength = req.reference_strength if references else None
        attempts = 0

        while True:
            attempts += 1
            # Retries move to a different seed region and weaken the reference.
            seed = (base_seed + i + (attempts - 1) * 1_000_003) % 2**32
            message = f"Generating image {i + 1}/{n}" + (
                f" (retry {attempts - 1}, result was too close to the reference)"
                if attempts > 1
                else ""
            )
            await progress(JobStatus.generating, start, message)

            async def on_progress(value: int, maximum: int, start=start, message=message) -> None:
                await progress(JobStatus.generating, start + span * value / max(maximum, 1), message)

            values = {"prompt": plan.prompt, "seed": seed, "width": gen_w, "height": gen_h}
            if references:
                values |= {"reference_image": uploaded[ref_index], "reference_strength": strength}
            outputs = await engine.run(workflow.build(values), on_progress=on_progress)

            # 3. Post-process into a web-ready file.
            web = await asyncio.to_thread(to_web_image, outputs[0], (out_w, out_h), webp_quality)

            # 4. Originality check against the references; retry if it looks copied.
            originality = None
            if scorer and references:
                originality = await asyncio.to_thread(
                    check_originality, scorer, web, references, policy, attempts
                )
                if not originality.passed and attempts <= policy.max_retries:
                    logger.info(
                        "Job %s image %d too similar (clip %.3f, phash %d), retrying",
                        job.id, i + 1, originality.max_clip_similarity,
                        originality.min_phash_distance,
                    )
                    strength = round(strength * policy.retry_strength_factor, 3)
                    continue
            break

        clip_score = (
            await asyncio.to_thread(scorer.clip_score, web, plan.prompt) if scorer else None
        )

        filename = f"{slug}-{i + 1}.webp"
        (job_dir / filename).write_bytes(web)
        images.append(
            GeneratedImage(
                url=f"{public_base_url.rstrip('/')}/files/{job.id}/{filename}",
                filename=filename,
                width=out_w,
                height=out_h,
                seed=seed,
                prompt=plan.prompt,
                alt_text=plan.alt_text,
                reference_index=ref_index,
                reference_strength=strength,
                scores=ImageScores(clip_score=clip_score, originality=originality)
                if scorer
                else None,
            )
        )
        logger.info("Job %s: saved %s", job.id, filename)

    return images
