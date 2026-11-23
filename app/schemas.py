from datetime import datetime, timezone
from enum import Enum

from typing import Annotated

from pydantic import BaseModel, Field


class ImageSize(str, Enum):
    featured = "featured"  # Open Graph / featured image
    inline = "inline"  # in-article image, 3:2
    square = "square"  # social media


# size preset -> (final output size, generation size)
# FLUX works best around 1 megapixel with sides that are multiples of 16,
# so we generate close to the target aspect ratio and then crop/resize.
SIZE_PRESETS: dict[ImageSize, tuple[tuple[int, int], tuple[int, int]]] = {
    ImageSize.featured: ((1200, 630), (1376, 720)),
    ImageSize.inline: ((1200, 800), (1248, 832)),
    ImageSize.square: ((1080, 1080), (1088, 1088)),
}


class JobRequest(BaseModel):
    title: str = Field(min_length=3, max_length=300, description="Article title")
    content: str | None = Field(
        default=None, max_length=20_000, description="Article body or summary (optional)"
    )
    num_images: int = Field(default=1, ge=1, le=4)
    size: ImageSize = ImageSize.featured
    seed: int | None = Field(
        default=None, ge=0, le=2**32 - 1, description="Fix for reproducible results"
    )
    references: list[Annotated[str, Field(max_length=14_000_000)]] = Field(
        default_factory=list,
        max_length=4,
        description=(
            "Optional reference images (http(s) URL or data:image/...;base64 URI). "
            "Images are generated in their style/composition with FLUX Redux, then checked "
            "so they are not copies. Image i uses reference i % len(references)."
        ),
    )
    reference_strength: float = Field(
        default=0.5,
        ge=0.05,
        le=1.0,
        description="How closely to follow the reference (1.0 = very close)",
    )


class JobStatus(str, Enum):
    queued = "queued"
    planning = "planning"  # LLM is writing prompts
    generating = "generating"  # FLUX is rendering
    completed = "completed"
    failed = "failed"


class ImagePlan(BaseModel):
    prompt: str
    alt_text: str


class Originality(BaseModel):
    """Similarity of the generated image to the closest reference image."""

    max_clip_similarity: float  # cosine, 1.0 = same content/composition
    min_phash_distance: int  # Hamming distance of perceptual hashes, 0 = near copy
    passed: bool
    attempts: int  # generations needed (retries use a new seed and weaker reference)


class ImageScores(BaseModel):
    clip_score: float | None = None  # image-prompt alignment, CLIPScore (x100)
    originality: Originality | None = None


class GeneratedImage(BaseModel):
    url: str
    filename: str
    width: int
    height: int
    seed: int
    prompt: str
    alt_text: str
    reference_index: int | None = None
    reference_strength: float | None = None
    scores: ImageScores | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Job(BaseModel):
    id: str
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    message: str | None = None
    request: JobRequest
    reference_files: list[str] = Field(default_factory=list)
    images: list[GeneratedImage] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str
