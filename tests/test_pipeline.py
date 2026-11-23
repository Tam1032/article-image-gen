from io import BytesIO

import pytest
from PIL import Image

from app.pipeline import OriginalityPolicy, PipelineError, run_job
from app.schemas import ImagePlan, ImageSize, Job, JobRequest, JobStatus
from app.services.postprocess import to_web_image
from app.services.workflows import WorkflowRegistry
from tests.conftest import make_png


class FakePlanner:
    async def plan_images(self, title, content, num_images):
        return [ImagePlan(prompt=f"prompt {i}", alt_text=f"alt {i}") for i in range(num_images)]


class FakeEngine:
    def __init__(self, missing=()):
        self.graphs = []
        self.freed = 0
        self.uploads = []
        self.missing = list(missing)

    async def free_memory(self):
        self.freed += 1

    async def upload_image(self, data, filename):
        self.uploads.append(filename)
        return filename

    async def missing_models(self, refs):
        return [r.filename for r in refs if r.filename in self.missing]

    async def run(self, graph, on_progress=None):
        self.graphs.append(graph)
        if on_progress:
            await on_progress(10, 20)
        latent = graph["27"]["inputs"]
        return [make_png(latent["width"], latent["height"])]


def test_to_web_image_crops_and_resizes():
    out = to_web_image(make_png(1376, 720), (1200, 630))
    img = Image.open(BytesIO(out))
    assert img.format == "WEBP"
    assert img.size == (1200, 630)


async def test_run_job_end_to_end(tmp_path, workflows_dir):
    job = Job(
        id="job1",
        request=JobRequest(
            title="Metro số 1 TP.HCM vận hành", num_images=2, size=ImageSize.featured, seed=7
        ),
    )
    engine = FakeEngine()
    reports = []

    async def report(status, progress, message):
        reports.append((status, progress))

    images = await run_job(
        job,
        planner=FakePlanner(),
        engine=engine,
        workflow=WorkflowRegistry(workflows_dir).get("flux_txt2img"),
        output_dir=tmp_path,
        public_base_url="http://example.test/",
        report=report,
    )

    assert [img.seed for img in images] == [7, 8]
    assert images[0].filename == "metro-so-1-tp-hcm-van-hanh-1.webp"
    assert images[0].url == "http://example.test/files/job1/metro-so-1-tp-hcm-van-hanh-1.webp"
    assert (tmp_path / "job1" / images[1].filename).exists()
    assert engine.graphs[0]["6"]["inputs"]["text"] == "prompt 0"
    assert engine.graphs[0]["27"]["inputs"]["width"] == 1376
    assert engine.freed == 1
    assert reports[0][0] == JobStatus.planning
    progress = [p for _, p in reports]
    assert progress == sorted(progress) and progress[-1] <= 1.0


class FakeScorer:
    """First `copies` checks look like a copy of the reference, later ones do not."""

    def __init__(self, copies=0):
        self.copies = copies
        self.checks = 0

    def image_similarity(self, a, b):
        self.checks += 1
        return 0.99 if self.checks <= self.copies else 0.6

    def phash_distance(self, a, b):
        return 2 if self.checks <= self.copies else 30

    def clip_score(self, image, text):
        return 31.5


def make_job(**kwargs):
    defaults = dict(title="Metro số 1 TP.HCM vận hành", num_images=1, seed=7)
    return Job(id="job1", request=JobRequest(**{**defaults, **kwargs}))


async def run(job, engine, workflows_dir, tmp_path, **kwargs):
    registry = WorkflowRegistry(workflows_dir)
    reports = []

    async def collect(status, progress, message):
        reports.append(progress)

    images = await run_job(
        job,
        planner=FakePlanner(),
        engine=engine,
        workflow=registry.get("flux_txt2img"),
        reference_workflow=registry.get("flux_redux"),
        output_dir=tmp_path,
        public_base_url="http://example.test",
        report=collect,
        **kwargs,
    )
    return images, reports


async def test_references_use_redux_workflow(tmp_path, workflows_dir):
    engine = FakeEngine()
    refs = [make_png(64, 64, "red"), make_png(64, 64, "blue")]
    images, _ = await run(
        make_job(num_images=3, reference_strength=0.4), engine, workflows_dir, tmp_path,
        references=refs, scorer=FakeScorer(),
    )

    assert engine.uploads == ["job1-ref-1.png", "job1-ref-2.png"]
    # image i uses reference i % len(references)
    assert [g["40"]["inputs"]["image"] for g in engine.graphs] == [
        "job1-ref-1.png", "job1-ref-2.png", "job1-ref-1.png",
    ]
    assert engine.graphs[0]["44"]["inputs"]["strength"] == 0.4
    assert [img.reference_index for img in images] == [0, 1, 0]
    assert images[0].scores.clip_score == 31.5
    assert images[0].scores.originality.passed
    assert images[0].scores.originality.attempts == 1


async def test_too_similar_result_is_regenerated(tmp_path, workflows_dir):
    engine = FakeEngine()
    images, reports = await run(
        make_job(reference_strength=0.5), engine, workflows_dir, tmp_path,
        references=[make_png(64, 64)], scorer=FakeScorer(copies=1),
    )

    assert len(engine.graphs) == 2  # first result looked copied -> one retry
    first, second = engine.graphs
    assert first["31"]["inputs"]["seed"] != second["31"]["inputs"]["seed"]
    assert second["44"]["inputs"]["strength"] == pytest.approx(0.3)  # 0.5 * 0.6
    orig = images[0].scores.originality
    assert orig.passed and orig.attempts == 2
    assert images[0].reference_strength == pytest.approx(0.3)
    assert images[0].seed == second["31"]["inputs"]["seed"]
    assert reports == sorted(reports)  # progress never goes backwards on retry


async def test_retries_are_bounded(tmp_path, workflows_dir):
    engine = FakeEngine()
    images, _ = await run(
        make_job(), engine, workflows_dir, tmp_path,
        references=[make_png(64, 64)], scorer=FakeScorer(copies=99),
        policy=OriginalityPolicy(max_retries=2),
    )
    assert len(engine.graphs) == 3
    orig = images[0].scores.originality
    assert not orig.passed and orig.attempts == 3  # kept, but flagged


async def test_missing_model_fails_before_generating(tmp_path, workflows_dir):
    engine = FakeEngine(missing=["flux1-redux-dev.safetensors"])
    with pytest.raises(PipelineError, match="flux1-redux-dev"):
        await run(make_job(), engine, workflows_dir, tmp_path, references=[make_png(64, 64)])
    assert engine.graphs == [] and engine.freed == 0


async def test_no_scorer_means_no_scores(tmp_path, workflows_dir):
    images, _ = await run(make_job(), FakeEngine(), workflows_dir, tmp_path)
    assert images[0].scores is None and images[0].reference_index is None
