from io import BytesIO

from PIL import Image

from app.pipeline import run_job
from app.schemas import ImagePlan, ImageSize, Job, JobRequest, JobStatus
from app.services.postprocess import to_web_image
from app.services.workflows import WorkflowRegistry
from tests.conftest import make_png


class FakePlanner:
    async def plan_images(self, title, content, num_images):
        return [ImagePlan(prompt=f"prompt {i}", alt_text=f"alt {i}") for i in range(num_images)]


class FakeEngine:
    def __init__(self):
        self.graphs = []
        self.freed = 0

    async def free_memory(self):
        self.freed += 1

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
