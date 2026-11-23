"""Gradio demo UI. A thin client of the HTTP API: ``python -m app.ui``.

Open http://localhost:7860 (through an SSH tunnel on a remote server).
"""

import base64
import os
import time
from io import BytesIO

import gradio as gr
import httpx
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
POLL_S = 1.0


def _data_uri(path: str) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((1024, 1024))
        buf = BytesIO()
        img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _caption(i: int, image: dict) -> str:
    parts = [f"#{i + 1} · seed {image['seed']}", image["alt_text"]]
    scores = image.get("scores") or {}
    if scores.get("clip_score") is not None:
        parts.append(f"CLIP score {scores['clip_score']:.1f}")
    orig = scores.get("originality")
    if orig:
        verdict = "original" if orig["passed"] else "TOO SIMILAR"
        parts.append(
            f"vs reference: copy sim {orig['max_copy_similarity']:.2f}, "
            f"pHash dist {orig['min_phash_distance']} → {verdict} "
            f"({orig['attempts']} attempt{'s' if orig['attempts'] > 1 else ''})"
        )
    return " | ".join(parts)


def generate(title, content, num_images, size, seed, files, strength):
    payload = {
        "title": title,
        "content": content or None,
        "num_images": int(num_images),
        "size": size,
        "seed": int(seed) if seed not in (None, "") and int(seed) >= 0 else None,
        "references": [_data_uri(f) for f in (files or [])],
        "reference_strength": strength,
    }
    with httpx.Client(base_url=API_URL, timeout=60) as http:
        resp = http.post("/v1/jobs", json=payload)
        if resp.status_code != 202:
            raise gr.Error(f"API error {resp.status_code}: {resp.text[:500]}")
        job_id = resp.json()["job_id"]

        started = time.monotonic()
        while True:
            job = http.get(f"/v1/jobs/{job_id}").json()
            elapsed = time.monotonic() - started
            status = (
                f"**{job['status']}** · {job['progress'] * 100:.0f}% · {elapsed:.0f}s"
                + (f" · {job['message']}" if job.get("message") else "")
            )
            if job["status"] in ("completed", "failed"):
                break
            yield status, [], None
            time.sleep(POLL_S)

        if job["status"] == "failed":
            raise gr.Error(job.get("error") or "Job failed")

        gallery = []
        for i, image in enumerate(job["images"]):
            # Fetch through the API (works without exposing port 8000 to the browser).
            data = http.get(f"/files/{job_id}/{image['filename']}").content
            gallery.append((Image.open(BytesIO(data)), _caption(i, image)))
        yield f"**completed** in {elapsed:.0f}s · job `{job_id}`", gallery, job


with gr.Blocks(title="Article Image Generator") as demo:
    gr.Markdown(
        "# Article Image Generator\n"
        "Article title → LLM image prompts → FLUX.1-dev (ComfyUI) → web-ready WebP + alt text. "
        "Add reference images to generate new, original images in a similar style (FLUX Redux)."
    )
    with gr.Row():
        with gr.Column(scale=1):
            title = gr.Textbox(label="Article title", lines=2)
            content = gr.Textbox(label="Article content / summary (optional)", lines=6)
            with gr.Row():
                num_images = gr.Slider(1, 4, value=2, step=1, label="Images")
                size = gr.Dropdown(["featured", "inline", "square"], value="featured", label="Size")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            with gr.Accordion("Reference images (optional)", open=False):
                files = gr.File(file_count="multiple", file_types=["image"], label="Up to 4 images")
                strength = gr.Slider(
                    0.05, 1.0, value=0.3, step=0.05, label="Reference strength (1.0 = very close)"
                )
            run = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            status = gr.Markdown()
            gallery = gr.Gallery(label="Results", columns=2, height="auto")
            with gr.Accordion("Job JSON", open=False):
                raw = gr.JSON()

    run.click(
        generate,
        inputs=[title, content, num_images, size, seed, files, strength],
        outputs=[status, gallery, raw],
    )
    gr.Examples(
        [
            ["Metro số 1 TP.HCM chính thức vận hành thương mại", ""],
            ["5 cách tiết kiệm điện cho gia đình trong mùa nắng nóng", ""],
            ["How small cafés use AI to plan their weekly menu", ""],
        ],
        inputs=[title, content],
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("UI_PORT", 7860)))
