"""Turns an article into image prompts using a local LLM served by Ollama."""

import json
import logging

import httpx
from pydantic import ValidationError

from app.schemas import ImagePlan

logger = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 6000

SYSTEM_PROMPT = """You are an art director who creates illustration briefs for online articles.
Given an article title (and optionally its content), write image prompts for the FLUX text-to-image model.

Rules for each prompt:
- Written in English, 40-80 words, concrete and visual: main subject, setting, composition, lighting, camera or art style.
- Every prompt must show a DIFFERENT visual angle of the topic (different subject, scene or composition).
- Default to photorealistic editorial photography unless the topic is abstract, then use a clean illustration style.
- Never include text, letters, logos, watermarks, brand names, or the name or likeness of a real person.

For each prompt also write alt_text: one sentence describing the image for accessibility and SEO,
in the SAME LANGUAGE as the article title.

Answer with JSON only."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "alt_text": {"type": "string"},
                },
                "required": ["prompt", "alt_text"],
            },
        }
    },
    "required": ["images"],
}


class LLMError(RuntimeError):
    pass


def build_user_message(title: str, content: str | None, num_images: int) -> str:
    parts = [f"Article title: {title.strip()}"]
    if content and content.strip():
        parts.append(f"Article content:\n{content.strip()[:MAX_CONTENT_CHARS]}")
    parts.append(f"Number of image prompts to write: {num_images}")
    return "\n\n".join(parts)


def parse_plans(raw: str, num_images: int) -> list[ImagePlan]:
    """Parse the LLM JSON answer into exactly ``num_images`` plans."""
    try:
        items = json.loads(raw)["images"]
        plans = [ImagePlan.model_validate(item) for item in items]
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
        raise LLMError(f"Invalid LLM response: {raw[:300]!r}") from exc

    plans = [p for p in plans if p.prompt.strip()]
    if not plans:
        raise LLMError("LLM returned no usable prompts")
    # Too few prompts: reuse them (each image still gets a different seed).
    while len(plans) < num_images:
        plans.append(plans[len(plans) % len(plans)])
    return plans[:num_images]


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        think: bool = False,
        temperature: float = 0.7,
        timeout_s: float = 300,
        unload_after_use: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.think = think
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.unload_after_use = unload_after_use

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=5) as http:
                return (await http.get("/api/tags")).status_code == 200
        except httpx.HTTPError:
            return False

    async def plan_images(
        self, title: str, content: str | None, num_images: int, retries: int = 1
    ) -> list[ImagePlan]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(title, content, num_images)},
            ],
            "format": RESPONSE_SCHEMA,  # structured output
            "stream": False,
            "think": self.think,
            "options": {"temperature": self.temperature},
            # keep_alive=0 unloads the model right away so FLUX gets the VRAM.
            "keep_alive": 0 if self.unload_after_use else "10m",
        }
        last_error: Exception | None = None
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s) as http:
            for attempt in range(retries + 1):
                resp = await http.post("/api/chat", json=payload)
                if resp.status_code != 200:
                    raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:500]}")
                try:
                    return parse_plans(resp.json()["message"]["content"], num_images)
                except LLMError as exc:
                    last_error = exc
                    logger.warning("Attempt %d: %s", attempt + 1, exc)
        raise LLMError(str(last_error))
