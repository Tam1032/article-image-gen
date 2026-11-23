"""Image quality checks that run in the worker, on CPU.

- **Prompt alignment** (CLIP score): cosine similarity between the image and its
  prompt, x100, the usual CLIPScore convention. Higher = closer to the prompt.
- **Originality** against reference images:
  - CLIP image-image cosine similarity (semantic/compositional closeness),
  - perceptual-hash Hamming distance (0 = pixel-level near copy, 64 = unrelated).

Requires the optional ``scoring`` dependency group (torch CPU, transformers,
imagehash). ``load_scorer`` returns None when it is not installed.
"""

from __future__ import annotations

import logging
from io import BytesIO

from PIL import Image

logger = logging.getLogger(__name__)


def _open(data: bytes) -> Image.Image:
    with Image.open(BytesIO(data)) as img:
        return img.convert("RGB")


class ClipScorer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    # Projected embeddings computed explicitly (stable across transformers versions).
    def _image_embedding(self, data: bytes):
        inputs = self.processor(images=_open(data), return_tensors="pt")
        with self._torch.inference_mode():
            pooled = self.model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
            emb = self.model.visual_projection(pooled)
        return emb / emb.norm(dim=-1, keepdim=True)

    def _text_embedding(self, text: str):
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        with self._torch.inference_mode():
            pooled = self.model.text_model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).pooler_output
            emb = self.model.text_projection(pooled)
        return emb / emb.norm(dim=-1, keepdim=True)

    def clip_score(self, image: bytes, text: str) -> float:
        cos = (self._image_embedding(image) @ self._text_embedding(text).T).item()
        return round(max(cos, 0.0) * 100, 2)

    def image_similarity(self, a: bytes, b: bytes) -> float:
        cos = (self._image_embedding(a) @ self._image_embedding(b).T).item()
        return round(cos, 4)

    @staticmethod
    def phash_distance(a: bytes, b: bytes) -> int:
        import imagehash

        return int(imagehash.phash(_open(a)) - imagehash.phash(_open(b)))


def load_scorer(model_name: str) -> ClipScorer | None:
    try:
        scorer = ClipScorer(model_name)
    except ImportError:
        logger.warning("Scoring disabled: install the 'scoring' dependency group")
        return None
    except Exception:  # e.g. no network to download the model
        logger.exception("Scoring disabled: could not load %s", model_name)
        return None
    logger.info("Loaded scorer %s", model_name)
    return scorer
