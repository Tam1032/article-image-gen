"""Image quality checks that run in the worker, on CPU.

- **Prompt alignment** (CLIP score): cosine similarity between the image and its
  prompt, x100, the usual CLIPScore convention. Higher = closer to the prompt.
- **Originality** against reference images:
  - DINOv2 image-image cosine similarity: the copy detector. Self-supervised
    features keep layout/structure, so they catch a reproduced composition
    that CLIP (trained to match captions) rates as merely "similar topic".
  - CLIP image-image similarity: reported for information only.
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


class Scorer:
    def __init__(
        self,
        clip_model: str = "openai/clip-vit-base-patch32",
        copy_model: str = "facebook/dinov2-small",
    ):
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        self._torch = torch
        self.clip = CLIPModel.from_pretrained(clip_model).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.dino = AutoModel.from_pretrained(copy_model).eval()
        self.dino_processor = AutoImageProcessor.from_pretrained(copy_model)

    # Embeddings computed explicitly (stable across transformers versions).
    def _clip_image(self, data: bytes):
        inputs = self.clip_processor(images=_open(data), return_tensors="pt")
        with self._torch.inference_mode():
            pooled = self.clip.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
            emb = self.clip.visual_projection(pooled)
        return emb / emb.norm(dim=-1, keepdim=True)

    def _clip_text(self, text: str):
        inputs = self.clip_processor(
            text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        with self._torch.inference_mode():
            pooled = self.clip.text_model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).pooler_output
            emb = self.clip.text_projection(pooled)
        return emb / emb.norm(dim=-1, keepdim=True)

    def _dino(self, data: bytes):
        inputs = self.dino_processor(images=_open(data), return_tensors="pt")
        with self._torch.inference_mode():
            emb = self.dino(pixel_values=inputs["pixel_values"]).last_hidden_state[:, 0]
        return emb / emb.norm(dim=-1, keepdim=True)

    def clip_score(self, image: bytes, text: str) -> float:
        cos = (self._clip_image(image) @ self._clip_text(text).T).item()
        return round(max(cos, 0.0) * 100, 2)

    def image_similarity(self, a: bytes, b: bytes) -> float:
        return round((self._clip_image(a) @ self._clip_image(b).T).item(), 4)

    def copy_similarity(self, a: bytes, b: bytes) -> float:
        return round((self._dino(a) @ self._dino(b).T).item(), 4)

    @staticmethod
    def phash_distance(a: bytes, b: bytes) -> int:
        import imagehash

        return int(imagehash.phash(_open(a)) - imagehash.phash(_open(b)))


def load_scorer(clip_model: str, copy_model: str) -> Scorer | None:
    try:
        scorer = Scorer(clip_model, copy_model)
    except ImportError:
        logger.warning("Scoring disabled: install the 'scoring' dependency group")
        return None
    except Exception:  # e.g. no network to download the models
        logger.exception("Scoring disabled: could not load %s / %s", clip_model, copy_model)
        return None
    logger.info("Loaded scorer (%s, %s)", clip_model, copy_model)
    return scorer
