"""Async client for the ComfyUI HTTP + WebSocket API."""

import asyncio
import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import websockets

from app.services.workflows import ModelRef

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], Awaitable[None]]


def _combo_options(spec: Any) -> list[str] | None:
    """Options of a COMBO input in /object_info (old and new formats)."""
    if isinstance(spec, list) and spec:
        if isinstance(spec[0], list):  # [["a.safetensors", ...], {...}]
            return spec[0]
        if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            return spec[1].get("options")
    return None


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, base_url: str, timeout_s: float = 600):
        self.base_url = base_url.rstrip("/")
        self.ws_url = re.sub(r"^http", "ws", self.base_url)  # http->ws, https->wss
        self.timeout_s = timeout_s

    def _http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=60)

    async def health(self) -> bool:
        try:
            async with self._http() as http:
                return (await http.get("/system_stats")).status_code == 200
        except httpx.HTTPError:
            return False

    async def free_memory(self) -> None:
        """Unload models from VRAM (e.g. so the LLM can use the GPU)."""
        async with self._http() as http:
            await http.post("/free", json={"unload_models": True, "free_memory": True})

    async def upload_image(self, data: bytes, filename: str) -> str:
        """Upload an input image (e.g. a reference); returns the name for LoadImage."""
        async with self._http() as http:
            resp = await http.post(
                "/upload/image",
                files={"image": (filename, data, "image/png")},
                data={"overwrite": "true"},
            )
        if resp.status_code != 200:
            raise ComfyError(f"Image upload failed: {resp.text[:500]}")
        info = resp.json()
        return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]

    async def missing_models(self, refs: list[ModelRef]) -> list[str]:
        """Return the model files from ``refs`` that ComfyUI does not have.

        Uses /object_info to read each loader's list of available files. If that
        list cannot be read, the file is assumed present (ComfyUI will still
        reject the workflow with its own error).
        """
        missing = []
        async with self._http() as http:
            for ref in refs:
                try:
                    resp = await http.get(f"/object_info/{ref.node_class}")
                    spec = resp.json()[ref.node_class]["input"]["required"][ref.input]
                except (httpx.HTTPError, KeyError, ValueError):
                    continue
                options = _combo_options(spec)
                if options is not None and ref.filename not in options:
                    missing.append(ref.filename)
        return missing

    async def run(
        self, graph: dict[str, Any], on_progress: ProgressCallback | None = None
    ) -> list[bytes]:
        """Queue a workflow, wait until it finishes and return the output images."""
        client_id = uuid.uuid4().hex
        # Connect the WebSocket *before* queueing so no message is missed.
        async with (
            self._http() as http,
            websockets.connect(
                f"{self.ws_url}/ws?clientId={client_id}", max_size=None
            ) as ws,
        ):
            resp = await http.post("/prompt", json={"prompt": graph, "client_id": client_id})
            if resp.status_code != 200:
                raise ComfyError(f"ComfyUI rejected the workflow: {resp.text[:1000]}")
            prompt_id = resp.json()["prompt_id"]
            logger.info("Queued ComfyUI prompt %s", prompt_id)

            try:
                await asyncio.wait_for(
                    self._wait_for_completion(ws, prompt_id, on_progress), self.timeout_s
                )
            except TimeoutError:
                await http.post("/interrupt")
                raise ComfyError(f"Generation timed out after {self.timeout_s}s") from None

            return await self._collect_images(http, prompt_id)

    @staticmethod
    async def _wait_for_completion(
        ws, prompt_id: str, on_progress: ProgressCallback | None
    ) -> None:
        async for raw in ws:
            if isinstance(raw, bytes):  # binary latent previews
                continue
            msg = json.loads(raw)
            msg_type, data = msg.get("type"), msg.get("data", {})
            if data.get("prompt_id") not in (None, prompt_id):
                continue  # message for another client's prompt

            if msg_type == "progress" and on_progress:
                await on_progress(data["value"], data["max"])
            elif msg_type == "execution_error":
                raise ComfyError(
                    f"{data.get('node_type')} failed: {data.get('exception_message')}"
                )
            elif msg_type == "execution_interrupted":
                raise ComfyError("Generation was interrupted")
            elif msg_type == "execution_success":
                return
            elif msg_type == "executing" and data.get("node") is None:
                return  # older ComfyUI versions signal completion this way
        raise ComfyError("ComfyUI WebSocket closed before the prompt finished")

    @staticmethod
    async def _collect_images(http: httpx.AsyncClient, prompt_id: str) -> list[bytes]:
        history = (await http.get(f"/history/{prompt_id}")).json().get(prompt_id)
        if not history:
            raise ComfyError(f"No history found for prompt {prompt_id}")

        images: list[bytes] = []
        for node_output in history.get("outputs", {}).values():
            for image in node_output.get("images", []):
                resp = await http.get(
                    "/view",
                    params={
                        "filename": image["filename"],
                        "subfolder": image.get("subfolder", ""),
                        "type": image.get("type", "output"),
                    },
                )
                resp.raise_for_status()
                images.append(resp.content)
        if not images:
            raise ComfyError("The workflow finished but produced no images")
        return images
