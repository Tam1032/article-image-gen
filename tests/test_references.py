import base64

import pytest
from PIL import Image
from io import BytesIO

from app.services.comfy import _combo_options
from app.services.references import InvalidReference, describe, load_reference
from app.services.workflows import WorkflowRegistry
from tests.conftest import make_png

MB = 1_000_000


def data_uri(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


async def test_data_uri_is_decoded_and_downscaled():
    out = await load_reference(data_uri(make_png(2048, 1024)), max_bytes=10 * MB)
    img = Image.open(BytesIO(out))
    assert img.format == "PNG" and img.size == (1024, 512)


@pytest.mark.parametrize(
    "ref, match",
    [
        ("data:text/plain;base64,aGVsbG8=", "data URI"),
        ("data:image/png;base64,@@@", "base64"),
        (data_uri(b"not an image"), "not a valid image"),
        ("ftp://example.com/a.png", "http"),
        ("http://127.0.0.1:8188/view?filename=x.png", "not a public address"),
        ("http://localhost/a.png", "not a public address"),
    ],
)
async def test_bad_references_are_rejected(ref, match):
    with pytest.raises(InvalidReference, match=match):
        await load_reference(ref, max_bytes=10 * MB)


async def test_size_limit():
    with pytest.raises(InvalidReference, match="larger"):
        await load_reference(data_uri(make_png(64, 64)), max_bytes=10)


def test_describe_shortens_data_uris():
    uri = data_uri(make_png(64, 64))
    assert len(describe(uri)) < 60
    assert describe("https://example.com/a.png") == "https://example.com/a.png"


def test_redux_workflow_and_required_models(workflows_dir):
    wf = WorkflowRegistry(workflows_dir).get("flux_redux")
    graph = wf.build({"prompt": "x", "seed": 1, "width": 512, "height": 512,
                      "reference_image": "ref.png"})
    assert graph["40"]["inputs"]["image"] == "ref.png"
    assert graph["44"]["inputs"]["strength"] == 0.5
    assert {m.filename for m in wf.required_models()} == {
        "flux1-dev-fp8.safetensors",
        "sigclip_vision_patch14_384.safetensors",
        "flux1-redux-dev.safetensors",
    }


def test_combo_options_formats():
    assert _combo_options([["a.safetensors", "b.safetensors"], {}]) == ["a.safetensors", "b.safetensors"]
    assert _combo_options(["COMBO", {"options": ["a.safetensors"]}]) == ["a.safetensors"]
    assert _combo_options(["INT", {"default": 1}]) is None
