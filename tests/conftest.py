from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def workflows_dir() -> Path:
    return ROOT / "workflows"


def make_png(width: int, height: int, color: str = "steelblue") -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), color).save(buf, "PNG")
    return buf.getvalue()
