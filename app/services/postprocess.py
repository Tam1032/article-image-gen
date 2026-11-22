from io import BytesIO

from PIL import Image, ImageOps


def to_web_image(data: bytes, size: tuple[int, int], quality: int = 85) -> bytes:
    """Center-crop to the target aspect ratio, resize, and encode as WebP.

    Re-encoding also drops the ComfyUI workflow metadata embedded in the PNG.
    """
    with Image.open(BytesIO(data)) as img:
        img = ImageOps.fit(img.convert("RGB"), size, Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, "WEBP", quality=quality, method=6)
    return buf.getvalue()
