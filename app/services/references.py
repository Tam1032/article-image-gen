"""Reference images sent with a job: fetch, validate and normalise.

A reference is either an ``http(s)://`` URL or a ``data:image/...;base64,`` URI.
The API resolves references when the job is created, so bad input fails fast
with a 422 instead of failing later in the worker.
"""

import asyncio
import base64
import binascii
import ipaddress
import socket
from io import BytesIO
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_SIDE = 1024  # Redux encodes at 384 px; 1024 keeps detail for scoring


class InvalidReference(ValueError):
    pass


def describe(reference: str) -> str:
    """Short form stored with the job (data URIs can be megabytes long)."""
    if reference.startswith("data:"):
        return f"{reference[:30]}... ({len(reference)} chars)"
    return reference


def normalize(data: bytes) -> bytes:
    """Decode any common image format, fix orientation, downscale, re-encode as PNG."""
    try:
        with Image.open(BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidReference(f"not a valid image ({exc})") from None
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def decode_data_uri(uri: str, max_bytes: int) -> bytes:
    header, sep, payload = uri.partition(",")
    if not sep or not header.startswith("data:image/") or not header.endswith(";base64"):
        raise InvalidReference("data URI must look like data:image/<type>;base64,<data>")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise InvalidReference("invalid base64 in data URI") from None
    if len(data) > max_bytes:
        raise InvalidReference(f"image larger than {max_bytes // 1_000_000} MB")
    return data


async def _check_public_host(host: str) -> None:
    """Reject hosts that resolve to private/loopback addresses (basic SSRF guard)."""
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except socket.gaierror:
        raise InvalidReference(f"cannot resolve host '{host}'") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise InvalidReference(f"host '{host}' is not a public address")


async def fetch_url(url: str, max_bytes: int, allow_private: bool = False) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise InvalidReference("only http(s) URLs are supported")
    if not allow_private:
        await _check_public_host(parsed.hostname)

    # Redirects are not followed: they could point to an internal address.
    async with httpx.AsyncClient(timeout=20, follow_redirects=False) as http:
        try:
            async with http.stream("GET", url) as resp:
                if resp.status_code != 200:
                    raise InvalidReference(f"download failed with HTTP {resp.status_code}")
                chunks, size = [], 0
                async for chunk in resp.aiter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise InvalidReference(f"image larger than {max_bytes // 1_000_000} MB")
                    chunks.append(chunk)
        except httpx.HTTPError as exc:
            raise InvalidReference(f"download failed ({exc.__class__.__name__})") from None
    return b"".join(chunks)


async def load_reference(reference: str, max_bytes: int, allow_private: bool = False) -> bytes:
    """Resolve one reference to normalised PNG bytes."""
    if reference.startswith("data:"):
        data = decode_data_uri(reference, max_bytes)
    else:
        data = await fetch_url(reference, max_bytes, allow_private)
    return await asyncio.to_thread(normalize, data)
