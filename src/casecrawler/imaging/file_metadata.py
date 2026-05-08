from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any


SUPPORTED_IMAGE_EXTENSIONS = {".dcm", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def image_file_metadata(path: str | Path) -> dict[str, Any]:
    image_path = Path(path)
    data = image_path.read_bytes()
    width, height = _image_dimensions(image_path, data)
    metadata: dict[str, Any] = {
        "byte_size": len(data),
        "mime_type": image_mime_type(image_path),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    if width is not None and height is not None:
        metadata["width"] = width
        metadata["height"] = height
    return metadata


def image_mime_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".dcm":
        return "application/dicom"
    return "application/octet-stream"


def has_supported_image_signature(path: str | Path) -> bool:
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    if suffix == ".dcm":
        return True
    try:
        signature = image_path.read_bytes()[:16]
    except OSError:
        return False
    if suffix == ".png":
        return signature.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix in {".jpg", ".jpeg"}:
        return signature.startswith(b"\xff\xd8\xff")
    if suffix in {".tif", ".tiff"}:
        return signature.startswith((b"II*\x00", b"MM\x00*"))
    if suffix == ".webp":
        return signature.startswith(b"RIFF") and signature[8:12] == b"WEBP"
    return False


def raster_dimensions(path: str | Path) -> tuple[int | None, int | None]:
    image_path = Path(path)
    try:
        data = image_path.read_bytes()
    except OSError:
        return None, None
    return _image_dimensions(image_path, data)


def _image_dimensions(path: Path, data: bytes) -> tuple[int | None, int | None]:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return _png_dimensions(data)
    if suffix in {".jpg", ".jpeg"}:
        return _jpeg_dimensions(data)
    if suffix == ".webp":
        return _webp_dimensions(data)
    return None, None


def _png_dimensions(data: bytes) -> tuple[int | None, int | None]:
    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None, None
    return struct.unpack(">II", data[16:24])


def _jpeg_dimensions(data: bytes) -> tuple[int | None, int | None]:
    if len(data) < 4 or not data.startswith(b"\xff\xd8"):
        return None, None
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(data):
            return None, None
        segment_length = int.from_bytes(data[index : index + 2], "big")
        if segment_length < 2 or index + segment_length > len(data):
            return None, None
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            height = int.from_bytes(data[index + 3 : index + 5], "big")
            width = int.from_bytes(data[index + 5 : index + 7], "big")
            return width, height
        index += segment_length
    return None, None


def _webp_dimensions(data: bytes) -> tuple[int | None, int | None]:
    if len(data) < 30 or not (data.startswith(b"RIFF") and data[8:12] == b"WEBP"):
        return None, None
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height
    return None, None
