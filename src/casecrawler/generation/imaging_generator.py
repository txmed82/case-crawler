from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from casecrawler.integrations.huggingface import require_package
from casecrawler.generation.imaging_models import (
    ImagingModelProfile,
    resolve_imaging_model_profile,
)
from casecrawler.generation.imaging_templates import (
    build_imaging_report,
    infer_imaging_labels,
)
from casecrawler.models.synthetic import ImagingAsset


class ImageLike(Protocol):
    def save(self, path: str | Path) -> None: ...


class DiffusersResult(Protocol):
    images: list[ImageLike]


class ImagingGenerator:
    def __init__(
        self,
        diffusers_pipeline=None,
        diffusers_model_id: str = "stabilityai/stable-diffusion-2-1",
        imaging_model_profile: str | ImagingModelProfile | None = None,
    ) -> None:
        profile = (
            imaging_model_profile
            if isinstance(imaging_model_profile, ImagingModelProfile)
            else resolve_imaging_model_profile(imaging_model_profile)
        )
        self._diffusers_pipeline = diffusers_pipeline
        self._imaging_model_profile = profile
        self._diffusers_model_id = profile.model_id if profile else diffusers_model_id

    def generate_placeholder(
        self,
        output_dir: str,
        prompt: str,
        modality: str = "XR",
        body_region: str = "chest",
    ) -> ImagingAsset:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        image_id = f"img-placeholder-{uuid4()}"
        file_path = Path(output_dir) / f"{image_id}.png"
        _write_placeholder_png(file_path, prompt=prompt, modality=modality)
        labels = infer_imaging_labels(prompt, modality)
        return ImagingAsset(
            image_id=image_id,
            modality=modality,
            body_region=body_region,
            prompt=prompt,
            file_path=str(file_path),
            report_text=build_imaging_report(
                prompt=prompt,
                modality=modality,
                body_region=body_region,
                labels=labels,
            ),
            labels=labels,
            generation_backend="placeholder",
        )

    def generate_diffusers(
        self,
        output_dir: str,
        prompt: str,
        modality: str = "XR",
        body_region: str = "chest",
        negative_prompt: str | None = None,
    ) -> ImagingAsset:
        profile = self._imaging_model_profile
        if profile:
            if not profile.is_compatible(modality, body_region):
                raise ValueError(
                    f"Imaging model profile '{profile.name}' is incompatible with "
                    f"requested {modality} {body_region}."
                )
            prompt = profile.render_prompt(prompt)
            modality = modality if modality != "XR" else profile.modality
            body_region = body_region if body_region != "chest" else profile.body_region
            negative_prompt = negative_prompt or profile.default_negative_prompt
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        image_id = f"img-{uuid4()}"
        file_path = output_path / f"{image_id}.png"
        if self._diffusers_pipeline is None:
            self._diffusers_pipeline = self._load_diffusers_pipeline()
        pipeline = self._diffusers_pipeline
        result = pipeline(prompt=prompt, negative_prompt=negative_prompt)
        if not getattr(result, "images", None):
            raise RuntimeError("Diffusers backend returned no images.")
        result.images[0].save(file_path)
        labels = infer_imaging_labels(prompt, modality)
        return ImagingAsset(
            image_id=image_id,
            modality=modality,
            body_region=body_region,
            prompt=prompt,
            file_path=str(file_path),
            report_text=build_imaging_report(
                prompt=prompt,
                modality=modality,
                body_region=body_region,
                labels=labels,
            ),
            labels=labels,
            generation_backend=(
                f"diffusers:{profile.name}:{self._diffusers_model_id}"
                if profile
                else f"diffusers:{self._diffusers_model_id}"
            ),
        )

    def _load_diffusers_pipeline(self):
        diffusers = require_package("diffusers", "imaging")
        pipeline = diffusers.DiffusionPipeline.from_pretrained(self._diffusers_model_id)
        if hasattr(pipeline, "to"):
            return pipeline.to("cpu")
        return pipeline


def _write_placeholder_png(
    path: Path,
    *,
    prompt: str,
    modality: str,
    width: int = 128,
    height: int = 128,
) -> None:
    seed = zlib.crc32(f"{modality}:{prompt}".encode("utf-8"))
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            radial = abs(x - width // 2) + abs(y - height // 2)
            texture = ((x * 17 + y * 31 + seed) % 53) - 26
            intensity = max(0, min(255, 210 - radial * 2 + texture))
            row.append(intensity)
        rows.append(b"\x00" + bytes(row))
    raw = b"".join(rows)
    png = [
        b"\x89PNG\r\n\x1a\n",
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(raw)),
        _png_chunk(b"IEND", b""),
    ]
    path.write_bytes(b"".join(png))


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )
