from __future__ import annotations

import json
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Any, Protocol
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
from casecrawler.imaging.file_metadata import image_file_metadata
from casecrawler.models.synthetic import ImagingAsset


EXTERNAL_IMAGING_TIMEOUT_SECONDS = 180.0


class ImageLike(Protocol):
    def save(self, path: str | Path) -> None: ...


class DiffusersResult(Protocol):
    images: list[ImageLike]


class ExternalImagingRunner(Protocol):
    def __call__(self, command: list[str], payload: str) -> str: ...


class ImagingGenerator:
    def __init__(
        self,
        diffusers_pipeline=None,
        diffusers_model_id: str = "stabilityai/stable-diffusion-2-1",
        imaging_model_profile: str | ImagingModelProfile | None = None,
        external_command: list[str] | None = None,
        external_runner: ExternalImagingRunner | None = None,
    ) -> None:
        if external_command is not None and not external_command:
            raise ValueError("external_command must not be empty when provided.")
        profile = (
            imaging_model_profile
            if isinstance(imaging_model_profile, ImagingModelProfile)
            else resolve_imaging_model_profile(imaging_model_profile)
        )
        self._diffusers_pipeline = diffusers_pipeline
        self._imaging_model_profile = profile
        self._diffusers_model_id = profile.model_id if profile else diffusers_model_id
        self._external_command = external_command
        self._external_runner = external_runner or _run_external_command

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
        metadata = _asset_generation_metadata(
            file_path=file_path,
            backend="placeholder",
            profile=None,
            command=None,
        )
        # Strict export modes (multimodal_jsonl --strict, release packages)
        # filter on this flag. Placeholder PNGs are clearly marked as such
        # so they cannot quietly leak into a downstream training set.
        metadata["is_placeholder"] = True
        metadata["placeholder_reason"] = (
            "Synthetic placeholder image generated for offline / CI use. "
            "Not suitable for vision-model training. Configure an HF imaging "
            "backend (hf_local / hf_endpoint) for real images."
        )
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
            metadata=metadata,
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
        backend = (
            f"diffusers:{profile.name}:{self._diffusers_model_id}"
            if profile
            else f"diffusers:{self._diffusers_model_id}"
        )
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
            generation_backend=backend,
            metadata=_asset_generation_metadata(
                file_path=file_path,
                backend=backend,
                profile=profile,
                command=None,
            ),
        )

    def generate_external(
        self,
        output_dir: str,
        prompt: str,
        modality: str = "XR",
        body_region: str = "chest",
    ) -> ImagingAsset:
        if self._external_command is None:
            raise RuntimeError("External imaging backend requires an imaging command.")
        payload = json.dumps(
            {
                "output_dir": output_dir,
                "prompt": prompt,
                "modality": modality,
                "body_region": body_region,
            },
            sort_keys=True,
        )
        output = self._external_runner(self._external_command, payload)
        try:
            raw_asset = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError("External imaging backend returned invalid JSON.") from exc
        if isinstance(raw_asset, dict) and isinstance(raw_asset.get("asset"), dict):
            raw_asset = raw_asset["asset"]
        if not isinstance(raw_asset, dict):
            raise RuntimeError(
                "External imaging backend must return an ImagingAsset object or "
                "an object with an asset object."
            )
        return _external_imaging_asset(
            raw_asset,
            backend=f"external:{' '.join(self._external_command)}",
            command=self._external_command,
            prompt=prompt,
            modality=modality,
            body_region=body_region,
        )

    def _load_diffusers_pipeline(self):
        diffusers = require_package("diffusers", "imaging")
        pipeline = diffusers.DiffusionPipeline.from_pretrained(self._diffusers_model_id)
        if hasattr(pipeline, "to"):
            return pipeline.to("cpu")
        return pipeline


def _external_imaging_asset(
    asset: dict,
    *,
    backend: str,
    command: list[str],
    prompt: str,
    modality: str,
    body_region: str,
) -> ImagingAsset:
    labels = infer_imaging_labels(prompt, modality)
    file_path = asset.get("file_path")
    supplied_metadata = asset.get("metadata")
    metadata = supplied_metadata if isinstance(supplied_metadata, dict) else {}
    metadata = {
        **_asset_generation_metadata(
            file_path=Path(file_path) if isinstance(file_path, str) else None,
            backend=backend,
            profile=None,
            command=command,
        ),
        **metadata,
    }
    payload = {
        "image_id": f"img-external-{uuid4()}",
        "modality": modality,
        "body_region": body_region,
        "prompt": prompt,
        "report_text": build_imaging_report(
            prompt=prompt,
            modality=modality,
            body_region=body_region,
            labels=labels,
        ),
        "labels": labels,
        "generation_backend": backend,
        **asset,
        "metadata": metadata,
    }
    if not payload.get("generation_backend"):
        payload["generation_backend"] = backend
    if not isinstance(payload.get("metadata"), dict):
        payload["metadata"] = metadata
    return ImagingAsset.model_validate(payload)


def _asset_generation_metadata(
    *,
    file_path: Path | None,
    backend: str,
    profile: ImagingModelProfile | None,
    command: list[str] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "generation_backend": backend,
        "artifact_contract": "casecrawler.models.synthetic.ImagingAsset",
    }
    if profile is not None:
        metadata["model_profile"] = {
            "name": profile.name,
            "model_id": profile.model_id,
            "adapter_type": profile.adapter_type,
            "license": profile.license,
            "gated": profile.gated,
            "use_policy": profile.use_policy,
            "validation_requirements": list(profile.validation_requirements),
        }
    if command is not None:
        metadata["external_command"] = list(command)
        metadata["external_contract"] = {
            "stdin": ["output_dir", "prompt", "modality", "body_region"],
            "stdout": "ImagingAsset JSON or {'asset': ImagingAsset JSON}",
        }
    if file_path is not None and file_path.exists() and file_path.is_file():
        metadata["file"] = image_file_metadata(file_path)
    return metadata


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


def _run_external_command(command: list[str], payload: str) -> str:
    try:
        result = subprocess.run(
            command,
            input=payload,
            capture_output=True,
            check=True,
            text=True,
            timeout=EXTERNAL_IMAGING_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "External imaging backend timed out after "
            f"{EXTERNAL_IMAGING_TIMEOUT_SECONDS:.0f}s: {command!r}."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "External imaging backend failed with exit code "
            f"{exc.returncode}: {command!r}. stdout={exc.stdout!r} stderr={exc.stderr!r}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"External imaging backend could not be executed: {command!r}."
        ) from exc
    return result.stdout
