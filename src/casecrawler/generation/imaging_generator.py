from __future__ import annotations

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
        labels = infer_imaging_labels(prompt, modality)
        return ImagingAsset(
            image_id="placeholder",
            modality=modality,
            body_region=body_region,
            prompt=prompt,
            file_path=None,
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
