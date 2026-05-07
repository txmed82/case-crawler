from __future__ import annotations

from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from casecrawler.integrations.huggingface import require_package
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
    ) -> None:
        self._diffusers_pipeline = diffusers_pipeline
        self._diffusers_model_id = diffusers_model_id

    def generate_placeholder(
        self,
        output_dir: str,
        prompt: str,
        modality: str = "XR",
        body_region: str = "chest",
    ) -> ImagingAsset:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return ImagingAsset(
            image_id="placeholder",
            modality=modality,
            body_region=body_region,
            prompt=prompt,
            file_path=None,
            report_text=(
                "Synthetic imaging placeholder. Configure a diffusers backend "
                "to render pixels."
            ),
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
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        image_id = f"img-{uuid5(NAMESPACE_URL, f'{self._diffusers_model_id}:{prompt}')}"
        file_path = output_path / f"{image_id}.png"
        pipeline = self._diffusers_pipeline or self._load_diffusers_pipeline()
        result = pipeline(prompt=prompt, negative_prompt=negative_prompt)
        if not getattr(result, "images", None):
            raise RuntimeError("Diffusers backend returned no images.")
        result.images[0].save(file_path)
        return ImagingAsset(
            image_id=image_id,
            modality=modality,
            body_region=body_region,
            prompt=prompt,
            file_path=str(file_path),
            report_text=(
                f"Synthetic {modality} image of the {body_region}. "
                "Generated pixels require downstream clinical validation."
            ),
            generation_backend=f"diffusers:{self._diffusers_model_id}",
        )

    def _load_diffusers_pipeline(self):
        diffusers = require_package("diffusers", "imaging")
        pipeline = diffusers.DiffusionPipeline.from_pretrained(self._diffusers_model_id)
        if hasattr(pipeline, "to"):
            return pipeline.to("cpu")
        return pipeline
