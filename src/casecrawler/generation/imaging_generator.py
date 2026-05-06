from __future__ import annotations

from pathlib import Path

from casecrawler.models.synthetic import ImagingAsset


class ImagingGenerator:
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

