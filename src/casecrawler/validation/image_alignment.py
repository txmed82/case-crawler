from __future__ import annotations

from casecrawler.models.synthetic import ImagingAsset


class ImageAlignmentValidator:
    def score(self, asset: ImagingAsset) -> float:
        if asset.prompt and asset.report_text:
            return 1.0
        return 0.0
