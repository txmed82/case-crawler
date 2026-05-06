from __future__ import annotations

import re

from casecrawler.models.synthetic import ImagingAsset


class ImageAlignmentValidator:
    def score(self, asset: ImagingAsset) -> float:
        prompt_tokens = self._tokens(asset.prompt)
        report_tokens = self._tokens(asset.report_text)
        if not prompt_tokens or not report_tokens:
            return 0.0
        overlap = prompt_tokens & report_tokens
        union = prompt_tokens | report_tokens
        return len(overlap) / len(union)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stopwords = {"a", "an", "and", "for", "in", "of", "the", "to", "with"}
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2 and token not in stopwords
        }
