from __future__ import annotations

import re
from collections.abc import Callable

from casecrawler.integrations.huggingface import require_package
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
        allowed_acronyms = {"ct", "mr", "xr", "us"}
        stopwords = {"a", "an", "and", "for", "in", "of", "the", "to", "with"}
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if (len(token) > 2 or token in allowed_acronyms)
            and token not in stopwords
        }


def validate_radiology_label_consistency(asset: ImagingAsset) -> list[str]:
    """Check that radiology labels are supported by prompt/report text."""
    evidence_text = f"{asset.prompt} {asset.report_text}"
    issues = []
    for label in asset.labels:
        label_terms = _label_terms(label.display, label.code)
        if not label_terms:
            continue
        if not any(_contains_term(evidence_text, term) for term in label_terms):
            issues.append(
                f"Radiology label {label.display!r} is not supported by prompt/report text."
            )
            continue
        negated_terms = [
            term
            for term in label_terms
            if _contains_negated_term(asset.report_text, term)
        ]
        if negated_terms:
            issues.append(
                f"Radiology label {label.display!r} is negated in report text."
            )
    return issues


def _label_terms(display: str, code: str) -> set[str]:
    terms = {display.lower(), code.replace("_", " ").lower()}
    terms.update(_RADIOLOGY_SYNONYMS.get(display.lower(), set()))
    terms.update(_RADIOLOGY_SYNONYMS.get(code.replace("_", " ").lower(), set()))
    return {term for term in terms if term}


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text.lower()) is not None


def _contains_negated_term(text: str, term: str) -> bool:
    lowered = text.lower()
    return any(
        re.search(rf"\b{negation}\s+(?:\w+\s+){{0,3}}{re.escape(term)}\b", lowered)
        for negation in _NEGATION_TERMS
    )


_NEGATION_TERMS = {
    "absent",
    "negative for",
    "no",
    "without",
}

_RADIOLOGY_SYNONYMS: dict[str, set[str]] = {
    "atelectasis": {"volume loss", "linear opacity"},
    "cardiomegaly": {"enlarged cardiac silhouette", "enlarged heart"},
    "consolidation": {"airspace opacity", "airspace disease"},
    "edema": {"pulmonary edema", "interstitial edema"},
    "effusion": {"pleural effusion"},
    "fracture": {"osseous fracture"},
    "opacity": {"opacification", "airspace opacity"},
    "pleural effusion": {"effusion"},
    "pneumonia": {"consolidation", "airspace opacity"},
    "pneumothorax": {"pleural air"},
    "pulmonary edema": {"edema", "interstitial edema"},
}


class BiomedCLIPImageValidator:
    def __init__(
        self,
        scorer: Callable[[str, str], float] | None = None,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    ) -> None:
        self._scorer = scorer
        self._model_name = model_name

    def score(self, asset: ImagingAsset) -> float:
        if not asset.file_path:
            return 0.0
        scorer = self._scorer or self._load_scorer()
        return max(0.0, min(1.0, float(scorer(asset.file_path, asset.report_text))))

    def _load_scorer(self):
        open_clip = require_package("open_clip", "imaging")
        torch = require_package("torch", "imaging")
        pil = require_package("PIL", "imaging")
        model, _, preprocess = open_clip.create_model_and_transforms(self._model_name)
        tokenizer = open_clip.get_tokenizer(self._model_name)
        model.eval()

        def _score(image_path: str, report_text: str) -> float:
            image = preprocess(pil.Image.open(image_path)).unsqueeze(0)
            text = tokenizer([report_text])
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).item()
            return (similarity + 1.0) / 2.0

        self._scorer = _score
        return _score
