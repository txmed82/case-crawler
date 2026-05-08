from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from casecrawler.integrations.huggingface import require_package
from casecrawler.models.synthetic import ImagingAsset, Modality, ValidationIssue


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


def validate_image_file_asset(asset: ImagingAsset) -> list[ValidationIssue]:
    if asset.generation_backend == "placeholder" and asset.file_path is None:
        return []
    field_prefix = f"imaging.{asset.image_id}"
    issues: list[ValidationIssue] = []
    if not asset.file_path:
        return [
            ValidationIssue(
                severity="error",
                modality=Modality.IMAGING,
                field=f"{field_prefix}.file_path",
                message="Generated image asset has no file path.",
            )
        ]
    path = Path(asset.file_path)
    if not path.exists():
        return [
            ValidationIssue(
                severity="error",
                modality=Modality.IMAGING,
                field=f"{field_prefix}.file_path",
                message=f"Generated image file does not exist: {asset.file_path}.",
            )
        ]
    if path.stat().st_size <= 0:
        issues.append(
            ValidationIssue(
                severity="error",
                modality=Modality.IMAGING,
                field=f"{field_prefix}.file_size",
                message=f"Generated image file is empty: {asset.file_path}.",
            )
        )
    if path.suffix.lower() not in _SUPPORTED_IMAGE_EXTENSIONS:
        issues.append(
            ValidationIssue(
                severity="error",
                modality=Modality.IMAGING,
                field=f"{field_prefix}.file_format",
                message=(
                    "Generated image file extension must be one of "
                    f"{', '.join(sorted(_SUPPORTED_IMAGE_EXTENSIONS))}."
                ),
            )
        )
    elif not _has_supported_image_signature(path):
        issues.append(
            ValidationIssue(
                severity="error",
                modality=Modality.IMAGING,
                field=f"{field_prefix}.file_signature",
                message=f"Generated image file signature is invalid: {asset.file_path}.",
            )
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

_SUPPORTED_IMAGE_EXTENSIONS = {".dcm", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _has_supported_image_signature(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".dcm":
        return True
    try:
        signature = path.read_bytes()[:16]
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


class MedGemmaImageTextValidator:
    def __init__(
        self,
        analyzer: Callable[[str, str], str | float | dict[str, Any]] | None = None,
        model_id: str = "google/medgemma-4b-it",
    ) -> None:
        self._analyzer = analyzer
        self._model_id = model_id

    def score(self, asset: ImagingAsset) -> float:
        if not asset.file_path:
            return 0.0
        analyzer = self._analyzer or self._load_analyzer()
        return _coerce_alignment_score(analyzer(asset.file_path, asset.report_text))

    def _load_analyzer(self):
        transformers = require_package("transformers", "hf")
        pil = require_package("PIL", "imaging")
        pipe = transformers.pipeline(
            "image-text-to-text",
            model=self._model_id,
        )

        def _analyze(image_path: str, report_text: str) -> str:
            prompt = (
                "You are validating synthetic medical image training data. "
                "Compare the image to this radiology report and return only JSON "
                'with a numeric "score" from 0 to 1 and a short "rationale". '
                f"Report: {report_text}"
            )
            result = pipe(images=pil.Image.open(image_path), text=prompt)
            if isinstance(result, list) and result:
                first = result[0]
                if isinstance(first, dict):
                    return str(
                        first.get("generated_text")
                        or first.get("text")
                        or first.get("answer")
                        or ""
                    )
            return str(result)

        self._analyzer = _analyze
        return _analyze


def _coerce_alignment_score(value: str | float | dict[str, Any]) -> float:
    if isinstance(value, (int, float)):
        return _clamp_score(float(value))
    if isinstance(value, dict):
        return _safe_float_score(value.get("score", 0.0))
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        match = re.search(
            r"\b(?:score|alignment)\b[^0-9]*(0(?:\.\d+)?|1(?:\.0+)?)",
            str(value),
            flags=re.IGNORECASE,
        )
        if not match:
            return 0.0
        return _clamp_score(float(match.group(1)))
    if isinstance(parsed, dict):
        return _safe_float_score(parsed.get("score", 0.0))
    if isinstance(parsed, (int, float)):
        return _clamp_score(float(parsed))
    return 0.0


def _safe_float_score(value: Any) -> float:
    try:
        return _clamp_score(float(value))
    except (TypeError, ValueError):
        return 0.0


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))
