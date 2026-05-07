from __future__ import annotations

from casecrawler.models.synthetic import Modality, SyntheticRecord, ValidationIssue, ValidationReport
from casecrawler.validation.clinical_rules import (
    validate_lab_flags,
    validate_temporal_consistency,
    validate_text_structured_contradictions,
    validate_vitals,
)
from casecrawler.validation.image_alignment import (
    ImageAlignmentValidator,
    validate_radiology_label_consistency,
)
from casecrawler.validation.privacy import validate_privacy


class SyntheticValidator:
    def __init__(
        self,
        threshold: float = 0.8,
        image_alignment_validator: ImageAlignmentValidator | None = None,
    ) -> None:
        self._threshold = threshold
        self._image_alignment_validator = image_alignment_validator or ImageAlignmentValidator()

    def validate(self, record: SyntheticRecord) -> ValidationReport:
        issues = [
            *validate_temporal_consistency(record),
            *validate_lab_flags(record),
            *validate_vitals(record),
            *validate_text_structured_contradictions(record),
            *validate_privacy(record),
        ]
        modality_alignment_score = self._image_alignment_score(record)
        if modality_alignment_score is not None and modality_alignment_score < self._threshold:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.IMAGING,
                    field="imaging.alignment",
                    message="Image report text is weakly aligned with the image prompt.",
                )
            )
        for asset in record.imaging:
            for message in validate_radiology_label_consistency(asset):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        modality=Modality.IMAGING,
                        field=f"imaging.{asset.image_id}.labels",
                        message=message,
                    )
                )

        clinical_error_count = sum(
            1
            for issue in issues
            if issue.severity == "error" and issue.field != "privacy"
        )
        schema_score = 1.0
        clinical_score = max(0.0, 1.0 - 0.25 * clinical_error_count)
        privacy_score = 0.0 if any(issue.field == "privacy" for issue in issues) else 1.0
        utility_score = (
            1.0
            if (
                record.documents
                or record.labs
                or record.vitals
                or record.time_series
                or record.imaging
            )
            else 0.0
        )
        approved = (
            schema_score >= self._threshold
            and clinical_score >= self._threshold
            and privacy_score >= self._threshold
            and utility_score >= self._threshold
            and (
                modality_alignment_score is None
                or modality_alignment_score >= self._threshold
            )
            and not issues
        )
        return ValidationReport(
            schema_score=schema_score,
            clinical_consistency_score=clinical_score,
            privacy_score=privacy_score,
            utility_score=utility_score,
            modality_alignment_score=modality_alignment_score,
            approved=approved,
            issues=issues,
        )

    def _image_alignment_score(self, record: SyntheticRecord) -> float | None:
        if not record.imaging:
            return None
        scores = [self._image_alignment_validator.score(asset) for asset in record.imaging]
        return sum(scores) / len(scores)
