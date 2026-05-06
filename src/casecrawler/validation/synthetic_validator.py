from __future__ import annotations

from casecrawler.models.synthetic import SyntheticRecord, ValidationReport
from casecrawler.validation.clinical_rules import validate_lab_flags, validate_vitals
from casecrawler.validation.privacy import validate_privacy


class SyntheticValidator:
    def __init__(self, threshold: float = 0.8) -> None:
        self._threshold = threshold

    def validate(self, record: SyntheticRecord) -> ValidationReport:
        issues = [
            *validate_lab_flags(record),
            *validate_vitals(record),
            *validate_privacy(record),
        ]

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
            and not issues
        )
        return ValidationReport(
            schema_score=schema_score,
            clinical_consistency_score=clinical_score,
            privacy_score=privacy_score,
            utility_score=utility_score,
            modality_alignment_score=None,
            approved=approved,
            issues=issues,
        )
