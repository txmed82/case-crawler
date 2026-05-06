from __future__ import annotations

from casecrawler.models.synthetic import Modality, SyntheticRecord, ValidationIssue


def validate_lab_flags(record: SyntheticRecord) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for lab in record.labs:
        if (
            isinstance(lab.value, int | float)
            and lab.reference_low is not None
            and lab.reference_high is not None
        ):
            outside = lab.value < lab.reference_low or lab.value > lab.reference_high
            if outside and not lab.flag:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        modality=Modality.LABS,
                        field="labs.flag",
                        message=f"{lab.name} is outside reference range but has no flag.",
                    )
                )
    return issues


def validate_vitals(record: SyntheticRecord) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for vital in record.vitals:
        if vital.name == "SpO2" and not 0 <= vital.value <= 100:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.VITALS,
                    field="vitals.SpO2",
                    message="SpO2 must be between 0 and 100.",
                )
            )
        if vital.name == "HR" and not 0 < vital.value < 260:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.VITALS,
                    field="vitals.HR",
                    message="Heart rate is outside a plausible clinical range.",
                )
            )
    return issues

