from __future__ import annotations

import re
from datetime import datetime

from casecrawler.models.synthetic import Modality, SyntheticRecord, ValidationIssue


def validate_temporal_consistency(record: SyntheticRecord) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for encounter in record.encounters:
        start = _parse_datetime(encounter.start)
        end = _parse_datetime(encounter.end) if encounter.end else None
        if start is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.STRUCTURED_EHR,
                    field="encounters.start",
                    message=f"Encounter {encounter.encounter_id} has an invalid start time.",
                )
            )
        if encounter.end and end is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.STRUCTURED_EHR,
                    field="encounters.end",
                    message=f"Encounter {encounter.encounter_id} has an invalid end time.",
                )
            )
        if start is not None and end is not None and end < start:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.STRUCTURED_EHR,
                    field="encounters.period",
                    message=f"Encounter {encounter.encounter_id} ends before it starts.",
                )
            )

    for medication in record.medication_history:
        start = _parse_datetime(medication.start) if medication.start else None
        end = _parse_datetime(medication.end) if medication.end else None
        if medication.start and start is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.STRUCTURED_EHR,
                    field="medication_history.start",
                    message=f"Medication {medication.name} has an invalid start time.",
                )
            )
        if medication.end and end is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.STRUCTURED_EHR,
                    field="medication_history.end",
                    message=f"Medication {medication.name} has an invalid end time.",
                )
            )
        if start is not None and end is not None and end < start:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.STRUCTURED_EHR,
                    field="medication_history.period",
                    message=f"Medication {medication.name} ends before it starts.",
                )
            )

    for channel in record.time_series:
        previous: datetime | None = None
        for point in channel.points:
            timestamp = _parse_datetime(point.timestamp)
            if timestamp is None:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        modality=Modality.TIME_SERIES,
                        field="time_series.timestamp",
                        message=(
                            f"Time series channel {channel.name} has an invalid "
                            f"timestamp: {point.timestamp}."
                        ),
                    )
                )
                continue
            if previous is not None and timestamp < previous:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        modality=Modality.TIME_SERIES,
                        field="time_series.order",
                        message=f"Time series channel {channel.name} is not chronological.",
                    )
                )
                break
            previous = timestamp
    return issues


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
        normalized_name = _normalize_name(vital.name)
        if normalized_name in {"spo2", "oxygen-saturation"} and not 0 <= vital.value <= 100:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.VITALS,
                    field="vitals.SpO2",
                    message="SpO2 must be between 0 and 100.",
                )
            )
        if normalized_name in {"hr", "heart-rate"} and not 0 < vital.value < 260:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.VITALS,
                    field="vitals.HR",
                    message="Heart rate is outside a plausible clinical range.",
                )
            )
        if normalized_name in {"temperature", "temp"} and not 25 <= vital.value <= 45:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.VITALS,
                    field="vitals.temperature",
                    message="Temperature is outside a plausible clinical range.",
                )
            )
    return issues


def validate_text_structured_contradictions(
    record: SyntheticRecord,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    text = _document_text(record)
    if not text:
        return issues

    lactate_high = any(
        _normalize_name(lab.name) == "lactate"
        and isinstance(lab.value, int | float)
        and lab.reference_high is not None
        and lab.value > lab.reference_high
        for lab in record.labs
    )
    if lactate_high and re.search(r"\b(normal lactate|lactate (?:is )?normal)\b", text):
        issues.append(
            ValidationIssue(
                severity="error",
                modality=Modality.CLINICAL_TEXT,
                field="documents.lactate",
                message="Clinical text says lactate is normal but structured labs are high.",
            )
        )

    febrile_vital = any(
        _normalize_name(vital.name) in {"temperature", "temp"}
        and vital.value >= 38.0
        for vital in record.vitals
    )
    if febrile_vital and re.search(r"\b(afebrile|no fever|denies fever)\b", text):
        issues.append(
            ValidationIssue(
                severity="error",
                modality=Modality.CLINICAL_TEXT,
                field="documents.fever",
                message="Clinical text denies fever but structured temperature is febrile.",
            )
        )

    hypoxic_vital = any(
        _normalize_name(vital.name) in {"spo2", "oxygen-saturation"}
        and vital.value < 90
        for vital in record.vitals
    )
    if hypoxic_vital and re.search(r"\b(normal oxygenation|oxygenating well)\b", text):
        issues.append(
            ValidationIssue(
                severity="error",
                modality=Modality.CLINICAL_TEXT,
                field="documents.oxygenation",
                message="Clinical text says oxygenation is normal but SpO2 is hypoxic.",
            )
        )
    return issues


def _document_text(record: SyntheticRecord) -> str:
    return "\n".join(
        " ".join(part for part in [document.clean_text, document.messy_text] if part)
        for document in record.documents
    ).lower()


def _normalize_name(value: str) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", value.lower()))


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
