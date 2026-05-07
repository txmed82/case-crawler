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


def validate_time_series_waveforms(record: SyntheticRecord) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for channel in record.time_series:
        if channel.sampling_rate_hz is not None and channel.sampling_rate_hz <= 0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.TIME_SERIES,
                    field="time_series.sampling_rate_hz",
                    message=f"Time series channel {channel.name} has a non-positive sampling rate.",
                )
            )
        if channel.name == "ecg_lead_ii":
            issues.extend(_validate_ecg_channel(channel))
        if channel.name == "pleth":
            issues.extend(_validate_pleth_channel(channel))
    return issues


def validate_time_series_structured_alignment(
    record: SyntheticRecord,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    vital_targets = {
        "heart_rate": _first_numeric_vital(record, {"hr", "heart-rate"}),
        "systolic_bp": _first_numeric_vital(
            record,
            {"sbp", "systolic-bp", "systolic-blood-pressure"},
        ),
        "spo2": _first_numeric_vital(record, {"spo2", "oxygen-saturation"}),
    }
    lab_targets = {
        f"lab_{_slug(lab.name)}": float(lab.value)
        for lab in record.labs
        if isinstance(lab.value, int | float)
    }
    for channel in record.time_series:
        observed = _first_channel_value(channel)
        if observed is None:
            continue
        target = vital_targets.get(channel.name)
        field = f"time_series.{channel.name}.alignment"
        tolerance = max(10.0, abs(target or 0) * 0.35)
        if target is None and channel.name in lab_targets:
            target = lab_targets[channel.name]
            tolerance = max(1.0, abs(target) * 0.5)
        if target is None:
            continue
        if abs(observed - target) > tolerance:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.TIME_SERIES,
                    field=field,
                    message=(
                        f"Time series channel {channel.name} starts at {observed}, "
                        f"which conflicts with structured value {target}."
                    ),
                )
            )
    return issues


def validate_lab_flags(record: SyntheticRecord) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for lab in record.labs:
        if (
            lab.reference_low is not None
            and lab.reference_high is not None
            and lab.reference_low > lab.reference_high
        ):
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.LABS,
                    field="labs.reference_range",
                    message=f"{lab.name} reference_low is greater than reference_high.",
                )
            )
            continue
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
            if lab.flag:
                normalized_flag = lab.flag.lower()
                if lab.value < lab.reference_low and normalized_flag.startswith("h"):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            modality=Modality.LABS,
                            field="labs.flag_direction",
                            message=f"{lab.name} is below range but flagged high.",
                        )
                    )
                if lab.value > lab.reference_high and normalized_flag.startswith("l"):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            modality=Modality.LABS,
                            field="labs.flag_direction",
                            message=f"{lab.name} is above range but flagged low.",
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
        if normalized_name in {"respiratory-rate", "rr"} and not 0 < vital.value < 80:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.VITALS,
                    field="vitals.respiratory_rate",
                    message="Respiratory rate is outside a plausible clinical range.",
                )
            )
        if normalized_name in {
            "sbp",
            "systolic-bp",
            "systolic-blood-pressure",
        } and not 40 <= vital.value <= 300:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.VITALS,
                    field="vitals.SBP",
                    message="Systolic blood pressure is outside a plausible clinical range.",
                )
            )
        if normalized_name in {
            "dbp",
            "diastolic-bp",
            "diastolic-blood-pressure",
        } and not 20 <= vital.value <= 180:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.VITALS,
                    field="vitals.DBP",
                    message="Diastolic blood pressure is outside a plausible clinical range.",
                )
            )
    return issues


def validate_medication_history(record: SyntheticRecord) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    valid_statuses = {
        "active",
        "completed",
        "entered-in-error",
        "intended",
        "on-hold",
        "stopped",
        "unknown",
    }
    valid_routes = {
        "im",
        "inhaled",
        "intranasal",
        "iv",
        "nebulized",
        "oral",
        "po",
        "subcutaneous",
        "sublingual",
        "topical",
    }
    for medication in record.medication_history:
        if not medication.name.strip():
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.STRUCTURED_EHR,
                    field="medication_history.name",
                    message="Medication statement has an empty medication name.",
                )
            )
        normalized_status = medication.status.lower()
        if normalized_status not in valid_statuses:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.STRUCTURED_EHR,
                    field="medication_history.status",
                    message=f"Medication {medication.name or '<empty>'} has an invalid status.",
                )
            )
        if medication.route and medication.route.lower() not in valid_routes:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.STRUCTURED_EHR,
                    field="medication_history.route",
                    message=f"Medication {medication.name or '<empty>'} has an unsupported route.",
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


def validate_radiology_document_alignment(
    record: SyntheticRecord,
) -> list[ValidationIssue]:
    if not record.imaging:
        return []
    radiology_text = " ".join(
        " ".join(part for part in [document.clean_text, document.messy_text] if part)
        for document in record.documents
        if document.note_type == "radiology_report"
    )
    if not radiology_text:
        return []
    issues: list[ValidationIssue] = []
    for asset in record.imaging:
        for label in asset.labels:
            label_terms = _label_terms(label.display, label.code)
            if not any(_contains_term(radiology_text, term) for term in label_terms):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        modality=Modality.CLINICAL_TEXT,
                        field="documents.radiology_report",
                        message=(
                            f"Radiology document does not support imaging label "
                            f"{label.display!r} for asset {asset.image_id}."
                        ),
                    )
                )
                continue
            if any(_contains_negated_term(radiology_text, term) for term in label_terms):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        modality=Modality.CLINICAL_TEXT,
                        field="documents.radiology_report",
                        message=(
                            f"Radiology document negates imaging label "
                            f"{label.display!r} for asset {asset.image_id}."
                        ),
                    )
                )
    return issues


def _document_text(record: SyntheticRecord) -> str:
    return "\n".join(
        " ".join(part for part in [document.clean_text, document.messy_text] if part)
        for document in record.documents
    ).lower()


def _label_terms(display: str, code: str) -> set[str]:
    terms = {display.lower(), code.replace("_", " ").lower()}
    return {term for term in terms if term}


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text.lower()) is not None


def _contains_negated_term(text: str, term: str) -> bool:
    lowered = text.lower()
    return any(
        re.search(rf"\b{negation}\s+(?:\w+\s+){{0,3}}{re.escape(term)}\b", lowered)
        for negation in {"absent", "negative for", "no", "without"}
    )


def _normalize_name(value: str) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", value.lower()))


def _slug(value: str) -> str:
    return re.sub(r"\W+", "_", value.lower()).strip("_")


def _first_numeric_vital(record: SyntheticRecord, names: set[str]) -> float | None:
    for vital in record.vitals:
        if _normalize_name(vital.name) in names:
            return float(vital.value)
    return None


def _first_channel_value(channel) -> float | None:
    if not channel.points:
        return None
    values = channel.points[0].values
    candidate = values.get("value")
    if isinstance(candidate, int | float):
        return float(candidate)
    if len(values) == 1:
        only_value = next(iter(values.values()))
        if isinstance(only_value, int | float):
            return float(only_value)
    return None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _validate_ecg_channel(channel) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if channel.sampling_rate_hz is None or not 50 <= channel.sampling_rate_hz <= 500:
        issues.append(
            ValidationIssue(
                severity="error",
                modality=Modality.TIME_SERIES,
                field="time_series.sampling_rate_hz",
                message="ECG waveform sampling rate must be between 50 and 500 Hz.",
            )
        )
    for point in channel.points:
        millivolts = point.values.get("millivolts")
        if isinstance(millivolts, int | float) and not -5 <= millivolts <= 5:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.TIME_SERIES,
                    field="time_series.ecg_lead_ii.millivolts",
                    message="ECG lead II millivolts are outside a plausible synthetic range.",
                )
            )
            break
    issues.extend(_validate_phase_values(channel, "time_series.ecg_lead_ii.phase"))
    return issues


def _validate_pleth_channel(channel) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if channel.sampling_rate_hz is None or not 10 <= channel.sampling_rate_hz <= 250:
        issues.append(
            ValidationIssue(
                severity="error",
                modality=Modality.TIME_SERIES,
                field="time_series.sampling_rate_hz",
                message="Pleth waveform sampling rate must be between 10 and 250 Hz.",
            )
        )
    for point in channel.points:
        amplitude = point.values.get("amplitude")
        if isinstance(amplitude, int | float) and not 0 <= amplitude <= 2:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.TIME_SERIES,
                    field="time_series.pleth.amplitude",
                    message="Pleth amplitude is outside a plausible normalized range.",
                )
            )
            break
    issues.extend(_validate_phase_values(channel, "time_series.pleth.phase"))
    return issues


def _validate_phase_values(channel, field: str) -> list[ValidationIssue]:
    for point in channel.points:
        phase = point.values.get("phase")
        if isinstance(phase, int | float) and not 0 <= phase <= 1:
            return [
                ValidationIssue(
                    severity="error",
                    modality=Modality.TIME_SERIES,
                    field=field,
                    message=f"Waveform phase for channel {channel.name} must be between 0 and 1.",
                )
            ]
    return []
