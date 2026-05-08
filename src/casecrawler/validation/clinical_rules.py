from __future__ import annotations

import re
from datetime import datetime, timezone

from casecrawler.generation.imaging_templates import get_imaging_template
from casecrawler.models.synthetic import Modality, SyntheticRecord, ValidationIssue


def validate_temporal_consistency(record: SyntheticRecord) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    encounter_windows: list[tuple[datetime, datetime | None]] = []
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
        if start is not None and (
            (encounter.end is None and end is None) or (end is not None and end >= start)
        ):
            encounter_windows.append((start, end))

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

    for lab in record.labs:
        observed_at = _parse_datetime(lab.effective_time)
        if observed_at is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.LABS,
                    field="labs.effective_time",
                    message=f"Lab {lab.name} has an invalid effective time.",
                )
            )
            continue
        if _outside_encounter_windows(observed_at, encounter_windows):
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.LABS,
                    field="labs.effective_time",
                    message=f"Lab {lab.name} is timestamped outside all encounter windows.",
                )
            )

    for vital in record.vitals:
        observed_at = _parse_datetime(vital.effective_time)
        if observed_at is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.VITALS,
                    field="vitals.effective_time",
                    message=f"Vital {vital.name} has an invalid effective time.",
                )
            )
            continue
        if _outside_encounter_windows(observed_at, encounter_windows):
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.VITALS,
                    field="vitals.effective_time",
                    message=f"Vital {vital.name} is timestamped outside all encounter windows.",
                )
            )

    for document in record.documents:
        authored_at = _parse_datetime(document.timestamp)
        if authored_at is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.CLINICAL_TEXT,
                    field="documents.timestamp",
                    message=f"Document {document.document_id} has an invalid timestamp.",
                )
            )
            continue
        if _outside_encounter_windows(authored_at, encounter_windows):
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.CLINICAL_TEXT,
                    field="documents.timestamp",
                    message=(
                        f"Document {document.document_id} is timestamped outside "
                        "all encounter windows."
                    ),
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


def validate_time_series_trends(record: SyntheticRecord) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    lab_targets = {
        f"lab_{_slug(lab.name)}": _lab_trend_target(lab)
        for lab in record.labs
        if isinstance(lab.value, int | float)
    }
    for channel in record.time_series:
        first = _first_channel_value(channel)
        last = _last_channel_value(channel)
        if first is None or last is None:
            continue
        lab_target = lab_targets.get(channel.name)
        if lab_target is not None:
            direction = _trend_direction(first, last, lab_target)
            if direction == "away":
                issues.append(
                    ValidationIssue(
                        severity="error",
                        modality=Modality.TIME_SERIES,
                        field=f"time_series.{channel.name}.trend",
                        message=(
                            f"Time series channel {channel.name} trends away from "
                            "the structured lab reference range."
                        ),
                    )
                )
            continue
        if channel.name == "spo2" and first < 90 and last < first - 2:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.TIME_SERIES,
                    field="time_series.spo2.trend",
                    message="Hypoxic SpO2 time series declines from an already low baseline.",
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


def validate_medication_safety(record: SyntheticRecord) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    egfr = _first_numeric_lab(record, {"egfr", "estimated-gfr"})
    potassium = _first_numeric_lab(record, {"potassium", "k"})
    inr = _first_numeric_lab(record, {"inr", "international-normalized-ratio"})
    systolic_bp = _first_numeric_vital(
        record,
        {"sbp", "systolic-bp", "systolic-blood-pressure"},
    )

    if egfr is not None and egfr < 30 and _has_active_medication(record, {"metformin"}):
        issues.append(
            ValidationIssue(
                severity="error",
                modality=Modality.STRUCTURED_EHR,
                field="medication_history.metformin_renal",
                message="Active metformin conflicts with severe renal impairment.",
            )
        )

    hyperkalemia_medications = {
        "amiloride",
        "eplerenone",
        "lisinopril",
        "losartan",
        "potassium",
        "potassium-chloride",
        "spironolactone",
        "triamterene",
    }
    if (
        potassium is not None
        and potassium >= 5.5
        and _has_active_medication(record, hyperkalemia_medications)
    ):
        issues.append(
            ValidationIssue(
                severity="error",
                modality=Modality.STRUCTURED_EHR,
                field="medication_history.hyperkalemia",
                message="Active potassium-raising therapy conflicts with hyperkalemia.",
            )
        )

    if inr is not None and inr >= 4.0 and _has_active_medication(record, {"warfarin"}):
        issues.append(
            ValidationIssue(
                severity="error",
                modality=Modality.STRUCTURED_EHR,
                field="medication_history.warfarin_inr",
                message="Active warfarin conflicts with a supratherapeutic INR.",
            )
        )

    nitrate_medications = {
        "isosorbide",
        "isosorbide-dinitrate",
        "isosorbide-mononitrate",
        "nitroglycerin",
    }
    if (
        systolic_bp is not None
        and systolic_bp < 90
        and _has_active_medication(record, nitrate_medications)
    ):
        issues.append(
            ValidationIssue(
                severity="error",
                modality=Modality.STRUCTURED_EHR,
                field="medication_history.nitrate_hypotension",
                message="Active nitrate therapy conflicts with hypotension.",
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


def validate_document_extracted_fact_alignment(
    record: SyntheticRecord,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    lab_values = {
        _normalize_name(lab.name): float(lab.value)
        for lab in record.labs
        if isinstance(lab.value, int | float)
    }
    vital_values = {
        _normalize_name(vital.name): float(vital.value) for vital in record.vitals
    }
    medication_names = {_normalize_name(medication.name) for medication in record.medication_history}
    imaging_asset_ids = {asset.image_id for asset in record.imaging}
    imaging_labels = {
        _normalize_name(term)
        for asset in record.imaging
        for label in asset.labels
        for term in (label.display, label.code)
        if term
    }

    for document in record.documents:
        facts = document.extracted_facts
        for item in _fact_list(facts.get("lab_values")):
            name = _normalize_name(str(item.get("name", "")))
            value = item.get("value")
            if name not in lab_values or not isinstance(value, int | float):
                continue
            if _numeric_conflict(float(value), lab_values[name]):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        modality=Modality.CLINICAL_TEXT,
                        field="documents.extracted_facts.lab_values",
                        message=(
                            f"Document {document.document_id} extracted lab {name}="
                            f"{value} conflicts with structured value {lab_values[name]}."
                        ),
                    )
                )
        for item in _fact_list(facts.get("vital_values")):
            name = _normalize_name(str(item.get("name", "")))
            value = item.get("value")
            if name not in vital_values or not isinstance(value, int | float):
                continue
            if _numeric_conflict(float(value), vital_values[name]):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        modality=Modality.CLINICAL_TEXT,
                        field="documents.extracted_facts.vital_values",
                        message=(
                            f"Document {document.document_id} extracted vital {name}="
                            f"{value} conflicts with structured value {vital_values[name]}."
                        ),
                    )
                )
        extracted_medications = {
            _normalize_name(str(item))
            for item in facts.get("medications", [])
            if isinstance(item, str)
        }
        unsupported_medications = extracted_medications - medication_names
        if unsupported_medications and medication_names:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.CLINICAL_TEXT,
                    field="documents.extracted_facts.medications",
                    message=(
                        f"Document {document.document_id} extracted medication names "
                        f"not present in structured medication history: "
                        f"{', '.join(sorted(unsupported_medications))}."
                    ),
                )
            )
        extracted_image_ids = {
            str(item).strip()
            for item in facts.get("imaging_asset_ids", [])
            if isinstance(item, str) and item.strip()
        }
        unsupported_image_ids = extracted_image_ids - imaging_asset_ids
        if unsupported_image_ids:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.CLINICAL_TEXT,
                    field="documents.extracted_facts.imaging_asset_ids",
                    message=(
                        f"Document {document.document_id} extracted imaging asset IDs "
                        f"not present in structured imaging assets: "
                        f"{', '.join(sorted(unsupported_image_ids))}."
                    ),
                )
            )
        extracted_imaging_labels = {
            _normalize_name(str(item))
            for item in facts.get("imaging_labels", [])
            if isinstance(item, str)
        }
        unsupported_imaging_labels = extracted_imaging_labels - imaging_labels
        if unsupported_imaging_labels:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.CLINICAL_TEXT,
                    field="documents.extracted_facts.imaging_labels",
                    message=(
                        f"Document {document.document_id} extracted imaging labels "
                        f"not present in structured imaging labels: "
                        f"{', '.join(sorted(unsupported_imaging_labels))}."
                    ),
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


def validate_imaging_asset_clinical_shape(
    record: SyntheticRecord,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for asset in record.imaging:
        modality = asset.modality.upper().strip()
        body_region = asset.body_region.lower().strip()
        template = get_imaging_template(modality)
        if template is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.IMAGING,
                    field=f"imaging.{asset.image_id}.modality",
                    message=f"Imaging asset {asset.image_id} uses unsupported modality {asset.modality!r}.",
                )
            )
        elif body_region not in {region.lower() for region in template.valid_body_regions}:
            valid_regions = ", ".join(template.valid_body_regions)
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.IMAGING,
                    field=f"imaging.{asset.image_id}.body_region",
                    message=(
                        f"Imaging asset {asset.image_id} uses body region "
                        f"{asset.body_region!r}, which is not valid for {modality}. "
                        f"Expected one of: {valid_regions}."
                    ),
                )
            )
        if not asset.report_text.strip():
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.IMAGING,
                    field=f"imaging.{asset.image_id}.report_text",
                    message=f"Imaging asset {asset.image_id} has empty radiology report text.",
                )
            )
        if not asset.prompt.strip():
            issues.append(
                ValidationIssue(
                    severity="error",
                    modality=Modality.IMAGING,
                    field=f"imaging.{asset.image_id}.prompt",
                    message=f"Imaging asset {asset.image_id} has empty generation prompt.",
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


def _first_numeric_lab(record: SyntheticRecord, names: set[str]) -> float | None:
    for lab in record.labs:
        if _normalize_name(lab.name) in names and isinstance(lab.value, int | float):
            return float(lab.value)
    return None


def _has_active_medication(record: SyntheticRecord, names: set[str]) -> bool:
    inactive_statuses = {"completed", "entered-in-error", "stopped"}
    for medication in record.medication_history:
        if medication.status.lower() in inactive_statuses:
            continue
        normalized = _normalize_name(medication.name)
        if any(normalized == name or normalized.startswith(f"{name}-") for name in names):
            return True
    return False


def _fact_list(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _numeric_conflict(observed: float, expected: float) -> bool:
    tolerance = max(0.5, abs(expected) * 0.2)
    return abs(observed - expected) > tolerance


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


def _last_channel_value(channel) -> float | None:
    if not channel.points:
        return None
    values = channel.points[-1].values
    candidate = values.get("value")
    if isinstance(candidate, int | float):
        return float(candidate)
    if len(values) == 1:
        only_value = next(iter(values.values()))
        if isinstance(only_value, int | float):
            return float(only_value)
    return None


def _lab_trend_target(lab) -> float | None:
    value = float(lab.value)
    if lab.reference_high is not None and value > lab.reference_high:
        return float(lab.reference_high)
    if lab.reference_low is not None and value < lab.reference_low:
        return float(lab.reference_low)
    return None


def _trend_direction(first: float, last: float, target: float) -> str:
    initial_distance = abs(first - target)
    final_distance = abs(last - target)
    if final_distance > initial_distance + max(0.5, initial_distance * 0.1):
        return "away"
    return "toward_or_stable"


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _outside_encounter_windows(
    timestamp: datetime,
    encounter_windows: list[tuple[datetime, datetime | None]],
) -> bool:
    if not encounter_windows:
        return False
    return not any(
        start <= timestamp and (end is None or timestamp <= end)
        for start, end in encounter_windows
    )


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
