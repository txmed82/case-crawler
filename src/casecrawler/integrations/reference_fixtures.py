from __future__ import annotations

from casecrawler.integrations.huggingface import (
    REFERENCE_DATASETS,
    import_reference_rows,
)
from casecrawler.integrations.synthea import SyntheaAdapter
from casecrawler.models.synthetic import (
    ClinicalDocument,
    Code,
    ComplexityProfile,
    Encounter,
    LabObservation,
    MedicationStatement,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    TimeSeriesChannel,
    TimeSeriesPoint,
    ValidationReport,
    VitalObservation,
)
from casecrawler.storage.dataset_store import DatasetStore


FIXTURE_REFERENCE_KEYS = [
    "synthea_fhir",
    "clinical_notes_to_fhir",
    "medsynth_dialogue_note",
    "technetium_i",
    "synthclinicalnotes",
    "augmented_clinical_notes",
    "clinical_timeseries_reference",
    "synthchex_75k",
    "radiology_report_consistency",
    "synthetic_chest_xray_pneumonia",
]


def import_reference_fixture(
    reference_key: str,
    *,
    dataset_id: str,
    limit: int | None = None,
) -> list[SyntheticRecord]:
    if reference_key == "synthea_fhir":
        return _limit_records(_synthea_fixture_records(dataset_id), limit)
    if reference_key == "clinical_timeseries_reference":
        return _limit_records(_clinical_timeseries_fixture_records(dataset_id), limit)
    try:
        rows = _FIXTURE_ROWS[reference_key]
        spec = REFERENCE_DATASETS[reference_key]
    except KeyError as exc:
        choices = ", ".join(FIXTURE_REFERENCE_KEYS)
        raise KeyError(
            f"Unknown bundled reference fixture {reference_key!r}. "
            f"Choose from: {choices}."
        ) from exc
    return import_reference_rows(
        rows,
        dataset_id=dataset_id,
        reference_key=reference_key,
        limit=limit,
        spec=spec,
    )


def seed_recommended_reference_fixtures(
    store: DatasetStore,
    *,
    dataset_id: str,
    dataset_id_prefix: str | None = None,
    limit: int | None = None,
) -> dict:
    manifest = store.get_manifest(dataset_id)
    reference_keys = _string_list(manifest.metadata.get("recommended_reference_keys"))
    imported = []
    skipped = []
    unavailable = []
    prefix = dataset_id_prefix or f"{dataset_id}-fixture"
    for reference_key in reference_keys:
        if reference_key not in FIXTURE_REFERENCE_KEYS:
            unavailable.append(reference_key)
            continue
        existing_dataset_id = store.find_reference_dataset_id(
            [reference_key],
            exclude_dataset_id=dataset_id,
        )
        if existing_dataset_id:
            skipped.append(
                {
                    "reference_key": reference_key,
                    "dataset_id": existing_dataset_id,
                    "reason": "already_imported",
                }
            )
            continue
        fixture_dataset_id = f"{prefix}-{_dataset_id_token(reference_key)}"
        records = import_reference_fixture(
            reference_key,
            dataset_id=fixture_dataset_id,
            limit=limit,
        )
        for record in records:
            store.save_record(record)
        imported.append(
            {
                "reference_key": reference_key,
                "dataset_id": fixture_dataset_id,
                "record_count": len(records),
            }
        )
    return {
        "dataset_id": dataset_id,
        "recommended_reference_keys": reference_keys,
        "imported": imported,
        "skipped": skipped,
        "unavailable": unavailable,
    }


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _dataset_id_token(value: str) -> str:
    return value.lower().replace("_", "-").replace("/", "-")


def _limit_records(
    records: list[SyntheticRecord],
    limit: int | None,
) -> list[SyntheticRecord]:
    return records[:limit] if limit is not None else records


def _synthea_fixture_records(dataset_id: str) -> list[SyntheticRecord]:
    return [
        SyntheaAdapter().import_fhir_resources(
            _SYNTHEA_RESOURCES,
            dataset_id=dataset_id,
            source_ref={
                "path": "casecrawler-bundled-fixture",
                "format": "fhir_resources",
                "patient_id": "fixture-synthea-patient",
            },
        )
    ]


def _clinical_timeseries_fixture_records(dataset_id: str) -> list[SyntheticRecord]:
    return [
        SyntheticRecord(
            record_id="fixture-timeseries-icu-1",
            dataset_id=dataset_id,
            topic="sepsis",
            complexity=ComplexityProfile.COMPLEX,
            modalities=[
                Modality.STRUCTURED_EHR,
                Modality.CLINICAL_TEXT,
                Modality.LABS,
                Modality.VITALS,
                Modality.TIME_SERIES,
            ],
            patient=SyntheticPatient(
                patient_id="fixture-timeseries-patient-1",
                age=67,
                sex="female",
                demographics={"source": "casecrawler-bundled-fixture"},
            ),
            encounters=[
                Encounter(
                    encounter_id="fixture-timeseries-enc-1",
                    start="2026-01-01T00:00:00",
                    end="2026-01-01T06:00:00",
                    setting="icu",
                    reason="Sepsis with shock physiology",
                    diagnoses=[
                        Code(
                            system="http://snomed.info/sct",
                            code="91302008",
                            display="Sepsis",
                        )
                    ],
                )
            ],
            labs=[
                LabObservation(
                    name="Lactate",
                    loinc="2524-7",
                    value=4.1,
                    unit="mmol/L",
                    reference_low=0.5,
                    reference_high=2.2,
                    flag="H",
                    effective_time="2026-01-01T00:15:00",
                    specimen="blood",
                ),
                LabObservation(
                    name="Creatinine",
                    loinc="2160-0",
                    value=1.8,
                    unit="mg/dL",
                    reference_low=0.6,
                    reference_high=1.2,
                    flag="H",
                    effective_time="2026-01-01T00:15:00",
                    specimen="serum",
                ),
            ],
            vitals=[
                VitalObservation(
                    name="Heart rate",
                    value=122,
                    unit="/min",
                    effective_time="2026-01-01T00:00:00",
                ),
                VitalObservation(
                    name="Systolic blood pressure",
                    value=86,
                    unit="mmHg",
                    effective_time="2026-01-01T00:00:00",
                ),
                VitalObservation(
                    name="SpO2",
                    value=92,
                    unit="%",
                    effective_time="2026-01-01T00:00:00",
                ),
            ],
            medication_history=[
                MedicationStatement(
                    name="Norepinephrine",
                    rxnorm="7512",
                    dose="0.08 mcg/kg/min",
                    route="IV",
                    frequency="continuous",
                    status="active",
                    start="2026-01-01T00:30:00",
                ),
                MedicationStatement(
                    name="Ceftriaxone",
                    rxnorm="2193",
                    dose="2 g",
                    route="IV",
                    frequency="daily",
                    status="active",
                    start="2026-01-01T00:20:00",
                ),
            ],
            time_series=[
                TimeSeriesChannel(
                    name="heart_rate",
                    unit="/min",
                    generation_backend="casecrawler-reference-fixture",
                    sampling_rate_hz=None,
                    points=[
                        TimeSeriesPoint(
                            timestamp="2026-01-01T00:00:00",
                            values={"value": 122},
                        ),
                        TimeSeriesPoint(
                            timestamp="2026-01-01T01:00:00",
                            values={"value": 116},
                        ),
                        TimeSeriesPoint(
                            timestamp="2026-01-01T02:00:00",
                            values={"value": 109},
                        ),
                    ],
                ),
                TimeSeriesChannel(
                    name="lactate",
                    unit="mmol/L",
                    generation_backend="casecrawler-reference-fixture",
                    sampling_rate_hz=None,
                    points=[
                        TimeSeriesPoint(
                            timestamp="2026-01-01T00:15:00",
                            values={"value": 4.1},
                        ),
                        TimeSeriesPoint(
                            timestamp="2026-01-01T02:15:00",
                            values={"value": 3.2},
                        ),
                        TimeSeriesPoint(
                            timestamp="2026-01-01T04:15:00",
                            values={"value": 2.4},
                        ),
                    ],
                ),
            ],
            documents=[
                ClinicalDocument(
                    document_id="fixture-timeseries-nursing-1",
                    note_type="nursing_note",
                    author_role="nurse",
                    timestamp="2026-01-01T02:30:00",
                    clean_text=(
                        "Nursing note: heart rate improving from 122 to 109 "
                        "after fluids and norepinephrine; lactate downtrending."
                    ),
                    messy_text="nsg: HR 122->109, lactate 4.1->2.4, NE cont",
                    extracted_facts={
                        "time_series_channels": ["heart_rate", "lactate"],
                        "medications": ["Norepinephrine", "Ceftriaxone"],
                    },
                )
            ],
            provenance=Provenance(
                generator="casecrawler-bundled-reference-fixture",
                model=None,
                source_refs=[
                    {
                        "reference_key": "clinical_timeseries_reference",
                        "source": "casecrawler-bundled-fixture",
                    }
                ],
                created_at="2026-01-01T06:00:00",
            ),
            validation=ValidationReport(
                schema_score=1.0,
                clinical_consistency_score=0.95,
                privacy_score=1.0,
                utility_score=0.95,
                modality_alignment_score=None,
                approved=True,
            ),
            metadata={
                "reference_key": "clinical_timeseries_reference",
                "reference_dataset": (
                    "casecrawler-bundled-fixture:clinical_timeseries_reference"
                ),
                "reference_license": "synthetic-fixture",
                "reference_split": "fixture",
            },
        )
    ]


_FIXTURE_ROWS: dict[str, list[dict]] = {
    "clinical_notes_to_fhir": [
        {
            "exampleId": "fixture-fhir-1",
            "difficulty": "moderate",
            "scenario": "Emergency sepsis evaluation with abnormal lactate.",
            "note": "Patient: 67-year-old female with fever, hypotension, and sepsis.",
            "fhir_bundle": (
                '{"resourceType":"Bundle","type":"collection","entry":['
                '{"resource":{"resourceType":"Observation","id":"obs-lactate",'
                '"code":{"coding":[{"system":"http://loinc.org","code":"2524-7",'
                '"display":"Lactate"}],"text":"Lactate"},'
                '"valueQuantity":{"value":4.1,"unit":"mmol/L"},'
                '"effectiveDateTime":"2026-01-01T00:00:00",'
                '"referenceRange":[{"low":{"value":0.5},"high":{"value":2.2}}]}},'
                '{"resource":{"resourceType":"Observation","id":"obs-hr",'
                '"category":[{"coding":[{"code":"vital-signs"}]}],'
                '"code":{"text":"Heart rate"},'
                '"valueQuantity":{"value":118,"unit":"/min"},'
                '"effectiveDateTime":"2026-01-01T00:05:00"}},'
                '{"resource":{"resourceType":"MedicationStatement","id":"med-abx",'
                '"medicationCodeableConcept":{"text":"Ceftriaxone"},'
                '"status":"active","dosage":[{"route":{"text":"IV"},"text":"2 g daily"}]}},'
                '{"resource":{"resourceType":"Condition","id":"cond-sepsis",'
                '"code":{"coding":[{"system":"http://snomed.info/sct","code":"91302008",'
                '"display":"Sepsis"}],"text":"Sepsis"}}}]}'
            ),
            "valid": True,
        }
    ],
    "medsynth_dialogue_note": [
        {
            "Note": "SOAP Note: 61-year-old male with pneumonia treated with ceftriaxone.",
            "Dialogue": "[doctor] Any allergies? [patient] None. [doctor] We will start antibiotics.",
            "ICD10": "J18.9",
            "ICD10_desc": "Pneumonia, unspecified organism",
        }
    ],
    "technetium_i": [
        {
            "note_id": "fixture-technetium-1",
            "note_type": "discharge_summary",
            "text": (
                "DISCHARGE SUMMARY\nPatient Name: Jones, Mary\n"
                "Patient is a 72-year-old female admitted with heart failure."
            ),
            "phi_annotations": [
                {"entity_type": "NAME", "text": "Jones, Mary", "start": 32, "end": 43},
                {"entity_type": "AGE", "text": "72-year-old", "start": 55, "end": 66},
            ],
            "icd_codes": ["428.0"],
            "quality_score": 0.93,
        }
    ],
    "synthclinicalnotes": [
        {
            "ground_truth": "Progress Note: 54-year-old male with acute kidney injury.",
            "model_input": "Create an inpatient progress note for AKI.",
        }
    ],
    "augmented_clinical_notes": [
        {
            "idx": "fixture-augmented-1",
            "full_note": "Clinical note: 49-year-old female with diabetic ketoacidosis.",
            "conversation": "Patient reports polyuria and abdominal pain.",
            "summary": "DKA treated with fluids and insulin.",
        }
    ],
    "synthchex_75k": [
        {
            "label": "pneumonia",
            "image": None,
        }
    ],
    "radiology_report_consistency": [
        {
            "case_id": "fixture-rad-1",
            "study": "Chest radiograph",
            "modality": "XR",
            "report_excerpt": "Right lower lobe opacity concerning for pneumonia.",
            "imaging_findings": "right lower lobe opacity pneumonia",
            "expected_decision": "consistent",
            "consistency_issue": "none",
        }
    ],
    "synthetic_chest_xray_pneumonia": [
        {
            "label": "1",
            "image": None,
        }
    ],
}


_SYNTHEA_RESOURCES = [
    {
        "resourceType": "Patient",
        "id": "fixture-synthea-patient",
        "gender": "female",
        "birthDate": "1959-01-01",
    },
    {
        "resourceType": "Encounter",
        "id": "fixture-synthea-encounter",
        "period": {"start": "2026-01-01T00:00:00"},
        "reasonCode": [{"text": "sepsis"}],
    },
    {
        "resourceType": "Condition",
        "id": "fixture-synthea-condition",
        "code": {
            "coding": [
                {
                    "system": "http://snomed.info/sct",
                    "code": "91302008",
                    "display": "Sepsis",
                }
            ],
            "text": "Sepsis",
        },
    },
    {
        "resourceType": "Observation",
        "id": "fixture-synthea-lactate",
        "code": {"text": "Lactate"},
        "valueQuantity": {"value": 3.8, "unit": "mmol/L"},
        "effectiveDateTime": "2026-01-01T00:10:00",
        "referenceRange": [{"low": {"value": 0.5}, "high": {"value": 2.2}}],
    },
    {
        "resourceType": "Observation",
        "id": "fixture-synthea-hr",
        "category": [{"coding": [{"code": "vital-signs"}]}],
        "code": {"text": "Heart rate"},
        "valueQuantity": {"value": 116, "unit": "/min"},
        "effectiveDateTime": "2026-01-01T00:05:00",
    },
    {
        "resourceType": "MedicationStatement",
        "id": "fixture-synthea-med",
        "medicationCodeableConcept": {"text": "Ceftriaxone"},
        "status": "active",
        "dosage": [{"route": {"text": "IV"}, "text": "2 g daily"}],
    },
]
