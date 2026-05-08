from casecrawler.models.synthetic import (
    ClinicalDocument,
    Code,
    Encounter,
    ImagingAsset,
    ComplexityProfile,
    LabObservation,
    MedicationStatement,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    TimeSeriesChannel,
    TimeSeriesPoint,
    VitalObservation,
)
from casecrawler.validation.synthetic_validator import SyntheticValidator


def _record(**overrides):
    data = {
        "record_id": "rec-1",
        "dataset_id": "ds-1",
        "topic": "sepsis",
        "complexity": ComplexityProfile.MODERATE,
        "modalities": [Modality.LABS, Modality.VITALS],
        "patient": SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        "encounters": [],
        "labs": [
            LabObservation(
                name="Lactate",
                value=4.8,
                unit="mmol/L",
                reference_low=0.5,
                reference_high=2.0,
                flag="critical",
                effective_time="2026-05-06T08:30:00",
            )
        ],
        "vitals": [
            VitalObservation(
                name="HR",
                value=118,
                unit="/min",
                effective_time="2026-05-06T08:00:00",
            )
        ],
        "provenance": Provenance(
            generator="unit-test",
            created_at="2026-05-06T09:00:00",
        ),
    }
    data.update(overrides)
    return SyntheticRecord(**data)


def test_validator_approves_plausible_record():
    report = SyntheticValidator().validate(_record())

    assert report.approved is True
    assert report.clinical_consistency_score >= 0.8


def test_validator_rejects_missing_lab_flag():
    bad = _record(
        labs=[
            LabObservation(
                name="Lactate",
                value=4.8,
                unit="mmol/L",
                reference_low=0.5,
                reference_high=2.0,
                flag=None,
                effective_time="2026-05-06T08:30:00",
            )
        ]
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "labs.flag" for issue in report.issues)


def test_validator_rejects_invalid_lab_reference_ranges_and_flag_direction():
    bad = _record(
        labs=[
            LabObservation(
                name="Sodium",
                value=130,
                unit="mmol/L",
                reference_low=145,
                reference_high=135,
                flag="H",
                effective_time="2026-05-06T08:30:00",
            ),
            LabObservation(
                name="Potassium",
                value=2.8,
                unit="mmol/L",
                reference_low=3.5,
                reference_high=5.1,
                flag="H",
                effective_time="2026-05-06T08:35:00",
            ),
            LabObservation(
                name="Creatinine",
                value=2.4,
                unit="mg/dL",
                reference_low=0.6,
                reference_high=1.3,
                flag="L",
                effective_time="2026-05-06T08:40:00",
            ),
        ]
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "labs.reference_range" for issue in report.issues)
    assert sum(issue.field == "labs.flag_direction" for issue in report.issues) == 2


def test_validator_rejects_known_lab_and_vital_unit_conflicts():
    bad = _record(
        labs=[
            LabObservation(
                name="Lactate",
                value=4.8,
                unit="mg/dL",
                reference_low=0.5,
                reference_high=2.0,
                flag="H",
                effective_time="2026-05-06T08:30:00",
            )
        ],
        vitals=[
            VitalObservation(
                name="SpO2",
                value=88,
                unit="mmHg",
                effective_time="2026-05-06T08:00:00",
            )
        ],
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "labs.unit" for issue in report.issues)
    assert any(issue.field == "vitals.unit" for issue in report.issues)


def test_validator_rejects_implausible_respiratory_rate_and_blood_pressure():
    bad = _record(
        vitals=[
            VitalObservation(
                name="Respiratory rate",
                value=90,
                unit="/min",
                effective_time="2026-05-06T08:00:00",
            ),
            VitalObservation(
                name="SBP",
                value=310,
                unit="mmHg",
                effective_time="2026-05-06T08:01:00",
            ),
            VitalObservation(
                name="DBP",
                value=-5,
                unit="mmHg",
                effective_time="2026-05-06T08:02:00",
            ),
        ]
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "vitals.respiratory_rate" for issue in report.issues)
    assert any(issue.field == "vitals.SBP" for issue in report.issues)
    assert any(issue.field == "vitals.DBP" for issue in report.issues)


def test_validator_rejects_invalid_medication_history_entries():
    bad = _record(
        medication_history=[
            MedicationStatement(
                name="",
                route="intravenous",
                status="active",
                start="2026-05-06",
            ),
            MedicationStatement(
                name="Ceftriaxone",
                route="telepathy",
                status="dispensed-maybe",
                start="2026-05-06",
            ),
        ]
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "medication_history.name" for issue in report.issues)
    assert any(issue.field == "medication_history.route" for issue in report.issues)
    assert any(issue.field == "medication_history.status" for issue in report.issues)


def test_validator_rejects_medication_lab_safety_conflicts():
    bad = _record(
        labs=[
            LabObservation(
                name="eGFR",
                value=22,
                unit="mL/min/1.73m2",
                reference_low=60,
                reference_high=120,
                flag="L",
                effective_time="2026-05-06T08:30:00",
            ),
            LabObservation(
                name="Potassium",
                value=6.2,
                unit="mmol/L",
                reference_low=3.5,
                reference_high=5.1,
                flag="H",
                effective_time="2026-05-06T08:35:00",
            ),
            LabObservation(
                name="INR",
                value=5.4,
                unit="ratio",
                reference_low=0.8,
                reference_high=1.2,
                flag="H",
                effective_time="2026-05-06T08:40:00",
            ),
        ],
        medication_history=[
            MedicationStatement(
                name="Metformin",
                dose="1000 mg",
                route="oral",
                status="active",
                start="2026-05-01",
            ),
            MedicationStatement(
                name="Potassium chloride",
                dose="20 mEq",
                route="oral",
                status="active",
                start="2026-05-01",
            ),
            MedicationStatement(
                name="Warfarin",
                dose="5 mg",
                route="oral",
                status="active",
                start="2026-05-01",
            ),
        ],
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "medication_history.metformin_renal" for issue in report.issues)
    assert any(issue.field == "medication_history.hyperkalemia" for issue in report.issues)
    assert any(issue.field == "medication_history.warfarin_inr" for issue in report.issues)


def test_validator_rejects_medication_vital_safety_conflicts():
    bad = _record(
        vitals=[
            VitalObservation(
                name="SBP",
                value=82,
                unit="mmHg",
                effective_time="2026-05-06T08:00:00",
            )
        ],
        medication_history=[
            MedicationStatement(
                name="Nitroglycerin",
                dose="0.4 mg",
                route="sublingual",
                status="active",
                start="2026-05-06",
            )
        ],
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "medication_history.nitrate_hypotension" for issue in report.issues)


def test_validator_rejects_lab_time_series_trending_away_from_reference_range():
    bad = _record(
        labs=[
            LabObservation(
                name="Lactate",
                value=4.8,
                unit="mmol/L",
                reference_low=0.5,
                reference_high=2.0,
                flag="H",
                effective_time="2026-05-06T08:30:00",
            ),
            LabObservation(
                name="Sodium",
                value=128,
                unit="mmol/L",
                reference_low=136,
                reference_high=145,
                flag="L",
                effective_time="2026-05-06T08:35:00",
            ),
        ],
        time_series=[
            TimeSeriesChannel(
                name="lab_lactate",
                unit="mmol/L",
                points=[
                    TimeSeriesPoint(
                        timestamp="2026-05-06T08:30:00",
                        values={"value": 4.8},
                    ),
                    TimeSeriesPoint(
                        timestamp="2026-05-06T09:30:00",
                        values={"value": 6.1},
                    ),
                ],
            ),
            TimeSeriesChannel(
                name="lab_sodium",
                unit="mmol/L",
                points=[
                    TimeSeriesPoint(
                        timestamp="2026-05-06T08:35:00",
                        values={"value": 128},
                    ),
                    TimeSeriesPoint(
                        timestamp="2026-05-06T09:35:00",
                        values={"value": 123},
                    ),
                ],
            ),
        ],
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "time_series.lab_lactate.trend" for issue in report.issues)
    assert any(issue.field == "time_series.lab_sodium.trend" for issue in report.issues)


def test_validator_rejects_hypoxic_spo2_time_series_trending_down():
    bad = _record(
        vitals=[
            VitalObservation(
                name="SpO2",
                value=88,
                unit="%",
                effective_time="2026-05-06T08:00:00",
            )
        ],
        time_series=[
            TimeSeriesChannel(
                name="spo2",
                unit="%",
                points=[
                    TimeSeriesPoint(
                        timestamp="2026-05-06T08:00:00",
                        values={"value": 88},
                    ),
                    TimeSeriesPoint(
                        timestamp="2026-05-06T09:00:00",
                        values={"value": 82},
                    ),
                ],
            )
        ],
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "time_series.spo2.trend" for issue in report.issues)


def test_validator_rejects_document_extracted_fact_conflicts():
    bad = _record(
        documents=[
            ClinicalDocument(
                document_id="doc-facts",
                note_type="progress_note",
                author_role="physician",
                timestamp="2026-05-06T09:00:00",
                clean_text="Progress note with extracted facts.",
                extracted_facts={
                    "lab_values": [
                        {
                            "name": "Lactate",
                            "value": 1.1,
                            "unit": "mmol/L",
                        }
                    ],
                    "vital_values": [
                        {
                            "name": "HR",
                            "value": 72,
                            "unit": "/min",
                        }
                    ],
                    "medications": ["Vancomycin"],
                },
            )
        ],
        medication_history=[
            MedicationStatement(
                name="Ceftriaxone",
                route="iv",
                status="active",
            )
        ],
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "documents.extracted_facts.lab_values" for issue in report.issues)
    assert any(issue.field == "documents.extracted_facts.vital_values" for issue in report.issues)
    assert any(issue.field == "documents.extracted_facts.medications" for issue in report.issues)


def test_validator_rejects_document_extracted_procedure_fact_conflicts():
    bad = _record(
        encounters=[
            Encounter(
                encounter_id="enc-1",
                start="2026-05-06T08:00:00",
                setting="inpatient",
                reason="sepsis",
                procedures=[
                    Code(
                        system="synthetic",
                        code="central_line",
                        display="Central venous catheter placement",
                    )
                ],
            )
        ],
        documents=[
            ClinicalDocument(
                document_id="doc-procedure-facts",
                note_type="progress_note",
                author_role="physician",
                timestamp="2026-05-06T09:00:00",
                clean_text="Progress note with procedure facts.",
                extracted_facts={
                    "procedures": [
                        "Central venous catheter placement",
                        "Exploratory laparotomy",
                    ],
                    "procedure_details": [
                        {
                            "encounter_id": "enc-1",
                            "system": "synthetic",
                            "code": "central_line",
                            "display": "Central venous catheter placement",
                        },
                        {
                            "encounter_id": "enc-1",
                            "system": "synthetic",
                            "code": "laparotomy",
                            "display": "Exploratory laparotomy",
                        },
                    ],
                },
            )
        ],
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "documents.extracted_facts.procedures" for issue in report.issues)
    assert any(
        issue.field == "documents.extracted_facts.procedure_details"
        for issue in report.issues
    )


def test_validator_rejects_document_extracted_imaging_fact_conflicts():
    bad = _record(
        modalities=[Modality.CLINICAL_TEXT, Modality.IMAGING],
        imaging=[
            ImagingAsset(
                image_id="img-1",
                modality="XR",
                body_region="chest",
                prompt="portable chest x-ray with right lower lobe pneumonia",
                report_text="Right lower lobe pneumonia.",
                labels=[
                    Code(
                        system="synthetic",
                        code="pneumonia",
                        display="Pneumonia",
                    )
                ],
                generation_backend="placeholder",
            )
        ],
        documents=[
            ClinicalDocument(
                document_id="doc-rad",
                note_type="radiology_report",
                author_role="radiologist",
                timestamp="2026-05-06T09:00:00",
                clean_text="Radiology report confirms pneumonia.",
                extracted_facts={
                    "imaging_asset_ids": ["img-1", "img-missing"],
                    "imaging_labels": ["Pneumonia", "Appendicitis"],
                },
            )
        ],
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(
        issue.field == "documents.extracted_facts.imaging_asset_ids"
        for issue in report.issues
    )
    assert any(
        issue.field == "documents.extracted_facts.imaging_labels"
        for issue in report.issues
    )


def test_validator_rejects_phi_like_text():
    bad = _record(metadata={"free_text": "Call patient at 555-123-4567 tomorrow."})

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "privacy" for issue in report.issues)


def test_validator_rejects_common_phi_like_identifiers():
    bad = _record(
        documents=[
            ClinicalDocument(
                document_id="doc-phi",
                note_type="ed_note",
                author_role="physician",
                timestamp="2026-05-06T09:00:00",
                clean_text=(
                    "MRN: 12345678. DOB 01/02/1960. "
                    "Patient lives at 123 Main Street."
                ),
            )
        ]
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    messages = " ".join(issue.message for issue in report.issues if issue.field == "privacy")
    assert "medical record number" in messages
    assert "date of birth" in messages
    assert "street address" in messages


class FakeImageAlignmentValidator:
    def __init__(self, score: float):
        self._score = score

    def score(self, asset: ImagingAsset) -> float:
        return self._score


def test_validator_records_modality_alignment_score_for_images(tmp_path):
    image_path = tmp_path / "xray.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic image bytes")
    record = _record(
        modalities=[Modality.IMAGING],
        imaging=[
            ImagingAsset(
                image_id="img-1",
                modality="XR",
                body_region="chest",
                prompt="portable chest x-ray pulmonary edema",
                file_path=str(image_path),
                report_text="portable chest x-ray pulmonary edema",
                generation_backend="unit-test",
            )
        ],
    )

    report = SyntheticValidator(
        image_alignment_validator=FakeImageAlignmentValidator(0.9)
    ).validate(record)

    assert report.approved is True
    assert report.modality_alignment_score == 0.9


def test_validator_rejects_low_image_alignment():
    record = _record(
        modalities=[Modality.IMAGING],
        imaging=[
            ImagingAsset(
                image_id="img-1",
                modality="XR",
                body_region="chest",
                prompt="portable chest x-ray pulmonary edema",
                file_path="xray.png",
                report_text="unrelated report",
                generation_backend="unit-test",
            )
        ],
    )

    report = SyntheticValidator(
        image_alignment_validator=FakeImageAlignmentValidator(0.2)
    ).validate(record)

    assert report.approved is False
    assert any(issue.field == "imaging.alignment" for issue in report.issues)


def test_validator_rejects_missing_artifact_generation_backend():
    bad = _record(
        modalities=[Modality.TIME_SERIES, Modality.IMAGING],
        time_series=[
            TimeSeriesChannel(
                name="heart_rate",
                unit="/min",
                generation_backend=" ",
                points=[
                    TimeSeriesPoint(
                        timestamp="2026-05-06T10:00:00",
                        values={"value": 110},
                    )
                ],
            )
        ],
        imaging=[
            ImagingAsset(
                image_id="img-1",
                modality="XR",
                body_region="chest",
                prompt="portable chest x-ray pulmonary edema",
                file_path="xray.png",
                report_text="portable chest x-ray pulmonary edema",
                generation_backend="",
            )
        ],
    )

    report = SyntheticValidator(
        image_alignment_validator=FakeImageAlignmentValidator(0.9)
    ).validate(bad)

    assert report.approved is False
    assert any(
        issue.field == "time_series.heart_rate.generation_backend"
        for issue in report.issues
    )
    assert any(
        issue.field == "imaging.img-1.generation_backend"
        for issue in report.issues
    )


def test_validator_rejects_missing_generated_image_file():
    record = _record(
        modalities=[Modality.IMAGING],
        imaging=[
            ImagingAsset(
                image_id="img-1",
                modality="XR",
                body_region="chest",
                prompt="portable chest x-ray pulmonary edema",
                file_path="missing-xray.png",
                report_text="portable chest x-ray pulmonary edema",
                generation_backend="diffusers:model",
            )
        ],
    )

    report = SyntheticValidator(
        image_alignment_validator=FakeImageAlignmentValidator(0.9)
    ).validate(record)

    assert report.approved is False
    assert any(issue.field == "imaging.img-1.file_path" for issue in report.issues)


def test_validator_rejects_empty_or_unsupported_generated_image_file(tmp_path):
    empty_image = tmp_path / "empty.gif"
    empty_image.write_bytes(b"")
    record = _record(
        modalities=[Modality.IMAGING],
        imaging=[
            ImagingAsset(
                image_id="img-1",
                modality="XR",
                body_region="chest",
                prompt="portable chest x-ray pulmonary edema",
                file_path=str(empty_image),
                report_text="portable chest x-ray pulmonary edema",
                generation_backend="diffusers:model",
            )
        ],
    )

    report = SyntheticValidator(
        image_alignment_validator=FakeImageAlignmentValidator(0.9)
    ).validate(record)

    assert report.approved is False
    assert any(issue.field == "imaging.img-1.file_size" for issue in report.issues)
    assert any(issue.field == "imaging.img-1.file_format" for issue in report.issues)


def test_validator_rejects_inconsistent_radiology_labels():
    record = _record(
        modalities=[Modality.IMAGING],
        imaging=[
            ImagingAsset(
                image_id="img-1",
                modality="XR",
                body_region="chest",
                prompt="portable chest x-ray pulmonary edema",
                file_path="xray.png",
                report_text="Portable chest radiograph without pneumothorax.",
                labels=[
                    Code(
                        system="synthetic",
                        code="pneumothorax",
                        display="Pneumothorax",
                    )
                ],
                generation_backend="unit-test",
            )
        ],
    )

    report = SyntheticValidator(
        image_alignment_validator=FakeImageAlignmentValidator(0.9)
    ).validate(record)

    assert report.approved is False
    assert any(issue.field == "imaging.img-1.labels" for issue in report.issues)


def test_validator_rejects_invalid_imaging_modality_region_and_report_text():
    record = _record(
        modalities=[Modality.IMAGING],
        imaging=[
            ImagingAsset(
                image_id="img-1",
                modality="XR",
                body_region="brain",
                prompt="brain x-ray showing stroke",
                file_path="brain-xray.png",
                report_text="   ",
                generation_backend="unit-test",
            ),
            ImagingAsset(
                image_id="img-2",
                modality="PET",
                body_region="whole_body",
                prompt="whole body PET",
                file_path="pet.png",
                report_text="Synthetic PET report.",
                generation_backend="unit-test",
            ),
        ],
    )

    report = SyntheticValidator(
        image_alignment_validator=FakeImageAlignmentValidator(0.9)
    ).validate(record)

    assert report.approved is False
    assert any(issue.field == "imaging.img-1.body_region" for issue in report.issues)
    assert any(issue.field == "imaging.img-1.report_text" for issue in report.issues)
    assert any(issue.field == "imaging.img-2.modality" for issue in report.issues)


def test_validator_rejects_radiology_document_that_negates_imaging_label():
    record = _record(
        modalities=[Modality.IMAGING, Modality.CLINICAL_TEXT],
        imaging=[
            ImagingAsset(
                image_id="img-1",
                modality="CT",
                body_region="abdomen",
                prompt="CT abdomen appendicitis",
                file_path="appendicitis.png",
                report_text="CT abdomen demonstrates appendicitis.",
                labels=[
                    Code(
                        system="synthetic",
                        code="appendicitis",
                        display="Appendicitis",
                    )
                ],
                generation_backend="unit-test",
            )
        ],
        documents=[
            ClinicalDocument(
                document_id="doc-rad",
                note_type="radiology_report",
                author_role="radiologist",
                timestamp="2026-05-06T09:00:00",
                clean_text="CT abdomen without appendicitis.",
            )
        ],
    )

    report = SyntheticValidator(
        image_alignment_validator=FakeImageAlignmentValidator(0.9)
    ).validate(record)

    assert report.approved is False
    assert any(issue.field == "documents.radiology_report" for issue in report.issues)


def test_validator_rejects_radiology_document_missing_imaging_label_evidence():
    record = _record(
        modalities=[Modality.IMAGING, Modality.CLINICAL_TEXT],
        imaging=[
            ImagingAsset(
                image_id="img-1",
                modality="XR",
                body_region="chest",
                prompt="portable chest x-ray pulmonary edema",
                file_path="xray.png",
                report_text="portable chest x-ray pulmonary edema",
                labels=[
                    Code(
                        system="synthetic",
                        code="pulmonary_edema",
                        display="Pulmonary edema",
                    )
                ],
                generation_backend="unit-test",
            )
        ],
        documents=[
            ClinicalDocument(
                document_id="doc-rad",
                note_type="radiology_report",
                author_role="radiologist",
                timestamp="2026-05-06T09:00:00",
                clean_text="Portable chest radiograph reviewed.",
            )
        ],
    )

    report = SyntheticValidator(
        image_alignment_validator=FakeImageAlignmentValidator(0.9)
    ).validate(record)

    assert report.approved is False
    assert any(issue.field == "documents.radiology_report" for issue in report.issues)


def test_validator_scans_nested_record_fields_for_phi():
    bad = _record(
        patient=SyntheticPatient(
            patient_id="pat-1",
            age=64,
            sex="male",
            demographics={"contact": {"email": "patient@example.com"}},
        )
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert report.clinical_consistency_score == 1.0
    assert any(issue.field == "privacy" for issue in report.issues)


def test_validator_rejects_encounter_and_medication_temporal_inversions():
    bad = _record(
        encounters=[
            Encounter(
                encounter_id="enc-1",
                start="2026-05-06T12:00:00",
                end="2026-05-06T10:00:00",
                setting="ed",
                reason="fever",
            )
        ],
        medication_history=[
            MedicationStatement(
                name="Ceftriaxone",
                status="completed",
                start="2026-05-07",
                end="2026-05-06",
            )
        ],
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "encounters.period" for issue in report.issues)
    assert any(issue.field == "medication_history.period" for issue in report.issues)


def test_validator_rejects_artifacts_outside_encounter_window():
    bad = _record(
        encounters=[
            Encounter(
                encounter_id="enc-1",
                start="2026-05-06T08:00:00",
                end="2026-05-06T12:00:00",
                setting="ed",
                reason="fever",
            )
        ],
        labs=[
            LabObservation(
                name="Lactate",
                value=3.2,
                unit="mmol/L",
                reference_low=0.5,
                reference_high=2.0,
                flag="high",
                effective_time="2026-05-06T14:00:00",
            )
        ],
        vitals=[
            VitalObservation(
                name="Temperature",
                value=38.4,
                unit="C",
                effective_time="2026-05-06T07:30:00",
            )
        ],
        documents=[
            ClinicalDocument(
                document_id="doc-1",
                note_type="physician_note",
                author_role="physician",
                timestamp="2026-05-06T13:00:00",
                clean_text="Fever evaluation.",
            )
        ],
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "labs.effective_time" for issue in report.issues)
    assert any(issue.field == "vitals.effective_time" for issue in report.issues)
    assert any(
        issue.field == "documents.timestamp"
        and "outside all encounter windows" in issue.message
        for issue in report.issues
    )


def test_validator_does_not_treat_invalid_encounter_end_as_open_window():
    bad = _record(
        encounters=[
            Encounter(
                encounter_id="enc-1",
                start="2026-05-06T08:00:00",
                end="not-a-date",
                setting="ed",
                reason="fever",
            )
        ],
        labs=[
            LabObservation(
                name="Lactate",
                value=3.2,
                unit="mmol/L",
                reference_low=0.5,
                reference_high=2.0,
                flag="high",
                effective_time="2026-05-07T14:00:00",
            )
        ],
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "encounters.end" for issue in report.issues)
    assert not any(
        issue.field == "labs.effective_time"
        and "outside all encounter windows" in issue.message
        for issue in report.issues
    )


def test_validator_rejects_non_chronological_time_series():
    bad = _record(
        modalities=[Modality.TIME_SERIES],
        time_series=[
            TimeSeriesChannel(
                name="heart_rate",
                unit="/min",
                points=[
                    TimeSeriesPoint(
                        timestamp="2026-05-06T11:00:00",
                        values={"heart_rate": 110},
                    ),
                    TimeSeriesPoint(
                        timestamp="2026-05-06T10:00:00",
                        values={"heart_rate": 115},
                    ),
                ],
            )
        ],
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "time_series.order" for issue in report.issues)


def test_validator_rejects_implausible_waveform_channels():
    bad = _record(
        modalities=[Modality.TIME_SERIES],
        time_series=[
            TimeSeriesChannel(
                name="ecg_lead_ii",
                unit="mV",
                sampling_rate_hz=2,
                points=[
                    TimeSeriesPoint(
                        timestamp="2026-05-06T10:00:00",
                        values={"millivolts": 8.5, "phase": 1.4},
                    )
                ],
            ),
            TimeSeriesChannel(
                name="pleth",
                unit="relative",
                sampling_rate_hz=25,
                points=[
                    TimeSeriesPoint(
                        timestamp="2026-05-06T10:00:00",
                        values={"amplitude": -0.2, "phase": 0.1},
                    )
                ],
            ),
        ],
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "time_series.sampling_rate_hz" for issue in report.issues)
    assert any(issue.field == "time_series.ecg_lead_ii.millivolts" for issue in report.issues)
    assert any(issue.field == "time_series.ecg_lead_ii.phase" for issue in report.issues)
    assert any(issue.field == "time_series.pleth.amplitude" for issue in report.issues)


def test_validator_rejects_time_series_that_conflicts_with_structured_values():
    bad = _record(
        vitals=[
            VitalObservation(
                name="HR",
                value=118,
                unit="/min",
                effective_time="2026-05-06T08:00:00",
            )
        ],
        labs=[
            LabObservation(
                name="Lactate",
                value=4.8,
                unit="mmol/L",
                reference_low=0.5,
                reference_high=2.0,
                flag="critical",
                effective_time="2026-05-06T08:30:00",
            )
        ],
        time_series=[
            TimeSeriesChannel(
                name="heart_rate",
                unit="/min",
                points=[
                    TimeSeriesPoint(
                        timestamp="2026-05-06T08:00:00",
                        values={"value": 35},
                    )
                ],
            ),
            TimeSeriesChannel(
                name="lab_lactate",
                unit="mmol/L",
                points=[
                    TimeSeriesPoint(
                        timestamp="2026-05-06T08:30:00",
                        values={"value": 0.6},
                    )
                ],
            ),
        ],
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "time_series.heart_rate.alignment" for issue in report.issues)
    assert any(issue.field == "time_series.lab_lactate.alignment" for issue in report.issues)


def test_validator_rejects_text_structured_lab_and_vital_contradictions():
    bad = _record(
        labs=[
            LabObservation(
                name="Lactate",
                value=5.2,
                unit="mmol/L",
                reference_low=0.5,
                reference_high=2.0,
                flag="critical",
                effective_time="2026-05-06T08:30:00",
            )
        ],
        vitals=[
            VitalObservation(
                name="Temperature",
                value=39.2,
                unit="C",
                effective_time="2026-05-06T08:00:00",
            )
        ],
        documents=[
            ClinicalDocument(
                document_id="doc-1",
                note_type="ed_note",
                author_role="physician",
                timestamp="2026-05-06T09:00:00",
                clean_text="Patient is afebrile and lactate is normal.",
            )
        ],
    )

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "documents.lactate" for issue in report.issues)
    assert any(issue.field == "documents.fever" for issue in report.issues)
