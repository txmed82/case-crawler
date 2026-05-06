from casecrawler.models.synthetic import (
    ComplexityProfile,
    LabObservation,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
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


def test_validator_rejects_phi_like_text():
    bad = _record(metadata={"free_text": "Call patient at 555-123-4567 tomorrow."})

    report = SyntheticValidator().validate(bad)

    assert report.approved is False
    assert any(issue.field == "privacy" for issue in report.issues)

