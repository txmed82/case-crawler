"""Regression tests for Phase 4d (validator tightening)."""

from __future__ import annotations

from casecrawler.models.synthetic import (
    ClinicalDocument,
    ComplexityProfile,
    GroundingBundle,
    GroundingCitation,
    LabObservation,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
)
from casecrawler.validation.clinical_rules import (
    validate_text_structured_contradictions,
)
from casecrawler.validation.privacy import validate_privacy
from casecrawler.validation.synthetic_validator import SyntheticValidator


def _record(**overrides) -> SyntheticRecord:
    data = {
        "record_id": "rec-1",
        "dataset_id": "ds-1",
        "topic": "sepsis",
        "complexity": ComplexityProfile.MODERATE,
        "modalities": [Modality.LABS, Modality.CLINICAL_TEXT],
        "patient": SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        "encounters": [],
        "labs": [],
        "vitals": [],
        "documents": [],
        "provenance": Provenance(
            generator="unit-test", created_at="2026-05-06T09:00:00"
        ),
    }
    data.update(overrides)
    return SyntheticRecord(**data)


# ---------- Privacy scope ---------------------------------------------------


def test_privacy_scan_does_not_trip_on_provenance_email():
    """The tool's own contact email lives on `provenance` (e.g. NCBI
    ENTREZ_EMAIL). It must NOT trip the patient PHI EMAIL regex."""
    record = _record(
        provenance=Provenance(
            generator="unit-test",
            created_at="2026-05-06T09:00:00",
            source_refs=[
                {
                    "source": "pubmed",
                    "contact": "tool-owner@casecrawler.dev",
                }
            ],
        ),
    )
    issues = validate_privacy(record)
    assert all(i.field != "privacy" for i in issues), issues


def test_privacy_scan_does_not_trip_on_grounding_citation_url():
    """Grounding citations come from third-party catalogs and may carry
    URLs or DOIs that look address-like; they're not patient PHI."""
    grounding = GroundingBundle(
        topic="sepsis",
        retrieved_at="2026-05-06T09:00:00Z",
        citations=[
            GroundingCitation(
                chunk_id="c1",
                source="pubmed",
                source_document_id="pubmed:1",
                score=0.9,
                credibility="guideline",
                snippet="Standard of care from 1234 Beacon Avenue clinic.",
                url="https://example.com/study",
            )
        ],
    )
    record = _record(metadata={"grounding": grounding.model_dump()})
    issues = validate_privacy(record)
    assert all(i.field != "privacy" for i in issues), issues


def test_privacy_scan_still_catches_phi_in_documents():
    record = _record(
        documents=[
            ClinicalDocument(
                document_id="doc-1",
                note_type="ed_note",
                author_role="physician",
                timestamp="2026-05-06T09:00:00",
                clean_text="Call patient at 555-123-4567 to confirm.",
            )
        ]
    )
    issues = validate_privacy(record)
    assert any(i.field == "privacy" for i in issues)


def test_privacy_scan_still_catches_phi_in_demographics():
    """Patient demographics CAN carry PHI -- that's a real risk we still
    want to catch. Only provenance / grounding are excluded."""
    record = _record(
        patient=SyntheticPatient(
            patient_id="pat-1",
            age=64,
            sex="male",
            demographics={"contact": {"email": "patient@example.com"}},
        )
    )
    issues = validate_privacy(record)
    assert any(i.field == "privacy" for i in issues)


# ---------- Contradiction judge wiring -------------------------------------


def test_contradiction_judge_extends_regex_findings():
    record = _record(
        labs=[
            LabObservation(
                name="Lactate",
                value=4.5,
                unit="mmol/L",
                reference_low=0.5,
                reference_high=2.0,
                flag="H",
                effective_time="2026-05-06T08:30:00",
            )
        ],
        documents=[
            ClinicalDocument(
                document_id="doc-1",
                note_type="ed_note",
                author_role="physician",
                timestamp="2026-05-06T09:00:00",
                # Phrase the regex doesn't catch.
                clean_text="Lactic acid was within reassuring limits.",
            )
        ],
    )

    judge_calls = []

    def fake_judge(rec, text):
        judge_calls.append((rec.record_id, "lactic acid" in text.lower()))
        return [
            {
                "field": "documents.lactate",
                "message": "Note implies normal lactic acid; structured lactate is high.",
            }
        ]

    issues = validate_text_structured_contradictions(record, judge=fake_judge)
    assert judge_calls and judge_calls[0][1] is True
    assert any(i.field == "documents.lactate" for i in issues)


def test_contradiction_judge_returning_non_list_does_not_crash():
    """A misbehaving judge that returns a non-list truthy value (e.g.
    ``True`` or a string) must not crash validation; the regex fallback
    should still produce its issue list."""
    record = _record(
        labs=[
            LabObservation(
                name="Lactate",
                value=4.5,
                unit="mmol/L",
                reference_low=0.5,
                reference_high=2.0,
                flag="H",
                effective_time="2026-05-06T08:30:00",
            )
        ],
        documents=[
            ClinicalDocument(
                document_id="doc-1",
                note_type="ed_note",
                author_role="physician",
                timestamp="2026-05-06T09:00:00",
                clean_text="Lactate is normal.",
            )
        ],
    )

    def chaotic_judge(rec, text):
        return True  # not a list

    issues = validate_text_structured_contradictions(record, judge=chaotic_judge)
    # Regex still fires; non-list judge output is dropped quietly.
    assert any(i.field == "documents.lactate" for i in issues)


def test_contradiction_judge_failure_falls_back_to_regex():
    record = _record(
        labs=[
            LabObservation(
                name="Lactate",
                value=4.5,
                unit="mmol/L",
                reference_low=0.5,
                reference_high=2.0,
                flag="H",
                effective_time="2026-05-06T08:30:00",
            )
        ],
        documents=[
            ClinicalDocument(
                document_id="doc-1",
                note_type="ed_note",
                author_role="physician",
                timestamp="2026-05-06T09:00:00",
                # Phrase the regex DOES catch.
                clean_text="Lactate is normal.",
            )
        ],
    )

    def angry_judge(rec, text):
        raise RuntimeError("judge service unreachable")

    issues = validate_text_structured_contradictions(record, judge=angry_judge)
    # Regex still fires.
    assert any(i.field == "documents.lactate" for i in issues)


def test_synthetic_validator_passes_judge_through_to_contradictions():
    record = _record(
        labs=[
            LabObservation(
                name="Lactate",
                value=4.5,
                unit="mmol/L",
                reference_low=0.5,
                reference_high=2.0,
                flag="H",
                effective_time="2026-05-06T08:30:00",
            )
        ],
        documents=[
            ClinicalDocument(
                document_id="doc-1",
                note_type="ed_note",
                author_role="physician",
                timestamp="2026-05-06T09:00:00",
                clean_text="Lactic acid trending well.",
            )
        ],
    )

    def fake_judge(rec, text):
        return [
            {
                "field": "documents.contradiction",
                "message": "Narrative downplays the abnormal lactate.",
            }
        ]

    validator = SyntheticValidator(threshold=0.0, contradiction_judge=fake_judge)
    report = validator.validate(record)
    fields = {issue.field for issue in report.issues}
    assert "documents.contradiction" in fields
