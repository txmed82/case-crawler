import pytest

from casecrawler.models.blueprint import (
    BlueprintEvidence,
    BlueprintValidationReport,
    ClinicalBlueprint,
    GenerationRole,
    JudgeReport,
    ReleaseReadinessTier,
)
from casecrawler.models.synthetic import ComplexityProfile, Modality
from casecrawler.storage.dataset_store import DatasetStore


def _blueprint() -> ClinicalBlueprint:
    return ClinicalBlueprint(
        blueprint_id="bp-1",
        dataset_id="ds-1",
        cohort_plan_id="plan-1",
        archetype_name="anticoagulation decision",
        organ_system="cardiovascular",
        setting="outpatient",
        patient={"age": 72, "sex": "female", "race": "not specified"},
        chief_concern="Atrial fibrillation anticoagulation follow-up.",
        diagnoses=[
            {
                "name": "atrial fibrillation",
                "supporting_findings": ["ECG confirms AF"],
            }
        ],
        differential=[
            {
                "name": "supraventricular tachycardia",
                "rationale": "Palpitations without ECG confirmation can mimic AF.",
            }
        ],
        timeline=[{"time": "2026-05-06T10:00:00", "event": "Follow-up visit"}],
        expected_labs=[
            {
                "name": "Creatinine",
                "value": 1.4,
                "unit": "mg/dL",
                "effective_time": "2026-05-06T10:10:00",
            }
        ],
        expected_vitals=[
            {
                "name": "Heart rate",
                "value": 92,
                "unit": "beats/min",
                "effective_time": "2026-05-06T10:05:00",
            }
        ],
        medications=[
            {
                "name": "Apixaban",
                "dose": "5 mg",
                "route": "oral",
                "frequency": "twice daily",
                "status": "active",
            }
        ],
        orders=[
            {
                "order_type": "lab",
                "display": "Basic metabolic panel",
                "priority": "routine",
            }
        ],
        clinical_reasoning_targets=["Review renal dosing and bleeding risk."],
        safety_constraints=["Review bleeding risk before anticoagulation."],
        uncertainty_points=["Renal function may require dose adjustment."],
        intended_tasks=["medication_reconciliation"],
        evidence=BlueprintEvidence(
            supported_claims=["AF anticoagulation requires renal-dose review."],
            citations=[{"source": "dailymed", "claim": "renal-dose review"}],
        ),
        metadata={"difficulty": "complex"},
    )


def _release_ready_report() -> BlueprintValidationReport:
    judge = JudgeReport(
        report_id="judge-1",
        dataset_id="ds-1",
        artifact_id="bp-1",
        role=GenerationRole.JUDGE,
        score=0.93,
        passed=True,
        rubric="blueprint_plausibility",
    )
    return BlueprintValidationReport(
        blueprint_id="bp-1",
        tier=ReleaseReadinessTier.RESEARCH_RELEASE_READY,
        schema_valid=True,
        clinically_plausible=True,
        grounded=True,
        judge_validated=True,
        judge_reports=[judge],
    )


def test_blueprint_materializer_creates_persistable_synthetic_record(tmp_path):
    from casecrawler.generation.blueprint_materializer import BlueprintMaterializer

    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))

    record = BlueprintMaterializer(
        created_at="2026-05-06T10:00:00"
    ).materialize(
        _blueprint(),
        validation_report=_release_ready_report(),
        store=store,
        require_release_ready=True,
    )

    assert record.record_id.startswith("rec-")
    assert record.dataset_id == "ds-1"
    assert record.topic == "cardiovascular"
    assert record.complexity == ComplexityProfile.COMPLEX
    assert set(record.modalities) == {
        Modality.STRUCTURED_EHR,
        Modality.CLINICAL_TEXT,
        Modality.LABS,
        Modality.VITALS,
    }
    assert record.patient.age == 72
    assert record.patient.sex == "female"
    assert record.encounters[0].setting == "outpatient"
    assert record.encounters[0].reason == "Atrial fibrillation anticoagulation follow-up."
    assert record.encounters[0].diagnoses[0].display == "atrial fibrillation"
    assert record.labs[0].name == "Creatinine"
    assert record.vitals[0].name == "Heart rate"
    assert record.medication_history[0].name == "Apixaban"
    assert record.orders[0].display == "Basic metabolic panel"
    assert "renal dosing" in record.documents[0].clean_text
    assert record.provenance.generator == "blueprint-materializer"
    assert record.provenance.created_at == "2026-05-06T10:00:00"
    assert record.provenance.source_refs[0]["blueprint_id"] == "bp-1"
    assert record.metadata["blueprint_id"] == "bp-1"
    assert record.metadata["cohort_plan_id"] == "plan-1"
    assert record.metadata["release_readiness_tier"] == "research_release_ready"
    assert record.metadata["intended_tasks"] == ["medication_reconciliation"]
    assert store.get_record(record.record_id) == record


def test_blueprint_materializer_blocks_when_release_ready_required():
    from casecrawler.generation.blueprint_materializer import BlueprintMaterializer

    with pytest.raises(ValueError, match="research release ready"):
        BlueprintMaterializer().materialize(
            _blueprint(),
            validation_report=BlueprintValidationReport(
                blueprint_id="bp-1",
                tier=ReleaseReadinessTier.CLINICALLY_PLAUSIBLE,
                schema_valid=True,
                clinically_plausible=True,
                grounded=True,
            ),
            require_release_ready=True,
        )
