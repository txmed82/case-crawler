from casecrawler.models.blueprint import (
    BlueprintEvidence,
    BlueprintValidationReport,
    ClinicalBlueprint,
    CohortArchetype,
    CohortPlan,
    GenerationAttempt,
    GenerationAttemptStatus,
    GenerationRole,
    JudgeReport,
    ReleaseReadinessTier,
)
from casecrawler.models.synthetic import (
    ComplexityProfile,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
)
from casecrawler.storage.dataset_store import DatasetStore


def _cohort_plan() -> CohortPlan:
    return CohortPlan(
        plan_id="plan-1",
        request="Generate diverse cardiovascular decision cases.",
        target_count=2,
        domains=["cardiology"],
        settings=["outpatient"],
        archetypes=[
            CohortArchetype(
                name="anticoagulation decision",
                organ_system="cardiovascular",
                setting="outpatient",
                target_count=2,
                acuity_mix={"routine": 0.5, "urgent": 0.5},
                difficulty_mix={"moderate": 1.0},
                required_modalities=[
                    Modality.STRUCTURED_EHR,
                    Modality.CLINICAL_TEXT,
                ],
                task_targets=["medication_reconciliation"],
            )
        ],
        required_grounding=True,
        created_by=GenerationRole.PLANNER,
    )


def _blueprint(blueprint_id: str = "bp-1") -> ClinicalBlueprint:
    return ClinicalBlueprint(
        blueprint_id=blueprint_id,
        dataset_id="ds-1",
        cohort_plan_id="plan-1",
        archetype_name="anticoagulation decision",
        organ_system="cardiovascular",
        setting="outpatient",
        patient={"age": 72, "sex": "female"},
        chief_concern="Atrial fibrillation anticoagulation follow-up.",
        diagnoses=[
            {
                "name": "atrial fibrillation",
                "supporting_findings": ["ECG confirms AF"],
            }
        ],
        clinical_reasoning_targets=["Review renal dosing and bleeding risk."],
        evidence=BlueprintEvidence(
            supported_claims=["AF anticoagulation requires renal-dose review."],
            citations=[{"source": "dailymed", "claim": "renal-dose review"}],
        ),
    )


def _record() -> SyntheticRecord:
    return SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="cardiology",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.STRUCTURED_EHR, Modality.CLINICAL_TEXT],
        patient=SyntheticPatient(patient_id="pat-1", age=72, sex="female"),
        encounters=[],
        provenance=Provenance(generator="unit-test", created_at="2026-05-06T10:00:00"),
    )


def test_dataset_store_round_trips_cohort_plans_and_blueprints(tmp_path):
    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    plan = _cohort_plan()
    blueprint = _blueprint()

    store.save_cohort_plan(plan)
    store.save_blueprint(blueprint)

    assert store.get_cohort_plan("plan-1") == plan
    assert store.list_cohort_plans()[0].plan_id == "plan-1"
    assert store.get_blueprint("bp-1") == blueprint
    assert store.list_blueprints(dataset_id="ds-1") == [blueprint]
    assert store.list_blueprints(cohort_plan_id="plan-1") == [blueprint]


def test_dataset_store_tracks_attempts_and_judge_reports_by_artifact(tmp_path):
    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    attempt = GenerationAttempt(
        attempt_id="attempt-1",
        dataset_id="ds-1",
        role=GenerationRole.BLUEPRINT_GENERATOR,
        status=GenerationAttemptStatus.REPAIR_REQUESTED,
        provider="openrouter",
        model="anthropic/claude-sonnet-4-6",
        prompt_hash="abc123",
        input_tokens=100,
        output_tokens=75,
        errors=["missing supporting finding"],
        artifact_id="bp-1",
    )
    judge_report = JudgeReport(
        report_id="judge-1",
        dataset_id="ds-1",
        artifact_id="bp-1",
        role=GenerationRole.JUDGE,
        score=0.92,
        passed=True,
        rubric="blueprint_plausibility",
        findings=[{"criterion": "diagnostic_support", "passed": True}],
    )
    validation = BlueprintValidationReport(
        blueprint_id="bp-1",
        tier=ReleaseReadinessTier.JUDGE_VALIDATED,
        schema_valid=True,
        clinically_plausible=True,
        grounded=True,
        judge_validated=True,
        judge_reports=[judge_report],
    )

    store.save_generation_attempt(attempt)
    store.save_judge_report(judge_report)
    store.save_blueprint_validation_report(validation)

    assert store.get_generation_attempt("attempt-1") == attempt
    assert store.list_generation_attempts(dataset_id="ds-1") == [attempt]
    assert store.list_generation_attempts(artifact_id="bp-1") == [attempt]
    assert store.get_judge_report("judge-1") == judge_report
    assert store.list_judge_reports(artifact_id="bp-1") == [judge_report]
    assert store.get_blueprint_validation_report("bp-1") == validation


def test_dataset_manifest_includes_blueprint_persistence_counts(tmp_path):
    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    store.save_record(_record())
    store.save_cohort_plan(_cohort_plan())
    store.save_blueprint(_blueprint("bp-1"))
    store.save_blueprint(_blueprint("bp-2"))
    store.save_generation_attempt(
        GenerationAttempt(
            attempt_id="attempt-1",
            dataset_id="ds-1",
            role=GenerationRole.PLANNER,
            status=GenerationAttemptStatus.SUCCEEDED,
            provider="openai",
            model="gpt-4.1",
        )
    )
    store.save_judge_report(
        JudgeReport(
            report_id="judge-1",
            dataset_id="ds-1",
            artifact_id="bp-1",
            role=GenerationRole.JUDGE,
            score=0.88,
            passed=True,
            rubric="blueprint_plausibility",
        )
    )

    manifest = store.get_manifest("ds-1")

    assert manifest.metadata["cohort_plan_ids"] == ["plan-1"]
    assert manifest.metadata["blueprint_count"] == 2
    assert manifest.metadata["generation_attempt_count"] == 1
    assert manifest.metadata["judge_report_count"] == 1
