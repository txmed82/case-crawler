import json

from casecrawler.export.blueprint_release import (
    build_blueprint_release_summary,
    write_blueprint_release_summary,
)
from casecrawler.generation.blueprint_materializer import BlueprintMaterializer
from casecrawler.models.blueprint import (
    BlueprintEvidence,
    BlueprintValidationReport,
    ClinicalBlueprint,
    GenerationAttempt,
    GenerationAttemptStatus,
    GenerationRole,
    JudgeReport,
    ReleaseReadinessTier,
)
from casecrawler.storage.dataset_store import DatasetStore


def _blueprint(blueprint_id: str, *, organ_system: str = "cardiovascular"):
    return ClinicalBlueprint(
        blueprint_id=blueprint_id,
        dataset_id="ds-1",
        cohort_plan_id="plan-1",
        archetype_name="anticoagulation decision",
        organ_system=organ_system,
        setting="outpatient",
        patient={"age": 72, "sex": "female"},
        chief_concern="Atrial fibrillation anticoagulation follow-up.",
        diagnoses=[
            {
                "name": "atrial fibrillation",
                "supporting_findings": ["ECG confirms AF"],
            }
        ],
        evidence=BlueprintEvidence(
            supported_claims=["AF anticoagulation requires renal-dose review."],
            citations=[{"source": "dailymed", "claim": "renal-dose review"}],
        ),
    )


def _judge_report(blueprint_id: str, *, passed: bool = True) -> JudgeReport:
    return JudgeReport(
        report_id=f"judge-{blueprint_id}",
        dataset_id="ds-1",
        artifact_id=blueprint_id,
        role=GenerationRole.JUDGE,
        score=0.93 if passed else 0.41,
        passed=passed,
        rubric="blueprint_plausibility",
    )


def _validation(
    blueprint_id: str,
    *,
    tier: ReleaseReadinessTier,
    judge_report: JudgeReport | None = None,
) -> BlueprintValidationReport:
    return BlueprintValidationReport(
        blueprint_id=blueprint_id,
        tier=tier,
        schema_valid=True,
        clinically_plausible=tier
        in {
            ReleaseReadinessTier.CLINICALLY_PLAUSIBLE,
            ReleaseReadinessTier.JUDGE_VALIDATED,
            ReleaseReadinessTier.RESEARCH_RELEASE_READY,
        },
        grounded=tier == ReleaseReadinessTier.RESEARCH_RELEASE_READY,
        judge_validated=tier
        in {
            ReleaseReadinessTier.JUDGE_VALIDATED,
            ReleaseReadinessTier.RESEARCH_RELEASE_READY,
        },
        judge_reports=[judge_report] if judge_report is not None else [],
        issues=[] if tier == ReleaseReadinessTier.RESEARCH_RELEASE_READY else [
            {"field": "evidence.citations", "message": "missing citation"}
        ],
    )


def test_build_blueprint_release_summary_aggregates_readiness_and_materialization(
    tmp_path,
):
    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    ready_blueprint = _blueprint("bp-ready")
    draft_blueprint = _blueprint("bp-draft", organ_system="endocrine")
    ready_judge = _judge_report("bp-ready")
    draft_judge = _judge_report("bp-draft", passed=False)
    store.save_blueprint(ready_blueprint)
    store.save_blueprint(draft_blueprint)
    store.save_judge_report(ready_judge)
    store.save_judge_report(draft_judge)
    store.save_blueprint_validation_report(
        _validation(
            "bp-ready",
            tier=ReleaseReadinessTier.RESEARCH_RELEASE_READY,
            judge_report=ready_judge,
        )
    )
    store.save_blueprint_validation_report(
        _validation(
            "bp-draft",
            tier=ReleaseReadinessTier.SCHEMA_VALID,
            judge_report=draft_judge,
        )
    )
    store.save_generation_attempt(
        GenerationAttempt(
            attempt_id="attempt-1",
            dataset_id="ds-1",
            role=GenerationRole.REPAIR,
            status=GenerationAttemptStatus.REPAIR_REQUESTED,
            provider="openai",
            model="repair-model",
            artifact_id="bp-draft",
        )
    )
    BlueprintMaterializer(created_at="2026-05-06T10:00:00").materialize(
        ready_blueprint,
        validation_report=store.get_blueprint_validation_report("bp-ready"),
        store=store,
        require_release_ready=True,
    )

    summary = build_blueprint_release_summary(store, "ds-1")

    assert summary["dataset_id"] == "ds-1"
    assert summary["blueprint_count"] == 2
    assert summary["validation_report_count"] == 2
    assert summary["research_release_ready_count"] == 1
    assert summary["materialized_record_count"] == 1
    assert summary["materialized_blueprint_ids"] == ["bp-ready"]
    assert summary["missing_materialized_blueprint_ids"] == []
    assert summary["judge_report_count"] == 2
    assert summary["passing_judge_report_count"] == 1
    assert summary["attempt_counts"]["repair_requested"] == 1
    assert summary["tier_counts"]["research_release_ready"] == 1
    assert summary["tier_counts"]["schema_valid"] == 1
    assert summary["tier_counts"]["missing"] == 0
    assert summary["organ_system_counts"] == {
        "cardiovascular": 1,
        "endocrine": 1,
    }
    assert summary["non_ready_blueprints"][0]["blueprint_id"] == "bp-draft"


def test_write_blueprint_release_summary_writes_json(tmp_path):
    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    store.save_blueprint(_blueprint("bp-1"))
    output = tmp_path / "summary.json"

    summary = write_blueprint_release_summary(store, "ds-1", output)

    assert json.loads(output.read_text()) == summary
    assert summary["blueprint_count"] == 1
