from __future__ import annotations

from casecrawler.models.blueprint import (
    BlueprintValidationReport,
    ClinicalBlueprint,
    JudgeReport,
    ReleaseReadinessTier,
)


class BlueprintValidator:
    def __init__(self, *, require_grounding: bool = True) -> None:
        self._require_grounding = require_grounding

    def validate(
        self,
        blueprint: ClinicalBlueprint,
        *,
        judge_reports: list[JudgeReport] | None = None,
    ) -> BlueprintValidationReport:
        judges = judge_reports or []
        issues = self._issues(blueprint)
        schema_valid = True
        grounded = self._grounded(blueprint)
        clinically_plausible = not any(
            issue["severity"] == "error" and issue["field"] != "evidence.citations"
            for issue in issues
        )
        if self._require_grounding and not grounded:
            clinically_plausible = False
        judge_validated = bool(judges) and all(report.passed for report in judges)
        tier = self._tier(
            schema_valid=schema_valid,
            clinically_plausible=clinically_plausible,
            grounded=grounded,
            judge_validated=judge_validated,
        )
        return BlueprintValidationReport(
            blueprint_id=blueprint.blueprint_id,
            tier=tier,
            schema_valid=schema_valid,
            clinically_plausible=clinically_plausible,
            grounded=grounded,
            judge_validated=judge_validated,
            issues=issues,
            judge_reports=judges,
        )

    def _issues(self, blueprint: ClinicalBlueprint) -> list[dict[str, str]]:
        issues: list[dict[str, str]] = []
        if not blueprint.patient.get("age"):
            issues.append(
                {
                    "severity": "error",
                    "field": "patient.age",
                    "message": "Blueprint patient must include age.",
                }
            )
        if not blueprint.patient.get("sex"):
            issues.append(
                {
                    "severity": "error",
                    "field": "patient.sex",
                    "message": "Blueprint patient must include sex.",
                }
            )
        if not blueprint.clinical_reasoning_targets:
            issues.append(
                {
                    "severity": "warning",
                    "field": "clinical_reasoning_targets",
                    "message": "Blueprint has no explicit reasoning target.",
                }
            )
        if blueprint.evidence.unsupported_claims:
            issues.append(
                {
                    "severity": "error",
                    "field": "evidence.unsupported_claims",
                    "message": "Blueprint contains unsupported evidence claims.",
                }
            )
        if self._require_grounding and not blueprint.evidence.citations:
            issues.append(
                {
                    "severity": "error",
                    "field": "evidence.citations",
                    "message": "Grounding is required but blueprint has no citations.",
                }
            )
        return issues

    def _grounded(self, blueprint: ClinicalBlueprint) -> bool:
        if not self._require_grounding:
            return True
        return bool(blueprint.evidence.citations)

    def _tier(
        self,
        *,
        schema_valid: bool,
        clinically_plausible: bool,
        grounded: bool,
        judge_validated: bool,
    ) -> ReleaseReadinessTier:
        if schema_valid and clinically_plausible and grounded and judge_validated:
            return ReleaseReadinessTier.RESEARCH_RELEASE_READY
        if schema_valid and clinically_plausible and grounded:
            return ReleaseReadinessTier.CLINICALLY_PLAUSIBLE
        if schema_valid:
            return ReleaseReadinessTier.SCHEMA_VALID
        return ReleaseReadinessTier.DRAFT
