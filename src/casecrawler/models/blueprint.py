from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from casecrawler.models.synthetic import Modality, StrictModel


class GenerationRole(str, Enum):
    PLANNER = "planner"
    BLUEPRINT_GENERATOR = "blueprint_generator"
    ARTIFACT_GENERATOR = "artifact_generator"
    JUDGE = "judge"
    REPAIR = "repair"


class GenerationAttemptStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REPAIR_REQUESTED = "repair_requested"


class ReleaseReadinessTier(str, Enum):
    DRAFT = "draft"
    SCHEMA_VALID = "schema_valid"
    CLINICALLY_PLAUSIBLE = "clinically_plausible"
    JUDGE_VALIDATED = "judge_validated"
    RESEARCH_RELEASE_READY = "research_release_ready"


class CohortArchetype(StrictModel):
    name: str
    organ_system: str
    setting: str
    target_count: int = Field(ge=1)
    acuity_mix: dict[str, float] = Field(default_factory=dict)
    difficulty_mix: dict[str, float] = Field(default_factory=dict)
    required_modalities: list[Modality] = Field(default_factory=list)
    task_targets: list[str] = Field(default_factory=list)
    safety_constraints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CohortPlan(StrictModel):
    plan_id: str
    request: str
    target_count: int = Field(ge=1)
    domains: list[str] = Field(default_factory=list)
    settings: list[str] = Field(default_factory=list)
    archetypes: list[CohortArchetype] = Field(min_length=1)
    required_grounding: bool = False
    diversity_targets: dict[str, Any] = Field(default_factory=dict)
    created_by: GenerationRole
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_archetype_counts_to_match_target_count(self) -> "CohortPlan":
        archetype_total = sum(archetype.target_count for archetype in self.archetypes)
        if archetype_total != self.target_count:
            raise ValueError(
                "target_count must equal the sum of archetype target_count values"
            )
        return self

    @property
    def archetype_counts(self) -> dict[str, int]:
        return {
            archetype.name: archetype.target_count for archetype in self.archetypes
        }


class BlueprintEvidence(StrictModel):
    supported_claims: list[str] = Field(default_factory=list)
    inferred_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)


class BlueprintDiagnosis(StrictModel):
    name: str
    supporting_findings: list[str] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BlueprintDifferentialDiagnosis(StrictModel):
    name: str
    rationale: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClinicalBlueprint(StrictModel):
    blueprint_id: str
    dataset_id: str
    cohort_plan_id: str
    archetype_name: str
    organ_system: str
    setting: str
    patient: dict[str, Any]
    chief_concern: str
    diagnoses: list[BlueprintDiagnosis] = Field(min_length=1)
    differential: list[BlueprintDifferentialDiagnosis] = Field(default_factory=list)
    comorbidities: list[str] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    expected_labs: list[dict[str, Any]] = Field(default_factory=list)
    expected_vitals: list[dict[str, Any]] = Field(default_factory=list)
    medications: list[dict[str, Any]] = Field(default_factory=list)
    orders: list[dict[str, Any]] = Field(default_factory=list)
    clinical_reasoning_targets: list[str] = Field(default_factory=list)
    safety_constraints: list[str] = Field(default_factory=list)
    uncertainty_points: list[str] = Field(default_factory=list)
    intended_tasks: list[str] = Field(default_factory=list)
    evidence: BlueprintEvidence = Field(default_factory=BlueprintEvidence)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def primary_diagnosis(self) -> str:
        return self.diagnoses[0].name

    @property
    def has_unsupported_claims(self) -> bool:
        return bool(self.evidence.unsupported_claims)

    @property
    def required_modalities(self) -> list[Modality]:
        modalities = [Modality.STRUCTURED_EHR]
        if (
            self.chief_concern
            or self.clinical_reasoning_targets
            or self.safety_constraints
            or self.uncertainty_points
        ):
            modalities.append(Modality.CLINICAL_TEXT)
        return modalities


class GenerationRolePolicy(StrictModel):
    role: GenerationRole
    provider: str
    model: str
    temperature: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationAttempt(StrictModel):
    attempt_id: str
    dataset_id: str
    role: GenerationRole
    status: GenerationAttemptStatus
    provider: str
    model: str
    prompt_hash: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)
    artifact_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class JudgeReport(StrictModel):
    report_id: str
    dataset_id: str
    artifact_id: str
    role: GenerationRole
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    rubric: str
    findings: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BlueprintValidationReport(StrictModel):
    blueprint_id: str
    tier: ReleaseReadinessTier = ReleaseReadinessTier.DRAFT
    schema_valid: bool = False
    clinically_plausible: bool = False
    grounded: bool = False
    judge_validated: bool = False
    issues: list[dict[str, Any]] = Field(default_factory=list)
    judge_reports: list[JudgeReport] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def research_release_ready(self) -> bool:
        return (
            self.tier == ReleaseReadinessTier.RESEARCH_RELEASE_READY
            and self.schema_valid
            and self.clinically_plausible
            and self.grounded
            and self.judge_validated
            and not self.issues
        )
