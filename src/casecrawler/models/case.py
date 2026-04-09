from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from casecrawler.models.blueprint import CaseBlueprint
from casecrawler.models.phase import CasePhase


class DifficultyLevel(str, Enum):
    MEDICAL_STUDENT = "medical_student"
    RESIDENT = "resident"
    ATTENDING = "attending"


class Patient(BaseModel):
    age: int
    sex: str
    demographics: str


class GroundTruth(BaseModel):
    diagnosis: str
    optimal_next_step: str
    rationale: str
    key_findings: list[str]


class DecisionChoice(BaseModel):
    choice: str
    is_correct: bool
    error_type: str | None = None
    reasoning: str
    outcome: str
    consequence: str | None = None
    next_decision: str | None = None


class Complication(BaseModel):
    trigger: str
    detail: str
    event: str
    outcome: str


class ReviewResult(BaseModel):
    accuracy_score: float
    pedagogy_score: float
    bias_score: float
    approved: bool
    notes: list[str]


class GeneratedCase(BaseModel):
    case_id: str
    topic: str
    difficulty: DifficultyLevel
    specialty: list[str]
    patient: Patient
    vignette: str
    decision_prompt: str
    ground_truth: GroundTruth
    decision_tree: list[DecisionChoice]
    complications: list[Complication]
    blueprint: CaseBlueprint | None = None
    phases: list[CasePhase] = []
    review: ReviewResult | None = None
    sources: list[dict]
    metadata: dict

    def is_multi_step(self) -> bool:
        return len(self.phases) > 1
