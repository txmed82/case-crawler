from __future__ import annotations
from pydantic import BaseModel

class PhaseBlueprint(BaseModel):
    phase_number: int
    time_offset: str
    clinical_context: str
    available_diagnostics: list[str]
    pending_diagnostics: list[str]
    decision_type: str
    correct_action: str
    key_reasoning: str

class BranchPoint(BaseModel):
    phase_number: int
    branch_type: str
    trigger_action_quality: str
    description: str

class CaseBlueprint(BaseModel):
    diagnosis: str
    clinical_arc: str
    phase_count: int
    phases: list[PhaseBlueprint]
    branching_points: list[BranchPoint]
    expected_complications: list[str]
