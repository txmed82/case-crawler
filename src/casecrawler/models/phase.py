from __future__ import annotations
from pydantic import BaseModel
from casecrawler.models.diagnostics import ImagingResult, LabResult, VitalSigns

class PhaseDecision(BaseModel):
    action: str
    is_optimal: bool
    quality: str
    reasoning: str
    clinical_outcome: str
    time_cost: str | None = None
    leads_to_phase: int | None = None

class PhaseOutcome(BaseModel):
    optimal_next_phase: int | None
    patient_status: str
    narrative_transition: str

class CasePhase(BaseModel):
    phase_number: int
    time_offset: str
    narrative: str
    vitals: VitalSigns | None = None
    lab_results: list[LabResult] = []
    imaging_results: list[ImagingResult] = []
    clinical_update: str | None = None
    decisions: list[PhaseDecision] = []
    phase_outcome: PhaseOutcome
