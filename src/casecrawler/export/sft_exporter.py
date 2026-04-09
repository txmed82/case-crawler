from __future__ import annotations
from casecrawler.models.case import GeneratedCase
from casecrawler.models.export import Conversation, Turn
from casecrawler.models.phase import CasePhase

SFT_SYSTEM_PROMPT = (
    "You are a physician evaluating a patient. At each step, you will receive "
    "clinical information and must decide on the next action. Explain your "
    "clinical reasoning before stating your decision."
)

def _build_system_turn(phase: CasePhase) -> str:
    parts = [phase.narrative]
    if phase.vitals:
        v = phase.vitals
        parts.append(
            f"\nVitals: HR {v.hr}, BP {v.bp_systolic}/{v.bp_diastolic}, "
            f"RR {v.rr}, SpO2 {v.spo2}%, Temp {v.temp_c}°C"
            + (f", GCS {v.gcs}" if v.gcs is not None else "")
        )
    for lr in phase.lab_results:
        values_str = ", ".join(f"{lv.name} {lv.value} {lv.unit}" for lv in lr.values)
        parts.append(f"\n{lr.panel} ({lr.timestamp}): {values_str}")
    for ir in phase.imaging_results:
        findings_str = "; ".join(f"{f.structure}: {f.observation}" for f in ir.findings)
        parts.append(f"\n{ir.modality} {ir.body_region} ({ir.timestamp}): {findings_str}. Impression: {ir.impression}")
    if phase.clinical_update:
        parts.append(f"\nUpdate: {phase.clinical_update}")
    return "".join(parts)

def _build_assistant_turn(phase: CasePhase, use_optimal: bool = True) -> str:
    if not phase.decisions:
        return "No clinical decision options available for this phase."
    if use_optimal:
        decision = next((d for d in phase.decisions if d.is_optimal), phase.decisions[0])
    else:
        decision = next((d for d in phase.decisions if not d.is_optimal), phase.decisions[-1])
    return (
        f"Based on the clinical presentation, I would {decision.action}. "
        f"Reasoning: {decision.reasoning}. "
        f"Expected outcome: {decision.clinical_outcome}."
    )

def export_sft_conversation(case: GeneratedCase, include_wrong_path: bool = False) -> Conversation:
    turns: list[Turn] = []
    use_optimal = not include_wrong_path
    for phase in case.phases:
        turns.append(Turn(role="system", content=_build_system_turn(phase)))
        turns.append(Turn(role="assistant", content=_build_assistant_turn(phase, use_optimal=use_optimal)))
    return Conversation(case_id=case.case_id, system_prompt=SFT_SYSTEM_PROMPT, turns=turns)
