from __future__ import annotations
from casecrawler.models.case import GeneratedCase
from casecrawler.models.export import Action, Episode, Observation, TimeStep

DEFAULT_REWARD_MAP: dict[str, float] = {
    "optimal": 1.0,
    "acceptable": 0.6,
    "suboptimal": 0.2,
    "harmful": -0.5,
    "catastrophic": -1.0,
}

def export_rl_episode(case: GeneratedCase, reward_map: dict[str, float] | None = None) -> Episode:
    rewards = reward_map or DEFAULT_REWARD_MAP
    steps: list[TimeStep] = []

    for phase in case.phases:
        vitals_dict = phase.vitals.model_dump() if phase.vitals else None
        lab_dicts = [lr.model_dump() for lr in phase.lab_results]
        imaging_dicts = [ir.model_dump() for ir in phase.imaging_results]

        actions: list[Action] = []
        reward_table: dict[str, float] = {}
        optimal_action_id = ""

        for i, decision in enumerate(phase.decisions):
            action_id = f"p{phase.phase_number}_a{i}"
            actions.append(Action(id=action_id, description=decision.action, quality=decision.quality))
            reward_table[action_id] = rewards.get(decision.quality, 0.0)
            if decision.is_optimal:
                optimal_action_id = action_id

        steps.append(TimeStep(
            step_number=phase.phase_number,
            observation=Observation(narrative=phase.narrative, vitals=vitals_dict, new_lab_results=lab_dicts, new_imaging_results=imaging_dicts, pending_orders=[], time_elapsed=phase.time_offset),
            action_space=actions,
            optimal_action=optimal_action_id,
            reward_table=reward_table,
        ))

    return Episode(case_id=case.case_id, difficulty=case.difficulty.value, specialty=case.specialty, steps=steps)
