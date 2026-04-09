from __future__ import annotations
from pydantic import BaseModel

class Observation(BaseModel):
    narrative: str
    vitals: dict | None
    new_lab_results: list[dict]
    new_imaging_results: list[dict]
    pending_orders: list[str]
    time_elapsed: str

class Action(BaseModel):
    id: str
    description: str
    quality: str

class TimeStep(BaseModel):
    step_number: int
    observation: Observation
    action_space: list[Action]
    optimal_action: str
    reward_table: dict[str, float]

class Episode(BaseModel):
    case_id: str
    difficulty: str
    specialty: list[str]
    steps: list[TimeStep]

class Turn(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    case_id: str
    system_prompt: str
    turns: list[Turn]
