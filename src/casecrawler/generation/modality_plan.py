from __future__ import annotations

from pydantic import BaseModel, Field

from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import Modality


class ModalityPlan(BaseModel):
    topic: str
    cohort_size: int
    modalities: list[Modality]
    cohort_constraints: dict = Field(default_factory=dict)
    document_types: list[str] = Field(default_factory=list)
    lab_panels: list[str] = Field(default_factory=list)
    vital_signs: list[str] = Field(default_factory=list)
    time_series_channels: list[str] = Field(default_factory=list)
    imaging_views: list[str] = Field(default_factory=list)


class ModalityPlanner:
    def build(self, req: GenerationRequest) -> ModalityPlan:
        document_types: list[str] = []
        lab_panels: list[str] = []
        vital_signs: list[str] = []
        time_series_channels: list[str] = []
        imaging_views: list[str] = []

        if Modality.CLINICAL_TEXT in req.modalities:
            document_types = [
                "ed_note",
                "progress_note",
                "nursing_note",
                "discharge_summary",
                "radiology_report",
            ]
        if Modality.LABS in req.modalities:
            lab_panels = ["cbc", "metabolic_panel", "lactate", "inflammatory_markers"]
        if Modality.VITALS in req.modalities:
            vital_signs = ["heart_rate", "systolic_bp", "spo2", "temperature"]
        if Modality.TIME_SERIES in req.modalities:
            time_series_channels = ["heart_rate", "systolic_bp", "spo2", "lactate"]
        if Modality.IMAGING in req.modalities:
            imaging_views = ["portable_chest_xray"]

        return ModalityPlan(
            topic=req.topic,
            cohort_size=req.count,
            modalities=req.modalities,
            cohort_constraints=req.cohort_constraints,
            document_types=document_types,
            lab_panels=lab_panels,
            vital_signs=vital_signs,
            time_series_channels=time_series_channels,
            imaging_views=imaging_views,
        )
