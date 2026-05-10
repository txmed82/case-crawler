from __future__ import annotations

from pydantic import BaseModel, Field

from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import Modality


class ModalityPlan(BaseModel):
    """Plan for which time-series channels and imaging views to generate.

    Earlier revisions also carried `document_types` / `lab_panels` /
    `vital_signs` fields, but those were populated by ``ModalityPlanner.build``
    and never read by any downstream generator (the structured + text
    generators consult the modality list directly), so they have been removed.
    """

    topic: str
    cohort_size: int
    modalities: list[Modality]
    cohort_constraints: dict = Field(default_factory=dict)
    time_series_channels: list[str] = Field(default_factory=list)
    imaging_views: list[str] = Field(default_factory=list)


class ModalityPlanner:
    def build(self, req: GenerationRequest) -> ModalityPlan:
        time_series_channels: list[str] = []
        imaging_views: list[str] = []

        if Modality.TIME_SERIES in req.modalities:
            time_series_channels = [
                "heart_rate",
                "systolic_bp",
                "spo2",
                "lactate",
                "ecg_lead_ii",
                "pleth",
            ]
        if Modality.IMAGING in req.modalities:
            imaging_views = ["portable_chest_xray"]

        return ModalityPlan(
            topic=req.topic,
            cohort_size=req.count,
            modalities=req.modalities,
            cohort_constraints=req.cohort_constraints,
            time_series_channels=time_series_channels,
            imaging_views=imaging_views,
        )
