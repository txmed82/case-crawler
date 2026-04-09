from __future__ import annotations

from pydantic import BaseModel


class VitalSigns(BaseModel):
    hr: int
    bp_systolic: int
    bp_diastolic: int
    rr: int
    spo2: float
    temp_c: float
    gcs: int | None = None


class LabValue(BaseModel):
    name: str
    value: float
    unit: str
    reference_low: float
    reference_high: float
    flag: str | None = None


class LabResult(BaseModel):
    panel: str
    values: list[LabValue]
    timestamp: str


class ImagingFinding(BaseModel):
    structure: str
    observation: str
    severity: str | None = None
    laterality: str | None = None


class ImagingResult(BaseModel):
    modality: str
    body_region: str
    indication: str
    findings: list[ImagingFinding]
    impression: str
    timestamp: str
