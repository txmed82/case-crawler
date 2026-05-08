from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ComplexityProfile(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    RARE = "rare"


class Modality(str, Enum):
    STRUCTURED_EHR = "structured_ehr"
    CLINICAL_TEXT = "clinical_text"
    LABS = "labs"
    VITALS = "vitals"
    TIME_SERIES = "time_series"
    IMAGING = "imaging"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Code(StrictModel):
    system: str
    code: str
    display: str


class Provenance(StrictModel):
    generator: str
    model: str | None = None
    source_refs: list[dict] = Field(default_factory=list)
    prompt_hash: str | None = None
    created_at: str


class SyntheticPatient(StrictModel):
    patient_id: str
    age: int
    sex: str
    demographics: dict = Field(default_factory=dict)
    social_history: dict = Field(default_factory=dict)


class Encounter(StrictModel):
    encounter_id: str
    start: str
    end: str | None = None
    setting: str
    reason: str
    diagnoses: list[Code] = Field(default_factory=list)
    procedures: list[Code] = Field(default_factory=list)


class LabObservation(StrictModel):
    name: str
    loinc: str | None = None
    value: float | str
    unit: str
    reference_low: float | None = None
    reference_high: float | None = None
    flag: str | None = None
    effective_time: str
    specimen: str | None = None


class VitalObservation(StrictModel):
    name: str
    value: float
    unit: str
    effective_time: str


class MedicationStatement(StrictModel):
    name: str
    rxnorm: str | None = None
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    status: str = "unknown"
    start: str | None = None
    end: str | None = None


class AllergyIntolerance(StrictModel):
    substance: str
    code: str | None = None
    system: str | None = None
    reaction: str | None = None
    severity: str | None = None
    status: str = "active"
    recorded_at: str | None = None


class TimeSeriesPoint(StrictModel):
    timestamp: str
    values: dict[str, float]


class TimeSeriesChannel(StrictModel):
    name: str
    unit: str
    generation_backend: str = "deterministic"
    sampling_rate_hz: float | None = None
    points: list[TimeSeriesPoint]


class ClinicalDocument(StrictModel):
    document_id: str
    note_type: str
    author_role: str
    timestamp: str
    clean_text: str
    messy_text: str | None = None
    extracted_facts: dict = Field(default_factory=dict)


class ImagingAsset(StrictModel):
    image_id: str
    modality: str
    body_region: str
    prompt: str
    file_path: str | None = None
    report_text: str
    labels: list[Code] = Field(default_factory=list)
    generation_backend: str


class ValidationIssue(StrictModel):
    severity: str
    modality: Modality
    field: str
    message: str


class ValidationReport(StrictModel):
    schema_score: float
    clinical_consistency_score: float
    privacy_score: float
    utility_score: float
    approved: bool
    modality_alignment_score: float | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)


class SyntheticRecord(StrictModel):
    record_id: str
    dataset_id: str
    topic: str
    complexity: ComplexityProfile
    modalities: list[Modality]
    patient: SyntheticPatient
    encounters: list[Encounter]
    labs: list[LabObservation] = Field(default_factory=list)
    vitals: list[VitalObservation] = Field(default_factory=list)
    medication_history: list[MedicationStatement] = Field(default_factory=list)
    allergies: list[AllergyIntolerance] = Field(default_factory=list)
    time_series: list[TimeSeriesChannel] = Field(default_factory=list)
    documents: list[ClinicalDocument] = Field(default_factory=list)
    imaging: list[ImagingAsset] = Field(default_factory=list)
    provenance: Provenance
    validation: ValidationReport | None = None
    metadata: dict = Field(default_factory=dict)
