from __future__ import annotations

from dataclasses import dataclass, field

from casecrawler.models.dataset import ExportFormat, GenerationRequest
from casecrawler.models.synthetic import ComplexityProfile, Modality


@dataclass(frozen=True)
class GenerationRecipe:
    name: str
    description: str
    modalities: list[Modality]
    export_formats: list[ExportFormat]
    complexity: ComplexityProfile = ComplexityProfile.MODERATE
    cohort_constraints: dict = field(default_factory=dict)
    validation_threshold: float = 0.8
    recommended_reference_keys: list[str] = field(default_factory=list)
    benchmark_min_overall_score: float = 0.75
    benchmark_min_metric_score: float = 0.5


RECIPES: dict[str, GenerationRecipe] = {
    "full_multimodal_acute_care": GenerationRecipe(
        name="full_multimodal_acute_care",
        description=(
            "Broad acute-care training set with structured EHR, notes, labs, "
            "vitals, time series, imaging assets, and multimodal exports."
        ),
        complexity=ComplexityProfile.COMPLEX,
        modalities=[
            Modality.STRUCTURED_EHR,
            Modality.CLINICAL_TEXT,
            Modality.LABS,
            Modality.VITALS,
            Modality.TIME_SERIES,
            Modality.IMAGING,
        ],
        export_formats=[
            ExportFormat.MULTIMODAL_JSONL,
            ExportFormat.SFT_JSONL,
            ExportFormat.FHIR_NDJSON,
            ExportFormat.PARQUET,
        ],
        cohort_constraints={
            "topic_mix": [
                {"topic": "sepsis", "weight": 2},
                {"topic": "pneumonia", "weight": 2},
                {"topic": "heart failure", "weight": 1},
                {"topic": "pulmonary embolism", "weight": 1},
            ],
            "age_min": 45,
            "age_max": 88,
            "sexes": ["female", "male"],
        },
        validation_threshold=0.75,
        recommended_reference_keys=[
            "clinical_notes_to_fhir",
            "synthchex_75k",
            "radiology_report_consistency",
        ],
        benchmark_min_overall_score=0.7,
        benchmark_min_metric_score=0.45,
    ),
    "radiology_cxr_report": GenerationRecipe(
        name="radiology_cxr_report",
        description=(
            "Chest imaging and radiology-report dataset for multimodal "
            "image-text alignment and report-generation fine tuning."
        ),
        modalities=[
            Modality.STRUCTURED_EHR,
            Modality.CLINICAL_TEXT,
            Modality.IMAGING,
        ],
        export_formats=[ExportFormat.MULTIMODAL_JSONL, ExportFormat.SFT_JSONL],
        cohort_constraints={
            "topic_mix": [
                "pneumonia",
                "heart failure",
                "status asthmaticus",
                "pulmonary embolism",
            ],
        },
        validation_threshold=0.7,
        recommended_reference_keys=[
            "synthchex_75k",
            "rexgradient_160k",
            "radiology_report_consistency",
        ],
        benchmark_min_overall_score=0.7,
        benchmark_min_metric_score=0.45,
    ),
    "icu_timeseries_notes": GenerationRecipe(
        name="icu_timeseries_notes",
        description=(
            "Longitudinal ICU-style cohort emphasizing vitals, labs, waveform-like "
            "time series, medication history, and nursing/physician notes."
        ),
        complexity=ComplexityProfile.COMPLEX,
        modalities=[
            Modality.STRUCTURED_EHR,
            Modality.CLINICAL_TEXT,
            Modality.LABS,
            Modality.VITALS,
            Modality.TIME_SERIES,
        ],
        export_formats=[ExportFormat.SFT_JSONL, ExportFormat.PARQUET],
        cohort_constraints={
            "topic_mix": [
                {"topic": "sepsis", "weight": 2},
                {"topic": "acute kidney injury", "weight": 1},
                {"topic": "diabetic ketoacidosis", "weight": 1},
            ],
            "age_min": 35,
            "age_max": 90,
        },
        validation_threshold=0.8,
        recommended_reference_keys=[
            "synthclinicalnotes",
            "augmented_clinical_notes",
            "clinical_notes_to_fhir",
        ],
        benchmark_min_overall_score=0.75,
        benchmark_min_metric_score=0.5,
    ),
}


def list_generation_recipes() -> list[GenerationRecipe]:
    return list(RECIPES.values())


def apply_generation_recipe(req: GenerationRequest) -> GenerationRequest:
    if not req.recipe:
        return req
    try:
        recipe = RECIPES[req.recipe]
    except KeyError as exc:
        choices = ", ".join(sorted(RECIPES))
        raise ValueError(f"Unknown generation recipe {req.recipe!r}. Choose from: {choices}.") from exc

    default = GenerationRequest(topic=req.topic)
    cohort_constraints = {
        **recipe.cohort_constraints,
        **req.cohort_constraints,
    }
    updates = {
        "cohort_constraints": cohort_constraints,
    }
    if req.modalities == default.modalities:
        updates["modalities"] = recipe.modalities
    if req.export_formats == default.export_formats:
        updates["export_formats"] = recipe.export_formats
    if req.complexity == default.complexity:
        updates["complexity"] = recipe.complexity
    if req.validation_threshold == default.validation_threshold:
        updates["validation_threshold"] = recipe.validation_threshold
    return req.model_copy(update=updates)
