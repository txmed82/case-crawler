import pytest

from casecrawler.generation.recipes import (
    apply_generation_recipe,
    list_generation_recipes,
)
from casecrawler.models.dataset import ExportFormat, GenerationRequest
from casecrawler.models.synthetic import ComplexityProfile, Modality


def test_generation_recipe_catalog_includes_multimodal_training_profiles():
    recipes = {recipe.name: recipe for recipe in list_generation_recipes()}

    assert "full_multimodal_acute_care" in recipes
    assert Modality.IMAGING in recipes["full_multimodal_acute_care"].modalities
    assert Modality.TIME_SERIES in recipes["full_multimodal_acute_care"].modalities
    assert ExportFormat.MULTIMODAL_JSONL in recipes["radiology_cxr_report"].export_formats
    assert recipes["icu_timeseries_notes"].cohort_constraints["topic_mix"]
    assert recipes["longitudinal_chronic_care"].cohort_constraints["encounter_count"] == 4
    assert ExportFormat.MEDICATION_RECONCILIATION_JSONL in (
        recipes["longitudinal_chronic_care"].export_formats
    )
    assert "synthchex_75k" in recipes["radiology_cxr_report"].recommended_reference_keys
    assert "rexgradient_160k" in recipes["radiology_cxr_report"].recommended_reference_keys
    assert "synthea_fhir" in recipes["full_multimodal_acute_care"].recommended_reference_keys
    assert "medsynth_dialogue_note" in recipes["full_multimodal_acute_care"].recommended_reference_keys
    assert "synthea_fhir" in recipes["icu_timeseries_notes"].recommended_reference_keys
    assert "technetium_i" in recipes["icu_timeseries_notes"].recommended_reference_keys
    assert (
        "clinical_timeseries_reference"
        in recipes["icu_timeseries_notes"].recommended_reference_keys
    )
    assert recipes["full_multimodal_acute_care"].benchmark_min_overall_score == 0.7


def test_apply_generation_recipe_fills_default_request_fields():
    req = apply_generation_recipe(
        GenerationRequest(topic="acute care", recipe="full_multimodal_acute_care")
    )

    assert req.complexity == ComplexityProfile.COMPLEX
    assert req.validation_threshold == 0.75
    assert req.modalities == [
        Modality.STRUCTURED_EHR,
        Modality.CLINICAL_TEXT,
        Modality.LABS,
        Modality.VITALS,
        Modality.TIME_SERIES,
        Modality.IMAGING,
    ]
    assert req.export_formats == [
        ExportFormat.MULTIMODAL_JSONL,
        ExportFormat.SFT_JSONL,
        ExportFormat.NOTE_FACT_SFT_JSONL,
        ExportFormat.CLINICAL_OBSERVATION_JSONL,
        ExportFormat.MEDICATION_RECONCILIATION_JSONL,
        ExportFormat.TIME_SERIES_JSONL,
        ExportFormat.FHIR_NDJSON,
        ExportFormat.PARQUET,
    ]
    assert req.cohort_constraints["topic_mix"][0]["topic"] == "sepsis"


def test_apply_generation_recipe_preserves_explicit_overrides():
    req = apply_generation_recipe(
        GenerationRequest(
            topic="acute care",
            recipe="full_multimodal_acute_care",
            modalities=[Modality.CLINICAL_TEXT],
            export_formats=[ExportFormat.CHAT_JSONL],
            complexity=ComplexityProfile.RARE,
            cohort_constraints={"age_min": 70},
        )
    )

    assert req.modalities == [Modality.CLINICAL_TEXT]
    assert req.export_formats == [ExportFormat.CHAT_JSONL]
    assert req.complexity == ComplexityProfile.RARE
    assert req.cohort_constraints["age_min"] == 70
    assert req.cohort_constraints["age_max"] == 88


def test_icu_recipe_recommends_note_fact_sft_export():
    req = apply_generation_recipe(
        GenerationRequest(topic="sepsis", recipe="icu_timeseries_notes")
    )

    assert ExportFormat.NOTE_FACT_SFT_JSONL in req.export_formats
    assert ExportFormat.CLINICAL_OBSERVATION_JSONL in req.export_formats
    assert ExportFormat.MEDICATION_RECONCILIATION_JSONL in req.export_formats
    assert ExportFormat.TIME_SERIES_JSONL in req.export_formats


def test_chronic_care_recipe_builds_longitudinal_training_request():
    req = apply_generation_recipe(
        GenerationRequest(topic="chronic care", recipe="longitudinal_chronic_care")
    )

    assert req.modalities == [
        Modality.STRUCTURED_EHR,
        Modality.CLINICAL_TEXT,
        Modality.LABS,
        Modality.VITALS,
        Modality.TIME_SERIES,
    ]
    assert req.cohort_constraints["encounter_count"] == 4
    assert req.cohort_constraints["topic_mix"][0]["topic"] == "type 2 diabetes"
    assert ExportFormat.CLINICAL_OBSERVATION_JSONL in req.export_formats
    assert ExportFormat.MEDICATION_RECONCILIATION_JSONL in req.export_formats


def test_apply_generation_recipe_rejects_unknown_recipe():
    with pytest.raises(ValueError, match="Unknown generation recipe"):
        apply_generation_recipe(GenerationRequest(topic="sepsis", recipe="missing"))
