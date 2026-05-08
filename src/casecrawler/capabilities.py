from __future__ import annotations

from casecrawler.export.fine_tuning import REQUIRED_RELEASE_COVERAGE_KEYS


def release_coverage_requirements() -> list[dict[str, str]]:
    """Return strict multimodal release coverage requirements for clients."""
    return [
        {
            "key": key,
            "description": release_coverage_requirement_description(key),
        }
        for key in sorted(REQUIRED_RELEASE_COVERAGE_KEYS)
    ]


def release_coverage_requirement_description(key: str) -> str:
    descriptions = {
        "approved_records": "Every record is validation-approved or human-review approved.",
        "benchmark_reference": (
            "At least one recommended benchmark reference dataset is resolved."
        ),
        "discharge_summaries": "Clinical text includes physician discharge summaries.",
        "imaging_model_policy": (
            "Generated image records include model license/use-policy metadata."
        ),
        "lab_reports": "Clinical text includes lab-report documents.",
        "labs": "Structured lab observations are present.",
        "medication_administration_records": (
            "Clinical text includes medication administration records."
        ),
        "medication_history": "Structured medication history is present.",
        "messy_clinical_text": "Clinical documents include messy or noisy text variants.",
        "modality_alignment_scores": "Image/report alignment validation scores are present.",
        "no_blocking_quality_issues": "Quality validation reports no blocking issues.",
        "nursing_notes": "Clinical text includes nursing notes.",
        "physician_notes": "Clinical text includes physician-authored notes.",
        "radiology_images": "Radiology image assets are backed by local image files.",
        "radiology_reports": "Clinical text includes radiology reports.",
        "records": "The dataset has at least one record.",
        "structured_ehr": "Structured encounters, diagnoses, and extracted facts are present.",
        "task_reference_coverage": (
            "Task-specific benchmark reference readiness is complete."
        ),
        "time_series": "Time-series channels and points are present.",
        "validation_reports": "Every record has a validation report.",
        "vital_signs_flowsheets": "Clinical text includes vital-sign flowsheets.",
        "vitals": "Structured vital observations are present.",
    }
    return descriptions.get(key, key.replace("_", " "))
