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
        "clinical_text_model_policy": (
            "Clinical text records include generator license/use-policy metadata."
        ),
        "allergy_intolerances": (
            "Structured allergy/intolerance safety facts are present."
        ),
        "clinical_orders": (
            "Structured clinical orders are present for order-aware training tasks."
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


def reference_dataset_capabilities() -> list[dict[str, object]]:
    """Return configured benchmark/reference datasets for clients."""
    from casecrawler.integrations.huggingface import REFERENCE_DATASETS
    from casecrawler.integrations.reference_fixtures import FIXTURE_REFERENCE_KEYS
    from casecrawler.integrations.synthea import (
        SYNTHEA_REFERENCE_DESCRIPTION,
        SYNTHEA_REFERENCE_KEY,
    )

    datasets: list[dict[str, object]] = [
        {
            "key": SYNTHEA_REFERENCE_KEY,
            "repo_id": None,
            "split": None,
            "license": "synthetic-local",
            "description": SYNTHEA_REFERENCE_DESCRIPTION,
            "image_field": None,
            "image_label_field": None,
            "image_modality": None,
            "image_body_region": None,
            "lab_values_field": None,
            "vital_values_field": None,
            "medications_field": None,
            "time_series_field": None,
            "gated": False,
            "use_policy": "local_synthea_import",
            "source": "synthea",
            "fixture_available": SYNTHEA_REFERENCE_KEY in FIXTURE_REFERENCE_KEYS,
        }
    ]
    datasets.extend(
        {
            "key": key,
            "repo_id": spec.repo_id,
            "split": spec.split,
            "license": spec.license,
            "description": spec.description,
            "image_field": spec.image_field,
            "image_label_field": spec.image_label_field,
            "image_modality": spec.image_modality,
            "image_body_region": spec.image_body_region,
            "lab_values_field": spec.lab_values_field,
            "vital_values_field": spec.vital_values_field,
            "medications_field": spec.medications_field,
            "time_series_field": spec.time_series_field,
            "gated": spec.gated,
            "use_policy": spec.use_policy,
            "source": "huggingface",
            "fixture_available": key in FIXTURE_REFERENCE_KEYS,
        }
        for key, spec in REFERENCE_DATASETS.items()
    )
    known_keys = {str(dataset["key"]) for dataset in datasets}
    for key in FIXTURE_REFERENCE_KEYS:
        if key in known_keys:
            continue
        datasets.append(
            {
                "key": key,
                "repo_id": None,
                "split": "fixture",
                "license": "synthetic-fixture",
                "description": _fixture_reference_description(key),
                "image_field": None,
                "image_label_field": None,
                "image_modality": None,
                "image_body_region": None,
                "lab_values_field": None,
                "vital_values_field": None,
                "medications_field": None,
                "time_series_field": "time_series",
                "gated": False,
                "use_policy": "offline_benchmark_fixture",
                "source": "casecrawler-fixture",
                "fixture_available": True,
            }
        )
    return datasets


def _fixture_reference_description(key: str) -> str:
    descriptions = {
        "clinical_timeseries_reference": (
            "Bundled ICU-style synthetic time-series reference with labs, vitals, "
            "medication history, and nursing-note context."
        )
    }
    return descriptions.get(key, "Bundled synthetic benchmark fixture.")


def image_validator_capabilities() -> list[dict[str, object]]:
    """Return configured image-text validator backends for clients."""
    from casecrawler.validation.image_alignment import list_image_validator_profiles

    return [
        {
            "key": profile.key,
            "backend": profile.backend,
            "description": profile.description,
            "requires": profile.requires,
            "model_id": profile.model_id,
            "license": profile.license,
            "gated": profile.gated,
            "use_policy": profile.use_policy,
            "notes": profile.notes,
        }
        for profile in list_image_validator_profiles()
    ]


def clinical_text_model_capabilities() -> list[dict[str, object]]:
    """Return configured clinical text generation adapter profiles."""
    from casecrawler.generation.clinical_text_models import (
        list_clinical_text_model_profiles,
    )

    return [
        {
            "name": profile.name,
            "adapter_type": profile.adapter_type,
            "reference": profile.reference,
            "model_id": profile.model_id,
            "license": profile.license,
            "gated": profile.gated,
            "use_policy": profile.use_policy,
            "command_template": profile.command_template,
            "input_contract": profile.input_contract,
            "output_contract": profile.output_contract,
            "validation_requirements": profile.validation_requirements,
            "notes": profile.notes,
        }
        for profile in list_clinical_text_model_profiles()
    ]
