import pytest

from casecrawler.generation.clinical_text_models import (
    list_clinical_text_model_profiles,
    resolve_clinical_text_model_profile,
)


def test_clinical_text_model_catalog_lists_external_note_generators():
    profiles = {
        profile.name: profile for profile in list_clinical_text_model_profiles()
    }

    assert profiles["medgemma_4b_it"].adapter_type == "external_command"
    assert profiles["medgemma_4b_it"].model_id == "google/medgemma-4b-it"
    assert profiles["medgemma_4b_it"].license == "health-ai-developer-foundations"
    assert profiles["medgemma_4b_it"].command_template == [
        "hf-note-sample",
        "--model",
        "google/medgemma-4b-it",
    ]
    assert profiles["medgemma_4b_it"].output_contract["stdout_json"] == (
        "ClinicalDocument[] or {'documents': ClinicalDocument[]}"
    )
    assert "privacy_phi_scan" in profiles["medgemma_4b_it"].validation_requirements
    assert profiles["meditron_7b"].model_id == "epfl-llm/meditron-7b"
    assert profiles["meditron_7b"].license == "llama2"
    assert (
        profiles["generic_external_note_generator"].use_policy
        == "bring_your_own_model_validate_outputs"
    )


def test_resolve_clinical_text_model_profile_rejects_unknown_profiles():
    with pytest.raises(ValueError, match="Unknown clinical text model profile"):
        resolve_clinical_text_model_profile("missing")
