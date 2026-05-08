import pytest

from casecrawler.generation.imaging_models import (
    list_imaging_model_profiles,
    resolve_imaging_model_profile,
)


def test_imaging_model_catalog_lists_medical_profiles():
    profiles = {profile.name: profile for profile in list_imaging_model_profiles()}

    assert "prompt2medimage" in profiles
    assert "medisyn" in profiles
    assert "chexgenbench_sana_e20" in profiles
    assert "stable_diffusion_chest_xray" in profiles
    assert "cxr_pneumonia_dreambooth" in profiles
    assert profiles["medisyn"].model_id == "hiesingerlab/MediSyn"
    assert profiles["medisyn"].license == "cc-by-nc-nd-4.0"
    assert profiles["medisyn"].gated is False
    assert profiles["medisyn"].use_policy == "non_commercial_no_derivatives_review_before_release"
    assert profiles["medisyn"].adapter_type == "diffusers"
    assert profiles["medisyn"].input_contract["inputs"] == [
        "prompt",
        "negative_prompt",
        "modality",
        "body_region",
    ]
    assert profiles["medisyn"].output_contract["artifact"] == "ImagingAsset"
    assert "image_file_signature" in profiles["medisyn"].validation_requirements
    assert profiles["medisyn"].is_compatible("CT", "abdomen") is True
    assert profiles["cxr_pneumonia_dreambooth"].modality == "XR"
    assert profiles["cxr_pneumonia_dreambooth"].is_compatible("XR", "chest") is True
    assert profiles["cxr_pneumonia_dreambooth"].is_compatible("CT", "abdomen") is False
    assert (
        profiles["chexgenbench_sana_e20"].model_id
        == "raman07/CheXGenBench-Models-Sana-e20"
    )
    assert profiles["chexgenbench_sana_e20"].license is None
    assert profiles["chexgenbench_sana_e20"].use_policy == (
        "model_card_missing_review_terms_and_validate_privacy_utility"
    )
    assert profiles["chexgenbench_sana_e20"].adapter_type == "diffusers"
    assert profiles["chexgenbench_sana_e20"].command_template == [
        "casecrawler",
        "generate-dataset",
        "<topic>",
        "--imaging-backend",
        "diffusers",
        "--imaging-model-profile",
        "chexgenbench_sana_e20",
    ]
    assert (
        "radiology_label_evidence"
        in profiles["chexgenbench_sana_e20"].validation_requirements
    )
    assert profiles["chexgenbench_sana_e20"].is_compatible("XR", "chest") is True
    assert profiles["prompt2medimage"].is_compatible("CT", "abdomen") is True
    assert profiles["roentgen_v2_gated"].license == "restricted"
    assert profiles["roentgen_v2_gated"].gated is True
    assert profiles["roentgen_v2_gated"].use_policy == "credentialed_mimic_cxr_terms_required"


def test_resolve_imaging_model_profile_rejects_unknown_profiles():
    with pytest.raises(ValueError, match="Unknown imaging model profile"):
        resolve_imaging_model_profile("missing")
