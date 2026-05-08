import pytest

from casecrawler.generation.imaging_models import (
    list_imaging_model_profiles,
    resolve_imaging_model_profile,
)


def test_imaging_model_catalog_lists_medical_profiles():
    profiles = {profile.name: profile for profile in list_imaging_model_profiles()}

    assert "prompt2medimage" in profiles
    assert "medisyn" in profiles
    assert "stable_diffusion_chest_xray" in profiles
    assert "cxr_pneumonia_dreambooth" in profiles
    assert profiles["medisyn"].model_id == "hiesingerlab/MediSyn"
    assert profiles["medisyn"].license == "cc-by-nc-nd-4.0"
    assert profiles["medisyn"].is_compatible("CT", "abdomen") is True
    assert profiles["cxr_pneumonia_dreambooth"].modality == "XR"
    assert profiles["cxr_pneumonia_dreambooth"].is_compatible("XR", "chest") is True
    assert profiles["cxr_pneumonia_dreambooth"].is_compatible("CT", "abdomen") is False
    assert profiles["prompt2medimage"].is_compatible("CT", "abdomen") is True
    assert profiles["roentgen_v2_gated"].license == "restricted"


def test_resolve_imaging_model_profile_rejects_unknown_profiles():
    with pytest.raises(ValueError, match="Unknown imaging model profile"):
        resolve_imaging_model_profile("missing")
