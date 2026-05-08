import pytest

from casecrawler.generation.timeseries_models import (
    list_time_series_model_profiles,
    resolve_time_series_model_profile,
)


def test_time_series_model_catalog_lists_external_references():
    profiles = {profile.name: profile for profile in list_time_series_model_profiles()}

    assert "timediff" in profiles
    assert "rawmed" in profiles
    assert profiles["timediff"].adapter_type == "external_command"
    assert profiles["timediff"].model_id == "MuhangTian/TimeDiff"
    assert profiles["timediff"].license == "mit"
    assert profiles["timediff"].gated is False
    assert profiles["timediff"].use_policy == "wrap_external_sampler_validate_outputs"
    assert profiles["rawmed"].use_policy == "research_reference_validate_outputs"


def test_resolve_time_series_model_profile_rejects_unknown_profiles():
    with pytest.raises(ValueError, match="Unknown time-series model profile"):
        resolve_time_series_model_profile("missing")
