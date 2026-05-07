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


def test_resolve_time_series_model_profile_rejects_unknown_profiles():
    with pytest.raises(ValueError, match="Unknown time-series model profile"):
        resolve_time_series_model_profile("missing")
