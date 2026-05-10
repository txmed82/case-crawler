"""Regression tests for Phase 2 (dead-code removal + legacy-key fail-fast)."""

from __future__ import annotations

import importlib

import pytest

from casecrawler.models.dataset import GenerationRequest


def test_legacy_min_age_raises_with_migration_hint():
    with pytest.raises(ValueError) as exc:
        GenerationRequest(topic="sepsis", cohort_constraints={"min_age": 30})
    msg = str(exc.value)
    assert "min_age" in msg
    assert "age_min" in msg
    assert "retired" in msg.lower() or "no longer" in msg.lower()


def test_legacy_max_age_raises():
    with pytest.raises(ValueError) as exc:
        GenerationRequest(topic="sepsis", cohort_constraints={"max_age": 80})
    assert "max_age" in str(exc.value)
    assert "age_max" in str(exc.value)


def test_legacy_sex_cycle_raises():
    with pytest.raises(ValueError) as exc:
        GenerationRequest(topic="sepsis", cohort_constraints={"sex_cycle": "male"})
    assert "sex_cycle" in str(exc.value)
    assert "sexes" in str(exc.value)


def test_canonical_keys_accepted():
    req = GenerationRequest(
        topic="sepsis",
        cohort_constraints={"age_min": 30, "age_max": 80, "sexes": ["male", "female"]},
    )
    assert req.cohort_constraints["age_min"] == 30


def test_diagnostics_module_is_gone():
    with pytest.raises(ModuleNotFoundError) as exc:
        importlib.import_module("casecrawler.models.diagnostics")
    assert exc.value.name == "casecrawler.models.diagnostics"


def test_ingest_first_field_removed():
    fields = GenerationRequest.model_fields
    assert "ingest_first" not in fields
