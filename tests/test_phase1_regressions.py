"""Regression tests for Phase 1 critical-bug fixes.

Each test below maps to one fix from the Phase 1 punch list. If any of these
start failing, a regression has reintroduced the original bug.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from casecrawler.generation.structured_generator import (
    _keyword_matches,
    _profile_for_topic,
)
from casecrawler.llm.anthropic_provider import AnthropicProvider
from casecrawler.llm.openai_provider import OpenAIProvider
from casecrawler.llm.openrouter_provider import OpenRouterProvider
from casecrawler.models.synthetic import (
    ComplexityProfile,
    LabObservation,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    VitalObservation,
)
from casecrawler.validation.clinical_rules import validate_vitals
from casecrawler.validation.synthetic_validator import SyntheticValidator


# --- Fix 1: word-boundary topic matching --------------------------------------


def test_pe_keyword_matches_word_not_substring():
    # The "pe" PE keyword must not greedily match other words that contain "pe".
    assert _keyword_matches("pe", "suspected pe with chest pain")
    assert _keyword_matches("pe", "PE on imaging")
    assert not _keyword_matches("pe", "preeclampsia")
    assert not _keyword_matches("pe", "type 2 diabetes")
    assert not _keyword_matches("pe", "petechiae")


def test_preeclampsia_topic_does_not_route_to_pulmonary_embolism_profile():
    # Preeclampsia is not in any profile; it must hit the fallback, not the PE
    # profile (which would attach heparin, troponin, BNP — clinically wrong).
    profile = _profile_for_topic("preeclampsia")
    assert profile.diagnosis_code != "pulmonary_embolism"
    medication_names = {m["name"].lower() for m in profile.medications}
    assert "heparin" not in medication_names


def test_pe_topic_still_routes_to_pulmonary_embolism():
    profile = _profile_for_topic("suspected pe")
    assert profile.diagnosis_code == "pulmonary_embolism"


# --- Fix 2: validation threshold gate -----------------------------------------


def _vital_record(value: float, unit: str = "C") -> SyntheticRecord:
    return SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="generic",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.VITALS],
        patient=SyntheticPatient(patient_id="pat-1", age=40, sex="female"),
        encounters=[],
        labs=[],
        vitals=[
            VitalObservation(
                name="Temperature",
                value=value,
                unit=unit,
                effective_time="2026-05-06T08:00:00",
            ),
        ],
        provenance=Provenance(
            generator="unit-test",
            created_at="2026-05-06T09:00:00",
        ),
    )


def test_threshold_zero_accepts_records_with_clinical_errors():
    # Two clinical errors (lab + vitals) drive clinical_score to 0.5. With a
    # threshold of 0.0, the record must still be approved — the previous
    # `and not issues` short-circuit rejected anything with even one issue.
    bad = SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.LABS, Modality.VITALS],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        encounters=[],
        labs=[
            LabObservation(
                name="Sodium",
                value=130,
                unit="mmol/L",
                reference_low=135,
                reference_high=145,
                flag=None,  # missing flag → clinical error
                effective_time="2026-05-06T08:30:00",
            )
        ],
        vitals=[
            VitalObservation(
                name="HR",
                value=999,  # implausible HR → clinical error
                unit="/min",
                effective_time="2026-05-06T08:00:00",
            )
        ],
        provenance=Provenance(
            generator="unit-test",
            created_at="2026-05-06T09:00:00",
        ),
    )

    strict = SyntheticValidator(threshold=0.8).validate(bad)
    lenient = SyntheticValidator(threshold=0.0).validate(bad)

    assert strict.approved is False
    assert lenient.approved is True, (
        "threshold=0.0 should accept records when scores meet the bar, "
        "regardless of how many issues fired"
    )
    assert lenient.issues, "issues list must still be populated for lenient mode"


# --- Fix 3: temperature unit handling -----------------------------------------


def test_temperature_in_fahrenheit_within_range_passes():
    record = _vital_record(98.6, unit="F")
    issues = validate_vitals(record)
    temp_issues = [i for i in issues if i.field == "vitals.temperature"]
    assert temp_issues == [], f"98.6F should be plausible, got: {temp_issues}"


def test_temperature_in_fahrenheit_out_of_range_fails():
    # 200F = ~93C, well outside plausible
    record = _vital_record(200.0, unit="F")
    issues = validate_vitals(record)
    assert any(i.field == "vitals.temperature" for i in issues)


def test_temperature_in_celsius_still_validated():
    record = _vital_record(50.0, unit="C")  # 50C is biologically implausible
    issues = validate_vitals(record)
    assert any(i.field == "vitals.temperature" for i in issues)

    ok = _vital_record(37.5, unit="C")
    assert all(i.field != "vitals.temperature" for i in validate_vitals(ok))


# --- Fix 4: Anthropic empty content guard -------------------------------------


@pytest.mark.asyncio
async def test_anthropic_generate_raises_on_empty_content():
    provider = AnthropicProvider(api_key="test", model="claude-opus-4-7")
    mock_response = MagicMock()
    mock_response.content = []
    mock_response.stop_reason = "max_tokens"
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 0
    mock_response.model = "claude-opus-4-7"

    with patch.object(
        provider._client.messages,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        with pytest.raises(ValueError) as exc_info:
            await provider.generate("hello")
    message = str(exc_info.value)
    assert "no content blocks" in message
    assert "max_tokens" in message, "stop_reason should be surfaced in the error"


@pytest.mark.asyncio
async def test_anthropic_generate_raises_on_non_text_first_block():
    provider = AnthropicProvider(api_key="test", model="claude-opus-4-7")
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="tool_use")]
    mock_response.stop_reason = "tool_use"
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 0
    mock_response.model = "claude-opus-4-7"

    with patch.object(
        provider._client.messages,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        with pytest.raises(ValueError) as exc_info:
            await provider.generate("hello")
    message = str(exc_info.value)
    assert "non-text first block" in message
    assert "tool_use" in message, "block type should be surfaced in the error"


# --- Fix 5: OpenAI / OpenRouter ValidationError handling ----------------------


class _DemoSchema(BaseModel):
    age: int
    sex: str


def _mock_chat_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    response.model = "test-model"
    return response


@pytest.mark.asyncio
async def test_openai_structured_invalid_json_raises_value_error():
    provider = OpenAIProvider(api_key="test", model="gpt-4")
    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=_mock_chat_response("not valid json {"),
    ):
        with pytest.raises(ValueError, match="not valid JSON"):
            await provider.generate_structured("p", _DemoSchema)


@pytest.mark.asyncio
async def test_openai_structured_schema_mismatch_raises_value_error():
    provider = OpenAIProvider(api_key="test", model="gpt-4")
    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=_mock_chat_response(json.dumps({"unrelated": "shape"})),
    ):
        with pytest.raises(ValueError, match="did not match schema"):
            await provider.generate_structured("p", _DemoSchema)


@pytest.mark.asyncio
async def test_openrouter_structured_schema_mismatch_raises_value_error():
    provider = OpenRouterProvider(api_key="test", model="anthropic/claude-opus")
    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=_mock_chat_response(json.dumps({"unrelated": "shape"})),
    ):
        with pytest.raises(ValueError, match="did not match schema"):
            await provider.generate_structured("p", _DemoSchema)


# --- Fix 6: export manifest only on successful completion ---------------------


def test_export_manifest_not_written_on_partial_stream(monkeypatch):
    """The streaming export endpoint must NOT record a completed manifest if
    the iteration was aborted mid-stream (client disconnect, generator error).
    Previously the `finally` block fired on every code path, recording false
    manifest entries with truncated record counts.

    This test exercises the real `stream_jsonl_export` helper from
    `casecrawler.api.routes.datasets` so a regression in the production code
    path actually fails the test.
    """

    from casecrawler.api.routes import datasets as datasets_routes
    from casecrawler.models.dataset import ExportFormat

    saved_calls: list[dict] = []

    class _StubStore:
        def save_export_manifest(self, **kwargs):
            saved_calls.append(kwargs)

    # `export_record_payloads` is patched to (a) yield one payload per record
    # in the happy path, and (b) raise after the first record on the failure
    # path — simulating a downstream serialization error or client disconnect.
    def _ok_payloads(record, _fmt):
        yield {"record_id": record}

    def _explode_payloads(record, _fmt):
        if record == "rec-1":
            yield {"record_id": record}
            return
        raise RuntimeError("client disconnect simulation")

    # --- happy path: full iteration writes exactly one manifest ---
    monkeypatch.setattr(datasets_routes, "export_record_payloads", _ok_payloads)
    list(
        datasets_routes.stream_jsonl_export(
            store=_StubStore(),
            dataset_id="ds-x",
            export_format=ExportFormat.SFT_JSONL,
            records=["rec-1", "rec-2", "rec-3"],
            benchmark_metadata={"k": "v"},
        )
    )
    assert len(saved_calls) == 1
    assert saved_calls[0]["record_count"] == 3
    assert saved_calls[0]["dataset_id"] == "ds-x"
    assert saved_calls[0]["metadata"]["transport"] == "api"
    assert saved_calls[0]["metadata"]["k"] == "v"

    # --- partial / aborted stream: no manifest written ---
    saved_calls.clear()
    monkeypatch.setattr(datasets_routes, "export_record_payloads", _explode_payloads)
    gen = datasets_routes.stream_jsonl_export(
        store=_StubStore(),
        dataset_id="ds-y",
        export_format=ExportFormat.SFT_JSONL,
        records=["rec-1", "rec-2", "rec-3"],
        benchmark_metadata={},
    )
    with pytest.raises(RuntimeError):
        list(gen)
    assert saved_calls == [], (
        "no manifest may be saved when stream_jsonl_export aborts mid-stream"
    )
