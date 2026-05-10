"""Regression tests for Phase 4c (real DPO / RL / tool-call exports)."""

from __future__ import annotations

import json

import pytest

from casecrawler.export.fine_tuning import (
    export_dpo_record,
    export_rl_record,
    export_tool_call_record,
)
from casecrawler.generation.judges import (
    recommend_judges,
    warn_if_judge_collides_with_generator,
)
from casecrawler.generation.preference_pipeline import (
    PreferenceCandidate,
    PreferenceConfig,
    PreferencePair,
    build_preference_pair,
)
from casecrawler.models.synthetic import (
    Code,
    ComplexityProfile,
    Encounter,
    GroundingBundle,
    GroundingCitation,
    LabObservation,
    MedicationStatement,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    VitalObservation,
)


def _make_record(*, with_grounding: bool = False) -> SyntheticRecord:
    metadata: dict = {}
    if with_grounding:
        metadata["grounding"] = GroundingBundle(
            topic="sepsis",
            retrieved_at="2026-05-06T09:00:00Z",
            citations=[
                GroundingCitation(
                    chunk_id="c1",
                    source="pubmed",
                    source_document_id="pubmed:1",
                    score=0.9,
                    credibility="guideline",
                )
            ],
        ).model_dump()
    return SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="sepsis",
        complexity=ComplexityProfile.COMPLEX,
        modalities=[Modality.LABS, Modality.VITALS],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        encounters=[
            Encounter(
                encounter_id="enc-1",
                start="2026-05-06T10:00:00",
                end="2026-05-06T14:00:00",
                setting="emergency",
                reason="fever",
                diagnoses=[
                    Code(
                        system="http://snomed.info/sct",
                        code="91302008",
                        display="Sepsis",
                    )
                ],
                procedures=[],
            )
        ],
        labs=[
            LabObservation(
                name="Lactate",
                loinc="2524-7",
                value=4.2,
                unit="mmol/L",
                reference_low=0.5,
                reference_high=2.2,
                flag="high",
                effective_time="2026-05-06T10:15:00",
                specimen="blood",
            ),
            LabObservation(
                name="Sodium",
                value=138,
                unit="mmol/L",
                reference_low=135,
                reference_high=145,
                effective_time="2026-05-06T10:15:00",
            ),
        ],
        vitals=[
            VitalObservation(
                name="Heart rate",
                value=122,
                unit="/min",
                effective_time="2026-05-06T10:10:00",
            ),
        ],
        medication_history=[
            MedicationStatement(
                name="Norepinephrine",
                rxnorm="7512",
                dose="0.08 mcg/kg/min",
                route="IV",
                frequency="continuous",
                start="2026-05-06T11:00:00",
            ),
        ],
        provenance=Provenance(
            generator="unit-test", created_at="2026-05-06T09:00:00"
        ),
        metadata=metadata,
    )


# ---------- Preference pipeline -------------------------------------------


def test_build_preference_pair_returns_typed_pair():
    pair = build_preference_pair(_make_record())
    assert isinstance(pair, PreferencePair)
    assert pair.record_id == "rec-1"
    assert pair.dataset_id == "ds-1"
    assert pair.chosen.score >= pair.rejected.score
    assert pair.chosen.text != pair.rejected.text
    assert all(isinstance(c, PreferenceCandidate) for c in pair.candidates)
    assert pair.selection_strategy in {"abnormal_aware", "rs_dpo"}


def test_build_preference_pair_propagates_grounding_citations():
    pair = build_preference_pair(_make_record(with_grounding=True))
    assert len(pair.citations) == 1
    assert pair.citations[0]["chunk_id"] == "c1"


def test_build_preference_pair_strategy_is_abnormal_aware_when_findings_differ():
    """Lactate is flagged abnormal on the fixture record. The deterministic
    candidate factory produces variants where some surface Lactate and some
    don't, so abnormal_aware selection should kick in."""
    pair = build_preference_pair(_make_record())
    # The chosen candidate should cover at least as many abnormal findings
    # as the rejected one.
    assert (
        pair.chosen.abnormal_findings_covered
        >= pair.rejected.abnormal_findings_covered
    )
    assert pair.abnormal_findings == ["Lactate"]


def test_build_preference_pair_with_custom_factory():
    record = _make_record()
    captured = {}

    def factory(rec, n):
        captured["n"] = n
        return ["high quality response", "low quality"]

    pair = build_preference_pair(
        record,
        config=PreferenceConfig(n_candidates=2),
        candidate_factory=factory,
    )
    assert captured["n"] == 2
    assert {c.text for c in pair.candidates} == {
        "high quality response",
        "low quality",
    }


def test_build_preference_pair_requires_two_candidates():
    record = _make_record()
    with pytest.raises(ValueError, match="at least 2"):
        build_preference_pair(
            record,
            config=PreferenceConfig(n_candidates=1),
            candidate_factory=lambda rec, n: ["only one"],
        )


# ---------- DPO / RL exports use the new pipeline -------------------------


def test_dpo_export_carries_real_preference_data():
    record = _make_record(with_grounding=True)
    payload = export_dpo_record(record)

    assert payload["chosen"][0]["content"]
    assert payload["rejected"][0]["content"]
    assert payload["chosen"][0]["content"] != payload["rejected"][0]["content"]
    assert payload["scores"]["chosen"] >= payload["scores"]["rejected"]
    assert payload["scores"]["delta"] >= 0
    assert payload["candidates"]
    assert payload["selection_strategy"] in {"abnormal_aware", "rs_dpo"}
    assert payload["citations"]
    assert payload["citations"][0]["chunk_id"] == "c1"
    assert payload["metadata"]["preference_construction"] == payload["selection_strategy"]


def test_rl_export_emits_flat_prompt_response_reward():
    record = _make_record()
    payload = export_rl_record(record)

    assert payload["prompt"][0]["role"] == "system"
    assert payload["response"][0]["role"] == "assistant"
    assert 0.0 <= payload["reward"] <= 1.0
    assert isinstance(payload["candidate_rewards"], list)
    assert all(0.0 <= r <= 1.0 for r in payload["candidate_rewards"])
    # Legacy episode preserved for backwards compatibility.
    assert payload["episode"]["steps"]


# ---------- Tool-call schema ----------------------------------------------


def test_tool_call_export_uses_real_clinical_tool_surface():
    record = _make_record()
    payload = export_tool_call_record(record)

    tool_names = {tool["function"]["name"] for tool in payload["tools"]}
    assert "lookup_lab" in tool_names
    assert "order_imaging" in tool_names
    assert "prescribe" in tool_names
    assert "record_diagnosis" in tool_names

    assistant = payload["messages"][-1]
    call_names = [c["function"]["name"] for c in assistant["tool_calls"]]
    # Lactate (high) should drive a lookup_lab call.
    assert "lookup_lab" in call_names
    # Norepinephrine (active medication) should drive a prescribe call.
    assert "prescribe" in call_names
    # Sepsis on the encounter should drive a record_diagnosis call.
    assert "record_diagnosis" in call_names


def test_tool_call_lookup_lab_arguments_are_valid_json_with_schema_fields():
    record = _make_record()
    payload = export_tool_call_record(record)
    assistant = payload["messages"][-1]
    lab_call = next(
        c for c in assistant["tool_calls"] if c["function"]["name"] == "lookup_lab"
    )
    args = json.loads(lab_call["function"]["arguments"])
    assert args["lab_name"] == "Lactate"
    assert args["patient_id"] == "pat-1"
    assert args["loinc"] == "2524-7"


# ---------- Judge curation + collision warning ----------------------------


def test_recommend_judges_default_returns_one_per_provider():
    recs = recommend_judges()
    providers = {rec.provider for rec in recs}
    assert {"anthropic", "openai", "openrouter", "ollama"} <= providers
    assert all(rec.is_default for rec in recs)


def test_recommend_judges_for_specific_provider_includes_alternates():
    recs = recommend_judges("anthropic")
    assert recs
    assert recs[0].provider == "anthropic"
    assert recs[0].is_default
    # Alternates follow.
    assert any(not r.is_default for r in recs[1:])


def test_recommend_judges_unknown_provider_raises():
    with pytest.raises(KeyError, match="Unknown provider"):
        recommend_judges("banana")


def test_warn_if_judge_collides_returns_message_when_same_provider():
    msg = warn_if_judge_collides_with_generator(
        judge_provider="anthropic", generator_provider="anthropic"
    )
    assert msg
    assert "Self-judging" in msg


def test_warn_if_judge_collides_silent_when_different_providers():
    msg = warn_if_judge_collides_with_generator(
        judge_provider="anthropic", generator_provider="openai"
    )
    assert msg is None


def test_warn_if_judge_collides_silent_when_either_is_none():
    assert warn_if_judge_collides_with_generator(
        judge_provider=None, generator_provider="anthropic"
    ) is None
    assert warn_if_judge_collides_with_generator(
        judge_provider="anthropic", generator_provider=None
    ) is None
