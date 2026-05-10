"""Regression tests for Phase 4a (RAG retriever wiring)."""

from __future__ import annotations

from typing import Any

import pytest

from casecrawler.generation.retriever import Retriever
from casecrawler.generation.synthetic_pipeline import SyntheticPipeline
from casecrawler.models.config import GroundingConfig
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import (
    ComplexityProfile,
    GroundingBundle,
    GroundingCitation,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    ValidationReport,
    VitalObservation,
)
from casecrawler.validation.synthetic_validator import SyntheticValidator


# ---------- Retriever.fetch_grounding ---------------------------------------


class _FakeStore:
    def __init__(self, hits: list[dict[str, Any]]):
        self._hits = hits

    def search(self, query: str, n_results: int = 10, source: str | None = None):
        return self._hits[:n_results]


def _hit(chunk_id: str, *, score: float, credibility: str, source: str = "pubmed") -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"Body text for {chunk_id} {'x' * 600}",
        "score": score,
        "metadata": {
            "source_document_id": f"{source}:doc-1",
            "source": source,
            "specialty": "internal_medicine",
            "credibility": credibility,
            "doi": "10.1000/x",
            "url": f"https://example.com/{chunk_id}",
        },
    }


def test_fetch_grounding_returns_bundle_ranked_by_credibility():
    store = _FakeStore(
        [
            _hit("a", score=0.5, credibility="preprint"),
            _hit("b", score=0.4, credibility="guideline"),
            _hit("c", score=0.3, credibility="peer_reviewed"),
        ]
    )
    bundle = Retriever(store).fetch_grounding(topic="sepsis", k=10)

    assert isinstance(bundle, GroundingBundle)
    assert bundle.topic == "sepsis"
    # Guideline first, then peer_reviewed, then preprint.
    assert [c.chunk_id for c in bundle.citations] == ["b", "c", "a"]
    assert all(isinstance(c, GroundingCitation) for c in bundle.citations)
    # Snippet capped to 280 chars.
    assert all(len(c.snippet) <= 280 for c in bundle.citations)


def test_fetch_grounding_empty_store_returns_empty_bundle():
    bundle = Retriever(_FakeStore([])).fetch_grounding(topic="anything")
    assert bundle.citations == []
    # By default, retriever does not flip fallback_used; that's the
    # pipeline's job.
    assert bundle.fallback_used is False


# ---------- Validator gates approval on citations ---------------------------


def _record(metadata: dict | None = None) -> SyntheticRecord:
    return SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.VITALS],
        patient=SyntheticPatient(patient_id="pat-1", age=40, sex="female"),
        encounters=[],
        labs=[],
        vitals=[
            VitalObservation(
                name="HR",
                value=88,
                unit="/min",
                effective_time="2026-05-06T08:00:00",
            ),
        ],
        provenance=Provenance(generator="unit-test", created_at="2026-05-06T09:00:00"),
        metadata=metadata or {},
    )


def test_validator_with_grounding_required_rejects_when_no_citations():
    record = _record()
    report = SyntheticValidator(threshold=0.0, require_grounding=True).validate(record)
    assert report.approved is False
    assert any(
        i.field == "metadata.grounding.citations" for i in report.issues
    ), report.issues


def test_validator_with_grounding_required_accepts_when_citations_present():
    record = _record(
        metadata={
            "grounding": GroundingBundle(
                topic="sepsis",
                retrieved_at="2026-05-06T09:00:00",
                citations=[
                    GroundingCitation(
                        chunk_id="c1",
                        source="pubmed",
                        source_document_id="pubmed:doc-1",
                        score=0.9,
                        credibility="guideline",
                        snippet="Sepsis is defined as...",
                    )
                ],
            ).model_dump()
        }
    )
    report = SyntheticValidator(threshold=0.0, require_grounding=True).validate(record)
    assert report.approved is True


def test_validator_with_grounding_required_accepts_fallback_bundle():
    """When the operator chose ``fallback='template'`` and retrieval came up
    empty, the bundle is flagged ``fallback_used=True`` and the validator
    must NOT reject -- the operator explicitly opted into degraded mode."""
    record = _record(
        metadata={
            "grounding": GroundingBundle(
                topic="rare_disease",
                retrieved_at="2026-05-06T09:00:00",
                citations=[],
                fallback_used=True,
                fallback_reason="no_chunks_in_index",
            ).model_dump()
        }
    )
    report = SyntheticValidator(threshold=0.0, require_grounding=True).validate(record)
    assert report.approved is True


def test_validator_default_does_not_require_grounding():
    """Existing default behaviour (and existing tests) must keep working."""
    record = _record()
    report = SyntheticValidator(threshold=0.0).validate(record)
    # No grounding-related issue.
    assert all(
        i.field != "metadata.grounding.citations" for i in report.issues
    )


# ---------- Pipeline grounding wiring --------------------------------------


def test_pipeline_grounding_disabled_returns_none():
    """With grounding disabled, _fetch_grounding_for_topic short-circuits."""
    pipeline = SyntheticPipeline()
    pipeline._config.synthetic.grounding = GroundingConfig(enabled=False)
    bundle = pipeline._fetch_grounding_for_topic(
        GenerationRequest(topic="sepsis")
    )
    assert bundle is None


def test_pipeline_grounding_uses_injected_retriever():
    """When enabled, the injected retriever's bundle is attached verbatim."""

    class _StubRetriever:
        def fetch_grounding(self, topic, modalities=None, k=8):
            return GroundingBundle(
                topic=topic,
                retrieved_at="2026-05-06T09:00:00",
                citations=[
                    GroundingCitation(
                        chunk_id="c1",
                        source="pubmed",
                        source_document_id="pubmed:doc-1",
                        score=0.95,
                        credibility="guideline",
                    )
                ],
            )

    pipeline = SyntheticPipeline(retriever=_StubRetriever())
    pipeline._config.synthetic.grounding = GroundingConfig(enabled=True, k=4)
    bundle = pipeline._fetch_grounding_for_topic(
        GenerationRequest(topic="sepsis")
    )
    assert bundle is not None
    assert bundle.fallback_used is False
    assert [c.chunk_id for c in bundle.citations] == ["c1"]


def test_pipeline_grounding_template_fallback_on_empty():
    """fallback='template' must produce a flagged empty bundle, not raise."""

    class _EmptyRetriever:
        def fetch_grounding(self, topic, modalities=None, k=8):
            return GroundingBundle(
                topic=topic, retrieved_at="2026-05-06T09:00:00", citations=[]
            )

    pipeline = SyntheticPipeline(retriever=_EmptyRetriever())
    pipeline._config.synthetic.grounding = GroundingConfig(
        enabled=True, fallback="template"
    )
    bundle = pipeline._fetch_grounding_for_topic(
        GenerationRequest(topic="rare_disease")
    )
    assert bundle is not None
    assert bundle.fallback_used is True
    assert bundle.fallback_reason == "no_chunks_in_index"


def test_pipeline_grounding_fallback_fail_raises_on_empty():
    class _EmptyRetriever:
        def fetch_grounding(self, topic, modalities=None, k=8):
            return GroundingBundle(
                topic=topic, retrieved_at="2026-05-06T09:00:00", citations=[]
            )

    pipeline = SyntheticPipeline(retriever=_EmptyRetriever())
    pipeline._config.synthetic.grounding = GroundingConfig(
        enabled=True, fallback="fail"
    )
    with pytest.raises(RuntimeError, match="returned no citations"):
        pipeline._fetch_grounding_for_topic(
            GenerationRequest(topic="rare_disease")
        )


def test_pipeline_grounding_fallback_fail_raises_on_retriever_error():
    class _BrokenRetriever:
        def fetch_grounding(self, topic, modalities=None, k=8):
            raise RuntimeError("chroma down")

    pipeline = SyntheticPipeline(retriever=_BrokenRetriever())
    pipeline._config.synthetic.grounding = GroundingConfig(
        enabled=True, fallback="fail"
    )
    with pytest.raises(RuntimeError, match="chroma down"):
        pipeline._fetch_grounding_for_topic(
            GenerationRequest(topic="sepsis")
        )


def test_pipeline_grounding_fallback_template_swallows_retriever_error():
    class _BrokenRetriever:
        def fetch_grounding(self, topic, modalities=None, k=8):
            raise RuntimeError("chroma down")

    pipeline = SyntheticPipeline(retriever=_BrokenRetriever())
    pipeline._config.synthetic.grounding = GroundingConfig(
        enabled=True, fallback="template"
    )
    bundle = pipeline._fetch_grounding_for_topic(
        GenerationRequest(topic="sepsis")
    )
    assert bundle is not None
    assert bundle.fallback_used is True
    assert bundle.fallback_reason == "retrieval_error"


# ---------- ValidationReport must serialize unchanged ----------------------


def test_validation_report_round_trip_with_grounding_issue():
    record = _record()
    report = SyntheticValidator(threshold=0.0, require_grounding=True).validate(record)
    payload = report.model_dump_json()
    restored = ValidationReport.model_validate_json(payload)
    assert restored.approved == report.approved
    assert any(
        i.field == "metadata.grounding.citations" for i in restored.issues
    )
