import tempfile

import pytest

from casecrawler.models.case import (
    Complication,
    DecisionChoice,
    DifficultyLevel,
    GeneratedCase,
    GroundTruth,
    Patient,
    ReviewResult,
)
from casecrawler.storage.case_store import CaseStore


def _make_case(case_id: str = "test-1", topic: str = "SAH", difficulty: str = "resident") -> GeneratedCase:
    return GeneratedCase(
        case_id=case_id,
        topic=topic,
        difficulty=DifficultyLevel(difficulty),
        specialty=["neurosurgery", "emergency_medicine"],
        patient=Patient(age=42, sex="female", demographics="Healthy"),
        vignette="A 42-year-old woman presents with thunderclap headache.",
        decision_prompt="What would you do next?",
        ground_truth=GroundTruth(
            diagnosis="SAH", optimal_next_step="CT head",
            rationale="Most sensitive", key_findings=["thunderclap headache"],
        ),
        decision_tree=[
            DecisionChoice(
                choice="CT head", is_correct=True, error_type=None,
                reasoning="Correct", outcome="SAH confirmed",
                consequence=None, next_decision=None,
            ),
        ],
        complications=[
            Complication(trigger="delayed_diagnosis", detail="6h", event="Rebleed", outcome="Death"),
        ],
        review=ReviewResult(
            accuracy_score=0.95, pedagogy_score=0.9, bias_score=0.95, approved=True, notes=[],
        ),
        sources=[{"type": "pubmed", "reference": "PMID:123", "chunk_ids": ["abc"]}],
        metadata={"generated_at": "2026-04-08", "model": "claude-sonnet-4-6", "retry_count": 0},
    )


def test_save_and_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaseStore(db_path=f"{tmpdir}/cases.db")
        case = _make_case()
        store.save(case)
        retrieved = store.get("test-1")
        assert retrieved is not None
        assert retrieved.case_id == "test-1"
        assert retrieved.topic == "SAH"
        assert retrieved.difficulty == DifficultyLevel.RESIDENT


def test_get_nonexistent():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaseStore(db_path=f"{tmpdir}/cases.db")
        assert store.get("nonexistent") is None


def test_list_all():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaseStore(db_path=f"{tmpdir}/cases.db")
        store.save(_make_case("c1", "SAH", "resident"))
        store.save(_make_case("c2", "MI", "attending"))
        store.save(_make_case("c3", "SAH", "medical_student"))
        cases = store.list_cases()
        assert len(cases) == 3


def test_list_filter_topic():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaseStore(db_path=f"{tmpdir}/cases.db")
        store.save(_make_case("c1", "SAH", "resident"))
        store.save(_make_case("c2", "MI", "attending"))
        cases = store.list_cases(topic="SAH")
        assert len(cases) == 1
        assert cases[0].topic == "SAH"


def test_list_filter_difficulty():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaseStore(db_path=f"{tmpdir}/cases.db")
        store.save(_make_case("c1", "SAH", "resident"))
        store.save(_make_case("c2", "SAH", "attending"))
        cases = store.list_cases(difficulty="attending")
        assert len(cases) == 1
        assert cases[0].difficulty == DifficultyLevel.ATTENDING


def test_list_filter_min_accuracy():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaseStore(db_path=f"{tmpdir}/cases.db")
        store.save(_make_case("c1"))
        low_acc = _make_case("c2")
        low_acc = low_acc.model_copy(
            update={"review": low_acc.review.model_copy(update={"accuracy_score": 0.5})}
        )
        store.save(low_acc)
        cases = store.list_cases(min_accuracy=0.8)
        assert len(cases) == 1
        assert cases[0].case_id == "c1"


def test_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaseStore(db_path=f"{tmpdir}/cases.db")
        assert store.count() == 0
        store.save(_make_case())
        assert store.count() == 1


def test_export_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = CaseStore(db_path=f"{tmpdir}/cases.db")
        store.save(_make_case("c1", "SAH"))
        store.save(_make_case("c2", "MI"))
        lines = store.export_jsonl()
        assert len(lines) == 2
        import json
        parsed = json.loads(lines[0])
        assert "case_id" in parsed
