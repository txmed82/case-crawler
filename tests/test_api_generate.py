from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from casecrawler.api.app import create_app
from casecrawler.models.case import (
    Complication, DecisionChoice, DifficultyLevel,
    GeneratedCase, GroundTruth, Patient, ReviewResult,
)


def _fake_case(case_id="test-1"):
    return GeneratedCase(
        case_id=case_id, topic="SAH", difficulty=DifficultyLevel.RESIDENT,
        specialty=["neurosurgery"],
        patient=Patient(age=42, sex="female", demographics="Healthy"),
        vignette="A 42-year-old woman...", decision_prompt="What next?",
        ground_truth=GroundTruth(diagnosis="SAH", optimal_next_step="CT",
                                  rationale="Sensitive", key_findings=["headache"]),
        decision_tree=[DecisionChoice(choice="CT", is_correct=True, error_type=None,
                                       reasoning="Right", outcome="Confirmed",
                                       consequence=None, next_decision=None)],
        complications=[Complication(trigger="delay", detail="6h", event="Rebleed", outcome="Death")],
        review=ReviewResult(accuracy_score=0.95, pedagogy_score=0.9, bias_score=0.92, approved=True, notes=[]),
        sources=[], metadata={"generated_at": "2026-04-08", "model": "test", "retry_count": 0},
    )


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_generate_endpoint(client):
    with patch("casecrawler.api.routes.generate.run_generation"):
        resp = client.post("/api/generate", json={"topic": "SAH"})
        assert resp.status_code == 202
        assert "job_id" in resp.json()


def test_cases_list(client):
    with patch("casecrawler.api.routes.cases.get_case_store") as mock:
        mock.return_value.list_cases.return_value = [_fake_case()]
        resp = client.get("/api/cases")
        assert resp.status_code == 200
        assert len(resp.json()["cases"]) == 1


def test_cases_detail(client):
    with patch("casecrawler.api.routes.cases.get_case_store") as mock:
        mock.return_value.get.return_value = _fake_case()
        resp = client.get("/api/cases/test-1")
        assert resp.status_code == 200
        assert resp.json()["case_id"] == "test-1"


def test_cases_not_found(client):
    with patch("casecrawler.api.routes.cases.get_case_store") as mock:
        mock.return_value.get.return_value = None
        resp = client.get("/api/cases/nonexistent")
        assert resp.status_code == 404
