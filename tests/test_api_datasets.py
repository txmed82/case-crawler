import json

from fastapi.testclient import TestClient

from casecrawler.api.routes import datasets as datasets_routes
from casecrawler.api.app import app
from casecrawler.models.config import AppConfig, SyntheticConfig
from casecrawler.models.synthetic import (
    ComplexityProfile,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
)
from casecrawler.storage.dataset_store import DatasetStore


def test_generate_dataset_api_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    response = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["generated"] == 1
    assert body["approved"] == 1
    assert body["total_records"] == 1


def test_generate_dataset_api_rejects_unbounded_counts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = AppConfig(synthetic=SyntheticConfig(max_api_generation_count=1))
    monkeypatch.setattr(datasets_routes, "get_config", lambda: config)
    client = TestClient(app)

    response = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 2})

    assert response.status_code == 422
    assert "less than or equal to 1" in response.json()["detail"]


def test_dataset_api_lists_and_exports_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    dataset_id = generated.json()["dataset_id"]

    listed = client.get("/api/datasets")
    exported = client.get(
        f"/api/datasets/{dataset_id}/export",
        params={"export_format": "sft_jsonl"},
    )

    assert listed.status_code == 200
    assert listed.json()["datasets"][0]["dataset_id"] == dataset_id
    assert exported.status_code == 200
    first_line = exported.text.strip().splitlines()[0]
    assert json.loads(first_line)["dataset_id"] == dataset_id


def test_dataset_api_lists_and_saves_human_reviews(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = DatasetStore()
    store.save_record(
        SyntheticRecord(
            record_id="rec-review",
            dataset_id="ds-review",
            topic="sepsis",
            complexity=ComplexityProfile.MODERATE,
            modalities=[Modality.CLINICAL_TEXT],
            patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
            encounters=[],
            provenance=Provenance(
                generator="unit-test",
                created_at="2026-05-06T10:00:00",
            ),
        )
    )
    client = TestClient(app)

    queue = client.get("/api/datasets/ds-review/reviews")
    reviewed = client.post(
        "/api/records/rec-review/review",
        json={
            "status": "approved",
            "reviewer": "clinical-reviewer",
            "notes": ["Approved for fine-tuning export."],
        },
    )
    queue_after = client.get("/api/datasets/ds-review/reviews")

    assert queue.status_code == 200
    assert queue.json()["records"][0]["record_id"] == "rec-review"
    assert reviewed.status_code == 200
    assert reviewed.json()["effective_approved"] is True
    assert reviewed.json()["human_review"]["reviewer"] == "clinical-reviewer"
    assert queue_after.json()["records"] == []


def test_dataset_api_serves_dataset_and_model_cards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    dataset_id = generated.json()["dataset_id"]

    dataset_card = client.get(f"/api/datasets/{dataset_id}/card")
    model_card = client.get(
        f"/api/datasets/{dataset_id}/card",
        params={"kind": "model"},
    )

    assert dataset_card.status_code == 200
    assert "# Dataset Card:" in dataset_card.text
    assert model_card.status_code == 200
    assert "# Model Card:" in model_card.text


def test_dataset_api_benchmarks_against_reference_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    reference = client.post(
        "/api/datasets/generate",
        json={"topic": "heart failure", "count": 1},
    )
    dataset_id = generated.json()["dataset_id"]
    reference_dataset_id = reference.json()["dataset_id"]

    response = client.get(
        f"/api/datasets/{dataset_id}/benchmark",
        params={"reference_dataset_id": reference_dataset_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["generated_dataset_id"] == dataset_id
    assert body["reference_dataset_id"] == reference_dataset_id
    assert body["overall_score"] >= 0
    assert any(metric["name"] == "modality_overlap" for metric in body["metrics"])


def test_dataset_api_benchmark_reports_missing_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    dataset_id = generated.json()["dataset_id"]

    response = client.get(
        f"/api/datasets/{dataset_id}/benchmark",
        params={"reference_dataset_id": "ds-missing"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "reference dataset not found"


def test_dataset_api_rejects_invalid_limits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    list_response = client.get("/api/datasets", params={"limit": -1})
    detail_response = client.get("/api/datasets/ds-missing", params={"limit": 5000})

    assert list_response.status_code == 422
    assert detail_response.status_code == 422
