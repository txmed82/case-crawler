import json

from fastapi.testclient import TestClient

from casecrawler.api.routes import datasets as datasets_routes
from casecrawler.api.app import app
from casecrawler.models.config import AppConfig, SyntheticConfig


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
