from fastapi.testclient import TestClient

from casecrawler.api.app import app
from casecrawler.config import load_config


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
    config_file = tmp_path / "config.yaml"
    config_file.write_text("synthetic:\n  max_api_generation_count: 1\n")
    load_config(str(config_file))
    client = TestClient(app)

    response = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 2})

    assert response.status_code == 422
    assert "less than or equal to 1" in response.json()["detail"]
