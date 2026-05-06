from fastapi.testclient import TestClient

from casecrawler.api.app import app


def test_generate_dataset_api_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    response = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["generated"] == 1
    assert body["approved"] == 1

