import pytest
from fastapi.testclient import TestClient

from casecrawler.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_legacy_generate_endpoint_is_not_registered(client):
    resp = client.post("/api/generate", json={"topic": "SAH"})

    assert resp.status_code == 404


def test_legacy_cases_endpoints_are_not_registered(client):
    assert client.get("/api/cases").status_code == 404
    assert client.get("/api/cases/test-1").status_code == 404
