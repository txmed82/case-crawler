from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from casecrawler.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_sources_endpoint(client):
    resp = client.get("/api/sources")
    assert resp.status_code == 200
    data = resp.json()
    assert "available" in data
    assert "unavailable" in data
    # pubmed should always be available
    names = [s["name"] for s in data["available"]]
    assert "pubmed" in names


def test_ingest_endpoint(client):
    with patch("casecrawler.api.routes.ingest.run_ingestion") as mock_run:
        mock_run.return_value = None
        resp = client.post("/api/ingest", json={"query": "test topic"})
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "running"


def test_ingest_status_not_found(client):
    resp = client.get("/api/ingest/nonexistent-id")
    assert resp.status_code == 404


def test_search_endpoint(client):
    with patch("casecrawler.api.routes.search.get_store") as mock_store:
        mock_instance = MagicMock()
        mock_instance.search.return_value = [
            {
                "chunk_id": "abc",
                "text": "Test result text.",
                "metadata": {"source": "pubmed", "credibility": "peer_reviewed"},
                "score": 0.95,
            }
        ]
        mock_store.return_value = mock_instance
        resp = client.get("/api/search", params={"q": "test query"})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["score"] == 0.95
