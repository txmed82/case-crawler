import os
from unittest.mock import patch

from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource
from casecrawler.sources.registry import SourceRegistry


class FakeFreeSrc(BaseSource):
    name = "fake_free"
    requires_keys: list[str] = []

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        return [
            Document(
                source=self.name, source_id="1", title="Fake", content="Fake content",
                content_type="abstract",
                metadata=DocumentMetadata(credibility=CredibilityLevel.PEER_REVIEWED),
            )
        ]

    async def fetch(self, document_id: str) -> Document:
        return Document(
            source=self.name, source_id=document_id, title="Fake", content="Full content",
            content_type="abstract",
            metadata=DocumentMetadata(credibility=CredibilityLevel.PEER_REVIEWED),
        )


class FakePaidSrc(BaseSource):
    name = "fake_paid"
    requires_keys: list[str] = ["FAKE_PAID_API_KEY"]

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        return []

    async def fetch(self, document_id: str) -> Document:
        raise NotImplementedError


def test_registry_discovers_free_source():
    registry = SourceRegistry()
    registry.discover()
    assert "fake_free" in registry.available_source_names


def test_registry_excludes_paid_without_key():
    with patch.dict(os.environ, {}, clear=True):
        registry = SourceRegistry()
        registry.discover()
        assert "fake_paid" not in registry.available_source_names


def test_registry_includes_paid_with_key():
    with patch.dict(os.environ, {"FAKE_PAID_API_KEY": "test-key"}):
        registry = SourceRegistry()
        registry.discover()
        assert "fake_paid" in registry.available_source_names


def test_registry_get_source():
    registry = SourceRegistry()
    registry.discover()
    source = registry.get("fake_free")
    assert source is not None
    assert source.name == "fake_free"


def test_registry_get_missing_source():
    registry = SourceRegistry()
    registry.discover()
    assert registry.get("nonexistent") is None


def test_registry_all_sources_info():
    with patch.dict(os.environ, {}, clear=True):
        registry = SourceRegistry()
        registry.discover()
        info = registry.all_sources_info()
        paid_info = [s for s in info if s["name"] == "fake_paid"]
        assert len(paid_info) == 1
        assert paid_info[0]["available"] is False
        assert paid_info[0]["missing_keys"] == ["FAKE_PAID_API_KEY"]
