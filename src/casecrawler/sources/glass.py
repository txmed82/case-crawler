from __future__ import annotations

import httpx

from casecrawler.config import get_env
from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

# Note: Glass Health API URL and response format are based on their developer docs.
# Adjust once you have confirmed API access.
BASE_URL = "https://glass.health/api/v1"


class GlassHealthSource(BaseSource):
    name = "glass"
    requires_keys: list[str] = ["GLASS_API_KEY"]

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        api_key = get_env("GLASS_API_KEY")
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/search",
                params={"q": query, "limit": str(limit)},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            return [self._parse_result(r) for r in data.get("results", [])]

    async def fetch(self, document_id: str) -> Document:
        api_key = get_env("GLASS_API_KEY")
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/content/{document_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            return self._parse_result(resp.json())

    def _parse_result(self, result: dict) -> Document:
        return Document(
            source="glass",
            source_id=result.get("id", ""),
            title=result.get("title", ""),
            content=result.get("content", ""),
            content_type="curated",
            metadata=DocumentMetadata(
                specialty=[result["category"]] if result.get("category") else [],
                credibility=CredibilityLevel.CURATED,
                url=result.get("url"),
            ),
        )
