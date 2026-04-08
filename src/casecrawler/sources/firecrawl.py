from __future__ import annotations

import httpx

from casecrawler.config import get_env
from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

BASE_URL = "https://api.firecrawl.dev/v1"


class FirecrawlSource(BaseSource):
    name = "firecrawl"
    requires_keys: list[str] = ["FIRECRAWL_API_KEY"]

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        api_key = get_env("FIRECRAWL_API_KEY")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BASE_URL}/search",
                json={"query": query, "limit": limit},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            return [self._parse_result(r) for r in data.get("data", [])]

    async def fetch(self, document_id: str) -> Document:
        """Scrape a specific URL."""
        api_key = get_env("FIRECRAWL_API_KEY")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BASE_URL}/scrape",
                json={"url": document_id, "formats": ["markdown"]},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            return Document(
                source="firecrawl",
                source_id=document_id,
                title=data.get("title", document_id),
                content=data.get("markdown", ""),
                content_type="full_text",
                metadata=DocumentMetadata(
                    credibility=CredibilityLevel.PEER_REVIEWED,
                    url=document_id,
                ),
            )

    def _parse_result(self, result: dict) -> Document:
        url = result.get("url", "")
        return Document(
            source="firecrawl",
            source_id=url,
            title=result.get("title", url),
            content=result.get("markdown", ""),
            content_type="full_text",
            metadata=DocumentMetadata(
                credibility=CredibilityLevel.PEER_REVIEWED,
                url=url,
            ),
        )
