from __future__ import annotations

import httpx

from casecrawler.config import get_env
from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

# Note: Anna's Archive API endpoint and response format should be confirmed
# against their actual API documentation once you have access.
BASE_URL = "https://annas-archive.gl/api/v1"


class AnnasArchiveSource(BaseSource):
    name = "annas_archive"
    requires_keys: list[str] = ["ANNAS_ARCHIVE_API_KEY"]

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        api_key = get_env("ANNAS_ARCHIVE_API_KEY")
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/search",
                params={"q": query, "limit": str(limit), "content": "scidb"},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            return [self._parse_result(r) for r in data.get("results", [])]

    async def fetch(self, document_id: str) -> Document:
        api_key = get_env("ANNAS_ARCHIVE_API_KEY")
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/content/{document_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            return self._parse_result(resp.json())

    def _parse_result(self, result: dict) -> Document:
        authors_raw = result.get("author", "")
        authors = [a.strip() for a in authors_raw.split(";") if a.strip()] if authors_raw else []

        return Document(
            source="annas_archive",
            source_id=result.get("id", ""),
            title=result.get("title", ""),
            content=result.get("content", ""),
            content_type="full_text",
            metadata=DocumentMetadata(
                authors=authors,
                credibility=CredibilityLevel.PEER_REVIEWED,
                doi=result.get("doi"),
            ),
        )
