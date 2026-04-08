from __future__ import annotations

from datetime import date, timedelta

import httpx

from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

BASE_URL = "https://api.medrxiv.org"


class MedRxivSource(BaseSource):
    name = "medrxiv"
    requires_keys: list[str] = []

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        """Fetch recent preprints and filter by query terms.

        Note: The medRxiv API has no keyword search endpoint.
        We fetch recent papers and filter client-side.
        """
        async with httpx.AsyncClient() as client:
            # Fetch recent papers (last 30 days)
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
            url = f"{BASE_URL}/details/medrxiv/{start_date}/{end_date}/0/json"

            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            collection = data.get("collection", [])
            query_lower = query.lower()
            query_terms = query_lower.split()

            # Filter: any query term appears in title or abstract
            matching = []
            for item in collection:
                title = item.get("title", "").lower()
                abstract = item.get("abstract", "").lower()
                if any(term in title or term in abstract for term in query_terms):
                    matching.append(self._parse_item(item))
                    if len(matching) >= limit:
                        break

            return matching

    async def fetch(self, document_id: str) -> Document:
        """Fetch a specific preprint by DOI."""
        async with httpx.AsyncClient() as client:
            url = f"{BASE_URL}/details/medrxiv/{document_id}/na/json"
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("collection", [])
            if not items:
                raise ValueError(f"No preprint found for DOI {document_id}")
            return self._parse_item(items[0])

    def _parse_item(self, item: dict) -> Document:
        doi = item.get("doi", "")
        title = item.get("title", "")
        abstract = item.get("abstract", "")
        authors_str = item.get("authors", "")
        date_str = item.get("date", "")
        category = item.get("category", "")

        authors = [a.strip() for a in authors_str.split(";") if a.strip()]

        pub_date = None
        if date_str:
            try:
                parts = date_str.split("-")
                pub_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
            except (ValueError, IndexError):
                pass

        metadata = DocumentMetadata(
            authors=authors,
            publication_date=pub_date,
            specialty=[category] if category else [],
            credibility=CredibilityLevel.PREPRINT,
            url=f"https://www.medrxiv.org/content/{doi}",
            doi=doi,
        )

        return Document(
            source="medrxiv",
            source_id=doi,
            title=title,
            content=abstract,
            content_type="abstract",
            metadata=metadata,
        )
