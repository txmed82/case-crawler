from __future__ import annotations

import httpx

from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

BASE_URL = "https://rxnav.nlm.nih.gov/REST"


class RxNormSource(BaseSource):
    name = "rxnorm"
    requires_keys: list[str] = []

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/drugs.json",
                params={"name": query},
            )
            resp.raise_for_status()
            data = resp.json()

            documents = []
            drug_group = data.get("drugGroup", {})
            for group in drug_group.get("conceptGroup", []):
                for prop in group.get("conceptProperties", []):
                    if len(documents) >= limit:
                        break
                    documents.append(self._parse_concept(prop))
            return documents

    async def fetch(self, document_id: str) -> Document:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{BASE_URL}/rxcui/{document_id}/properties.json",
            )
            resp.raise_for_status()
            data = resp.json()
            props = data.get("properties", {})
            return Document(
                source="rxnorm",
                source_id=document_id,
                title=props.get("name", ""),
                content=f"Drug: {props.get('name', '')}\nRxCUI: {document_id}\nTerm Type: {props.get('tty', '')}",
                content_type="drug_info",
                metadata=DocumentMetadata(
                    credibility=CredibilityLevel.CURATED,
                    url=f"https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm={document_id}",
                ),
            )

    def _parse_concept(self, prop: dict) -> Document:
        rxcui = prop.get("rxcui", "")
        name = prop.get("name", "")
        tty = prop.get("tty", "")

        content = f"Drug: {name}\nRxCUI: {rxcui}\nTerm Type: {tty}"

        return Document(
            source="rxnorm",
            source_id=rxcui,
            title=name,
            content=content,
            content_type="drug_info",
            metadata=DocumentMetadata(
                credibility=CredibilityLevel.CURATED,
                url=f"https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm={rxcui}",
            ),
        )
