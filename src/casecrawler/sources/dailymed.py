from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx

from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

# LOINC codes for key drug label sections
SECTION_CODES = {
    "34067-9": "INDICATIONS AND USAGE",
    "34068-7": "DOSAGE AND ADMINISTRATION",
    "34070-3": "CONTRAINDICATIONS",
    "43685-7": "WARNINGS AND PRECAUTIONS",
    "34084-4": "ADVERSE REACTIONS",
    "34073-7": "DRUG INTERACTIONS",
    "34089-3": "DESCRIPTION",
    "34090-1": "CLINICAL PHARMACOLOGY",
    "34066-1": "BOXED WARNING",
    "34088-5": "OVERDOSAGE",
}

NS = {"hl7": "urn:hl7-org:v3"}


class DailyMedSource(BaseSource):
    name = "dailymed"
    requires_keys: list[str] = []

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        async with httpx.AsyncClient() as client:
            # Step 1: Search for SPLs by drug name
            resp = await client.get(
                f"{BASE_URL}/spls.json",
                params={"drug_name": query, "pagesize": str(min(limit, 100)), "page": "1"},
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", [])

            # Step 2: Fetch full SPL XML for each result
            documents = []
            for item in items:
                setid = item.get("setid", "")
                title = item.get("title", "")
                try:
                    doc = await self._fetch_spl(client, setid, title)
                    documents.append(doc)
                except Exception:
                    continue  # Skip SPLs that fail to parse

            return documents

    async def fetch(self, document_id: str) -> Document:
        async with httpx.AsyncClient() as client:
            return await self._fetch_spl(client, document_id, "")

    async def _fetch_spl(self, client: httpx.AsyncClient, setid: str, title: str) -> Document:
        resp = await client.get(f"{BASE_URL}/spls/{setid}.xml")
        resp.raise_for_status()
        content = self._parse_spl_xml(resp.text)

        metadata = DocumentMetadata(
            credibility=CredibilityLevel.FDA_LABEL,
            url=f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}",
        )

        return Document(
            source="dailymed",
            source_id=setid,
            title=title,
            content=content,
            content_type="drug_label",
            metadata=metadata,
        )

    def _parse_spl_xml(self, xml_text: str) -> str:
        root = ET.fromstring(xml_text)
        sections = []

        for section in root.iter("{urn:hl7-org:v3}section"):
            code_elem = section.find("hl7:code", NS)
            if code_elem is None:
                continue
            code = code_elem.get("code", "")
            header = SECTION_CODES.get(code)
            if header is None:
                continue

            # Extract all text content from the section
            text_elem = section.find("hl7:text", NS)
            if text_elem is not None:
                text = "".join(text_elem.itertext()).strip()
                if text:
                    sections.append(f"{header}\n{text}")

        return "\n\n".join(sections)
