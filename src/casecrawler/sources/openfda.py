from __future__ import annotations

from datetime import date

import httpx

from casecrawler.config import get_env
from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

LABEL_URL = "https://api.fda.gov/drug/label.json"


class OpenFDASource(BaseSource):
    name = "openfda"
    requires_keys: list[str] = []

    def _base_params(self) -> dict:
        params: dict = {}
        api_key = get_env("OPENFDA_API_KEY")
        if api_key:
            params["api_key"] = api_key
        return params

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        async with httpx.AsyncClient() as client:
            params = {
                **self._base_params(),
                "search": f'openfda.generic_name:"{query}"+openfda.brand_name:"{query}"',
                "limit": str(min(limit, 100)),
            }
            resp = await client.get(LABEL_URL, params=params)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()
            return [self._parse_label(r) for r in data.get("results", [])]

    async def fetch(self, document_id: str) -> Document:
        async with httpx.AsyncClient() as client:
            params = {
                **self._base_params(),
                "search": f'set_id:"{document_id}"',
                "limit": "1",
            }
            resp = await client.get(LABEL_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                raise ValueError(f"No label found for set_id {document_id}")
            return self._parse_label(results[0])

    def _parse_label(self, result: dict) -> Document:
        openfda = result.get("openfda", {})
        brand = openfda.get("brand_name", [""])[0] if openfda.get("brand_name") else ""
        generic = openfda.get("generic_name", [""])[0] if openfda.get("generic_name") else ""
        manufacturer = openfda.get("manufacturer_name", [""])[0] if openfda.get("manufacturer_name") else ""
        set_id = result.get("set_id", "")

        title = f"{brand} ({generic})" if brand and generic else brand or generic or set_id

        # Build content from label sections
        sections = []
        for field, header in [
            ("indications_and_usage", "INDICATIONS AND USAGE"),
            ("dosage_and_administration", "DOSAGE AND ADMINISTRATION"),
            ("contraindications", "CONTRAINDICATIONS"),
            ("warnings", "WARNINGS"),
            ("warnings_and_cautions", "WARNINGS AND PRECAUTIONS"),
            ("adverse_reactions", "ADVERSE REACTIONS"),
            ("drug_interactions", "DRUG INTERACTIONS"),
            ("boxed_warning", "BOXED WARNING"),
            ("overdosage", "OVERDOSAGE"),
        ]:
            values = result.get(field, [])
            if values:
                text = values[0] if isinstance(values, list) else values
                sections.append(f"{header}\n{text}")

        content = "\n\n".join(sections)

        # Parse date
        pub_date = None
        effective = result.get("effective_time", "")
        if len(effective) >= 8:
            try:
                pub_date = date(int(effective[:4]), int(effective[4:6]), int(effective[6:8]))
            except ValueError:
                pass

        metadata = DocumentMetadata(
            authors=[manufacturer] if manufacturer else [],
            publication_date=pub_date,
            specialty=[],
            credibility=CredibilityLevel.FDA_LABEL,
            url=f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}" if set_id else None,
        )

        return Document(
            source="openfda",
            source_id=set_id,
            title=title,
            content=content,
            content_type="drug_label",
            metadata=metadata,
        )
