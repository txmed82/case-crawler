from __future__ import annotations

import httpx

from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


class ClinicalTrialsSource(BaseSource):
    name = "clinicaltrials"
    requires_keys: list[str] = []

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        async with httpx.AsyncClient() as client:
            params = {
                "query.cond": query,
                "pageSize": str(min(limit, 100)),
                "countTotal": "true",
                "format": "json",
            }
            resp = await client.get(BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            return [self._parse_study(s) for s in data.get("studies", [])]

    async def fetch(self, document_id: str) -> Document:
        """Fetch a single study by NCT ID."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BASE_URL}/{document_id}", params={"format": "json"})
            resp.raise_for_status()
            data = resp.json()
            return self._parse_study(data)

    def _parse_study(self, study: dict) -> Document:
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status = proto.get("statusModule", {})
        eligibility = proto.get("eligibilityModule", {})
        arms = proto.get("armsInterventionsModule", {})
        outcomes = proto.get("outcomesModule", {})
        design = proto.get("designModule", {})

        nct_id = ident.get("nctId", "")
        title = ident.get("briefTitle", "") or ident.get("officialTitle", "")

        # Build structured content
        sections = []

        sections.append(f"Title: {title}")
        sections.append(f"NCT ID: {nct_id}")
        sections.append(f"Status: {status.get('overallStatus', 'Unknown')}")

        phases = design.get("phases", [])
        if phases:
            sections.append(f"Phase: {', '.join(phases)}")

        criteria = eligibility.get("eligibilityCriteria", "")
        if criteria:
            age_range = ""
            min_age = eligibility.get("minimumAge", "")
            max_age = eligibility.get("maximumAge", "")
            if min_age or max_age:
                age_range = f" (Age: {min_age} - {max_age})"
            sections.append(f"Eligibility{age_range}:\n{criteria}")

        interventions = arms.get("interventions", [])
        if interventions:
            interv_text = "\n".join(
                f"- {i.get('type', '')}: {i.get('name', '')} — {i.get('description', '')}"
                for i in interventions
            )
            sections.append(f"Interventions:\n{interv_text}")

        primary = outcomes.get("primaryOutcomes", [])
        if primary:
            out_text = "\n".join(
                f"- {o.get('measure', '')} ({o.get('timeFrame', '')})" for o in primary
            )
            sections.append(f"Primary Outcomes:\n{out_text}")

        secondary = outcomes.get("secondaryOutcomes", [])
        if secondary:
            out_text = "\n".join(
                f"- {o.get('measure', '')} ({o.get('timeFrame', '')})" for o in secondary
            )
            sections.append(f"Secondary Outcomes:\n{out_text}")

        content = "\n\n".join(sections)

        metadata = DocumentMetadata(
            credibility=CredibilityLevel.PEER_REVIEWED,
            url=f"https://clinicaltrials.gov/study/{nct_id}",
        )

        return Document(
            source="clinicaltrials",
            source_id=nct_id,
            title=title,
            content=content,
            content_type="trial_protocol",
            metadata=metadata,
        )
