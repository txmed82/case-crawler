from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date

import httpx

from casecrawler.config import get_env
from casecrawler.models.document import (
    CredibilityLevel,
    Document,
    DocumentMetadata,
)
from casecrawler.sources.base import BaseSource

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

GUIDELINE_TYPES = {"Practice Guideline", "Guideline"}


class PubMedSource(BaseSource):
    name = "pubmed"
    requires_keys: list[str] = []

    def _base_params(self) -> dict:
        params: dict = {"tool": "casecrawler", "email": "casecrawler@example.com"}
        api_key = get_env("NCBI_API_KEY")
        if api_key:
            params["api_key"] = api_key
        return params

    async def search(self, query: str, limit: int = 20) -> list[Document]:
        async with httpx.AsyncClient() as client:
            params = {
                **self._base_params(),
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": str(limit),
                "sort": "relevance",
            }
            resp = await client.get(ESEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            pmids = data["esearchresult"]["idlist"]

            if not pmids:
                return []

            params = {
                **self._base_params(),
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            }
            resp = await client.get(EFETCH_URL, params=params)
            resp.raise_for_status()
            return self._parse_articles(resp.text)

    async def fetch(self, document_id: str) -> Document:
        async with httpx.AsyncClient() as client:
            params = {
                **self._base_params(),
                "db": "pubmed",
                "id": document_id,
                "retmode": "xml",
            }
            resp = await client.get(EFETCH_URL, params=params)
            resp.raise_for_status()
            docs = self._parse_articles(resp.text)
            if not docs:
                raise ValueError(f"No article found for PMID {document_id}")
            return docs[0]

    def _parse_articles(self, xml_text: str) -> list[Document]:
        root = ET.fromstring(xml_text)
        documents = []

        for article_elem in root.findall(".//PubmedArticle"):
            citation = article_elem.find("MedlineCitation")
            if citation is None:
                continue

            pmid_elem = citation.find("PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""

            art = citation.find("Article")
            if art is None:
                continue

            title_elem = art.find("ArticleTitle")
            title = title_elem.text or "" if title_elem is not None else ""

            abstract_parts = []
            abstract_elem = art.find("Abstract")
            if abstract_elem is not None:
                for at in abstract_elem.findall("AbstractText"):
                    label = at.get("Label")
                    text = at.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
            content = "\n\n".join(abstract_parts)

            authors = []
            author_list = art.find("AuthorList")
            if author_list is not None:
                for author in author_list.findall("Author"):
                    last = author.findtext("LastName", "")
                    fore = author.findtext("ForeName", "")
                    collective = author.findtext("CollectiveName", "")
                    if collective:
                        authors.append(collective)
                    elif last:
                        authors.append(f"{last} {fore}".strip())

            doi = None
            eloc = art.find("ELocationID[@EIdType='doi']")
            if eloc is not None:
                doi = eloc.text

            pub_date = self._parse_pub_date(art)

            pub_types = set()
            pt_list = citation.find("PublicationTypeList")
            if pt_list is not None:
                for pt in pt_list.findall("PublicationType"):
                    if pt.text:
                        pub_types.add(pt.text)

            if pub_types & GUIDELINE_TYPES:
                credibility = CredibilityLevel.GUIDELINE
            else:
                credibility = CredibilityLevel.PEER_REVIEWED

            metadata = DocumentMetadata(
                authors=authors,
                publication_date=pub_date,
                specialty=[],
                credibility=credibility,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                doi=doi,
            )

            documents.append(
                Document(
                    source="pubmed",
                    source_id=pmid,
                    title=title,
                    content=content,
                    content_type="abstract",
                    metadata=metadata,
                )
            )

        return documents

    def _parse_pub_date(self, article_elem: ET.Element) -> date | None:
        journal_issue = article_elem.find(".//JournalIssue/PubDate")
        if journal_issue is None:
            return None
        year_text = journal_issue.findtext("Year")
        if not year_text:
            return None
        year = int(year_text)
        month_text = journal_issue.findtext("Month", "").lower()
        month = MONTH_MAP.get(month_text, 1)
        day_text = journal_issue.findtext("Day", "1")
        try:
            return date(year, month, int(day_text))
        except ValueError:
            return date(year, month, 1)
