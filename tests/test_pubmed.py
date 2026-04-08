import re

import pytest

from casecrawler.models.document import CredibilityLevel
from casecrawler.sources.pubmed import PubMedSource

ESEARCH_URL_RE = re.compile(r"https://eutils\.ncbi\.nlm\.nih\.gov/entrez/eutils/esearch\.fcgi.*")
EFETCH_URL_RE = re.compile(r"https://eutils\.ncbi\.nlm\.nih\.gov/entrez/eutils/efetch\.fcgi.*")

ESEARCH_RESPONSE = {
    "esearchresult": {
        "count": "2",
        "retmax": "2",
        "retstart": "0",
        "idlist": ["38901234", "38905678"],
    }
}

EFETCH_XML = """<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>38901234</PMID>
      <Article>
        <ArticleTitle>Management of Subarachnoid Hemorrhage: A Clinical Guideline</ArticleTitle>
        <Abstract>
          <AbstractText>This guideline addresses the management of aneurysmal subarachnoid hemorrhage including diagnosis and treatment.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>John</ForeName>
          </Author>
          <Author>
            <LastName>Doe</LastName>
            <ForeName>Jane</ForeName>
          </Author>
        </AuthorList>
        <Journal>
          <JournalIssue>
            <PubDate><Year>2024</Year><Month>Mar</Month></PubDate>
          </JournalIssue>
        </Journal>
        <ELocationID EIdType="doi">10.1234/sah.2024</ELocationID>
      </Article>
      <PublicationTypeList>
        <PublicationType>Practice Guideline</PublicationType>
      </PublicationTypeList>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>38905678</PMID>
      <Article>
        <ArticleTitle>Outcomes After SAH: A Retrospective Study</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">SAH is a severe condition.</AbstractText>
          <AbstractText Label="RESULTS">Outcomes varied significantly.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Lee</LastName>
            <ForeName>Alice</ForeName>
          </Author>
        </AuthorList>
        <Journal>
          <JournalIssue>
            <PubDate><Year>2024</Year><Month>Feb</Month></PubDate>
          </JournalIssue>
        </Journal>
        <ELocationID EIdType="doi">10.5678/sah.2024</ELocationID>
      </Article>
      <PublicationTypeList>
        <PublicationType>Journal Article</PublicationType>
      </PublicationTypeList>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


@pytest.fixture
def pubmed():
    return PubMedSource()


@pytest.mark.asyncio
async def test_pubmed_search(pubmed, httpx_mock):
    httpx_mock.add_response(
        url=ESEARCH_URL_RE,
        json=ESEARCH_RESPONSE,
    )
    httpx_mock.add_response(
        url=EFETCH_URL_RE,
        text=EFETCH_XML,
    )
    docs = await pubmed.search("subarachnoid hemorrhage", limit=2)
    assert len(docs) == 2
    assert docs[0].source == "pubmed"
    assert docs[0].source_id == "38901234"
    assert docs[0].title == "Management of Subarachnoid Hemorrhage: A Clinical Guideline"
    assert "management" in docs[0].content.lower()
    assert docs[0].metadata.authors == ["Smith John", "Doe Jane"]
    assert docs[0].metadata.doi == "10.1234/sah.2024"
    assert docs[0].metadata.credibility == CredibilityLevel.GUIDELINE


@pytest.mark.asyncio
async def test_pubmed_structured_abstract(pubmed, httpx_mock):
    httpx_mock.add_response(
        url=ESEARCH_URL_RE,
        json=ESEARCH_RESPONSE,
    )
    httpx_mock.add_response(
        url=EFETCH_URL_RE,
        text=EFETCH_XML,
    )
    docs = await pubmed.search("SAH", limit=2)
    assert "BACKGROUND" in docs[1].content or "SAH is a severe" in docs[1].content


@pytest.mark.asyncio
async def test_pubmed_credibility_detection(pubmed, httpx_mock):
    httpx_mock.add_response(
        url=ESEARCH_URL_RE,
        json=ESEARCH_RESPONSE,
    )
    httpx_mock.add_response(
        url=EFETCH_URL_RE,
        text=EFETCH_XML,
    )
    docs = await pubmed.search("SAH", limit=2)
    assert docs[0].metadata.credibility == CredibilityLevel.GUIDELINE
    assert docs[1].metadata.credibility == CredibilityLevel.PEER_REVIEWED


def test_pubmed_is_available():
    assert PubMedSource.is_available() is True


def test_pubmed_name():
    assert PubMedSource.name == "pubmed"
