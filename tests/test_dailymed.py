import re

import pytest

from casecrawler.models.document import CredibilityLevel
from casecrawler.sources.dailymed import DailyMedSource

SEARCH_RESPONSE = {
    "data": [
        {
            "setid": "23cb5001-6aeb-434a-b2e8-aaf8bbab9dc6",
            "title": "METFORMIN HYDROCHLORIDE tablet",
            "published_date": "Jan 15, 2024",
            "spl_version": 5,
        }
    ],
    "metadata": {"total_elements": 1},
}

SPL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<document xmlns="urn:hl7-org:v3">
  <component>
    <structuredBody>
      <component>
        <section>
          <code code="34067-9"/>
          <title>INDICATIONS AND USAGE</title>
          <text>Metformin is indicated for type 2 diabetes.</text>
        </section>
      </component>
      <component>
        <section>
          <code code="34070-3"/>
          <title>CONTRAINDICATIONS</title>
          <text>Severe renal impairment.</text>
        </section>
      </component>
      <component>
        <section>
          <code code="34084-4"/>
          <title>ADVERSE REACTIONS</title>
          <text>Diarrhea, nausea, vomiting.</text>
        </section>
      </component>
    </structuredBody>
  </component>
</document>"""


@pytest.fixture
def dailymed():
    return DailyMedSource()


@pytest.mark.asyncio
async def test_dailymed_search(dailymed, httpx_mock):
    httpx_mock.add_response(
        url=re.compile(r"https://dailymed\.nlm\.nih\.gov/dailymed/services/v2/spls\.json"),
        json=SEARCH_RESPONSE,
    )
    httpx_mock.add_response(
        url=re.compile(r"https://dailymed\.nlm\.nih\.gov/dailymed/services/v2/spls/23cb5001"),
        text=SPL_XML,
        headers={"content-type": "application/xml"},
    )
    docs = await dailymed.search("metformin", limit=5)
    assert len(docs) == 1
    assert docs[0].source == "dailymed"
    assert docs[0].content_type == "drug_label"
    assert "type 2 diabetes" in docs[0].content
    assert "renal impairment" in docs[0].content
    assert docs[0].metadata.credibility == CredibilityLevel.FDA_LABEL


def test_dailymed_is_available():
    assert DailyMedSource.is_available() is True
