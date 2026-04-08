import re

import pytest

from casecrawler.models.document import CredibilityLevel
from casecrawler.sources.openfda import OpenFDASource

LABEL_RESPONSE = {
    "meta": {"results": {"total": 1}},
    "results": [
        {
            "set_id": "abc-123",
            "openfda": {
                "brand_name": ["ADVIL"],
                "generic_name": ["IBUPROFEN"],
                "manufacturer_name": ["Pfizer"],
            },
            "indications_and_usage": ["For temporary relief of minor aches and pains."],
            "contraindications": ["Do not use if you have had an allergic reaction to ibuprofen."],
            "adverse_reactions": ["Nausea, dizziness, headache."],
            "dosage_and_administration": ["Adults: 200-400mg every 4-6 hours."],
            "drug_interactions": ["Avoid concurrent use with aspirin."],
            "warnings": ["Stomach bleeding warning."],
            "effective_time": "20240115",
        }
    ],
}


@pytest.fixture
def openfda():
    return OpenFDASource()


@pytest.mark.asyncio
async def test_openfda_search(openfda, httpx_mock):
    httpx_mock.add_response(
        url=re.compile(r"https://api\.fda\.gov/drug/label\.json"),
        json=LABEL_RESPONSE,
    )
    docs = await openfda.search("ibuprofen", limit=5)
    assert len(docs) == 1
    assert docs[0].source == "openfda"
    assert docs[0].content_type == "drug_label"
    assert "INDICATIONS" in docs[0].content
    assert "aches and pains" in docs[0].content
    assert docs[0].metadata.credibility == CredibilityLevel.FDA_LABEL


def test_openfda_is_available():
    assert OpenFDASource.is_available() is True
