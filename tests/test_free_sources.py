import re

import pytest

from casecrawler.models.document import CredibilityLevel
from casecrawler.sources.rxnorm import RxNormSource
from casecrawler.sources.medrxiv import MedRxivSource
from casecrawler.sources.clinicaltrials import ClinicalTrialsSource


# --- RxNorm ---

RXNORM_DRUGS_RESPONSE = {
    "drugGroup": {
        "conceptGroup": [
            {
                "tty": "SBD",
                "conceptProperties": [
                    {
                        "rxcui": "596928",
                        "name": "duloxetine 20 MG Delayed Release Oral Capsule [Cymbalta]",
                        "tty": "SBD",
                    },
                    {
                        "rxcui": "596930",
                        "name": "duloxetine 30 MG Delayed Release Oral Capsule [Cymbalta]",
                        "tty": "SBD",
                    },
                ],
            }
        ]
    }
}


@pytest.fixture
def rxnorm():
    return RxNormSource()


@pytest.mark.asyncio
async def test_rxnorm_search(rxnorm, httpx_mock):
    httpx_mock.add_response(
        url=re.compile(r"https://rxnav\.nlm\.nih\.gov/REST/drugs\.json"),
        json=RXNORM_DRUGS_RESPONSE,
    )
    docs = await rxnorm.search("cymbalta", limit=5)
    assert len(docs) == 2
    assert docs[0].source == "rxnorm"
    assert docs[0].source_id == "596928"
    assert "duloxetine" in docs[0].content.lower()
    assert docs[0].metadata.credibility == CredibilityLevel.CURATED


# --- medRxiv ---

MEDRXIV_RESPONSE = {
    "messages": [{"status": "ok", "count": 1, "total": 1}],
    "collection": [
        {
            "doi": "10.1101/2024.01.01.123456",
            "title": "Novel Biomarkers in Subarachnoid Hemorrhage",
            "authors": "Smith, J.; Doe, A.",
            "date": "2024-01-15",
            "abstract": "We identified novel biomarkers for SAH prognosis.",
            "category": "neurology",
            "server": "medrxiv",
        }
    ],
}


@pytest.fixture
def medrxiv():
    return MedRxivSource()


@pytest.mark.asyncio
async def test_medrxiv_search(medrxiv, httpx_mock):
    httpx_mock.add_response(
        url=re.compile(r"https://api\.medrxiv\.org/details/medrxiv"),
        json=MEDRXIV_RESPONSE,
    )
    docs = await medrxiv.search("subarachnoid hemorrhage", limit=5)
    assert len(docs) == 1
    assert docs[0].source == "medrxiv"
    assert docs[0].metadata.credibility == CredibilityLevel.PREPRINT
    assert "biomarkers" in docs[0].content.lower()


# --- ClinicalTrials ---

CT_RESPONSE = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00000123",
                    "briefTitle": "Study of Drug X for SAH",
                    "officialTitle": "A Randomized Trial of Drug X in SAH Patients",
                },
                "statusModule": {
                    "overallStatus": "RECRUITING",
                },
                "eligibilityModule": {
                    "eligibilityCriteria": "Inclusion: Adults 18-65\nExclusion: Pregnancy",
                    "sex": "ALL",
                    "minimumAge": "18 Years",
                    "maximumAge": "65 Years",
                },
                "armsInterventionsModule": {
                    "interventions": [
                        {"type": "DRUG", "name": "Drug X", "description": "Experimental drug"},
                    ]
                },
                "outcomesModule": {
                    "primaryOutcomes": [
                        {"measure": "Mortality at 30 days", "timeFrame": "30 days"},
                    ]
                },
                "designModule": {
                    "studyType": "INTERVENTIONAL",
                    "phases": ["PHASE3"],
                },
            }
        }
    ],
    "nextPageToken": None,
}


@pytest.fixture
def ct():
    return ClinicalTrialsSource()


@pytest.mark.asyncio
async def test_clinicaltrials_search(ct, httpx_mock):
    httpx_mock.add_response(
        url=re.compile(r"https://clinicaltrials\.gov/api/v2/studies"),
        json=CT_RESPONSE,
    )
    docs = await ct.search("subarachnoid hemorrhage", limit=5)
    assert len(docs) == 1
    assert docs[0].source == "clinicaltrials"
    assert docs[0].source_id == "NCT00000123"
    assert "Drug X" in docs[0].content
    assert "18-65" in docs[0].content or "18 Years" in docs[0].content
    assert docs[0].content_type == "trial_protocol"
    assert docs[0].metadata.credibility == CredibilityLevel.PEER_REVIEWED
