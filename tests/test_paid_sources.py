import os
import re
from unittest.mock import patch

import pytest

from casecrawler.models.document import CredibilityLevel
from casecrawler.sources.glass import GlassHealthSource
from casecrawler.sources.annas_archive import AnnasArchiveSource
from casecrawler.sources.firecrawl import FirecrawlSource


# --- Glass Health ---


def test_glass_requires_key():
    assert GlassHealthSource.requires_keys == ["GLASS_API_KEY"]


def test_glass_unavailable_without_key():
    with patch.dict(os.environ, {}, clear=True):
        assert GlassHealthSource.is_available() is False


def test_glass_available_with_key():
    with patch.dict(os.environ, {"GLASS_API_KEY": "test-key"}):
        assert GlassHealthSource.is_available() is True


@pytest.mark.asyncio
async def test_glass_search(httpx_mock):
    with patch.dict(os.environ, {"GLASS_API_KEY": "test-key"}):
        httpx_mock.add_response(
            url=re.compile(r"https://glass\.health"),
            json={
                "results": [
                    {
                        "id": "glass-123",
                        "title": "Subarachnoid Hemorrhage",
                        "content": "SAH is a type of stroke caused by bleeding into the subarachnoid space.",
                        "category": "neurosurgery",
                    }
                ]
            },
        )
        source = GlassHealthSource()
        docs = await source.search("subarachnoid hemorrhage", limit=5)
        assert len(docs) == 1
        assert docs[0].source == "glass"
        assert docs[0].metadata.credibility == CredibilityLevel.CURATED


# --- Anna's Archive ---


def test_annas_requires_key():
    assert AnnasArchiveSource.requires_keys == ["ANNAS_ARCHIVE_API_KEY"]


def test_annas_unavailable_without_key():
    with patch.dict(os.environ, {}, clear=True):
        assert AnnasArchiveSource.is_available() is False


@pytest.mark.asyncio
async def test_annas_search(httpx_mock):
    with patch.dict(os.environ, {"ANNAS_ARCHIVE_API_KEY": "test-key"}):
        httpx_mock.add_response(
            url=re.compile(r"https://annas-archive\.gl"),
            json={
                "results": [
                    {
                        "id": "md5:abc123",
                        "title": "Principles of Neurosurgery",
                        "author": "Smith J",
                        "content": "Chapter 12: Management of SAH...",
                    }
                ]
            },
        )
        source = AnnasArchiveSource()
        docs = await source.search("subarachnoid hemorrhage", limit=5)
        assert len(docs) == 1
        assert docs[0].source == "annas_archive"


# --- Firecrawl ---


def test_firecrawl_requires_key():
    assert FirecrawlSource.requires_keys == ["FIRECRAWL_API_KEY"]


def test_firecrawl_unavailable_without_key():
    with patch.dict(os.environ, {}, clear=True):
        assert FirecrawlSource.is_available() is False


@pytest.mark.asyncio
async def test_firecrawl_search(httpx_mock):
    with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"}):
        httpx_mock.add_response(
            url=re.compile(r"https://api\.firecrawl\.dev/v1/search"),
            json={
                "data": [
                    {
                        "url": "https://example.com/guideline",
                        "title": "SAH Management Guideline",
                        "markdown": "# SAH Guidelines\n\nManagement of SAH includes...",
                    }
                ]
            },
        )
        source = FirecrawlSource()
        docs = await source.search("subarachnoid hemorrhage guidelines", limit=5)
        assert len(docs) == 1
        assert docs[0].source == "firecrawl"
        assert docs[0].content_type == "full_text"
