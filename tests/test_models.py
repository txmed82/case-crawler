from datetime import date

from casecrawler.models.document import (
    Chunk,
    CredibilityLevel,
    Document,
    DocumentMetadata,
)


def test_document_creation():
    meta = DocumentMetadata(
        authors=["Smith J", "Doe A"],
        publication_date=date(2024, 1, 15),
        specialty=["neurosurgery"],
        credibility=CredibilityLevel.PEER_REVIEWED,
        url="https://example.com/article",
        doi="10.1234/test",
    )
    doc = Document(
        source="pubmed",
        source_id="12345678",
        title="Test Article",
        content="This is the abstract text.",
        content_type="abstract",
        metadata=meta,
    )
    assert doc.source == "pubmed"
    assert doc.metadata.credibility == CredibilityLevel.PEER_REVIEWED
    assert doc.metadata.authors == ["Smith J", "Doe A"]


def test_document_metadata_defaults():
    meta = DocumentMetadata(credibility=CredibilityLevel.PREPRINT)
    assert meta.authors == []
    assert meta.publication_date is None
    assert meta.specialty == []
    assert meta.url is None
    assert meta.doi is None


def test_credibility_ordering():
    levels = list(CredibilityLevel)
    assert CredibilityLevel.GUIDELINE in levels
    assert CredibilityLevel.PEER_REVIEWED in levels
    assert CredibilityLevel.PREPRINT in levels
    assert CredibilityLevel.CURATED in levels
    assert CredibilityLevel.FDA_LABEL in levels


def test_chunk_creation():
    meta = DocumentMetadata(credibility=CredibilityLevel.PEER_REVIEWED)
    chunk = Chunk(
        chunk_id="abc123",
        source_document_id="pubmed:12345678",
        text="Some chunk text here.",
        position=0,
        metadata=meta,
    )
    assert chunk.chunk_id == "abc123"
    assert chunk.position == 0
    assert chunk.metadata.credibility == CredibilityLevel.PEER_REVIEWED
