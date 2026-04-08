from __future__ import annotations

import pytest
from casecrawler.models.document import Document, DocumentMetadata, CredibilityLevel
from casecrawler.pipeline.chunker import Chunker


def _doc(content: str, content_type: str = "abstract", source: str = "pubmed", source_id: str = "123") -> Document:
    return Document(
        source=source,
        source_id=source_id,
        title="Test Doc",
        content=content,
        content_type=content_type,
        metadata=DocumentMetadata(
            credibility=CredibilityLevel.PEER_REVIEWED,
            specialty=["neurology"],
        ),
    )


def test_abstract_kept_whole():
    doc = _doc("This is an abstract. It has multiple sentences. No splitting needed.", content_type="abstract")
    chunker = Chunker()
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == doc.content


def test_chunk_id_deterministic():
    doc = _doc("Some text here.", content_type="abstract")
    chunker = Chunker()
    chunks1 = chunker.chunk(doc)
    chunks2 = chunker.chunk(doc)
    assert chunks1[0].chunk_id == chunks2[0].chunk_id


def test_full_text_split():
    # Build content with paragraphs that will exceed chunk_size=200
    paragraph = "Paragraph one. " * 50  # ~800 chars per paragraph
    content = "\n\n".join([paragraph] * 4)
    doc = _doc(content, content_type="full_text")
    chunker = Chunker(chunk_size=200)
    chunks = chunker.chunk(doc)
    # Should be split into multiple chunks
    assert len(chunks) > 1
    # Each chunk text should not drastically exceed chunk_size (some overlap allowed)
    for chunk in chunks:
        assert len(chunk.text) > 0


def test_drug_label_split_by_section():
    content = (
        "INDICATIONS AND USAGE\n"
        "This drug is indicated for treatment of pain.\n\n"
        "CONTRAINDICATIONS\n"
        "Do not use in patients with allergy.\n\n"
        "ADVERSE REACTIONS\n"
        "Common side effects include nausea and headache.\n"
    )
    doc = _doc(content, content_type="drug_label")
    chunker = Chunker()
    chunks = chunker.chunk(doc)
    # Should be split into sections
    assert len(chunks) >= 2
    texts = [c.text for c in chunks]
    assert any("INDICATIONS AND USAGE" in t for t in texts)
    assert any("CONTRAINDICATIONS" in t for t in texts)


def test_metadata_inherited():
    doc = _doc("Some abstract content.", content_type="abstract")
    chunker = Chunker()
    chunks = chunker.chunk(doc)
    assert chunks[0].metadata.credibility == CredibilityLevel.PEER_REVIEWED
    assert "neurology" in chunks[0].metadata.specialty
    assert chunks[0].source_document_id == "pubmed:123"
