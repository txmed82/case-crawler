from __future__ import annotations

import tempfile

import pytest
from casecrawler.models.document import Chunk, DocumentMetadata, CredibilityLevel
from casecrawler.pipeline.tagger import Tagger
from casecrawler.pipeline.embedder import Embedder
from casecrawler.pipeline.store import Store


def _chunk(text: str, chunk_id: str = "abc123", position: int = 0, specialty: list[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_document_id="pubmed:1",
        text=text,
        position=position,
        metadata=DocumentMetadata(
            credibility=CredibilityLevel.PEER_REVIEWED,
            specialty=specialty or [],
            authors=["Smith J"],
            doi="10.1000/test",
            url="https://example.com",
        ),
    )


# ── Tagger tests ──────────────────────────────────────────────────────────────

def test_tagger_adds_specialty():
    tagger = Tagger()
    chunk = _chunk("The neurosurgical team performed a craniotomy.")
    result = tagger.tag(chunk)
    assert "neurosurgery" in result.metadata.specialty


def test_tagger_multiple_specialties():
    tagger = Tagger()
    chunk = _chunk("Cardiac surgery was combined with neurology consult.")
    result = tagger.tag(chunk)
    specialties = result.metadata.specialty
    assert "cardiology" in specialties or "cardiothoracic_surgery" in specialties
    assert "neurology" in specialties


def test_tagger_preserves_existing_specialty():
    tagger = Tagger()
    chunk = _chunk("A general discussion.", specialty=["oncology"])
    result = tagger.tag(chunk)
    assert "oncology" in result.metadata.specialty


# ── Embedder tests ────────────────────────────────────────────────────────────

def test_embedder_produces_vectors():
    embedder = Embedder()
    chunk = _chunk("The patient presented with chest pain.")
    results = embedder.embed([chunk])
    assert len(results) == 1
    c, vec = results[0]
    assert len(vec) == 384  # all-MiniLM-L6-v2 output dim


def test_embedder_batch():
    embedder = Embedder()
    chunks = [
        _chunk("First chunk text.", chunk_id="id1", position=0),
        _chunk("Second chunk text.", chunk_id="id2", position=1),
        _chunk("Third chunk text.", chunk_id="id3", position=2),
    ]
    results = embedder.embed(chunks)
    assert len(results) == 3
    for c, vec in results:
        assert len(vec) == 384


# ── Store tests ───────────────────────────────────────────────────────────────

def test_store_and_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Store(chroma_dir=tmpdir)
        embedder = Embedder()
        chunk = _chunk("The patient was diagnosed with epilepsy.", chunk_id="epi001")
        pairs = embedder.embed([chunk])
        store.store(pairs)
        results = store.search("epilepsy diagnosis", n_results=1)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "epi001"
        assert "score" in results[0]
        assert 0.0 <= results[0]["score"] <= 1.0


def test_store_deduplication():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Store(chroma_dir=tmpdir)
        embedder = Embedder()
        chunk = _chunk("Duplicate content for upsert test.", chunk_id="dup001")
        pairs = embedder.embed([chunk])
        store.store(pairs)
        store.store(pairs)  # store same chunk again
        assert store.count == 1
