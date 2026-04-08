import tempfile

from casecrawler.generation.retriever import Retriever
from casecrawler.models.document import Chunk, CredibilityLevel, DocumentMetadata
from casecrawler.pipeline.embedder import Embedder
from casecrawler.pipeline.store import Store


def _store_test_chunks(store: Store, embedder: Embedder) -> None:
    chunks = [
        Chunk(
            chunk_id="c1", source_document_id="pubmed:1", text="SAH is a neurosurgical emergency requiring immediate CT.",
            position=0, metadata=DocumentMetadata(credibility=CredibilityLevel.GUIDELINE, specialty=["neurosurgery"]),
        ),
        Chunk(
            chunk_id="c2", source_document_id="pubmed:2", text="Thunderclap headache is the hallmark of SAH.",
            position=0, metadata=DocumentMetadata(credibility=CredibilityLevel.PEER_REVIEWED),
        ),
        Chunk(
            chunk_id="c3", source_document_id="medrxiv:3", text="Novel biomarkers for SAH prognosis.",
            position=0, metadata=DocumentMetadata(credibility=CredibilityLevel.PREPRINT),
        ),
    ]
    embedded = embedder.embed(chunks)
    store.store(embedded)


def test_retriever_returns_chunks():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Store(chroma_dir=tmpdir)
        embedder = Embedder()
        _store_test_chunks(store, embedder)
        retriever = Retriever(store=store)
        results = retriever.retrieve("subarachnoid hemorrhage", limit=10)
        assert len(results) >= 1
        assert "text" in results[0]
        assert "credibility" in results[0]
        assert "chunk_id" in results[0]


def test_retriever_orders_by_credibility():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Store(chroma_dir=tmpdir)
        embedder = Embedder()
        _store_test_chunks(store, embedder)
        retriever = Retriever(store=store)
        results = retriever.retrieve("SAH", limit=10)
        # Guidelines should come before preprints
        credibilities = [r["credibility"] for r in results]
        if "guideline" in credibilities and "preprint" in credibilities:
            assert credibilities.index("guideline") < credibilities.index("preprint")


def test_retriever_includes_source_info():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Store(chroma_dir=tmpdir)
        embedder = Embedder()
        _store_test_chunks(store, embedder)
        retriever = Retriever(store=store)
        results = retriever.retrieve("SAH", limit=10)
        for r in results:
            assert "source_document_id" in r
            assert "chunk_id" in r
