from __future__ import annotations

import threading

from casecrawler.config import get_config
from casecrawler.models.document import Document
from casecrawler.pipeline.chunker import Chunker
from casecrawler.pipeline.embedder import Embedder
from casecrawler.pipeline.store import Store
from casecrawler.pipeline.tagger import Tagger


_SHARED_ORCHESTRATOR: "PipelineOrchestrator | None" = None
_SHARED_ORCHESTRATOR_LOCK = threading.Lock()


class PipelineOrchestrator:
    def __init__(
        self,
        chroma_dir: str | None = None,
        embedding_model: str | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> None:
        cfg = get_config()

        resolved_chroma_dir = chroma_dir or cfg.storage.chroma_persist_dir
        resolved_model = embedding_model or cfg.embedding.model
        resolved_chunk_size = chunk_size if chunk_size is not None else cfg.chunking.default_chunk_size
        resolved_overlap = overlap if overlap is not None else cfg.chunking.overlap

        self._chunker = Chunker(chunk_size=resolved_chunk_size, overlap=resolved_overlap)
        self._tagger = Tagger()
        self._embedder = Embedder(model_name=resolved_model)
        self.store = Store(
            chroma_dir=resolved_chroma_dir,
            query_embedder=self._embedder.embed_text,
        )

    def process(self, documents: list[Document]) -> dict[str, int]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self._chunker.chunk(doc))

        tagged = self._tagger.tag_all(all_chunks)
        pairs = self._embedder.embed(tagged)
        self.store.store(pairs)

        return {"documents": len(documents), "chunks": len(all_chunks)}


def get_shared_orchestrator() -> PipelineOrchestrator:
    """Return a process-wide PipelineOrchestrator.

    ChromaDB's PersistentClient acquires a file lock on its data directory;
    creating a new orchestrator (and therefore a new client) per request
    causes contention or outright failures when concurrent ingest jobs run.
    Long-lived consumers should share a single instance via this accessor.
    """

    global _SHARED_ORCHESTRATOR
    with _SHARED_ORCHESTRATOR_LOCK:
        if _SHARED_ORCHESTRATOR is None:
            _SHARED_ORCHESTRATOR = PipelineOrchestrator()
    return _SHARED_ORCHESTRATOR


def reset_shared_orchestrator() -> None:
    """Drop the shared orchestrator. Tests use this to start each case fresh."""

    global _SHARED_ORCHESTRATOR
    with _SHARED_ORCHESTRATOR_LOCK:
        _SHARED_ORCHESTRATOR = None
