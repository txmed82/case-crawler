from __future__ import annotations

import chromadb
from chromadb.config import Settings

from casecrawler.models.document import Chunk


class Store:
    def __init__(self, chroma_dir: str = "./data/chroma") -> None:
        self._client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="casecrawler",
            metadata={"hnsw:space": "cosine"},
        )

    def store(self, pairs: list[tuple[Chunk, list[float]]]) -> None:
        if not pairs:
            return

        ids = [chunk.chunk_id for chunk, _ in pairs]
        embeddings = [vec for _, vec in pairs]
        documents = [chunk.text for chunk, _ in pairs]
        metadatas = [self._build_metadata(chunk) for chunk, _ in pairs]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(self, query: str, n_results: int = 10, source: str | None = None) -> list[dict]:
        where = {"source": source} if source else None
        query_kwargs: dict = {
            "query_texts": [query],
            "n_results": min(n_results, max(self._collection.count(), 1)),
        }
        if where:
            query_kwargs["where"] = where
        results = self._collection.query(**query_kwargs)
        output = []
        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            output.append(
                {
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - distance,
                }
            )
        return output

    @property
    def count(self) -> int:
        return self._collection.count()

    @staticmethod
    def _build_metadata(chunk: Chunk) -> dict:
        source = chunk.source_document_id.split(":")[0] if ":" in chunk.source_document_id else ""
        return {
            "source_document_id": chunk.source_document_id,
            "source": source,
            "position": chunk.position,
            "credibility": chunk.metadata.credibility.value,
            "specialty": ",".join(chunk.metadata.specialty),
            "authors": ",".join(chunk.metadata.authors),
            "doi": chunk.metadata.doi or "",
            "url": chunk.metadata.url or "",
        }
