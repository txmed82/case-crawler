from __future__ import annotations

from sentence_transformers import SentenceTransformer

from casecrawler.models.document import Chunk


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        texts = [c.text for c in chunks]
        vectors = self.model.encode(texts, show_progress_bar=False)
        return [(chunk, vec.tolist()) for chunk, vec in zip(chunks, vectors)]
