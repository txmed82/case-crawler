from __future__ import annotations

from pydantic import BaseModel


class SourcesConfig(BaseModel):
    priority: list[str] = [
        "pubmed", "glass", "openfda", "annas_archive",
        "dailymed", "rxnorm", "medrxiv", "clinicaltrials", "firecrawl",
    ]
    disabled: list[str] = []


class IngestionConfig(BaseModel):
    default_limit_per_source: int = 20
    sources: SourcesConfig = SourcesConfig()


class ChunkingConfig(BaseModel):
    default_chunk_size: int = 500
    overlap: int = 50


class EmbeddingConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"


class StorageConfig(BaseModel):
    chroma_persist_dir: str = "./data/chroma"


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class LlmConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    ollama_base_url: str = "http://localhost:11434"


class GenerationConfig(BaseModel):
    max_retries: int = 3
    review_threshold: float = 0.7
    default_difficulty: str = "resident"
    retriever_chunk_count: int = 25


class AppConfig(BaseModel):
    ingestion: IngestionConfig = IngestionConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    storage: StorageConfig = StorageConfig()
    api: ApiConfig = ApiConfig()
    llm: LlmConfig = LlmConfig()
    generation: GenerationConfig = GenerationConfig()
