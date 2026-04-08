from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel


class CredibilityLevel(str, Enum):
    GUIDELINE = "guideline"
    PEER_REVIEWED = "peer_reviewed"
    PREPRINT = "preprint"
    CURATED = "curated"
    FDA_LABEL = "fda_label"


class DocumentMetadata(BaseModel):
    authors: list[str] = []
    publication_date: date | None = None
    specialty: list[str] = []
    credibility: CredibilityLevel
    url: str | None = None
    doi: str | None = None


class Document(BaseModel):
    source: str
    source_id: str
    title: str
    content: str
    content_type: str
    metadata: DocumentMetadata


class Chunk(BaseModel):
    chunk_id: str
    source_document_id: str
    text: str
    position: int
    metadata: DocumentMetadata
