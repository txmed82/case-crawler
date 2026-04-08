from __future__ import annotations

import os
from abc import ABC, abstractmethod

from casecrawler.models.document import Document


class BaseSource(ABC):
    name: str
    requires_keys: list[str] = []

    @abstractmethod
    async def search(self, query: str, limit: int = 20) -> list[Document]:
        """Search the source for documents matching a query."""

    @abstractmethod
    async def fetch(self, document_id: str) -> Document:
        """Fetch full content for a specific document."""

    @classmethod
    def is_available(cls) -> bool:
        return all(os.environ.get(key) for key in cls.requires_keys)

    @classmethod
    def missing_keys(cls) -> list[str]:
        return [key for key in cls.requires_keys if not os.environ.get(key)]
