"""Source registry.

Discovery used to walk ``BaseSource.__subclasses__()`` from whichever
modules happened to be imported by the time ``discover()`` ran. That made
the available-source list a function of import order — surprising, and
fragile when tests imported sources in a different sequence.

The new shape:

- An explicit ``_REGISTERED_SOURCES`` list of concrete source classes is
  defined here, so adding a new source means appending to one list.
- ``SourceRegistry`` instances filter that list against current env
  state (so ``is_available`` reflects ``os.environ`` at construction
  time, not module-import time).
- ``get_registry()`` returns a lazily-built, thread-safe process-wide
  singleton for the common case where call sites just want "the"
  registry; tests can still ``SourceRegistry()`` a fresh instance to
  reflect a patched env.
"""

from __future__ import annotations

import threading

from casecrawler.sources.annas_archive import AnnasArchiveSource
from casecrawler.sources.base import BaseSource
from casecrawler.sources.clinicaltrials import ClinicalTrialsSource
from casecrawler.sources.dailymed import DailyMedSource
from casecrawler.sources.firecrawl import FirecrawlSource
from casecrawler.sources.glass import GlassHealthSource
from casecrawler.sources.medrxiv import MedRxivSource
from casecrawler.sources.openfda import OpenFDASource
from casecrawler.sources.pubmed import PubMedSource
from casecrawler.sources.rxnorm import RxNormSource

_REGISTERED_SOURCES: tuple[type[BaseSource], ...] = (
    PubMedSource,
    ClinicalTrialsSource,
    DailyMedSource,
    OpenFDASource,
    MedRxivSource,
    RxNormSource,
    GlassHealthSource,
    FirecrawlSource,
    AnnasArchiveSource,
)


class SourceRegistry:
    def __init__(
        self,
        source_classes: tuple[type[BaseSource], ...] | None = None,
    ) -> None:
        self._source_classes = source_classes or _REGISTERED_SOURCES
        self._sources: dict[str, BaseSource] = {}

    def discover(self) -> None:
        for source_cls in self._source_classes:
            if source_cls.is_available():
                self._sources[source_cls.name] = source_cls()

    @property
    def available_source_names(self) -> list[str]:
        return list(self._sources.keys())

    def get(self, name: str) -> BaseSource | None:
        return self._sources.get(name)

    def get_sources(self, names: list[str] | None = None) -> list[BaseSource]:
        if names is None:
            return list(self._sources.values())
        return [self._sources[n] for n in names if n in self._sources]

    def all_sources_info(self) -> list[dict]:
        info = []
        for source_cls in self._source_classes:
            available = source_cls.is_available()
            entry = {
                "name": source_cls.name,
                "requires_keys": list(source_cls.requires_keys),
                "available": available,
            }
            if not available:
                entry["missing_keys"] = source_cls.missing_keys()
            info.append(entry)
        return info


_singleton_lock = threading.Lock()
_singleton: SourceRegistry | None = None


def get_registry() -> SourceRegistry:
    """Return a lazily-built, process-wide ``SourceRegistry``.

    The first call constructs and discovers; subsequent calls return the
    same instance. Thread-safe via a module-level lock so concurrent
    request handlers don't race on first access.
    """

    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is None:
            registry = SourceRegistry()
            registry.discover()
            _singleton = registry
    return _singleton


def reset_registry() -> None:
    """Drop the cached singleton. Intended for tests that mutate the env."""

    global _singleton
    with _singleton_lock:
        _singleton = None
