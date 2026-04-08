from __future__ import annotations

from casecrawler.sources.base import BaseSource


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, BaseSource] = {}

    def discover(self) -> None:
        for source_cls in BaseSource.__subclasses__():
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
        for source_cls in BaseSource.__subclasses__():
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
