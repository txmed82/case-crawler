from __future__ import annotations


def require_package(import_name: str, extra: str):
    try:
        return __import__(import_name)
    except ImportError as exc:
        raise RuntimeError(f"Install casecrawler[{extra}] to use this backend.") from exc

