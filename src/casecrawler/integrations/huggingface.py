from __future__ import annotations

import importlib


def require_package(import_name: str, extra: str):
    try:
        return importlib.import_module(import_name)
    except ModuleNotFoundError as exc:
        if exc.name != import_name:
            raise
        raise RuntimeError(f"Install casecrawler[{extra}] to use this backend.") from exc
