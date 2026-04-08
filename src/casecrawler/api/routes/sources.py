from __future__ import annotations

from fastapi import APIRouter

from casecrawler.sources.registry import SourceRegistry

router = APIRouter()


@router.get("/sources")
async def list_sources():
    registry = SourceRegistry()
    info = registry.all_sources_info()
    available = [s for s in info if s["available"]]
    unavailable = [s for s in info if not s["available"]]
    return {"available": available, "unavailable": unavailable}
