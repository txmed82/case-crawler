from __future__ import annotations

from fastapi import APIRouter, Query

from casecrawler.config import get_config
from casecrawler.pipeline.store import Store

router = APIRouter()


def get_store() -> Store:
    config = get_config()
    return Store(chroma_dir=config.storage.chroma_persist_dir)


@router.get("/search")
async def search_chunks(
    q: str = Query(..., description="Search query"),
    source: str | None = Query(None, description="Filter by source name"),
    limit: int = Query(10, description="Max results", le=100),
):
    store = get_store()
    results = store.search(q, n_results=limit, source=source)
    return {"results": results}
