from __future__ import annotations

import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from casecrawler.config import get_config
from casecrawler.pipeline.orchestrator import get_shared_orchestrator
from casecrawler.sources.registry import get_registry

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory job store. Sufficient for local single-process use; not durable
# across process restarts. The job dict has no eviction; we accept unbounded
# growth for an OSS-local tool. A multi-process deployment would replace
# this with Redis or similar.
_jobs: dict[str, dict] = {}


class IngestRequest(BaseModel):
    query: str
    sources: list[str] | None = None
    limit: int | None = None


class IngestResponse(BaseModel):
    job_id: str
    status: str


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def start_ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "summary": {}, "elapsed_seconds": 0}
    background_tasks.add_task(run_ingestion, job_id, req.query, req.sources, req.limit)
    return IngestResponse(job_id=job_id, status="running")


@router.get("/ingest/{job_id}")
async def get_ingest_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **_jobs[job_id]}


async def run_ingestion(
    job_id: str,
    query: str,
    source_names: list[str] | None,
    limit: int | None,
) -> None:
    config = get_config()
    limit = limit or config.ingestion.default_limit_per_source
    start = time.time()

    try:
        registry = get_registry()
        active_sources = registry.get_sources(source_names)

        # Fan out searches. Per-source failures are logged and recorded in
        # the job summary so callers can see *why* a source returned nothing.
        per_source_errors: dict[str, str] = {}

        async def search_one(source):
            try:
                return source.name, await source.search(query, limit=limit)
            except Exception as exc:
                logger.exception(
                    "Source %s search failed for query %r", source.name, query
                )
                per_source_errors[source.name] = f"{type(exc).__name__}: {exc}"
                return source.name, []

        results = await asyncio.gather(*[search_one(s) for s in active_sources])

        pipeline = get_shared_orchestrator()
        summary = {}
        for source_name, docs in results:
            if docs:
                result = pipeline.process(docs)
                summary[source_name] = result

        job_summary = {
            "status": "completed",
            "summary": summary,
            "elapsed_seconds": round(time.time() - start, 1),
        }
        if per_source_errors:
            job_summary["source_errors"] = per_source_errors
        _jobs[job_id] = job_summary
    except Exception as e:
        _jobs[job_id] = {
            "status": "failed",
            "error": str(e),
            "elapsed_seconds": round(time.time() - start, 1),
        }
