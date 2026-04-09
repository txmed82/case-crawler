from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from casecrawler.config import get_config
from casecrawler.generation.pipeline import GenerationPipeline
from casecrawler.generation.retriever import Retriever
from casecrawler.llm.factory import get_provider
from casecrawler.pipeline.store import Store
from casecrawler.storage.case_store import CaseStore

router = APIRouter()

_jobs: dict[str, dict] = {}


class GenerateRequest(BaseModel):
    topic: str
    difficulty: str | None = None
    count: int = 1
    ingest_first: bool = False
    multi_step: bool = False


@router.post("/generate", status_code=202)
async def start_generation(req: GenerateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running"}
    background_tasks.add_task(run_generation, job_id, req.topic, req.difficulty, req.count, req.multi_step)
    return {"job_id": job_id, "status": "running"}


@router.get("/generate/{job_id}")
async def get_generation_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **_jobs[job_id]}


async def run_generation(job_id: str, topic: str, difficulty: str | None, count: int, multi_step: bool = False) -> None:
    config = get_config()
    difficulty = difficulty or config.generation.default_difficulty
    start = time.time()

    try:
        provider = get_provider(config.llm.provider, config.llm.model, base_url=config.llm.ollama_base_url)
        store = Store(chroma_dir=config.storage.chroma_persist_dir)
        retriever = Retriever(store=store)
        if multi_step:
            from casecrawler.generation.multi_step_pipeline import MultiStepPipeline
            pipeline = MultiStepPipeline(
                provider=provider, retriever=retriever,
                max_retries=config.generation.max_retries,
                review_threshold=config.generation.review_threshold,
            )
        else:
            pipeline = GenerationPipeline(
                provider=provider, retriever=retriever,
                max_retries=config.generation.max_retries,
                review_threshold=config.generation.review_threshold,
            )

        result = await pipeline.generate_batch(topic=topic, count=count, difficulty=difficulty)

        case_store = CaseStore()
        for case in result["cases"]:
            case_store.save(case)

        _jobs[job_id] = {
            "status": "completed",
            "cases_generated": result["generated"],
            "cases_failed": result["failed"],
            "elapsed_seconds": round(time.time() - start, 1),
        }
    except Exception as e:
        _jobs[job_id] = {
            "status": "failed",
            "error": str(e),
            "elapsed_seconds": round(time.time() - start, 1),
        }
