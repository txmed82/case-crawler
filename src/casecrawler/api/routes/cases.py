from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from casecrawler.storage.case_store import CaseStore

router = APIRouter()


def get_case_store() -> CaseStore:
    return CaseStore()


@router.get("/cases")
async def list_cases(
    topic: str | None = Query(None),
    difficulty: str | None = Query(None),
    min_accuracy: float | None = Query(None),
    limit: int = Query(20, le=100),
):
    store = get_case_store()
    cases = store.list_cases(topic=topic, difficulty=difficulty, min_accuracy=min_accuracy, limit=limit)
    return {
        "cases": [case.model_dump() for case in cases],
        "total": len(cases),
    }


@router.get("/cases/export")
async def export_cases(
    topic: str | None = Query(None),
    difficulty: str | None = Query(None),
):
    store = get_case_store()
    lines = store.export_jsonl(topic=topic, difficulty=difficulty)

    def generate():
        for line in lines:
            yield line + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.get("/cases/{case_id}")
async def get_case(case_id: str):
    store = get_case_store()
    case = store.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return case.model_dump()
