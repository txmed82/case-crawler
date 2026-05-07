from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse

from casecrawler.config import get_config
from casecrawler.export.cards import build_dataset_card, build_model_card
from casecrawler.export.fine_tuning import export_record
from casecrawler.generation.synthetic_pipeline import SyntheticPipeline
from casecrawler.models.dataset import (
    ExportFormat,
    GenerationRequest,
    HumanReviewDecision,
)
from casecrawler.storage.dataset_store import DatasetStore
from casecrawler.validation.benchmark import DatasetBenchmark

router = APIRouter()


@router.get("/datasets")
def list_datasets(limit: int = Query(100, ge=1, le=1000)):
    store = DatasetStore()
    return {"datasets": [manifest.model_dump() for manifest in store.list_manifests(limit=limit)]}


@router.post("/datasets/generate")
async def generate_dataset(req: GenerationRequest):
    config = get_config()
    max_count = config.synthetic.max_api_generation_count
    max_returned = config.synthetic.max_api_returned_records
    if req.count > max_count:
        raise HTTPException(
            status_code=422,
            detail=f"count must be less than or equal to {max_count}",
        )

    result = await SyntheticPipeline().generate(req)
    store = DatasetStore()
    for record in result["records"]:
        store.save_record(record)
    returned_records = result["records"][:max_returned]
    return {
        "dataset_id": result["dataset_id"],
        "generated": result["generated"],
        "approved": result["approved"],
        "total_records": len(result["records"]),
        "records": [record.model_dump() for record in returned_records],
    }


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, limit: int = Query(100, ge=1, le=1000)):
    store = DatasetStore()
    try:
        manifest = store.get_manifest(dataset_id)
    except KeyError as err:
        raise HTTPException(status_code=404, detail="dataset not found") from err
    records = store.list_records(dataset_id=dataset_id, limit=limit)
    return {
        "manifest": manifest.model_dump(),
        "records": [record.model_dump() for record in records],
    }


@router.get("/datasets/{dataset_id}/reviews")
def list_dataset_review_queue(
    dataset_id: str,
    limit: int = Query(100, ge=1, le=1000),
    include_reviewed: bool = False,
):
    store = DatasetStore()
    if not store.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="dataset not found")
    return {
        "dataset_id": dataset_id,
        "records": [
            item.model_dump()
            for item in store.list_review_queue(
                dataset_id=dataset_id,
                include_reviewed=include_reviewed,
                limit=limit,
            )
        ],
    }


@router.post("/records/{record_id}/review")
def save_record_review(record_id: str, decision: HumanReviewDecision):
    store = DatasetStore()
    try:
        record = store.save_human_review(record_id, decision)
    except KeyError as err:
        raise HTTPException(status_code=404, detail="record not found") from err
    return {
        "record_id": record.record_id,
        "dataset_id": record.dataset_id,
        "human_review": record.metadata["human_review"],
        "effective_approved": store.effective_approved(record),
    }


@router.get("/datasets/{dataset_id}/card", response_class=PlainTextResponse)
def get_dataset_card(dataset_id: str, kind: str = Query("dataset", pattern="^(dataset|model)$")):
    store = DatasetStore()
    try:
        manifest = store.get_manifest(dataset_id)
    except KeyError as err:
        raise HTTPException(status_code=404, detail="dataset not found") from err
    records = list(store.iter_records(dataset_id=dataset_id))
    if kind == "dataset":
        return build_dataset_card(manifest, records)
    return build_model_card(manifest, records)


@router.get("/datasets/{dataset_id}/benchmark")
def benchmark_dataset(
    dataset_id: str,
    reference_dataset_id: str = Query(..., min_length=1),
):
    store = DatasetStore()
    if not store.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="dataset not found")
    if not store.dataset_exists(reference_dataset_id):
        raise HTTPException(status_code=404, detail="reference dataset not found")
    generated_records = list(store.iter_records(dataset_id=dataset_id))
    reference_records = list(store.iter_records(dataset_id=reference_dataset_id))
    try:
        report = DatasetBenchmark().compare(generated_records, reference_records)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    return report.model_dump()


@router.get("/datasets/{dataset_id}/export")
def export_dataset(
    dataset_id: str,
    export_format: ExportFormat = ExportFormat.SFT_JSONL,
):
    store = DatasetStore()
    if not store.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="dataset not found")

    def _iter_jsonl():
        record_count = 0
        byte_count = 0
        try:
            for record in store.iter_records(dataset_id=dataset_id):
                line = json.dumps(export_record(record, export_format), sort_keys=True)
                record_count += 1
                byte_count += len(line.encode("utf-8")) + 1
                yield line + "\n"
        finally:
            store.save_export_manifest(
                dataset_id=dataset_id,
                export_format=export_format,
                file_path=(
                    f"api://datasets/{dataset_id}/export?"
                    f"export_format={export_format.value}"
                ),
                record_count=record_count,
                metadata={"transport": "api", "jsonl_bytes": byte_count},
            )

    return StreamingResponse(_iter_jsonl(), media_type="application/x-ndjson")
