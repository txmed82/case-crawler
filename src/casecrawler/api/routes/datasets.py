from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from casecrawler.config import get_config
from casecrawler.export.fine_tuning import export_record
from casecrawler.generation.synthetic_pipeline import SyntheticPipeline
from casecrawler.models.dataset import ExportFormat, GenerationRequest
from casecrawler.storage.dataset_store import DatasetStore

router = APIRouter()


@router.get("/datasets")
async def list_datasets(limit: int = 100):
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
async def get_dataset(dataset_id: str, limit: int = 100):
    store = DatasetStore()
    records = store.list_records(dataset_id=dataset_id, limit=limit)
    if not records:
        raise HTTPException(status_code=404, detail="dataset not found")
    return {
        "manifest": store.get_manifest(dataset_id).model_dump(),
        "records": [record.model_dump() for record in records],
    }


@router.get("/datasets/{dataset_id}/export")
async def export_dataset(dataset_id: str, format: ExportFormat = ExportFormat.SFT_JSONL):
    store = DatasetStore()
    records = store.list_records(dataset_id=dataset_id)
    if not records:
        raise HTTPException(status_code=404, detail="dataset not found")
    exported = [export_record(record, format) for record in records]
    store.save_export_manifest(
        dataset_id=dataset_id,
        export_format=format,
        file_path=f"api://datasets/{dataset_id}/export?format={format.value}",
        record_count=len(exported),
        metadata={"transport": "api", "jsonl_bytes": len(_jsonl(exported))},
    )
    return {
        "dataset_id": dataset_id,
        "format": format.value,
        "record_count": len(exported),
        "records": exported,
    }


def _jsonl(records: list[dict]) -> str:
    return "\n".join(json.dumps(record, sort_keys=True) for record in records)
