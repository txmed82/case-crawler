from __future__ import annotations

from fastapi import APIRouter, HTTPException

from casecrawler.config import get_config
from casecrawler.generation.synthetic_pipeline import SyntheticPipeline
from casecrawler.models.dataset import GenerationRequest
from casecrawler.storage.dataset_store import DatasetStore

router = APIRouter()


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
