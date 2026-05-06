from __future__ import annotations

from fastapi import APIRouter

from casecrawler.generation.synthetic_pipeline import SyntheticPipeline
from casecrawler.models.dataset import GenerationRequest
from casecrawler.storage.dataset_store import DatasetStore

router = APIRouter()


@router.post("/datasets/generate")
async def generate_dataset(req: GenerationRequest):
    result = await SyntheticPipeline().generate(req)
    store = DatasetStore()
    for record in result["records"]:
        store.save_record(record)
    return {
        "dataset_id": result["dataset_id"],
        "generated": result["generated"],
        "approved": result["approved"],
        "records": [record.model_dump() for record in result["records"]],
    }
