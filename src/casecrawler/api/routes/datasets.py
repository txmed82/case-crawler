from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from casecrawler.config import get_config
from casecrawler.export.cards import build_dataset_card, build_model_card
from casecrawler.export.fine_tuning import export_parquet_bytes, export_record_payloads
from casecrawler.generation.synthetic_pipeline import SyntheticPipeline
from casecrawler.models.dataset import (
    ExportFormat,
    GenerationRequest,
    HumanReviewDecision,
)
from casecrawler.models.synthetic import ComplexityProfile, Modality
from casecrawler.storage.dataset_store import DatasetStore
from casecrawler.validation.benchmark import DatasetBenchmark
from casecrawler.validation.quality import build_dataset_quality_report

router = APIRouter()


class ReferenceImportRequest(BaseModel):
    reference_key: str | None = Field(default=None, min_length=1)
    dataset_id: str = Field(min_length=1)
    repo_id: str | None = Field(default=None, min_length=1)
    split: str | None = None
    license: str | None = None
    note_field: str = "note"
    question_field: str | None = None
    answer_field: str | None = None
    task_field: str | None = None
    patient_id_field: str | None = None
    image_field: str | None = None
    image_label_field: str | None = None
    image_modality: str = "XR"
    image_body_region: str = "chest"
    limit: int | None = Field(default=None, ge=1)


class SyntheaImportRequest(BaseModel):
    path: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)


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

    try:
        result = await SyntheticPipeline().generate(req)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
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


@router.get("/datasets/reference-catalog")
def list_reference_catalog():
    from casecrawler.integrations.huggingface import REFERENCE_DATASETS
    from casecrawler.integrations.synthea import (
        SYNTHEA_REFERENCE_DESCRIPTION,
        SYNTHEA_REFERENCE_KEY,
    )

    datasets = [
        {
            "key": SYNTHEA_REFERENCE_KEY,
            "repo_id": None,
            "split": None,
            "license": "synthetic-local",
            "description": SYNTHEA_REFERENCE_DESCRIPTION,
            "image_field": None,
            "image_label_field": None,
            "image_modality": None,
            "image_body_region": None,
            "gated": False,
            "use_policy": "local_synthea_import",
            "source": "synthea",
        }
    ]
    datasets.extend(
        {
            "key": key,
            "repo_id": spec.repo_id,
            "split": spec.split,
            "license": spec.license,
            "description": spec.description,
            "image_field": spec.image_field,
            "image_label_field": spec.image_label_field,
            "image_modality": spec.image_modality,
            "image_body_region": spec.image_body_region,
            "gated": spec.gated,
            "use_policy": spec.use_policy,
            "source": "huggingface",
        }
        for key, spec in REFERENCE_DATASETS.items()
    )
    return {
        "datasets": datasets
    }


@router.get("/datasets/capabilities")
def list_dataset_capabilities():
    from casecrawler.generation.imaging_models import list_imaging_model_profiles
    from casecrawler.generation.recipes import list_generation_recipes
    from casecrawler.generation.structured_generator import list_clinical_profile_catalog
    from casecrawler.generation.timeseries_models import list_time_series_model_profiles

    return {
        "modalities": [modality.value for modality in Modality],
        "complexity_profiles": [profile.value for profile in ComplexityProfile],
        "export_formats": [export_format.value for export_format in ExportFormat],
        "generation_recipes": [
            {
                "name": recipe.name,
                "description": recipe.description,
                "complexity": recipe.complexity.value,
                "modalities": [modality.value for modality in recipe.modalities],
                "export_formats": [
                    export_format.value for export_format in recipe.export_formats
                ],
                "cohort_constraints": recipe.cohort_constraints,
                "validation_threshold": recipe.validation_threshold,
                "recommended_reference_keys": recipe.recommended_reference_keys,
                "benchmark_thresholds": {
                    "min_overall_score": recipe.benchmark_min_overall_score,
                    "min_metric_score": recipe.benchmark_min_metric_score,
                },
            }
            for recipe in list_generation_recipes()
        ],
        "cohort_constraints": [
            "age_min",
            "age_max",
            "sexes",
            "sex_cycle",
            "topic_mix",
            "topic_mix_weights",
            "base_time",
        ],
        "imaging_model_profiles": [
            {
                "name": profile.name,
                "model_id": profile.model_id,
                "modality": profile.modality,
                "body_region": profile.body_region,
                "license": profile.license,
                "notes": profile.notes,
            }
            for profile in list_imaging_model_profiles()
        ],
        "time_series_model_profiles": [
            {
                "name": profile.name,
                "adapter_type": profile.adapter_type,
                "reference": profile.reference,
                "notes": profile.notes,
            }
            for profile in list_time_series_model_profiles()
        ],
        "clinical_profiles": [
            {
                "key": profile.key,
                "keywords": list(profile.keywords),
                "diagnosis_display": profile.diagnosis_display,
                "diagnosis_code": profile.diagnosis_code,
                "lab_names": profile.lab_names,
                "vital_names": profile.vital_names,
                "medication_names": profile.medication_names,
            }
            for profile in list_clinical_profile_catalog()
        ],
        "validators": [
            {
                "key": "lexical",
                "requires": [],
                "description": "No-key prompt/report token overlap validator.",
            },
            {
                "key": "biomedclip",
                "requires": ["casecrawler[imaging]"],
                "description": "Optional BiomedCLIP image-text alignment scorer.",
            },
            {
                "key": "medgemma",
                "requires": ["casecrawler[hf]", "casecrawler[imaging]", "accepted model terms"],
                "description": "Optional gated MedGemma image/report reasoning validator.",
            },
        ],
    }


@router.post("/datasets/reference-import")
def import_reference_dataset(req: ReferenceImportRequest):
    from casecrawler.integrations.huggingface import (
        REFERENCE_DATASETS,
        import_reference_rows,
        load_huggingface_dataset,
        load_reference_dataset,
        reference_dataset_spec,
    )

    if req.repo_id:
        split = req.split or "train"
        spec = reference_dataset_spec(
            repo_id=req.repo_id,
            split=split,
            license=req.license or "unspecified",
            note_field=req.note_field,
            question_field=req.question_field,
            answer_field=req.answer_field,
                task_field=req.task_field,
                patient_id_field=req.patient_id_field,
                image_field=req.image_field,
                image_label_field=req.image_label_field,
                image_modality=req.image_modality,
                image_body_region=req.image_body_region,
                description="User-specified Hugging Face reference dataset.",
            )
        try:
            rows = load_huggingface_dataset(req.repo_id, split=split, streaming=True)
            records = import_reference_rows(
                rows,
                dataset_id=req.dataset_id,
                reference_key=req.reference_key or req.repo_id,
                split=split,
                limit=req.limit,
                spec=spec,
            )
        except RuntimeError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        reference_key = req.reference_key or req.repo_id
    elif req.reference_key not in REFERENCE_DATASETS:
        raise HTTPException(status_code=404, detail="reference dataset not found")
    else:
        assert req.reference_key is not None
        spec = REFERENCE_DATASETS[req.reference_key]
        try:
            rows = load_reference_dataset(
                req.reference_key,
                split=req.split,
                streaming=True,
            )
            records = import_reference_rows(
                rows,
                dataset_id=req.dataset_id,
                reference_key=req.reference_key,
                split=req.split,
                limit=req.limit,
            )
        except RuntimeError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        reference_key = req.reference_key

    store = DatasetStore()
    for record in records:
        store.save_record(record)
    return {
        "dataset_id": req.dataset_id,
        "imported": len(records),
        "reference_key": reference_key,
        "repo_id": spec.repo_id,
        "split": req.split or spec.split,
        "license": spec.license,
    }


@router.post("/datasets/synthea-import")
def import_synthea_fhir(req: SyntheaImportRequest):
    from json import JSONDecodeError

    from casecrawler.integrations.synthea import SyntheaAdapter

    try:
        records = SyntheaAdapter().import_fhir_path(req.path, dataset_id=req.dataset_id)
    except (OSError, JSONDecodeError) as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    if not records:
        raise HTTPException(
            status_code=404,
            detail="no Synthea FHIR JSON bundles or NDJSON files found",
        )
    store = DatasetStore()
    for record in records:
        store.save_record(record)
    return {
        "dataset_id": req.dataset_id,
        "imported": len(records),
        "source": "synthea_fhir",
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


@router.get("/datasets/{dataset_id}/quality")
def get_dataset_quality(dataset_id: str):
    from casecrawler.validation.benchmark_selection import build_benchmark_plan_summary

    store = DatasetStore()
    if not store.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="dataset not found")
    records = list(store.iter_records(dataset_id=dataset_id))
    return build_dataset_quality_report(
        dataset_id,
        records,
        effective_approved=store.effective_approved,
        benchmark_plan=build_benchmark_plan_summary(store, dataset_id),
    ).model_dump()


@router.get("/datasets/{dataset_id}/exports")
def list_dataset_exports(
    dataset_id: str,
    limit: int = Query(100, ge=1, le=1000),
):
    store = DatasetStore()
    if not store.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="dataset not found")
    return {
        "dataset_id": dataset_id,
        "exports": [
            export_manifest.model_dump()
            for export_manifest in store.list_export_manifests(
                dataset_id=dataset_id,
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


@router.get("/datasets/{dataset_id}/images/{image_id}")
def get_dataset_image(dataset_id: str, image_id: str):
    store = DatasetStore()
    if not store.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="dataset not found")
    for record in store.iter_records(dataset_id=dataset_id):
        for image in record.imaging:
            if image.image_id != image_id:
                continue
            if not image.file_path:
                raise HTTPException(status_code=404, detail="image file not available")
            image_path = Path(image.file_path).expanduser()
            if not image_path.is_file():
                raise HTTPException(status_code=404, detail="image file not found")
            return FileResponse(
                image_path,
                filename=image_path.name,
                media_type=_image_media_type(image_path),
            )
    raise HTTPException(status_code=404, detail="image not found")


@router.get("/datasets/{dataset_id}/benchmark")
def benchmark_dataset(
    dataset_id: str,
    reference_dataset_id: str = Query(..., min_length=1),
    min_overall_score: float = Query(0.75, ge=0.0, le=1.0),
    min_metric_score: float = Query(0.5, ge=0.0, le=1.0),
):
    store = DatasetStore()
    if not store.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="dataset not found")
    if not store.dataset_exists(reference_dataset_id):
        raise HTTPException(status_code=404, detail="reference dataset not found")
    generated_records = list(store.iter_records(dataset_id=dataset_id))
    reference_records = list(store.iter_records(dataset_id=reference_dataset_id))
    try:
        report = DatasetBenchmark(
            min_overall_score=min_overall_score,
            min_metric_score=min_metric_score,
        ).compare(generated_records, reference_records)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    return report.model_dump()


@router.get("/datasets/{dataset_id}/benchmark-plan")
def benchmark_plan(dataset_id: str):
    from casecrawler.validation.benchmark_selection import build_benchmark_plan_summary

    store = DatasetStore()
    if not store.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="dataset not found")
    return build_benchmark_plan_summary(store, dataset_id)


@router.get("/datasets/{dataset_id}/benchmark-suite")
def benchmark_suite(dataset_id: str):
    from casecrawler.validation.benchmark_selection import run_recommended_benchmark_suite

    store = DatasetStore()
    if not store.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="dataset not found")
    try:
        return run_recommended_benchmark_suite(store, dataset_id)
    except LookupError as err:
        raise HTTPException(status_code=409, detail=str(err)) from err
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err


@router.get("/datasets/{dataset_id}/export")
def export_dataset(
    dataset_id: str,
    export_format: ExportFormat = ExportFormat.SFT_JSONL,
    allow_blocked: bool = False,
    reference_dataset_id: str | None = Query(default=None, min_length=1),
    auto_benchmark: bool = False,
    min_overall_score: float = Query(0.75, ge=0.0, le=1.0),
    min_metric_score: float = Query(0.5, ge=0.0, le=1.0),
):
    store = DatasetStore()
    if not store.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="dataset not found")
    records = list(store.iter_records(dataset_id=dataset_id))
    if not allow_blocked:
        report = build_dataset_quality_report(
            dataset_id,
            records,
            effective_approved=store.effective_approved,
        )
        if not report.export_ready:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Dataset is not ready for fine-tuning export. "
                    f"Blockers: {report.issue_counts_by_field}. "
                    "Set allow_blocked=true to export anyway."
                ),
            )
    try:
        from casecrawler.validation.benchmark_selection import resolve_benchmark_gate

        benchmark_gate = resolve_benchmark_gate(
            store,
            dataset_id=dataset_id,
            reference_dataset_id=reference_dataset_id,
            auto_benchmark=auto_benchmark,
            min_overall_score=min_overall_score,
            min_metric_score=min_metric_score,
        )
    except KeyError as err:
        raise HTTPException(status_code=404, detail="reference dataset not found") from err
    except LookupError as err:
        raise HTTPException(status_code=409, detail=str(err)) from err

    benchmark_metadata = {}
    if benchmark_gate:
        reference_records = list(
            store.iter_records(dataset_id=benchmark_gate.reference_dataset_id)
        )
        try:
            benchmark_report = DatasetBenchmark(
                min_overall_score=benchmark_gate.min_overall_score,
                min_metric_score=benchmark_gate.min_metric_score,
            ).compare(records, reference_records)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        benchmark_metadata = {
            "benchmark_reference_dataset_id": benchmark_gate.reference_dataset_id,
            "benchmark_reference_key": benchmark_gate.reference_key,
            "benchmark_auto_selected": benchmark_gate.auto_selected,
            "benchmark_overall_score": benchmark_report.overall_score,
            "benchmark_passed": benchmark_report.passed,
            "benchmark_failing_metrics": benchmark_report.failing_metrics,
            "benchmark_thresholds": benchmark_report.thresholds,
        }
        if not benchmark_report.passed and not allow_blocked:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Dataset failed benchmark gate for fine-tuning export. "
                    f"Overall score: {benchmark_report.overall_score}. "
                    f"Failing metrics: {benchmark_report.failing_metrics}. "
                    "Set allow_blocked=true to export anyway."
                ),
            )

    if export_format == ExportFormat.PARQUET:
        try:
            payload, record_count = export_parquet_bytes(records)
        except ImportError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        store.save_export_manifest(
            dataset_id=dataset_id,
            export_format=export_format,
            file_path=(
                f"api://datasets/{dataset_id}/export?"
                f"export_format={export_format.value}"
            ),
            record_count=record_count,
            metadata={
                "transport": "api",
                "parquet_bytes": len(payload),
                **benchmark_metadata,
            },
        )
        return Response(
            content=payload,
            media_type="application/vnd.apache.parquet",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{dataset_id}.parquet"'
                )
            },
        )

    def _iter_jsonl():
        record_count = 0
        byte_count = 0
        try:
            for record in records:
                for payload in export_record_payloads(record, export_format):
                    line = json.dumps(payload, sort_keys=True)
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
                metadata={
                    "transport": "api",
                    "jsonl_bytes": byte_count,
                    **benchmark_metadata,
                },
            )

    return StreamingResponse(_iter_jsonl(), media_type="application/x-ndjson")


def _image_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"
