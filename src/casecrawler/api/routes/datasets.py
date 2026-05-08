from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from casecrawler.config import get_config
from casecrawler.export.cards import build_dataset_card, build_model_card
from casecrawler.capabilities import (
    image_validator_capabilities,
    reference_dataset_capabilities,
    release_coverage_requirements,
)
from casecrawler.export.fine_tuning import (
    export_jsonl_split_package,
    export_parquet_bytes,
    export_record_payloads,
    summarize_export_task_coverage,
)
from casecrawler.export.release_audit import build_objective_coverage_audit
from casecrawler.generation.synthetic_pipeline import SyntheticPipeline
from casecrawler.models.dataset import (
    ExportFormat,
    GenerationRequest,
    HumanReviewDecision,
)
from casecrawler.models.synthetic import ComplexityProfile, Modality
from casecrawler.storage.dataset_store import DatasetStore
from casecrawler.validation.benchmark import (
    DatasetBenchmark,
    benchmark_profile_artifact,
    parse_benchmark_profile_artifact,
    profile_records,
)
from casecrawler.validation.quality import build_dataset_quality_report, export_profile_blocker

router = APIRouter()


class ReferenceImportRequest(BaseModel):
    reference_key: str | None = Field(default=None, min_length=1)
    dataset_id: str = Field(min_length=1)
    fixture: bool = False
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
    lab_values_field: str | None = None
    vital_values_field: str | None = None
    medications_field: str | None = None
    time_series_field: str | None = None
    limit: int | None = Field(default=None, ge=1)


class SyntheaImportRequest(BaseModel):
    path: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)


class BenchmarkProfileCompareRequest(BaseModel):
    profile: dict
    reference_profile: dict
    min_overall_score: float = Field(default=0.75, ge=0.0, le=1.0)
    min_metric_score: float = Field(default=0.5, ge=0.0, le=1.0)


class ReleasePackageRequest(BaseModel):
    topic: str = Field(min_length=1)
    count: int = Field(default=1, ge=1)
    recipe: str = "full_multimodal_acute_care"
    export_format: ExportFormat = ExportFormat.MULTIMODAL_JSONL
    train_ratio: float = Field(default=0.8, ge=0.0)
    validation_ratio: float = Field(default=0.1, ge=0.0)
    test_ratio: float = Field(default=0.1, ge=0.0)
    seed: str = "casecrawler"
    imaging_backend: Literal["placeholder", "diffusers"] = "placeholder"
    imaging_model_profile: str | None = Field(default=None, min_length=1)
    diffusers_model_id: str | None = Field(default=None, min_length=1)
    fixture_limit: int = Field(default=1, ge=1)
    min_overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    min_metric_score: float = Field(default=0.0, ge=0.0, le=1.0)


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


@router.post("/datasets/release-package")
async def generate_release_package(req: ReleasePackageRequest):
    config = get_config()
    if req.count > config.synthetic.max_api_generation_count:
        raise HTTPException(
            status_code=422,
            detail=(
                "count must be less than or equal to "
                f"{config.synthetic.max_api_generation_count}"
            ),
        )
    if req.export_format == ExportFormat.PARQUET:
        raise HTTPException(
            status_code=422,
            detail="Release package export writes JSONL split packages, not parquet.",
        )
    try:
        generation_request = GenerationRequest(
            topic=req.topic,
            count=req.count,
            recipe=req.recipe,
            export_formats=[req.export_format],
            imaging_backend=req.imaging_backend,
            imaging_model_profile=req.imaging_model_profile,
            diffusers_model_id=req.diffusers_model_id,
        )
        result = await SyntheticPipeline().generate(generation_request)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    from casecrawler.integrations.reference_fixtures import (
        seed_recommended_reference_fixtures,
    )
    from casecrawler.validation.benchmark_selection import (
        build_benchmark_plan_summary,
        run_recommended_benchmark_suite,
    )

    store = DatasetStore()
    records = result["records"]
    for record in records:
        store.save_record(record)
    dataset_id = result["dataset_id"]
    seeded_references = seed_recommended_reference_fixtures(
        store,
        dataset_id=dataset_id,
        dataset_id_prefix=f"{dataset_id}-release-fixture",
        limit=req.fixture_limit,
    )
    benchmark_plan_summary = build_benchmark_plan_summary(store, dataset_id)
    reference_dataset_id = benchmark_plan_summary.get("resolved_reference_dataset_id")
    if not isinstance(reference_dataset_id, str) or not reference_dataset_id:
        raise HTTPException(
            status_code=409,
            detail=(
                "Generated dataset has no seeded benchmark reference. "
                f"Missing: {benchmark_plan_summary.get('missing_reference_keys', [])}."
            ),
        )
    reference_records = list(store.iter_records(dataset_id=reference_dataset_id))
    try:
        benchmark_report = DatasetBenchmark(
            min_overall_score=req.min_overall_score,
            min_metric_score=req.min_metric_score,
        ).compare(records, reference_records)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    try:
        benchmark_suite = run_recommended_benchmark_suite(
            store,
            dataset_id,
            min_overall_score=req.min_overall_score,
            min_metric_score=req.min_metric_score,
        )
    except (LookupError, ValueError) as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    benchmark_reference_key = benchmark_plan_summary.get("resolved_reference_key")
    benchmark_plan = {
        "recommended_reference_keys": benchmark_plan_summary["recommended_reference_keys"],
        "ready": benchmark_report.passed,
        "resolved_reference_dataset_id": reference_dataset_id
        if benchmark_report.passed
        else None,
        "missing_reference_keys": []
        if benchmark_report.passed
        else [benchmark_reference_key or reference_dataset_id],
        "thresholds": benchmark_report.thresholds,
        "task_export_reference_readiness": benchmark_plan_summary[
            "task_export_reference_readiness"
        ],
    }
    quality_report = build_dataset_quality_report(
        dataset_id,
        records,
        effective_approved=store.effective_approved,
        benchmark_plan=benchmark_plan,
    )
    if not quality_report.export_ready:
        raise HTTPException(
            status_code=409,
            detail=(
                "Generated dataset is not ready for fine-tuning export. "
                f"Blockers: {quality_report.issue_counts_by_field}."
            ),
        )
    profile_blocker = export_profile_blocker(quality_report, req.export_format.value)
    if profile_blocker:
        raise HTTPException(status_code=409, detail=profile_blocker)
    if not quality_report.multimodal_release_ready:
        raise HTTPException(
            status_code=409,
            detail=(
                "Generated dataset is not multimodal-release-ready. "
                f"Missing: {quality_report.multimodal_release_missing}."
            ),
        )

    manifest_snapshot = store.get_manifest(dataset_id)
    try:
        with TemporaryDirectory() as temp_dir:
            task_coverage = summarize_export_task_coverage(
                records,
                req.export_format,
            )
            objective_coverage = build_objective_coverage_audit(
                quality_report=quality_report,
                benchmark_suite=benchmark_suite,
                manifest={
                    "task_coverage": task_coverage,
                    "image_artifacts": {
                        image.image_id: True
                        for record in records
                        for image in record.imaging
                        if image.file_path
                    },
                    "audit_artifacts": {
                        name: True
                        for name in (
                            "benchmark_profile.json",
                            "benchmark_report.json",
                            "benchmark_suite_report.json",
                            "dataset_card.md",
                            "model_card.md",
                            "quality_report.json",
                            "release_package_summary.json",
                        )
                    },
                },
            )
            manifest = export_jsonl_split_package(
                records,
                temp_dir,
                req.export_format,
                dataset_id=dataset_id,
                train_ratio=req.train_ratio,
                validation_ratio=req.validation_ratio,
                test_ratio=req.test_ratio,
                seed=req.seed,
                audit_artifacts={
                    "quality_report.json": quality_report.model_dump(mode="json"),
                    "benchmark_profile.json": benchmark_profile_artifact(
                        profile_records(records)
                    ),
                    "benchmark_report.json": benchmark_report.model_dump(mode="json"),
                    "benchmark_suite_report.json": benchmark_suite,
                    "dataset_card.md": build_dataset_card(manifest_snapshot, records),
                    "model_card.md": build_model_card(manifest_snapshot, records),
                    "release_package_summary.json": {
                        "dataset_id": dataset_id,
                        "generated": result["generated"],
                        "approved": result["approved"],
                        "seeded_references": seeded_references,
                        "task_coverage": task_coverage,
                        "objective_coverage": objective_coverage,
                        "quality_report": {
                            "export_ready": quality_report.export_ready,
                            "multimodal_release_ready": (
                                quality_report.multimodal_release_ready
                            ),
                            "multimodal_release_missing": (
                                quality_report.multimodal_release_missing
                            ),
                            "core_artifact_coverage": quality_report.core_artifact_coverage,
                            "clinical_text_model_policy_counts": (
                                quality_report.clinical_text_model_policy_counts
                            ),
                            "time_series_channel_counts": (
                                quality_report.time_series_channel_counts
                            ),
                            "time_series_model_policy_counts": (
                                quality_report.time_series_model_policy_counts
                            ),
                            "image_validator_policy_counts": (
                                quality_report.image_validator_policy_counts
                            ),
                            "mean_time_series_points": (
                                quality_report.mean_time_series_points
                            ),
                            "mean_time_series_duration_hours": (
                                quality_report.mean_time_series_duration_hours
                            ),
                            "mean_imaging_prompt_chars": (
                                quality_report.mean_imaging_prompt_chars
                            ),
                            "mean_imaging_report_chars": (
                                quality_report.mean_imaging_report_chars
                            ),
                            "imaging_report_label_evidence_rate": (
                                quality_report.imaging_report_label_evidence_rate
                            ),
                            "mean_imaging_width": quality_report.mean_imaging_width,
                            "mean_imaging_height": quality_report.mean_imaging_height,
                            "mean_modality_alignment_score": (
                                quality_report.mean_modality_alignment_score
                            ),
                        },
                        "benchmark": {
                            "reference_dataset_id": reference_dataset_id,
                            "reference_key": benchmark_reference_key,
                            "passed": benchmark_report.passed,
                            "overall_score": benchmark_report.overall_score,
                            "failing_metrics": benchmark_report.failing_metrics,
                            "thresholds": benchmark_report.thresholds,
                        },
                        "benchmark_suite": {
                            "passed": benchmark_suite["passed"],
                            "reference_count": benchmark_suite["reference_count"],
                            "mean_overall_score": benchmark_suite[
                                "mean_overall_score"
                            ],
                            "task_export_results": benchmark_suite[
                                "task_export_results"
                            ],
                        },
                    },
                },
            )
            payload = BytesIO()
            with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(Path(temp_dir).rglob("*")):
                    if path.is_file():
                        archive.write(path, arcname=path.relative_to(temp_dir))
            zip_bytes = payload.getvalue()
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    store.save_export_manifest(
        dataset_id=dataset_id,
        export_format=req.export_format,
        file_path=(
            f"api://datasets/release-package?export_format={req.export_format.value}"
        ),
        record_count=manifest["example_count"],
        metadata={
            "transport": "api",
            "split_package": True,
            "release_package": True,
            "zip_bytes": len(zip_bytes),
            "seed": manifest["seed"],
            "ratios": manifest["ratios"],
            "audit_artifacts": {
                name: Path(path).name for name, path in manifest["audit_artifacts"].items()
            },
            "image_artifacts": manifest["image_artifacts"],
            "image_artifact_count": len(manifest["image_artifacts"]),
            "multimodal_release_ready": quality_report.multimodal_release_ready,
            "multimodal_release_missing": quality_report.multimodal_release_missing,
            "core_artifact_coverage": quality_report.core_artifact_coverage,
            "benchmark_reference_dataset_id": reference_dataset_id,
            "benchmark_reference_key": benchmark_reference_key,
            "benchmark_auto_selected": True,
            "benchmark_overall_score": benchmark_report.overall_score,
            "benchmark_passed": benchmark_report.passed,
            "benchmark_suite_passed": benchmark_suite["passed"],
            "benchmark_suite_reference_count": benchmark_suite["reference_count"],
            "benchmark_failing_metrics": benchmark_report.failing_metrics,
            "benchmark_thresholds": benchmark_report.thresholds,
            "objective_coverage": objective_coverage,
            "splits": {
                name: {
                    "record_count": data["record_count"],
                    "example_count": data["example_count"],
                }
                for name, data in manifest["splits"].items()
            },
        },
    )
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{dataset_id}-{req.export_format.value}-release.zip"'
            ),
            "X-CaseCrawler-Dataset-Id": dataset_id,
        },
    )


@router.get("/datasets/reference-catalog")
def list_reference_catalog():
    return {"datasets": reference_dataset_capabilities()}


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
        "release_coverage_requirements": release_coverage_requirements(),
        "reference_datasets": reference_dataset_capabilities(),
        "imaging_model_profiles": [
            {
                "name": profile.name,
                "model_id": profile.model_id,
                "adapter_type": profile.adapter_type,
                "modality": profile.modality,
                "body_region": profile.body_region,
                "license": profile.license,
                "gated": profile.gated,
                "use_policy": profile.use_policy,
                "command_template": profile.command_template,
                "input_contract": profile.input_contract,
                "output_contract": profile.output_contract,
                "validation_requirements": profile.validation_requirements,
                "notes": profile.notes,
            }
            for profile in list_imaging_model_profiles()
        ],
        "time_series_model_profiles": [
            {
                "name": profile.name,
                "adapter_type": profile.adapter_type,
                "reference": profile.reference,
                "model_id": profile.model_id,
                "license": profile.license,
                "gated": profile.gated,
                "use_policy": profile.use_policy,
                "command_template": profile.command_template,
                "input_contract": profile.input_contract,
                "output_contract": profile.output_contract,
                "validation_requirements": profile.validation_requirements,
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
        "validators": image_validator_capabilities(),
    }


@router.post("/datasets/reference-import")
def import_reference_dataset(req: ReferenceImportRequest):
    if req.fixture:
        from casecrawler.integrations.reference_fixtures import import_reference_fixture

        if not req.reference_key:
            raise HTTPException(
                status_code=422,
                detail="reference_key is required for bundled fixture imports",
            )
        try:
            records = import_reference_fixture(
                req.reference_key,
                dataset_id=req.dataset_id,
                limit=req.limit,
            )
        except KeyError as err:
            raise HTTPException(status_code=404, detail=str(err)) from err
        store = DatasetStore()
        for record in records:
            store.save_record(record)
        return {
            "dataset_id": req.dataset_id,
            "imported": len(records),
            "reference_key": req.reference_key,
            "repo_id": "casecrawler-bundled-fixture",
            "split": "fixture",
            "license": "synthetic-fixture",
        }

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
            lab_values_field=req.lab_values_field,
            vital_values_field=req.vital_values_field,
            medications_field=req.medications_field,
            time_series_field=req.time_series_field,
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
            detail="no Synthea FHIR JSON, FHIR NDJSON, or CSV files found",
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


@router.get("/datasets/{dataset_id}/benchmark-profile")
def get_benchmark_profile(dataset_id: str):
    store = DatasetStore()
    if not store.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="dataset not found")
    records = list(store.iter_records(dataset_id=dataset_id))
    try:
        return benchmark_profile_artifact(profile_records(records))
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err


@router.post("/benchmark-profile")
def benchmark_profile(req: BenchmarkProfileCompareRequest):
    try:
        generated_profile = parse_benchmark_profile_artifact(req.profile)
        reference_profile = parse_benchmark_profile_artifact(req.reference_profile)
        report = DatasetBenchmark(
            min_overall_score=req.min_overall_score,
            min_metric_score=req.min_metric_score,
        ).compare_profiles(generated_profile, reference_profile)
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


@router.post("/datasets/{dataset_id}/reference-fixtures")
def seed_reference_fixtures(
    dataset_id: str,
    dataset_id_prefix: str | None = Query(default=None, min_length=1),
    limit: int | None = Query(default=None, ge=1),
):
    from casecrawler.integrations.reference_fixtures import (
        seed_recommended_reference_fixtures,
    )

    store = DatasetStore()
    if not store.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="dataset not found")
    return seed_recommended_reference_fixtures(
        store,
        dataset_id=dataset_id,
        dataset_id_prefix=dataset_id_prefix,
        limit=limit,
    )


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
    report = build_dataset_quality_report(
        dataset_id,
        records,
        effective_approved=store.effective_approved,
    )
    if not allow_blocked:
        if not report.export_ready:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Dataset is not ready for fine-tuning export. "
                    f"Blockers: {report.issue_counts_by_field}. "
                    "Set allow_blocked=true to export anyway."
                ),
            )
        profile_blocker = export_profile_blocker(report, export_format.value)
        if profile_blocker:
            raise HTTPException(
                status_code=409,
                detail=f"{profile_blocker} Set allow_blocked=true to export anyway.",
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


@router.get("/datasets/{dataset_id}/export-splits")
def export_dataset_splits(
    dataset_id: str,
    export_format: ExportFormat = ExportFormat.SFT_JSONL,
    train_ratio: float = Query(0.8, ge=0.0),
    validation_ratio: float = Query(0.1, ge=0.0),
    test_ratio: float = Query(0.1, ge=0.0),
    seed: str = "casecrawler",
    allow_blocked: bool = False,
    require_multimodal_release: bool = False,
    reference_dataset_id: str | None = Query(default=None, min_length=1),
    auto_benchmark: bool = False,
    min_overall_score: float = Query(0.75, ge=0.0, le=1.0),
    min_metric_score: float = Query(0.5, ge=0.0, le=1.0),
):
    store = DatasetStore()
    if not store.dataset_exists(dataset_id):
        raise HTTPException(status_code=404, detail="dataset not found")
    if export_format == ExportFormat.PARQUET:
        raise HTTPException(
            status_code=422,
            detail="Split package export writes JSONL profiles, not parquet.",
        )
    records = list(store.iter_records(dataset_id=dataset_id))
    report = build_dataset_quality_report(
        dataset_id,
        records,
        effective_approved=store.effective_approved,
    )
    if not allow_blocked:
        if not report.export_ready:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Dataset is not ready for split fine-tuning export. "
                    f"Blockers: {report.issue_counts_by_field}. "
                    "Set allow_blocked=true to export anyway."
                ),
            )
        profile_blocker = export_profile_blocker(report, export_format.value)
        if profile_blocker:
            raise HTTPException(
                status_code=409,
                detail=f"{profile_blocker} Set allow_blocked=true to export anyway.",
            )
    try:
        from casecrawler.validation.benchmark_selection import (
            benchmark_suite_from_report,
            resolve_benchmark_gate,
        )

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
    benchmark_audit = {}
    benchmark_plan = None
    benchmark_suite = None
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
        benchmark_audit["benchmark_report.json"] = benchmark_report.model_dump(mode="json")
        benchmark_suite = benchmark_suite_from_report(
            dataset_id=dataset_id,
            benchmark_report=benchmark_report,
            reference_dataset_id=benchmark_gate.reference_dataset_id,
            reference_key=benchmark_gate.reference_key,
        )
        benchmark_audit["benchmark_suite_report.json"] = benchmark_suite
        benchmark_plan = {
            "recommended_reference_keys": [
                benchmark_gate.reference_key or benchmark_gate.reference_dataset_id
            ],
            "ready": benchmark_report.passed,
            "resolved_reference_dataset_id": benchmark_gate.reference_dataset_id
            if benchmark_report.passed
            else None,
            "missing_reference_keys": []
            if benchmark_report.passed
            else [benchmark_gate.reference_key or benchmark_gate.reference_dataset_id],
            "thresholds": benchmark_report.thresholds,
            "task_export_reference_readiness": {},
        }
        if not benchmark_report.passed and not allow_blocked:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Dataset failed benchmark gate for split fine-tuning export. "
                    f"Overall score: {benchmark_report.overall_score}. "
                    f"Failing metrics: {benchmark_report.failing_metrics}. "
                    "Set allow_blocked=true to export anyway."
                ),
            )
    if benchmark_plan is not None:
        report = build_dataset_quality_report(
            dataset_id,
            records,
            effective_approved=store.effective_approved,
            benchmark_plan=benchmark_plan,
        )
    if require_multimodal_release and not report.multimodal_release_ready:
        raise HTTPException(
            status_code=409,
            detail=(
                "Dataset is not ready for multimodal release package export. "
                f"Missing: {report.multimodal_release_missing}."
            ),
        )
    manifest_snapshot = store.get_manifest(dataset_id)
    try:
        with TemporaryDirectory() as temp_dir:
            manifest = export_jsonl_split_package(
                records,
                temp_dir,
                export_format,
                dataset_id=dataset_id,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
                seed=seed,
                audit_artifacts={
                    "quality_report.json": report.model_dump(mode="json"),
                    "benchmark_profile.json": benchmark_profile_artifact(
                        profile_records(records)
                    ),
                    "dataset_card.md": build_dataset_card(manifest_snapshot, records),
                    "model_card.md": build_model_card(manifest_snapshot, records),
                    **benchmark_audit,
                },
            )
            payload = BytesIO()
            with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(Path(temp_dir).rglob("*")):
                    if path.is_file():
                        archive.write(path, arcname=path.relative_to(temp_dir))
            zip_bytes = payload.getvalue()
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    store.save_export_manifest(
        dataset_id=dataset_id,
        export_format=export_format,
        file_path=(
            f"api://datasets/{dataset_id}/export-splits?"
            f"export_format={export_format.value}"
        ),
        record_count=manifest["example_count"],
        metadata={
            "transport": "api",
            "split_package": True,
            "zip_bytes": len(zip_bytes),
            "seed": manifest["seed"],
            "ratios": manifest["ratios"],
            "audit_artifacts": {
                name: Path(path).name for name, path in manifest["audit_artifacts"].items()
            },
            "image_artifacts": manifest["image_artifacts"],
            "image_artifact_count": len(manifest["image_artifacts"]),
            "multimodal_release_ready": report.multimodal_release_ready,
            "multimodal_release_missing": report.multimodal_release_missing,
            "core_artifact_coverage": report.core_artifact_coverage,
            **benchmark_metadata,
            **(
                {
                    "benchmark_suite_passed": benchmark_suite["passed"],
                    "benchmark_suite_reference_count": benchmark_suite[
                        "reference_count"
                    ],
                }
                if benchmark_suite
                else {}
            ),
            "splits": {
                name: {
                    "record_count": data["record_count"],
                    "example_count": data["example_count"],
                }
                for name, data in manifest["splits"].items()
            },
        },
    )
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{dataset_id}-{export_format.value}-splits.zip"'
            )
        },
    )


def _image_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"
