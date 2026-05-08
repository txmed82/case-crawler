import json
import zipfile
from io import BytesIO

from fastapi.testclient import TestClient

from casecrawler.api.routes import datasets as datasets_routes
from casecrawler.api.app import app
from casecrawler.models.config import AppConfig, SyntheticConfig
from casecrawler.models.synthetic import (
    ComplexityProfile,
    ImagingAsset,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    ValidationReport,
)
from casecrawler.storage.dataset_store import DatasetStore


def test_generate_dataset_api_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    response = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["generated"] == 1
    assert body["approved"] == 1
    assert body["total_records"] == 1


def test_generate_dataset_api_rejects_unbounded_counts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = AppConfig(synthetic=SyntheticConfig(max_api_generation_count=1))
    monkeypatch.setattr(datasets_routes, "get_config", lambda: config)
    client = TestClient(app)

    response = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 2})

    assert response.status_code == 422
    assert "less than or equal to 1" in response.json()["detail"]


def test_generate_dataset_api_reports_unknown_recipe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/datasets/generate",
        json={"topic": "sepsis", "recipe": "missing"},
    )

    assert response.status_code == 422
    assert "Unknown generation recipe" in response.json()["detail"]


def test_dataset_api_lists_and_exports_records(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    dataset_id = generated.json()["dataset_id"]

    listed = client.get("/api/datasets")
    exported = client.get(
        f"/api/datasets/{dataset_id}/export",
        params={"export_format": "sft_jsonl"},
    )

    assert listed.status_code == 200
    assert listed.json()["datasets"][0]["dataset_id"] == dataset_id
    assert exported.status_code == 200
    first_line = exported.text.strip().splitlines()[0]
    assert json.loads(first_line)["dataset_id"] == dataset_id


def test_dataset_api_exports_note_fact_sft_examples(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    dataset_id = generated.json()["dataset_id"]

    exported = client.get(
        f"/api/datasets/{dataset_id}/export",
        params={"export_format": "note_fact_sft_jsonl"},
    )

    assert exported.status_code == 200
    lines = [json.loads(line) for line in exported.text.strip().splitlines()]
    assert len(lines) > 1
    assert {line["task"] for line in lines} == {"extract_clinical_facts_from_note"}
    assert all(line["dataset_id"] == dataset_id for line in lines)
    assert all(line["document_id"] for line in lines)


def test_dataset_api_exports_split_fine_tuning_package(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 3})
    dataset_id = generated.json()["dataset_id"]

    exported = client.get(
        f"/api/datasets/{dataset_id}/export-splits",
        params={
            "export_format": "sft_jsonl",
            "train_ratio": 0.34,
            "validation_ratio": 0.33,
            "test_ratio": 0.33,
            "seed": "unit-test",
        },
    )
    listed = client.get(f"/api/datasets/{dataset_id}/exports")

    assert exported.status_code == 200
    assert exported.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(BytesIO(exported.content)) as archive:
        assert sorted(archive.namelist()) == [
            "benchmark_profile.json",
            "dataset_card.md",
            "manifest.json",
            "model_card.md",
            "quality_report.json",
            "test.jsonl",
            "train.jsonl",
            "validation.jsonl",
        ]
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["record_count"] == 3
        assert manifest["example_count"] == 3
        assert manifest["splits"]["train"]["record_count"] == 1
        assert manifest["splits"]["validation"]["record_count"] == 1
        assert manifest["splits"]["test"]["record_count"] == 1
        assert json.loads(archive.read("quality_report.json"))["export_ready"] is True
        benchmark_profile = json.loads(archive.read("benchmark_profile.json"))
        assert benchmark_profile["artifact_type"] == "casecrawler_benchmark_profile"
        assert benchmark_profile["profile"]["dataset_id"] == dataset_id
        assert "Dataset Card" in archive.read("dataset_card.md").decode()
    assert listed.json()["exports"][0]["metadata"]["split_package"] is True
    assert listed.json()["exports"][0]["metadata"]["transport"] == "api"
    assert listed.json()["exports"][0]["metadata"]["multimodal_release_ready"] is False
    assert "benchmark_reference" in listed.json()["exports"][0]["metadata"][
        "multimodal_release_missing"
    ]
    assert listed.json()["exports"][0]["metadata"]["core_artifact_coverage"]["records"] is True
    assert "benchmark_profile.json" in listed.json()["exports"][0]["metadata"][
        "audit_artifacts"
    ]


def test_dataset_api_blocks_profile_specific_export_when_artifacts_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post(
        "/api/datasets/generate",
        json={
            "topic": "sepsis",
            "count": 1,
            "modalities": ["clinical_text"],
        },
    )
    dataset_id = generated.json()["dataset_id"]

    blocked = client.get(
        f"/api/datasets/{dataset_id}/export",
        params={"export_format": "medication_reconciliation_jsonl"},
    )
    allowed = client.get(
        f"/api/datasets/{dataset_id}/export",
        params={
            "export_format": "medication_reconciliation_jsonl",
            "allow_blocked": "true",
        },
    )

    assert blocked.status_code == 409
    assert "Export profile medication_reconciliation_jsonl is not ready" in blocked.json()["detail"]
    assert "medications" in blocked.json()["detail"]
    assert allowed.status_code == 200


def test_dataset_api_blocks_profile_specific_split_export_when_artifacts_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post(
        "/api/datasets/generate",
        json={
            "topic": "sepsis",
            "count": 1,
            "modalities": ["clinical_text"],
        },
    )
    dataset_id = generated.json()["dataset_id"]

    blocked = client.get(
        f"/api/datasets/{dataset_id}/export-splits",
        params={"export_format": "clinical_observation_jsonl"},
    )

    assert blocked.status_code == 409
    assert "Export profile clinical_observation_jsonl is not ready" in blocked.json()["detail"]
    assert "labs_or_vitals" in blocked.json()["detail"]


def test_dataset_api_can_require_multimodal_release_for_split_export(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    dataset_id = generated.json()["dataset_id"]

    blocked = client.get(
        f"/api/datasets/{dataset_id}/export-splits",
        params={"require_multimodal_release": "true"},
    )

    assert blocked.status_code == 409
    assert "not ready for multimodal release package export" in blocked.json()["detail"]
    assert "benchmark_reference" in blocked.json()["detail"]


def test_dataset_api_generates_release_package_with_fixture_references(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/datasets/release-package",
        json={
            "topic": "sepsis",
            "count": 1,
            "seed": "unit-test",
        },
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    dataset_id = response.headers["x-casecrawler-dataset-id"]
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        names = sorted(archive.namelist())
        image_files = [name for name in names if name.startswith("images/")]
        time_series_files = [
            name for name in names if name.startswith("time_series/")
        ]
        assert len(image_files) == 1
        assert time_series_files
        assert names == [
            "benchmark_profile.json",
            "benchmark_report.json",
            "benchmark_suite_report.json",
            "dataset_card.md",
            image_files[0],
            "manifest.json",
            "model_card.md",
            "quality_report.json",
            "release_package_summary.json",
            "test.jsonl",
            *time_series_files,
            "train.jsonl",
            "validation.jsonl",
        ]
        manifest = json.loads(archive.read("manifest.json"))
        quality = json.loads(archive.read("quality_report.json"))
        benchmark = json.loads(archive.read("benchmark_report.json"))
        benchmark_suite = json.loads(archive.read("benchmark_suite_report.json"))
        summary = json.loads(archive.read("release_package_summary.json"))
    exports = DatasetStore().list_export_manifests(dataset_id=dataset_id)

    assert manifest["dataset_id"] == dataset_id
    assert manifest["export_format"] == "multimodal_jsonl"
    assert manifest["record_count"] == 1
    assert next(iter(manifest["image_artifacts"].values()))["package_path"] == image_files[0]
    assert image_files[0] in manifest["files"]
    assert {
        artifact["package_path"]
        for artifact in manifest["time_series_artifacts"].values()
    } == set(time_series_files)
    assert set(time_series_files).issubset(manifest["files"])
    assert quality["multimodal_release_ready"] is True
    assert benchmark["passed"] is True
    assert benchmark_suite["passed"] is True
    assert benchmark_suite["reference_count"] >= 1
    assert summary["dataset_id"] == dataset_id
    assert summary["task_coverage"] == manifest["task_coverage"]
    assert summary["quality_report"]["multimodal_release_ready"] is True
    assert summary["quality_report"]["imaging_report_label_evidence_rate"] is not None
    assert summary["quality_report"]["mean_imaging_report_chars"] > 0
    assert summary["quality_report"]["time_series_channel_counts"]
    assert summary["quality_report"]["mean_time_series_points"] > 0
    assert summary["quality_report"]["mean_time_series_duration_hours"] >= 0
    assert summary["benchmark"]["passed"] is True
    assert summary["benchmark_suite"]["passed"] is True
    assert summary["benchmark_suite"]["reference_count"] == benchmark_suite["reference_count"]
    assert summary["seeded_references"]["imported"]
    assert exports[0].metadata["release_package"] is True
    assert exports[0].metadata["multimodal_release_ready"] is True
    assert exports[0].metadata["image_artifact_count"] == 1
    assert next(iter(exports[0].metadata["image_artifacts"].values()))[
        "package_path"
    ] == image_files[0]
    assert exports[0].metadata["benchmark_passed"] is True
    assert exports[0].metadata["benchmark_suite_passed"] is True


def test_dataset_api_split_export_can_require_benchmark_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 3})
    reference = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 3})
    dataset_id = generated.json()["dataset_id"]
    reference_dataset_id = reference.json()["dataset_id"]

    exported = client.get(
        f"/api/datasets/{dataset_id}/export-splits",
        params={
            "export_format": "sft_jsonl",
            "reference_dataset_id": reference_dataset_id,
            "min_overall_score": 0,
            "min_metric_score": 0,
        },
    )
    listed = client.get(f"/api/datasets/{dataset_id}/exports")

    assert exported.status_code == 200
    with zipfile.ZipFile(BytesIO(exported.content)) as archive:
        assert "benchmark_report.json" in archive.namelist()
        assert "benchmark_suite_report.json" in archive.namelist()
        benchmark_report = json.loads(archive.read("benchmark_report.json"))
        benchmark_suite = json.loads(archive.read("benchmark_suite_report.json"))
        assert benchmark_report["reference_dataset_id"] == reference_dataset_id
        assert benchmark_report["passed"] is True
        assert benchmark_suite["reference_count"] == 1
        assert benchmark_suite["passed"] is True
        assert benchmark_suite["results"][0]["reference_dataset_id"] == reference_dataset_id
    metadata = listed.json()["exports"][0]["metadata"]
    assert metadata["benchmark_reference_dataset_id"] == reference_dataset_id
    assert metadata["benchmark_passed"] is True
    assert metadata["benchmark_suite_passed"] is True
    assert metadata["benchmark_suite_reference_count"] == 1


def test_dataset_api_export_can_require_benchmark_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    reference = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    dataset_id = generated.json()["dataset_id"]
    reference_dataset_id = reference.json()["dataset_id"]

    exported = client.get(
        f"/api/datasets/{dataset_id}/export",
        params={
            "export_format": "sft_jsonl",
            "reference_dataset_id": reference_dataset_id,
            "min_overall_score": 0.0,
            "min_metric_score": 0.0,
        },
    )

    assert exported.status_code == 200
    first_line = exported.text.strip().splitlines()[0]
    assert json.loads(first_line)["dataset_id"] == dataset_id


def test_dataset_api_export_can_auto_select_recipe_benchmark_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post(
        "/api/datasets/generate",
        json={"topic": "sepsis", "count": 1, "recipe": "icu_timeseries_notes"},
    )
    reference = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    dataset_id = generated.json()["dataset_id"]
    reference_dataset_id = reference.json()["dataset_id"]
    store = DatasetStore()
    for record in store.list_records(dataset_id=reference_dataset_id):
        store.save_record(
            record.model_copy(
                update={
                    "metadata": {
                        **record.metadata,
                        "reference_key": "synthclinicalnotes",
                        "reference_dataset": "IntelLabs/SynthClinicalNotes",
                    }
                }
            )
        )

    exported = client.get(
        f"/api/datasets/{dataset_id}/export",
        params={
            "export_format": "sft_jsonl",
            "auto_benchmark": "true",
            "allow_blocked": "true",
        },
    )
    listed = client.get(f"/api/datasets/{dataset_id}/exports")

    assert exported.status_code == 200
    assert listed.json()["exports"][0]["metadata"]["benchmark_reference_dataset_id"] == (
        reference_dataset_id
    )
    assert listed.json()["exports"][0]["metadata"]["benchmark_reference_key"] == (
        "synthclinicalnotes"
    )
    assert listed.json()["exports"][0]["metadata"]["benchmark_auto_selected"] is True
    assert listed.json()["exports"][0]["metadata"]["benchmark_thresholds"] == {
        "min_overall_score": 0.75,
        "min_metric_score": 0.5,
    }


def test_dataset_api_reports_recipe_benchmark_plan_readiness(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post(
        "/api/datasets/generate",
        json={"topic": "sepsis", "count": 1, "recipe": "icu_timeseries_notes"},
    )
    reference = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    dataset_id = generated.json()["dataset_id"]
    reference_dataset_id = reference.json()["dataset_id"]
    store = DatasetStore()
    for record in store.list_records(dataset_id=reference_dataset_id):
        store.save_record(
            record.model_copy(
                update={
                    "metadata": {
                        **record.metadata,
                        "reference_key": "synthclinicalnotes",
                        "reference_dataset": "IntelLabs/SynthClinicalNotes",
                    }
                }
            )
        )

    response = client.get(f"/api/datasets/{dataset_id}/benchmark-plan")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == dataset_id
    assert body["primary_recipe"] == "icu_timeseries_notes"
    assert body["recommended_reference_keys"] == [
        "synthea_fhir",
        "synthclinicalnotes",
        "augmented_clinical_notes",
        "medsynth_dialogue_note",
        "clinical_notes_to_fhir",
        "technetium_i",
    ]
    assert body["resolved_reference_dataset_id"] == reference_dataset_id
    assert body["resolved_reference_key"] == "synthclinicalnotes"
    assert body["ready"] is True
    assert body["thresholds"] == {"min_overall_score": 0.75, "min_metric_score": 0.5}
    note_fact_readiness = body["task_export_reference_readiness"][
        "note_fact_sft_jsonl"
    ]
    assert note_fact_readiness["available_reference_keys"] == ["synthclinicalnotes"]
    assert note_fact_readiness["missing_reference_keys"] == [
        "augmented_clinical_notes",
        "clinical_notes_to_fhir",
        "technetium_i",
    ]
    assert note_fact_readiness["ready"] is False


def test_dataset_api_seeds_recipe_reference_fixtures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post(
        "/api/datasets/generate",
        json={"topic": "sepsis", "count": 1, "recipe": "icu_timeseries_notes"},
    )
    dataset_id = generated.json()["dataset_id"]

    seeded = client.post(
        f"/api/datasets/{dataset_id}/reference-fixtures",
        params={"dataset_id_prefix": "fixture-ref"},
    )
    plan = client.get(f"/api/datasets/{dataset_id}/benchmark-plan")

    assert seeded.status_code == 200
    body = seeded.json()
    assert body["unavailable"] == []
    assert {item["reference_key"] for item in body["imported"]} == {
        "synthea_fhir",
        "synthclinicalnotes",
        "augmented_clinical_notes",
        "medsynth_dialogue_note",
        "clinical_notes_to_fhir",
        "technetium_i",
    }
    assert plan.json()["ready"] is True
    assert plan.json()["missing_reference_keys"] == []


def test_dataset_api_quality_report_includes_recipe_benchmark_readiness(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post(
        "/api/datasets/generate",
        json={"topic": "sepsis", "count": 1, "recipe": "icu_timeseries_notes"},
    )
    dataset_id = generated.json()["dataset_id"]

    response = client.get(f"/api/datasets/{dataset_id}/quality")

    assert response.status_code == 200
    body = response.json()
    assert body["benchmark_ready"] is False
    assert body["recommended_reference_keys"] == [
        "synthea_fhir",
        "synthclinicalnotes",
        "augmented_clinical_notes",
        "medsynth_dialogue_note",
        "clinical_notes_to_fhir",
        "technetium_i",
    ]
    assert "clinical_notes_to_fhir" in body["missing_reference_keys"]
    assert body["benchmark_thresholds"] == {
        "min_overall_score": 0.75,
        "min_metric_score": 0.5,
    }
    assert body["task_export_reference_readiness"]["time_series_jsonl"][
        "recommended_reference_keys"
    ] == ["synthea_fhir"]


def test_dataset_api_runs_recipe_benchmark_suite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post(
        "/api/datasets/generate",
        json={"topic": "sepsis", "count": 1, "recipe": "icu_timeseries_notes"},
    )
    first_reference = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    second_reference = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    dataset_id = generated.json()["dataset_id"]
    first_reference_id = first_reference.json()["dataset_id"]
    second_reference_id = second_reference.json()["dataset_id"]
    store = DatasetStore()
    for dataset_id_to_mark, reference_key in [
        (first_reference_id, "synthclinicalnotes"),
        (second_reference_id, "clinical_notes_to_fhir"),
    ]:
        for record in store.list_records(dataset_id=dataset_id_to_mark):
            store.save_record(
                record.model_copy(
                    update={
                        "metadata": {
                            **record.metadata,
                            "reference_key": reference_key,
                            "reference_dataset": reference_key,
                        }
                    }
                )
            )

    response = client.get(f"/api/datasets/{dataset_id}/benchmark-suite")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == dataset_id
    assert body["reference_count"] == 2
    assert body["thresholds"] == {"min_overall_score": 0.75, "min_metric_score": 0.5}
    assert {
        result["reference_key"] for result in body["results"]
    } == {"synthclinicalnotes", "clinical_notes_to_fhir"}
    assert {
        result["reference_dataset_id"] for result in body["results"]
    } == {first_reference_id, second_reference_id}
    note_fact_results = body["task_export_results"]["note_fact_sft_jsonl"]
    assert note_fact_results["reference_count"] == 2
    assert note_fact_results["missing_reference_keys"] == [
        "augmented_clinical_notes",
        "technetium_i",
    ]


def test_dataset_api_lists_export_manifests(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    reference = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    dataset_id = generated.json()["dataset_id"]
    reference_dataset_id = reference.json()["dataset_id"]

    exported = client.get(
        f"/api/datasets/{dataset_id}/export",
        params={
            "export_format": "sft_jsonl",
            "reference_dataset_id": reference_dataset_id,
            "min_overall_score": 0.0,
            "min_metric_score": 0.0,
        },
    )
    listed = client.get(f"/api/datasets/{dataset_id}/exports")

    assert exported.status_code == 200
    assert listed.status_code == 200
    body = listed.json()
    assert body["dataset_id"] == dataset_id
    assert body["exports"][0]["export_format"] == "sft_jsonl"
    assert body["exports"][0]["metadata"]["transport"] == "api"
    assert body["exports"][0]["metadata"]["benchmark_passed"] is True
    assert (
        body["exports"][0]["metadata"]["benchmark_reference_dataset_id"]
        == reference_dataset_id
    )


def test_dataset_api_export_blocks_failed_benchmark_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    reference = client.post(
        "/api/datasets/generate",
        json={"topic": "heart failure", "count": 1},
    )
    dataset_id = generated.json()["dataset_id"]
    reference_dataset_id = reference.json()["dataset_id"]

    blocked = client.get(
        f"/api/datasets/{dataset_id}/export",
        params={
            "export_format": "sft_jsonl",
            "reference_dataset_id": reference_dataset_id,
            "min_overall_score": 1.0,
            "min_metric_score": 1.0,
        },
    )
    allowed = client.get(
        f"/api/datasets/{dataset_id}/export",
        params={
            "export_format": "sft_jsonl",
            "reference_dataset_id": reference_dataset_id,
            "min_overall_score": 1.0,
            "min_metric_score": 1.0,
            "allow_blocked": "true",
        },
    )

    assert blocked.status_code == 409
    assert "failed benchmark gate" in blocked.json()["detail"]
    assert allowed.status_code == 200


def test_dataset_api_exports_parquet_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    dataset_id = generated.json()["dataset_id"]

    def fake_export_parquet_bytes(records):
        records = list(records)
        assert len(records) == 1
        assert records[0].dataset_id == dataset_id
        return b"PAR1synthetic-parquet", len(records)

    monkeypatch.setattr(datasets_routes, "export_parquet_bytes", fake_export_parquet_bytes)

    exported = client.get(
        f"/api/datasets/{dataset_id}/export",
        params={"export_format": "parquet"},
    )

    assert exported.status_code == 200
    assert exported.content == b"PAR1synthetic-parquet"
    assert exported.headers["content-type"] == "application/vnd.apache.parquet"
    assert f'filename="{dataset_id}.parquet"' in exported.headers["content-disposition"]


def test_dataset_api_serves_dataset_image_asset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-image")
    store = DatasetStore()
    store.save_record(
        SyntheticRecord(
            record_id="rec-image",
            dataset_id="ds-image",
            topic="pneumonia",
            complexity=ComplexityProfile.MODERATE,
            modalities=[Modality.IMAGING],
            patient=SyntheticPatient(patient_id="pat-1", age=64, sex="female"),
            encounters=[],
            imaging=[
                ImagingAsset(
                    image_id="img-1",
                    modality="XR",
                    body_region="chest",
                    prompt="portable chest x-ray pneumonia",
                    file_path=str(image_path),
                    report_text="Right lower lobe opacity.",
                    generation_backend="unit-test",
                )
            ],
            provenance=Provenance(generator="unit-test", created_at="2026-01-01T00:00:00"),
        )
    )
    client = TestClient(app)

    served = client.get("/api/datasets/ds-image/images/img-1")
    missing = client.get("/api/datasets/ds-image/images/img-missing")

    assert served.status_code == 200
    assert served.headers["content-type"] == "image/png"
    assert served.content == b"\x89PNG\r\n\x1a\nsynthetic-image"
    assert missing.status_code == 404


def test_dataset_api_export_blocks_unready_dataset_without_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = DatasetStore()
    store.save_record(
        SyntheticRecord(
            record_id="rec-blocked",
            dataset_id="ds-blocked",
            topic="sepsis",
            complexity=ComplexityProfile.MODERATE,
            modalities=[Modality.CLINICAL_TEXT],
            patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
            encounters=[],
            provenance=Provenance(generator="unit-test", created_at="2026-01-01T00:00:00"),
            validation=ValidationReport(
                schema_score=1.0,
                clinical_consistency_score=1.0,
                privacy_score=1.0,
                utility_score=1.0,
                approved=True,
            ),
        )
    )
    client = TestClient(app)

    blocked = client.get(
        "/api/datasets/ds-blocked/export",
        params={"export_format": "sft_jsonl"},
    )
    allowed = client.get(
        "/api/datasets/ds-blocked/export",
        params={"export_format": "sft_jsonl", "allow_blocked": "true"},
    )

    assert blocked.status_code == 409
    assert "not ready for fine-tuning export" in blocked.json()["detail"]
    assert "clinical_text.missing_artifacts" in blocked.json()["detail"]
    assert allowed.status_code == 200
    assert json.loads(allowed.text.strip())["record_id"] == "rec-blocked"


def test_dataset_api_blocks_export_until_required_human_review(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post(
        "/api/datasets/generate",
        json={"topic": "sepsis", "count": 1, "require_human_review": True},
    )
    dataset_id = generated.json()["dataset_id"]
    record_id = generated.json()["records"][0]["record_id"]

    blocked = client.get(
        f"/api/datasets/{dataset_id}/export",
        params={"export_format": "sft_jsonl"},
    )
    reviewed = client.post(
        f"/api/records/{record_id}/review",
        json={"status": "approved", "reviewer": "clinical-reviewer"},
    )
    allowed = client.get(
        f"/api/datasets/{dataset_id}/export",
        params={"export_format": "sft_jsonl"},
    )

    assert blocked.status_code == 409
    assert "human_review.missing" in blocked.json()["detail"]
    assert reviewed.status_code == 200
    assert allowed.status_code == 200
    assert json.loads(allowed.text.strip())["record_id"] == record_id


def test_dataset_api_lists_and_saves_human_reviews(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = DatasetStore()
    store.save_record(
        SyntheticRecord(
            record_id="rec-review",
            dataset_id="ds-review",
            topic="sepsis",
            complexity=ComplexityProfile.MODERATE,
            modalities=[Modality.CLINICAL_TEXT],
            patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
            encounters=[],
            provenance=Provenance(
                generator="unit-test",
                created_at="2026-05-06T10:00:00",
            ),
        )
    )
    client = TestClient(app)

    queue = client.get("/api/datasets/ds-review/reviews")
    reviewed = client.post(
        "/api/records/rec-review/review",
        json={
            "status": "approved",
            "reviewer": "clinical-reviewer",
            "notes": ["Approved for fine-tuning export."],
        },
    )
    queue_after = client.get("/api/datasets/ds-review/reviews")

    assert queue.status_code == 200
    assert queue.json()["records"][0]["record_id"] == "rec-review"
    assert reviewed.status_code == 200
    assert reviewed.json()["effective_approved"] is True
    assert reviewed.json()["human_review"]["reviewer"] == "clinical-reviewer"
    assert queue_after.json()["records"] == []


def test_dataset_api_serves_quality_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    dataset_id = generated.json()["dataset_id"]

    response = client.get(f"/api/datasets/{dataset_id}/quality")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == dataset_id
    assert body["record_count"] == 1
    assert body["approved_count"] == 1
    assert body["export_ready"] is True
    assert "clinical_text" in body["modality_counts"]
    assert "lab_values" in body["extracted_fact_key_counts"]
    assert body["time_series_backend_counts"] == {}
    assert body["imaging_backend_counts"] == {}
    assert body["imaging_model_policy_counts"] == {}
    assert body["diagnosis_code_system_counts"]["synthetic"] >= 1
    assert body["diagnosis_code_counts"]["synthetic:sepsis"] >= 1
    assert body["phi_entity_counts"] == {}


def test_dataset_api_quality_report_handles_missing_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    response = client.get("/api/datasets/ds-missing/quality")

    assert response.status_code == 404
    assert response.json()["detail"] == "dataset not found"


def test_dataset_api_serves_dataset_and_model_cards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    dataset_id = generated.json()["dataset_id"]

    dataset_card = client.get(f"/api/datasets/{dataset_id}/card")
    model_card = client.get(
        f"/api/datasets/{dataset_id}/card",
        params={"kind": "model"},
    )

    assert dataset_card.status_code == 200
    assert "# Dataset Card:" in dataset_card.text
    assert model_card.status_code == 200
    assert "# Model Card:" in model_card.text


def test_dataset_api_benchmarks_against_reference_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    reference = client.post(
        "/api/datasets/generate",
        json={"topic": "heart failure", "count": 1},
    )
    dataset_id = generated.json()["dataset_id"]
    reference_dataset_id = reference.json()["dataset_id"]

    response = client.get(
        f"/api/datasets/{dataset_id}/benchmark",
        params={"reference_dataset_id": reference_dataset_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["generated_dataset_id"] == dataset_id
    assert body["reference_dataset_id"] == reference_dataset_id
    assert body["overall_score"] >= 0
    assert body["thresholds"] == {"min_overall_score": 0.75, "min_metric_score": 0.5}
    assert any(metric["name"] == "modality_overlap" for metric in body["metrics"])


def test_dataset_api_exports_and_compares_benchmark_profiles(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 2})
    reference = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 2})
    dataset_id = generated.json()["dataset_id"]
    reference_dataset_id = reference.json()["dataset_id"]

    generated_profile = client.get(f"/api/datasets/{dataset_id}/benchmark-profile")
    reference_profile = client.get(
        f"/api/datasets/{reference_dataset_id}/benchmark-profile"
    )
    compared = client.post(
        "/api/benchmark-profile",
        json={
            "profile": generated_profile.json(),
            "reference_profile": reference_profile.json(),
            "min_overall_score": 0.2,
            "min_metric_score": 0,
        },
    )

    assert generated_profile.status_code == 200
    assert reference_profile.status_code == 200
    assert generated_profile.json()["artifact_type"] == "casecrawler_benchmark_profile"
    assert compared.status_code == 200
    body = compared.json()
    assert body["generated_dataset_id"] == dataset_id
    assert body["reference_dataset_id"] == reference_dataset_id
    assert body["thresholds"] == {"min_overall_score": 0.2, "min_metric_score": 0.0}


def test_dataset_api_rejects_invalid_benchmark_profile_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/benchmark-profile",
        json={
            "profile": {"artifact_type": "unknown"},
            "reference_profile": {"artifact_type": "unknown"},
        },
    )

    assert response.status_code == 422
    assert "unsupported artifact_type" in response.json()["detail"]


def test_dataset_api_uses_custom_benchmark_thresholds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    reference = client.post(
        "/api/datasets/generate",
        json={"topic": "sepsis", "count": 1},
    )
    dataset_id = generated.json()["dataset_id"]
    reference_dataset_id = reference.json()["dataset_id"]

    response = client.get(
        f"/api/datasets/{dataset_id}/benchmark",
        params={
            "reference_dataset_id": reference_dataset_id,
            "min_overall_score": 1.0,
            "min_metric_score": 1.0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["thresholds"] == {"min_overall_score": 1.0, "min_metric_score": 1.0}
    assert "passed" in body
    assert "failing_metrics" in body


def test_dataset_api_rejects_invalid_benchmark_thresholds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    response = client.get(
        "/api/datasets/ds-any/benchmark",
        params={
            "reference_dataset_id": "ds-ref",
            "min_overall_score": 1.1,
        },
    )

    assert response.status_code == 422


def test_dataset_api_benchmark_reports_missing_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    generated = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    dataset_id = generated.json()["dataset_id"]

    response = client.get(
        f"/api/datasets/{dataset_id}/benchmark",
        params={"reference_dataset_id": "ds-missing"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "reference dataset not found"


def test_dataset_api_lists_hf_reference_catalog(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    response = client.get("/api/datasets/reference-catalog")

    assert response.status_code == 200
    datasets = response.json()["datasets"]
    assert any(item["key"] == "asclepius" for item in datasets)
    assert any(item["key"] == "rexgradient_160k" for item in datasets)
    assert any(item["key"] == "synthea_fhir" for item in datasets)
    assert any(item["key"] == "technetium_i" for item in datasets)
    asclepius = next(item for item in datasets if item["key"] == "asclepius")
    rexgradient = next(item for item in datasets if item["key"] == "rexgradient_160k")
    synthea = next(item for item in datasets if item["key"] == "synthea_fhir")
    technetium = next(item for item in datasets if item["key"] == "technetium_i")
    assert asclepius["repo_id"] == "starmpcc/Asclepius-Synthetic-Clinical-Notes"
    assert asclepius["license"]
    assert rexgradient["repo_id"] == "rajpurkarlab/ReXGradient-160K"
    assert rexgradient["license"] == "rexgradient-non-commercial-gated"
    assert synthea["repo_id"] is None
    assert synthea["source"] == "synthea"
    assert synthea["use_policy"] == "local_synthea_import"
    assert technetium["repo_id"] == "temlm-foundation/Technetium-I"
    assert technetium["license"] == "eupl-1.2"
    assert rexgradient["gated"] is True
    assert rexgradient["use_policy"] == "non_commercial_research_only"
    assert rexgradient["image_modality"] == "XR"
    assert rexgradient["fixture_available"] is False
    assert technetium["fixture_available"] is True


def test_dataset_api_lists_generation_capabilities(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    response = client.get("/api/datasets/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert "clinical_text" in body["modalities"]
    assert "sft_jsonl" in body["export_formats"]
    assert "full_multimodal_acute_care" in {
        recipe["name"] for recipe in body["generation_recipes"]
    }
    radiology_recipe = next(
        recipe
        for recipe in body["generation_recipes"]
        if recipe["name"] == "radiology_cxr_report"
    )
    assert "synthchex_75k" in radiology_recipe["recommended_reference_keys"]
    assert radiology_recipe["benchmark_thresholds"]["min_overall_score"] == 0.7
    assert "topic_mix" in body["cohort_constraints"]
    assert "topic_mix_weights" in body["cohort_constraints"]
    release_requirements = {
        requirement["key"]: requirement["description"]
        for requirement in body["release_coverage_requirements"]
    }
    assert "lab_reports" in release_requirements
    assert "vital_signs_flowsheets" in release_requirements
    assert "medication_administration_records" in release_requirements
    assert "discharge_summaries" in release_requirements
    assert "radiology_images" in release_requirements
    assert "benchmark reference" in release_requirements["benchmark_reference"].lower()
    references = {dataset["key"]: dataset for dataset in body["reference_datasets"]}
    assert references["synthchex_75k"]["source"] == "huggingface"
    assert references["synthchex_75k"]["image_modality"] == "XR"
    assert references["radiology_report_consistency"]["fixture_available"] is True
    assert references["synthea_fhir"]["source"] == "synthea"
    assert "cxr_pneumonia_dreambooth" in {
        profile["name"] for profile in body["imaging_model_profiles"]
    }
    medisyn = next(
        profile
        for profile in body["imaging_model_profiles"]
        if profile["name"] == "medisyn"
    )
    assert medisyn["model_id"] == "hiesingerlab/MediSyn"
    assert medisyn["license"] == "cc-by-nc-nd-4.0"
    assert medisyn["gated"] is False
    assert (
        medisyn["use_policy"]
        == "non_commercial_no_derivatives_review_before_release"
    )
    roentgen = next(
        profile
        for profile in body["imaging_model_profiles"]
        if profile["name"] == "roentgen_v2_gated"
    )
    assert roentgen["gated"] is True
    assert roentgen["use_policy"] == "credentialed_mimic_cxr_terms_required"
    time_series_profiles = {
        profile["name"]: profile for profile in body["time_series_model_profiles"]
    }
    assert time_series_profiles["timediff"]["model_id"] == "MuhangTian/TimeDiff"
    assert time_series_profiles["timediff"]["license"] == "mit"
    assert time_series_profiles["timediff"]["gated"] is False
    assert (
        time_series_profiles["timediff"]["use_policy"]
        == "wrap_external_sampler_validate_outputs"
    )
    assert "sepsis" in {profile["key"] for profile in body["clinical_profiles"]}
    sepsis = next(profile for profile in body["clinical_profiles"] if profile["key"] == "sepsis")
    assert "Lactate" in sepsis["lab_names"]
    assert "Ceftriaxone" in sepsis["medication_names"]
    validators = {validator["key"]: validator for validator in body["validators"]}
    assert validators["biomedclip"]["model_id"] == (
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    assert validators["biomedclip"]["license"] == "mit"
    assert validators["medgemma"]["model_id"] == "google/medgemma-4b-it"
    assert validators["medgemma"]["gated"] is True
    assert "accepted model terms" in validators["medgemma"]["requires"]


def test_dataset_api_imports_hf_reference_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    def fake_load_reference_dataset(key, *, split=None, streaming=True):
        assert key == "asclepius"
        assert split == "validation"
        assert streaming is True
        return [
            {
                "patient_id": "ref-1",
                "note": "Progress Note: 60-year-old male with sepsis.",
                "question": "Summarize.",
                "answer": "Sepsis.",
                "task": "Summarization",
            }
        ]

    monkeypatch.setattr(
        "casecrawler.integrations.huggingface.load_reference_dataset",
        fake_load_reference_dataset,
    )

    response = client.post(
        "/api/datasets/reference-import",
        json={
            "reference_key": "asclepius",
            "dataset_id": "ds-hf-reference",
            "split": "validation",
            "limit": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == "ds-hf-reference"
    assert body["imported"] == 1
    assert body["reference_key"] == "asclepius"
    assert DatasetStore().dataset_exists("ds-hf-reference")
    manifest = DatasetStore().get_manifest("ds-hf-reference")
    assert manifest.metadata["primary_reference_key"] == "asclepius"


def test_dataset_api_imports_custom_hf_reference_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    def fake_load_huggingface_dataset(repo_id, *, split, streaming=True):
        assert repo_id == "org/custom-synthetic-notes"
        assert split == "eval"
        assert streaming is True
        return [
            {
                "subject_id": "ref-1",
                "clinical_note": "Progress Note: 57-year-old female with COPD.",
                "prompt": "Extract diagnosis.",
                "completion": "COPD.",
                "task_name": "extraction",
                "labs": [
                    {
                        "name": "PaCO2",
                        "value": 51,
                        "unit": "mmHg",
                        "flag": "H",
                        "effective_time": "2026-01-01T00:00:00",
                    }
                ],
                "vitals": [
                    {
                        "name": "SpO2",
                        "value": 91,
                        "unit": "%",
                        "effective_time": "2026-01-01T00:05:00",
                    }
                ],
                "medications": [{"name": "Albuterol", "route": "inhaled"}],
                "signals": [
                    {
                        "name": "respiratory_rate",
                        "unit": "/min",
                        "points": [
                            {
                                "timestamp": "2026-01-01T00:05:00",
                                "values": {"value": 24},
                            }
                        ],
                    }
                ],
            }
        ]

    monkeypatch.setattr(
        "casecrawler.integrations.huggingface.load_huggingface_dataset",
        fake_load_huggingface_dataset,
    )

    response = client.post(
        "/api/datasets/reference-import",
        json={
            "repo_id": "org/custom-synthetic-notes",
            "dataset_id": "ds-custom-reference",
            "split": "eval",
            "license": "cc-by-4.0",
            "note_field": "clinical_note",
            "question_field": "prompt",
            "answer_field": "completion",
            "task_field": "task_name",
            "patient_id_field": "subject_id",
            "lab_values_field": "labs",
            "vital_values_field": "vitals",
            "medications_field": "medications",
            "time_series_field": "signals",
            "limit": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == "ds-custom-reference"
    assert body["reference_key"] == "org/custom-synthetic-notes"
    assert body["repo_id"] == "org/custom-synthetic-notes"
    record = DatasetStore().list_records(dataset_id="ds-custom-reference")[0]
    assert record.metadata["reference_key"] == "org/custom-synthetic-notes"
    assert record.metadata["reference_license"] == "cc-by-4.0"
    assert record.documents[0].extracted_facts["instruction"] == "Extract diagnosis."
    assert record.labs[0].name == "PaCO2"
    assert record.vitals[0].name == "SpO2"
    assert record.medication_history[0].name == "Albuterol"
    assert record.time_series[0].name == "respiratory_rate"


def test_dataset_api_imports_bundled_reference_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/datasets/reference-import",
        json={
            "reference_key": "clinical_notes_to_fhir",
            "dataset_id": "ds-fixture-reference",
            "fixture": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["imported"] == 1
    assert body["repo_id"] == "casecrawler-bundled-fixture"
    manifest = DatasetStore().get_manifest("ds-fixture-reference")
    record = DatasetStore().list_records(dataset_id="ds-fixture-reference")[0]
    assert manifest.metadata["primary_reference_key"] == "clinical_notes_to_fhir"
    assert record.labs[0].name == "Lactate"
    assert record.medication_history[0].name == "Ceftriaxone"


def test_dataset_api_imports_synthea_fhir_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    bundle_dir = tmp_path / "synthea"
    bundle_dir.mkdir()
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-1",
                    "gender": "female",
                    "birthDate": "1970-01-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-1",
                    "period": {"start": "2026-01-01T00:00:00"},
                    "reasonCode": [{"text": "sepsis"}],
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "Lactate"},
                    "valueQuantity": {"value": 3.4, "unit": "mmol/L"},
                    "effectiveDateTime": "2026-01-01T01:00:00",
                }
            },
        ],
    }
    (bundle_dir / "patient.json").write_text(json.dumps(bundle))

    response = client.post(
        "/api/datasets/synthea-import",
        json={"path": str(bundle_dir), "dataset_id": "ds-synthea"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "dataset_id": "ds-synthea",
        "imported": 1,
        "source": "synthea_fhir",
    }
    stored = client.get("/api/datasets/ds-synthea")
    assert stored.status_code == 200
    assert stored.json()["records"][0]["patient"]["patient_id"] == "pat-1"
    assert stored.json()["records"][0]["labs"][0]["name"] == "Lactate"
    assert stored.json()["manifest"]["metadata"]["primary_reference_key"] == "synthea_fhir"
    assert stored.json()["manifest"]["metadata"]["reference_keys"] == {"synthea_fhir": 1}


def test_dataset_api_imports_synthea_fhir_ndjson_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    bundle_dir = tmp_path / "synthea-ndjson"
    bundle_dir.mkdir()
    (bundle_dir / "Patient.ndjson").write_text(
        json.dumps({"resourceType": "Patient", "id": "pat-ndjson"}) + "\n"
    )

    response = client.post(
        "/api/datasets/synthea-import",
        json={"path": str(bundle_dir), "dataset_id": "ds-synthea-ndjson"},
    )

    assert response.status_code == 200
    assert response.json()["imported"] == 1
    stored = client.get("/api/datasets/ds-synthea-ndjson")
    assert stored.json()["records"][0]["patient"]["patient_id"] == "pat-ndjson"
    assert stored.json()["records"][0]["metadata"]["source_format"] == "fhir_ndjson"


def test_dataset_api_reports_empty_synthea_import_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    response = client.post(
        "/api/datasets/synthea-import",
        json={"path": str(empty_dir), "dataset_id": "ds-synthea"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "no Synthea FHIR JSON bundles or NDJSON files found"


def test_dataset_api_reports_unknown_hf_reference_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/datasets/reference-import",
        json={"reference_key": "missing", "dataset_id": "ds-missing"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "reference dataset not found"


def test_dataset_api_rejects_invalid_limits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    list_response = client.get("/api/datasets", params={"limit": -1})
    detail_response = client.get("/api/datasets/ds-missing", params={"limit": 5000})

    assert list_response.status_code == 422
    assert detail_response.status_code == 422
