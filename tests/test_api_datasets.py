import json

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
    asclepius = next(item for item in datasets if item["key"] == "asclepius")
    assert asclepius["repo_id"] == "starmpcc/Asclepius-Synthetic-Clinical-Notes"
    assert asclepius["license"]


def test_dataset_api_lists_generation_capabilities(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    response = client.get("/api/datasets/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert "clinical_text" in body["modalities"]
    assert "sft_jsonl" in body["export_formats"]
    assert "topic_mix" in body["cohort_constraints"]
    assert "topic_mix_weights" in body["cohort_constraints"]
    assert "cxr_pneumonia_dreambooth" in {
        profile["name"] for profile in body["imaging_model_profiles"]
    }
    assert "timediff" in {
        profile["name"] for profile in body["time_series_model_profiles"]
    }
    assert "sepsis" in {profile["key"] for profile in body["clinical_profiles"]}
    sepsis = next(profile for profile in body["clinical_profiles"] if profile["key"] == "sepsis")
    assert "Lactate" in sepsis["lab_names"]
    assert "Ceftriaxone" in sepsis["medication_names"]
    assert "biomedclip" in {validator["key"] for validator in body["validators"]}
    assert "medgemma" in {validator["key"] for validator in body["validators"]}


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
            "limit": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == "ds-custom-reference"
    assert body["reference_key"] == "org/custom-synthetic-notes"
    assert body["repo_id"] == "org/custom-synthetic-notes"
    record = DatasetStore().list_records(dataset_id="ds-custom-reference")[0]
    assert record.metadata["reference_license"] == "cc-by-4.0"
    assert record.documents[0].extracted_facts["instruction"] == "Extract diagnosis."


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
    assert response.json()["detail"] == "no Synthea FHIR JSON bundles found"


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
