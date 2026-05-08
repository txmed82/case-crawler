import json
import hashlib
import struct
import zipfile
import zlib

from casecrawler.export.fine_tuning import (
    REQUIRED_RELEASE_COVERAGE_KEYS,
    export_clinical_observation_records,
    export_dpo_record,
    export_chat_record,
    export_fhir_record,
    export_jsonl_split_package,
    export_medication_reconciliation_records,
    export_multimodal_record,
    export_note_fact_sft_records,
    export_parquet_record,
    export_record,
    export_record_payloads,
    export_rl_record,
    export_sft_record,
    export_time_series_records,
    export_tool_call_record,
    verify_fhir_bundle,
    verify_fhir_ndjson_export,
    verify_jsonl_split_package,
)
from casecrawler.models.synthetic import (
    ClinicalDocument,
    Code,
    ComplexityProfile,
    Encounter,
    ImagingAsset,
    LabObservation,
    MedicationStatement,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    TimeSeriesChannel,
    TimeSeriesPoint,
    VitalObservation,
)
from casecrawler.validation.benchmark import (
    DatasetBenchmark,
    benchmark_profile_artifact,
    profile_records,
)


def test_export_sft_record_contains_messages():
    record = SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.CLINICAL_TEXT],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        encounters=[],
        documents=[
            ClinicalDocument(
                document_id="doc-1",
                note_type="ed_note",
                author_role="physician",
                timestamp="2026-05-06T10:00:00",
                clean_text="Patient has fever, hypotension, elevated lactate.",
            )
        ],
        provenance=Provenance(
            generator="unit-test",
            created_at="2026-05-06T10:00:00",
        ),
    )

    exported = export_sft_record(record, task="summarize")

    assert exported["record_id"] == "rec-1"
    assert exported["messages"][0]["role"] == "system"
    assert exported["messages"][1]["role"] == "user"
    assert exported["messages"][2]["role"] == "assistant"


def test_export_sft_record_includes_structured_context_without_documents():
    record = _multimodal_record().model_copy(update={"documents": []})

    exported = export_sft_record(record, task="extract")

    user_message = exported["messages"][1]["content"]
    assert "Structured facts:" in user_message
    assert "Lactate" in user_message
    assert "Ceftriaxone" in user_message
    assert "img-1" in user_message


def test_export_sft_extract_record_targets_full_structured_context():
    record = _multimodal_record()

    exported = export_sft_record(record, task="extract")
    assistant_payload = json.loads(exported["messages"][2]["content"])

    assert assistant_payload["record_id"] == "rec-1"
    assert assistant_payload["patient"]["age"] == 64
    assert assistant_payload["diagnoses"][0]["display"] == "Sepsis"
    assert assistant_payload["procedures"][0]["display"] == (
        "Central venous catheter placement"
    )
    assert assistant_payload["labs"][0]["name"] == "Lactate"
    assert assistant_payload["vitals"][0]["name"] == "Heart rate"
    assert assistant_payload["medication_history"][0]["name"] == "Ceftriaxone"
    assert assistant_payload["time_series"][0]["name"] == "heart_rate"
    assert assistant_payload["documents"][0]["document_id"] == "doc-1"
    assert assistant_payload["imaging"][0]["image_id"] == "img-1"
    assert assistant_payload["provenance"]["generator"] == "unit-test"
    assert assistant_payload["synthetic"] is True


def test_export_jsonl_split_package_writes_manifest_and_stable_splits(tmp_path):
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(5)
    ]

    manifest = export_jsonl_split_package(
        records,
        tmp_path,
        "clinical_observation_jsonl",
        dataset_id="ds-split",
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        seed="unit-test",
        audit_artifacts={
            "quality_report.json": {"export_ready": True},
            "dataset_card.md": "# Dataset Card\n",
        },
    )
    repeated = export_jsonl_split_package(
        records,
        tmp_path / "repeat",
        "clinical_observation_jsonl",
        dataset_id="ds-split",
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        seed="unit-test",
    )

    assert manifest["dataset_id"] == "ds-split"
    assert manifest["export_format"] == "clinical_observation_jsonl"
    assert manifest["record_count"] == 5
    assert manifest["splits"]["train"]["record_count"] == 3
    assert manifest["splits"]["validation"]["record_count"] == 1
    assert manifest["splits"]["test"]["record_count"] == 1
    assert set(manifest["audit_artifacts"]) == {
        "dataset_card.md",
        "quality_report.json",
    }
    assert set(manifest["files"]) == {
        "dataset_card.md",
        "quality_report.json",
        "test.jsonl",
        "train.jsonl",
        "validation.jsonl",
    }
    assert manifest["splits"]["train"]["example_count"] == 6
    assert manifest["splits"]["train"]["record_ids"] == repeated["splits"]["train"]["record_ids"]
    assert (tmp_path / "manifest.json").exists()
    assert json.loads((tmp_path / "quality_report.json").read_text())["export_ready"] is True
    assert (tmp_path / "dataset_card.md").read_text() == "# Dataset Card\n"
    assert manifest["files"]["train.jsonl"]["byte_size"] == (tmp_path / "train.jsonl").stat().st_size
    assert manifest["files"]["train.jsonl"]["sha256"] == hashlib.sha256(
        (tmp_path / "train.jsonl").read_bytes()
    ).hexdigest()
    assert (tmp_path / "train.jsonl").read_text().count("\n") == 6
    first_payload = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    assert first_payload["task"] in {
        "clinical_lab_observation_interpretation",
        "clinical_vital_observation_interpretation",
    }


def test_export_jsonl_split_package_copies_file_backed_images(tmp_path):
    image_path = tmp_path / "source-cxr.png"
    image_path.write_bytes(_png_bytes(width=32, height=32))
    record = _multimodal_record().model_copy(
        update={
            "dataset_id": "ds-split",
            "imaging": [
                _multimodal_record().imaging[0].model_copy(
                    update={"file_path": str(image_path)}
                )
            ],
        }
    )

    manifest = export_jsonl_split_package(
        [record],
        tmp_path / "package",
        "multimodal_jsonl",
        dataset_id="ds-split",
    )
    image_key = "rec-1:img-1"
    package_path = manifest["image_artifacts"][image_key]["package_path"]
    copied_image = tmp_path / "package" / package_path

    assert package_path == "images/rec-1-img-1.png"
    assert copied_image.read_bytes() == image_path.read_bytes()
    assert package_path in manifest["files"]
    assert manifest["files"][package_path]["byte_size"] == image_path.stat().st_size
    report = verify_jsonl_split_package(tmp_path / "package")
    assert report["valid"] is True
    assert report["checked_files"][package_path]["exists"] is True


def test_verify_jsonl_split_package_requires_release_image_artifacts(tmp_path):
    image_path = tmp_path / "source-cxr.png"
    image_path.write_bytes(_png_bytes(width=32, height=32))
    record = _multimodal_record().model_copy(
        update={
            "dataset_id": "ds-split",
            "imaging": [
                _multimodal_record().imaging[0].model_copy(
                    update={"file_path": str(image_path)}
                )
            ],
        }
    )
    export_jsonl_split_package(
        [record],
        tmp_path / "package",
        "multimodal_jsonl",
        dataset_id="ds-split",
        audit_artifacts={
            "quality_report.json": {
                "dataset_id": "ds-split",
                "record_count": 1,
                "approved_count": 1,
                "approval_rate": 1.0,
                "export_ready": True,
                "core_artifact_coverage": {
                    key: True for key in REQUIRED_RELEASE_COVERAGE_KEYS
                },
                "multimodal_release_ready": True,
                "multimodal_release_missing": [],
            }
        },
    )
    manifest_path = tmp_path / "package" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["image_artifacts"] = {}
    manifest_path.write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path / "package")

    assert report["valid"] is False
    assert any(issue["field"] == "image_artifacts" for issue in report["issues"])


def test_verify_jsonl_split_package_validates_image_artifact_manifest_files(tmp_path):
    image_path = tmp_path / "source-cxr.png"
    image_path.write_bytes(_png_bytes(width=32, height=32))
    record = _multimodal_record().model_copy(
        update={
            "dataset_id": "ds-split",
            "imaging": [
                _multimodal_record().imaging[0].model_copy(
                    update={"file_path": str(image_path)}
                )
            ],
        }
    )
    export_jsonl_split_package(
        [record],
        tmp_path / "package",
        "multimodal_jsonl",
        dataset_id="ds-split",
    )
    manifest_path = tmp_path / "package" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    image_artifact = next(iter(manifest["image_artifacts"].values()))
    manifest["files"].pop(image_artifact["package_path"])
    manifest_path.write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path / "package")

    assert report["valid"] is False
    assert any(
        issue["field"].endswith(".package_path")
        and "missing from manifest files" in issue["message"]
        for issue in report["issues"]
    )


def test_verify_jsonl_split_package_accepts_valid_moved_package(tmp_path):
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(4)
    ]
    export_jsonl_split_package(
        records,
        tmp_path,
        "sft_jsonl",
        dataset_id="ds-split",
        train_ratio=0.5,
        validation_ratio=0.25,
        test_ratio=0.25,
        seed="unit-test",
        audit_artifacts={
            "quality_report.json": {
                "dataset_id": "ds-split",
                "record_count": 4,
                "approved_count": 4,
                "approval_rate": 1.0,
                "export_ready": True,
                "core_artifact_coverage": {"records": True},
                "multimodal_release_ready": False,
                "multimodal_release_missing": ["benchmark_reference"],
            }
        },
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for file_metadata in manifest["files"].values():
        file_metadata["path"] = "/moved/package/location"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))

    report = verify_jsonl_split_package(tmp_path)

    assert report["valid"] is True
    assert report["dataset_id"] == "ds-split"
    assert report["quality_report"]["multimodal_release_ready"] is False
    assert report["quality_report"]["multimodal_release_missing"] == [
        "benchmark_reference"
    ]
    assert report["checked_files"]["train.jsonl"]["exists"] is True
    assert report["splits"]["train"]["example_count"] == 2


def test_verify_jsonl_split_package_accepts_zip_archive(tmp_path):
    package_dir = tmp_path / "package"
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(3)
    ]
    export_jsonl_split_package(
        records,
        package_dir,
        "sft_jsonl",
        dataset_id="ds-split",
        train_ratio=0.34,
        validation_ratio=0.33,
        test_ratio=0.33,
        seed="unit-test",
    )
    archive_path = tmp_path / "package.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in package_dir.iterdir():
            archive.write(path, arcname=path.name)

    report = verify_jsonl_split_package(archive_path)

    assert report["valid"] is True
    assert report["archive"] is True
    assert report["package_dir"] == str(archive_path)
    assert report["splits"]["test"]["example_count"] == 1


def test_verify_jsonl_split_package_validates_fhir_split_examples(tmp_path):
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(3)
    ]
    export_jsonl_split_package(
        records,
        tmp_path,
        "fhir_ndjson",
        dataset_id="ds-split",
        train_ratio=0.34,
        validation_ratio=0.33,
        test_ratio=0.33,
        seed="unit-test",
    )
    train_payload = json.loads((tmp_path / "train.jsonl").read_text().splitlines()[0])
    observation = next(
        entry["resource"]
        for entry in train_payload["entry"]
        if entry["resource"]["resourceType"] == "Observation"
    )
    observation.pop("subject")
    (tmp_path / "train.jsonl").write_text(json.dumps(train_payload) + "\n")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"]["train.jsonl"]["byte_size"] = (tmp_path / "train.jsonl").stat().st_size
    manifest["files"]["train.jsonl"]["sha256"] = hashlib.sha256(
        (tmp_path / "train.jsonl").read_bytes()
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path)

    assert report["valid"] is False
    assert any(
        issue["field"].startswith("train.jsonl.line.1.Observation.subject")
        for issue in report["issues"]
    )


def test_verify_jsonl_split_package_validates_benchmark_profile_artifact(tmp_path):
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(3)
    ]
    export_jsonl_split_package(
        records,
        tmp_path,
        "sft_jsonl",
        dataset_id="ds-split",
        train_ratio=0.34,
        validation_ratio=0.33,
        test_ratio=0.33,
        seed="unit-test",
        audit_artifacts={
            "benchmark_profile.json": benchmark_profile_artifact(
                profile_records(records)
            )
        },
    )
    profile_payload = json.loads((tmp_path / "benchmark_profile.json").read_text())
    profile_payload["profile"]["dataset_id"] = "ds-other"
    (tmp_path / "benchmark_profile.json").write_text(json.dumps(profile_payload))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"]["benchmark_profile.json"]["byte_size"] = (
        tmp_path / "benchmark_profile.json"
    ).stat().st_size
    manifest["files"]["benchmark_profile.json"]["sha256"] = hashlib.sha256(
        (tmp_path / "benchmark_profile.json").read_bytes()
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path)

    assert report["valid"] is False
    assert any(
        issue["field"]
        == "audit_artifacts.benchmark_profile.json.profile.dataset_id"
        for issue in report["issues"]
    )


def test_verify_jsonl_split_package_validates_benchmark_report_artifact(tmp_path):
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(3)
    ]
    reference_records = [
        record.model_copy(update={"record_id": f"ref-{index}", "dataset_id": "ds-ref"})
        for index, record in enumerate(records)
    ]
    benchmark_report = DatasetBenchmark(
        min_overall_score=0.0,
        min_metric_score=0.0,
    ).compare(records, reference_records)
    export_jsonl_split_package(
        records,
        tmp_path,
        "sft_jsonl",
        dataset_id="ds-split",
        train_ratio=0.34,
        validation_ratio=0.33,
        test_ratio=0.33,
        seed="unit-test",
        audit_artifacts={
            "benchmark_report.json": benchmark_report.model_dump(mode="json")
        },
    )
    report_payload = json.loads((tmp_path / "benchmark_report.json").read_text())
    report_payload["generated_dataset_id"] = "ds-other"
    (tmp_path / "benchmark_report.json").write_text(json.dumps(report_payload))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"]["benchmark_report.json"]["byte_size"] = (
        tmp_path / "benchmark_report.json"
    ).stat().st_size
    manifest["files"]["benchmark_report.json"]["sha256"] = hashlib.sha256(
        (tmp_path / "benchmark_report.json").read_bytes()
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path)

    assert report["valid"] is False
    assert any(
        issue["field"]
        == "audit_artifacts.benchmark_report.json.generated_dataset_id"
        for issue in report["issues"]
    )


def test_verify_jsonl_split_package_validates_benchmark_suite_artifact(tmp_path):
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(3)
    ]
    reference_records = [
        record.model_copy(update={"record_id": f"ref-{index}", "dataset_id": "ds-ref"})
        for index, record in enumerate(records)
    ]
    benchmark_report = DatasetBenchmark(
        min_overall_score=0.0,
        min_metric_score=0.0,
    ).compare(records, reference_records)
    benchmark_suite = {
        "dataset_id": "ds-split",
        "primary_recipe": "full_multimodal_acute_care",
        "recommended_reference_keys": ["synthchex_75k"],
        "task_export_results": {},
        "reference_count": 1,
        "passed": True,
        "mean_overall_score": benchmark_report.overall_score,
        "thresholds": benchmark_report.thresholds,
        "results": [
            {
                "reference_key": "synthchex_75k",
                "reference_dataset_id": "ds-ref",
                "passed": True,
                "overall_score": benchmark_report.overall_score,
                "failing_metrics": [],
                "report": benchmark_report.model_dump(mode="json"),
            }
        ],
    }
    export_jsonl_split_package(
        records,
        tmp_path,
        "sft_jsonl",
        dataset_id="ds-split",
        train_ratio=0.34,
        validation_ratio=0.33,
        test_ratio=0.33,
        seed="unit-test",
        audit_artifacts={"benchmark_suite_report.json": benchmark_suite},
    )
    suite_payload = json.loads((tmp_path / "benchmark_suite_report.json").read_text())
    suite_payload["dataset_id"] = "ds-other"
    suite_payload["results"][0]["report"]["generated_dataset_id"] = "ds-other"
    (tmp_path / "benchmark_suite_report.json").write_text(json.dumps(suite_payload))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"]["benchmark_suite_report.json"]["byte_size"] = (
        tmp_path / "benchmark_suite_report.json"
    ).stat().st_size
    manifest["files"]["benchmark_suite_report.json"]["sha256"] = hashlib.sha256(
        (tmp_path / "benchmark_suite_report.json").read_bytes()
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path)
    issue_fields = {issue["field"] for issue in report["issues"]}

    assert report["valid"] is False
    assert "audit_artifacts.benchmark_suite_report.json.dataset_id" in issue_fields
    assert (
        "audit_artifacts.benchmark_suite_report.json."
        "results.0.report.generated_dataset_id"
    ) in issue_fields


def test_verify_jsonl_split_package_validates_quality_report_dataset_id(tmp_path):
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(3)
    ]
    export_jsonl_split_package(
        records,
        tmp_path,
        "sft_jsonl",
        dataset_id="ds-split",
        train_ratio=0.34,
        validation_ratio=0.33,
        test_ratio=0.33,
        seed="unit-test",
        audit_artifacts={
            "quality_report.json": {
                "dataset_id": "ds-other",
                "record_count": 3,
                "approved_count": 3,
                "approval_rate": 1.0,
                "export_ready": True,
            }
        },
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"]["quality_report.json"]["byte_size"] = (
        tmp_path / "quality_report.json"
    ).stat().st_size
    manifest["files"]["quality_report.json"]["sha256"] = hashlib.sha256(
        (tmp_path / "quality_report.json").read_bytes()
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path)

    assert report["valid"] is False
    assert any(
        issue["field"] == "audit_artifacts.quality_report.json.dataset_id"
        for issue in report["issues"]
    )


def test_verify_jsonl_split_package_validates_quality_report_release_fields(tmp_path):
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(3)
    ]
    export_jsonl_split_package(
        records,
        tmp_path,
        "sft_jsonl",
        dataset_id="ds-split",
        train_ratio=0.34,
        validation_ratio=0.33,
        test_ratio=0.33,
        seed="unit-test",
        audit_artifacts={
            "quality_report.json": {
                "dataset_id": "ds-split",
                "record_count": 3,
                "approved_count": 3,
                "approval_rate": 1.0,
                "export_ready": True,
                "core_artifact_coverage": {"records": "yes"},
                "multimodal_release_ready": "false",
                "multimodal_release_missing": [False],
            }
        },
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"]["quality_report.json"]["byte_size"] = (
        tmp_path / "quality_report.json"
    ).stat().st_size
    manifest["files"]["quality_report.json"]["sha256"] = hashlib.sha256(
        (tmp_path / "quality_report.json").read_bytes()
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path)

    assert report["valid"] is False
    issue_fields = {issue["field"] for issue in report["issues"]}
    assert "audit_artifacts.quality_report.json.core_artifact_coverage" in issue_fields
    assert "audit_artifacts.quality_report.json.multimodal_release_ready" in issue_fields
    assert "audit_artifacts.quality_report.json.multimodal_release_missing" in issue_fields


def test_verify_jsonl_split_package_requires_release_coverage_keys(tmp_path):
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(3)
    ]
    export_jsonl_split_package(
        records,
        tmp_path,
        "sft_jsonl",
        dataset_id="ds-split",
        train_ratio=0.34,
        validation_ratio=0.33,
        test_ratio=0.33,
        seed="unit-test",
        audit_artifacts={
            "quality_report.json": {
                "dataset_id": "ds-split",
                "record_count": 3,
                "approved_count": 3,
                "approval_rate": 1.0,
                "export_ready": True,
                "core_artifact_coverage": {"records": True},
                "multimodal_release_ready": True,
                "multimodal_release_missing": [],
            }
        },
    )

    report = verify_jsonl_split_package(tmp_path)

    assert report["valid"] is False
    assert any(
        issue["field"] == "audit_artifacts.quality_report.json.core_artifact_coverage"
        and "missing required release coverage keys" in issue["message"]
        for issue in report["issues"]
    )


def test_verify_jsonl_split_package_rejects_ready_release_with_false_coverage(tmp_path):
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(3)
    ]
    coverage = {key: True for key in REQUIRED_RELEASE_COVERAGE_KEYS}
    coverage["lab_reports"] = False
    export_jsonl_split_package(
        records,
        tmp_path,
        "sft_jsonl",
        dataset_id="ds-split",
        train_ratio=0.34,
        validation_ratio=0.33,
        test_ratio=0.33,
        seed="unit-test",
        audit_artifacts={
            "quality_report.json": {
                "dataset_id": "ds-split",
                "record_count": 3,
                "approved_count": 3,
                "approval_rate": 1.0,
                "export_ready": True,
                "core_artifact_coverage": coverage,
                "multimodal_release_ready": True,
                "multimodal_release_missing": [],
            }
        },
    )

    report = verify_jsonl_split_package(tmp_path)

    assert report["valid"] is False
    assert any(
        issue["field"] == "audit_artifacts.quality_report.json.core_artifact_coverage"
        and "false release coverage keys" in issue["message"]
        and "lab_reports" in issue["message"]
        for issue in report["issues"]
    )


def test_verify_jsonl_split_package_validates_card_headers(tmp_path):
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(3)
    ]
    export_jsonl_split_package(
        records,
        tmp_path,
        "sft_jsonl",
        dataset_id="ds-split",
        train_ratio=0.34,
        validation_ratio=0.33,
        test_ratio=0.33,
        seed="unit-test",
        audit_artifacts={
            "dataset_card.md": "# Dataset Card: wrong-dataset\n",
            "model_card.md": "# Model Card: wrong-dataset synthetic generation pipeline\n",
        },
    )

    report = verify_jsonl_split_package(tmp_path)

    assert report["valid"] is False
    assert any(
        issue["field"] == "audit_artifacts.dataset_card.md.title"
        for issue in report["issues"]
    )
    assert any(
        issue["field"] == "audit_artifacts.model_card.md.title"
        for issue in report["issues"]
    )


def test_verify_jsonl_split_package_rejects_zip_with_unsafe_path(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../manifest.json", "{}")

    report = verify_jsonl_split_package(archive_path)

    assert report["valid"] is False
    assert report["issues"][0]["field"] == "zip"


def test_verify_jsonl_split_package_rejects_tampered_checksum_and_counts(tmp_path):
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(3)
    ]
    export_jsonl_split_package(
        records,
        tmp_path,
        "sft_jsonl",
        dataset_id="ds-split",
        train_ratio=0.34,
        validation_ratio=0.33,
        test_ratio=0.33,
        seed="unit-test",
    )
    with (tmp_path / "train.jsonl").open("a") as f:
        f.write('{"record_id": "rec-extra"}\n')

    report = verify_jsonl_split_package(tmp_path)

    assert report["valid"] is False
    issue_fields = {issue["field"] for issue in report["issues"]}
    assert "files.train.jsonl.byte_size" in issue_fields
    assert "files.train.jsonl.sha256" in issue_fields
    assert "splits.train.example_count" in issue_fields
    assert "splits.train.record_ids" in issue_fields


def test_export_note_fact_sft_records_creates_document_level_examples():
    record = _multimodal_record()

    examples = export_note_fact_sft_records(record)

    assert len(examples) == 1
    example = examples[0]
    assert example["record_id"] == "rec-1"
    assert example["document_id"] == "doc-1"
    assert example["task"] == "extract_clinical_facts_from_note"
    assert "pt fever hypotn lactate hi" in example["messages"][1]["content"]
    target = json.loads(example["messages"][2]["content"])
    assert target["document"]["document_id"] == "doc-1"
    assert target["document"]["extracted_facts"] == {
        "lab_values": [{"name": "Lactate", "value": 4.2, "unit": "mmol/L"}],
        "vital_values": [{"name": "Heart rate", "value": 122, "unit": "/min"}],
        "medications": ["Ceftriaxone"],
        "imaging_labels": ["Opacity"],
    }
    assert target["record_context"]["labs"][0]["name"] == "Lactate"
    assert target["record_context"]["vitals"][0]["name"] == "Heart rate"
    assert target["record_context"]["medication_history"][0]["name"] == "Ceftriaxone"
    assert target["record_context"]["diagnoses"][0]["display"] == "Sepsis"
    assert target["record_context"]["procedures"][0]["display"] == (
        "Central venous catheter placement"
    )
    assert target["record_context"]["imaging_labels"][0]["labels"][0]["display"] == "Opacity"
    assert example["metadata"]["note_type"] == "ed_note"
    assert example["metadata"]["export_profile"] == "note_fact_sft_jsonl"


def test_export_record_dispatches_chat_and_multimodal():
    record = SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
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

    chat = export_chat_record(record)
    multimodal = export_multimodal_record(record)

    assert export_record(record, "chat_jsonl") == chat
    assert export_record(record, "multimodal_jsonl") == multimodal
    assert chat["messages"][0]["role"] == "system"
    assert multimodal["images"] == []
    assert multimodal["clinical_context"]["record_id"] == "rec-1"


def test_export_record_dispatches_note_fact_sft_profile():
    record = _multimodal_record()

    exported = export_record(record, "note_fact_sft_jsonl")
    payloads = export_record_payloads(record, "note_fact_sft_jsonl")

    assert exported["metadata"]["export_profile"] == "note_fact_sft_jsonl"
    assert exported["examples"] == payloads
    assert payloads[0]["document_id"] == "doc-1"


def test_export_clinical_observation_records_creates_lab_and_vital_examples():
    record = _multimodal_record()

    examples = export_clinical_observation_records(record)

    assert [example["task"] for example in examples] == [
        "clinical_lab_observation_interpretation",
        "clinical_vital_observation_interpretation",
    ]
    lab = examples[0]
    assert lab["input"]["observation_kind"] == "lab"
    assert lab["input"]["observation"]["name"] == "Lactate"
    assert lab["target"] == {
        "name": "Lactate",
        "loinc": "2524-7",
        "value": 4.2,
        "unit": "mmol/L",
        "reference_low": 0.5,
        "reference_high": 2.2,
        "flag": "high",
        "effective_time": "2026-05-06T10:15:00",
        "specimen": "blood",
        "abnormal": True,
    }
    assert lab["metadata"]["export_profile"] == "clinical_observation_jsonl"
    assert lab["metadata"]["observation_kind"] == "lab"
    vital = examples[1]
    assert vital["input"]["observation_kind"] == "vital"
    assert vital["target"] == {
        "name": "Heart rate",
        "value": 122.0,
        "unit": "/min",
        "effective_time": "2026-05-06T10:10:00",
        "abnormal": True,
        "direction": "high",
    }
    assert vital["clinical_context"]["diagnoses"][0]["display"] == "Sepsis"


def test_export_clinical_observation_records_derives_lab_flags_from_reference_range():
    record = _multimodal_record().model_copy(
        update={
            "labs": [
                LabObservation(
                    name="Potassium",
                    value=2.9,
                    unit="mmol/L",
                    reference_low=3.5,
                    reference_high=5.0,
                    effective_time="2026-05-06T10:15:00",
                )
            ],
            "vitals": [],
        }
    )

    examples = export_clinical_observation_records(record)

    assert examples[0]["target"]["flag"] == "L"
    assert examples[0]["target"]["abnormal"] is True


def test_export_time_series_records_creates_channel_level_training_examples():
    record = _multimodal_record().model_copy(
        update={
            "time_series": [
                TimeSeriesChannel(
                    name="heart_rate",
                    unit="/min",
                    sampling_rate_hz=0.2,
                    points=[
                        TimeSeriesPoint(
                            timestamp="2026-05-06T10:00:00",
                            values={"heart_rate": 118},
                        ),
                        TimeSeriesPoint(
                            timestamp="2026-05-06T10:05:00",
                            values={"heart_rate": 122},
                        ),
                    ],
                    generation_backend="deterministic",
                )
            ]
        }
    )

    examples = export_time_series_records(record)

    assert len(examples) == 1
    example = examples[0]
    assert example["record_id"] == "rec-1"
    assert example["task"] == "clinical_time_series_forecasting"
    assert example["channel"]["name"] == "heart_rate"
    assert example["channel"]["sampling_rate_hz"] == 0.2
    assert example["input"]["points"] == [
        {"timestamp": "2026-05-06T10:00:00", "values": {"heart_rate": 118.0}}
    ]
    assert example["target"]["points"] == [
        {"timestamp": "2026-05-06T10:05:00", "values": {"heart_rate": 122.0}}
    ]
    assert example["clinical_context"]["labs"][0]["name"] == "Lactate"
    assert example["clinical_context"]["vitals"][0]["name"] == "Heart rate"
    assert example["clinical_context"]["diagnoses"][0]["display"] == "Sepsis"
    assert example["metadata"]["export_profile"] == "time_series_jsonl"


def test_export_medication_reconciliation_records_creates_medication_level_examples():
    record = _multimodal_record()

    examples = export_medication_reconciliation_records(record)

    assert len(examples) == 1
    example = examples[0]
    assert example["record_id"] == "rec-1"
    assert example["task"] == "medication_reconciliation"
    assert example["input"]["candidate_medication"] == "Ceftriaxone"
    assert example["input"]["notes"][0]["extracted_medications"] == ["Ceftriaxone"]
    assert example["target"]["normalized_name"] == "Ceftriaxone"
    assert example["target"]["rxnorm"] == "2193"
    assert example["target"]["dose"] == "2 g"
    assert example["target"]["route"] == "IV"
    assert example["target"]["frequency"] == "daily"
    assert example["target"]["status"] == "active"
    assert example["target"]["active"] is True
    assert example["clinical_context"]["medication_history"][0]["name"] == "Ceftriaxone"
    assert example["metadata"]["export_profile"] == "medication_reconciliation_jsonl"


def test_export_medication_reconciliation_records_marks_inactive_medications():
    record = _multimodal_record().model_copy(
        update={
            "medication_history": [
                MedicationStatement(
                    name="Warfarin",
                    rxnorm="11289",
                    dose="5 mg",
                    route="oral",
                    frequency="daily",
                    status="stopped",
                    start="2026-05-01",
                    end="2026-05-06",
                )
            ]
        }
    )

    examples = export_medication_reconciliation_records(record)

    assert examples[0]["target"]["active"] is False
    assert examples[0]["target"]["period"] == {
        "start": "2026-05-01",
        "end": "2026-05-06",
    }


def test_export_record_dispatches_time_series_profile_as_multiple_payloads():
    record = _multimodal_record()

    exported = export_record(record, "time_series_jsonl")
    payloads = export_record_payloads(record, "time_series_jsonl")

    assert exported["metadata"]["export_profile"] == "time_series_jsonl"
    assert exported["examples"] == payloads
    assert payloads[0]["channel"]["name"] == "heart_rate"


def test_export_record_dispatches_clinical_observation_profile():
    record = _multimodal_record()

    exported = export_record(record, "clinical_observation_jsonl")
    payloads = export_record_payloads(record, "clinical_observation_jsonl")

    assert exported["metadata"]["export_profile"] == "clinical_observation_jsonl"
    assert exported["examples"] == payloads
    assert payloads[0]["target"]["name"] == "Lactate"


def test_export_record_dispatches_medication_reconciliation_profile():
    record = _multimodal_record()

    exported = export_record(record, "medication_reconciliation_jsonl")
    payloads = export_record_payloads(record, "medication_reconciliation_jsonl")

    assert exported["metadata"]["export_profile"] == "medication_reconciliation_jsonl"
    assert exported["examples"] == payloads
    assert payloads[0]["target"]["normalized_name"] == "Ceftriaxone"


def test_export_multimodal_record_preserves_imaging_labels_and_alignment_tasks():
    record = _multimodal_record()

    exported = export_multimodal_record(record)

    assert exported["clinical_context"]["imaging"][0]["image_id"] == "img-1"
    assert exported["clinical_context"]["diagnoses"][0]["display"] == "Sepsis"
    assert exported["clinical_context"]["procedures"][0]["display"] == (
        "Central venous catheter placement"
    )
    assert exported["images"][0]["labels"] == [
        {
            "system": "https://casecrawler.dev/synthetic-radiology-labels",
            "code": "opacity",
            "display": "Opacity",
        }
    ]
    assert exported["image_text_pairs"] == [
        {
            "image_id": "img-1",
            "text": "Right lower lobe opacity concerning for pneumonia.",
            "task": "radiology_image_report_alignment",
            "labels": ["Opacity"],
        }
    ]
    supervised_tasks = {task["task"]: task for task in exported["supervised_tasks"]}
    assert set(supervised_tasks) == {
        "radiology_image_report_alignment",
        "radiology_report_generation",
        "radiology_label_extraction",
    }
    assert supervised_tasks["radiology_image_report_alignment"]["target"]["labels"] == [
        "Opacity"
    ]
    assert supervised_tasks["radiology_report_generation"]["input"]["labels"] == [
        "Opacity"
    ]
    assert supervised_tasks["radiology_report_generation"]["target"]["report_text"] == (
        "Right lower lobe opacity concerning for pneumonia."
    )
    assert supervised_tasks["radiology_label_extraction"]["input"]["report_text"] == (
        "Right lower lobe opacity concerning for pneumonia."
    )
    assert supervised_tasks["radiology_label_extraction"]["target"]["labels"] == [
        "Opacity"
    ]


def test_export_multimodal_record_inlines_existing_image_bytes(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png_bytes(width=64, height=48))
    record = _multimodal_record().model_copy(
        update={
            "imaging": [
                _multimodal_record().imaging[0].model_copy(
                    update={"file_path": str(image_path)}
                )
            ]
        }
    )

    exported = export_multimodal_record(record)

    image = exported["images"][0]
    assert image["image_base64"]
    assert image["image_metadata"]["mime_type"] == "image/png"
    assert image["image_metadata"]["width"] == 64
    assert image["image_metadata"]["height"] == 48
    assert image["image_metadata"]["byte_size"] == image_path.stat().st_size
    assert len(image["image_metadata"]["sha256"]) == 64
    assert exported["supervised_tasks"][0]["input"]["image_metadata"]["width"] == 64


def test_export_fhir_record_contains_training_bundle_resources():
    record = _multimodal_record()

    exported = export_fhir_record(record)
    resources = [entry["resource"] for entry in exported["entry"]]
    resource_types = {resource["resourceType"] for resource in resources}

    assert exported["resourceType"] == "Bundle"
    assert exported["type"] == "collection"
    assert "Patient" in resource_types
    assert "Encounter" in resource_types
    assert "Observation" in resource_types
    assert "Condition" in resource_types
    assert "Procedure" in resource_types
    assert "MedicationStatement" in resource_types
    assert "DocumentReference" in resource_types
    assert "DiagnosticReport" in resource_types
    assert "Provenance" in resource_types
    assert any(
        resource["resourceType"] == "Observation"
        and resource["code"]["coding"][0]["code"] == "2524-7"
        for resource in resources
        if resource["resourceType"] == "Observation"
        and resource["code"].get("coding")
    )
    conditions = [
        resource for resource in resources if resource["resourceType"] == "Condition"
    ]
    assert conditions[0]["code"]["coding"][0]["code"] == "91302008"
    encounter = next(
        resource for resource in resources if resource["resourceType"] == "Encounter"
    )
    assert encounter["diagnosis"][0]["condition"]["reference"] == (
        f"Condition/{conditions[0]['id']}"
    )
    procedures = [
        resource for resource in resources if resource["resourceType"] == "Procedure"
    ]
    assert procedures[0]["code"]["coding"][0]["display"] == "Central venous catheter placement"
    assert procedures[0]["encounter"]["reference"] == "Encounter/enc-1"
    lab = next(
        resource
        for resource in resources
        if resource["resourceType"] == "Observation"
        and resource["code"].get("coding")
        and resource["code"]["coding"][0]["code"] == "2524-7"
    )
    vital = next(
        resource
        for resource in resources
        if resource["resourceType"] == "Observation"
        and resource["id"].startswith("rec-1-vital-heart-rate")
    )
    assert lab["encounter"]["reference"] == "Encounter/enc-1"
    assert vital["encounter"]["reference"] == "Encounter/enc-1"
    time_series = next(
        resource
        for resource in resources
        if resource["resourceType"] == "Observation"
        and resource["id"] == "rec-1-timeseries-heart-rate"
    )
    assert time_series["effectivePeriod"] == {
        "start": "2026-05-06T10:00:00",
        "end": "2026-05-06T10:00:00",
    }
    assert time_series["encounter"]["reference"] == "Encounter/enc-1"
    assert time_series["component"][0]["code"]["text"] == "heart_rate"
    assert time_series["component"][0]["extension"][0] == {
        "url": "https://casecrawler.dev/fhir/StructureDefinition/sample-timestamp",
        "valueDateTime": "2026-05-06T10:00:00",
    }
    assert time_series["component"][0]["extension"][1] == {
        "url": "https://casecrawler.dev/fhir/StructureDefinition/sample-encounter",
        "valueReference": {"reference": "Encounter/enc-1"},
    }
    assert verify_fhir_bundle(exported)["valid"] is True


def test_verify_fhir_bundle_rejects_missing_subject_and_observation_value():
    bundle = export_fhir_record(_multimodal_record())
    resources = [entry["resource"] for entry in bundle["entry"]]
    observation = next(resource for resource in resources if resource["resourceType"] == "Observation")
    observation.pop("subject")
    observation.pop("valueQuantity", None)
    observation.pop("valueString", None)

    report = verify_fhir_bundle(bundle)

    assert report["valid"] is False
    issue_fields = {issue["field"] for issue in report["issues"]}
    assert "Observation.subject" in issue_fields
    assert "Observation.value" in issue_fields


def test_verify_fhir_ndjson_export_reports_invalid_lines(tmp_path):
    path = tmp_path / "fhir.ndjson"
    path.write_text(json.dumps(export_fhir_record(_multimodal_record())) + "\nnot-json\n")

    report = verify_fhir_ndjson_export(path)

    assert report["valid"] is False
    assert report["bundle_count"] == 1
    assert any(issue["field"] == "line.2" for issue in report["issues"])


def test_export_fhir_record_links_longitudinal_observations_to_encounters():
    base = _multimodal_record()
    record = base.model_copy(
        update={
            "encounters": [
                *base.encounters,
                Encounter(
                    encounter_id="enc-2",
                    start="2026-05-07T10:00:00",
                    end="2026-05-07T14:00:00",
                    setting="inpatient",
                    reason="Sepsis reassessment",
                    diagnoses=base.encounters[0].diagnoses,
                ),
            ],
            "labs": [
                *base.labs,
                LabObservation(
                    name="Lactate",
                    loinc="2524-7",
                    value=2.1,
                    unit="mmol/L",
                    reference_low=0.5,
                    reference_high=2.2,
                    effective_time="2026-05-07T10:15:00",
                ),
            ],
            "vitals": [
                *base.vitals,
                VitalObservation(
                    name="Heart rate",
                    value=94,
                    unit="/min",
                    effective_time="2026-05-07T10:10:00",
                ),
            ],
            "time_series": [
                TimeSeriesChannel(
                    name="heart_rate",
                    unit="/min",
                    points=[
                        TimeSeriesPoint(
                            timestamp="2026-05-06T10:00:00",
                            values={"heart_rate": 118},
                        ),
                        TimeSeriesPoint(
                            timestamp="2026-05-07T10:00:00",
                            values={"heart_rate": 94},
                        ),
                    ],
                )
            ],
        }
    )

    exported = export_fhir_record(record)
    resources = [entry["resource"] for entry in exported["entry"]]
    labs = [
        resource
        for resource in resources
        if resource["resourceType"] == "Observation"
        and resource["id"].startswith("rec-1-lab-lactate")
    ]
    vitals = [
        resource
        for resource in resources
        if resource["resourceType"] == "Observation"
        and resource["id"].startswith("rec-1-vital-heart-rate")
    ]
    time_series = next(
        resource
        for resource in resources
        if resource["resourceType"] == "Observation"
        and resource["id"] == "rec-1-timeseries-heart-rate"
    )

    assert [lab["encounter"]["reference"] for lab in labs] == [
        "Encounter/enc-1",
        "Encounter/enc-2",
    ]
    assert [vital["encounter"]["reference"] for vital in vitals] == [
        "Encounter/enc-1",
        "Encounter/enc-2",
    ]
    assert "encounter" not in time_series
    assert [
        component["extension"][1]["valueReference"]["reference"]
        for component in time_series["component"]
    ] == ["Encounter/enc-1", "Encounter/enc-2"]


def test_export_fhir_record_preserves_waveform_sampling_metadata():
    record = _multimodal_record().model_copy(
        update={
            "time_series": [
                TimeSeriesChannel(
                    name="ecg_lead_ii",
                    unit="mV",
                    sampling_rate_hz=125,
                    points=[
                        TimeSeriesPoint(
                            timestamp="2026-05-06T10:00:00",
                            values={"millivolts": 0.12},
                        ),
                        TimeSeriesPoint(
                            timestamp="2026-05-06T10:00:01",
                            values={"millivolts": 0.09},
                        ),
                    ],
                )
            ]
        }
    )

    exported = export_fhir_record(record)
    resources = [entry["resource"] for entry in exported["entry"]]
    waveform = next(
        resource
        for resource in resources
        if resource["resourceType"] == "Observation"
        and resource["id"] == "rec-1-timeseries-ecg-lead-ii"
    )

    assert waveform["effectivePeriod"] == {
        "start": "2026-05-06T10:00:00",
        "end": "2026-05-06T10:00:01",
    }
    assert waveform["extension"][0] == {
        "url": "https://casecrawler.dev/fhir/StructureDefinition/sampling-rate-hz",
        "valueDecimal": 125,
    }


def test_export_parquet_record_flattens_modalities_for_tabular_storage():
    record = _multimodal_record()

    exported = export_parquet_record(record)

    assert exported["record_id"] == "rec-1"
    assert exported["patient_age"] == 64
    assert exported["patient_sex"] == "male"
    assert '"labs"' in exported["modalities"]
    assert "Lactate" in exported["labs_json"]
    assert "Sepsis" in exported["diagnoses_json"]
    assert "Central venous catheter placement" in exported["procedures_json"]
    assert "Central venous catheter placement" in exported["procedure_names_json"]
    assert "ed_note" in exported["documents_json"]
    assert exported["synthetic"] is True


def test_export_record_dispatches_fhir_and_parquet():
    record = _multimodal_record()

    assert export_record(record, "fhir_ndjson") == export_fhir_record(record)
    assert export_record(record, "parquet") == export_parquet_record(record)


def test_export_tool_call_record_contains_clinical_extraction_call():
    record = _multimodal_record()

    exported = export_tool_call_record(record)
    assistant = exported["messages"][-1]

    assert exported["tools"][0]["function"]["name"] == "emit_synthetic_clinical_facts"
    assert assistant["tool_calls"][0]["function"]["name"] == "emit_synthetic_clinical_facts"
    assert "Lactate" in assistant["tool_calls"][0]["function"]["arguments"]
    assert "img-1" in assistant["tool_calls"][0]["function"]["arguments"]
    assert "Central venous catheter placement" in assistant["tool_calls"][0]["function"][
        "arguments"
    ]
    assert "procedures" in exported["tools"][0]["function"]["parameters"]["required"]
    assert exported["metadata"]["export_profile"] == "tool_call_jsonl"


def test_export_dpo_record_contains_preferred_and_rejected_answers():
    record = _multimodal_record()

    exported = export_dpo_record(record)

    assert exported["prompt"][0]["role"] == "system"
    assert "chosen" in exported
    assert "rejected" in exported
    assert "synthetic" in exported["chosen"][0]["content"].lower()
    assert "ignore" in exported["rejected"][0]["content"].lower()
    assert exported["metadata"]["export_profile"] == "dpo_jsonl"


def test_export_rl_record_contains_rewarded_clinical_actions():
    record = _multimodal_record()

    exported = export_rl_record(record)
    step = exported["steps"][0]

    assert exported["record_id"] == "rec-1"
    assert step["observation"]["patient"]["age"] == 64
    assert step["optimal_action"] == "review_structured_record"
    assert step["reward_table"]["review_structured_record"] == 1.0
    assert step["reward_table"]["disregard_synthetic_provenance"] < 0
    assert exported["metadata"]["export_profile"] == "rl_jsonl"


def test_export_record_dispatches_training_profiles():
    record = _multimodal_record()

    assert export_record(record, "tool_call_jsonl") == export_tool_call_record(record)
    assert export_record(record, "dpo_jsonl") == export_dpo_record(record)
    assert export_record(record, "rl_jsonl") == export_rl_record(record)


def _multimodal_record() -> SyntheticRecord:
    return SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="sepsis",
        complexity=ComplexityProfile.COMPLEX,
        modalities=[
            Modality.STRUCTURED_EHR,
            Modality.CLINICAL_TEXT,
            Modality.LABS,
            Modality.VITALS,
            Modality.TIME_SERIES,
            Modality.IMAGING,
        ],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        encounters=[
            Encounter(
                encounter_id="enc-1",
                start="2026-05-06T10:00:00",
                end="2026-05-06T14:00:00",
                setting="emergency",
                reason="Fever and hypotension",
                diagnoses=[
                    Code(
                        system="http://snomed.info/sct",
                        code="91302008",
                        display="Sepsis",
                    )
                ],
                procedures=[
                    Code(
                        system="http://snomed.info/sct",
                        code="232717009",
                        display="Central venous catheter placement",
                    )
                ],
            )
        ],
        labs=[
            LabObservation(
                name="Lactate",
                loinc="2524-7",
                value=4.2,
                unit="mmol/L",
                reference_low=0.5,
                reference_high=2.2,
                flag="high",
                effective_time="2026-05-06T10:15:00",
                specimen="blood",
            )
        ],
        vitals=[
            VitalObservation(
                name="Heart rate",
                value=122,
                unit="/min",
                effective_time="2026-05-06T10:10:00",
            )
        ],
        medication_history=[
            MedicationStatement(
                name="Ceftriaxone",
                rxnorm="2193",
                dose="2 g",
                route="IV",
                frequency="daily",
                status="active",
                start="2026-05-06",
            )
        ],
        time_series=[
            TimeSeriesChannel(
                name="heart_rate",
                unit="/min",
                points=[
                    TimeSeriesPoint(
                        timestamp="2026-05-06T10:00:00",
                        values={"heart_rate": 118},
                    )
                ],
            )
        ],
        documents=[
            ClinicalDocument(
                document_id="doc-1",
                note_type="ed_note",
                author_role="physician",
                timestamp="2026-05-06T10:30:00",
                clean_text="Patient has fever, hypotension, and elevated lactate.",
                messy_text="pt fever hypotn lactate hi",
                extracted_facts={
                    "lab_values": [
                        {"name": "Lactate", "value": 4.2, "unit": "mmol/L"}
                    ],
                    "vital_values": [
                        {"name": "Heart rate", "value": 122, "unit": "/min"}
                    ],
                    "medications": ["Ceftriaxone"],
                    "imaging_labels": ["Opacity"],
                },
            )
        ],
        imaging=[
            ImagingAsset(
                image_id="img-1",
                modality="xray",
                body_region="chest",
                prompt="Synthetic chest x-ray with right lower lobe opacity",
                report_text="Right lower lobe opacity concerning for pneumonia.",
                labels=[
                    Code(
                        system="https://casecrawler.dev/synthetic-radiology-labels",
                        code="opacity",
                        display="Opacity",
                    )
                ],
                generation_backend="placeholder",
            )
        ],
        provenance=Provenance(
            generator="unit-test",
            created_at="2026-05-06T10:00:00",
        ),
    )


def _png_bytes(*, width: int, height: int) -> bytes:
    raw = b"".join(b"\x00" + (b"\x80" * width) for _ in range(height))
    chunks = [
        b"\x89PNG\r\n\x1a\n",
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(raw)),
        _png_chunk(b"IEND", b""),
    ]
    return b"".join(chunks)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )
