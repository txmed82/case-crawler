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
    AllergyIntolerance,
    ClinicalDocument,
    ClinicalOrder,
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
    assert assistant_payload["allergies"][0]["substance"] == "Penicillin"
    assert assistant_payload["orders"][0]["display"] == "Lactate"
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
    assert manifest["task_coverage"] == {
        "clinical_lab_observation_interpretation": 5,
        "clinical_vital_observation_interpretation": 5,
    }
    assert manifest["splits"]["train"]["task_coverage"] == {
        "clinical_lab_observation_interpretation": 3,
        "clinical_vital_observation_interpretation": 3,
    }
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
    image_artifact = manifest["image_artifacts"][image_key]
    copied_image = tmp_path / "package" / package_path

    assert package_path == "images/rec-1-img-1.png"
    assert image_artifact["record_id"] == "rec-1"
    assert image_artifact["image_id"] == "img-1"
    assert image_artifact["modality"] == "xray"
    assert image_artifact["body_region"] == "chest"
    assert image_artifact["generation_backend"] == "placeholder"
    assert image_artifact["prompt"] == "Synthetic chest x-ray with right lower lobe opacity"
    assert image_artifact["report_text"] == "Right lower lobe opacity concerning for pneumonia."
    assert image_artifact["labels"][0]["code"] == "opacity"
    assert copied_image.read_bytes() == image_path.read_bytes()
    assert package_path in manifest["files"]
    assert manifest["files"][package_path]["byte_size"] == image_path.stat().st_size
    exported = json.loads((tmp_path / "package" / "train.jsonl").read_text())
    assert exported["images"][0]["package_path"] == package_path
    assert exported["image_text_pairs"][0]["package_path"] == package_path
    supervised_tasks = {task["task"]: task for task in exported["supervised_tasks"]}
    assert (
        supervised_tasks["radiology_image_report_alignment"]["input"]["package_path"]
        == package_path
    )
    report = verify_jsonl_split_package(tmp_path / "package")
    assert report["valid"] is True
    assert report["checked_files"][package_path]["exists"] is True


def test_export_jsonl_split_package_writes_time_series_artifacts(tmp_path):
    record = _multimodal_record().model_copy(
        update={
            "dataset_id": "ds-split",
            "metadata": {
                **_multimodal_record().metadata,
                "time_series_model_policy": {
                    "profile": "timediff",
                    "model_id": "MuhangTian/TimeDiff",
                    "license": "mit",
                    "gated": False,
                    "use_policy": "wrap_external_sampler_validate_outputs",
                },
            },
        }
    )

    manifest = export_jsonl_split_package(
        [record],
        tmp_path / "package",
        "time_series_jsonl",
        dataset_id="ds-split",
    )
    artifact_key = "rec-1:heart_rate"
    artifact = manifest["time_series_artifacts"][artifact_key]
    package_path = artifact["package_path"]
    copied_payload = json.loads((tmp_path / "package" / package_path).read_text())

    assert package_path == "time_series/rec-1-heart-rate.json"
    assert artifact["record_id"] == "rec-1"
    assert artifact["channel_name"] == "heart_rate"
    assert artifact["unit"] == "/min"
    assert artifact["generation_backend"] == "deterministic"
    assert artifact["time_series_model_policy"] == {
        "profile": "timediff",
        "model_id": "MuhangTian/TimeDiff",
        "license": "mit",
        "gated": False,
        "use_policy": "wrap_external_sampler_validate_outputs",
    }
    assert artifact["point_count"] == 1
    assert copied_payload["record_id"] == "rec-1"
    assert copied_payload["channel"]["name"] == "heart_rate"
    assert copied_payload["channel"]["points"][0]["values"]["heart_rate"] == 118
    assert package_path in manifest["files"]
    exported = json.loads((tmp_path / "package" / "train.jsonl").read_text())
    assert exported["channel"]["package_path"] == package_path

    report = verify_jsonl_split_package(tmp_path / "package")
    assert report["valid"] is True
    assert report["checked_files"][package_path]["exists"] is True


def test_verify_jsonl_split_package_requires_release_time_series_artifacts(tmp_path):
    record = _multimodal_record().model_copy(update={"dataset_id": "ds-split"})
    export_jsonl_split_package(
        [record],
        tmp_path / "package",
        "time_series_jsonl",
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
    manifest["time_series_artifacts"] = {}
    manifest_path.write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path / "package")

    assert report["valid"] is False
    assert any(issue["field"] == "time_series_artifacts" for issue in report["issues"])


def test_verify_jsonl_split_package_validates_task_coverage_summary(tmp_path):
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(3)
    ]
    export_jsonl_split_package(
        records,
        tmp_path / "package",
        "clinical_observation_jsonl",
        dataset_id="ds-split",
        train_ratio=0.34,
        validation_ratio=0.33,
        test_ratio=0.33,
        seed="unit-test",
    )
    manifest_path = tmp_path / "package" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["task_coverage"]["clinical_lab_observation_interpretation"] = 99
    manifest["splits"]["train"]["task_coverage"] = {}
    manifest_path.write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path / "package")
    issue_fields = {issue["field"] for issue in report["issues"]}

    assert report["valid"] is False
    assert "task_coverage" in issue_fields
    assert "splits.train.task_coverage" in issue_fields


def test_verify_jsonl_split_package_requires_release_task_coverage(tmp_path):
    record = _multimodal_record().model_copy(update={"dataset_id": "ds-split"})
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
    manifest["task_coverage"] = {}
    manifest_path.write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path / "package")

    assert report["valid"] is False
    assert any(issue["field"] == "task_coverage" for issue in report["issues"])


def test_verify_jsonl_split_package_validates_time_series_artifact_metadata(tmp_path):
    record = _multimodal_record().model_copy(update={"dataset_id": "ds-split"})
    export_jsonl_split_package(
        [record],
        tmp_path / "package",
        "time_series_jsonl",
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
    artifact = next(iter(manifest["time_series_artifacts"].values()))
    artifact["record_id"] = "rec-other"
    artifact["channel_name"] = ""
    artifact.pop("generation_backend")
    artifact["time_series_model_policy"] = {"profile": "timediff"}
    artifact["point_count"] = 0
    manifest["files"].pop(artifact["package_path"])
    manifest_path.write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path / "package")
    issue_fields = {issue["field"] for issue in report["issues"]}

    assert report["valid"] is False
    assert any(field.endswith(".record_id") for field in issue_fields)
    assert any(field.endswith(".channel_name") for field in issue_fields)
    assert any(field.endswith(".generation_backend") for field in issue_fields)
    assert any(field.endswith(".time_series_model_policy.license") for field in issue_fields)
    assert any(field.endswith(".point_count") for field in issue_fields)
    assert any(field.endswith(".package_path") for field in issue_fields)


def test_verify_jsonl_split_package_validates_time_series_jsonl_paths(tmp_path):
    record = _multimodal_record().model_copy(update={"dataset_id": "ds-split"})
    export_jsonl_split_package(
        [record],
        tmp_path / "package",
        "time_series_jsonl",
        dataset_id="ds-split",
    )
    train_path = tmp_path / "package" / "train.jsonl"
    payload = json.loads(train_path.read_text())
    payload["channel"]["package_path"] = "../unsafe.json"
    train_path.write_text(json.dumps(payload) + "\n")
    manifest_path = tmp_path / "package" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["train.jsonl"]["byte_size"] = train_path.stat().st_size
    manifest["files"]["train.jsonl"]["sha256"] = hashlib.sha256(
        train_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path / "package")
    issue_messages = [issue["message"] for issue in report["issues"]]

    assert report["valid"] is False
    assert any("not safe" in message for message in issue_messages)


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


def test_verify_jsonl_split_package_requires_release_image_artifact_metadata(tmp_path):
    image_path = tmp_path / "source-cxr.png"
    image_path.write_bytes(_png_bytes(width=32, height=32))
    record = _multimodal_record().model_copy(
        update={
            "dataset_id": "ds-split",
            "metadata": {
                "imaging_model_policy": {
                    "profile": "stable_diffusion_chest_xray",
                    "model_id": "danyalmalik/stable-diffusion-chest-xray",
                    "license": "creativeml-openrail-m",
                    "gated": False,
                    "use_policy": "openrail_review_outputs_before_release",
                },
                "image_validator_policy": {
                    "profile": "lexical",
                    "backend": "lexical",
                    "model_id": None,
                    "license": "casecrawler",
                    "gated": False,
                    "use_policy": "deterministic_screening_only",
                }
            },
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
    image_artifact = next(iter(manifest["image_artifacts"].values()))
    image_artifact["record_id"] = "rec-other"
    image_artifact["image_id"] = "img-other"
    image_artifact.pop("generation_backend")
    image_artifact["labels"] = [{"system": "synthetic", "code": "opacity"}]
    image_artifact["imaging_model_policy"].pop("use_policy")
    image_artifact["image_validator_policy"].pop("backend")
    image_artifact.pop("metadata")
    manifest_path.write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path / "package")
    issue_fields = {issue["field"] for issue in report["issues"]}

    assert report["valid"] is False
    assert any(field.endswith(".record_id") for field in issue_fields)
    assert any(field.endswith(".image_id") for field in issue_fields)
    assert any(field.endswith(".generation_backend") for field in issue_fields)
    assert any(field.endswith(".labels") for field in issue_fields)
    assert any(
        field.endswith(".imaging_model_policy.use_policy")
        for field in issue_fields
    )
    assert any(
        field.endswith(".image_validator_policy.backend")
        for field in issue_fields
    )
    assert any(field.endswith(".metadata") for field in issue_fields)


def test_verify_jsonl_split_package_validates_multimodal_jsonl_image_paths(tmp_path):
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
    train_path = tmp_path / "package" / "train.jsonl"
    payload = json.loads(train_path.read_text())
    payload["images"][0]["package_path"] = "images/not-declared.png"
    payload["image_text_pairs"][0]["package_path"] = "../unsafe.png"
    payload["supervised_tasks"][0]["input"]["package_path"] = "images/not-declared.png"
    train_path.write_text(json.dumps(payload) + "\n")
    manifest_path = tmp_path / "package" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["train.jsonl"]["byte_size"] = train_path.stat().st_size
    manifest["files"]["train.jsonl"]["sha256"] = hashlib.sha256(
        train_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path / "package")
    issue_messages = [issue["message"] for issue in report["issues"]]

    assert report["valid"] is False
    assert any("not declared in manifest image_artifacts" in message for message in issue_messages)
    assert any("not safe" in message for message in issue_messages)
    assert any("missing from manifest files" in message for message in issue_messages)


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


def test_verify_jsonl_split_package_rejects_inconsistent_passing_benchmark_suite(
    tmp_path,
):
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
        "recommended_reference_keys": [
            "synthclinicalnotes",
            "radiology_report_consistency",
        ],
        "task_export_results": {
            "multimodal_jsonl": {
                "recommended_reference_keys": ["radiology_report_consistency"],
                "reference_count": 0,
                "missing_reference_keys": ["radiology_report_consistency"],
                "passed": True,
            }
        },
        "reference_count": 1,
        "passed": True,
        "mean_overall_score": benchmark_report.overall_score,
        "thresholds": benchmark_report.thresholds,
        "results": [
            {
                "reference_key": "synthclinicalnotes",
                "reference_dataset_id": "ds-ref",
                "passed": False,
                "overall_score": benchmark_report.overall_score,
                "failing_metrics": ["imaging_label_overlap"],
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

    report = verify_jsonl_split_package(tmp_path)
    issue_fields = {issue["field"] for issue in report["issues"]}

    assert report["valid"] is False
    assert (
        "audit_artifacts.benchmark_suite_report.json.recommended_reference_keys"
        in issue_fields
    )
    assert "audit_artifacts.benchmark_suite_report.json.passed" in issue_fields
    assert (
        "audit_artifacts.benchmark_suite_report.json.task_export_results."
        "multimodal_jsonl.missing_reference_keys"
    ) in issue_fields
    assert (
        "audit_artifacts.benchmark_suite_report.json.task_export_results."
        "multimodal_jsonl.reference_count"
    ) in issue_fields


def test_verify_jsonl_split_package_validates_release_package_summary_artifact(
    tmp_path,
):
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(3)
    ]
    release_summary = {
        "dataset_id": "ds-split",
        "generated": 3,
        "approved": 3,
        "seeded_references": {"imported": ["ds-ref"]},
        "task_coverage": [],
        "quality_report": {
            "export_ready": True,
            "multimodal_release_ready": True,
            "multimodal_release_missing": ["radiology_images"],
            "core_artifact_coverage": {
                key: True for key in REQUIRED_RELEASE_COVERAGE_KEYS
            },
            "mean_imaging_prompt_chars": "long",
            "mean_imaging_report_chars": False,
            "imaging_report_label_evidence_rate": 1.5,
            "time_series_channel_counts": [],
            "mean_time_series_points": "many",
            "mean_time_series_duration_hours": False,
        },
        "benchmark": {
            "reference_dataset_id": "ds-ref",
            "reference_key": "synthclinicalnotes",
            "passed": True,
            "overall_score": 0.9,
            "failing_metrics": ["imaging_label_overlap"],
            "thresholds": {"min_overall_score": 0.75, "min_metric_score": 0.5},
        },
        "benchmark_suite": {
            "passed": True,
            "reference_count": 0,
            "mean_overall_score": 0.9,
            "task_export_results": [],
        },
        "objective_coverage": {
            "objective": "Generate multimodal synthetic healthcare training data.",
            "complete": True,
            "missing": ["radiology_images"],
            "criteria": {
                "records": {
                    "requirement": "Synthetic records are generated.",
                    "satisfied": True,
                    "artifacts": ["quality_report.json"],
                    "evidence": {"record_count": 3},
                },
                "labs": {
                    "requirement": "Lab observations are present.",
                    "satisfied": "yes",
                    "artifacts": ["quality_report.json"],
                    "evidence": {},
                },
            },
        },
    }
    release_summary["quality_report"]["core_artifact_coverage"]["radiology_images"] = False
    export_jsonl_split_package(
        records,
        tmp_path,
        "sft_jsonl",
        dataset_id="ds-split",
        train_ratio=0.34,
        validation_ratio=0.33,
        test_ratio=0.33,
        seed="unit-test",
        audit_artifacts={"release_package_summary.json": release_summary},
    )
    summary_payload = json.loads((tmp_path / "release_package_summary.json").read_text())
    summary_payload["dataset_id"] = "ds-other"
    (tmp_path / "release_package_summary.json").write_text(json.dumps(summary_payload))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"]["release_package_summary.json"]["byte_size"] = (
        tmp_path / "release_package_summary.json"
    ).stat().st_size
    manifest["files"]["release_package_summary.json"]["sha256"] = hashlib.sha256(
        (tmp_path / "release_package_summary.json").read_bytes()
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path)
    issue_fields = {issue["field"] for issue in report["issues"]}

    assert report["valid"] is False
    assert "audit_artifacts.release_package_summary.json.dataset_id" in issue_fields
    assert (
        "audit_artifacts.release_package_summary.json.quality_report."
        "core_artifact_coverage"
    ) in issue_fields
    assert (
        "audit_artifacts.release_package_summary.json.quality_report."
        "multimodal_release_missing"
    ) in issue_fields
    assert (
        "audit_artifacts.release_package_summary.json.quality_report."
        "mean_imaging_prompt_chars"
    ) in issue_fields
    assert (
        "audit_artifacts.release_package_summary.json.quality_report."
        "mean_imaging_report_chars"
    ) in issue_fields
    assert (
        "audit_artifacts.release_package_summary.json.quality_report."
        "imaging_report_label_evidence_rate"
    ) in issue_fields
    assert (
        "audit_artifacts.release_package_summary.json.quality_report."
        "time_series_channel_counts"
    ) in issue_fields
    assert (
        "audit_artifacts.release_package_summary.json.quality_report."
        "mean_time_series_points"
    ) in issue_fields
    assert (
        "audit_artifacts.release_package_summary.json.quality_report."
        "mean_time_series_duration_hours"
    ) in issue_fields
    assert "audit_artifacts.release_package_summary.json.task_coverage" in issue_fields
    assert (
        "audit_artifacts.release_package_summary.json.benchmark.failing_metrics"
        in issue_fields
    )
    assert (
        "audit_artifacts.release_package_summary.json.benchmark_suite.reference_count"
        in issue_fields
    )
    assert (
        "audit_artifacts.release_package_summary.json.benchmark_suite."
        "task_export_results"
    ) in issue_fields
    assert (
        "audit_artifacts.release_package_summary.json.benchmark_suite.results"
        in issue_fields
    )
    assert (
        "audit_artifacts.release_package_summary.json.objective_coverage.missing"
        in issue_fields
    )
    assert (
        "audit_artifacts.release_package_summary.json.objective_coverage.criteria"
        in issue_fields
    )
    assert (
        "audit_artifacts.release_package_summary.json.objective_coverage."
        "criteria.labs.satisfied"
    ) in issue_fields


def test_verify_jsonl_split_package_rejects_passing_release_summary_with_failed_task_benchmarks(
    tmp_path,
):
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(3)
    ]
    release_summary = {
        "dataset_id": "ds-split",
        "generated": 3,
        "approved": 3,
        "seeded_references": {"imported": ["ds-ref"]},
        "quality_report": {
            "export_ready": True,
            "multimodal_release_ready": False,
            "multimodal_release_missing": [],
            "core_artifact_coverage": {
                key: False for key in REQUIRED_RELEASE_COVERAGE_KEYS
            },
        },
        "benchmark": {
            "reference_dataset_id": "ds-ref",
            "reference_key": "synthchex_75k",
            "passed": True,
            "overall_score": 0.9,
            "failing_metrics": [],
            "thresholds": {"min_overall_score": 0.75, "min_metric_score": 0.5},
        },
        "benchmark_suite": {
            "passed": True,
            "reference_count": 1,
            "mean_overall_score": 0.9,
            "task_export_results": {
                "multimodal_jsonl": {
                    "recommended_reference_keys": [
                        "synthchex_75k",
                        "radiology_report_consistency",
                    ],
                    "reference_count": 0,
                    "missing_reference_keys": ["radiology_report_consistency"],
                    "passed": False,
                    "mean_overall_score": None,
                    "results": [],
                }
            },
        },
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
        audit_artifacts={"release_package_summary.json": release_summary},
    )

    report = verify_jsonl_split_package(tmp_path)
    issue_fields = {issue["field"] for issue in report["issues"]}

    assert report["valid"] is False
    assert (
        "audit_artifacts.release_package_summary.json.benchmark_suite."
        "task_export_results.multimodal_jsonl.reference_count"
    ) in issue_fields
    assert (
        "audit_artifacts.release_package_summary.json.benchmark_suite."
        "task_export_results.multimodal_jsonl.passed"
    ) in issue_fields
    assert (
        "audit_artifacts.release_package_summary.json.benchmark_suite."
        "task_export_results.multimodal_jsonl.missing_reference_keys"
    ) in issue_fields


def test_verify_jsonl_split_package_rejects_passing_release_summary_with_missing_benchmark_results(
    tmp_path,
):
    records = [
        _multimodal_record().model_copy(
            update={"record_id": f"rec-{index}", "dataset_id": "ds-split"}
        )
        for index in range(3)
    ]
    release_summary = {
        "dataset_id": "ds-split",
        "generated": 3,
        "approved": 3,
        "seeded_references": {"imported": ["ds-ref"]},
        "quality_report": {
            "export_ready": True,
            "multimodal_release_ready": False,
            "multimodal_release_missing": [],
            "core_artifact_coverage": {
                key: False for key in REQUIRED_RELEASE_COVERAGE_KEYS
            },
        },
        "benchmark": {
            "reference_dataset_id": "ds-ref",
            "reference_key": "synthchex_75k",
            "passed": True,
            "overall_score": 0.9,
            "failing_metrics": [],
            "thresholds": {"min_overall_score": 0.75, "min_metric_score": 0.5},
        },
        "benchmark_suite": {
            "passed": True,
            "reference_count": 2,
            "mean_overall_score": 0.9,
            "recommended_reference_keys": [
                "synthchex_75k",
                "radiology_report_consistency",
            ],
            "task_export_results": {},
            "results": [
                {
                    "reference_key": "synthchex_75k",
                    "reference_dataset_id": "ds-ref",
                    "passed": True,
                    "overall_score": 0.9,
                    "failing_metrics": [],
                    "report": {},
                }
            ],
        },
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
        audit_artifacts={"release_package_summary.json": release_summary},
    )

    report = verify_jsonl_split_package(tmp_path)
    issue_fields = {issue["field"] for issue in report["issues"]}

    assert report["valid"] is False
    assert (
        "audit_artifacts.release_package_summary.json.benchmark_suite."
        "reference_count"
    ) in issue_fields
    assert (
        "audit_artifacts.release_package_summary.json.benchmark_suite."
        "recommended_reference_keys"
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
                "mean_imaging_prompt_chars": "long",
                "mean_imaging_report_chars": False,
                "imaging_report_label_evidence_rate": 1.5,
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
    assert "audit_artifacts.quality_report.json.mean_imaging_prompt_chars" in issue_fields
    assert "audit_artifacts.quality_report.json.mean_imaging_report_chars" in issue_fields
    assert (
        "audit_artifacts.quality_report.json.imaging_report_label_evidence_rate"
        in issue_fields
    )
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


def test_verify_jsonl_split_package_requires_release_row_provenance_and_policies(
    tmp_path,
):
    record = _multimodal_record().model_copy(
        update={
            "dataset_id": "ds-split",
            "metadata": {
                "clinical_text_model_policy": {
                    "backend": "llm",
                    "provider": "ollama",
                    "model_id": "medgemma-local",
                    "license": "provider_terms",
                    "gated": False,
                    "use_policy": (
                        "synthetic_clinical_text_review_outputs_before_release"
                    ),
                },
                "imaging_model_policy": {
                    "profile": "cxr_pneumonia_dreambooth",
                    "backend": "diffusers",
                    "model_id": "rexgradient/synthetic-chest-xray-pneumonia",
                    "license": "cc-by-nc-4.0",
                    "gated": False,
                    "use_policy": (
                        "non_commercial_no_derivatives_review_before_release"
                    ),
                },
                "image_validator_policy": {
                    "profile": "biomedclip",
                    "backend": "BiomedCLIPImageValidator",
                    "model_id": (
                        "hf-hub:microsoft/"
                        "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                    ),
                    "license": "mit",
                    "gated": False,
                    "use_policy": "open_validation_model_review_alignment_scores",
                },
                "time_series_model_policy": {
                    "profile": "timediff",
                    "backend": "external",
                    "model_id": "MuhangTian/TimeDiff",
                    "license": "mit",
                    "gated": False,
                    "use_policy": "wrap_external_generator_validate_outputs",
                },
            },
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
                "multimodal_release_ready": False,
                "multimodal_release_missing": ["radiology_images"],
                "clinical_text_model_policy_counts": {
                    (
                        "backend=llm|provider=ollama|model_id=medgemma-local|"
                        "gated=false|use_policy=synthetic"
                    ): 1
                },
                "imaging_model_policy_counts": {
                    "profile=cxr_pneumonia_dreambooth": 1
                },
                "image_validator_policy_counts": {"profile=biomedclip": 1},
                "time_series_model_policy_counts": {"profile=timediff": 1},
            }
        },
    )
    train_path = tmp_path / "package" / "train.jsonl"
    payload = json.loads(train_path.read_text())
    payload["metadata"].pop("provenance")
    payload["metadata"].pop("clinical_text_model_policy")
    payload["metadata"].pop("imaging_model_policy")
    payload["metadata"].pop("image_validator_policy")
    payload["metadata"].pop("time_series_model_policy")
    train_path.write_text(json.dumps(payload) + "\n")
    manifest_path = tmp_path / "package" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["train.jsonl"]["byte_size"] = train_path.stat().st_size
    manifest["files"]["train.jsonl"]["sha256"] = hashlib.sha256(
        train_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    report = verify_jsonl_split_package(tmp_path / "package")
    issue_fields = {issue["field"] for issue in report["issues"]}

    assert report["valid"] is False
    assert "train.jsonl.line.1.metadata.provenance" in issue_fields
    assert "train.jsonl.line.1.metadata.clinical_text_model_policy" in issue_fields
    assert "train.jsonl.line.1.metadata.imaging_model_policy" in issue_fields
    assert "train.jsonl.line.1.metadata.image_validator_policy" in issue_fields
    assert "train.jsonl.line.1.metadata.time_series_model_policy" in issue_fields


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
    assert target["record_context"]["orders"][0]["display"] == "Lactate"
    assert target["record_context"]["diagnoses"][0]["display"] == "Sepsis"
    assert target["record_context"]["procedures"][0]["display"] == (
        "Central venous catheter placement"
    )
    assert target["record_context"]["imaging_labels"][0]["labels"][0]["display"] == "Opacity"
    assert example["metadata"]["note_type"] == "ed_note"
    assert example["metadata"]["export_profile"] == "note_fact_sft_jsonl"


def test_export_training_rows_preserve_provenance_and_model_policies():
    policy_metadata = {
        "clinical_text_model_policy": {
            "backend": "llm",
            "provider": "ollama",
            "model_id": "medgemma-local",
            "license": "provider_terms",
            "gated": False,
            "use_policy": "synthetic_clinical_text_review_outputs_before_release",
        },
        "imaging_model_policy": {
            "profile": "cxr_pneumonia_dreambooth",
            "backend": "diffusers",
            "model_id": "rexgradient/synthetic-chest-xray-pneumonia",
            "license": "cc-by-nc-4.0",
            "gated": False,
            "use_policy": "non_commercial_no_derivatives_review_before_release",
        },
        "image_validator_policy": {
            "profile": "biomedclip",
            "backend": "BiomedCLIPImageValidator",
            "model_id": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "license": "mit",
            "gated": False,
            "use_policy": "open_validation_model_review_alignment_scores",
        },
        "time_series_model_policy": {
            "profile": "timediff",
            "backend": "external",
            "model_id": "MuhangTian/TimeDiff",
            "license": "research",
            "gated": False,
            "use_policy": "wrap_external_generator_validate_outputs",
        },
    }
    record = _multimodal_record().model_copy(
        update={
            "metadata": policy_metadata,
            "provenance": Provenance(
                generator="unit-test-generator",
                model="unit-test-model",
                source_refs=[{"source": "fixture", "id": "ref-1"}],
                prompt_hash="abc123",
                created_at="2026-05-06T10:00:00",
            ),
        }
    )

    exported = [
        export_sft_record(record),
        export_chat_record(record),
        export_multimodal_record(record),
        export_note_fact_sft_records(record)[0],
        export_clinical_observation_records(record)[0],
        export_time_series_records(record)[0],
        export_medication_reconciliation_records(record)[0],
    ]

    for row in exported:
        metadata = row["metadata"]
        assert metadata["provenance"]["generator"] == "unit-test-generator"
        assert metadata["provenance"]["model"] == "unit-test-model"
        assert metadata["provenance"]["source_refs"] == [
            {"source": "fixture", "id": "ref-1"}
        ]
        assert metadata["provenance"]["prompt_hash"] == "abc123"
        assert metadata["clinical_text_model_policy"] == (
            policy_metadata["clinical_text_model_policy"]
        )
        assert metadata["imaging_model_policy"] == policy_metadata["imaging_model_policy"]
        assert metadata["image_validator_policy"] == (
            policy_metadata["image_validator_policy"]
        )
        assert metadata["time_series_model_policy"] == (
            policy_metadata["time_series_model_policy"]
        )


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
                    update={
                        "file_path": str(image_path),
                        "metadata": {
                            "generation_backend": "placeholder",
                            "file": {"width": 64, "height": 48},
                        },
                    }
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
    assert image["metadata"]["file"] == {"width": 64, "height": 48}
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
    assert "AllergyIntolerance" in resource_types
    assert "ServiceRequest" in resource_types
    assert "MedicationRequest" in resource_types
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
    allergy = next(
        resource
        for resource in resources
        if resource["resourceType"] == "AllergyIntolerance"
    )
    assert allergy["code"]["coding"][0]["display"] == "Penicillin"
    assert allergy["reaction"][0]["manifestation"][0]["text"] == "hives"
    service_request = next(
        resource for resource in resources if resource["resourceType"] == "ServiceRequest"
    )
    medication_request = next(
        resource for resource in resources if resource["resourceType"] == "MedicationRequest"
    )
    assert service_request["code"]["coding"][0]["display"] == "Lactate"
    assert medication_request["medicationCodeableConcept"]["coding"][0]["display"] == (
        "Ceftriaxone"
    )
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
        allergies=[
            AllergyIntolerance(
                substance="Penicillin",
                code="7980",
                system="RxNorm",
                reaction="hives",
                severity="moderate",
                recorded_at="2026-05-01",
            )
        ],
        orders=[
            ClinicalOrder(
                order_id="ord-lactate",
                order_type="laboratory",
                display="Lactate",
                code="2524-7",
                system="LOINC",
                status="completed",
                priority="stat",
                ordered_at="2026-05-06T10:00:00",
                encounter_id="enc-1",
            ),
            ClinicalOrder(
                order_id="ord-ceftriaxone",
                order_type="medication",
                display="Ceftriaxone",
                code="2193",
                system="RxNorm",
                status="active",
                priority="stat",
                ordered_at="2026-05-06T10:00:00",
                encounter_id="enc-1",
            ),
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
