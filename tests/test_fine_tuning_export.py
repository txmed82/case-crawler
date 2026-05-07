from casecrawler.export.fine_tuning import (
    export_dpo_record,
    export_chat_record,
    export_fhir_record,
    export_multimodal_record,
    export_parquet_record,
    export_record,
    export_rl_record,
    export_sft_record,
    export_tool_call_record,
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


def test_export_multimodal_record_preserves_imaging_labels_and_alignment_tasks():
    record = _multimodal_record()

    exported = export_multimodal_record(record)

    assert exported["clinical_context"]["imaging"][0]["image_id"] == "img-1"
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
    assert exported["supervised_tasks"][0]["target"]["labels"] == ["Opacity"]


def test_export_multimodal_record_inlines_existing_image_bytes(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"synthetic image bytes")
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

    assert exported["images"][0]["image_base64"] == "c3ludGhldGljIGltYWdlIGJ5dGVz"
    assert exported["images"][0]["image_mime_type"] == "image/png"


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


def test_export_parquet_record_flattens_modalities_for_tabular_storage():
    record = _multimodal_record()

    exported = export_parquet_record(record)

    assert exported["record_id"] == "rec-1"
    assert exported["patient_age"] == 64
    assert exported["patient_sex"] == "male"
    assert '"labs"' in exported["modalities"]
    assert "Lactate" in exported["labs_json"]
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
