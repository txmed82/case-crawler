import pytest

from casecrawler.generation.synthetic_pipeline import SyntheticPipeline
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import Code, ImagingAsset, Modality
from casecrawler.validation.synthetic_validator import SyntheticValidator


@pytest.mark.asyncio
async def test_synthetic_pipeline_generates_valid_records():
    pipeline = SyntheticPipeline(validator=SyntheticValidator())

    result = await pipeline.generate(GenerationRequest(topic="sepsis", count=2))

    assert result["generated"] == 2
    assert result["approved"] == 2
    assert len(result["records"]) == 2
    assert result["records"][0].documents
    assert result["records"][0].labs
    assert {
        "ed_note",
        "progress_note",
        "nursing_note",
        "discharge_summary",
        "lab_report",
        "vital_signs_flowsheet",
        "medication_administration_record",
    }.issubset({document.note_type for document in result["records"][0].documents})


@pytest.mark.asyncio
async def test_synthetic_pipeline_applies_generation_recipe():
    pipeline = SyntheticPipeline(validator=SyntheticValidator())

    result = await pipeline.generate(
        GenerationRequest(
            topic="acute care",
            recipe="radiology_cxr_report",
            count=2,
            imaging_backend="placeholder",
        )
    )

    assert result["plan"].modalities == [
        Modality.STRUCTURED_EHR,
        Modality.CLINICAL_TEXT,
        Modality.IMAGING,
    ]
    assert {record.topic for record in result["records"]}.issubset(
        {"pneumonia", "heart failure", "status asthmaticus", "pulmonary embolism"}
    )
    assert result["records"][0].metadata["generation_overrides"]["recipe"] == (
        "radiology_cxr_report"
    )
    assert result["records"][0].imaging


@pytest.mark.asyncio
async def test_synthetic_pipeline_generates_topic_mix_cohorts():
    pipeline = SyntheticPipeline(validator=SyntheticValidator())

    result = await pipeline.generate(
        GenerationRequest(
            topic="mixed acute care cohort",
            count=4,
            cohort_constraints={"topic_mix": ["sepsis", "pneumonia"]},
        )
    )

    assert [record.topic for record in result["records"]] == [
        "sepsis",
        "pneumonia",
        "sepsis",
        "pneumonia",
    ]
    assert result["records"][0].metadata["cohort_constraints"]["topic_mix"] == [
        "sepsis",
        "pneumonia",
    ]


@pytest.mark.asyncio
async def test_synthetic_pipeline_generates_weighted_topic_mix_cohorts():
    pipeline = SyntheticPipeline(validator=SyntheticValidator())

    result = await pipeline.generate(
        GenerationRequest(
            topic="weighted acute care cohort",
            count=6,
            cohort_constraints={
                "topic_mix": [
                    {"topic": "sepsis", "weight": 2},
                    {"topic": "pneumonia", "weight": 1},
                ]
            },
        )
    )

    assert [record.topic for record in result["records"]] == [
        "sepsis",
        "sepsis",
        "pneumonia",
        "sepsis",
        "sepsis",
        "pneumonia",
    ]


@pytest.mark.asyncio
async def test_synthetic_pipeline_generates_topic_mix_with_weight_map():
    pipeline = SyntheticPipeline(validator=SyntheticValidator())

    result = await pipeline.generate(
        GenerationRequest(
            topic="weighted acute care cohort",
            count=4,
            cohort_constraints={
                "topic_mix": ["sepsis", "pneumonia"],
                "topic_mix_weights": {"sepsis": 3},
            },
        )
    )

    assert [record.topic for record in result["records"]] == [
        "sepsis",
        "sepsis",
        "sepsis",
        "pneumonia",
    ]


@pytest.mark.asyncio
async def test_synthetic_pipeline_rejects_invalid_topic_mix_weight():
    pipeline = SyntheticPipeline(validator=SyntheticValidator())

    with pytest.raises(ValueError, match="topic_mix weights"):
        await pipeline.generate(
            GenerationRequest(
                topic="weighted acute care cohort",
                cohort_constraints={"topic_mix": [{"topic": "sepsis", "weight": 0}]},
            )
        )


class FakeImagingGenerator:
    def __init__(self):
        self.diffusers_calls = []

    def generate_diffusers(
        self,
        output_dir: str,
        prompt: str,
        modality: str = "XR",
        body_region: str = "chest",
        negative_prompt: str | None = None,
    ):
        self.diffusers_calls.append((output_dir, prompt, modality, body_region))
        return ImagingAsset(
            image_id="img-test",
            modality=modality,
            body_region=body_region,
            prompt=prompt,
            file_path="fake.png",
            report_text=f"Synthetic {modality} image for {prompt}",
            labels=[
                Code(
                    system="synthetic",
                    code="pneumonia",
                    display="Pneumonia",
                )
            ],
            generation_backend="diffusers:test",
        )

    def generate_placeholder(
        self,
        output_dir: str,
        prompt: str,
        modality: str = "XR",
        body_region: str = "chest",
    ):
        raise AssertionError("Expected diffusers backend.")


class FakeTimeSeriesGenerator:
    def __init__(self):
        self.calls = []

    def add_time_series(self, record, channels=None, points=6):
        self.calls.append((record.record_id, channels, points))
        return record


class FakeTextGenerator:
    def __init__(self, provider=None):
        self.provider = provider
        self.calls = []

    async def add_documents_async(self, record):
        self.calls.append(record.record_id)
        return record


@pytest.mark.asyncio
async def test_synthetic_pipeline_uses_configured_diffusers_backend(tmp_path):
    imaging_generator = FakeImagingGenerator()
    pipeline = SyntheticPipeline(
        imaging_generator=imaging_generator,
        validator=SyntheticValidator(),
        image_output_dir=str(tmp_path),
        image_backend="diffusers",
    )

    result = await pipeline.generate(
        GenerationRequest(topic="pneumonia", count=1, modalities=[Modality.IMAGING])
    )

    assert result["records"][0].imaging[0].generation_backend == "diffusers:test"
    assert imaging_generator.diffusers_calls[0][0] == str(tmp_path)
    assert "right lower lobe opacity" in imaging_generator.diffusers_calls[0][1]
    assert imaging_generator.diffusers_calls[0][2:] == ("XR", "chest")


@pytest.mark.asyncio
async def test_synthetic_pipeline_allows_request_imaging_backend_override(tmp_path):
    imaging_generator = FakeImagingGenerator()
    pipeline = SyntheticPipeline(
        imaging_generator=imaging_generator,
        validator=SyntheticValidator(),
        image_output_dir=str(tmp_path),
        image_backend="placeholder",
    )

    result = await pipeline.generate(
        GenerationRequest(
            topic="pneumonia",
            count=1,
            modalities=[Modality.IMAGING],
            imaging_backend="diffusers",
        )
    )

    assert result["records"][0].imaging[0].generation_backend == "diffusers:test"
    assert imaging_generator.diffusers_calls[0][0] == str(tmp_path)


@pytest.mark.asyncio
async def test_synthetic_pipeline_uses_request_imaging_model_profile(monkeypatch, tmp_path):
    created = []

    class RequestScopedImagingGenerator(FakeImagingGenerator):
        def __init__(self, diffusers_model_id: str, imaging_model_profile: str):
            super().__init__()
            created.append((diffusers_model_id, imaging_model_profile))

    monkeypatch.setattr(
        "casecrawler.generation.synthetic_pipeline.ImagingGenerator",
        RequestScopedImagingGenerator,
    )
    pipeline = SyntheticPipeline(
        imaging_generator=FakeImagingGenerator(),
        validator=SyntheticValidator(),
        image_output_dir=str(tmp_path),
        image_backend="placeholder",
    )

    result = await pipeline.generate(
        GenerationRequest(
            topic="pneumonia",
            count=1,
            modalities=[Modality.IMAGING],
            imaging_backend="diffusers",
            imaging_model_profile="cxr_pneumonia_dreambooth",
            diffusers_model_id="hf/test-cxr",
        )
    )

    assert created == [("hf/test-cxr", "cxr_pneumonia_dreambooth")]
    assert result["records"][0].imaging[0].generation_backend == "diffusers:test"


@pytest.mark.asyncio
async def test_synthetic_pipeline_allows_request_time_series_backend_override(
    monkeypatch,
    tmp_path,
):
    created = []

    class RequestScopedTimeSeriesGenerator(FakeTimeSeriesGenerator):
        def __init__(self, external_command):
            super().__init__()
            created.append(external_command)

    monkeypatch.setattr(
        "casecrawler.generation.synthetic_pipeline.TimeSeriesGenerator",
        RequestScopedTimeSeriesGenerator,
    )
    pipeline = SyntheticPipeline(
        time_series_generator=FakeTimeSeriesGenerator(),
        validator=SyntheticValidator(),
        image_output_dir=str(tmp_path),
        image_backend="placeholder",
    )

    result = await pipeline.generate(
        GenerationRequest(
            topic="sepsis",
            count=1,
            modalities=[Modality.TIME_SERIES],
            time_series_backend="external",
            time_series_model_profile="timediff",
            time_series_command=["timediff-sample"],
        )
    )

    assert created == [["timediff-sample"]]
    assert result["generated"] == 1


@pytest.mark.asyncio
async def test_synthetic_pipeline_allows_request_clinical_text_backend_override(
    monkeypatch,
    tmp_path,
):
    created_providers = []
    created_generators = []

    class FakeProvider:
        pass

    class RequestScopedTextGenerator(FakeTextGenerator):
        def __init__(self, provider=None):
            super().__init__(provider=provider)
            created_generators.append(provider)

    def fake_get_provider(provider_name, model, **kwargs):
        created_providers.append((provider_name, model, kwargs))
        return FakeProvider()

    monkeypatch.setattr(
        "casecrawler.generation.synthetic_pipeline.TextGenerator",
        RequestScopedTextGenerator,
    )
    monkeypatch.setattr(
        "casecrawler.generation.synthetic_pipeline.get_provider",
        fake_get_provider,
    )
    pipeline = SyntheticPipeline(
        text_generator=FakeTextGenerator(),
        validator=SyntheticValidator(),
        image_output_dir=str(tmp_path),
        image_backend="placeholder",
    )

    result = await pipeline.generate(
        GenerationRequest(
            topic="sepsis",
            count=1,
            modalities=[Modality.CLINICAL_TEXT],
            clinical_text_backend="llm",
            llm_provider="ollama",
            llm_model="medgemma-local",
            ollama_base_url="http://localhost:11434",
        )
    )

    assert created_providers == [
        ("ollama", "medgemma-local", {"base_url": "http://localhost:11434"})
    ]
    assert len(created_generators) == 1
    assert result["generated"] == 1


@pytest.mark.asyncio
async def test_synthetic_pipeline_honors_request_validation_threshold(tmp_path):
    pipeline = SyntheticPipeline(
        image_output_dir=str(tmp_path),
        image_backend="placeholder",
    )

    relaxed = await pipeline.generate(
        GenerationRequest(
            topic="pneumonia",
            count=1,
            modalities=[Modality.IMAGING],
            validation_threshold=0.3,
        )
    )
    strict = await pipeline.generate(
        GenerationRequest(
            topic="pneumonia",
            count=1,
            modalities=[Modality.IMAGING],
            validation_threshold=0.8,
        )
    )

    assert relaxed["records"][0].validation.modality_alignment_score < 0.8
    assert relaxed["approved"] == 1
    assert strict["approved"] == 0
    assert any(
        issue.field == "imaging.alignment"
        for issue in strict["records"][0].validation.issues
    )


@pytest.mark.asyncio
async def test_synthetic_pipeline_keeps_unrequested_modalities_empty(tmp_path):
    pipeline = SyntheticPipeline(
        image_output_dir=str(tmp_path),
        image_backend="placeholder",
    )

    result = await pipeline.generate(
        GenerationRequest(
            topic="sepsis",
            count=1,
            modalities=[Modality.IMAGING],
            validation_threshold=0.3,
        )
    )
    record = result["records"][0]

    assert record.modalities == [Modality.IMAGING]
    assert record.labs == []
    assert record.vitals == []
    assert record.medication_history == []
    assert record.documents == []
    assert record.time_series == []
    assert record.imaging


@pytest.mark.asyncio
async def test_synthetic_pipeline_placeholder_imaging_uses_topic_aware_labels(tmp_path):
    pipeline = SyntheticPipeline(
        validator=SyntheticValidator(),
        image_output_dir=str(tmp_path),
        image_backend="placeholder",
    )

    result = await pipeline.generate(
        GenerationRequest(topic="heart failure", count=1, modalities=[Modality.IMAGING])
    )

    image = result["records"][0].imaging[0]

    assert "pulmonary edema" in image.prompt
    assert "Pulmonary edema" in image.report_text
    assert any(label.display == "Pulmonary edema" for label in image.labels)


@pytest.mark.asyncio
async def test_synthetic_pipeline_uses_topic_specific_imaging_modalities(tmp_path):
    pipeline = SyntheticPipeline(
        validator=SyntheticValidator(),
        image_output_dir=str(tmp_path),
        image_backend="placeholder",
    )

    stroke = await pipeline.generate(
        GenerationRequest(topic="ischemic stroke", count=1, modalities=[Modality.IMAGING])
    )
    pe = await pipeline.generate(
        GenerationRequest(topic="pulmonary embolism", count=1, modalities=[Modality.IMAGING])
    )
    aki = await pipeline.generate(
        GenerationRequest(topic="acute kidney injury", count=1, modalities=[Modality.IMAGING])
    )

    assert stroke["records"][0].imaging[0].modality == "CT"
    assert stroke["records"][0].imaging[0].body_region == "head"
    assert pe["records"][0].imaging[0].modality == "CTA"
    assert pe["records"][0].imaging[0].body_region == "chest"
    assert aki["records"][0].imaging[0].modality == "US"
    assert aki["records"][0].imaging[0].body_region == "abdomen"


@pytest.mark.asyncio
async def test_synthetic_pipeline_uses_expanded_profile_imaging_specs(tmp_path):
    pipeline = SyntheticPipeline(
        validator=SyntheticValidator(),
        image_output_dir=str(tmp_path),
        image_backend="placeholder",
    )

    asthma = await pipeline.generate(
        GenerationRequest(topic="status asthmaticus", count=1, modalities=[Modality.IMAGING])
    )
    pancreatitis = await pipeline.generate(
        GenerationRequest(topic="acute pancreatitis", count=1, modalities=[Modality.IMAGING])
    )
    appendicitis = await pipeline.generate(
        GenerationRequest(topic="appendicitis", count=1, modalities=[Modality.IMAGING])
    )
    pyelo = await pipeline.generate(
        GenerationRequest(topic="pyelonephritis", count=1, modalities=[Modality.IMAGING])
    )
    meningitis = await pipeline.generate(
        GenerationRequest(topic="bacterial meningitis", count=1, modalities=[Modality.IMAGING])
    )
    seizure = await pipeline.generate(
        GenerationRequest(topic="status epilepticus", count=1, modalities=[Modality.IMAGING])
    )

    asthma_image = asthma["records"][0].imaging[0]
    pancreatitis_image = pancreatitis["records"][0].imaging[0]
    appendicitis_image = appendicitis["records"][0].imaging[0]
    pyelo_image = pyelo["records"][0].imaging[0]
    meningitis_image = meningitis["records"][0].imaging[0]
    seizure_image = seizure["records"][0].imaging[0]

    assert (asthma_image.modality, asthma_image.body_region) == ("XR", "chest")
    assert any(label.display == "Hyperinflation" for label in asthma_image.labels)
    assert (pancreatitis_image.modality, pancreatitis_image.body_region) == ("CT", "abdomen")
    assert any(label.display == "Peripancreatic inflammation" for label in pancreatitis_image.labels)
    assert (appendicitis_image.modality, appendicitis_image.body_region) == ("CT", "abdomen")
    assert any(label.display == "Appendicitis" for label in appendicitis_image.labels)
    assert (pyelo_image.modality, pyelo_image.body_region) == ("CT", "abdomen")
    assert any(label.display == "Pyelonephritis" for label in pyelo_image.labels)
    assert (meningitis_image.modality, meningitis_image.body_region) == ("CT", "head")
    assert "no acute hemorrhage" in meningitis_image.prompt
    assert (seizure_image.modality, seizure_image.body_region) == ("CT", "head")
    assert "postictal" in seizure_image.prompt


@pytest.mark.asyncio
async def test_synthetic_pipeline_clinical_text_radiology_report_uses_generated_imaging(tmp_path):
    pipeline = SyntheticPipeline(
        validator=SyntheticValidator(),
        image_output_dir=str(tmp_path),
        image_backend="placeholder",
    )

    result = await pipeline.generate(
        GenerationRequest(
            topic="acute pancreatitis",
            count=1,
            modalities=[Modality.CLINICAL_TEXT, Modality.IMAGING],
        )
    )
    record = result["records"][0]
    radiology_report = next(
        document for document in record.documents if document.note_type == "radiology_report"
    )

    assert record.imaging[0].modality == "CT"
    assert record.imaging[0].body_region == "abdomen"
    assert "CT abdomen" in radiology_report.clean_text
    assert "Peripancreatic inflammation" in radiology_report.clean_text
    assert record.imaging[0].image_id in radiology_report.extracted_facts["imaging_asset_ids"]


@pytest.mark.asyncio
async def test_synthetic_pipeline_rejects_unknown_image_backend(tmp_path):
    pipeline = SyntheticPipeline(
        validator=SyntheticValidator(),
        image_output_dir=str(tmp_path),
        image_backend="unknown",
    )

    with pytest.raises(ValueError, match="Unknown synthetic imaging backend"):
        await pipeline.generate(
            GenerationRequest(topic="pneumonia", count=1, modalities=[Modality.IMAGING])
        )
