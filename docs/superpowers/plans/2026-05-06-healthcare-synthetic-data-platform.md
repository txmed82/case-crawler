# Healthcare Synthetic Data Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Convert CaseCrawler from a medical education case generator into a multimodal synthetic healthcare data generator for AI fine-tuning and evaluation.

**Architecture:** Keep the durable backend pieces: source ingestion, Chroma retrieval, LLM provider abstraction, Pydantic structured generation, review/retry, SQLite storage, CLI, API, and JSONL export. Replace education-first concepts with dataset-first generation: patient timelines, clinical notes, labs, vitals, waveforms/time series, imaging assets, validation reports, provenance, and fine-tuning export profiles. External models and tools are wrapped behind generator and validator interfaces so the platform can use hosted APIs, Hugging Face models, local diffusers, Synthea, and future EHR time-series models without hard-coding one vendor.

**Tech Stack:** Python 3.12, Pydantic v2, Click, FastAPI, SQLite, ChromaDB, sentence-transformers, optional Hugging Face `transformers`/`diffusers`/`datasets`, existing OpenAI/Anthropic/OpenRouter/Ollama providers, React/Vite frontend.

---

## Research Notes

Use these as implementation anchors:

- Synthea is the best open-source baseline for full synthetic patient histories and standards-based output. It exports FHIR, C-CDA, and CSV and models patients from birth through present using disease modules.
- SYNDERAI is a useful reference for FHIR-first synthetic lab reports, discharge reports, patient summaries, privacy principles, and clinically consistent interdependencies.
- TimeDiff is a relevant open-source reference for mixed-type EHR time-series generation with utility and privacy evaluation code.
- RawMed and similar 2025 work are relevant for future multi-table time-series EHR generation, but the repo should begin with schema/rule/LLM generation and pluggable model hooks.
- Asclepius Synthetic Clinical Notes is useful as a reference dataset for clinical note and instruction formats, not as the core data model.
- NHSE synthetic clinical notes is a newer reference for a no-real-data synthetic notes pipeline.
- MedGemma can be used as an optional medical text/image reasoning validator or generator when its gated license is acceptable.
- BiomedCLIP is a practical open medical image-text alignment validator for generated medical images and captions.
- Medical image generation should start narrow. Chest X-ray diffusion models and CheXGenBench/SynthCheX-style evaluation are a reasonable first slice. Seg2med-style CT/MR generation is promising but should be treated as a later specialized adapter because it depends on anatomical masks/phantoms.

## What Exists Now

### Keep

| Path | Why it stays |
|---|---|
| `src/casecrawler/sources/` | Good source-adapter shape for PubMed, OpenFDA, DailyMed, RxNorm, medRxiv, ClinicalTrials, Glass, Anna's Archive, and Firecrawl. |
| `src/casecrawler/pipeline/` | Normalization, chunking, embeddings, tagging, Chroma storage, and retrieval remain useful for grounding synthetic data. |
| `src/casecrawler/llm/` | Existing provider abstraction supports hosted and local models. |
| `src/casecrawler/generation/multi_step_pipeline.py` | Plan-render-review orchestration is the right pattern, but the output object must become dataset records rather than education cases. |
| `src/casecrawler/generation/lab_panels.py` | Useful starting reference ranges for lab validation. |
| `src/casecrawler/models/diagnostics.py` | Useful seed for labs, imaging findings, and vitals, but too shallow for full dataset generation. |
| `src/casecrawler/export/` | Exporter pattern is correct. Output profiles need expansion. |
| `src/casecrawler/api/` and `src/casecrawler/cli.py` | Keep both. Rename commands and API resources around datasets/jobs/exports. |

### Remove Or Demote

| Current concept | Replacement |
|---|---|
| `DifficultyLevel` values `medical_student`, `resident`, `attending` | `ComplexityProfile`: `simple`, `moderate`, `complex`, `rare`, plus configurable cohort mix. |
| `pedagogy_score` | `utility_score`, `schema_score`, `clinical_consistency_score`, `privacy_score`, `modality_alignment_score`. |
| Case player UI | Dataset workbench: generation jobs, schema preview, validation reports, exports. |
| `decision_tree` as primary data | Optional training task view derived from patient timeline. |
| `vignette` as primary data | Multiple documents: ED note, progress note, discharge summary, radiology report, lab report, nursing note, messy OCR/message variants. |
| `case_id` | `record_id` / `patient_id` / `dataset_id`, with stable provenance. |

## Target Data Model

Create a dataset-first schema in `src/casecrawler/models/synthetic.py`:

```python
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class ComplexityProfile(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    RARE = "rare"


class Modality(str, Enum):
    STRUCTURED_EHR = "structured_ehr"
    CLINICAL_TEXT = "clinical_text"
    LABS = "labs"
    VITALS = "vitals"
    TIME_SERIES = "time_series"
    IMAGING = "imaging"


class Code(BaseModel):
    system: str
    code: str
    display: str


class Provenance(BaseModel):
    generator: str
    model: str | None = None
    source_refs: list[dict] = Field(default_factory=list)
    prompt_hash: str | None = None
    created_at: str


class SyntheticPatient(BaseModel):
    patient_id: str
    age: int
    sex: str
    demographics: dict = Field(default_factory=dict)
    social_history: dict = Field(default_factory=dict)


class Encounter(BaseModel):
    encounter_id: str
    start: str
    end: str | None = None
    setting: str
    reason: str
    diagnoses: list[Code] = Field(default_factory=list)
    procedures: list[Code] = Field(default_factory=list)


class LabObservation(BaseModel):
    name: str
    loinc: str | None = None
    value: float | str
    unit: str
    reference_low: float | None = None
    reference_high: float | None = None
    flag: str | None = None
    effective_time: str
    specimen: str | None = None


class VitalObservation(BaseModel):
    name: str
    value: float
    unit: str
    effective_time: str


class TimeSeriesPoint(BaseModel):
    timestamp: str
    values: dict[str, float]


class TimeSeriesChannel(BaseModel):
    name: str
    unit: str
    sampling_rate_hz: float | None = None
    points: list[TimeSeriesPoint]


class ClinicalDocument(BaseModel):
    document_id: str
    note_type: str
    author_role: str
    timestamp: str
    clean_text: str
    messy_text: str | None = None
    extracted_facts: dict = Field(default_factory=dict)


class ImagingAsset(BaseModel):
    image_id: str
    modality: str
    body_region: str
    prompt: str
    file_path: str | None = None
    report_text: str
    labels: list[Code] = Field(default_factory=list)
    generation_backend: str


class ValidationIssue(BaseModel):
    severity: str
    modality: Modality
    field: str
    message: str


class ValidationReport(BaseModel):
    schema_score: float
    clinical_consistency_score: float
    privacy_score: float
    utility_score: float
    modality_alignment_score: float | None = None
    approved: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class SyntheticRecord(BaseModel):
    record_id: str
    dataset_id: str
    topic: str
    complexity: ComplexityProfile
    modalities: list[Modality]
    patient: SyntheticPatient
    encounters: list[Encounter]
    labs: list[LabObservation] = Field(default_factory=list)
    vitals: list[VitalObservation] = Field(default_factory=list)
    time_series: list[TimeSeriesChannel] = Field(default_factory=list)
    documents: list[ClinicalDocument] = Field(default_factory=list)
    imaging: list[ImagingAsset] = Field(default_factory=list)
    provenance: Provenance
    validation: ValidationReport | None = None
    metadata: dict = Field(default_factory=dict)
```

## File Structure

### New Files

| File | Responsibility |
|---|---|
| `src/casecrawler/models/synthetic.py` | Dataset-first multimodal record schema. |
| `src/casecrawler/models/dataset.py` | Generation request, dataset manifest, export manifest. |
| `src/casecrawler/generation/synthetic_pipeline.py` | Orchestrates plan -> modality generation -> validation -> storage. |
| `src/casecrawler/generation/modality_plan.py` | Builds a grounded plan for requested modalities and cohort composition. |
| `src/casecrawler/generation/text_generator.py` | Generates clean and messy clinical notes from structured facts. |
| `src/casecrawler/generation/structured_generator.py` | Generates patient, encounters, diagnoses, medications, procedures, vitals, labs. |
| `src/casecrawler/generation/timeseries_generator.py` | Generates longitudinal vitals/labs/waveform-like time series. |
| `src/casecrawler/generation/imaging_generator.py` | Dispatches image generation backends and report generation. |
| `src/casecrawler/validation/synthetic_validator.py` | Composite validator for schema, clinical consistency, privacy, and utility. |
| `src/casecrawler/validation/clinical_rules.py` | Deterministic physiology, lab range, temporal, and contradiction checks. |
| `src/casecrawler/validation/privacy.py` | PHI-like string and memorization-risk checks. |
| `src/casecrawler/validation/image_alignment.py` | Optional BiomedCLIP/MedGemma image-text validation adapter. |
| `src/casecrawler/storage/dataset_store.py` | SQLite storage for datasets, records, validation reports, and export manifests. |
| `src/casecrawler/export/fine_tuning.py` | SFT, chat, tool-call, multimodal, RL, DPO-style JSONL exporters. |
| `src/casecrawler/integrations/huggingface.py` | Optional Hugging Face model/dataset helpers. |
| `src/casecrawler/integrations/synthea.py` | Optional Synthea subprocess/import adapter. |
| `src/casecrawler/api/routes/datasets.py` | Dataset CRUD, generation jobs, export endpoints. |
| `tests/test_synthetic_models.py` | Schema tests. |
| `tests/test_synthetic_validator.py` | Deterministic validation tests. |
| `tests/test_synthetic_pipeline.py` | Pipeline orchestration tests with fake generators. |
| `tests/test_fine_tuning_export.py` | Export profile tests. |

### Existing Files To Modify

| File | Changes |
|---|---|
| `README.md` | Reposition as healthcare synthetic data generator; remove medical education as a primary use case. |
| `pyproject.toml` | Add optional extras for `hf`, `imaging`, `synthea`, and `parquet`. |
| `config.example.yaml` | Add generation backends, validation thresholds, export profiles, and modality defaults. |
| `src/casecrawler/cli.py` | Add `generate-dataset`, `datasets`, `validate`, and `export` commands; keep old `generate` temporarily as legacy. |
| `src/casecrawler/api/app.py` | Register dataset routes. |
| `ui/src/App.tsx` | Replace case player navigation with dataset workbench navigation. |
| `src/casecrawler/generation/prompts.py` | Add dataset planning, document generation, and validation prompts; remove pedagogy language from new path. |

---

## Task 1: Introduce Synthetic Record Models

**Files:**
- Create: `src/casecrawler/models/synthetic.py`
- Test: `tests/test_synthetic_models.py`

- [x] **Step 1: Write failing schema tests**

```python
from casecrawler.models.synthetic import (
    ClinicalDocument,
    Code,
    ComplexityProfile,
    Encounter,
    LabObservation,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    ValidationReport,
    VitalObservation,
)


def test_synthetic_record_with_text_labs_and_vitals():
    record = SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="heart failure exacerbation",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.STRUCTURED_EHR, Modality.CLINICAL_TEXT, Modality.LABS, Modality.VITALS],
        patient=SyntheticPatient(patient_id="pat-1", age=72, sex="female"),
        encounters=[
            Encounter(
                encounter_id="enc-1",
                start="2026-05-06T08:00:00",
                setting="emergency_department",
                reason="dyspnea",
                diagnoses=[Code(system="ICD-10-CM", code="I50.9", display="Heart failure, unspecified")],
            )
        ],
        labs=[
            LabObservation(
                name="BNP",
                loinc="30934-4",
                value=1220.0,
                unit="pg/mL",
                reference_low=0.0,
                reference_high=100.0,
                flag="H",
                effective_time="2026-05-06T08:45:00",
            )
        ],
        vitals=[
            VitalObservation(name="SpO2", value=89.0, unit="%", effective_time="2026-05-06T08:05:00")
        ],
        documents=[
            ClinicalDocument(
                document_id="doc-1",
                note_type="ed_note",
                author_role="physician",
                timestamp="2026-05-06T09:15:00",
                clean_text="Patient presents with progressive dyspnea and edema.",
                messy_text="pt w/ prog dyspnea + edema, BNP 1220",
            )
        ],
        provenance=Provenance(generator="unit-test", created_at="2026-05-06T09:30:00"),
        validation=ValidationReport(
            schema_score=1.0,
            clinical_consistency_score=0.95,
            privacy_score=1.0,
            utility_score=0.9,
            approved=True,
        ),
    )

    assert record.record_id == "rec-1"
    assert record.complexity == ComplexityProfile.MODERATE
    assert record.labs[0].flag == "H"
    assert record.validation.approved is True
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_synthetic_models.py -v
```

Expected: fail with `ModuleNotFoundError: No module named 'casecrawler.models.synthetic'`.

- [x] **Step 3: Implement `src/casecrawler/models/synthetic.py`**

Use the complete target data model from the `Target Data Model` section.

- [x] **Step 4: Run the test to verify it passes**

Run:

```bash
python -m pytest tests/test_synthetic_models.py -v
```

Expected: pass.

---

## Task 2: Add Dataset Request And Manifest Models

**Files:**
- Create: `src/casecrawler/models/dataset.py`
- Test: `tests/test_dataset_models.py`

- [x] **Step 1: Write failing tests**

```python
from casecrawler.models.dataset import DatasetManifest, ExportFormat, GenerationRequest
from casecrawler.models.synthetic import ComplexityProfile, Modality


def test_generation_request_defaults():
    req = GenerationRequest(topic="sepsis", count=25)
    assert req.count == 25
    assert req.complexity == ComplexityProfile.MODERATE
    assert Modality.CLINICAL_TEXT in req.modalities
    assert Modality.LABS in req.modalities


def test_dataset_manifest_records_validation_summary():
    manifest = DatasetManifest(
        dataset_id="ds-1",
        name="sepsis-multimodal-v1",
        topic="sepsis",
        requested_count=100,
        generated_count=97,
        approved_count=91,
        modalities=[Modality.CLINICAL_TEXT, Modality.LABS],
        export_formats=[ExportFormat.SFT_JSONL, ExportFormat.PARQUET],
        created_at="2026-05-06T12:00:00",
    )
    assert manifest.approved_count == 91
    assert ExportFormat.SFT_JSONL in manifest.export_formats
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_dataset_models.py -v
```

Expected: fail with `ModuleNotFoundError`.

- [x] **Step 3: Implement dataset models**

```python
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field

from casecrawler.models.synthetic import ComplexityProfile, Modality


class ExportFormat(str, Enum):
    RAW_JSONL = "raw_jsonl"
    SFT_JSONL = "sft_jsonl"
    CHAT_JSONL = "chat_jsonl"
    MULTIMODAL_JSONL = "multimodal_jsonl"
    RL_JSONL = "rl_jsonl"
    FHIR_NDJSON = "fhir_ndjson"
    PARQUET = "parquet"


class GenerationRequest(BaseModel):
    topic: str
    count: int = Field(default=1, ge=1)
    complexity: ComplexityProfile = ComplexityProfile.MODERATE
    modalities: list[Modality] = Field(
        default_factory=lambda: [
            Modality.STRUCTURED_EHR,
            Modality.CLINICAL_TEXT,
            Modality.LABS,
            Modality.VITALS,
        ]
    )
    cohort_constraints: dict = Field(default_factory=dict)
    export_formats: list[ExportFormat] = Field(default_factory=lambda: [ExportFormat.SFT_JSONL])
    ingest_first: bool = False
    validation_threshold: float = 0.8


class DatasetManifest(BaseModel):
    dataset_id: str
    name: str
    topic: str
    requested_count: int
    generated_count: int
    approved_count: int
    modalities: list[Modality]
    export_formats: list[ExportFormat]
    created_at: str
    metadata: dict = Field(default_factory=dict)
```

- [x] **Step 4: Run the test to verify it passes**

Run:

```bash
python -m pytest tests/test_dataset_models.py -v
```

Expected: pass.

---

## Task 3: Build Deterministic Clinical Validators

**Files:**
- Create: `src/casecrawler/validation/__init__.py`
- Create: `src/casecrawler/validation/clinical_rules.py`
- Create: `src/casecrawler/validation/privacy.py`
- Create: `src/casecrawler/validation/synthetic_validator.py`
- Test: `tests/test_synthetic_validator.py`

- [x] **Step 1: Write failing validator tests**

```python
from casecrawler.models.synthetic import (
    ComplexityProfile,
    LabObservation,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    ValidationIssue,
    VitalObservation,
)
from casecrawler.validation.synthetic_validator import SyntheticValidator


def _record(**overrides):
    data = {
        "record_id": "rec-1",
        "dataset_id": "ds-1",
        "topic": "sepsis",
        "complexity": ComplexityProfile.MODERATE,
        "modalities": [Modality.LABS, Modality.VITALS],
        "patient": SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        "encounters": [],
        "labs": [
            LabObservation(
                name="Lactate",
                value=4.8,
                unit="mmol/L",
                reference_low=0.5,
                reference_high=2.0,
                flag="critical",
                effective_time="2026-05-06T08:30:00",
            )
        ],
        "vitals": [
            VitalObservation(name="HR", value=118, unit="/min", effective_time="2026-05-06T08:00:00")
        ],
        "provenance": Provenance(generator="unit-test", created_at="2026-05-06T09:00:00"),
    }
    data.update(overrides)
    return SyntheticRecord(**data)


def test_validator_approves_plausible_record():
    report = SyntheticValidator().validate(_record())
    assert report.approved is True
    assert report.clinical_consistency_score >= 0.8


def test_validator_rejects_missing_lab_flag():
    bad = _record(labs=[
        LabObservation(
            name="Lactate",
            value=4.8,
            unit="mmol/L",
            reference_low=0.5,
            reference_high=2.0,
            flag=None,
            effective_time="2026-05-06T08:30:00",
        )
    ])
    report = SyntheticValidator().validate(bad)
    assert report.approved is False
    assert any(issue.field == "labs.flag" for issue in report.issues)


def test_validator_rejects_phi_like_text():
    bad = _record(metadata={"free_text": "Call patient at 555-123-4567 tomorrow."})
    report = SyntheticValidator().validate(bad)
    assert report.approved is False
    assert any(issue.field == "privacy" for issue in report.issues)
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_synthetic_validator.py -v
```

Expected: fail with missing validation modules.

- [x] **Step 3: Implement deterministic checks**

`src/casecrawler/validation/clinical_rules.py`:

```python
from __future__ import annotations

from casecrawler.models.synthetic import Modality, SyntheticRecord, ValidationIssue


def validate_lab_flags(record: SyntheticRecord) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for lab in record.labs:
        if isinstance(lab.value, (int, float)) and lab.reference_low is not None and lab.reference_high is not None:
            outside = lab.value < lab.reference_low or lab.value > lab.reference_high
            if outside and not lab.flag:
                issues.append(ValidationIssue(
                    severity="error",
                    modality=Modality.LABS,
                    field="labs.flag",
                    message=f"{lab.name} is outside reference range but has no flag.",
                ))
    return issues


def validate_vitals(record: SyntheticRecord) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for vital in record.vitals:
        if vital.name == "SpO2" and not 0 <= vital.value <= 100:
            issues.append(ValidationIssue(
                severity="error",
                modality=Modality.VITALS,
                field="vitals.SpO2",
                message="SpO2 must be between 0 and 100.",
            ))
        if vital.name == "HR" and not 0 < vital.value < 260:
            issues.append(ValidationIssue(
                severity="error",
                modality=Modality.VITALS,
                field="vitals.HR",
                message="Heart rate is outside a plausible clinical range.",
            ))
    return issues
```

`src/casecrawler/validation/privacy.py`:

```python
from __future__ import annotations

import re

from casecrawler.models.synthetic import Modality, SyntheticRecord, ValidationIssue

PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def _text_blobs(record: SyntheticRecord) -> list[str]:
    blobs = [str(record.metadata)]
    for document in record.documents:
        blobs.append(document.clean_text)
        if document.messy_text:
            blobs.append(document.messy_text)
    return blobs


def validate_privacy(record: SyntheticRecord) -> list[ValidationIssue]:
    text = "\n".join(_text_blobs(record))
    issues: list[ValidationIssue] = []
    for regex, label in [(PHONE_RE, "phone number"), (SSN_RE, "SSN"), (EMAIL_RE, "email")]:
        if regex.search(text):
            issues.append(ValidationIssue(
                severity="error",
                modality=Modality.CLINICAL_TEXT,
                field="privacy",
                message=f"Potential PHI-like {label} detected.",
            ))
    return issues
```

`src/casecrawler/validation/synthetic_validator.py`:

```python
from __future__ import annotations

from casecrawler.models.synthetic import SyntheticRecord, ValidationReport
from casecrawler.validation.clinical_rules import validate_lab_flags, validate_vitals
from casecrawler.validation.privacy import validate_privacy


class SyntheticValidator:
    def __init__(self, threshold: float = 0.8) -> None:
        self._threshold = threshold

    def validate(self, record: SyntheticRecord) -> ValidationReport:
        issues = []
        issues.extend(validate_lab_flags(record))
        issues.extend(validate_vitals(record))
        issues.extend(validate_privacy(record))

        error_count = sum(1 for issue in issues if issue.severity == "error")
        schema_score = 1.0
        clinical_score = max(0.0, 1.0 - 0.25 * error_count)
        privacy_score = 0.0 if any(issue.field == "privacy" for issue in issues) else 1.0
        utility_score = 1.0 if record.documents or record.labs or record.vitals or record.time_series or record.imaging else 0.0
        approved = (
            schema_score >= self._threshold
            and clinical_score >= self._threshold
            and privacy_score >= self._threshold
            and utility_score >= self._threshold
            and not issues
        )
        return ValidationReport(
            schema_score=schema_score,
            clinical_consistency_score=clinical_score,
            privacy_score=privacy_score,
            utility_score=utility_score,
            modality_alignment_score=None,
            approved=approved,
            issues=issues,
        )
```

- [x] **Step 4: Run validator tests**

Run:

```bash
python -m pytest tests/test_synthetic_validator.py -v
```

Expected: pass.

---

## Task 4: Add Dataset Storage

**Files:**
- Create: `src/casecrawler/storage/dataset_store.py`
- Test: `tests/test_dataset_store.py`

- [x] **Step 1: Write storage tests**

```python
from casecrawler.models.synthetic import ComplexityProfile, Modality, Provenance, SyntheticPatient, SyntheticRecord
from casecrawler.storage.dataset_store import DatasetStore


def test_dataset_store_round_trips_record(tmp_path):
    store = DatasetStore(db_path=str(tmp_path / "datasets.db"))
    record = SyntheticRecord(
        record_id="rec-1",
        dataset_id="ds-1",
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.CLINICAL_TEXT],
        patient=SyntheticPatient(patient_id="pat-1", age=64, sex="male"),
        encounters=[],
        provenance=Provenance(generator="unit-test", created_at="2026-05-06T10:00:00"),
    )

    store.save_record(record)

    assert store.get_record("rec-1").record_id == "rec-1"
    assert len(store.list_records(dataset_id="ds-1")) == 1
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_dataset_store.py -v
```

Expected: fail with missing `DatasetStore`.

- [x] **Step 3: Implement `DatasetStore`**

```python
from __future__ import annotations

import sqlite3

from casecrawler.models.synthetic import SyntheticRecord


class DatasetStore:
    def __init__(self, db_path: str = "./data/datasets.db") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS synthetic_records (
                record_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                complexity TEXT NOT NULL,
                approved INTEGER,
                record_json TEXT NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_synth_dataset ON synthetic_records(dataset_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_synth_topic ON synthetic_records(topic)")
        self._conn.commit()

    def save_record(self, record: SyntheticRecord) -> None:
        approved = None if record.validation is None else int(record.validation.approved)
        self._conn.execute(
            """INSERT OR REPLACE INTO synthetic_records
            (record_id, dataset_id, topic, complexity, approved, record_json)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                record.record_id,
                record.dataset_id,
                record.topic,
                record.complexity.value,
                approved,
                record.model_dump_json(),
            ),
        )
        self._conn.commit()

    def get_record(self, record_id: str) -> SyntheticRecord | None:
        row = self._conn.execute(
            "SELECT record_json FROM synthetic_records WHERE record_id = ?",
            (record_id,),
        ).fetchone()
        if row is None:
            return None
        return SyntheticRecord.model_validate_json(row["record_json"])

    def list_records(self, dataset_id: str | None = None, topic: str | None = None, approved: bool | None = None, limit: int = 1000) -> list[SyntheticRecord]:
        query = "SELECT record_json FROM synthetic_records WHERE 1=1"
        params: list = []
        if dataset_id:
            query += " AND dataset_id = ?"
            params.append(dataset_id)
        if topic:
            query += " AND topic = ?"
            params.append(topic)
        if approved is not None:
            query += " AND approved = ?"
            params.append(int(approved))
        query += " LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [SyntheticRecord.model_validate_json(row["record_json"]) for row in rows]
```

- [x] **Step 4: Run storage tests**

Run:

```bash
python -m pytest tests/test_dataset_store.py -v
```

Expected: pass.

---

## Task 5: Build The First Synthetic Pipeline Slice

**Files:**
- Create: `src/casecrawler/generation/structured_generator.py`
- Create: `src/casecrawler/generation/text_generator.py`
- Create: `src/casecrawler/generation/synthetic_pipeline.py`
- Test: `tests/test_synthetic_pipeline.py`

- [x] **Step 1: Write orchestration tests with fake generation**

```python
import pytest

from casecrawler.generation.synthetic_pipeline import SyntheticPipeline
from casecrawler.models.dataset import GenerationRequest
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
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_synthetic_pipeline.py -v
```

Expected: fail with missing `SyntheticPipeline`.

- [x] **Step 3: Implement a deterministic first slice**

This first implementation should be deliberately boring: it proves the new record shape, validation, and storage path without requiring API keys.

`src/casecrawler/generation/structured_generator.py`:

```python
from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import (
    Code,
    Encounter,
    LabObservation,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    VitalObservation,
)


class StructuredGenerator:
    def generate(self, dataset_id: str, req: GenerationRequest, index: int) -> SyntheticRecord:
        now = datetime.now().isoformat()
        patient = SyntheticPatient(
            patient_id=f"pat-{uuid4()}",
            age=45 + (index % 35),
            sex="female" if index % 2 else "male",
        )
        encounter = Encounter(
            encounter_id=f"enc-{uuid4()}",
            start=now,
            setting="emergency_department",
            reason=req.topic,
            diagnoses=[Code(system="synthetic", code=req.topic.replace(" ", "_"), display=req.topic)],
        )
        return SyntheticRecord(
            record_id=f"rec-{uuid4()}",
            dataset_id=dataset_id,
            topic=req.topic,
            complexity=req.complexity,
            modalities=req.modalities,
            patient=patient,
            encounters=[encounter],
            labs=[
                LabObservation(
                    name="WBC",
                    value=15.2,
                    unit="K/uL",
                    reference_low=4.5,
                    reference_high=11.0,
                    flag="H",
                    effective_time=now,
                ),
                LabObservation(
                    name="Lactate",
                    value=3.4,
                    unit="mmol/L",
                    reference_low=0.5,
                    reference_high=2.0,
                    flag="H",
                    effective_time=now,
                ),
            ],
            vitals=[
                VitalObservation(name="HR", value=112, unit="/min", effective_time=now),
                VitalObservation(name="SBP", value=92, unit="mmHg", effective_time=now),
                VitalObservation(name="SpO2", value=94, unit="%", effective_time=now),
            ],
            provenance=Provenance(generator="structured-generator", created_at=now),
        )
```

`src/casecrawler/generation/text_generator.py`:

```python
from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from casecrawler.models.synthetic import ClinicalDocument, SyntheticRecord


class TextGenerator:
    def add_documents(self, record: SyntheticRecord) -> SyntheticRecord:
        timestamp = datetime.now().isoformat()
        labs = ", ".join(f"{lab.name} {lab.value} {lab.unit}" for lab in record.labs)
        vitals = ", ".join(f"{vital.name} {vital.value}{vital.unit}" for vital in record.vitals)
        clean = (
            f"{record.patient.age}-year-old {record.patient.sex} patient presents with {record.topic}. "
            f"Initial vitals: {vitals}. Initial labs: {labs}. "
            "Assessment and plan document a synthetic but clinically plausible presentation."
        )
        messy = clean.replace("patient", "pt").replace("with", "w/").replace("Initial", "Init")
        document = ClinicalDocument(
            document_id=f"doc-{uuid4()}",
            note_type="ed_note",
            author_role="physician",
            timestamp=timestamp,
            clean_text=clean,
            messy_text=messy,
        )
        return record.model_copy(update={"documents": [*record.documents, document]})
```

`src/casecrawler/generation/synthetic_pipeline.py`:

```python
from __future__ import annotations

from uuid import uuid4

from casecrawler.generation.structured_generator import StructuredGenerator
from casecrawler.generation.text_generator import TextGenerator
from casecrawler.models.dataset import GenerationRequest
from casecrawler.validation.synthetic_validator import SyntheticValidator


class SyntheticPipeline:
    def __init__(
        self,
        structured_generator: StructuredGenerator | None = None,
        text_generator: TextGenerator | None = None,
        validator: SyntheticValidator | None = None,
    ) -> None:
        self._structured_generator = structured_generator or StructuredGenerator()
        self._text_generator = text_generator or TextGenerator()
        self._validator = validator or SyntheticValidator()

    async def generate(self, req: GenerationRequest) -> dict:
        dataset_id = f"ds-{uuid4()}"
        records = []
        approved = 0
        for index in range(req.count):
            record = self._structured_generator.generate(dataset_id=dataset_id, req=req, index=index)
            record = self._text_generator.add_documents(record)
            validation = self._validator.validate(record)
            record = record.model_copy(update={"validation": validation})
            records.append(record)
            if validation.approved:
                approved += 1
        return {
            "dataset_id": dataset_id,
            "generated": len(records),
            "approved": approved,
            "records": records,
        }
```

- [x] **Step 4: Run pipeline tests**

Run:

```bash
python -m pytest tests/test_synthetic_pipeline.py -v
```

Expected: pass.

---

## Task 6: Add Fine-Tuning Export Profiles

**Files:**
- Create: `src/casecrawler/export/fine_tuning.py`
- Test: `tests/test_fine_tuning_export.py`

- [x] **Step 1: Write export tests**

```python
from casecrawler.export.fine_tuning import export_sft_record
from casecrawler.models.synthetic import ClinicalDocument, ComplexityProfile, Modality, Provenance, SyntheticPatient, SyntheticRecord


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
        provenance=Provenance(generator="unit-test", created_at="2026-05-06T10:00:00"),
    )

    exported = export_sft_record(record, task="summarize")

    assert exported["record_id"] == "rec-1"
    assert exported["messages"][0]["role"] == "system"
    assert exported["messages"][1]["role"] == "user"
    assert exported["messages"][2]["role"] == "assistant"
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_fine_tuning_export.py -v
```

Expected: fail with missing module.

- [x] **Step 3: Implement SFT export**

```python
from __future__ import annotations

from casecrawler.models.synthetic import SyntheticRecord


def export_sft_record(record: SyntheticRecord, task: str = "summarize") -> dict:
    note_text = "\n\n".join(document.clean_text for document in record.documents)
    if task == "summarize":
        user = f"Summarize the following synthetic clinical record:\n\n{note_text}"
        assistant = f"Synthetic patient with {record.topic}; structured data includes {len(record.labs)} labs and {len(record.vitals)} vitals."
    elif task == "extract":
        user = f"Extract diagnoses, abnormal labs, and vital sign abnormalities from this synthetic note:\n\n{note_text}"
        assistant = {
            "topic": record.topic,
            "labs": [lab.model_dump() for lab in record.labs],
            "vitals": [vital.model_dump() for vital in record.vitals],
        }
    else:
        raise ValueError(f"Unknown SFT task: {task}")
    return {
        "record_id": record.record_id,
        "dataset_id": record.dataset_id,
        "task": task,
        "messages": [
            {"role": "system", "content": "You are a clinical AI assistant trained on synthetic healthcare data."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant if isinstance(assistant, str) else str(assistant)},
        ],
        "metadata": {
            "topic": record.topic,
            "complexity": record.complexity.value,
            "modalities": [m.value for m in record.modalities],
            "synthetic": True,
        },
    }
```

- [x] **Step 4: Run export tests**

Run:

```bash
python -m pytest tests/test_fine_tuning_export.py -v
```

Expected: pass.

---

## Task 7: Wire CLI Commands

**Files:**
- Modify: `src/casecrawler/cli.py`
- Test: `tests/test_cli_synthetic.py`

- [x] **Step 1: Write CLI tests**

```python
from click.testing import CliRunner

from casecrawler.cli import cli


def test_generate_dataset_command_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["generate-dataset", "sepsis", "--count", "1"])

    assert result.exit_code == 0
    assert "Generated: 1" in result.output
    assert "Approved: 1" in result.output
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_cli_synthetic.py -v
```

Expected: fail because `generate-dataset` does not exist.

- [x] **Step 3: Add `generate-dataset` command**

Add imports inside the function to avoid slowing legacy commands:

```python
@cli.command("generate-dataset")
@click.argument("topic")
@click.option("--count", default=1, type=int, help="Number of synthetic records to generate")
@click.option("--complexity", default="moderate", help="simple, moderate, complex, or rare")
def generate_dataset(topic: str, count: int, complexity: str) -> None:
    """Generate synthetic healthcare records for AI training."""
    from casecrawler.generation.synthetic_pipeline import SyntheticPipeline
    from casecrawler.models.dataset import GenerationRequest
    from casecrawler.models.synthetic import ComplexityProfile
    from casecrawler.storage.dataset_store import DatasetStore

    req = GenerationRequest(topic=topic, count=count, complexity=ComplexityProfile(complexity))
    result = asyncio.run(SyntheticPipeline().generate(req))
    store = DatasetStore()
    for record in result["records"]:
        store.save_record(record)
    click.echo(f"Dataset: {result['dataset_id']}")
    click.echo(f"Generated: {result['generated']}")
    click.echo(f"Approved: {result['approved']}")
```

- [x] **Step 4: Run CLI tests**

Run:

```bash
python -m pytest tests/test_cli_synthetic.py -v
```

Expected: pass.

---

## Task 8: Add API Dataset Route

**Files:**
- Create: `src/casecrawler/api/routes/datasets.py`
- Modify: `src/casecrawler/api/app.py`
- Test: `tests/test_api_datasets.py`

- [x] **Step 1: Write API tests**

```python
from fastapi.testclient import TestClient

from casecrawler.api.app import app


def test_generate_dataset_api_smoke():
    client = TestClient(app)
    response = client.post("/api/datasets/generate", json={"topic": "sepsis", "count": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["generated"] == 1
    assert body["approved"] == 1
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_api_datasets.py -v
```

Expected: fail with 404.

- [x] **Step 3: Implement route**

```python
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
```

In `src/casecrawler/api/app.py`, include:

```python
from casecrawler.api.routes import datasets

app.include_router(datasets.router, prefix="/api")
```

- [x] **Step 4: Run API tests**

Run:

```bash
python -m pytest tests/test_api_datasets.py -v
```

Expected: pass.

---

## Task 9: Integrate Optional Model Backends

**Files:**
- Create: `src/casecrawler/integrations/huggingface.py`
- Create: `src/casecrawler/generation/imaging_generator.py`
- Create: `src/casecrawler/validation/image_alignment.py`
- Modify: `pyproject.toml`
- Test: `tests/test_optional_backends.py`

- [x] **Step 1: Add optional dependency groups**

In `pyproject.toml`:

```toml
hf = ["huggingface_hub>=0.28", "datasets>=3.0", "transformers>=4.45"]
imaging = ["diffusers>=0.31", "torch>=2.4", "pillow>=10.0", "open-clip-torch>=2.26"]
parquet = ["pyarrow>=17.0", "pandas>=2.2"]
```

- [x] **Step 2: Add lazy import helpers**

```python
from __future__ import annotations


def require_package(import_name: str, extra: str):
    try:
        return __import__(import_name)
    except ImportError as exc:
        raise RuntimeError(f"Install casecrawler[{extra}] to use this backend.") from exc
```

- [x] **Step 3: Add imaging backend interface**

```python
from __future__ import annotations

from pathlib import Path

from casecrawler.models.synthetic import ImagingAsset


class ImagingGenerator:
    def generate_placeholder(self, output_dir: str, prompt: str, modality: str = "XR", body_region: str = "chest") -> ImagingAsset:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return ImagingAsset(
            image_id="placeholder",
            modality=modality,
            body_region=body_region,
            prompt=prompt,
            file_path=None,
            report_text="Synthetic imaging placeholder. Configure a diffusers backend to render pixels.",
            generation_backend="placeholder",
        )
```

- [x] **Step 4: Add tests that optional imports fail gracefully**

```python
from casecrawler.generation.imaging_generator import ImagingGenerator


def test_imaging_placeholder_does_not_require_diffusers(tmp_path):
    asset = ImagingGenerator().generate_placeholder(str(tmp_path), "portable chest x-ray with pulmonary edema")
    assert asset.generation_backend == "placeholder"
    assert asset.modality == "XR"
```

- [x] **Step 5: Run optional backend tests**

Run:

```bash
python -m pytest tests/test_optional_backends.py -v
```

Expected: pass.

---

## Task 10: Reposition Documentation And UI

**Files:**
- Modify: `README.md`
- Modify: `ui/src/App.tsx`
- Modify or replace: `ui/src/pages/GeneratePage.tsx`
- Modify or replace: `ui/src/pages/CasesPage.tsx`
- Remove from nav: `ui/src/pages/PlayCasePage.tsx`

- [x] **Step 1: Rewrite README opening**

Replace the current education-led introduction with:

```markdown
# CaseCrawler

CaseCrawler generates validated synthetic healthcare datasets for AI training and evaluation.

It combines grounded medical knowledge retrieval, structured clinical data generation, messy clinical text synthesis, labs, vitals, time-series signals, optional medical images, validation, and fine-tuning exports.

The goal is not to simulate a classroom case. The goal is to produce multimodal synthetic records that are ready to inspect, validate, and export as JSONL, FHIR NDJSON, parquet, or model-specific fine-tuning formats.
```

- [x] **Step 2: Replace nav labels**

In `ui/src/App.tsx`, use:

```tsx
<NavLink to="/" end className={navLinkClass}>Knowledge</NavLink>
<NavLink to="/search" className={navLinkClass}>Search</NavLink>
<NavLink to="/sources" className={navLinkClass}>Sources</NavLink>
<NavLink to="/generate" className={navLinkClass}>Generate Dataset</NavLink>
<NavLink to="/cases" className={navLinkClass}>Datasets</NavLink>
```

- [x] **Step 3: Remove player route from nav and future docs**

Keep the file temporarily if tests import it, but do not present it as a primary workflow. Later cleanup can delete the route after the dataset workbench exists.

- [x] **Step 4: Run frontend build**

Run:

```bash
cd ui && npm run build
```

Expected: build passes.

---

## Execution Order

1. Models: Tasks 1-2.
2. Validation: Task 3.
3. Storage: Task 4.
4. Pipeline: Task 5.
5. Export: Task 6.
6. CLI/API: Tasks 7-8.
7. Optional model hooks: Task 9.
8. Docs/UI repositioning: Task 10.

This order creates working software by Task 5 without requiring paid APIs, gated Hugging Face models, GPU image generation, or Synthea installation.

## Later Work

- Add Synthea adapter that shells out to a configured Synthea checkout and imports FHIR NDJSON/CSV into `SyntheticRecord`.
- Add Hugging Face dataset importers for Asclepius and NHSE synthetic notes as examples and regression fixtures.
- Add MedGemma validator/generator adapter for gated users.
- Add BiomedCLIP image-text alignment scoring.
- Add chest X-ray diffusers adapter and CheXGenBench-style report/label validation.
- Add TimeDiff/RawMed-style learned time-series adapters once the schema and validation path are stable.
- Add dataset cards and model cards for generated datasets.
- Add a manual review queue so humans can approve/reject records before export.

## Self-Review

- Spec coverage: The plan covers stripping education-first concepts, adding multimodal data types, wrapping external model backends, validation, storage, CLI/API, export, and docs/UI repositioning.
- Placeholder scan: No task requires unspecified implementation details; later work is explicitly out of scope for the first slice.
- Type consistency: `GenerationRequest`, `SyntheticRecord`, `ValidationReport`, `DatasetStore`, and `SyntheticPipeline` names are consistent across tasks.
