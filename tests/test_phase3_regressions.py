"""Regression tests for Phase 3 (persistence + concurrency)."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from casecrawler.models.dataset import HumanReviewDecision, HumanReviewStatus
from casecrawler.models.synthetic import (
    ComplexityProfile,
    Modality,
    Provenance,
    SyntheticPatient,
    SyntheticRecord,
    ValidationReport,
    VitalObservation,
)
from casecrawler.pipeline.orchestrator import (
    get_shared_orchestrator,
    reset_shared_orchestrator,
)
from casecrawler.sources.pubmed import PubMedSource
from casecrawler.storage.dataset_store import (
    DatasetStore,
    get_shared_store,
    reset_shared_stores,
)


def _record(record_id: str, dataset_id: str = "ds-1", approved: bool = True) -> SyntheticRecord:
    return SyntheticRecord(
        record_id=record_id,
        dataset_id=dataset_id,
        topic="sepsis",
        complexity=ComplexityProfile.MODERATE,
        modalities=[Modality.VITALS],
        patient=SyntheticPatient(patient_id=f"pat-{record_id}", age=40, sex="female"),
        encounters=[],
        labs=[],
        vitals=[
            VitalObservation(
                name="HR",
                value=88,
                unit="/min",
                effective_time="2026-05-06T08:00:00",
            ),
        ],
        provenance=Provenance(
            generator="unit-test",
            created_at="2026-05-06T09:00:00",
        ),
        validation=ValidationReport(
            schema_score=1.0,
            clinical_consistency_score=1.0,
            privacy_score=1.0,
            utility_score=1.0,
            modality_alignment_score=None,
            approved=approved,
            issues=[],
        ),
    )


# --- Shared store reuses one process-wide instance per resolved path -------


def test_get_shared_store_returns_same_instance(tmp_path):
    reset_shared_stores()
    db_path = str(tmp_path / "test.db")
    a = get_shared_store(db_path)
    b = get_shared_store(db_path)
    assert a is b


def test_get_shared_store_distinct_for_different_paths(tmp_path):
    reset_shared_stores()
    a = get_shared_store(str(tmp_path / "a.db"))
    b = get_shared_store(str(tmp_path / "b.db"))
    assert a is not b


def test_dataset_store_shared_classmethod(tmp_path):
    reset_shared_stores()
    db_path = str(tmp_path / "test.db")
    a = DatasetStore.shared(db_path)
    b = get_shared_store(db_path)
    assert a is b


# --- Concurrent writes are serialized without sqlite errors ---------------


def test_concurrent_save_record_does_not_raise(tmp_path):
    reset_shared_stores()
    db_path = str(tmp_path / "concurrent.db")
    store = get_shared_store(db_path)

    errors: list[BaseException] = []

    def _writer(start: int):
        for i in range(start, start + 20):
            try:
                store.save_record(_record(record_id=f"rec-{i}"))
            except BaseException as exc:  # capture & report
                errors.append(exc)
                return

    threads = [threading.Thread(target=_writer, args=(i * 100,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"writers raised: {errors!r}"
    # All 80 records committed.
    rows = store._conn.execute(
        "SELECT COUNT(*) FROM synthetic_records"
    ).fetchone()
    assert rows[0] == 80


# --- Schema migration adds requires_human_review column to legacy DBs ------


def test_legacy_schema_gets_requires_human_review_column(tmp_path):
    reset_shared_stores()
    db_path = str(tmp_path / "legacy.db")
    # Hand-create a pre-Phase-3 schema without `requires_human_review`.
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE synthetic_records (
            record_id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            complexity TEXT NOT NULL,
            approved INTEGER,
            record_json TEXT NOT NULL
        )"""
    )
    conn.commit()
    conn.close()

    store = DatasetStore(db_path=db_path)
    cols = {
        row["name"]
        for row in store._conn.execute("PRAGMA table_info(synthetic_records)").fetchall()
    }
    assert "requires_human_review" in cols


# --- list_review_queue uses the new SQL filter ----------------------------


def test_list_review_queue_skips_approved_records_via_sql(tmp_path):
    reset_shared_stores()
    store = get_shared_store(str(tmp_path / "queue.db"))

    # 50 approved (should be skipped) + 5 unapproved.
    for i in range(50):
        store.save_record(_record(record_id=f"ok-{i}", approved=True))
    for i in range(5):
        store.save_record(_record(record_id=f"bad-{i}", approved=False))

    queue = store.list_review_queue(include_reviewed=False)
    record_ids = {item.record_id for item in queue}
    assert record_ids == {f"bad-{i}" for i in range(5)}


def test_list_review_queue_includes_require_human_review_records(tmp_path):
    reset_shared_stores()
    store = get_shared_store(str(tmp_path / "rhr.db"))

    rec = _record(record_id="needs-review", approved=True)
    rec.metadata["require_human_review"] = True
    store.save_record(rec)

    queue = store.list_review_queue(include_reviewed=False)
    assert {item.record_id for item in queue} == {"needs-review"}


def test_list_review_queue_skips_rejected_human_review(tmp_path):
    reset_shared_stores()
    store = get_shared_store(str(tmp_path / "reject.db"))

    rec = _record(record_id="rejected", approved=False)
    store.save_record(rec)
    store.save_human_review(
        "rejected",
        HumanReviewDecision(status=HumanReviewStatus.REJECTED),
    )

    queue = store.list_review_queue(include_reviewed=False)
    assert queue == []


# --- list_manifests uses the cheap aggregated path -------------------------


def test_list_manifests_aggregates_counts_without_full_scan(tmp_path):
    reset_shared_stores()
    store = get_shared_store(str(tmp_path / "manifests.db"))

    for i in range(10):
        store.save_record(_record(record_id=f"a-{i}", dataset_id="ds-a", approved=True))
    for i in range(3):
        store.save_record(_record(record_id=f"b-{i}", dataset_id="ds-b", approved=False))

    manifests = {m.dataset_id: m for m in store.list_manifests()}
    assert manifests["ds-a"].generated_count == 10
    assert manifests["ds-a"].approved_count == 10
    assert manifests["ds-b"].generated_count == 3
    assert manifests["ds-b"].approved_count == 0
    # The fast manifest path replaces the per-record `record_ids` UUID bomb
    # with a `record_count` summary.
    assert "record_ids" not in manifests["ds-a"].metadata
    assert manifests["ds-a"].metadata["record_count"] == 10


# --- Shared orchestrator returns the same instance ------------------------


def test_get_shared_orchestrator_returns_same_instance(tmp_path, monkeypatch):
    reset_shared_orchestrator()
    monkeypatch.chdir(tmp_path)
    a = get_shared_orchestrator()
    b = get_shared_orchestrator()
    assert a is b


# --- PubMed source refuses to run without ENTREZ_EMAIL --------------------


def test_pubmed_requires_contact_email(monkeypatch):
    monkeypatch.delenv("ENTREZ_EMAIL", raising=False)
    monkeypatch.delenv("NCBI_EMAIL", raising=False)
    src = PubMedSource()
    with pytest.raises(RuntimeError, match="Entrez"):
        src._base_params()


def test_pubmed_accepts_ncbi_email_alias(monkeypatch):
    monkeypatch.delenv("ENTREZ_EMAIL", raising=False)
    monkeypatch.setenv("NCBI_EMAIL", "you@example.com")
    src = PubMedSource()
    assert src._base_params()["email"] == "you@example.com"
