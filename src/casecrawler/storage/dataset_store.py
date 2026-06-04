from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from casecrawler.generation.recipes import (
    RECIPES,
    recommended_reference_keys_for_exports,
    task_export_reference_keys,
)
from casecrawler.models.blueprint import (
    BlueprintValidationReport,
    ClinicalBlueprint,
    CohortPlan,
    GenerationAttempt,
    GenerationRole,
    JudgeReport,
)
from casecrawler.models.dataset import (
    DatasetManifest,
    ExportFormat,
    ExportManifest,
    HumanReviewDecision,
    HumanReviewSummary,
    HumanReviewStatus,
    ReviewQueueItem,
)
from casecrawler.models.synthetic import SyntheticRecord


_SHARED_INSTANCES: dict[str, "DatasetStore"] = {}
_SHARED_LOCK = threading.Lock()


class DatasetStore:
    """Sqlite-backed persistence for synthetic records and export manifests.

    Concurrency
    -----------
    The connection is opened with ``check_same_thread=False`` so multiple
    FastAPI worker threads can use the same instance, but SQLite still
    serializes writes. All write methods acquire ``self._write_lock`` to make
    that serialization explicit and to prevent two threads from interleaving
    a SELECT/UPDATE pair.

    Most call sites should obtain a process-wide instance via
    :func:`get_shared_store` (or :meth:`DatasetStore.shared`) so they share
    the connection and lock.
    """

    def __init__(self, db_path: str = "./data/datasets.db") -> None:
        self._db_path = db_path
        parent = Path(db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._write_lock = threading.Lock()
        self._create_tables()

    @classmethod
    def shared(cls, db_path: str = "./data/datasets.db") -> "DatasetStore":
        return get_shared_store(db_path)

    def _create_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS synthetic_records (
                record_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                complexity TEXT NOT NULL,
                approved INTEGER,
                requires_human_review INTEGER NOT NULL DEFAULT 0,
                record_json TEXT NOT NULL
            )
            """
        )
        # Idempotent migration for older schemas that pre-date the
        # ``requires_human_review`` column.
        existing_cols = {
            row["name"]
            for row in self._conn.execute(
                "PRAGMA table_info(synthetic_records)"
            ).fetchall()
        }
        if "requires_human_review" not in existing_cols:
            self._conn.execute(
                "ALTER TABLE synthetic_records "
                "ADD COLUMN requires_human_review INTEGER NOT NULL DEFAULT 0"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_synth_dataset ON synthetic_records(dataset_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_synth_topic ON synthetic_records(topic)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_synth_review ON "
            "synthetic_records(dataset_id, approved, requires_human_review)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS export_manifests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                export_format TEXT NOT NULL,
                file_path TEXT NOT NULL,
                record_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_export_manifests_dataset_created_id
            ON export_manifests(dataset_id, created_at DESC, id DESC)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_export_manifests_created_id
            ON export_manifests(created_at DESC, id DESC)
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cohort_plans (
                plan_id TEXT PRIMARY KEY,
                plan_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clinical_blueprints (
                blueprint_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                cohort_plan_id TEXT NOT NULL,
                archetype_name TEXT NOT NULL,
                blueprint_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_blueprints_dataset
            ON clinical_blueprints(dataset_id)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_blueprints_plan
            ON clinical_blueprints(cohort_plan_id)
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_attempts (
                attempt_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                role TEXT NOT NULL,
                status TEXT NOT NULL,
                artifact_id TEXT,
                attempt_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_generation_attempts_dataset
            ON generation_attempts(dataset_id)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_generation_attempts_artifact
            ON generation_attempts(artifact_id)
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS judge_reports (
                report_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                role TEXT NOT NULL,
                passed INTEGER NOT NULL,
                score REAL NOT NULL,
                report_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_judge_reports_dataset
            ON judge_reports(dataset_id)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_judge_reports_artifact
            ON judge_reports(artifact_id)
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS blueprint_validation_reports (
                blueprint_id TEXT PRIMARY KEY,
                tier TEXT NOT NULL,
                report_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def save_record(self, record: SyntheticRecord) -> None:
        effective_approved = self.effective_approved(record)
        approved = None if effective_approved is None else int(effective_approved)
        requires_review = int(bool(record.metadata.get("require_human_review")))
        with self._write_lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO synthetic_records
                (record_id, dataset_id, topic, complexity, approved,
                 requires_human_review, record_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.record_id,
                    record.dataset_id,
                    record.topic,
                    record.complexity.value,
                    approved,
                    requires_review,
                    record.model_dump_json(),
                ),
            )
            self._conn.commit()

    def save_human_review(
        self,
        record_id: str,
        decision: HumanReviewDecision,
    ) -> SyntheticRecord:
        record = self.get_record(record_id)
        if record is None:
            raise KeyError(f"Record {record_id} not found.")
        metadata = dict(record.metadata)
        metadata["human_review"] = decision.model_dump()
        updated = record.model_copy(update={"metadata": metadata})
        self.save_record(updated)
        return updated

    def save_cohort_plan(self, plan: CohortPlan) -> None:
        with self._write_lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO cohort_plans
                (plan_id, plan_json) VALUES (?, ?)""",
                (plan.plan_id, plan.model_dump_json()),
            )
            self._conn.commit()

    def get_cohort_plan(self, plan_id: str) -> CohortPlan | None:
        row = self._conn.execute(
            "SELECT plan_json FROM cohort_plans WHERE plan_id = ?",
            (plan_id,),
        ).fetchone()
        if row is None:
            return None
        return CohortPlan.model_validate_json(row["plan_json"])

    def list_cohort_plans(self, limit: int = 1000) -> list[CohortPlan]:
        rows = self._conn.execute(
            "SELECT plan_json FROM cohort_plans ORDER BY plan_id LIMIT ?",
            (limit,),
        ).fetchall()
        return [CohortPlan.model_validate_json(row["plan_json"]) for row in rows]

    def save_blueprint(self, blueprint: ClinicalBlueprint) -> None:
        with self._write_lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO clinical_blueprints
                (blueprint_id, dataset_id, cohort_plan_id, archetype_name, blueprint_json)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    blueprint.blueprint_id,
                    blueprint.dataset_id,
                    blueprint.cohort_plan_id,
                    blueprint.archetype_name,
                    blueprint.model_dump_json(),
                ),
            )
            self._conn.commit()

    def get_blueprint(self, blueprint_id: str) -> ClinicalBlueprint | None:
        row = self._conn.execute(
            "SELECT blueprint_json FROM clinical_blueprints WHERE blueprint_id = ?",
            (blueprint_id,),
        ).fetchone()
        if row is None:
            return None
        return ClinicalBlueprint.model_validate_json(row["blueprint_json"])

    def list_blueprints(
        self,
        dataset_id: str | None = None,
        cohort_plan_id: str | None = None,
        limit: int = 1000,
    ) -> list[ClinicalBlueprint]:
        query = "SELECT blueprint_json FROM clinical_blueprints WHERE 1=1"
        params: list = []
        if dataset_id:
            query += " AND dataset_id = ?"
            params.append(dataset_id)
        if cohort_plan_id:
            query += " AND cohort_plan_id = ?"
            params.append(cohort_plan_id)
        query += " ORDER BY blueprint_id LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [
            ClinicalBlueprint.model_validate_json(row["blueprint_json"])
            for row in rows
        ]

    def save_generation_attempt(self, attempt: GenerationAttempt) -> None:
        with self._write_lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO generation_attempts
                (attempt_id, dataset_id, role, status, artifact_id, attempt_json)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    attempt.attempt_id,
                    attempt.dataset_id,
                    attempt.role.value,
                    attempt.status.value,
                    attempt.artifact_id,
                    attempt.model_dump_json(),
                ),
            )
            self._conn.commit()

    def get_generation_attempt(self, attempt_id: str) -> GenerationAttempt | None:
        row = self._conn.execute(
            "SELECT attempt_json FROM generation_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        if row is None:
            return None
        return GenerationAttempt.model_validate_json(row["attempt_json"])

    def list_generation_attempts(
        self,
        dataset_id: str | None = None,
        artifact_id: str | None = None,
        role: GenerationRole | str | None = None,
        limit: int = 1000,
    ) -> list[GenerationAttempt]:
        query = "SELECT attempt_json FROM generation_attempts WHERE 1=1"
        params: list = []
        if dataset_id:
            query += " AND dataset_id = ?"
            params.append(dataset_id)
        if artifact_id:
            query += " AND artifact_id = ?"
            params.append(artifact_id)
        if role:
            role_value = role.value if isinstance(role, GenerationRole) else role
            query += " AND role = ?"
            params.append(role_value)
        query += " ORDER BY attempt_id LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [
            GenerationAttempt.model_validate_json(row["attempt_json"])
            for row in rows
        ]

    def save_judge_report(self, report: JudgeReport) -> None:
        with self._write_lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO judge_reports
                (report_id, dataset_id, artifact_id, role, passed, score, report_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    report.report_id,
                    report.dataset_id,
                    report.artifact_id,
                    report.role.value,
                    int(report.passed),
                    report.score,
                    report.model_dump_json(),
                ),
            )
            self._conn.commit()

    def save_judge_report_with_attempt(
        self,
        report: JudgeReport,
        attempt: GenerationAttempt,
    ) -> None:
        with self._write_lock:
            try:
                self._conn.execute(
                    """INSERT OR REPLACE INTO judge_reports
                    (report_id, dataset_id, artifact_id, role, passed, score, report_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        report.report_id,
                        report.dataset_id,
                        report.artifact_id,
                        report.role.value,
                        int(report.passed),
                        report.score,
                        report.model_dump_json(),
                    ),
                )
                self._conn.execute(
                    """INSERT OR REPLACE INTO generation_attempts
                    (attempt_id, dataset_id, role, status, artifact_id, attempt_json)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        attempt.attempt_id,
                        attempt.dataset_id,
                        attempt.role.value,
                        attempt.status.value,
                        attempt.artifact_id,
                        attempt.model_dump_json(),
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def get_judge_report(self, report_id: str) -> JudgeReport | None:
        row = self._conn.execute(
            "SELECT report_json FROM judge_reports WHERE report_id = ?",
            (report_id,),
        ).fetchone()
        if row is None:
            return None
        return JudgeReport.model_validate_json(row["report_json"])

    def list_judge_reports(
        self,
        dataset_id: str | None = None,
        artifact_id: str | None = None,
        limit: int = 1000,
    ) -> list[JudgeReport]:
        query = "SELECT report_json FROM judge_reports WHERE 1=1"
        params: list = []
        if dataset_id:
            query += " AND dataset_id = ?"
            params.append(dataset_id)
        if artifact_id:
            query += " AND artifact_id = ?"
            params.append(artifact_id)
        query += " ORDER BY report_id LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [JudgeReport.model_validate_json(row["report_json"]) for row in rows]

    def save_blueprint_validation_report(
        self,
        report: BlueprintValidationReport,
    ) -> None:
        with self._write_lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO blueprint_validation_reports
                (blueprint_id, tier, report_json) VALUES (?, ?, ?)""",
                (report.blueprint_id, report.tier.value, report.model_dump_json()),
            )
            self._conn.commit()

    def get_blueprint_validation_report(
        self,
        blueprint_id: str,
    ) -> BlueprintValidationReport | None:
        row = self._conn.execute(
            "SELECT report_json FROM blueprint_validation_reports WHERE blueprint_id = ?",
            (blueprint_id,),
        ).fetchone()
        if row is None:
            return None
        return BlueprintValidationReport.model_validate_json(row["report_json"])

    def get_human_review(
        self, record: SyntheticRecord
    ) -> HumanReviewDecision | None:
        payload = record.metadata.get("human_review")
        if payload is None:
            return None
        return HumanReviewDecision.model_validate(payload)

    def effective_approved(self, record: SyntheticRecord) -> bool | None:
        human_review = self.get_human_review(record)
        if human_review and human_review.status == HumanReviewStatus.APPROVED:
            return True
        if human_review and human_review.status in {
            HumanReviewStatus.REJECTED,
            HumanReviewStatus.NEEDS_REVISION,
        }:
            return False
        if record.validation is None:
            return None
        return record.validation.approved

    def list_review_queue(
        self,
        dataset_id: str | None = None,
        include_reviewed: bool = False,
        limit: int = 100,
    ) -> list[ReviewQueueItem]:
        items: list[ReviewQueueItem] = []
        # Push the bulk filter into SQL when possible: records that are not
        # approved (either explicitly false or unscored) plus any record that
        # was flagged as requiring human review even if it passed validation.
        if include_reviewed:
            row_iter = self._iter_record_rows(dataset_id=dataset_id)
        else:
            row_iter = self._iter_record_rows(
                dataset_id=dataset_id,
                review_queue_only=True,
            )
        for record in row_iter:
            human_review = self.get_human_review(record)
            effective_approved = self.effective_approved(record)
            requires_human_review = record.metadata.get("require_human_review") is True
            has_required_approval = (
                human_review is not None
                and human_review.status == HumanReviewStatus.APPROVED
            )
            if not include_reviewed and (
                (effective_approved is True and not (
                    requires_human_review and not has_required_approval
                ))
                or (
                    human_review is not None
                    and human_review.status == HumanReviewStatus.REJECTED
                )
            ):
                continue
            issues = record.validation.issues if record.validation else []
            items.append(
                ReviewQueueItem(
                    record_id=record.record_id,
                    dataset_id=record.dataset_id,
                    topic=record.topic,
                    complexity=record.complexity,
                    modalities=record.modalities,
                    validation_approved=(
                        None if record.validation is None else record.validation.approved
                    ),
                    human_review=human_review,
                    issue_count=len(issues),
                    blocking_issue_count=sum(
                        1 for issue in issues if issue.severity == "error"
                    ),
                )
            )
            if len(items) >= limit:
                break
        return items

    def human_review_summary(self, dataset_id: str | None = None) -> HumanReviewSummary:
        summary = HumanReviewSummary(dataset_id=dataset_id)
        for record in self._iter_record_rows(dataset_id=dataset_id):
            summary.total_records += 1
            review = self.get_human_review(record)
            if review is None or review.status == HumanReviewStatus.PENDING:
                summary.pending += 1
            elif review.status == HumanReviewStatus.APPROVED:
                summary.approved += 1
            elif review.status == HumanReviewStatus.REJECTED:
                summary.rejected += 1
            elif review.status == HumanReviewStatus.NEEDS_REVISION:
                summary.needs_revision += 1
            if record.metadata.get("require_human_review") is True:
                summary.required_human_review += 1
                if review is None or review.status != HumanReviewStatus.APPROVED:
                    summary.missing_required_review += 1
            if record.validation and any(
                issue.severity == "error" for issue in record.validation.issues
            ):
                summary.blocking_issue_records += 1
        return summary

    def get_record(self, record_id: str) -> SyntheticRecord | None:
        row = self._conn.execute(
            "SELECT record_json FROM synthetic_records WHERE record_id = ?",
            (record_id,),
        ).fetchone()
        if row is None:
            return None
        return SyntheticRecord.model_validate_json(row["record_json"])

    def list_records(
        self,
        dataset_id: str | None = None,
        topic: str | None = None,
        approved: bool | None = None,
        limit: int | None = 1000,
        offset: int = 0,
    ) -> list[SyntheticRecord]:
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
        query += " ORDER BY record_id"
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        rows = self._conn.execute(query, params).fetchall()
        return [SyntheticRecord.model_validate_json(row["record_json"]) for row in rows]

    def _iter_record_rows(
        self,
        dataset_id: str | None = None,
        review_queue_only: bool = False,
        page_size: int = 1000,
    ):
        offset = 0
        while True:
            query = "SELECT record_json FROM synthetic_records WHERE 1=1"
            params: list = []
            if dataset_id:
                query += " AND dataset_id = ?"
                params.append(dataset_id)
            if review_queue_only:
                # Either not approved (NULL or 0) OR explicitly flagged for
                # human review even if validator approved it.
                query += (
                    " AND ("
                    "approved IS NULL OR approved = 0"
                    " OR requires_human_review = 1"
                    ")"
                )
            query += " ORDER BY record_id LIMIT ? OFFSET ?"
            params.extend([page_size, offset])
            rows = self._conn.execute(query, params).fetchall()
            if not rows:
                return
            for row in rows:
                yield SyntheticRecord.model_validate_json(row["record_json"])
            if len(rows) < page_size:
                return
            offset += len(rows)

    def iter_records(
        self,
        dataset_id: str | None = None,
        topic: str | None = None,
        approved: bool | None = None,
        page_size: int = 1000,
    ):
        offset = 0
        while True:
            page = self.list_records(
                dataset_id=dataset_id,
                topic=topic,
                approved=approved,
                limit=page_size,
                offset=offset,
            )
            if not page:
                break
            yield from page
            offset += len(page)

    def dataset_exists(self, dataset_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM synthetic_records WHERE dataset_id = ? LIMIT 1",
            (dataset_id,),
        ).fetchone()
        return row is not None

    def list_dataset_ids(self, limit: int = 1000) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT dataset_id FROM synthetic_records ORDER BY dataset_id LIMIT ?",
            (limit,),
        ).fetchall()
        return [row["dataset_id"] for row in rows]

    def find_reference_dataset_id(
        self,
        reference_keys: list[str],
        *,
        exclude_dataset_id: str | None = None,
    ) -> str | None:
        if not reference_keys:
            return None
        candidates = [
            manifest
            for manifest in self.list_manifests()
            if manifest.dataset_id != exclude_dataset_id
        ]
        for reference_key in reference_keys:
            for manifest in candidates:
                if manifest.metadata.get("primary_reference_key") == reference_key:
                    return manifest.dataset_id
        return None

    def get_manifest(self, dataset_id: str) -> DatasetManifest:
        records = list(self.iter_records(dataset_id=dataset_id))
        if not records:
            raise KeyError(f"Dataset {dataset_id} not found.")
        return self._manifest_from_records(dataset_id, records)

    def list_manifests(self, limit: int = 1000) -> list[DatasetManifest]:
        """Return one manifest per dataset.

        The previous implementation called ``get_manifest`` for every dataset,
        which itself did a paginated full scan of every record per dataset --
        an O(N*M) deserialization on every list call. We now aggregate counts
        in a single SQL query and only fall back to ``iter_records`` to grab
        a single representative record per dataset for the manifest's
        ``first.topic`` / ``first.provenance.created_at`` fields.
        """

        dataset_ids = self.list_dataset_ids(limit)
        if not dataset_ids:
            return []
        placeholders = ",".join("?" for _ in dataset_ids)
        agg_rows = self._conn.execute(
            f"SELECT dataset_id, COUNT(*) AS total, "
            f"SUM(CASE WHEN approved = 1 THEN 1 ELSE 0 END) AS approved_count "
            f"FROM synthetic_records WHERE dataset_id IN ({placeholders}) "
            f"GROUP BY dataset_id",
            dataset_ids,
        ).fetchall()
        agg = {
            row["dataset_id"]: (row["total"], row["approved_count"] or 0)
            for row in agg_rows
        }

        manifests: list[DatasetManifest] = []
        for dataset_id in dataset_ids:
            row = self._conn.execute(
                "SELECT record_json FROM synthetic_records WHERE dataset_id = ? "
                "ORDER BY record_id LIMIT 1",
                (dataset_id,),
            ).fetchone()
            if row is None:
                continue
            first = SyntheticRecord.model_validate_json(row["record_json"])
            total, approved_count = agg.get(dataset_id, (0, 0))
            manifests.append(
                self._aggregated_manifest(
                    dataset_id=dataset_id,
                    first=first,
                    total=total,
                    approved_count=approved_count,
                )
            )
        return manifests

    def _aggregated_manifest(
        self,
        *,
        dataset_id: str,
        first: SyntheticRecord,
        total: int,
        approved_count: int,
    ) -> DatasetManifest:
        # We deliberately omit the per-record `record_ids` array from this
        # summary metadata; the previous implementation embedded an N-element
        # UUID list in every manifest, which was an O(N) serialization bomb
        # on every list call. Callers that need the IDs should query
        # ``list_records`` directly.
        # Recipe / reference metadata is derived from a single representative
        # record. This is correct for homogeneous datasets (the common case
        # by far) and degrades gracefully for mixed-recipe datasets, where
        # the trade-off is having a fast list endpoint vs. an exhaustive but
        # impractical one.
        sample = [first]
        export_formats = _manifest_export_formats(sample)
        recipe_metadata = _manifest_recipe_metadata(sample)
        reference_metadata = _manifest_reference_metadata(sample)
        latest_exports = [
            export_manifest.model_dump()
            for export_manifest in self.list_export_manifests(
                dataset_id=dataset_id,
                limit=5,
            )
        ]
        return DatasetManifest(
            dataset_id=dataset_id,
            name=f"{first.topic}-synthetic",
            topic=first.topic,
            requested_count=total,
            generated_count=total,
            approved_count=approved_count,
            modalities=first.modalities,
            export_formats=export_formats,
            created_at=first.provenance.created_at,
            metadata={
                "record_count": total,
                **recipe_metadata,
                **reference_metadata,
                **self._blueprint_manifest_metadata(dataset_id),
                "latest_exports": latest_exports,
            },
        )

    def _manifest_from_records(
        self,
        dataset_id: str,
        records: list[SyntheticRecord],
    ) -> DatasetManifest:
        modalities = []
        for record in records:
            for modality in record.modalities:
                if modality not in modalities:
                    modalities.append(modality)
        approved_count = sum(1 for record in records if self.effective_approved(record))
        first = records[0]
        export_formats = _manifest_export_formats(records)
        recipe_metadata = _manifest_recipe_metadata(records)
        reference_metadata = _manifest_reference_metadata(records)
        return DatasetManifest(
            dataset_id=dataset_id,
            name=f"{first.topic}-synthetic",
            topic=first.topic,
            requested_count=len(records),
            generated_count=len(records),
            approved_count=approved_count,
            modalities=modalities,
            export_formats=export_formats,
            created_at=first.provenance.created_at,
            metadata={
                "record_ids": [record.record_id for record in records],
                **recipe_metadata,
                **reference_metadata,
                **self._blueprint_manifest_metadata(dataset_id),
                "latest_exports": [
                    export_manifest.model_dump()
                    for export_manifest in self.list_export_manifests(
                        dataset_id=dataset_id,
                        limit=5,
                    )
                ],
            },
        )

    def _blueprint_manifest_metadata(self, dataset_id: str) -> dict:
        blueprint_rows = self._conn.execute(
            """
            SELECT COUNT(*) AS blueprint_count
            FROM clinical_blueprints
            WHERE dataset_id = ?
            """,
            (dataset_id,),
        ).fetchone()
        plan_rows = self._conn.execute(
            """
            SELECT DISTINCT cohort_plan_id
            FROM clinical_blueprints
            WHERE dataset_id = ?
            ORDER BY cohort_plan_id
            """,
            (dataset_id,),
        ).fetchall()
        attempt_row = self._conn.execute(
            """
            SELECT COUNT(*) AS generation_attempt_count
            FROM generation_attempts
            WHERE dataset_id = ?
            """,
            (dataset_id,),
        ).fetchone()
        judge_row = self._conn.execute(
            """
            SELECT COUNT(*) AS judge_report_count
            FROM judge_reports
            WHERE dataset_id = ?
            """,
            (dataset_id,),
        ).fetchone()

        blueprint_count = blueprint_rows["blueprint_count"] if blueprint_rows else 0
        attempt_count = (
            attempt_row["generation_attempt_count"] if attempt_row else 0
        )
        judge_count = judge_row["judge_report_count"] if judge_row else 0
        if blueprint_count == 0 and attempt_count == 0 and judge_count == 0:
            return {}
        return {
            "cohort_plan_ids": [row["cohort_plan_id"] for row in plan_rows],
            "blueprint_count": blueprint_count,
            "generation_attempt_count": attempt_count,
            "judge_report_count": judge_count,
        }

    def save_export_manifest(
        self,
        dataset_id: str,
        export_format: str | ExportFormat,
        file_path: str,
        record_count: int,
        metadata: dict | None = None,
    ) -> ExportManifest:
        resolved_format = ExportFormat(export_format)
        created_at = datetime.now().isoformat()
        export_manifest = ExportManifest(
            dataset_id=dataset_id,
            export_format=resolved_format,
            file_path=file_path,
            record_count=record_count,
            created_at=created_at,
            metadata=metadata or {},
        )
        with self._write_lock:
            self._conn.execute(
                """INSERT INTO export_manifests
                (dataset_id, export_format, file_path, record_count, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    dataset_id,
                    resolved_format.value,
                    file_path,
                    record_count,
                    created_at,
                    export_manifest.model_dump_json(),
                ),
            )
            self._conn.commit()
        return export_manifest

    def list_export_manifests(
        self,
        dataset_id: str | None = None,
        limit: int = 100,
    ) -> list[ExportManifest]:
        query = "SELECT metadata_json FROM export_manifests WHERE 1=1"
        params: list = []
        if dataset_id:
            query += " AND dataset_id = ?"
            params.append(dataset_id)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [
            ExportManifest.model_validate_json(row["metadata_json"])
            for row in rows
        ]


def get_shared_store(db_path: str = "./data/datasets.db") -> DatasetStore:
    """Return a process-wide DatasetStore for the given db_path.

    FastAPI route handlers and other long-lived consumers should use this
    rather than constructing a fresh ``DatasetStore`` per request -- creating
    a new sqlite connection on every call is wasteful and fights the write
    lock.

    The cache key is the resolved absolute path of ``db_path``, so callers
    that rely on a relative path (``./data/datasets.db``) and then change
    the working directory pick up the correct on-disk store rather than a
    stale cached instance.
    """

    key = str(Path(db_path).resolve())
    with _SHARED_LOCK:
        instance = _SHARED_INSTANCES.get(key)
        if instance is None:
            instance = DatasetStore(db_path=db_path)
            _SHARED_INSTANCES[key] = instance
    return instance


def reset_shared_stores() -> None:
    """Drop all shared instances. Tests use this to start each case fresh."""

    with _SHARED_LOCK:
        _SHARED_INSTANCES.clear()


def _manifest_export_formats(records: list[SyntheticRecord]) -> list[ExportFormat]:
    requested: list[ExportFormat] = []
    for record in records:
        for value in record.metadata.get("requested_export_formats", []):
            try:
                export_format = ExportFormat(value)
            except ValueError:
                continue
            if export_format not in requested:
                requested.append(export_format)
    return requested or list(ExportFormat)


def _manifest_recipe_metadata(records: list[SyntheticRecord]) -> dict:
    export_formats = _manifest_export_formats(records)
    recipe_counts: dict[str, int] = {}
    for record in records:
        overrides = record.metadata.get("generation_overrides", {})
        if not isinstance(overrides, dict):
            continue
        recipe = overrides.get("recipe")
        if isinstance(recipe, str) and recipe:
            recipe_counts[recipe] = recipe_counts.get(recipe, 0) + 1
    if not recipe_counts:
        reference_keys = recommended_reference_keys_for_exports(export_formats)
        if not reference_keys:
            return {}
        return {
            "recommended_reference_keys": reference_keys,
            "task_export_reference_keys": task_export_reference_keys(export_formats),
            "benchmark_thresholds": {
                "min_overall_score": 0.75,
                "min_metric_score": 0.5,
            },
        }
    selected_recipe = max(recipe_counts, key=recipe_counts.get)
    metadata = {
        "generation_recipes": dict(sorted(recipe_counts.items())),
        "primary_recipe": selected_recipe,
    }
    recipe_spec = RECIPES.get(selected_recipe)
    if recipe_spec is not None:
        export_reference_keys = recommended_reference_keys_for_exports(export_formats)
        reference_keys = list(recipe_spec.recommended_reference_keys)
        for reference_key in export_reference_keys:
            if reference_key not in reference_keys:
                reference_keys.append(reference_key)
        metadata["recommended_reference_keys"] = reference_keys
        metadata["task_export_reference_keys"] = task_export_reference_keys(export_formats)
        metadata["benchmark_thresholds"] = {
            "min_overall_score": recipe_spec.benchmark_min_overall_score,
            "min_metric_score": recipe_spec.benchmark_min_metric_score,
        }
    return metadata


def _manifest_reference_metadata(records: list[SyntheticRecord]) -> dict:
    reference_keys = _metadata_counts(records, "reference_key")
    reference_datasets = _metadata_counts(records, "reference_dataset")
    if not reference_keys and not reference_datasets:
        return {}
    metadata: dict[str, object] = {}
    if reference_keys:
        metadata["reference_keys"] = dict(sorted(reference_keys.items()))
        metadata["primary_reference_key"] = max(reference_keys, key=reference_keys.get)
    if reference_datasets:
        metadata["reference_datasets"] = dict(sorted(reference_datasets.items()))
        metadata["primary_reference_dataset"] = max(
            reference_datasets,
            key=reference_datasets.get,
        )
    return metadata


def _metadata_counts(records: list[SyntheticRecord], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.metadata.get(field)
        if isinstance(value, str) and value:
            counts[value] = counts.get(value, 0) + 1
    return counts
