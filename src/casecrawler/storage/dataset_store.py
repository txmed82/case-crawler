from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from casecrawler.generation.recipes import (
    RECIPES,
    recommended_reference_keys_for_exports,
    task_export_reference_keys,
)
from casecrawler.models.dataset import (
    DatasetManifest,
    ExportFormat,
    ExportManifest,
    HumanReviewDecision,
    HumanReviewStatus,
    ReviewQueueItem,
)
from casecrawler.models.synthetic import SyntheticRecord


class DatasetStore:
    def __init__(self, db_path: str = "./data/datasets.db") -> None:
        self._db_path = db_path
        parent = Path(db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS synthetic_records (
                record_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                complexity TEXT NOT NULL,
                approved INTEGER,
                record_json TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_synth_dataset ON synthetic_records(dataset_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_synth_topic ON synthetic_records(topic)"
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
        self._conn.commit()

    def save_record(self, record: SyntheticRecord) -> None:
        effective_approved = self.effective_approved(record)
        approved = None if effective_approved is None else int(effective_approved)
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
        for record in self.iter_records(dataset_id=dataset_id):
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
        return [self.get_manifest(dataset_id) for dataset_id in self.list_dataset_ids(limit)]

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
                "latest_exports": [
                    export_manifest.model_dump()
                    for export_manifest in self.list_export_manifests(
                        dataset_id=dataset_id,
                        limit=5,
                    )
                ],
            },
        )

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
