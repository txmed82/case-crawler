from __future__ import annotations

import sqlite3
from pathlib import Path

from casecrawler.models.synthetic import SyntheticRecord


class DatasetStore:
    def __init__(self, db_path: str = "./data/datasets.db") -> None:
        self._db_path = db_path
        parent = Path(db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
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

    def list_records(
        self,
        dataset_id: str | None = None,
        topic: str | None = None,
        approved: bool | None = None,
        limit: int = 1000,
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
        query += " LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [SyntheticRecord.model_validate_json(row["record_json"]) for row in rows]
