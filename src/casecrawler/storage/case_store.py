from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from casecrawler.models.case import GeneratedCase


class CaseStore:
    def __init__(self, db_path: str = "./data/cases.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                specialty TEXT NOT NULL,
                accuracy_score REAL,
                pedagogy_score REAL,
                bias_score REAL,
                model TEXT,
                generated_at TIMESTAMP,
                case_json TEXT NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_topic ON cases(topic)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_difficulty ON cases(difficulty)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_specialty ON cases(specialty)")
        self._conn.commit()

    def save(self, case: GeneratedCase) -> None:
        accuracy = case.review.accuracy_score if case.review else None
        pedagogy = case.review.pedagogy_score if case.review else None
        bias = case.review.bias_score if case.review else None
        self._conn.execute(
            """INSERT OR REPLACE INTO cases
            (case_id, topic, difficulty, specialty, accuracy_score, pedagogy_score, bias_score, model, generated_at, case_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                case.case_id,
                case.topic,
                case.difficulty.value,
                ",".join(case.specialty),
                accuracy,
                pedagogy,
                bias,
                case.metadata.get("model", ""),
                case.metadata.get("generated_at", datetime.now().isoformat()),
                case.model_dump_json(),
            ),
        )
        self._conn.commit()

    def get(self, case_id: str) -> GeneratedCase | None:
        row = self._conn.execute(
            "SELECT case_json FROM cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        if row is None:
            return None
        return GeneratedCase.model_validate_json(row["case_json"])

    def list_cases(
        self,
        topic: str | None = None,
        difficulty: str | None = None,
        min_accuracy: float | None = None,
        limit: int = 100,
    ) -> list[GeneratedCase]:
        query = "SELECT case_json FROM cases WHERE 1=1"
        params: list = []

        if topic:
            query += " AND topic = ?"
            params.append(topic)
        if difficulty:
            query += " AND difficulty = ?"
            params.append(difficulty)
        if min_accuracy is not None:
            query += " AND accuracy_score >= ?"
            params.append(min_accuracy)

        query += " ORDER BY generated_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [GeneratedCase.model_validate_json(row["case_json"]) for row in rows]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM cases").fetchone()
        return row["cnt"]

    def export_jsonl(
        self,
        topic: str | None = None,
        difficulty: str | None = None,
    ) -> list[str]:
        """Export cases as JSONL lines."""
        cases = self.list_cases(topic=topic, difficulty=difficulty, limit=10000)
        return [case.model_dump_json() for case in cases]
