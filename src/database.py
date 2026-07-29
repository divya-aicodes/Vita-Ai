"""SQLite persistence for anonymous prediction metadata and feedback."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from .config import DATABASE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    crop_name TEXT NOT NULL,
    disease_name TEXT NOT NULL,
    health_status TEXT NOT NULL CHECK (health_status IN ('Healthy', 'Diseased', 'Unknown')),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    second_prediction TEXT,
    third_prediction TEXT,
    image_quality_score REAL NOT NULL CHECK (image_quality_score >= 0 AND image_quality_score <= 100),
    inference_time_ms REAL NOT NULL CHECK (inference_time_ms >= 0),
    model_version TEXT NOT NULL,
    feedback TEXT CHECK (feedback IS NULL OR feedback IN ('Correct', 'Incorrect', 'Unsure'))
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_crop ON predictions(crop_name);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    training_date TEXT,
    test_accuracy REAL,
    macro_f1 REAL,
    input_size TEXT NOT NULL,
    notes TEXT
);
"""


class PredictionDatabase:
    def __init__(self, path: Path | str = DATABASE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(SCHEMA)

    def save_prediction(
        self,
        *,
        crop_name: str,
        disease_name: str,
        health_status: str,
        confidence: float,
        alternatives: list[str],
        image_quality_score: float,
        inference_time_ms: float,
        model_version: str,
    ) -> int:
        values = (alternatives + [None, None])[:2]
        with self.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions (
                    created_at, crop_name, disease_name, health_status, confidence,
                    second_prediction, third_prediction, image_quality_score,
                    inference_time_ms, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    crop_name,
                    disease_name,
                    health_status,
                    float(confidence),
                    values[0],
                    values[1],
                    float(image_quality_score),
                    float(inference_time_ms),
                    model_version,
                ),
            )
            return int(cursor.lastrowid)

    def update_feedback(self, prediction_id: int, feedback: str) -> bool:
        if feedback not in {"Correct", "Incorrect", "Unsure"}:
            raise ValueError("Feedback must be Correct, Incorrect, or Unsure.")
        with self.connection() as connection:
            cursor = connection.execute(
                "UPDATE predictions SET feedback = ? WHERE prediction_id = ?",
                (feedback, int(prediction_id)),
            )
            return cursor.rowcount == 1

    def recent_predictions(self, limit: int = 100) -> list[dict[str, object]]:
        safe_limit = max(1, min(int(limit), 1000))
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def analytics(self) -> dict[str, object]:
        with self.connection() as connection:
            summary = dict(
                connection.execute(
                    """
                    SELECT COUNT(*) AS total,
                           COALESCE(AVG(confidence), 0) AS average_confidence,
                           SUM(CASE WHEN health_status = 'Healthy' THEN 1 ELSE 0 END) AS healthy,
                           SUM(CASE WHEN health_status = 'Diseased' THEN 1 ELSE 0 END) AS diseased,
                           SUM(CASE WHEN confidence < 0.55 THEN 1 ELSE 0 END) AS low_confidence
                    FROM predictions
                    """
                ).fetchone()
            )
            crops = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT crop_name, COUNT(*) AS count
                    FROM predictions GROUP BY crop_name ORDER BY count DESC, crop_name
                    """
                ).fetchall()
            ]
            diseases = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT disease_name, COUNT(*) AS count
                    FROM predictions GROUP BY disease_name ORDER BY count DESC, disease_name LIMIT 10
                    """
                ).fetchall()
            ]
        return {"summary": summary, "by_crop": crops, "top_conditions": diseases}

    def clear_history(self) -> int:
        with self.connection() as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
            connection.execute("DELETE FROM predictions")
        return count

