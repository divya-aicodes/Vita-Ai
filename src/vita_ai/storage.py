"""SQLite persistence for anonymous screening metadata and optional feedback."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import DATABASE_PATH
from .domain import ScreeningResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS screenings (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    crop TEXT NOT NULL,
    status TEXT NOT NULL,
    primary_condition TEXT,
    match_score REAL,
    photo_status TEXT,
    symptoms_json TEXT NOT NULL,
    environments_json TEXT NOT NULL,
    distribution TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    screening_id TEXT NOT NULL UNIQUE,
    verdict TEXT NOT NULL CHECK (verdict IN ('helpful', 'incorrect', 'unsure')),
    confirmed_condition TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (screening_id) REFERENCES screenings(id) ON DELETE CASCADE
);
"""


def connect(path: Path = DATABASE_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.executescript(SCHEMA)
    return connection


def save_screening(
    *,
    crop: str,
    symptoms: list[str],
    environments: list[str],
    distribution: str,
    notes: str,
    result: ScreeningResult,
    path: Path = DATABASE_PATH,
) -> str:
    case_id = uuid.uuid4().hex
    primary = result.primary
    connection = connect(path)
    try:
        with connection:
            connection.execute(
                """
                INSERT INTO screenings (
                    id, created_at, crop, status, primary_condition, match_score,
                    photo_status, symptoms_json, environments_json, distribution, notes, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    datetime.now(UTC).isoformat(),
                    crop,
                    result.status,
                    primary.name if primary else None,
                    primary.match_score if primary else None,
                    result.photo.status if result.photo else None,
                    json.dumps(symptoms),
                    json.dumps(environments),
                    distribution,
                    notes.strip()[:1000],
                    json.dumps(result.as_dict()),
                ),
            )
    finally:
        connection.close()
    return case_id


def list_screenings(limit: int = 50, path: Path = DATABASE_PATH) -> list[dict[str, Any]]:
    connection = connect(path)
    try:
        rows = connection.execute(
            """
            SELECT s.*, f.verdict, f.confirmed_condition
            FROM screenings s
            LEFT JOIN feedback f ON f.screening_id = s.id
            ORDER BY s.created_at DESC
            LIMIT ?
            """,
            (max(1, min(limit, 200)),),
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def save_feedback(
    screening_id: str,
    verdict: str,
    confirmed_condition: str = "",
    path: Path = DATABASE_PATH,
) -> None:
    if verdict not in {"helpful", "incorrect", "unsure"}:
        raise ValueError("Unsupported feedback verdict.")
    connection = connect(path)
    try:
        with connection:
            connection.execute(
                """
                INSERT INTO feedback (screening_id, verdict, confirmed_condition, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(screening_id) DO UPDATE SET
                    verdict = excluded.verdict,
                    confirmed_condition = excluded.confirmed_condition,
                    created_at = excluded.created_at
                """,
                (
                    screening_id,
                    verdict,
                    confirmed_condition.strip()[:200] or None,
                    datetime.now(UTC).isoformat(),
                ),
            )
    finally:
        connection.close()
