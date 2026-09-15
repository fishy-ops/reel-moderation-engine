from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

GroundTruth = Literal["violation", "safe", "uncertain"]


class FeedbackSummary(BaseModel):
    total: int
    violations: int
    safe: int
    uncertain: int
    false_positive: int
    false_negative: int


def initialize_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS moderation_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_sha256 TEXT NOT NULL,
                report_sha256 TEXT NOT NULL,
                predicted_verdict TEXT NOT NULL,
                ground_truth TEXT NOT NULL CHECK (
                    ground_truth IN ('violation', 'safe', 'uncertain')
                ),
                event_start_seconds REAL,
                event_end_seconds REAL,
                reason TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS feedback_video_idx "
            "ON moderation_feedback(video_sha256)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS report_aggregates (
                video_sha256 TEXT PRIMARY KEY,
                impressions INTEGER NOT NULL,
                unique_reports INTEGER NOT NULL,
                trusted_reports INTEGER NOT NULL,
                reports_last_hour INTEGER NOT NULL,
                review_priority REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def record_feedback(
    database: Path,
    report_path: Path,
    ground_truth: GroundTruth,
    reason: str,
    reviewer: str = "operator",
    event_start_seconds: float | None = None,
    event_end_seconds: float | None = None,
) -> int:
    if event_start_seconds is not None and event_start_seconds < 0:
        raise ValueError("event_start_seconds cannot be negative")
    if (
        event_end_seconds is not None
        and event_start_seconds is not None
        and event_end_seconds < event_start_seconds
    ):
        raise ValueError("event_end_seconds cannot precede event_start_seconds")
    raw_report = report_path.read_bytes()
    report = json.loads(raw_report)
    initialize_database(database)
    with sqlite3.connect(database) as connection:
        cursor = connection.execute(
            """
            INSERT INTO moderation_feedback (
                video_sha256, report_sha256, predicted_verdict, ground_truth,
                event_start_seconds, event_end_seconds, reason, reviewer, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report["video_sha256"],
                hashlib.sha256(raw_report).hexdigest(),
                report["verdict"],
                ground_truth,
                event_start_seconds,
                event_end_seconds,
                reason,
                reviewer,
                datetime.now(UTC).isoformat(),
            ),
        )
        return int(cursor.lastrowid)


def summarize_feedback(database: Path) -> FeedbackSummary:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT predicted_verdict, ground_truth FROM moderation_feedback"
        ).fetchall()
    false_positive = sum(
        truth == "safe" and prediction != "APPROVED" for prediction, truth in rows
    )
    false_negative = sum(
        truth == "violation" and prediction == "APPROVED" for prediction, truth in rows
    )
    return FeedbackSummary(
        total=len(rows),
        violations=sum(truth == "violation" for _, truth in rows),
        safe=sum(truth == "safe" for _, truth in rows),
        uncertain=sum(truth == "uncertain" for _, truth in rows),
        false_positive=false_positive,
        false_negative=false_negative,
    )


def report_priority(
    model_risk: float,
    impressions: int,
    unique_reports: int,
    trusted_reports: int,
    reports_last_hour: int,
) -> float:
    """Prioritize review without allowing reports alone to auto-enforce."""

    if min(impressions, unique_reports, trusted_reports, reports_last_hour) < 0:
        raise ValueError("report counters cannot be negative")
    report_rate = min(1.0, unique_reports / max(impressions, 50) * 100.0)
    trust_ratio = min(1.0, trusted_reports / max(unique_reports, 1))
    velocity = min(1.0, reports_last_hour / 5.0)
    complement = (
        (1.0 - min(max(model_risk, 0.0), 1.0))
        * (1.0 - 0.35 * report_rate)
        * (1.0 - 0.20 * report_rate * trust_ratio)
        * (1.0 - 0.25 * velocity)
    )
    return min(0.94, 1.0 - complement)


def update_report_aggregate(
    database: Path,
    video_sha256: str,
    model_risk: float,
    impressions: int,
    unique_reports: int,
    trusted_reports: int,
    reports_last_hour: int,
) -> float:
    priority = report_priority(
        model_risk,
        impressions,
        unique_reports,
        trusted_reports,
        reports_last_hour,
    )
    initialize_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO report_aggregates (
                video_sha256, impressions, unique_reports, trusted_reports,
                reports_last_hour, review_priority, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_sha256) DO UPDATE SET
                impressions=excluded.impressions,
                unique_reports=excluded.unique_reports,
                trusted_reports=excluded.trusted_reports,
                reports_last_hour=excluded.reports_last_hour,
                review_priority=excluded.review_priority,
                updated_at=excluded.updated_at
            """,
            (
                video_sha256,
                impressions,
                unique_reports,
                trusted_reports,
                reports_last_hour,
                priority,
                datetime.now(UTC).isoformat(),
            ),
        )
    return priority
