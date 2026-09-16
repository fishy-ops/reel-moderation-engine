from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
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


class ConfigurationFeedbackSummary(FeedbackSummary):
    pipeline_version: str | None = None
    pipeline_source_sha256: str | None = None
    decision_config_sha256: str | None = None


def initialize_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection, connection:
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
                review_key TEXT,
                pipeline_version TEXT,
                pipeline_source_sha256 TEXT,
                decision_config_sha256 TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT
            )
            """
        )
        existing_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(moderation_feedback)")
        }
        for column in (
            "pipeline_version",
            "pipeline_source_sha256",
            "decision_config_sha256",
            "review_key",
            "updated_at",
        ):
            if column not in existing_columns:
                connection.execute(
                    f"ALTER TABLE moderation_feedback ADD COLUMN {column} TEXT"
                )
        seen_review_keys: set[str] = set()
        existing_reviews = connection.execute(
            """
            SELECT id, report_sha256, reviewer
            FROM moderation_feedback
            ORDER BY id DESC
            """
        ).fetchall()
        for feedback_id, report_digest, existing_reviewer in existing_reviews:
            review_key = hashlib.sha256(
                f"{report_digest}:{existing_reviewer}".encode()
            ).hexdigest()
            if review_key in seen_review_keys:
                continue
            seen_review_keys.add(review_key)
            connection.execute(
                "UPDATE moderation_feedback SET review_key = ? WHERE id = ?",
                (review_key, feedback_id),
            )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS feedback_review_key_idx "
            "ON moderation_feedback(review_key)"
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
    if (event_start_seconds is None) != (event_end_seconds is None):
        raise ValueError("event start and end must be supplied together")
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
    metadata = report.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    report_digest = hashlib.sha256(raw_report).hexdigest()
    review_key = hashlib.sha256(f"{report_digest}:{reviewer}".encode()).hexdigest()
    now = datetime.now(UTC).isoformat()
    initialize_database(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            """
            INSERT INTO moderation_feedback (
                video_sha256, report_sha256, predicted_verdict, ground_truth,
                event_start_seconds, event_end_seconds, reason, reviewer,
                review_key,
                pipeline_version, pipeline_source_sha256, decision_config_sha256,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(review_key) DO UPDATE SET
                ground_truth=excluded.ground_truth,
                event_start_seconds=excluded.event_start_seconds,
                event_end_seconds=excluded.event_end_seconds,
                reason=excluded.reason,
                pipeline_version=excluded.pipeline_version,
                pipeline_source_sha256=excluded.pipeline_source_sha256,
                decision_config_sha256=excluded.decision_config_sha256,
                updated_at=excluded.updated_at
            """,
            (
                report["video_sha256"],
                report_digest,
                report["verdict"],
                ground_truth,
                event_start_seconds,
                event_end_seconds,
                reason,
                reviewer,
                review_key,
                metadata.get("pipeline_version"),
                metadata.get("pipeline_source_sha256"),
                metadata.get("decision_config_sha256"),
                now,
                now,
            ),
        )
        row = connection.execute(
            "SELECT id FROM moderation_feedback WHERE review_key = ?",
            (review_key,),
        ).fetchone()
        if row is None:  # pragma: no cover - database invariant
            raise RuntimeError("feedback upsert did not produce a row")
        return int(row[0])


def _summarize_rows(rows: list[tuple[str, str]]) -> FeedbackSummary:
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


def summarize_feedback(database: Path) -> FeedbackSummary:
    initialize_database(database)
    with closing(sqlite3.connect(database)) as connection:
        rows = connection.execute(
            "SELECT predicted_verdict, ground_truth FROM moderation_feedback"
        ).fetchall()
    return _summarize_rows(rows)


def summarize_feedback_by_configuration(
    database: Path,
) -> list[ConfigurationFeedbackSummary]:
    initialize_database(database)
    with closing(sqlite3.connect(database)) as connection:
        rows = connection.execute(
            """
            SELECT pipeline_version, pipeline_source_sha256, decision_config_sha256,
                   predicted_verdict, ground_truth
            FROM moderation_feedback
            """
        ).fetchall()
    grouped: dict[
        tuple[str | None, str | None, str | None],
        list[tuple[str, str]],
    ] = defaultdict(list)
    for version, source_digest, config_digest, prediction, truth in rows:
        grouped[(version, source_digest, config_digest)].append((prediction, truth))
    summaries: list[ConfigurationFeedbackSummary] = []
    for key in sorted(grouped, key=lambda item: tuple(value or "" for value in item)):
        base = _summarize_rows(grouped[key])
        summaries.append(
            ConfigurationFeedbackSummary(
                **base.model_dump(),
                pipeline_version=key[0],
                pipeline_source_sha256=key[1],
                decision_config_sha256=key[2],
            )
        )
    return summaries


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
    initialize_database(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            """
            SELECT impressions, unique_reports, trusted_reports
            FROM report_aggregates
            WHERE video_sha256 = ?
            """,
            (video_sha256,),
        ).fetchone()
        if existing:
            impressions = max(impressions, int(existing[0]))
            unique_reports = max(unique_reports, int(existing[1]))
            trusted_reports = max(trusted_reports, int(existing[2]))
        priority = report_priority(
            model_risk,
            impressions,
            unique_reports,
            trusted_reports,
            reports_last_hour,
        )
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
