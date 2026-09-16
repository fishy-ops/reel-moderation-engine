import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from reel_moderation.feedback import (
    record_feedback,
    report_priority,
    summarize_feedback,
    summarize_feedback_by_configuration,
    update_report_aggregate,
)


def write_report(
    path: Path,
    verdict: str,
    digest: str,
    source_digest: str | None = None,
    config_digest: str | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "video_sha256": digest,
                "verdict": verdict,
                "metadata": {
                    "pipeline_version": "0.1.0" if source_digest else None,
                    "pipeline_source_sha256": source_digest,
                    "decision_config_sha256": config_digest,
                },
            }
        ),
        encoding="utf-8",
    )


def test_feedback_summary_identifies_false_decisions(tmp_path: Path) -> None:
    database = tmp_path / "feedback.sqlite"
    positive = tmp_path / "positive.json"
    negative = tmp_path / "negative.json"
    write_report(positive, "APPROVED", "positive")
    write_report(negative, "FLAG_FOR_HUMAN_REVIEW", "negative")
    record_feedback(database, positive, "violation", "missed exposure")
    record_feedback(database, negative, "safe", "revealing clothing")

    summary = summarize_feedback(database)
    assert summary.total == 2
    assert summary.false_negative == 1
    assert summary.false_positive == 1


def test_report_priority_is_bounded_and_reports_increase_priority() -> None:
    baseline = report_priority(0.3, 1000, 0, 0, 0)
    escalated = report_priority(0.3, 1000, 20, 15, 8)
    assert baseline == pytest.approx(0.3)
    assert baseline < escalated <= 0.94


def test_report_aggregate_is_upserted_without_reporter_identity(tmp_path: Path) -> None:
    database = tmp_path / "feedback.sqlite"
    first = update_report_aggregate(database, "digest", 0.2, 1000, 2, 1, 1)
    second = update_report_aggregate(database, "digest", 0.2, 1000, 20, 15, 8)
    assert second > first


def test_stale_report_update_cannot_reduce_cumulative_counts(tmp_path: Path) -> None:
    database = tmp_path / "feedback.sqlite"
    update_report_aggregate(database, "digest", 0.2, 1000, 20, 15, 5)

    priority = update_report_aggregate(database, "digest", 0.2, 100, 2, 1, 0)

    expected = report_priority(0.2, 1000, 20, 15, 0)
    assert priority == pytest.approx(expected)
    with closing(sqlite3.connect(database)) as connection:
        stored = connection.execute(
            """
            SELECT impressions, unique_reports, trusted_reports, reports_last_hour
            FROM report_aggregates
            WHERE video_sha256 = 'digest'
            """
        ).fetchone()
    assert stored == (1000, 20, 15, 0)


def test_feedback_is_grouped_by_exact_configuration(tmp_path: Path) -> None:
    database = tmp_path / "feedback.sqlite"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_report(first, "APPROVED", "first", "source-a", "config-a")
    write_report(second, "FLAG_FOR_HUMAN_REVIEW", "second", "source-b", "config-b")
    record_feedback(database, first, "violation", "miss")
    record_feedback(database, second, "safe", "false positive")

    summaries = summarize_feedback_by_configuration(database)

    assert len(summaries) == 2
    assert summaries[0].pipeline_source_sha256 == "source-a"
    assert summaries[0].false_negative == 1
    assert summaries[1].pipeline_source_sha256 == "source-b"
    assert summaries[1].false_positive == 1


def test_existing_feedback_database_is_migrated_in_place(tmp_path: Path) -> None:
    database = tmp_path / "feedback.sqlite"
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            """
            CREATE TABLE moderation_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_sha256 TEXT NOT NULL,
                report_sha256 TEXT NOT NULL,
                predicted_verdict TEXT NOT NULL,
                ground_truth TEXT NOT NULL,
                event_start_seconds REAL,
                event_end_seconds REAL,
                reason TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
    report = tmp_path / "report.json"
    write_report(report, "APPROVED", "digest", "source", "config")

    record_feedback(database, report, "safe", "migration test")

    with closing(sqlite3.connect(database)) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(moderation_feedback)")
        }
        stored = connection.execute(
            "SELECT pipeline_source_sha256, decision_config_sha256 "
            "FROM moderation_feedback"
        ).fetchone()
    assert {
        "pipeline_version",
        "pipeline_source_sha256",
        "decision_config_sha256",
        "review_key",
        "updated_at",
    } <= columns
    assert stored == ("source", "config")


def test_feedback_event_bounds_must_be_complete(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    write_report(report, "APPROVED", "digest")

    with pytest.raises(ValueError, match="supplied together"):
        record_feedback(
            tmp_path / "feedback.sqlite",
            report,
            "violation",
            "incomplete event",
            event_start_seconds=1.0,
        )


def test_retried_reviewer_decision_updates_one_adjudication(tmp_path: Path) -> None:
    database = tmp_path / "feedback.sqlite"
    report = tmp_path / "report.json"
    write_report(report, "APPROVED", "digest", "source", "config")

    first_id = record_feedback(database, report, "violation", "initial label")
    second_id = record_feedback(database, report, "safe", "corrected label")

    summary = summarize_feedback(database)
    assert second_id == first_id
    assert summary.total == 1
    assert summary.safe == 1
    assert summary.false_negative == 0
    with closing(sqlite3.connect(database)) as connection:
        stored = connection.execute(
            "SELECT reason, created_at, updated_at FROM moderation_feedback"
        ).fetchone()
    assert stored is not None
    assert stored[0] == "corrected label"
    assert stored[1] is not None
    assert stored[2] is not None
