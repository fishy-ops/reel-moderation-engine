import json
from pathlib import Path

import pytest

from reel_moderation.feedback import (
    record_feedback,
    report_priority,
    summarize_feedback,
    update_report_aggregate,
)


def write_report(path: Path, verdict: str, digest: str) -> None:
    path.write_text(
        json.dumps({"video_sha256": digest, "verdict": verdict}),
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
