from pathlib import Path

import pytest

from reel_moderation.evaluation import (
    EvaluationCase,
    compare_evaluation_reports,
    evaluate_cases,
    load_manifest,
)
from reel_moderation.schema import ModerationReport, Verdict


def _report(
    name: str,
    verdict: Verdict,
    duration: float = 10,
    processing: float = 2,
    timestamps: list[float] | None = None,
):
    return ModerationReport(
        video_name=name,
        video_sha256=f"hash-{name}",
        duration_seconds=duration,
        source_fps=30,
        total_frames_decoded=300,
        total_frames_analyzed=20,
        verdict=verdict,
        flagged_timestamps=timestamps or [],
        backend="fake",
        processing_seconds=processing,
    )


def test_evaluation_metrics_count_review_as_positive() -> None:
    cases = [
        EvaluationCase(id="positive", video=Path("positive.mp4"), expected_violation=True),
        EvaluationCase(
            id="negative", video=Path("negative.mp4"), expected_violation=False, slices=["safe"]
        ),
    ]
    reports = {
        "positive.mp4": _report("positive.mp4", Verdict.FLAG_FOR_HUMAN_REVIEW),
        "negative.mp4": _report("negative.mp4", Verdict.APPROVED),
    }

    result = evaluate_cases(cases, lambda path: reports[path.name])

    assert result.metrics["recall"] == 1
    assert result.metrics["false_positive_rate"] == 0
    assert result.metrics["review"] == 1
    assert result.per_slice["safe"]["true_negative"] == 1
    assert result.throughput_x_realtime == 5


def test_safe_review_is_separate_from_auto_block_false_positive() -> None:
    cases = [
        EvaluationCase(
            id="revealing-safe",
            video=Path("revealing-safe.mp4"),
            expected_violation=False,
            slices=["revealing-clothing"],
        )
    ]
    report = _report("revealing-safe.mp4", Verdict.FLAG_FOR_HUMAN_REVIEW)

    result = evaluate_cases(cases, lambda path: report)

    assert result.metrics["false_positive_rate"] == 1
    assert result.metrics["safe_review_rate"] == 1
    assert result.metrics["review_false_positive"] == 1
    assert result.metrics["auto_block_false_positive_rate"] == 0
    assert result.metrics["auto_block_false_positive"] == 0


def test_manifest_resolves_relative_paths_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '\n'.join(
            [
                '{"id":"same","video":"a.mp4","expected_violation":true}',
                '{"id":"same","video":"b.mp4","expected_violation":false}',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate case ids"):
        load_manifest(manifest)


def test_comparison_fails_on_accuracy_and_throughput_regression() -> None:
    cases = [
        EvaluationCase(
            id="positive",
            video=Path("positive.mp4"),
            expected_violation=True,
            slices=["brief"],
        ),
        EvaluationCase(
            id="negative",
            video=Path("negative.mp4"),
            expected_violation=False,
            slices=["safe"],
        ),
    ]
    baseline_reports = {
        "positive.mp4": _report("positive.mp4", Verdict.FLAG_FOR_HUMAN_REVIEW),
        "negative.mp4": _report("negative.mp4", Verdict.APPROVED),
    }
    candidate_reports = {
        "positive.mp4": _report(
            "positive.mp4", Verdict.APPROVED, duration=10, processing=4
        ),
        "negative.mp4": _report(
            "negative.mp4", Verdict.FLAG_FOR_HUMAN_REVIEW, duration=10, processing=4
        ),
    }
    baseline = evaluate_cases(cases, lambda path: baseline_reports[path.name])
    candidate = evaluate_cases(cases, lambda path: candidate_reports[path.name])

    comparison = compare_evaluation_reports(baseline, candidate)

    assert comparison.passed is False
    assert comparison.overall_delta["recall"] == -1
    assert comparison.overall_delta["false_positive_rate"] == 1
    assert comparison.overall_delta["throughput_ratio"] == 0.5
    assert any("slice 'brief' recall dropped" in item for item in comparison.failures)
    assert any("slice 'safe' false-positive rate increased" in item for item in comparison.failures)


def test_identical_evaluation_passes_regression_gate() -> None:
    cases = [
        EvaluationCase(id="positive", video=Path("positive.mp4"), expected_violation=True)
    ]
    reports = {
        "positive.mp4": _report("positive.mp4", Verdict.FLAG_FOR_HUMAN_REVIEW)
    }
    baseline = evaluate_cases(cases, lambda path: reports[path.name])

    comparison = compare_evaluation_reports(baseline, baseline)

    assert comparison.passed is True
    assert comparison.failures == []


def test_processing_error_cannot_hide_recall_regression() -> None:
    cases = [
        EvaluationCase(id="positive", video=Path("positive.mp4"), expected_violation=True)
    ]
    baseline = evaluate_cases(
        cases,
        lambda path: _report(path.name, Verdict.FLAG_FOR_HUMAN_REVIEW),
    )
    candidate = evaluate_cases(
        cases,
        lambda path: _report(path.name, Verdict.ERROR),
    )

    comparison = compare_evaluation_reports(baseline, candidate)

    assert comparison.passed is False
    assert comparison.overall_delta["recall"] is None
    assert "overall processing errors increased by 1" in comparison.failures


def test_event_localization_counts_on_and_off_target_timestamps() -> None:
    cases = [
        EvaluationCase(
            id="event",
            video=Path("event.mp4"),
            expected_violation=True,
            events=[{"start_seconds": 2.0, "end_seconds": 3.0}],
            slices=["brief"],
        )
    ]
    report = _report(
        "event.mp4",
        Verdict.FLAG_FOR_HUMAN_REVIEW,
        timestamps=[0.5, 1.6, 2.4, 3.6],
    )

    result = evaluate_cases(
        cases,
        lambda path: report,
        event_tolerance_seconds=0.5,
    )

    assert result.metrics["event_localization_recall"] == 1
    assert result.metrics["on_target_timestamps"] == 2
    assert result.metrics["off_target_timestamps"] == 2
    assert result.metrics["timestamp_precision"] == 0.5
    assert result.results[0].first_detection_latencies_seconds == [0]
    assert result.results[0].first_detection_offsets_seconds == pytest.approx([-0.4])
    assert result.metrics["mean_first_detection_offset_seconds"] == pytest.approx(-0.4)
    assert result.metrics["mean_absolute_onset_error_seconds"] == pytest.approx(0.4)


def test_video_level_review_without_timestamp_is_not_claimed_as_localized() -> None:
    cases = [
        EvaluationCase(
            id="semantic-only",
            video=Path("semantic-only.mp4"),
            expected_violation=True,
            events=[{"start_seconds": 2.0, "end_seconds": 3.0}],
        )
    ]
    report = _report("semantic-only.mp4", Verdict.FLAG_FOR_HUMAN_REVIEW)

    result = evaluate_cases(cases, lambda path: report)

    assert result.metrics["recall"] == 1
    assert result.metrics["event_localization_recall"] == 0
    assert result.results[0].missed_events == 1


def test_multiple_events_are_scored_independently() -> None:
    case = EvaluationCase(
        id="multiple",
        video=Path("multiple.mp4"),
        expected_violation=True,
        events=[
            {"start_seconds": 0.6, "end_seconds": 0.8},
            {"start_seconds": 5.8, "end_seconds": 6.2},
        ],
    )
    report = _report(
        "multiple.mp4",
        Verdict.FLAG_FOR_HUMAN_REVIEW,
        timestamps=[0.7, 9.0],
    )

    result = evaluate_cases([case], lambda path: report, event_tolerance_seconds=0.1)

    assert result.metrics["event_cases"] == 2
    assert result.metrics["localized_events"] == 1
    assert result.metrics["event_localization_recall"] == 0.5
    assert result.results[0].localized_events == 1
    assert result.results[0].missed_events == 1


def test_comparison_rejects_worse_onset_localization() -> None:
    case = EvaluationCase(
        id="timing",
        video=Path("timing.mp4"),
        expected_violation=True,
        events=[{"start_seconds": 2.0, "end_seconds": 3.0}],
        slices=["brief"],
    )
    baseline = evaluate_cases(
        [case],
        lambda path: _report(
            path.name,
            Verdict.FLAG_FOR_HUMAN_REVIEW,
            timestamps=[2.1],
        ),
    )
    candidate = evaluate_cases(
        [case],
        lambda path: _report(
            path.name,
            Verdict.FLAG_FOR_HUMAN_REVIEW,
            timestamps=[2.4],
        ),
    )

    comparison = compare_evaluation_reports(baseline, candidate)

    assert comparison.passed is False
    assert comparison.overall_delta["mean_absolute_onset_error_seconds"] == pytest.approx(
        0.3
    )
    assert any("onset error increased" in item for item in comparison.failures)


def test_comparison_rejects_promoting_safe_review_to_auto_block() -> None:
    case = EvaluationCase(
        id="revealing-safe",
        video=Path("revealing-safe.mp4"),
        expected_violation=False,
        slices=["revealing-clothing"],
    )
    baseline = evaluate_cases(
        [case],
        lambda path: _report(path.name, Verdict.FLAG_FOR_HUMAN_REVIEW),
    )
    candidate = evaluate_cases(
        [case],
        lambda path: _report(path.name, Verdict.AUTO_BLOCK),
    )

    comparison = compare_evaluation_reports(baseline, candidate)

    assert comparison.passed is False
    assert comparison.overall_delta["false_positive_rate"] == 0
    assert comparison.overall_delta["auto_block_false_positive_rate"] == 1
    assert any(
        "auto-block false-positive rate increased" in item
        for item in comparison.failures
    )
