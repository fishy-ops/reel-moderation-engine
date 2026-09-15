from pathlib import Path

import pytest

from reel_moderation.evaluation import EvaluationCase, evaluate_cases, load_manifest
from reel_moderation.schema import ModerationReport, Verdict


def _report(name: str, verdict: Verdict, duration: float = 10, processing: float = 2):
    return ModerationReport(
        video_name=name,
        video_sha256=f"hash-{name}",
        duration_seconds=duration,
        source_fps=30,
        total_frames_decoded=300,
        total_frames_analyzed=20,
        verdict=verdict,
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
