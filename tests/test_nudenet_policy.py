import numpy as np
import pytest

from reel_moderation.config import SemanticVerifierConfig
from reel_moderation.sampling import FrameCandidate
from reel_moderation.schema import Detection
from reel_moderation.scorers.nudenet import (
    NudeNetScorer,
    _apply_semantic_mode,
    _risk_from_detections,
)
from reel_moderation.scorers.semantic import SemanticScore


def detection(label: str, score: float) -> Detection:
    return Detection(label=label, score=score, source="test")


def test_covered_classes_are_not_violations() -> None:
    assert _risk_from_detections([detection("FEMALE_BREAST_COVERED", 0.99)]) == 0


def test_explicit_classes_raise_risk() -> None:
    risk = _risk_from_detections([detection("FEMALE_GENITALIA_EXPOSED", 0.9)])
    assert risk == pytest.approx(0.9)


def test_multiple_detections_accumulate_sublinearly() -> None:
    risk = _risk_from_detections(
        [
            detection("FEMALE_BREAST_EXPOSED", 0.8),
            detection("BUTTOCKS_EXPOSED", 0.7),
        ]
    )
    assert 0.8 < risk < 1.0


def test_overlapping_crop_duplicate_does_not_inflate_risk() -> None:
    risk = _risk_from_detections(
        [
            detection("FEMALE_BREAST_EXPOSED", 0.8),
            detection("FEMALE_BREAST_EXPOSED", 0.7),
        ]
    )
    assert risk == pytest.approx(0.8 * 0.95)


def test_tiling_runs_on_triggers_and_periodic_baselines() -> None:
    scorer = object.__new__(NudeNetScorer)
    scorer.tile_small_regions = True
    scorer.tile_interval_seconds = 2.0
    scorer._last_tiled_at = float("-inf")
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    assert scorer._should_tile(FrameCandidate(0, 0.0, rgb, {"baseline"}))
    assert not scorer._should_tile(FrameCandidate(15, 0.5, rgb, {"baseline"}))
    assert scorer._should_tile(
        FrameCandidate(16, 0.533, rgb, {"localized_overlay_change"})
    )
    assert scorer._should_tile(FrameCandidate(90, 3.0, rgb, {"baseline"}))


def test_semantic_verifier_runs_periodically_and_on_adaptive_changes() -> None:
    scorer = object.__new__(NudeNetScorer)
    scorer.semantic_verifier = object()
    scorer.semantic_config = SemanticVerifierConfig(
        baseline_interval_seconds=1.0,
        minimum_interval_seconds=0.25,
    )
    scorer._last_semantic_at = float("-inf")
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    assert scorer._should_verify_semantically(
        FrameCandidate(0, 0.0, rgb, {"baseline"}), 0.0
    )
    assert not scorer._should_verify_semantically(
        FrameCandidate(3, 0.1, rgb, {"localized_overlay_change"}), 0.8
    )
    assert scorer._should_verify_semantically(
        FrameCandidate(9, 0.3, rgb, {"localized_overlay_change"}), 0.0
    )
    assert scorer._should_verify_semantically(
        FrameCandidate(39, 1.3, rgb, {"baseline"}), 0.0
    )


def test_reset_clears_per_video_sampling_state() -> None:
    scorer = object.__new__(NudeNetScorer)
    scorer._last_tiled_at = 7.0
    scorer._last_semantic_at = 8.0

    scorer.reset()

    assert scorer._last_tiled_at == float("-inf")
    assert scorer._last_semantic_at == float("-inf")


def test_review_mode_keeps_frame_risk_unlocalized() -> None:
    assert _apply_semantic_mode(0.7, 0.3, "covered", "review") == 0.7
    assert _apply_semantic_mode(0.2, 0.55, "exposed", "review") == 0.2
    assert _apply_semantic_mode(0.7, 0.3, "covered", "enforce") == 0.3


def test_semantic_frames_are_batched(tmp_path) -> None:
    class BatchVerifier:
        name = "batch-test"

        def __init__(self) -> None:
            self.calls = 0

        def score_many(self, region_groups):
            self.calls += 1
            return [SemanticScore(0.7, 0.2, 0.1, 0) for _ in region_groups]

    scorer = object.__new__(NudeNetScorer)
    scorer.scratch_dir = tmp_path
    scorer.semantic_verifier = BatchVerifier()
    scorer.semantic_config = SemanticVerifierConfig(mode="audit")
    scorer._should_tile = lambda frame: False
    scorer._detect = lambda rgb, crop_name, directory: []
    scorer._should_verify_semantically = lambda frame, risk: True
    rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    frames = [
        FrameCandidate(0, 0.0, rgb, {"baseline"}),
        FrameCandidate(30, 1.0, rgb, {"baseline"}),
    ]

    scores = scorer.score_batch(frames)

    assert scorer.semantic_verifier.calls == 1
    assert len(scores) == 2
    assert all(score.verifier_signals["semantic_decision"] == "exposed" for score in scores)
