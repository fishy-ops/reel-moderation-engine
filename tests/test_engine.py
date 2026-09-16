from pathlib import Path

import numpy as np

from reel_moderation.config import EngineConfig
from reel_moderation.engine import ModerationEngine
from reel_moderation.sampling import AdaptiveSampler, FrameCandidate
from reel_moderation.schema import Detection, Verdict
from reel_moderation.scorers.base import FrameScore, FrameScorer


class ScriptedScorer(FrameScorer):
    name = "scripted-test"

    def score_batch(self, frames: list[FrameCandidate]) -> list[FrameScore]:
        scores = {0: 0.1, 1: 0.86, 2: 0.89}
        return [
            FrameScore(
                risk_score=scores[frame.frame_index],
                detections=(
                    [
                        Detection(
                            label="FEMALE_BREAST_EXPOSED",
                            score=scores[frame.frame_index],
                            source=self.name,
                        )
                    ]
                    if scores[frame.frame_index] > 0.4
                    else []
                ),
            )
            for frame in frames
        ]


class SemanticReviewScorer(FrameScorer):
    name = "semantic-review-test"

    def score_batch(self, frames: list[FrameCandidate]) -> list[FrameScore]:
        return [
            FrameScore(
                risk_score=0.0,
                raw_risk_score=0.0,
                verifier_signals={
                    "semantic_decision": "exposed",
                    "semantic_exposed": 0.7,
                },
            )
            for _ in frames
        ]


def fake_sample(self: AdaptiveSampler, _: Path):
    self.stats.source_fps = 30
    self.stats.duration_seconds = 0.2
    self.stats.decoded_frames = 6
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    yield FrameCandidate(0, 0.0, rgb, {"baseline"})
    yield FrameCandidate(1, 0.1, rgb, {"localized_overlay_change"})
    yield FrameCandidate(2, 0.2, rgb, {"dense_burst"})


def test_engine_builds_report_without_persisting_frames(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(AdaptiveSampler, "sample", fake_sample)
    video = tmp_path / "opaque-video-input.mp4"
    video.write_bytes(b"not decoded because the sampler is replaced")
    engine = ModerationEngine(
        EngineConfig(backend="dry-run", batch_size=2), scorer=ScriptedScorer()
    )

    report = engine.moderate(video)

    assert report.verdict == Verdict.AUTO_BLOCK
    assert report.total_frames_decoded == 6
    assert report.total_frames_analyzed == 3
    assert report.flagged_timestamps == [0.1, 0.2]
    assert len(report.video_sha256) == 64
    assert not list(tmp_path.glob("*.jpg"))


def test_secondary_review_does_not_invent_flagged_timestamps(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(AdaptiveSampler, "sample", fake_sample)
    video = tmp_path / "opaque-video-input.mp4"
    video.write_bytes(b"not decoded because the sampler is replaced")
    engine = ModerationEngine(
        EngineConfig(
            backend="dry-run",
            semantic_verifier={
                "backend": "none",
                "mode": "review",
                "review_minimum_votes": 2,
                "review_vote_gap_seconds": 1.25,
            },
            batch_size=2,
        ),
        scorer=SemanticReviewScorer(),
    )

    report = engine.moderate(video)

    assert report.verdict == Verdict.FLAG_FOR_HUMAN_REVIEW
    assert report.flagged_timestamps == []
    assert report.evidence_segments == []
    assert report.metadata["secondary_review"]["triggered"] is True
