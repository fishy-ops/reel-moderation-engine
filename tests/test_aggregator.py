from reel_moderation.aggregator import TemporalAggregator
from reel_moderation.schema import Detection, FrameEvidence, Verdict


def evidence(timestamp: float, score: float, index: int = 0) -> FrameEvidence:
    return FrameEvidence(
        frame_index=index,
        timestamp_seconds=timestamp,
        risk_score=score,
        detections=[
            Detection(
                label="FEMALE_BREAST_EXPOSED",
                score=score,
                source="test",
            )
        ],
    )


def test_no_risky_frames_are_approved() -> None:
    verdict, segments = TemporalAggregator().verdict([evidence(0.0, 0.1)])
    assert verdict == Verdict.APPROVED
    assert segments == []


def test_isolated_ambiguous_frame_goes_to_review() -> None:
    verdict, segments = TemporalAggregator().verdict([evidence(1.2, 0.7)])
    assert verdict == Verdict.FLAG_FOR_HUMAN_REVIEW
    assert len(segments) == 1


def test_two_strong_nearby_frames_auto_block() -> None:
    verdict, segments = TemporalAggregator().verdict(
        [evidence(2.0, 0.86), evidence(2.15, 0.88, index=1)]
    )
    assert verdict == Verdict.AUTO_BLOCK
    assert segments[0].frame_count == 2


def test_extreme_single_frame_can_auto_block() -> None:
    verdict, _ = TemporalAggregator().verdict([evidence(3.0, 0.99)])
    assert verdict == Verdict.AUTO_BLOCK


def test_temporally_separate_frames_do_not_form_false_cluster() -> None:
    verdict, segments = TemporalAggregator().verdict(
        [evidence(1.0, 0.84), evidence(3.0, 0.84, index=1)]
    )
    assert verdict == Verdict.FLAG_FOR_HUMAN_REVIEW
    assert len(segments) == 2
