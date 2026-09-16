from reel_moderation.schema import Detection, EvidenceSegment, FrameEvidence
from reel_moderation.temporal_verifier import (
    summarize_secondary_review,
    summarize_temporal_verification,
)


def test_temporal_verifier_counts_model_agreement_without_pixels() -> None:
    evidence = [
        FrameEvidence(
            frame_index=10,
            timestamp_seconds=1.0,
            risk_score=0.7,
            raw_risk_score=0.7,
            reasons=["dense_burst"],
            detections=[
                Detection(label="FEMALE_BREAST_EXPOSED", score=0.7, source="primary")
            ],
            verifier_signals={"semantic_decision": "exposed"},
        ),
        FrameEvidence(
            frame_index=11,
            timestamp_seconds=1.2,
            risk_score=0.6,
            raw_risk_score=0.6,
            reasons=["dense_burst"],
            detections=[
                Detection(label="FEMALE_BREAST_EXPOSED", score=0.6, source="primary")
            ],
            verifier_signals={"semantic_decision": "covered"},
        ),
    ]
    segment = EvidenceSegment(
        start_seconds=1.0,
        end_seconds=1.2,
        frame_count=2,
        max_risk_score=0.7,
        labels=["FEMALE_BREAST_EXPOSED"],
    )

    result = summarize_temporal_verification(evidence, [segment])

    assert result[0]["agreement_exposed_frames"] == 1
    assert result[0]["disagreement_covered_frames"] == 1
    assert result[0]["semantic_frames"] == 2


def test_secondary_review_requires_temporally_adjacent_votes() -> None:
    evidence = [
        FrameEvidence(
            frame_index=index,
            timestamp_seconds=timestamp,
            risk_score=0.0,
            raw_risk_score=0.0,
            verifier_signals={
                "semantic_decision": "exposed",
                "semantic_exposed": score,
            },
        )
        for index, timestamp, score in [(0, 1.0, 0.6), (1, 2.0, 0.7), (2, 8.0, 0.9)]
    ]

    result = summarize_secondary_review(evidence, minimum_votes=2, vote_gap_seconds=1.25)

    assert result["triggered"] is True
    assert result["strongest_cluster_votes"] == 2
    assert result["strongest_cluster_start"] == 1.0
    assert result["strongest_cluster_end"] == 2.0


def test_isolated_secondary_votes_do_not_trigger_review() -> None:
    evidence = [
        FrameEvidence(
            frame_index=index,
            timestamp_seconds=timestamp,
            risk_score=0.0,
            verifier_signals={"semantic_decision": "exposed"},
        )
        for index, timestamp in [(0, 1.0), (1, 5.0)]
    ]

    result = summarize_secondary_review(evidence, minimum_votes=2, vote_gap_seconds=1.25)

    assert result["triggered"] is False
