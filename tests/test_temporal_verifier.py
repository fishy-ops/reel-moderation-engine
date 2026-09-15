from reel_moderation.schema import Detection, EvidenceSegment, FrameEvidence
from reel_moderation.temporal_verifier import summarize_temporal_verification


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
