from __future__ import annotations

from reel_moderation.schema import EvidenceSegment, FrameEvidence


def summarize_secondary_review(
    evidence: list[FrameEvidence],
    minimum_votes: int = 2,
    vote_gap_seconds: float = 1.25,
) -> dict[str, bool | float | int | None]:
    """Find sustained semantic agreement without inventing policy timestamps.

    This signal may raise an otherwise approved video to human review. It is kept
    separate from evidence segments because the broad classifiers are not reliable
    enough to localize the exact policy event.
    """

    votes = sorted(
        (
            frame
            for frame in evidence
            if frame.verifier_signals.get("semantic_decision") == "exposed"
        ),
        key=lambda frame: frame.timestamp_seconds,
    )
    clusters: list[list[FrameEvidence]] = []
    for frame in votes:
        if not clusters or (
            frame.timestamp_seconds - clusters[-1][-1].timestamp_seconds
            > vote_gap_seconds
        ):
            clusters.append([frame])
        else:
            clusters[-1].append(frame)
    strongest = max(
        clusters,
        key=lambda cluster: (
            len(cluster),
            max(
                float(frame.verifier_signals.get("semantic_exposed", 0.0))
                for frame in cluster
            ),
        ),
        default=[],
    )
    strengths = [
        float(frame.verifier_signals.get("semantic_exposed", 0.0))
        for frame in strongest
    ]
    return {
        "triggered": len(strongest) >= minimum_votes,
        "minimum_votes": minimum_votes,
        "vote_gap_seconds": vote_gap_seconds,
        "total_exposed_votes": len(votes),
        "strongest_cluster_votes": len(strongest),
        "strongest_cluster_start": (
            strongest[0].timestamp_seconds if strongest else None
        ),
        "strongest_cluster_end": (
            strongest[-1].timestamp_seconds if strongest else None
        ),
        "strongest_cluster_max_score": max(strengths, default=None),
    }


def summarize_temporal_verification(
    evidence: list[FrameEvidence],
    segments: list[EvidenceSegment],
    context_seconds: float = 0.5,
) -> list[dict[str, float | int]]:
    """Summarize independent-model agreement around each candidate event.

    The summary is deliberately non-enforcing while the verifier is calibrated.
    It contains no pixels and can safely feed evaluation or reviewer-priority logic.
    """

    summaries: list[dict[str, float | int]] = []
    for segment in segments:
        nearby = [
            frame
            for frame in evidence
            if segment.start_seconds - context_seconds
            <= frame.timestamp_seconds
            <= segment.end_seconds + context_seconds
        ]
        semantic_frames = [frame for frame in nearby if frame.verifier_signals]
        exposed_votes = sum(
            frame.verifier_signals.get("semantic_decision") == "exposed"
            for frame in semantic_frames
        )
        covered_votes = sum(
            frame.verifier_signals.get("semantic_decision") == "covered"
            for frame in semantic_frames
        )
        uncertain_votes = len(semantic_frames) - exposed_votes - covered_votes
        primary_exposed_frames = sum(
            any(detection.label.endswith("_EXPOSED") for detection in frame.detections)
            for frame in nearby
        )
        agreement_frames = sum(
            frame.verifier_signals.get("semantic_decision") == "exposed"
            and any(
                detection.label.endswith("_EXPOSED") for detection in frame.detections
            )
            for frame in semantic_frames
        )
        disagreement_frames = sum(
            frame.verifier_signals.get("semantic_decision") == "covered"
            and any(
                detection.label.endswith("_EXPOSED") for detection in frame.detections
            )
            for frame in semantic_frames
        )
        summaries.append(
            {
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
                "duration_seconds": max(
                    0.0, round(segment.end_seconds - segment.start_seconds, 6)
                ),
                "primary_exposed_frames": primary_exposed_frames,
                "semantic_frames": len(semantic_frames),
                "semantic_exposed_votes": exposed_votes,
                "semantic_covered_votes": covered_votes,
                "semantic_uncertain_votes": uncertain_votes,
                "agreement_exposed_frames": agreement_frames,
                "disagreement_covered_frames": disagreement_frames,
            }
        )
    return summaries
