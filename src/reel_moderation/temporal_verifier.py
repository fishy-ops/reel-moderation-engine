from __future__ import annotations

from reel_moderation.schema import EvidenceSegment, FrameEvidence


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
