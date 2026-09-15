from __future__ import annotations

from collections import defaultdict

from reel_moderation.config import PolicyConfig
from reel_moderation.schema import EvidenceSegment, FrameEvidence, Verdict


class TemporalAggregator:
    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def segments(self, evidence: list[FrameEvidence]) -> list[EvidenceSegment]:
        risky = sorted(
            (item for item in evidence if item.risk_score >= self.config.review_threshold),
            key=lambda item: item.timestamp_seconds,
        )
        if not risky:
            return []

        clusters: list[list[FrameEvidence]] = [[risky[0]]]
        for item in risky[1:]:
            gap = item.timestamp_seconds - clusters[-1][-1].timestamp_seconds
            if gap <= self.config.cluster_gap_seconds:
                clusters[-1].append(item)
            else:
                clusters.append([item])

        output: list[EvidenceSegment] = []
        for cluster in clusters:
            labels = sorted(
                {
                    detection.label
                    for frame in cluster
                    for detection in frame.detections
                    if detection.score >= self.config.minimum_detection_confidence
                }
            )
            output.append(
                EvidenceSegment(
                    start_seconds=cluster[0].timestamp_seconds,
                    end_seconds=cluster[-1].timestamp_seconds,
                    frame_count=len(cluster),
                    max_risk_score=max(item.risk_score for item in cluster),
                    labels=labels,
                )
            )
        return output

    def verdict(self, evidence: list[FrameEvidence]) -> tuple[Verdict, list[EvidenceSegment]]:
        segments = self.segments(evidence)
        if not segments:
            return Verdict.APPROVED, []

        for segment in segments:
            if (
                segment.frame_count >= self.config.minimum_block_hits
                and segment.max_risk_score >= self.config.block_threshold
            ):
                return Verdict.AUTO_BLOCK, segments
            if segment.max_risk_score >= self.config.single_frame_block_threshold:
                return Verdict.AUTO_BLOCK, segments
        return Verdict.FLAG_FOR_HUMAN_REVIEW, segments


def summarize_labels(evidence: list[FrameEvidence]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for frame in evidence:
        for detection in frame.detections:
            counts[detection.label] += 1
    return dict(sorted(counts.items()))
