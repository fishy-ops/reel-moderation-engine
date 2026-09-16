from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable
from pathlib import Path

from reel_moderation.aggregator import TemporalAggregator, summarize_labels
from reel_moderation.config import EngineConfig
from reel_moderation.provenance import (
    PIPELINE_VERSION,
    decision_config_sha256,
    pipeline_source_sha256,
)
from reel_moderation.sampling import AdaptiveSampler, FrameCandidate
from reel_moderation.schema import FrameEvidence, ModerationReport, Verdict
from reel_moderation.scorers.base import DryRunScorer, FrameScorer
from reel_moderation.scorers.nudenet import NudeNetScorer
from reel_moderation.scorers.semantic import EnsembleVerifier, SiglipVerifier
from reel_moderation.temporal_verifier import (
    summarize_secondary_review,
    summarize_temporal_verification,
)
from reel_moderation.tracking import summarize_tracks


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _batches(items: Iterable[FrameCandidate], size: int) -> Iterable[list[FrameCandidate]]:
    batch: list[FrameCandidate] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class ModerationEngine:
    def __init__(
        self,
        config: EngineConfig | None = None,
        scorer: FrameScorer | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.scorer = scorer or self._make_scorer()
        self.aggregator = TemporalAggregator(self.config.policy)

    def _make_scorer(self) -> FrameScorer:
        if self.config.backend == "dry-run":
            return DryRunScorer()
        if self.config.backend == "nudenet":
            if self.config.scratch_dir is None:
                raise ValueError("scratch_dir must be configured for the NudeNet backend")
            semantic_verifier = None
            if self.config.semantic_verifier.backend == "siglip":
                semantic_verifier = SiglipVerifier(self.config.semantic_verifier.model_id)
            elif self.config.semantic_verifier.backend == "ensemble":
                semantic_verifier = EnsembleVerifier(
                    self.config.semantic_verifier.model_id,
                    self.config.semantic_verifier.classifier_model_id,
                )
            return NudeNetScorer(
                scratch_dir=self.config.scratch_dir,
                minimum_confidence=self.config.policy.minimum_detection_confidence,
                execution_provider=self.config.execution_provider,
                semantic_verifier=semantic_verifier,
                semantic_config=self.config.semantic_verifier,
            )
        raise ValueError(f"unknown moderation backend: {self.config.backend}")

    def moderate(self, video_path: Path) -> ModerationReport:
        video_path = video_path.expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

        started = time.monotonic()
        self.scorer.reset()
        sampler = AdaptiveSampler(self.config.sampling)
        evidence: list[FrameEvidence] = []
        warnings: list[str] = []

        for batch in _batches(sampler.sample(video_path), self.config.batch_size):
            scores = self.scorer.score_batch(batch)
            if len(scores) != len(batch):
                raise RuntimeError("scorer returned a different number of results than inputs")
            for candidate, score in zip(batch, scores, strict=True):
                warnings.extend(score.warnings)
                evidence.append(
                    FrameEvidence(
                        frame_index=candidate.frame_index,
                        timestamp_seconds=round(candidate.timestamp_seconds, 6),
                        risk_score=score.risk_score,
                        raw_risk_score=score.raw_risk_score,
                        reasons=sorted(candidate.reasons),
                        detections=score.detections,
                        verifier_signals=score.verifier_signals,
                    )
                )

        verdict, segments = self.aggregator.verdict(evidence)
        secondary_review = summarize_secondary_review(
            evidence,
            minimum_votes=self.config.semantic_verifier.review_minimum_votes,
            vote_gap_seconds=self.config.semantic_verifier.review_vote_gap_seconds,
        )
        if (
            self.config.semantic_verifier.mode == "review"
            and secondary_review["triggered"]
            and verdict == Verdict.APPROVED
        ):
            verdict = Verdict.FLAG_FOR_HUMAN_REVIEW
            warnings.append(
                "secondary semantic consensus raised the video to review; "
                "no exact policy timestamp was inferred"
            )
        risky_evidence = [
            item for item in evidence if item.risk_score >= self.config.policy.review_threshold
        ]
        semantic_audit = [
            {
                "timestamp_seconds": item.timestamp_seconds,
                "raw_risk_score": item.raw_risk_score,
                "signals": item.verifier_signals,
            }
            for item in evidence
            if item.verifier_signals
        ]
        top_candidates = [
            {
                "timestamp_seconds": item.timestamp_seconds,
                "raw_risk_score": item.raw_risk_score,
                "detections": [
                    {
                        "label": detection.label,
                        "score": detection.score,
                        "crop": detection.crop,
                    }
                    for detection in item.detections
                ],
            }
            for item in sorted(
                evidence,
                key=lambda candidate: candidate.raw_risk_score or 0.0,
                reverse=True,
            )[:20]
            if item.raw_risk_score
        ]
        if sampler.stats.capped_candidates:
            warnings.append(
                f"candidate cap dropped {sampler.stats.capped_candidates} frames; "
                "review sampling limits"
            )
        if self.scorer.name == "dry-run":
            warnings.append("dry-run backend does not perform content classification")

        return ModerationReport(
            video_name=video_path.name,
            video_sha256=_sha256(video_path),
            duration_seconds=round(sampler.stats.duration_seconds, 6),
            source_fps=sampler.stats.source_fps,
            total_frames_decoded=sampler.stats.decoded_frames,
            total_frames_analyzed=len(evidence),
            verdict=verdict,
            max_risk_score=max((item.risk_score for item in evidence), default=0.0),
            flagged_timestamps=[item.timestamp_seconds for item in risky_evidence],
            evidence_segments=segments,
            evidence=risky_evidence,
            backend=self.scorer.name,
            processing_seconds=round(time.monotonic() - started, 6),
            warnings=sorted(set(warnings)),
            metadata={
                "pipeline_version": PIPELINE_VERSION,
                "pipeline_source_sha256": pipeline_source_sha256(),
                "decision_config_sha256": decision_config_sha256(self.config),
                "label_counts": summarize_labels(risky_evidence),
                "sampling": self.config.sampling.model_dump(mode="json"),
                "policy": self.config.policy.model_dump(mode="json"),
                "semantic_verifier": self.config.semantic_verifier.model_dump(mode="json"),
                "semantic_audit": semantic_audit,
                "secondary_review": secondary_review,
                "top_candidates": top_candidates,
                "temporal_verification": summarize_temporal_verification(
                    evidence, segments
                ),
                "region_tracks": summarize_tracks(evidence),
            },
        )
