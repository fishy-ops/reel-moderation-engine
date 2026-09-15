from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from reel_moderation.sampling import FrameCandidate
from reel_moderation.schema import Detection


@dataclass(slots=True)
class FrameScore:
    risk_score: float
    detections: list[Detection] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class FrameScorer(ABC):
    name: str

    @abstractmethod
    def score_batch(self, frames: list[FrameCandidate]) -> list[FrameScore]:
        """Score frames without retaining their image data."""


class DryRunScorer(FrameScorer):
    """Exercises video plumbing only; it is not a moderation model."""

    name = "dry-run"

    def score_batch(self, frames: list[FrameCandidate]) -> list[FrameScore]:
        return [FrameScore(risk_score=0.0) for _ in frames]
