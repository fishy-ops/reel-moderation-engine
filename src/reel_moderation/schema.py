from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Verdict(StrEnum):
    APPROVED = "APPROVED"
    FLAG_FOR_HUMAN_REVIEW = "FLAG_FOR_HUMAN_REVIEW"
    AUTO_BLOCK = "AUTO_BLOCK"
    ERROR = "ERROR"


class Detection(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)
    box: tuple[int, int, int, int] | None = None
    source: str
    crop: str = "full"


class FrameEvidence(BaseModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    risk_score: float = Field(ge=0, le=1)
    raw_risk_score: float | None = Field(default=None, ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    detections: list[Detection] = Field(default_factory=list)
    verifier_signals: dict[str, float | str] = Field(default_factory=dict)


class EvidenceSegment(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    frame_count: int = Field(ge=1)
    max_risk_score: float = Field(ge=0, le=1)
    labels: list[str] = Field(default_factory=list)


class ModerationReport(BaseModel):
    schema_version: str = "1.0"
    video_name: str
    video_sha256: str
    duration_seconds: float = Field(ge=0)
    source_fps: float = Field(ge=0)
    total_frames_decoded: int = Field(ge=0)
    total_frames_analyzed: int = Field(ge=0)
    verdict: Verdict
    max_risk_score: float = Field(default=0, ge=0, le=1)
    flagged_timestamps: list[float] = Field(default_factory=list)
    evidence_segments: list[EvidenceSegment] = Field(default_factory=list)
    evidence: list[FrameEvidence] = Field(default_factory=list)
    backend: str
    processing_seconds: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def error(cls, video_path: Path, message: str, backend: str) -> ModerationReport:
        return cls(
            video_name=video_path.name,
            video_sha256="",
            duration_seconds=0,
            source_fps=0,
            total_frames_decoded=0,
            total_frames_analyzed=0,
            verdict=Verdict.ERROR,
            backend=backend,
            processing_seconds=0,
            warnings=[message],
        )
