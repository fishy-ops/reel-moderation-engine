from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SamplingConfig(BaseModel):
    """Controls cheap every-frame inspection and expensive-frame selection."""

    baseline_fps: float = Field(default=2.0, gt=0, le=30)
    thumbnail_size: int = Field(default=96, ge=32, le=256)
    global_change_threshold: float = Field(default=0.115, ge=0, le=1)
    local_change_threshold: float = Field(default=0.12, ge=0, le=1)
    luminance_flash_threshold: float = Field(default=0.13, ge=0, le=1)
    burst_seconds: float = Field(default=0.35, ge=0, le=3)
    max_candidates_per_minute: int = Field(default=900, ge=30, le=3600)


class PolicyConfig(BaseModel):
    review_threshold: float = Field(default=0.42, ge=0, le=1)
    block_threshold: float = Field(default=0.82, ge=0, le=1)
    single_frame_block_threshold: float = Field(default=0.985, ge=0, le=1)
    minimum_block_hits: int = Field(default=2, ge=1, le=30)
    cluster_gap_seconds: float = Field(default=0.45, gt=0, le=5)
    minimum_detection_confidence: float = Field(default=0.25, ge=0, le=1)

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> PolicyConfig:
        if not self.review_threshold < self.block_threshold <= self.single_frame_block_threshold:
            raise ValueError(
                "thresholds must satisfy review < block <= single-frame block"
            )
        return self


class EngineConfig(BaseModel):
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    batch_size: int = Field(default=16, ge=1, le=128)
    backend: str = "nudenet"
    execution_provider: Literal["auto", "cuda", "cpu"] = "auto"
    scratch_dir: Path | None = None
    keep_evidence_images: bool = False

    @model_validator(mode="after")
    def evidence_requires_scratch(self) -> EngineConfig:
        if self.keep_evidence_images and self.scratch_dir is None:
            raise ValueError("scratch_dir is required when keeping evidence images")
        return self
