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


class SemanticVerifierConfig(BaseModel):
    backend: Literal["none", "siglip", "ensemble"] = "none"
    model_id: str = "google/siglip2-base-patch16-224"
    classifier_model_id: str = "Falconsai/nsfw_image_detection"
    mode: Literal["audit", "review", "enforce"] = "audit"
    trigger_threshold: float = Field(default=0.32, ge=0, le=1)
    baseline_interval_seconds: float = Field(default=1.0, gt=0, le=10)
    minimum_interval_seconds: float = Field(default=0.25, gt=0, le=2)
    covered_probability_threshold: float = Field(default=0.60, ge=0, le=1)
    covered_margin: float = Field(default=0.20, ge=0, le=1)
    covered_risk_multiplier: float = Field(default=0.45, ge=0, le=1)
    exposed_probability_threshold: float = Field(default=0.50, ge=0, le=1)
    exposed_margin: float = Field(default=0.15, ge=0, le=1)
    exposed_boost: float = Field(default=0.25, ge=0, le=1)
    exposed_review_floor: float = Field(default=0.55, ge=0, le=1)

    @model_validator(mode="after")
    def semantic_intervals_are_ordered(self) -> SemanticVerifierConfig:
        if self.minimum_interval_seconds > self.baseline_interval_seconds:
            raise ValueError("semantic minimum interval cannot exceed baseline interval")
        return self


class EngineConfig(BaseModel):
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    semantic_verifier: SemanticVerifierConfig = Field(default_factory=SemanticVerifierConfig)
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
