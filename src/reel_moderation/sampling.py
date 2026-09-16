from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from reel_moderation.config import SamplingConfig

RGBFrame = NDArray[np.uint8]


@dataclass(slots=True)
class FrameCandidate:
    frame_index: int
    timestamp_seconds: float
    rgb: RGBFrame
    reasons: set[str] = field(default_factory=set)
    change_score: float = 0.0


@dataclass(slots=True)
class SamplingStats:
    source_fps: float = 0.0
    duration_seconds: float = 0.0
    decoded_frames: int = 0
    capped_candidates: int = 0


def _block_change(current: RGBFrame, previous: RGBFrame, blocks: int = 8) -> float:
    """Return the strongest local mean absolute change in a coarse grid."""

    delta = np.abs(current.astype(np.int16) - previous.astype(np.int16)).mean(axis=2) / 255.0
    height, width = delta.shape
    scores: list[float] = []
    for y in range(blocks):
        y0, y1 = y * height // blocks, (y + 1) * height // blocks
        for x in range(blocks):
            x0, x1 = x * width // blocks, (x + 1) * width // blocks
            cell = delta[y0:y1, x0:x1]
            if cell.size:
                scores.append(float(cell.mean()))
    return max(scores, default=0.0)


class AdaptiveSampler:
    """Decode every frame cheaply and retain frames around visual discontinuities.

    This specifically avoids the classic one-frame-per-second blind spot. Every
    decoded frame is inspected at thumbnail resolution. Uniform samples provide
    coverage while global, local-overlay, and luminance changes open dense bursts.
    """

    def __init__(self, config: SamplingConfig | None = None) -> None:
        self.config = config or SamplingConfig()
        self.stats = SamplingStats()

    def sample(self, video_path: Path) -> Iterator[FrameCandidate]:
        self.stats = SamplingStats()
        previous_thumb: RGBFrame | None = None
        next_baseline_at = 0.0
        burst_until = -1.0
        adaptive_emitted = 0

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            self.stats.source_fps = fps
            time_base = float(stream.time_base)
            max_adaptive_candidates = max(
                1,
                int(
                    self.config.max_candidates_per_minute
                    * max(float(stream.duration or 0) * time_base / 60.0, 1.0)
                ),
            )

            for frame_index, frame in enumerate(container.decode(stream)):
                timestamp = (
                    float(frame.pts * stream.time_base)
                    if frame.pts is not None
                    else (frame_index / fps if fps else 0.0)
                )
                timestamp = max(timestamp, 0.0)
                thumb = frame.reformat(
                    width=self.config.thumbnail_size,
                    height=self.config.thumbnail_size,
                    format="rgb24",
                ).to_ndarray()
                reasons: set[str] = set()
                change_score = 0.0

                if frame_index == 0 or timestamp + 1e-9 >= next_baseline_at:
                    reasons.add("baseline")
                    next_baseline_at = timestamp + 1.0 / self.config.baseline_fps

                if previous_thumb is not None:
                    global_change = float(
                        np.abs(thumb.astype(np.int16) - previous_thumb.astype(np.int16)).mean()
                        / 255.0
                    )
                    local_change = _block_change(thumb, previous_thumb)
                    current_luma = float(thumb.mean()) / 255.0
                    previous_luma = float(previous_thumb.mean()) / 255.0
                    flash_change = abs(current_luma - previous_luma)
                    change_score = max(global_change, local_change, flash_change)

                    if global_change >= self.config.global_change_threshold:
                        reasons.add("scene_change")
                    if local_change >= self.config.local_change_threshold:
                        reasons.add("localized_overlay_change")
                    if flash_change >= self.config.luminance_flash_threshold:
                        reasons.add("luminance_flash")
                    if reasons - {"baseline"}:
                        burst_until = max(burst_until, timestamp + self.config.burst_seconds)

                if timestamp <= burst_until:
                    reasons.add("dense_burst")

                if reasons:
                    baseline_due = "baseline" in reasons
                    if baseline_due or adaptive_emitted < max_adaptive_candidates:
                        if not baseline_due:
                            adaptive_emitted += 1
                        yield FrameCandidate(
                            frame_index=frame_index,
                            timestamp_seconds=timestamp,
                            rgb=frame.to_ndarray(format="rgb24"),
                            reasons=reasons,
                            change_score=change_score,
                        )
                    else:
                        self.stats.capped_candidates += 1

                previous_thumb = thumb
                self.stats.decoded_frames = frame_index + 1
                frame_duration = (
                    float(frame.duration * stream.time_base)
                    if frame.duration is not None
                    else (1.0 / fps if fps else 0.0)
                )
                self.stats.duration_seconds = max(
                    self.stats.duration_seconds,
                    timestamp + max(frame_duration, 0.0),
                )
