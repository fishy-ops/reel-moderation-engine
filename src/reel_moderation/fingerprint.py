from __future__ import annotations

import hashlib
from pathlib import Path

import av
import numpy as np
from pydantic import BaseModel, Field


class VideoFingerprint(BaseModel):
    schema_version: str = "1.0"
    video_sha256: str
    duration_seconds: float = Field(ge=0)
    sample_fps: float = Field(gt=0)
    frame_hashes: list[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _difference_hash(frame: av.VideoFrame) -> str:
    gray = frame.reformat(width=9, height=8, format="gray8").to_ndarray()
    bits = gray[:, 1:] > gray[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def fingerprint_video(path: Path, sample_fps: float = 2.0) -> VideoFingerprint:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    video = path.expanduser().resolve()
    hashes: list[str] = []
    duration = 0.0
    next_sample = 0.0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        for index, frame in enumerate(container.decode(stream)):
            timestamp = (
                float(frame.pts * stream.time_base)
                if frame.pts is not None
                else index / fps if fps else 0.0
            )
            duration = max(duration, timestamp)
            if timestamp + 1e-9 >= next_sample:
                hashes.append(_difference_hash(frame))
                next_sample = timestamp + 1.0 / sample_fps
    return VideoFingerprint(
        video_sha256=_sha256(video),
        duration_seconds=round(duration, 6),
        sample_fps=sample_fps,
        frame_hashes=hashes,
    )


def _mirrored_difference_hash(value: int) -> int:
    """Approximate the dHash produced by a horizontally mirrored frame."""

    mirrored = 0
    for row_index in range(8):
        shift = 56 - row_index * 8
        row = (value >> shift) & 0xFF
        reversed_row = int(f"{row:08b}"[::-1], 2)
        mirrored = (mirrored << 8) | ((~reversed_row) & 0xFF)
    return mirrored


def fingerprint_similarity(left: VideoFingerprint, right: VideoFingerprint) -> float:
    """Approximate similarity robust to trims, mirroring, and modest transcoding."""

    if not left.frame_hashes or not right.frame_hashes:
        return 0.0
    shorter, longer = sorted(
        (left.frame_hashes, right.frame_hashes), key=len
    )
    similarities = []
    for frame_hash in shorter:
        value = int(frame_hash, 16)
        distance = min(
            min(
                (value ^ candidate_value).bit_count(),
                (value ^ _mirrored_difference_hash(candidate_value)).bit_count(),
            )
            for candidate_value in (int(candidate, 16) for candidate in longer)
        )
        similarities.append(1.0 - distance / 64.0)
    return float(np.mean(similarities))
