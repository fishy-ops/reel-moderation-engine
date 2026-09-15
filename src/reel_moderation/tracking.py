from __future__ import annotations

from dataclasses import dataclass, field

from reel_moderation.schema import Detection, FrameEvidence


@dataclass(slots=True)
class RegionTrack:
    label: str
    crop: str
    observations: list[tuple[float, float, tuple[int, int, int, int]]] = field(
        default_factory=list
    )

    @property
    def start_seconds(self) -> float:
        return self.observations[0][0]

    @property
    def end_seconds(self) -> float:
        return self.observations[-1][0]

    @property
    def max_score(self) -> float:
        return max(item[1] for item in self.observations)


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection_width = max(0, min(lx + lw, rx + rw) - max(lx, rx))
    intersection_height = max(0, min(ly + lh, ry + rh) - max(ly, ry))
    intersection = intersection_width * intersection_height
    union = lw * lh + rw * rh - intersection
    return intersection / union if union else 0.0


def _eligible(detection: Detection) -> bool:
    return detection.box is not None and detection.label.endswith("_EXPOSED")


def build_region_tracks(
    evidence: list[FrameEvidence],
    maximum_gap_seconds: float = 0.5,
    minimum_iou: float = 0.12,
) -> list[RegionTrack]:
    tracks: list[RegionTrack] = []
    for frame in sorted(evidence, key=lambda item: item.timestamp_seconds):
        for detection in (item for item in frame.detections if _eligible(item)):
            assert detection.box is not None
            candidates = [
                track
                for track in tracks
                if track.label == detection.label
                and track.crop == detection.crop
                and frame.timestamp_seconds - track.end_seconds <= maximum_gap_seconds
                and _iou(track.observations[-1][2], detection.box) >= minimum_iou
            ]
            if candidates:
                track = max(candidates, key=lambda item: item.end_seconds)
            else:
                track = RegionTrack(label=detection.label, crop=detection.crop)
                tracks.append(track)
            track.observations.append(
                (frame.timestamp_seconds, detection.score, detection.box)
            )
    return tracks


def summarize_tracks(evidence: list[FrameEvidence]) -> list[dict[str, float | int | str]]:
    tracks = build_region_tracks(evidence)
    return [
        {
            "label": track.label,
            "crop": track.crop,
            "start_seconds": round(track.start_seconds, 6),
            "end_seconds": round(track.end_seconds, 6),
            "hits": len(track.observations),
            "max_score": round(track.max_score, 6),
        }
        for track in tracks
        if len(track.observations) >= 2
    ]
