from reel_moderation.schema import Detection, FrameEvidence
from reel_moderation.tracking import build_region_tracks, summarize_tracks


def frame(timestamp: float, box: tuple[int, int, int, int]) -> FrameEvidence:
    return FrameEvidence(
        frame_index=round(timestamp * 30),
        timestamp_seconds=timestamp,
        risk_score=0.7,
        detections=[
            Detection(
                label="FEMALE_BREAST_EXPOSED",
                score=0.7,
                box=box,
                source="test",
            )
        ],
    )


def test_nearby_overlapping_regions_form_track() -> None:
    evidence = [frame(1.0, (10, 10, 20, 20)), frame(1.1, (12, 11, 20, 20))]
    tracks = build_region_tracks(evidence)
    assert len(tracks) == 1
    assert len(tracks[0].observations) == 2
    assert summarize_tracks(evidence)[0]["hits"] == 2


def test_distant_regions_form_separate_tracks() -> None:
    tracks = build_region_tracks(
        [frame(1.0, (10, 10, 20, 20)), frame(1.1, (100, 100, 20, 20))]
    )
    assert len(tracks) == 2
