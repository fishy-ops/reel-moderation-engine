from pathlib import Path

import numpy as np

from reel_moderation.config import SamplingConfig
from reel_moderation.sampling import AdaptiveSampler, _block_change


def test_local_change_catches_small_overlay() -> None:
    previous = np.zeros((96, 96, 3), dtype=np.uint8)
    current = previous.copy()
    current[32:48, 32:48] = 255

    global_change = float(np.abs(current.astype(int) - previous.astype(int)).mean() / 255)
    local_change = _block_change(current, previous)

    assert global_change < 0.05
    assert local_change > 0.9


def test_no_change_is_zero() -> None:
    frame = np.full((96, 96, 3), 127, dtype=np.uint8)
    assert _block_change(frame, frame) == 0.0


def test_compressed_like_local_change_exceeds_high_recall_threshold() -> None:
    previous = np.zeros((96, 96, 3), dtype=np.uint8)
    current = previous.copy()
    current[32:50, 38:56, 0] = 190

    assert _block_change(current, previous) >= 0.12


class FakeFrame:
    def __init__(self, index: int, value: int) -> None:
        self.pts = index
        self.duration = 1
        self.array = np.full((96, 96, 3), value, dtype=np.uint8)

    def reformat(self, **kwargs):
        return self

    def to_ndarray(self, **kwargs):
        return self.array


class FakeStream:
    average_rate = 10
    time_base = 0.1
    duration = 50


class FakeContainer:
    def __init__(self) -> None:
        self.stream = FakeStream()
        self.streams = type("Streams", (), {"video": [self.stream]})()
        self.frames = [FakeFrame(index, 255 if index % 2 else 0) for index in range(50)]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def decode(self, stream):
        assert stream is self.stream
        return iter(self.frames)


def test_candidate_cap_never_removes_baseline_coverage(monkeypatch) -> None:
    container = FakeContainer()
    monkeypatch.setattr("reel_moderation.sampling.av.open", lambda path: container)
    sampler = AdaptiveSampler(
        SamplingConfig(
            baseline_fps=1,
            global_change_threshold=0,
            local_change_threshold=0,
            luminance_flash_threshold=0,
            max_candidates_per_minute=30,
        )
    )

    candidates = list(sampler.sample(Path("synthetic.mp4")))

    baseline_indices = {
        candidate.frame_index
        for candidate in candidates
        if "baseline" in candidate.reasons
    }
    assert baseline_indices == {0, 10, 20, 30, 40}
    assert sampler.stats.capped_candidates > 0
    assert sampler.stats.decoded_frames == 50
    assert sampler.stats.duration_seconds == 5.0
