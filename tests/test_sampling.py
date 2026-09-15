import numpy as np

from reel_moderation.sampling import _block_change


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
