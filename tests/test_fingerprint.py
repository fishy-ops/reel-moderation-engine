from reel_moderation.fingerprint import (
    VideoFingerprint,
    _mirrored_difference_hash,
    fingerprint_similarity,
)


def fingerprint(*hashes: str) -> VideoFingerprint:
    return VideoFingerprint(
        video_sha256="source",
        duration_seconds=1,
        sample_fps=2,
        frame_hashes=list(hashes),
    )


def test_fingerprint_similarity_matches_transcoded_near_duplicates() -> None:
    left = fingerprint("0000000000000000", "ffffffffffffffff")
    right = fingerprint("0000000000000001", "fffffffffffffffe", "aaaaaaaaaaaaaaaa")
    assert fingerprint_similarity(left, right) == 63 / 64


def test_empty_fingerprint_has_zero_similarity() -> None:
    assert fingerprint_similarity(fingerprint(), fingerprint("0" * 16)) == 0


def test_fingerprint_similarity_recognizes_horizontal_mirror() -> None:
    original = int("963ca55a0ff06987", 16)
    mirrored = _mirrored_difference_hash(original)

    assert fingerprint_similarity(
        fingerprint(f"{original:016x}"), fingerprint(f"{mirrored:016x}")
    ) == 1.0
