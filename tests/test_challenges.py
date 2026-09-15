from pathlib import Path

import pytest

from reel_moderation.challenges import generate_challenge


def test_challenge_requires_explicit_remote_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MODERATION_DERIVED_MEDIA_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="must be explicitly configured"):
        generate_challenge(tmp_path / "input.mp4", tmp_path / "output.mp4", "mirror")


def test_challenge_rejects_output_outside_root(tmp_path: Path, monkeypatch) -> None:
    allowed = tmp_path / "allowed"
    monkeypatch.setenv("MODERATION_DERIVED_MEDIA_ROOT", str(allowed))
    with pytest.raises(ValueError, match="outside"):
        generate_challenge(
            tmp_path / "input.mp4",
            tmp_path / "elsewhere" / "output.mp4",
            "mirror",
        )
