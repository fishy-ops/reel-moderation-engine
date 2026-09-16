from __future__ import annotations

import json
from pathlib import Path

import pytest

from reel_moderation import worker
from reel_moderation.schema import ModerationReport, Verdict


def report_for(path: Path) -> ModerationReport:
    return ModerationReport(
        video_name=path.name,
        video_sha256="digest",
        duration_seconds=1,
        source_fps=30,
        total_frames_decoded=30,
        total_frames_analyzed=3,
        verdict=Verdict.APPROVED,
        backend="test",
        processing_seconds=0.1,
    )


class FakeEngine:
    def moderate(self, path: Path) -> ModerationReport:
        return report_for(path)


def configure_job(tmp_path: Path, monkeypatch, delete: bool = False) -> Path:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    video = incoming / "job.mp4"
    video.write_bytes(b"synthetic")
    monkeypatch.setenv("MODERATION_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DELETE_INPUTS_AFTER_PROCESSING", str(delete).lower())
    monkeypatch.setattr(worker, "_engine_from_environment", lambda: FakeEngine())
    return video


def test_job_writes_atomic_report(tmp_path: Path, monkeypatch) -> None:
    video = configure_job(tmp_path, monkeypatch)

    result = worker.moderate_job(str(video))

    report_path = tmp_path / "reports" / "job.json"
    assert result["verdict"] == "APPROVED"
    assert json.loads(report_path.read_text(encoding="utf-8"))["video_sha256"] == "digest"
    assert not report_path.with_suffix(".json.tmp").exists()
    assert video.exists()


def test_job_deletes_input_when_retention_is_disabled(tmp_path: Path, monkeypatch) -> None:
    video = configure_job(tmp_path, monkeypatch, delete=True)

    worker.moderate_job(str(video))

    assert not video.exists()


def test_job_deletes_input_by_default(tmp_path: Path, monkeypatch) -> None:
    video = configure_job(tmp_path, monkeypatch)
    monkeypatch.delenv("DELETE_INPUTS_AFTER_PROCESSING")

    worker.moderate_job(str(video))

    assert not video.exists()


def test_failed_job_still_deletes_configured_input(tmp_path: Path, monkeypatch) -> None:
    video = configure_job(tmp_path, monkeypatch, delete=True)

    class FailingEngine:
        def moderate(self, path: Path) -> ModerationReport:
            raise RuntimeError("failure")

    monkeypatch.setattr(worker, "_engine_from_environment", lambda: FailingEngine())

    with pytest.raises(RuntimeError, match="failure"):
        worker.moderate_job(str(video))
    assert not video.exists()


def test_job_rejects_path_outside_data_directory(tmp_path: Path, monkeypatch) -> None:
    outside = tmp_path.parent / "outside.mp4"
    outside.write_bytes(b"synthetic")
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("MODERATION_DATA_DIR", str(data))

    with pytest.raises(ValueError, match="outside the configured data directory"):
        worker.moderate_job(str(outside))
    assert outside.exists()
    outside.unlink()


def test_temporary_report_is_removed_when_publish_fails(tmp_path: Path, monkeypatch) -> None:
    video = configure_job(tmp_path, monkeypatch)
    original_replace = Path.replace

    def fail_report_replace(path: Path, target: Path) -> Path:
        if path.suffix == ".tmp":
            raise OSError("publish failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_report_replace)

    with pytest.raises(OSError, match="publish failed"):
        worker.moderate_job(str(video))
    assert not (tmp_path / "reports" / "job.json.tmp").exists()
    assert not (tmp_path / "reports" / "job.json").exists()
