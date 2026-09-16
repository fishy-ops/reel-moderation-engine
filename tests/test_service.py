from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from fastapi.testclient import TestClient

from reel_moderation import service


class FakeQueue:
    def __init__(self, error: Exception | None = None, count: int = 0) -> None:
        self.error = error
        self.count = count
        self.calls: list[tuple[tuple, dict]] = []

    def enqueue(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return object()


def client_for(tmp_path: Path, monkeypatch, queue: FakeQueue | None = None) -> TestClient:
    monkeypatch.setattr(service, "DATA_DIR", tmp_path)
    monkeypatch.setattr(service, "queue", queue or FakeQueue())
    monkeypatch.setattr(service.redis_connection, "ping", lambda: True)
    monkeypatch.setattr(service, "_registered_worker_count", lambda: 1)
    monkeypatch.setattr(service, "_probe_video", lambda path: None)
    return TestClient(service.app)


def test_upload_is_queued_under_an_opaque_name(tmp_path, monkeypatch) -> None:
    queue = FakeQueue()
    client = client_for(tmp_path, monkeypatch, queue)

    response = client.post(
        "/api/v1/moderate",
        files={"file": ("private-original-name.MP4", b"synthetic-video", "video/mp4")},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert len(payload["task_id"]) == 32
    stored = tmp_path / "incoming" / f"{payload['task_id']}.mp4"
    assert stored.read_bytes() == b"synthetic-video"
    assert queue.calls[0][1]["job_id"] == payload["task_id"]
    assert "private-original-name" not in str(queue.calls[0])


def test_empty_upload_is_rejected_and_removed(tmp_path, monkeypatch) -> None:
    client = client_for(tmp_path, monkeypatch)

    response = client.post(
        "/api/v1/moderate",
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )

    assert response.status_code == 400
    assert not list((tmp_path / "incoming").iterdir())


def test_queue_failure_removes_sensitive_upload(tmp_path, monkeypatch) -> None:
    client = client_for(tmp_path, monkeypatch, FakeQueue(RuntimeError("offline")))

    response = client.post(
        "/api/v1/moderate",
        files={"file": ("video.mp4", b"synthetic-video", "video/mp4")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "moderation queue is unavailable"
    assert not list((tmp_path / "incoming").iterdir())


def test_upload_limit_removes_partial_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(service, "MAX_UPLOAD_BYTES", 3)
    client = client_for(tmp_path, monkeypatch)

    response = client.post(
        "/api/v1/moderate",
        files={"file": ("video.mp4", b"four", "video/mp4")},
    )

    assert response.status_code == 413
    assert not list((tmp_path / "incoming").iterdir())


def test_status_normalizes_rq_enum_value(monkeypatch) -> None:
    class QueueStatus(Enum):
        QUEUED = "queued"

    class FakeJob:
        is_finished = False
        is_failed = False
        result = None

        def get_status(self, refresh: bool):
            assert refresh is True
            return QueueStatus.QUEUED

    monkeypatch.setattr(
        service.Job,
        "fetch",
        classmethod(lambda cls, task_id, connection: FakeJob()),
    )
    client = TestClient(service.app)

    response = client.get("/api/v1/status/" + "a" * 32)

    assert response.status_code == 200
    assert response.json()["status"] == "queued"


def test_feedback_requires_complete_ordered_event_bounds() -> None:
    incomplete = service.FeedbackRequest.model_validate
    try:
        incomplete(
            {
                "task_id": "a" * 32,
                "ground_truth": "violation",
                "reason": "test",
                "event_start_seconds": 2.0,
            }
        )
    except ValueError:
        pass
    else:  # pragma: no cover - assertion branch
        raise AssertionError("an incomplete event range was accepted")


def test_report_signal_rejects_impossible_counts() -> None:
    client = TestClient(service.app)

    response = client.post(
        "/api/v1/report-signal",
        json={
            "task_id": "a" * 32,
            "impressions": 2,
            "unique_reports": 3,
            "trusted_reports": 1,
            "reports_last_hour": 1,
        },
    )

    assert response.status_code == 422


def test_feedback_endpoint_records_bounded_event(tmp_path, monkeypatch) -> None:
    client = client_for(tmp_path, monkeypatch)
    monkeypatch.setattr(service, "FEEDBACK_DATABASE", tmp_path / "feedback.sqlite")
    task_id = "a" * 32
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / f"{task_id}.json").write_text(
        '{"video_sha256":"digest","verdict":"FLAG_FOR_HUMAN_REVIEW",'
        '"metadata":{"pipeline_version":"test"}}',
        encoding="utf-8",
    )

    response = client.post(
        "/api/v1/feedback",
        json={
            "task_id": task_id,
            "ground_truth": "violation",
            "reason": "operator-confirmed brief event",
            "reviewer": "test-reviewer",
            "event_start_seconds": 2.0,
            "event_end_seconds": 3.0,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"feedback_id": 1}
    assert (tmp_path / "feedback.sqlite").is_file()


def test_report_signal_updates_priority_without_changing_verdict(
    tmp_path, monkeypatch
) -> None:
    class FinishedJob:
        is_finished = True
        result = {
            "video_sha256": "digest",
            "max_risk_score": 0.6,
            "verdict": "FLAG_FOR_HUMAN_REVIEW",
        }

    monkeypatch.setattr(
        service.Job,
        "fetch",
        classmethod(lambda cls, task_id, connection: FinishedJob()),
    )
    monkeypatch.setattr(service, "FEEDBACK_DATABASE", tmp_path / "feedback.sqlite")
    client = TestClient(service.app)

    response = client.post(
        "/api/v1/report-signal",
        json={
            "task_id": "a" * 32,
            "impressions": 100,
            "unique_reports": 10,
            "trusted_reports": 8,
            "reports_last_hour": 5,
        },
    )

    assert response.status_code == 200
    assert response.json()["recommendation"] == "expedite_human_review"
    assert 0.65 <= response.json()["review_priority"] <= 0.94
    assert FinishedJob.result["verdict"] == "FLAG_FOR_HUMAN_REVIEW"


def test_report_signal_rejects_unfinished_job(monkeypatch) -> None:
    class QueuedJob:
        is_finished = False
        result = None

    monkeypatch.setattr(
        service.Job,
        "fetch",
        classmethod(lambda cls, task_id, connection: QueuedJob()),
    )
    client = TestClient(service.app)

    response = client.post(
        "/api/v1/report-signal",
        json={
            "task_id": "a" * 32,
            "impressions": 1,
            "unique_reports": 0,
            "trusted_reports": 0,
            "reports_last_hour": 0,
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "moderation task is not finished"


def test_readiness_reports_queue_capacity(tmp_path, monkeypatch) -> None:
    client = client_for(tmp_path, monkeypatch, FakeQueue(count=3))
    monkeypatch.setattr(service, "MAX_QUEUED_JOBS", 10)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "queued_jobs": 3,
        "workers": 1,
        "capacity_remaining": 7,
    }


def test_upload_is_rejected_before_storage_when_no_worker(tmp_path, monkeypatch) -> None:
    client = client_for(tmp_path, monkeypatch)
    monkeypatch.setattr(service, "_registered_worker_count", lambda: 0)

    response = client.post(
        "/api/v1/moderate",
        files={"file": ("video.mp4", b"synthetic-video", "video/mp4")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "no moderation worker is available"
    assert not (tmp_path / "incoming").exists()


def test_upload_is_rejected_before_storage_when_queue_is_full(
    tmp_path, monkeypatch
) -> None:
    client = client_for(tmp_path, monkeypatch, FakeQueue(count=25))
    monkeypatch.setattr(service, "MAX_QUEUED_JOBS", 25)

    response = client.post(
        "/api/v1/moderate",
        files={"file": ("video.mp4", b"synthetic-video", "video/mp4")},
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "moderation queue is at capacity"
    assert not (tmp_path / "incoming").exists()


def test_invalid_container_is_removed_before_queueing(tmp_path, monkeypatch) -> None:
    queue = FakeQueue()
    client = client_for(tmp_path, monkeypatch, queue)

    def reject(path: Path) -> None:
        raise ValueError("invalid or unreadable video container")

    monkeypatch.setattr(service, "_probe_video", reject)

    response = client.post(
        "/api/v1/moderate",
        files={"file": ("video.mp4", b"not-a-video", "video/mp4")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "invalid or unreadable video container"
    assert queue.calls == []
    assert not list((tmp_path / "incoming").iterdir())


def test_probe_rejects_extreme_resolution(tmp_path, monkeypatch) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic")

    class ProbeResult:
        stdout = (
            '{"streams":[{"codec_type":"video","width":8000,"height":5000}],'
            '"format":{"duration":"10.0"}}'
        )

    monkeypatch.setattr(service.subprocess, "run", lambda *args, **kwargs: ProbeResult())

    try:
        service._probe_video(video)
    except ValueError as exc:
        assert str(exc) == "video resolution exceeds the configured limit"
    else:  # pragma: no cover - assertion branch
        raise AssertionError("extreme resolution was accepted")


def test_probe_rejects_container_without_video_stream(tmp_path, monkeypatch) -> None:
    video = tmp_path / "audio-only.mp4"
    video.write_bytes(b"synthetic")

    class ProbeResult:
        stdout = '{"streams":[],"format":{"duration":"10.0"}}'

    monkeypatch.setattr(service.subprocess, "run", lambda *args, **kwargs: ProbeResult())

    try:
        service._probe_video(video)
    except ValueError as exc:
        assert str(exc) == "video has no valid visual stream"
    else:  # pragma: no cover - assertion branch
        raise AssertionError("audio-only container was accepted")


def test_stale_opaque_upload_cleanup_is_narrow(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(service, "MAX_UPLOAD_RETENTION_SECONDS", 3600)
    stale = tmp_path / ("a" * 32 + ".mp4")
    recent = tmp_path / ("b" * 32 + ".mov")
    operator_named = tmp_path / "operator-source.mp4"
    unrelated = tmp_path / ("c" * 32 + ".txt")
    directory = tmp_path / ("d" * 32 + ".mp4")
    stale.write_bytes(b"stale")
    recent.write_bytes(b"recent")
    operator_named.write_bytes(b"operator")
    unrelated.write_bytes(b"unrelated")
    directory.mkdir()
    os.utime(stale, (1_000, 1_000))
    os.utime(operator_named, (1_000, 1_000))
    os.utime(unrelated, (1_000, 1_000))

    removed = service._remove_stale_uploads(tmp_path, now=10_000)

    assert removed == 1
    assert not stale.exists()
    assert recent.exists()
    assert operator_named.exists()
    assert unrelated.exists()
    assert directory.exists()
