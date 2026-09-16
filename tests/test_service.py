from __future__ import annotations

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
