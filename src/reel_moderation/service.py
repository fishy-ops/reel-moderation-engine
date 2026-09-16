from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, model_validator
from redis import Redis
from rq import Queue, Worker
from rq.job import Job

from reel_moderation.feedback import record_feedback, update_report_aggregate
from reel_moderation.worker import moderate_job

DATA_DIR = Path(os.environ.get("MODERATION_DATA_DIR", "/data"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(2 * 1024**3)))
MAX_QUEUED_JOBS = int(os.environ.get("MAX_QUEUED_JOBS", "25"))
MAX_VIDEO_SECONDS = float(os.environ.get("MAX_VIDEO_SECONDS", "300"))
MAX_VIDEO_PIXELS = int(os.environ.get("MAX_VIDEO_PIXELS", str(7680 * 4320)))
MAX_UPLOAD_RETENTION_SECONDS = int(
    os.environ.get("MAX_UPLOAD_RETENTION_SECONDS", str(24 * 3600))
)
FEEDBACK_DATABASE = Path(
    os.environ.get("MODERATION_FEEDBACK_DATABASE", "/data/reports/feedback.sqlite")
)
ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

redis_connection = Redis.from_url(REDIS_URL)
queue = Queue("moderation", connection=redis_connection, default_timeout="30m")

app = FastAPI(title="Reel Moderation Engine", version="0.1.0")


class SubmitResponse(BaseModel):
    task_id: str
    status: str


class StatusResponse(BaseModel):
    task_id: str
    status: str
    result: dict | None = None
    error: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready"]
    queued_jobs: int
    workers: int
    capacity_remaining: int


class FeedbackRequest(BaseModel):
    task_id: str
    ground_truth: Literal["violation", "safe", "uncertain"]
    reason: str = Field(min_length=1, max_length=500)
    reviewer: str = Field(default="operator", min_length=1, max_length=100)
    event_start_seconds: float | None = Field(default=None, ge=0)
    event_end_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def event_bounds_are_ordered(self) -> FeedbackRequest:
        if (self.event_start_seconds is None) != (self.event_end_seconds is None):
            raise ValueError("event start and end must be supplied together")
        if (
            self.event_start_seconds is not None
            and self.event_end_seconds is not None
            and self.event_end_seconds < self.event_start_seconds
        ):
            raise ValueError("event end cannot precede event start")
        return self


class ReportSignalRequest(BaseModel):
    task_id: str
    impressions: int = Field(ge=0)
    unique_reports: int = Field(ge=0)
    trusted_reports: int = Field(ge=0)
    reports_last_hour: int = Field(ge=0)

    @model_validator(mode="after")
    def report_counts_are_consistent(self) -> ReportSignalRequest:
        if self.unique_reports > self.impressions:
            raise ValueError("unique reports cannot exceed impressions")
        if self.trusted_reports > self.unique_reports:
            raise ValueError("trusted reports cannot exceed unique reports")
        if self.reports_last_hour > self.unique_reports:
            raise ValueError("recent reports cannot exceed unique reports")
        return self


class ReportSignalResponse(BaseModel):
    task_id: str
    review_priority: float
    recommendation: str


@app.get("/health")
def health() -> dict[str, str]:
    redis_connection.ping()
    return {"status": "ok"}


def _registered_worker_count() -> int:
    available_states = {"idle", "busy"}
    count = 0
    for worker in Worker.all(queue=queue):
        state = worker.get_state()
        if str(getattr(state, "value", state)) in available_states:
            count += 1
    return count


def _capacity() -> tuple[int, int]:
    redis_connection.ping()
    return int(queue.count), _registered_worker_count()


def _ensure_accepting_uploads() -> tuple[int, int]:
    try:
        queued_jobs, workers = _capacity()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="moderation queue is unavailable",
        ) from exc
    if workers < 1:
        raise HTTPException(status_code=503, detail="no moderation worker is available")
    if queued_jobs >= MAX_QUEUED_JOBS:
        raise HTTPException(status_code=429, detail="moderation queue is at capacity")
    return queued_jobs, workers


def _probe_video(path: Path) -> None:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file,crypto,data,pipe",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type,width,height:format=duration",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        payload = json.loads(completed.stdout)
        video_streams = [
            stream
            for stream in payload.get("streams", [])
            if stream.get("codec_type") == "video"
        ]
        duration = float(payload["format"]["duration"])
    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        raise ValueError("invalid or unreadable video container") from exc
    if not video_streams:
        raise ValueError("video has no valid visual stream")
    try:
        width = int(video_streams[0]["width"])
        height = int(video_streams[0]["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("video has no valid visual stream") from exc
    if duration <= 0 or width <= 0 or height <= 0:
        raise ValueError("video has no valid visual stream")
    if duration > MAX_VIDEO_SECONDS:
        raise ValueError("video duration exceeds the configured limit")
    if width * height > MAX_VIDEO_PIXELS:
        raise ValueError("video resolution exceeds the configured limit")


def _remove_stale_uploads(incoming: Path, now: float | None = None) -> int:
    """Remove abandoned opaque uploads without touching operator source media."""

    cutoff = (time.time() if now is None else now) - MAX_UPLOAD_RETENTION_SECONDS
    removed = 0
    for candidate in incoming.iterdir():
        if candidate.is_symlink() or not candidate.is_file():
            continue
        if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        task_id = candidate.stem
        if len(task_id) != 32 or any(char not in "0123456789abcdef" for char in task_id):
            continue
        try:
            if candidate.stat().st_mtime < cutoff:
                candidate.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


@app.get("/ready", response_model=ReadinessResponse)
def ready() -> ReadinessResponse:
    queued_jobs, workers = _ensure_accepting_uploads()
    return ReadinessResponse(
        status="ready",
        queued_jobs=queued_jobs,
        workers=workers,
        capacity_remaining=MAX_QUEUED_JOBS - queued_jobs,
    )


@app.post(
    "/api/v1/moderate",
    response_model=SubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_video(file: Annotated[UploadFile, File(...)]) -> SubmitResponse:
    _ensure_accepting_uploads()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="unsupported video container")

    task_id = uuid.uuid4().hex
    incoming = DATA_DIR / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    _remove_stale_uploads(incoming)
    destination = incoming / f"{task_id}{suffix}"
    written = 0
    try:
        with destination.open("xb") as target:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="video exceeds upload limit")
                target.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="video is empty")

    try:
        _probe_video(destination)
    except ValueError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        queue.enqueue(
            moderate_job,
            str(destination),
            job_id=task_id,
            result_ttl=7 * 24 * 3600,
            failure_ttl=7 * 24 * 3600,
        )
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=503,
            detail="moderation queue is unavailable",
        ) from exc
    return SubmitResponse(task_id=task_id, status="queued")


@app.get("/api/v1/status/{task_id}", response_model=StatusResponse)
def task_status(task_id: str) -> StatusResponse:
    _validate_task_id(task_id)
    try:
        job = Job.fetch(task_id, connection=redis_connection)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc

    job_status = job.get_status(refresh=True)
    if job.is_finished:
        return StatusResponse(task_id=task_id, status="finished", result=job.result)
    if job.is_failed:
        return StatusResponse(
            task_id=task_id,
            status="failed",
            error="moderation task failed; inspect restricted worker logs",
        )
    status_value = getattr(job_status, "value", job_status)
    return StatusResponse(task_id=task_id, status=str(status_value))


def _validate_task_id(task_id: str) -> None:
    if len(task_id) != 32 or any(char not in "0123456789abcdef" for char in task_id):
        raise HTTPException(status_code=400, detail="invalid task id")


@app.post("/api/v1/feedback")
def submit_feedback(request: FeedbackRequest) -> dict[str, int]:
    _validate_task_id(request.task_id)
    report_path = DATA_DIR / "reports" / f"{request.task_id}.json"
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="completed report not found")
    feedback_id = record_feedback(
        FEEDBACK_DATABASE,
        report_path,
        request.ground_truth,
        request.reason,
        reviewer=request.reviewer,
        event_start_seconds=request.event_start_seconds,
        event_end_seconds=request.event_end_seconds,
    )
    return {"feedback_id": feedback_id}


@app.post("/api/v1/report-signal", response_model=ReportSignalResponse)
def submit_report_signal(request: ReportSignalRequest) -> ReportSignalResponse:
    _validate_task_id(request.task_id)
    try:
        job = Job.fetch(request.task_id, connection=redis_connection)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc
    if not job.is_finished or not isinstance(job.result, dict):
        raise HTTPException(status_code=409, detail="moderation task is not finished")
    priority = update_report_aggregate(
        FEEDBACK_DATABASE,
        str(job.result["video_sha256"]),
        float(job.result["max_risk_score"]),
        request.impressions,
        request.unique_reports,
        request.trusted_reports,
        request.reports_last_hour,
    )
    recommendation = "expedite_human_review" if priority >= 0.65 else "monitor"
    return ReportSignalResponse(
        task_id=request.task_id,
        review_priority=priority,
        recommendation=recommendation,
    )
