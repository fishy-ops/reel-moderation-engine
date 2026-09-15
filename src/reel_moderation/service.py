from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from redis import Redis
from rq import Queue
from rq.job import Job

from reel_moderation.feedback import record_feedback, update_report_aggregate
from reel_moderation.worker import moderate_job

DATA_DIR = Path(os.environ.get("MODERATION_DATA_DIR", "/data"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(2 * 1024**3)))
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


class FeedbackRequest(BaseModel):
    task_id: str
    ground_truth: Literal["violation", "safe", "uncertain"]
    reason: str = Field(min_length=1, max_length=500)
    reviewer: str = Field(default="operator", min_length=1, max_length=100)
    event_start_seconds: float | None = Field(default=None, ge=0)
    event_end_seconds: float | None = Field(default=None, ge=0)


class ReportSignalRequest(BaseModel):
    task_id: str
    impressions: int = Field(ge=0)
    unique_reports: int = Field(ge=0)
    trusted_reports: int = Field(ge=0)
    reports_last_hour: int = Field(ge=0)


class ReportSignalResponse(BaseModel):
    task_id: str
    review_priority: float
    recommendation: str


@app.get("/health")
def health() -> dict[str, str]:
    redis_connection.ping()
    return {"status": "ok"}


@app.post(
    "/api/v1/moderate",
    response_model=SubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_video(file: Annotated[UploadFile, File(...)]) -> SubmitResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="unsupported video container")

    task_id = uuid.uuid4().hex
    incoming = DATA_DIR / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
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

    queue.enqueue(
        moderate_job,
        str(destination),
        job_id=task_id,
        result_ttl=7 * 24 * 3600,
        failure_ttl=7 * 24 * 3600,
    )
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
    return StatusResponse(task_id=task_id, status=str(job_status))


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
