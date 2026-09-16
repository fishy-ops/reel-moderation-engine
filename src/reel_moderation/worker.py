from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from reel_moderation.config import EngineConfig, PolicyConfig
from reel_moderation.engine import ModerationEngine


@lru_cache(maxsize=1)
def _engine_from_environment() -> ModerationEngine:
    scratch_dir = Path(os.environ.get("MODERATION_SCRATCH_DIR", "/moderation-scratch"))
    config = EngineConfig(
        backend=os.environ.get("MODERATION_BACKEND", "nudenet"),
        execution_provider=os.environ.get("MODERATION_EXECUTION_PROVIDER", "auto"),
        semantic_verifier={
            "backend": os.environ.get("MODERATION_SEMANTIC_BACKEND", "none"),
            "mode": os.environ.get("MODERATION_SEMANTIC_MODE", "audit"),
            "model_id": os.environ.get(
                "MODERATION_SEMANTIC_MODEL", "google/siglip2-base-patch16-224"
            ),
            "model_revision": os.environ.get(
                "MODERATION_SEMANTIC_MODEL_REVISION",
                "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
            ),
            "classifier_model_id": os.environ.get(
                "MODERATION_CLASSIFIER_MODEL", "Falconsai/nsfw_image_detection"
            ),
            "classifier_model_revision": os.environ.get(
                "MODERATION_CLASSIFIER_MODEL_REVISION",
                "96cb0d0342c7afb80cab76ecc58b265fa44da256",
            ),
        },
        policy=PolicyConfig(
            review_threshold=float(os.environ.get("MODERATION_REVIEW_THRESHOLD", "0.42")),
            block_threshold=float(os.environ.get("MODERATION_BLOCK_THRESHOLD", "0.82")),
        ),
        scratch_dir=scratch_dir,
        batch_size=int(os.environ.get("MODERATION_BATCH_SIZE", "16")),
    )
    return ModerationEngine(config)


def moderate_job(video_path: str) -> dict:
    path = Path(video_path).resolve()
    data_dir = Path(os.environ.get("MODERATION_DATA_DIR", "/data")).resolve()
    delete_inputs = os.environ.get("DELETE_INPUTS_AFTER_PROCESSING", "false").lower() == "true"
    if data_dir not in path.parents:
        raise ValueError("video path is outside the configured data directory")

    try:
        report = _engine_from_environment().moderate(path)
        reports_dir = data_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{path.stem}.json"
        temporary_report = report_path.with_suffix(".json.tmp")
        try:
            temporary_report.write_text(
                report.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_report.replace(report_path)
        finally:
            temporary_report.unlink(missing_ok=True)
        return report.model_dump(mode="json")
    finally:
        if delete_inputs:
            path.unlink(missing_ok=True)
