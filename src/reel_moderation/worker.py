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
            "classifier_model_id": os.environ.get(
                "MODERATION_CLASSIFIER_MODEL", "Falconsai/nsfw_image_detection"
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
        temporary_report.write_text(
            report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_report.replace(report_path)
        return report.model_dump(mode="json")
    finally:
        if delete_inputs:
            path.unlink(missing_ok=True)
