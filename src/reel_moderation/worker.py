from __future__ import annotations

import os
from pathlib import Path

from reel_moderation.config import EngineConfig
from reel_moderation.engine import ModerationEngine


def moderate_job(video_path: str) -> dict:
    path = Path(video_path).resolve()
    data_dir = Path(os.environ.get("MODERATION_DATA_DIR", "/data")).resolve()
    scratch_dir = Path(os.environ.get("MODERATION_SCRATCH_DIR", "/moderation-scratch"))
    delete_inputs = os.environ.get("DELETE_INPUTS_AFTER_PROCESSING", "false").lower() == "true"
    if data_dir not in path.parents:
        raise ValueError("video path is outside the configured data directory")

    config = EngineConfig(
        backend=os.environ.get("MODERATION_BACKEND", "nudenet"),
        execution_provider=os.environ.get("MODERATION_EXECUTION_PROVIDER", "auto"),
        scratch_dir=scratch_dir,
        batch_size=int(os.environ.get("MODERATION_BATCH_SIZE", "16")),
    )
    try:
        report = ModerationEngine(config).moderate(path)
        reports_dir = data_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{path.stem}.json"
        report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return report.model_dump(mode="json")
    finally:
        if delete_inputs:
            path.unlink(missing_ok=True)
