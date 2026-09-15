from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from reel_moderation.sampling import FrameCandidate, RGBFrame
from reel_moderation.schema import Detection
from reel_moderation.scorers.base import FrameScore, FrameScorer

# These detector classes represent visible anatomy. Covered-class detections are
# deliberately not treated as violations.
EXPOSED_CLASS_WEIGHTS: dict[str, float] = {
    "ANUS_EXPOSED": 1.00,
    "FEMALE_BREAST_EXPOSED": 0.95,
    "FEMALE_GENITALIA_EXPOSED": 1.00,
    "MALE_BREAST_EXPOSED": 0.35,
    "MALE_GENITALIA_EXPOSED": 1.00,
    "BUTTOCKS_EXPOSED": 0.72,
}


def _risk_from_detections(detections: list[Detection]) -> float:
    # The same anatomy can be detected in the full image and several overlapping
    # crops. Count only the strongest detection per label so tiling improves recall
    # without manufacturing repeated evidence from one object.
    strongest_by_label: dict[str, float] = {}
    for detection in detections:
        strongest_by_label[detection.label] = max(
            strongest_by_label.get(detection.label, 0.0), detection.score
        )
    complement = 1.0
    for label, score in strongest_by_label.items():
        weight = EXPOSED_CLASS_WEIGHTS.get(label, 0.0)
        complement *= 1.0 - min(1.0, score * weight)
    return min(1.0, max(0.0, 1.0 - complement))


def _crops(rgb: RGBFrame) -> list[tuple[str, RGBFrame]]:
    """Return full frame and overlapping crops to recover small reflections."""

    height, width = rgb.shape[:2]
    crops: list[tuple[str, RGBFrame]] = [("full", rgb)]
    if min(height, width) < 320:
        return crops

    overlap = 0.12
    half_h, half_w = int(height * (0.5 + overlap)), int(width * (0.5 + overlap))
    starts = [
        (0, 0, "top_left"),
        (0, max(width - half_w, 0), "top_right"),
        (max(height - half_h, 0), 0, "bottom_left"),
        (max(height - half_h, 0), max(width - half_w, 0), "bottom_right"),
    ]
    for y, x, name in starts:
        crops.append((name, rgb[y : y + half_h, x : x + half_w]))
    return crops


class NudeNetScorer(FrameScorer):
    """Fine-grained detector with in-memory/tmpfs evidence handling.

    NudeNet currently exposes a file-oriented API. Crops are therefore encoded
    into a caller-controlled scratch directory and deleted immediately after each
    batch. Production containers mount this directory as tmpfs, keeping decoded
    evidence out of persistent storage.
    """

    name = "nudenet"

    def __init__(
        self,
        scratch_dir: Path,
        minimum_confidence: float = 0.25,
        tile_small_regions: bool = True,
        tile_interval_seconds: float = 2.0,
        execution_provider: str = "auto",
    ) -> None:
        if scratch_dir is None:
            raise ValueError("NudeNet requires an explicit scratch_dir")
        scratch_dir.mkdir(parents=True, exist_ok=True)
        self.scratch_dir = scratch_dir
        self.minimum_confidence = minimum_confidence
        self.tile_small_regions = tile_small_regions
        self.tile_interval_seconds = tile_interval_seconds
        self._last_tiled_at = float("-inf")

        try:
            import nudenet
            import onnxruntime
            from nudenet import NudeDetector
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "NudeNet backend is not installed; install the 'nudenet' extra"
            ) from exc
        self.detector = NudeDetector()
        available = onnxruntime.get_available_providers()
        wants_cuda = execution_provider in {"auto", "cuda"}
        if wants_cuda and "CUDAExecutionProvider" in available:
            model_path = Path(nudenet.__file__).parent / "320n.onnx"
            self.detector.onnx_session = onnxruntime.InferenceSession(
                str(model_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.detector.input_name = self.detector.onnx_session.get_inputs()[0].name
        elif execution_provider == "cuda":
            raise RuntimeError("CUDA execution was required but is not available")
        providers = self.detector.onnx_session.get_providers()
        if execution_provider == "cuda" and "CUDAExecutionProvider" not in providers:
            raise RuntimeError(
                "CUDA execution was required but ONNX Runtime fell back to CPU; "
                "verify that the CUDA and cuDNN runtime versions match"
            )
        self.name = f"nudenet/{providers[0]}" if providers else "nudenet/unknown"

    def _should_tile(self, frame: FrameCandidate) -> bool:
        if not self.tile_small_regions:
            return False
        trigger_reasons = {
            "scene_change",
            "localized_overlay_change",
            "luminance_flash",
            "dense_burst",
        }
        triggered = bool(frame.reasons & trigger_reasons)
        interval_due = frame.timestamp_seconds - self._last_tiled_at >= self.tile_interval_seconds
        should_tile = frame.frame_index == 0 or triggered or interval_due
        if should_tile:
            self._last_tiled_at = frame.timestamp_seconds
        return should_tile

    def _detect(self, rgb: RGBFrame, crop_name: str, directory: Path) -> list[Detection]:
        from PIL import Image

        image_path = directory / f"{crop_name}.jpg"
        Image.fromarray(rgb).save(image_path, format="JPEG", quality=92)
        raw: list[dict[str, Any]] = self.detector.detect(str(image_path))
        detections: list[Detection] = []
        for item in raw:
            label = str(item.get("class", "UNKNOWN")).upper()
            score = float(item.get("score", 0.0))
            if score < self.minimum_confidence or label not in EXPOSED_CLASS_WEIGHTS:
                continue
            raw_box = item.get("box")
            box = tuple(int(value) for value in raw_box) if raw_box else None
            detections.append(
                Detection(
                    label=label,
                    score=min(max(score, 0.0), 1.0),
                    box=box,  # type: ignore[arg-type]
                    source=self.name,
                    crop=crop_name,
                )
            )
        return detections

    def score_batch(self, frames: list[FrameCandidate]) -> list[FrameScore]:
        output: list[FrameScore] = []
        for frame in frames:
            with tempfile.TemporaryDirectory(dir=self.scratch_dir) as tmp:
                directory = Path(tmp)
                regions = _crops(frame.rgb) if self._should_tile(frame) else [("full", frame.rgb)]
                detections: list[Detection] = []
                for crop_name, crop in regions:
                    detections.extend(self._detect(crop, crop_name, directory))

            risk = _risk_from_detections(detections)
            output.append(FrameScore(risk_score=min(risk, 1.0), detections=detections))
        return output
