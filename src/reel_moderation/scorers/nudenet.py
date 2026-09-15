from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from reel_moderation.config import SemanticVerifierConfig
from reel_moderation.sampling import FrameCandidate, RGBFrame
from reel_moderation.schema import Detection
from reel_moderation.scorers.base import FrameScore, FrameScorer
from reel_moderation.scorers.semantic import SemanticVerifier, adjust_risk

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

COVERED_CONTEXT_CLASSES = {
    "ANUS_COVERED",
    "BUTTOCKS_COVERED",
    "FEMALE_BREAST_COVERED",
    "FEMALE_GENITALIA_COVERED",
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


def _apply_semantic_mode(
    raw_risk: float,
    adjusted_risk: float,
    decision: str,
    mode: str,
) -> float:
    if mode == "enforce":
        return adjusted_risk
    if mode == "review" and decision == "exposed":
        return adjusted_risk
    return raw_risk


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
        semantic_verifier: SemanticVerifier | None = None,
        semantic_config: SemanticVerifierConfig | None = None,
    ) -> None:
        if scratch_dir is None:
            raise ValueError("NudeNet requires an explicit scratch_dir")
        scratch_dir.mkdir(parents=True, exist_ok=True)
        self.scratch_dir = scratch_dir
        self.minimum_confidence = minimum_confidence
        self.tile_small_regions = tile_small_regions
        self.tile_interval_seconds = tile_interval_seconds
        self._last_tiled_at = float("-inf")
        self._last_semantic_at = float("-inf")
        self.semantic_verifier = semantic_verifier
        self.semantic_config = semantic_config or SemanticVerifierConfig()

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
        if self.semantic_verifier:
            self.name += f"+{self.semantic_verifier.name}/{self.semantic_config.mode}"

    def reset(self) -> None:
        self._last_tiled_at = float("-inf")
        self._last_semantic_at = float("-inf")

    def _should_verify_semantically(self, frame: FrameCandidate, raw_risk: float) -> bool:
        if not self.semantic_verifier:
            return False
        elapsed = frame.timestamp_seconds - self._last_semantic_at
        if elapsed < self.semantic_config.minimum_interval_seconds:
            return False
        adaptive_reasons = {
            "scene_change",
            "localized_overlay_change",
            "luminance_flash",
            "dense_burst",
        }
        should_verify = (
            frame.frame_index == 0
            or raw_risk >= self.semantic_config.trigger_threshold
            or elapsed >= self.semantic_config.baseline_interval_seconds
            or bool(frame.reasons & adaptive_reasons)
        )
        if should_verify:
            self._last_semantic_at = frame.timestamp_seconds
        return should_verify

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
            policy_or_context = label in EXPOSED_CLASS_WEIGHTS or label in COVERED_CONTEXT_CLASSES
            if score < self.minimum_confidence or not policy_or_context:
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

            raw_risk = _risk_from_detections(detections)
            risk = raw_risk
            signals: dict[str, float | str] = {}
            if self._should_verify_semantically(frame, raw_risk):
                # Give the independent verifier full-frame and tiled views even
                # when the primary detector found no candidate. This is the path
                # that can recover complete primary-model misses.
                semantic_regions = _crops(frame.rgb)
                semantic_score = self.semantic_verifier.score(
                    [region for _, region in semantic_regions]
                )
                adjusted, decision = adjust_risk(raw_risk, semantic_score, self.semantic_config)
                signals = {**semantic_score.signals(), "semantic_decision": decision}
                risk = _apply_semantic_mode(
                    raw_risk,
                    adjusted,
                    decision,
                    self.semantic_config.mode,
                )
            output.append(
                FrameScore(
                    risk_score=min(risk, 1.0),
                    raw_risk_score=min(raw_risk, 1.0),
                    detections=detections,
                    verifier_signals=signals,
                )
            )
        return output
