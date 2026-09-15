from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

from pydantic import BaseModel, Field

from reel_moderation.schema import ModerationReport, Verdict


class EvaluationCase(BaseModel):
    id: str = Field(min_length=1)
    video: Path
    expected_violation: bool
    split: str = "test"
    slices: list[str] = Field(default_factory=list)


class MetricCounts(BaseModel):
    total: int = 0
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    review: int = 0
    auto_block: int = 0
    errors: int = 0

    @property
    def recall(self) -> float | None:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else None

    @property
    def precision(self) -> float | None:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else None

    @property
    def false_positive_rate(self) -> float | None:
        denominator = self.false_positive + self.true_negative
        return self.false_positive / denominator if denominator else None

    def serialized(self) -> dict[str, int | float | None]:
        return {
            **self.model_dump(),
            "recall": self.recall,
            "precision": self.precision,
            "false_positive_rate": self.false_positive_rate,
        }


class EvaluationResult(BaseModel):
    case_id: str
    video_sha256: str
    expected_violation: bool
    verdict: Verdict
    duration_seconds: float
    processing_seconds: float
    slices: list[str]


class EvaluationReport(BaseModel):
    schema_version: str = "1.0"
    split: str
    metrics: dict[str, int | float | None]
    per_slice: dict[str, dict[str, int | float | None]]
    media_seconds: float
    processing_seconds: float
    throughput_x_realtime: float | None
    results: list[EvaluationResult]


def load_manifest(path: Path, split: str = "test") -> list[EvaluationCase]:
    manifest = path.expanduser().resolve()
    cases: list[EvaluationCase] = []
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            case = EvaluationCase.model_validate_json(line)
        except Exception as exc:
            raise ValueError(f"invalid manifest line {line_number}: {exc}") from exc
        if not case.video.is_absolute():
            case.video = (manifest.parent / case.video).resolve()
        if case.split == split:
            cases.append(case)
    if not cases:
        raise ValueError(f"manifest has no cases in split {split!r}")
    duplicate_ids = {case.id for case in cases if sum(item.id == case.id for item in cases) > 1}
    if duplicate_ids:
        raise ValueError(f"duplicate case ids: {', '.join(sorted(duplicate_ids))}")
    return cases


def _counts(items: Iterable[tuple[EvaluationCase, ModerationReport]]) -> MetricCounts:
    counts = MetricCounts()
    for case, report in items:
        counts.total += 1
        if report.verdict == Verdict.ERROR:
            counts.errors += 1
            continue
        predicted_positive = report.verdict != Verdict.APPROVED
        if report.verdict == Verdict.FLAG_FOR_HUMAN_REVIEW:
            counts.review += 1
        elif report.verdict == Verdict.AUTO_BLOCK:
            counts.auto_block += 1
        if predicted_positive and case.expected_violation:
            counts.true_positive += 1
        elif predicted_positive:
            counts.false_positive += 1
        elif case.expected_violation:
            counts.false_negative += 1
        else:
            counts.true_negative += 1
    return counts


def evaluate_cases(
    cases: list[EvaluationCase],
    moderate: Callable[[Path], ModerationReport],
    split: str = "test",
) -> EvaluationReport:
    completed: list[tuple[EvaluationCase, ModerationReport]] = []
    results: list[EvaluationResult] = []
    for case in cases:
        report = moderate(case.video)
        completed.append((case, report))
        results.append(
            EvaluationResult(
                case_id=case.id,
                video_sha256=report.video_sha256,
                expected_violation=case.expected_violation,
                verdict=report.verdict,
                duration_seconds=report.duration_seconds,
                processing_seconds=report.processing_seconds,
                slices=case.slices,
            )
        )

    slice_items: dict[str, list[tuple[EvaluationCase, ModerationReport]]] = defaultdict(list)
    for case, report in completed:
        for slice_name in case.slices:
            slice_items[slice_name].append((case, report))

    media_seconds = sum(report.duration_seconds for _, report in completed)
    processing_seconds = sum(report.processing_seconds for _, report in completed)
    return EvaluationReport(
        split=split,
        metrics=_counts(completed).serialized(),
        per_slice={
            name: _counts(items).serialized() for name, items in sorted(slice_items.items())
        },
        media_seconds=media_seconds,
        processing_seconds=processing_seconds,
        throughput_x_realtime=(media_seconds / processing_seconds if processing_seconds else None),
        results=results,
    )


def write_report(report: EvaluationReport, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
