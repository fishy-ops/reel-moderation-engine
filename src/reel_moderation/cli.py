from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, cast

import typer

from reel_moderation.challenges import ChallengeVariant, generate_challenge
from reel_moderation.config import EngineConfig, PolicyConfig
from reel_moderation.engine import ModerationEngine
from reel_moderation.evaluation import (
    EvaluationReport,
    compare_evaluation_reports,
    evaluate_cases,
    load_manifest,
    write_report,
)
from reel_moderation.feedback import (
    GroundTruth,
    record_feedback,
    summarize_feedback,
    summarize_feedback_by_configuration,
)
from reel_moderation.fingerprint import (
    VideoFingerprint,
    fingerprint_similarity,
    fingerprint_video,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Adaptive video moderation for brief and evasive visual-policy violations.",
)


@app.command()
def moderate(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    backend: Annotated[str, typer.Option(help="nudenet or dry-run")] = "nudenet",
    execution_provider: Annotated[
        str, typer.Option(help="auto, cuda, or cpu")
    ] = "auto",
    semantic_backend: Annotated[
        str, typer.Option(help="none, siglip, or ensemble")
    ] = "none",
    semantic_mode: Annotated[
        str, typer.Option(help="audit, review, or enforce")
    ] = "audit",
    scratch_dir: Annotated[
        Path, typer.Option(help="Ephemeral directory for decoded detector inputs")
    ] = Path("/moderation-scratch"),
    batch_size: Annotated[int, typer.Option(min=1, max=128)] = 16,
    review_threshold: Annotated[float, typer.Option(min=0, max=1)] = 0.42,
) -> None:
    """Moderate one video and emit a structured JSON report."""

    config = EngineConfig(
        backend=backend,
        execution_provider=execution_provider,
        semantic_verifier={"backend": semantic_backend, "mode": semantic_mode},
        policy=PolicyConfig(review_threshold=review_threshold),
        scratch_dir=scratch_dir if backend == "nudenet" else None,
        batch_size=batch_size,
    )
    report = ModerationEngine(config).moderate(video)
    serialized = report.model_dump_json(indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
        typer.echo(str(output))
    else:
        typer.echo(serialized)


@app.command("show-config")
def show_config() -> None:
    """Print the default sampling and policy configuration."""

    typer.echo(json.dumps(EngineConfig().model_dump(mode="json"), indent=2))


@app.command("evaluate")
def evaluate(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    split: Annotated[str, typer.Option(help="Manifest split to evaluate")] = "test",
    backend: Annotated[str, typer.Option(help="nudenet or dry-run")] = "nudenet",
    execution_provider: Annotated[
        str, typer.Option(help="auto, cuda, or cpu")
    ] = "auto",
    semantic_backend: Annotated[
        str, typer.Option(help="none, siglip, or ensemble")
    ] = "none",
    semantic_mode: Annotated[
        str, typer.Option(help="audit, review, or enforce")
    ] = "audit",
    scratch_dir: Annotated[
        Path, typer.Option(help="Ephemeral directory for decoded detector inputs")
    ] = Path("/moderation-scratch"),
    review_threshold: Annotated[float, typer.Option(min=0, max=1)] = 0.42,
    event_tolerance_seconds: Annotated[float, typer.Option(min=0, max=5)] = 0.5,
) -> None:
    """Evaluate a labeled JSONL manifest without copying its media."""

    config = EngineConfig(
        backend=backend,
        execution_provider=execution_provider,
        semantic_verifier={"backend": semantic_backend, "mode": semantic_mode},
        policy=PolicyConfig(review_threshold=review_threshold),
        scratch_dir=scratch_dir if backend == "nudenet" else None,
    )
    engine = ModerationEngine(config)
    report = evaluate_cases(
        load_manifest(manifest, split=split),
        engine.moderate,
        split=split,
        event_tolerance_seconds=event_tolerance_seconds,
    )
    write_report(report, output)
    typer.echo(str(output))


@app.command("compare-evaluations")
def compare_evaluations_command(
    baseline: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    candidate: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    max_recall_drop: Annotated[float, typer.Option(min=0, max=1)] = 0.0,
    max_false_positive_rate_increase: Annotated[
        float, typer.Option(min=0, max=1)
    ] = 0.0,
    minimum_throughput_ratio: Annotated[float, typer.Option(min=0)] = 0.9,
) -> None:
    """Compare media-free reports and exit nonzero on a regression."""

    baseline_report = EvaluationReport.model_validate_json(
        baseline.read_text(encoding="utf-8")
    )
    candidate_report = EvaluationReport.model_validate_json(
        candidate.read_text(encoding="utf-8")
    )
    comparison = compare_evaluation_reports(
        baseline_report,
        candidate_report,
        max_recall_drop=max_recall_drop,
        max_false_positive_rate_increase=max_false_positive_rate_increase,
        minimum_throughput_ratio=minimum_throughput_ratio,
    )
    typer.echo(comparison.model_dump_json(indent=2))
    if not comparison.passed:
        raise typer.Exit(code=1)


@app.command("fingerprint")
def fingerprint_command(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    sample_fps: Annotated[float, typer.Option(min=0.25, max=10)] = 2.0,
) -> None:
    """Create a media-free perceptual video fingerprint."""

    result = fingerprint_video(video, sample_fps=sample_fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(str(output))


@app.command("compare-fingerprints")
def compare_fingerprints(
    left: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    right: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Compare two perceptual fingerprints without accessing their videos."""

    left_fingerprint = VideoFingerprint.model_validate_json(left.read_text(encoding="utf-8"))
    right_fingerprint = VideoFingerprint.model_validate_json(right.read_text(encoding="utf-8"))
    typer.echo(f"{fingerprint_similarity(left_fingerprint, right_fingerprint):.6f}")


@app.command("record-feedback")
def record_feedback_command(
    report: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    ground_truth: Annotated[str, typer.Option(help="violation, safe, or uncertain")],
    reason: Annotated[str, typer.Option(help="Short policy-grounded reviewer reason")],
    database: Annotated[Path, typer.Option()] = Path("/data/reports/feedback.sqlite"),
    reviewer: Annotated[str, typer.Option()] = "operator",
    event_start: Annotated[float | None, typer.Option(min=0)] = None,
    event_end: Annotated[float | None, typer.Option(min=0)] = None,
) -> None:
    """Record a reviewer decision without storing evidence imagery."""

    if ground_truth not in {"violation", "safe", "uncertain"}:
        raise typer.BadParameter("ground-truth must be violation, safe, or uncertain")
    feedback_id = record_feedback(
        database,
        report,
        cast(GroundTruth, ground_truth),
        reason,
        reviewer=reviewer,
        event_start_seconds=event_start,
        event_end_seconds=event_end,
    )
    typer.echo(str(feedback_id))


@app.command("feedback-summary")
def feedback_summary_command(
    database: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Summarize adjudicated feedback outcomes."""

    typer.echo(summarize_feedback(database).model_dump_json(indent=2))


@app.command("feedback-summary-by-configuration")
def feedback_summary_by_configuration_command(
    database: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Group reviewer outcomes by exact pipeline and decision configuration."""

    summaries = summarize_feedback_by_configuration(database)
    typer.echo(json.dumps([item.model_dump() for item in summaries], indent=2))


@app.command("generate-challenge")
def generate_challenge_command(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    variant: Annotated[
        str, typer.Option(help="mirror, compression, blur, reflection, or occlusion")
    ],
    event_start: Annotated[float, typer.Option(min=0)] = 1.0,
    event_end: Annotated[float, typer.Option(min=0)] = 2.0,
) -> None:
    """Generate one approved Aurora-only adversarial video derivative."""

    allowed = {"mirror", "compression", "blur", "reflection", "occlusion"}
    if variant not in allowed:
        raise typer.BadParameter(f"variant must be one of {', '.join(sorted(allowed))}")
    generate_challenge(
        video,
        output,
        cast(ChallengeVariant, variant),
        event_start_seconds=event_start,
        event_end_seconds=event_end,
    )
    typer.echo(str(output))


if __name__ == "__main__":
    app()
