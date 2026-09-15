from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from reel_moderation.config import EngineConfig
from reel_moderation.engine import ModerationEngine
from reel_moderation.evaluation import evaluate_cases, load_manifest, write_report

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
    scratch_dir: Annotated[
        Path, typer.Option(help="Ephemeral directory for decoded detector inputs")
    ] = Path("/moderation-scratch"),
    batch_size: Annotated[int, typer.Option(min=1, max=128)] = 16,
) -> None:
    """Moderate one video and emit a structured JSON report."""

    config = EngineConfig(
        backend=backend,
        execution_provider=execution_provider,
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
    scratch_dir: Annotated[
        Path, typer.Option(help="Ephemeral directory for decoded detector inputs")
    ] = Path("/moderation-scratch"),
) -> None:
    """Evaluate a labeled JSONL manifest without copying its media."""

    config = EngineConfig(
        backend=backend,
        execution_provider=execution_provider,
        scratch_dir=scratch_dir if backend == "nudenet" else None,
    )
    engine = ModerationEngine(config)
    report = evaluate_cases(load_manifest(manifest, split=split), engine.moderate, split=split)
    write_report(report, output)
    typer.echo(str(output))


if __name__ == "__main__":
    app()
