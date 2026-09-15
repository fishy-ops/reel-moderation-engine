from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Literal

ChallengeVariant = Literal["mirror", "compression", "blur", "reflection", "occlusion"]


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def generate_challenge(
    input_path: Path,
    output_path: Path,
    variant: ChallengeVariant,
    event_start_seconds: float = 1.0,
    event_end_seconds: float = 2.0,
) -> None:
    """Create an Aurora-only derivative from operator-approved source media."""

    configured_root = os.environ.get("MODERATION_DERIVED_MEDIA_ROOT")
    if not configured_root:
        raise RuntimeError("MODERATION_DERIVED_MEDIA_ROOT must be explicitly configured")
    allowed_root = Path(configured_root).resolve()
    destination = output_path.resolve()
    if not _inside(destination, allowed_root):
        raise ValueError("challenge output is outside the configured derived-media root")
    destination.parent.mkdir(parents=True, exist_ok=True)

    filters = {
        "mirror": "hflip",
        "compression": "scale=iw/2:ih/2,scale=iw*2:ih*2",
        "blur": "gblur=sigma=1.2",
        "reflection": (
            "split=2[base][copy];[copy]hflip,scale=iw*0.30:ih*0.30[ref];"
            "[base][ref]overlay=W-w-20:20"
        ),
        "occlusion": (
            f"drawbox=x=iw*0.25:y=ih*0.25:w=iw*0.5:h=ih*0.5:color=black:t=fill:"
            f"enable='between(t,{event_start_seconds},{event_end_seconds})'"
        ),
    }
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path.resolve()),
        "-vf",
        filters[variant],
        "-c:v",
        "libx264",
        "-crf",
        "28",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(destination),
    ]
    subprocess.run(command, check=True)
