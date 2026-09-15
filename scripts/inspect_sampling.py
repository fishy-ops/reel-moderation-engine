"""Print numeric frame-change diagnostics without saving decoded frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np

from reel_moderation.sampling import _block_change


def inspect(path: Path, thumbnail_size: int = 96) -> list[dict[str, float | int]]:
    previous = None
    observations: list[dict[str, float | int]] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            thumbnail = frame.reformat(
                width=thumbnail_size,
                height=thumbnail_size,
                format="rgb24",
            ).to_ndarray()
            if previous is not None:
                global_change = float(
                    np.abs(thumbnail.astype(np.int16) - previous.astype(np.int16)).mean()
                    / 255.0
                )
                local_change = _block_change(thumbnail, previous)
                flash_change = abs(float(thumbnail.mean() - previous.mean())) / 255.0
                observations.append(
                    {
                        "frame": index,
                        "global": round(global_change, 6),
                        "local": round(local_change, 6),
                        "flash": round(flash_change, 6),
                        "max": round(max(global_change, local_change, flash_change), 6),
                    }
                )
            previous = thumbnail
    return sorted(observations, key=lambda item: float(item["max"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(inspect(args.video)[: args.top], indent=2))


if __name__ == "__main__":
    main()
