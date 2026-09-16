from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from reel_moderation.config import EngineConfig

PIPELINE_VERSION = "0.1.0"


@lru_cache(maxsize=1)
def pipeline_source_sha256(package_root: Path | None = None) -> str:
    """Hash installed pipeline source without depending on Git metadata."""

    root = (package_root or Path(__file__).parent).resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def decision_config_sha256(config: EngineConfig) -> str:
    """Fingerprint every configured input that can affect a moderation decision."""

    decision_config = config.model_dump(
        mode="json",
        exclude={"scratch_dir", "keep_evidence_images"},
    )
    canonical = json.dumps(
        decision_config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
