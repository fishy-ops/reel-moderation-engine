from pathlib import Path

from reel_moderation.config import EngineConfig
from reel_moderation.provenance import decision_config_sha256, pipeline_source_sha256


def test_decision_fingerprint_ignores_storage_location() -> None:
    left = EngineConfig(scratch_dir=Path("/one"))
    right = EngineConfig(scratch_dir=Path("/two"))

    assert decision_config_sha256(left) == decision_config_sha256(right)


def test_decision_fingerprint_changes_with_policy() -> None:
    baseline = EngineConfig()
    candidate = EngineConfig(policy={"review_threshold": 0.5})

    assert decision_config_sha256(baseline) != decision_config_sha256(candidate)


def test_decision_fingerprint_changes_with_model_revision() -> None:
    baseline = EngineConfig()
    candidate = EngineConfig(semantic_verifier={"model_revision": "different"})

    assert decision_config_sha256(baseline) != decision_config_sha256(candidate)


def test_source_fingerprint_changes_with_source_bytes(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    first = pipeline_source_sha256(tmp_path)
    pipeline_source_sha256.cache_clear()
    source.write_text("value = 2\n", encoding="utf-8")

    second = pipeline_source_sha256(tmp_path)

    assert len(first) == 64
    assert first != second
