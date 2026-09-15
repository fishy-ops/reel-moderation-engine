import pytest

from reel_moderation.config import SemanticVerifierConfig
from reel_moderation.scorers.semantic import EnsembleVerifier, SemanticScore, adjust_risk


class FixedVerifier:
    def __init__(self, score: SemanticScore) -> None:
        self._score = score

    def score(self, regions):
        return self._score


def test_covered_semantic_consensus_reduces_risk() -> None:
    config = SemanticVerifierConfig(mode="enforce")
    score = SemanticScore(0.1, 0.8, 0.1, 0)
    adjusted, decision = adjust_risk(0.7, score, config)
    assert decision == "covered"
    assert adjusted == pytest.approx(0.315)


def test_exposed_semantic_consensus_boosts_risk() -> None:
    config = SemanticVerifierConfig(mode="enforce")
    score = SemanticScore(0.8, 0.1, 0.1, 0)
    adjusted, decision = adjust_risk(0.6, score, config)
    assert decision == "exposed"
    assert adjusted > 0.6


def test_uncertain_semantics_do_not_change_risk() -> None:
    config = SemanticVerifierConfig(mode="enforce")
    score = SemanticScore(0.4, 0.35, 0.25, 0)
    adjusted, decision = adjust_risk(0.6, score, config)
    assert decision == "uncertain"
    assert adjusted == 0.6


def test_ensemble_preserves_independent_classifier_signal() -> None:
    verifier = object.__new__(EnsembleVerifier)
    verifier._siglip = FixedVerifier(
        SemanticScore(0.8, 0.15, 0.05, 0, {"siglip_exposed": 0.8})
    )
    verifier._classifier = FixedVerifier(
        SemanticScore(0.8, 0.0, 0.2, 1, {"falcon_nsfw": 0.8})
    )

    score = verifier.score([object()])

    assert score.exposed_probability == pytest.approx(0.8)
    assert score.region_index == 1
    assert score.component_signals["falcon_nsfw"] == 0.8

