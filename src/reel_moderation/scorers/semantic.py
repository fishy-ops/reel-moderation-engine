from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from reel_moderation.config import SemanticVerifierConfig
from reel_moderation.sampling import RGBFrame

EXPOSED_PROMPTS = (
    "a photograph with bare intimate anatomy visibly exposed",
    "a photograph showing an uncovered nipple or bare breast",
    "a nude adult body with intimate parts clearly visible",
)
COVERED_PROMPTS = (
    "a person in revealing but opaque clothing that covers intimate anatomy",
    "a person wearing a low-cut top, bra, swimsuit, or lingerie with no bare nipple",
    "a clothed person whose chest and intimate anatomy remain covered",
)
OTHER_PROMPTS = (
    "an ordinary photograph with no nudity",
    "a non-explicit scene where no intimate anatomy is visible",
    "a person in normal clothing",
)


@dataclass(frozen=True, slots=True)
class SemanticScore:
    exposed_probability: float
    covered_probability: float
    other_probability: float
    region_index: int
    component_signals: dict[str, float] = field(default_factory=dict)

    def signals(self) -> dict[str, float | str]:
        return {
            "semantic_exposed": round(self.exposed_probability, 6),
            "semantic_covered": round(self.covered_probability, 6),
            "semantic_other": round(self.other_probability, 6),
            "semantic_region_index": float(self.region_index),
            **{
                key: round(value, 6)
                for key, value in sorted(self.component_signals.items())
            },
        }


class SemanticVerifier(Protocol):
    name: str

    def score(self, regions: list[RGBFrame]) -> SemanticScore: ...

    def score_many(self, region_groups: list[list[RGBFrame]]) -> list[SemanticScore]: ...


class SiglipVerifier:
    name = "siglip2"

    def __init__(self, model_id: str, revision: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "SigLIP verifier dependencies are not installed in this image"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("SigLIP verifier requires CUDA in the production worker")
        self._torch = torch
        self._device = torch.device("cuda")
        self._processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self._model = AutoModel.from_pretrained(
            model_id,
            dtype=torch.float16,
            revision=revision,
        ).to(self._device)
        self._model.eval()
        self._prompts = [*EXPOSED_PROMPTS, *COVERED_PROMPTS, *OTHER_PROMPTS]

    def score(self, regions: list[RGBFrame]) -> SemanticScore:
        if not regions:
            raise ValueError("at least one region is required for semantic verification")
        return self.score_many([regions])[0]

    def score_many(self, region_groups: list[list[RGBFrame]]) -> list[SemanticScore]:
        if not region_groups:
            return []
        if any(not regions for regions in region_groups):
            raise ValueError("each semantic verification group requires a region")
        regions = [region for group in region_groups for region in group]
        inputs = self._processor(
            text=self._prompts,
            images=regions,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output = self._model(**inputs)
            logits = output.logits_per_image.float()
            grouped = self._torch.stack(
                [
                    logits[:, 0:3].mean(dim=1),
                    logits[:, 3:6].mean(dim=1),
                    logits[:, 6:9].mean(dim=1),
                ],
                dim=1,
            )
            probabilities = self._torch.softmax(grouped, dim=1).cpu().numpy()
        scores: list[SemanticScore] = []
        offset = 0
        for regions_in_group in region_groups:
            group = probabilities[offset : offset + len(regions_in_group)]
            region_index = int(np.argmax(group[:, 0]))
            exposed, covered, other = group[region_index]
            scores.append(
                SemanticScore(
                    exposed_probability=float(exposed),
                    covered_probability=float(covered),
                    other_probability=float(other),
                    region_index=region_index,
                    component_signals={
                        "siglip_exposed": float(exposed),
                        "siglip_covered": float(covered),
                        "siglip_other": float(other),
                    },
                )
            )
            offset += len(regions_in_group)
        return scores


class FalconNsfwVerifier:
    name = "falcon-nsfw"

    def __init__(self, model_id: str, revision: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "Falcon NSFW verifier dependencies are not installed in this image"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("Falcon NSFW verifier requires CUDA in production")
        self._torch = torch
        self._device = torch.device("cuda")
        self._processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        self._model = AutoModelForImageClassification.from_pretrained(
            model_id,
            dtype=torch.float16,
            revision=revision,
        ).to(self._device)
        self._model.eval()
        labels = {
            int(index): str(label).lower()
            for index, label in self._model.config.id2label.items()
        }
        matches = [index for index, label in labels.items() if label == "nsfw"]
        if len(matches) != 1:
            raise RuntimeError("Falcon classifier does not expose exactly one NSFW label")
        self._nsfw_index = matches[0]

    def score(self, regions: list[RGBFrame]) -> SemanticScore:
        if not regions:
            raise ValueError("at least one region is required for semantic verification")
        return self.score_many([regions])[0]

    def score_many(self, region_groups: list[list[RGBFrame]]) -> list[SemanticScore]:
        if not region_groups:
            return []
        if any(not regions for regions in region_groups):
            raise ValueError("each semantic verification group requires a region")
        regions = [region for group in region_groups for region in group]
        inputs = self._processor(images=regions, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits.float()
            probabilities = self._torch.softmax(logits, dim=1).cpu().numpy()
        scores: list[SemanticScore] = []
        offset = 0
        for regions_in_group in region_groups:
            group = probabilities[offset : offset + len(regions_in_group)]
            region_index = int(np.argmax(group[:, self._nsfw_index]))
            nsfw_probability = float(group[region_index, self._nsfw_index])
            scores.append(
                SemanticScore(
                    exposed_probability=nsfw_probability,
                    covered_probability=0.0,
                    other_probability=1.0 - nsfw_probability,
                    region_index=region_index,
                    component_signals={"falcon_nsfw": nsfw_probability},
                )
            )
            offset += len(regions_in_group)
        return scores


class EnsembleVerifier:
    name = "siglip2+falcon-nsfw"

    def __init__(
        self,
        siglip_model_id: str,
        classifier_model_id: str,
        siglip_model_revision: str | None = None,
        classifier_model_revision: str | None = None,
    ) -> None:
        self._siglip = SiglipVerifier(siglip_model_id, siglip_model_revision)
        self._classifier = FalconNsfwVerifier(
            classifier_model_id,
            classifier_model_revision,
        )

    def score(self, regions: list[RGBFrame]) -> SemanticScore:
        return self.score_many([regions])[0]

    def score_many(self, region_groups: list[list[RGBFrame]]) -> list[SemanticScore]:
        siglip_scores = self._siglip.score_many(region_groups)
        classifier_scores = self._classifier.score_many(region_groups)
        return [
            self._combine(siglip, classifier)
            for siglip, classifier in zip(
                siglip_scores, classifier_scores, strict=True
            )
        ]

    @staticmethod
    def _combine(siglip: SemanticScore, classifier: SemanticScore) -> SemanticScore:
        # A geometric mean requires both differently trained models to contribute.
        # It prevents the broad binary classifier from dominating on revealing but
        # covered content merely because its standalone score is near one.
        exposed = float(
            np.sqrt(siglip.exposed_probability * classifier.exposed_probability)
        )
        remaining = 1.0 - exposed
        context_total = siglip.covered_probability + siglip.other_probability
        covered = (
            remaining * siglip.covered_probability / context_total
            if context_total
            else remaining / 2.0
        )
        other = remaining - covered
        return SemanticScore(
            exposed_probability=exposed,
            covered_probability=covered,
            other_probability=other,
            region_index=(
                classifier.region_index
                if classifier.exposed_probability >= siglip.exposed_probability
                else siglip.region_index
            ),
            component_signals={
                **siglip.component_signals,
                **classifier.component_signals,
                "siglip_region_index": float(siglip.region_index),
                "falcon_region_index": float(classifier.region_index),
            },
        )


def adjust_risk(
    raw_risk: float,
    score: SemanticScore,
    config: SemanticVerifierConfig,
) -> tuple[float, str]:
    covered_dominates = (
        score.covered_probability >= config.covered_probability_threshold
        and score.covered_probability - score.exposed_probability >= config.covered_margin
    )
    exposed_dominates = (
        score.exposed_probability >= config.exposed_probability_threshold
        and score.exposed_probability - score.covered_probability >= config.exposed_margin
    )
    if covered_dominates:
        return raw_risk * config.covered_risk_multiplier, "covered"
    if exposed_dominates:
        boosted = 1.0 - (1.0 - raw_risk) * (
            1.0 - config.exposed_boost * score.exposed_probability
        )
        return min(1.0, max(boosted, config.exposed_review_floor)), "exposed"
    return raw_risk, "uncertain"
