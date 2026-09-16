"""Reel Moderation Engine."""

from reel_moderation.engine import ModerationEngine
from reel_moderation.provenance import PIPELINE_VERSION
from reel_moderation.schema import ModerationReport, Verdict

__all__ = ["ModerationEngine", "ModerationReport", "Verdict"]
__version__ = PIPELINE_VERSION
