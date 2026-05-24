"""Runtime configuration for the classification service.

Lives outside ``rie-contracts`` because these knobs are adapter-local — they
tune how the classify package behaves, not the shape of any port or claim.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ClassifierConfig(BaseModel):
    """Adapter-local knobs for ``ClassificationService``.

    Attributes:
        multilingual_enabled: When ``True`` (default) the service runs
            language detection on every incoming clause, threads the detected
            code into the prompt, and selects per-language keyword lists
            from the pillar config (falling back to ``en`` with a
            ``language_match=False`` flag when the pillar has no entry for
            the detected language). When ``False``, language detection is
            skipped and every clause is treated as ``en``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    multilingual_enabled: bool = Field(default=True)


__all__ = ["ClassifierConfig"]
