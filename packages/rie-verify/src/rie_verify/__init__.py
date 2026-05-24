"""4-gate verification adapter.

Implements `VerifierPort`:

  * Tier A (deterministic, runs on every claim): span-existence, verbatim-match.
  * Tier B (model-based, runs only on Tier-A survivors): entailment, self-consistency.

Honest semantics — *any* disagreement between the NLI model and the second LLM
flags the claim for human review (never auto-resolved). See §6.5.
"""

from __future__ import annotations

from rie_verify.nli import FakeNliBackend, NliBackend, NliScores, TransformersNliBackend
from rie_verify.second_llm import FakeSecondLlm, OllamaSecondLlm, SecondLlmBackend
from rie_verify.service import VerificationService

__all__ = [
    "FakeNliBackend",
    "FakeSecondLlm",
    "NliBackend",
    "NliScores",
    "OllamaSecondLlm",
    "SecondLlmBackend",
    "TransformersNliBackend",
    "VerificationService",
]
