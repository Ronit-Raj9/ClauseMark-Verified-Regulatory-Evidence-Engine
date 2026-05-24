"""4-gate verification adapter.

Implements `VerifierPort`:

  * Tier A (deterministic, runs on every claim): span-existence, verbatim-match.
  * Tier B (model-based, runs only on Tier-A survivors): entailment, self-consistency.

Honest semantics — *any* disagreement between the NLI model and the second LLM
flags the claim for human review (never auto-resolved). See §6.5.

Phase 2 (opt-in): an additive 5th gate, "kg_grounding", checks that entities
mentioned in the claim's decomposition resolve in a per-document KG built from
heuristic legal-text NER. It runs only when ``VerificationService`` is
constructed with ``kg_gate_enabled=True`` and is consumed via the additive
``verify_with_kg(...)`` API. The canonical ``verify(...)`` 4-gate output and
the frozen ``rie_contracts.VerificationReport`` contract are unchanged.
"""

from __future__ import annotations

from rie_verify.kg_grounding import (
    KG_GATE_NAME,
    DocumentKg,
    Entity,
    EntityKind,
    KgGroundingResult,
    build_document_kg,
    extract_entities,
    run_kg_grounding_gate,
)
from rie_verify.nli import FakeNliBackend, NliBackend, NliScores, TransformersNliBackend
from rie_verify.second_llm import FakeSecondLlm, OllamaSecondLlm, SecondLlmBackend
from rie_verify.service import VerificationService

__all__ = [
    "KG_GATE_NAME",
    "DocumentKg",
    "Entity",
    "EntityKind",
    "FakeNliBackend",
    "FakeSecondLlm",
    "KgGroundingResult",
    "NliBackend",
    "NliScores",
    "OllamaSecondLlm",
    "SecondLlmBackend",
    "TransformersNliBackend",
    "VerificationService",
    "build_document_kg",
    "extract_entities",
    "run_kg_grounding_gate",
]
