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

from rie_verify.adversarial import (
    DEFAULT_LENSES,
    AdversarialPanel,
    FakeRefuter,
    LensVerdict,
    OllamaRefuter,
    PanelOutcome,
    RefutationLlm,
)
from rie_verify.confidence import (
    ConfidenceScreenResult,
    apply_confidence_routing,
    route_verifications_by_confidence,
    screen_confidence_on_gold,
)
from rie_verify.entity_grounding import (
    Entity as GroundingEntity,
)
from rie_verify.entity_grounding import (
    EntityExtractor,
    EntityGroundingChecker,
    FakeEntityExtractor,
    SpacyEntityExtractor,
)
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
from rie_verify.source_refetch import (
    FakeSourceFetcher,
    HttpSourceFetcher,
    SourceFetcher,
    verify_source_live,
)

__all__ = [
    "DEFAULT_LENSES",
    "KG_GATE_NAME",
    "AdversarialPanel",
    "ConfidenceScreenResult",
    "DocumentKg",
    "Entity",
    "EntityExtractor",
    "EntityGroundingChecker",
    "EntityKind",
    "FakeEntityExtractor",
    "FakeNliBackend",
    "FakeRefuter",
    "FakeSecondLlm",
    "FakeSourceFetcher",
    "GroundingEntity",
    "HttpSourceFetcher",
    "KgGroundingResult",
    "LensVerdict",
    "NliBackend",
    "NliScores",
    "OllamaRefuter",
    "OllamaSecondLlm",
    "PanelOutcome",
    "RefutationLlm",
    "SecondLlmBackend",
    "SourceFetcher",
    "SpacyEntityExtractor",
    "TransformersNliBackend",
    "VerificationService",
    "apply_confidence_routing",
    "build_document_kg",
    "extract_entities",
    "route_verifications_by_confidence",
    "run_kg_grounding_gate",
    "screen_confidence_on_gold",
    "verify_source_live",
]
