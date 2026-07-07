"""Pure business logic — domain rules, never adapters.

Imports `rie_contracts` only. Authority precedence, regime assembly logic,
self-consistency vote tally, citation rendering — all deterministic, all here.
"""

from rie_domain.authority import (
    AuthorityResolver,
    apply_authority_overrides,
    rank_authority,
)
from rie_domain.citation import (
    Citation,
    build_citation,
    extract_snippet,
    materialise_citations,
    render_citation_string,
)
from rie_domain.layer2 import (
    build_layer2_batch,
    build_layer2_from_pillar,
    build_layer2_recommendation,
    claim_text,
    find_coverage_for_claim,
    format_rationale,
    pick_band,
    recommend_band,
    regime_open_questions,
    score_bands_from_criteria,
    score_from_few_shots,
)
from rie_domain.regime import (
    REGIME_EDGE_TYPES,
    assemble_regime,
    assemble_regime_from_claim,
    consolidate_self_consistency_votes,
    derive_layer1_status,
    enrich_claim_regime,
)
from rie_domain.verification import derive_status

__all__ = [
    "REGIME_EDGE_TYPES",
    "AuthorityResolver",
    "Citation",
    "apply_authority_overrides",
    "assemble_regime",
    "assemble_regime_from_claim",
    "build_citation",
    "build_layer2_batch",
    "build_layer2_from_pillar",
    "build_layer2_recommendation",
    "claim_text",
    "consolidate_self_consistency_votes",
    "derive_layer1_status",
    "derive_status",
    "enrich_claim_regime",
    "extract_snippet",
    "find_coverage_for_claim",
    "format_rationale",
    "materialise_citations",
    "pick_band",
    "rank_authority",
    "recommend_band",
    "regime_open_questions",
    "render_citation_string",
    "score_bands_from_criteria",
    "score_from_few_shots",
]
