"""Pure business logic — domain rules, never adapters.

Imports `rie_contracts` only. Authority precedence, regime assembly logic,
self-consistency vote tally, citation rendering — all deterministic, all here.
"""

from rie_domain.authority import (
    AuthorityResolver,
    apply_authority_overrides,
    rank_authority,
)
from rie_domain.citation import build_citation, render_citation_string
from rie_domain.regime import (
    assemble_regime,
    consolidate_self_consistency_votes,
    derive_layer1_status,
)
from rie_domain.verification import derive_status

__all__ = [
    "AuthorityResolver",
    "apply_authority_overrides",
    "assemble_regime",
    "build_citation",
    "consolidate_self_consistency_votes",
    "derive_layer1_status",
    "derive_status",
    "rank_authority",
    "render_citation_string",
]
