"""Audit endpoint — full evidence package for the audit UI.

Returns, for a single jurisdiction:
- every claim (any layer1_status) with its latest verification report and
  the deterministic citations materialised from the document store
- every coverage record

This is the payload the Streamlit audit viewer renders side-by-side with the
source PDF, so it MUST contain enough to draw highlights without further
round-trips to the API.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from rie_persistence.models import ClaimRow
from rie_persistence.repository import (
    DocumentRepository,
    _to_claim,  # type: ignore[attr-defined]
)
from sqlalchemy import select

from rie_api.deps import get_repo
from rie_api.routes.claims import _materialise_citations
from rie_api.schemas import (
    ClaimDTO,
    CoverageDTO,
    EvidencePackage,
    EvidencePackageClaim,
    VerificationReportDTO,
)

router = APIRouter(prefix="/v1/audit", tags=["audit"])


@router.get(
    "/{jurisdiction}",
    response_model=EvidencePackage,
    summary="Full evidence package for one jurisdiction",
)
def get_audit_package(
    jurisdiction: str,
    repo: DocumentRepository = Depends(get_repo),
) -> EvidencePackage:
    # Pull every claim for this jurisdiction in a single session scan.
    with repo.session_factory() as session:
        rows = session.scalars(select(ClaimRow).where(ClaimRow.jurisdiction == jurisdiction)).all()
        claims = [_to_claim(r, list(r.spans)) for r in rows]

    package_claims: list[EvidencePackageClaim] = []
    for claim in claims:
        report = repo.latest_verification(claim.claim_id)
        package_claims.append(
            EvidencePackageClaim(
                claim=ClaimDTO.from_model(claim),
                verification=VerificationReportDTO.from_model(report) if report else None,
                citations=_materialise_citations(claim.evidence_spans, repo),
            )
        )

    coverage_records = list(repo.list_coverage(jurisdiction=jurisdiction))
    return EvidencePackage(
        jurisdiction=jurisdiction,
        claims=package_claims,
        coverage=[CoverageDTO.from_model(c) for c in coverage_records],
        generated_at=datetime.now(tz=UTC),
    )
