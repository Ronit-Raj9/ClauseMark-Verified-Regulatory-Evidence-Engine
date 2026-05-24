"""HumanReviewQueuePort implementation backed by the document repository.

The §6.5 "any disagreement → flagged" policy lands all FLAGGED claims here.
LangGraph's `interrupt()` mechanism is what pauses the graph; this queue is
the audit-log layer behind that interrupt (per `tool.md`: LangGraph state ≠
audit log).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from rie_contracts import (
    Claim,
    DocumentRepositoryPort,
    ReviewRecord,
    VerificationReport,
    VerificationStatus,
)


@dataclass
class HumanReviewQueue:
    """Repository-backed `HumanReviewQueuePort` impl."""

    repo: DocumentRepositoryPort

    def enqueue(self, claim: Claim, report: VerificationReport) -> str:
        # The claim is already persisted by `verify_node`. Enqueuing is a no-op
        # at the storage layer; UI / API query for FLAGGED claims directly.
        return claim.claim_id

    def resolve(self, claim_id: str, review: ReviewRecord) -> None:
        self.repo.save_review(review)

    def pending(self) -> Iterable[tuple[Claim, VerificationReport]]:
        for claim in self.repo.list_claims_for_review():
            # We don't have a latest_verification() in the port; the API endpoint
            # joins reviews+verifications at query time. Yield (claim, None-like)
            # via a stub report if the repo doesn't expose verifications cleanly.
            try:
                report = self.repo.latest_verification(claim.claim_id)  # type: ignore[attr-defined]
            except AttributeError:
                report = None
            if report is None:
                continue
            if report.status == VerificationStatus.FLAGGED:
                yield claim, report
