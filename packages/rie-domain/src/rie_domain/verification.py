"""Derive verification status from gate results.

Policy (matches §6.5 of the architecture):
  - any deterministic gate (span-existence, verbatim-match) fails → REJECTED
  - any model gate (entailment, self-consistency) fails OR disagrees → FLAGGED
  - all four gates pass → VERIFIED
"""

from __future__ import annotations

from collections.abc import Sequence

from rie_contracts import GateName, GateResult, VerificationStatus

_DETERMINISTIC: frozenset[GateName] = frozenset({GateName.SPAN_EXISTENCE, GateName.VERBATIM_MATCH})
_MODEL: frozenset[GateName] = frozenset({GateName.ENTAILMENT, GateName.SELF_CONSISTENCY})


def derive_status(gates: Sequence[GateResult]) -> tuple[VerificationStatus, list[str]]:
    by_name: dict[GateName, GateResult] = {g.gate: g for g in gates}
    failures: list[str] = []

    for gate in _DETERMINISTIC:
        if gate not in by_name:
            return VerificationStatus.REJECTED, [f"missing gate {gate}"]
        if not by_name[gate].passed:
            failures.append(f"{gate}: {by_name[gate].detail}")
    if failures:
        return VerificationStatus.REJECTED, failures

    for gate in _MODEL:
        if gate not in by_name:
            return VerificationStatus.FLAGGED, [f"missing gate {gate}"]
        if not by_name[gate].passed:
            failures.append(f"{gate}: {by_name[gate].detail}")
    if failures:
        return VerificationStatus.FLAGGED, failures

    return VerificationStatus.VERIFIED, []
