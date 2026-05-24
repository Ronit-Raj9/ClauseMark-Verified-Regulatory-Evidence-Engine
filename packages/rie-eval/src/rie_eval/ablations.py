"""§8 ablation harness.

The actual ablated pipeline runs are orchestrated externally (by `rie-orchestration`
or a script): for each ablation, the caller rebuilds the pipeline with the
component switched off, runs it over the gold set, and feeds the resulting
metric dict back here as ``ablated_metric``. This module records the delta in
a structured, exportable form.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

# §8: each canonical ablation has a stable name. Engine code reads these from
# config to decide which switch to flip; this module only records the result.
ABLATION_WITHOUT_RERANKER: str = "without_reranker"
ABLATION_WITHOUT_ENTAILMENT: str = "without_entailment"
ABLATION_WITHOUT_PARENT_DOC: str = "without_parent_document_retrieval"

CANONICAL_ABLATIONS: tuple[str, ...] = (
    ABLATION_WITHOUT_RERANKER,
    ABLATION_WITHOUT_ENTAILMENT,
    ABLATION_WITHOUT_PARENT_DOC,
)


@dataclass(frozen=True)
class AblationResult:
    """The effect of disabling one component on one metric."""

    name: str
    metric_key: str
    baseline: float
    ablated: float
    delta: float = field(init=False)

    def __post_init__(self) -> None:
        # Frozen dataclass — use object.__setattr__ to set the derived field.
        object.__setattr__(self, "delta", self.ablated - self.baseline)


def run_ablation(
    name: str,
    baseline_metric: float,
    ablated_metric: float,
    metric_key: str = "f1_macro",
) -> AblationResult:
    """Record a single ablation outcome."""
    return AblationResult(
        name=name,
        metric_key=metric_key,
        baseline=baseline_metric,
        ablated=ablated_metric,
    )


# Signature only — the actual ablated runs are executed by the orchestrator.
def run_all_ablations(
    baseline_metrics: Mapping[str, float],
    ablated_runner: Callable[[str], Mapping[str, float]],
    metric_key: str = "f1_macro",
    ablations: Sequence[str] = CANONICAL_ABLATIONS,
) -> list[AblationResult]:
    """Drive the §8 ablation matrix.

    ``ablated_runner(ablation_name)`` is supplied by the orchestrator: it
    rebuilds the pipeline with that component disabled and returns the
    resulting metrics. This function records the delta against the baseline
    for each ablation and returns them in order.
    """
    if metric_key not in baseline_metrics:
        raise KeyError(f"metric_key {metric_key!r} missing from baseline_metrics")
    baseline_value = float(baseline_metrics[metric_key])
    results: list[AblationResult] = []
    for ablation in ablations:
        ablated = ablated_runner(ablation)
        if metric_key not in ablated:
            raise KeyError(f"ablation {ablation!r}: metric_key {metric_key!r} missing")
        results.append(
            run_ablation(
                name=ablation,
                baseline_metric=baseline_value,
                ablated_metric=float(ablated[metric_key]),
                metric_key=metric_key,
            )
        )
    return results


__all__ = [
    "ABLATION_WITHOUT_ENTAILMENT",
    "ABLATION_WITHOUT_PARENT_DOC",
    "ABLATION_WITHOUT_RERANKER",
    "CANONICAL_ABLATIONS",
    "AblationResult",
    "run_ablation",
    "run_all_ablations",
]
