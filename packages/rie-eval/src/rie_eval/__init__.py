"""Gold-set evaluator + RIE-specific metrics (§8)."""

from rie_eval.ablations import (
    ABLATION_WITHOUT_ENTAILMENT,
    ABLATION_WITHOUT_PARENT_DOC,
    ABLATION_WITHOUT_RERANKER,
    CANONICAL_ABLATIONS,
    AblationResult,
    run_ablation,
    run_all_ablations,
)
from rie_eval.cost import CostTracker, RunCostReport, summarise_run_cost
from rie_eval.hallucinated_words import compute_hallucinated_words_rate
from rie_eval.metrics import NO_CLAIM_LABEL, confusion, macro_avg, prf
from rie_eval.ragas_metrics import (
    RAGAS_AVAILABLE,
    aggregate_recall,
    compute_context_recall,
    compute_faithfulness,
)
from rie_eval.service import Evaluator

__all__ = [
    "ABLATION_WITHOUT_ENTAILMENT",
    "ABLATION_WITHOUT_PARENT_DOC",
    "ABLATION_WITHOUT_RERANKER",
    "CANONICAL_ABLATIONS",
    "NO_CLAIM_LABEL",
    "RAGAS_AVAILABLE",
    "AblationResult",
    "CostTracker",
    "Evaluator",
    "RunCostReport",
    "aggregate_recall",
    "compute_context_recall",
    "compute_faithfulness",
    "compute_hallucinated_words_rate",
    "confusion",
    "macro_avg",
    "prf",
    "run_ablation",
    "run_all_ablations",
    "summarise_run_cost",
]
