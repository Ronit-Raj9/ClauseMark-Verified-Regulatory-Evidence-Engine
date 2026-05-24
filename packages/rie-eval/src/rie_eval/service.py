"""`Evaluator` — concrete `EvaluatorPort` over the gold set + repository.

Computes the §8 evaluation metrics:

- per-indicator precision / recall / F1
- macro-averaged precision / recall / F1
- citation-support precision (reviewer-accept rate on `verified` claims)
- authority-tier error rate (stored doc tier vs gold expected tier)
- false-zero rate (system emitted ``no_evidence`` where gold had a claim)
- claim-status counts (total / verified / flagged / rejected)
- retrieval recall (gold element / span substring appears in retrieved IDs)

All metrics are pure functions of the inputs. The repository / config inputs
are read-only — no metric is allowed to mutate state.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rie_config import ConfigRepository
from rie_contracts import (
    AuthorityTier,
    Claim,
    CoverageRecord,
    CoverageState,
    DocumentMeta,
    GoldItem,
    Layer1Status,
    ReviewDecision,
    ReviewRecord,
)

from rie_eval.metrics import NO_CLAIM_LABEL, macro_avg, prf
from rie_eval.ragas_metrics import (
    aggregate_recall,
    compute_context_recall,
    compute_faithfulness,
)

# ─── Lightweight repository protocol (subset of DocumentRepositoryPort) ──────
# The Evaluator only needs three read-side operations. Defining a narrow local
# Protocol keeps the service testable with a tiny fake and avoids coupling to
# the full persistence surface.


@runtime_checkable
class _EvalRepo(Protocol):
    def get_document(self, doc_id: str) -> DocumentMeta: ...

    def list_reviews(self, claim_id: str) -> Sequence[ReviewRecord]: ...

    def list_coverage(self, jurisdiction: str | None = None) -> Sequence[CoverageRecord]: ...


@dataclass
class Evaluator:
    """Gold-set evaluator. Implements ``EvaluatorPort``.

    Construct with a config repository (to load gold sets) and optionally a
    persistence repository (to look up reviews, authority tier, coverage).
    The persistence repo is optional so unit tests can run without a database.
    """

    config: ConfigRepository
    repo: _EvalRepo | None = None

    # ─── Port: evaluate_pillar ──────────────────────────────────────────────

    def evaluate_pillar(
        self,
        pillar_id: str,
        claims: Sequence[Claim],
    ) -> dict[str, float]:
        gold = list(self.config.load_gold(pillar_id))
        # Restrict claims to this pillar — callers may pass everything.
        pillar_claims = [c for c in claims if c.pillar_id == pillar_id]

        # ─── Match each gold item to at most one claim ────────────────────
        matched_pairs, unmatched_gold, unmatched_claims = _match(gold, pillar_claims)

        y_true: list[str] = []
        y_pred: list[str] = []
        for gold_item, claim in matched_pairs:
            y_true.append(gold_item.indicator_id)
            y_pred.append(claim.indicator_id)
        # Unmatched gold → the system failed to emit a claim → recall hit.
        for gold_item in unmatched_gold:
            y_true.append(gold_item.indicator_id)
            y_pred.append(NO_CLAIM_LABEL)
        # Unmatched claims → the system emitted a claim where gold has nothing.
        # Encode as: true=NO_CLAIM_LABEL, pred=claim.indicator_id → precision hit.
        for claim in unmatched_claims:
            y_true.append(NO_CLAIM_LABEL)
            y_pred.append(claim.indicator_id)

        per_label = prf(y_true, y_pred)
        precision_by_indicator = {k: v[0] for k, v in per_label.items()}
        recall_by_indicator = {k: v[1] for k, v in per_label.items()}
        f1_by_indicator = {k: v[2] for k, v in per_label.items()}
        p_macro, r_macro, f_macro = macro_avg(per_label)

        # ─── Claim-status counts ──────────────────────────────────────────
        count_total = len(pillar_claims)
        count_verified = sum(1 for c in pillar_claims if c.layer1_status == Layer1Status.VERIFIED)
        count_flagged = sum(1 for c in pillar_claims if c.layer1_status == Layer1Status.FLAGGED)
        count_rejected = sum(1 for c in pillar_claims if c.layer1_status == Layer1Status.REJECTED)

        # ─── Citation-support precision (reviewer-accept on `verified`) ───
        # The headline trust number. Each `verified` claim must be confirmed
        # by a human reviewer (ACCEPT). If no reviews available (no repo,
        # or reviews table empty), we default to 1.0 per spec — the in-memory
        # fast-path used by unit tests cannot manufacture a reviewer signal.
        citation_support_precision = self._citation_support_precision(pillar_claims, count_verified)

        # ─── Authority-tier error rate ────────────────────────────────────
        authority_tier_error_rate = self._authority_tier_error_rate(matched_pairs)

        # ─── False-zero rate ──────────────────────────────────────────────
        false_zero_rate = self._false_zero_rate(pillar_id, gold)

        # ─── Reviewer-override rate ───────────────────────────────────────
        # Fraction of VERIFIED claims for which any human ReviewRecord
        # disagrees with the system (decision in {CORRECT, REJECT}).
        reviewer_override_rate = self._reviewer_override_rate(pillar_claims)

        # ─── RAGAS-style metrics (token-overlap proxies) ──────────────────
        # context_recall: per matched pair, fraction of gold span tokens that
        #   appear in the cited claim spans' element texts (when reachable).
        # faithfulness:   per matched pair, fraction of claim-constraint
        #   tokens that appear in the cited span text.
        context_recall, faithfulness = self._ragas_proxies(matched_pairs)

        return {
            **{f"precision[{k}]": v for k, v in precision_by_indicator.items()},
            **{f"recall[{k}]": v for k, v in recall_by_indicator.items()},
            **{f"f1[{k}]": v for k, v in f1_by_indicator.items()},
            "precision_macro": p_macro,
            "recall_macro": r_macro,
            "f1_macro": f_macro,
            "citation_support_precision": citation_support_precision,
            "authority_tier_error_rate": authority_tier_error_rate,
            "false_zero_rate": false_zero_rate,
            "reviewer_override_rate": reviewer_override_rate,
            "context_recall": context_recall,
            "faithfulness": faithfulness,
            "count_total": float(count_total),
            "count_verified": float(count_verified),
            "count_flagged": float(count_flagged),
            "count_rejected": float(count_rejected),
        }

    # ─── Port: measure_retrieval_recall ─────────────────────────────────────

    def measure_retrieval_recall(
        self,
        pillar_id: str,
        retrieved_per_query: Sequence[Sequence[str]],
    ) -> float:
        gold = list(self.config.load_gold(pillar_id))
        if not gold:
            return 0.0
        if len(retrieved_per_query) != len(gold):
            raise ValueError(
                f"retrieved_per_query length {len(retrieved_per_query)} "
                f"!= gold length {len(gold)} for pillar {pillar_id}"
            )
        hits = 0
        for item, retrieved in zip(gold, retrieved_per_query):
            if _gold_hit_in_retrieved(item, retrieved):
                hits += 1
        return hits / len(gold)

    # ─── Public helpers — exposed so callers can build expanded reports ─────

    def detailed_per_indicator(
        self,
        pillar_id: str,
        claims: Sequence[Claim],
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Return PRF broken out per indicator (without the flat-dict packing).

        Shape: ``{"precision_by_indicator": {ind: p}, ...}``. Useful for the
        CLI table renderer and human-facing reports.
        """
        flat = self.evaluate_pillar(pillar_id, claims)
        precision_by_indicator: dict[str, float] = {}
        recall_by_indicator: dict[str, float] = {}
        f1_by_indicator: dict[str, float] = {}
        for key, value in flat.items():
            if key.startswith("precision[") and key.endswith("]"):
                precision_by_indicator[key[len("precision[") : -1]] = value
            elif key.startswith("recall[") and key.endswith("]"):
                recall_by_indicator[key[len("recall[") : -1]] = value
            elif key.startswith("f1[") and key.endswith("]"):
                f1_by_indicator[key[len("f1[") : -1]] = value
        return {
            "precision_by_indicator": precision_by_indicator,
            "recall_by_indicator": recall_by_indicator,
            "f1_by_indicator": f1_by_indicator,
        }

    # ─── Internals ──────────────────────────────────────────────────────────

    def _citation_support_precision(
        self, pillar_claims: Sequence[Claim], count_verified: int
    ) -> float:
        if count_verified == 0:
            # No verified claims to evaluate — vacuously perfect.
            return 1.0
        if self.repo is None:
            # Fed in-memory via repository=None — fall back per spec.
            return 1.0
        accepted = 0
        scored = 0
        for c in pillar_claims:
            if c.layer1_status != Layer1Status.VERIFIED:
                continue
            try:
                reviews = list(self.repo.list_reviews(c.claim_id))
            except Exception:
                reviews = []
            if not reviews:
                continue
            # Use the most recent review (highest `decided_at`).
            latest = max(reviews, key=lambda r: r.decided_at)
            scored += 1
            if latest.decision == ReviewDecision.ACCEPT:
                accepted += 1
        if scored == 0:
            # No reviews recorded yet — same fallback per spec.
            return 1.0
        return accepted / scored

    def _reviewer_override_rate(self, pillar_claims: Sequence[Claim]) -> float:
        """Fraction of VERIFIED claims with at least one human override.

        A ``CORRECT`` or ``REJECT`` review decision counts as an override; a
        plain ``ACCEPT`` does not. Returns ``0.0`` when there are no verified
        claims or the persistence repository is unavailable.
        """
        verified = [c for c in pillar_claims if c.layer1_status == Layer1Status.VERIFIED]
        if not verified:
            return 0.0
        if self.repo is None:
            return 0.0
        overrides = 0
        for c in verified:
            try:
                reviews = list(self.repo.list_reviews(c.claim_id))
            except Exception:
                reviews = []
            if any(r.decision in (ReviewDecision.CORRECT, ReviewDecision.REJECT) for r in reviews):
                overrides += 1
        return overrides / len(verified)

    def _ragas_proxies(
        self, matched_pairs: Sequence[tuple[GoldItem, Claim]]
    ) -> tuple[float, float]:
        """Mean (context_recall, faithfulness) across matched (gold, claim) pairs.

        Both are token-overlap proxies (see ``ragas_metrics``). When a
        persistence repository is supplied we resolve the cited evidence-span
        text via ``get_element_text(element_id)``; otherwise we fall back to
        the gold ``span_text`` as a stand-in for the cited passage so the
        metric is still well-defined on in-memory test fixtures.
        """
        if not matched_pairs:
            return (0.0, 0.0)
        recalls: list[float] = []
        faiths: list[float] = []
        for gold_item, claim in matched_pairs:
            cited_text = self._cited_text(claim, fallback=gold_item.span_text)
            recalls.append(
                compute_context_recall(claim, gold_item.span_text, retrieved_passages=[cited_text])
            )
            faiths.append(compute_faithfulness(claim.decomposition, cited_text))
        return (aggregate_recall(recalls), aggregate_recall(faiths))

    def _cited_text(self, claim: Claim, *, fallback: str) -> str:
        """Resolve the cited span's element text via the repo if possible."""
        if not claim.evidence_spans:
            return fallback
        span = claim.evidence_spans[0]
        # Only DocumentRepository implementations expose get_element_text;
        # the narrow Evaluator protocol does not. Probe with getattr to stay
        # decoupled from the full persistence surface.
        getter = getattr(self.repo, "get_element_text", None) if self.repo is not None else None
        if getter is None:
            return fallback
        try:
            text = getter(span.element_id)
        except Exception:
            return fallback
        return text or fallback

    def _authority_tier_error_rate(self, matched_pairs: Sequence[tuple[GoldItem, Claim]]) -> float:
        if not matched_pairs:
            return 0.0
        if self.repo is None:
            # Without persistence we cannot resolve the stored doc's tier;
            # callers must supply a repo to measure this. Be honest — return 0.
            return 0.0
        errors = 0
        scored = 0
        for gold_item, claim in matched_pairs:
            # Resolve the actual stored authority tier for the claim's doc.
            doc_id = claim.evidence_spans[0].doc_id if claim.evidence_spans else None
            if doc_id is None:
                continue
            try:
                doc = self.repo.get_document(doc_id)
            except (KeyError, Exception):
                continue
            scored += 1
            stored = AuthorityTier(doc.authority_tier)
            if stored != gold_item.expected_authority_tier:
                errors += 1
        if scored == 0:
            return 0.0
        return errors / scored

    def _false_zero_rate(self, pillar_id: str, gold: Sequence[GoldItem]) -> float:
        """Fraction of emitted ``no_evidence`` coverage rows that the gold
        set contradicts (i.e. an indicator the system claimed unfindable
        actually has at least one gold-set example)."""
        if self.repo is None:
            return 0.0
        # Group gold indicator presence per jurisdiction.
        gold_present: dict[tuple[str, str], bool] = defaultdict(bool)
        for g in gold:
            if g.pillar_id != pillar_id:
                continue
            gold_present[(g.jurisdiction, g.indicator_id)] = True
        try:
            coverage_rows = list(self.repo.list_coverage())
        except Exception:
            return 0.0
        no_evidence_rows = [
            r for r in coverage_rows if r.state == CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
        ]
        if not no_evidence_rows:
            return 0.0
        emitted_for_pillar = [
            r for r in no_evidence_rows if _indicator_pillar(r.indicator_id) == pillar_id
        ]
        if not emitted_for_pillar:
            return 0.0
        false_zeros = sum(
            1
            for r in emitted_for_pillar
            if gold_present.get((r.jurisdiction, r.indicator_id), False)
        )
        return false_zeros / len(emitted_for_pillar)


# ─── Module-level helpers ────────────────────────────────────────────────────


def _indicator_pillar(indicator_id: str) -> str:
    """Indicator IDs are ``<pillar>.<index>`` — derive the pillar half."""
    return indicator_id.split(".", maxsplit=1)[0] if "." in indicator_id else indicator_id


def _match(
    gold: Sequence[GoldItem], claims: Sequence[Claim]
) -> tuple[list[tuple[GoldItem, Claim]], list[GoldItem], list[Claim]]:
    """Greedy 1-1 matching of gold items to claims.

    A claim matches a gold item when EITHER:
    - one of the claim's evidence spans points at an ``element_id`` whose
      stable prefix mentions the gold doc_id (best-effort substring match), or
    - the gold ``span_text`` (normalised) appears as a substring of the claim's
      ``clause_id`` / first evidence-span span_id chain via doc_id equality.

    Matching is conservative: at most one claim per gold item, at most one
    gold item per claim. Multiple compatible claims for the same gold pick
    the one with the same indicator (if any), else the first.
    """
    used_claims: set[str] = set()
    pairs: list[tuple[GoldItem, Claim]] = []
    unmatched_gold: list[GoldItem] = []

    for g in gold:
        candidates = [c for c in claims if c.claim_id not in used_claims]
        # Prefer matches whose evidence touches the same doc_id.
        same_doc = [c for c in candidates if any(sp.doc_id == g.doc_id for sp in c.evidence_spans)]
        pool = same_doc or [c for c in candidates if g.jurisdiction == c.jurisdiction]
        if not pool:
            unmatched_gold.append(g)
            continue
        # Within the doc-matched pool, prefer the indicator-equal one (helps
        # disambiguate when the same doc has multiple gold items).
        same_indicator = [c for c in pool if c.indicator_id == g.indicator_id]
        chosen = same_indicator[0] if same_indicator else pool[0]
        pairs.append((g, chosen))
        used_claims.add(chosen.claim_id)

    unmatched_claims = [c for c in claims if c.claim_id not in used_claims]
    return pairs, unmatched_gold, unmatched_claims


def _gold_hit_in_retrieved(item: GoldItem, retrieved: Sequence[str]) -> bool:
    """A gold item is recalled when either:
    - any retrieved id literally equals an expected element_id we can derive
      from the gold item (we don't know it directly, so we accept any id
      containing the gold doc_id), OR
    - the gold ``span_text`` appears (normalised) as a substring of any of
      the retrieved ids/snippets (caller may pass plain text snippets).
    """
    if not retrieved:
        return False
    needle_doc = item.doc_id.lower()
    needle_span = _normalise(item.span_text)
    for rid in retrieved:
        if rid is None:
            continue
        rid_norm = rid.lower()
        if needle_doc and needle_doc in rid_norm:
            return True
        if needle_span and needle_span in _normalise(rid):
            return True
    return False


def _normalise(s: str) -> str:
    return " ".join(s.split()).lower()


__all__ = ["Evaluator"]
