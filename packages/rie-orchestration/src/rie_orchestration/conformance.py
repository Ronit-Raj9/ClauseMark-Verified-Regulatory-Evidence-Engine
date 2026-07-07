"""Conformance layer — materialise the judge-validated submission artefacts.

Turns the pipeline's final `RieState` into the exact 13-column CSV + JSON
contract via `rie_output`, with byte-true verbatim snippets, NEW/KNOWN
discovery tags (diffed against the Round-1 gold), and `P{pillar}-I{n}` display
indicator ids. This is the ONLY place submission files are written.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rie_contracts import (
    Claim,
    DocumentMeta,
    SourceRegistryEntry,
    VerificationStatus,
)
from rie_output import (
    EvidencePackageJson,
    ProvisionRecord,
    to_display_indicator,
    validate_rows,
    write_csv,
    write_json,
)

from rie_orchestration.state import RieState
from rie_orchestration.wiring import AdapterBundle

log = logging.getLogger(__name__)


def _source_meta(
    bundle: AdapterBundle, jurisdiction: str
) -> dict[str, SourceRegistryEntry]:
    try:
        entries = bundle.config.load_source_registry(jurisdiction)
    except Exception as e:  # noqa: BLE001 — registry optional in some runs
        log.warning("conformance: source registry load failed (%s)", e)
        return {}
    return {e.source_id: e for e in entries}


def _verbatim(bundle: AdapterBundle, claim: Claim) -> tuple[str, str, int]:
    """Return (verbatim_snippet, article, page) — byte-true from stored element."""
    span = claim.evidence_spans[0] if claim.evidence_spans else None
    try:
        element = bundle.repo.get_element(claim.clause_id)
    except Exception:  # noqa: BLE001
        element = None
    if element is None:
        return "", "", 0
    text = element.text
    if span is not None:
        # Spans are element-relative (0..len). Fall back to absolute-relative or
        # full text if offsets don't land inside this element.
        if 0 <= span.char_start < span.char_end <= len(text):
            snippet = text[span.char_start : span.char_end]
        else:
            rel_start = max(0, span.char_start - element.char_start)
            rel_end = max(rel_start, span.char_end - element.char_start)
            snippet = text[rel_start:rel_end] or text
    else:
        snippet = text
    article = element.legal_numbering or f"para {element.element_id.rsplit('_', 1)[-1]}"
    return snippet.strip(), article, element.page


def _discovery_tag(
    bundle: AdapterBundle, jurisdiction: str, claim: Claim, snippet: str, pillar_id: str
) -> str:
    try:
        from rie_ingest.discovery_tag import tag_discovery

        gold = list(bundle.config.load_gold(pillar_id))
        return tag_discovery(
            jurisdiction=jurisdiction,
            indicator_id=claim.indicator_id,
            doc_id=claim.evidence_spans[0].doc_id if claim.evidence_spans else "",
            span_text=snippet,
            gold=gold,
        )
    except Exception as e:  # noqa: BLE001 — tagger/gold optional
        log.debug("conformance: discovery tag fallback NEW (%s)", e)
        return "NEW"


def build_provision_records(
    state: RieState, bundle: AdapterBundle
) -> list[ProvisionRecord]:
    """Assemble one ProvisionRecord per non-rejected claim."""
    jurisdiction = state.get("jurisdiction", "")
    sources = _source_meta(bundle, jurisdiction)
    docs: dict[str, DocumentMeta] = {d.doc_id: d for d in state.get("documents", [])}
    verifications = state.get("verifications", {})

    rows: list[ProvisionRecord] = []
    for claim in state.get("claims", []):
        report = verifications.get(claim.claim_id)
        if report is not None and report.status == VerificationStatus.REJECTED:
            continue  # withhold rejected — never ship an unverifiable claim

        doc_id = claim.evidence_spans[0].doc_id if claim.evidence_spans else claim.clause_id
        doc = docs.get(doc_id)
        src = sources.get(doc_id)
        snippet, article, page = _verbatim(bundle, claim)
        tag = _discovery_tag(bundle, jurisdiction, claim, snippet, claim.pillar_id)

        status = report.status.value if report is not None else "pending_verification"
        conf = "" if claim.model_confidence is None else f"{claim.model_confidence:.2f}"
        rationale = (
            f"{claim.decomposition.subject}: {claim.decomposition.constraint}"
        )[:300]

        rows.append(
            ProvisionRecord(
                economy=jurisdiction,
                law_name=(doc.title if doc else (src.title if src else doc_id)),
                law_number_ref=(
                    (src.law_number_ref if src and src.law_number_ref else "")
                    or (src.title if src else "")
                ),
                last_amended=(src.last_amended if src and src.last_amended else ""),
                indicator_id=to_display_indicator(claim.indicator_id),
                article=article,
                discovery_tag=tag,
                location_reference=f"page {page}" if page else "",
                verbatim_snippet=snippet,
                mapping_rationale=rationale,
                source_url=(doc.source_url if doc and doc.source_url else (src.source_url if src and src.source_url else "")),
                confidence=conf,
                notes=f"layer1_status={status}",
            )
        )
    return rows


def write_submission(
    state: RieState,
    bundle: AdapterBundle,
    *,
    out_dir: Path,
    economy: str,
    pillar_ids: list[str],
    model_version: str = "fake-deterministic",
    cost: dict[str, Any] | None = None,
    timestamp: str = "run",
) -> dict[str, Any]:
    """Write `<economy>_P<pillars>_<ts>.csv` + `.json`; return paths + validation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_provision_records(state, bundle)
    violations = validate_rows(rows)

    # EvidencePackageJson.cost is numeric-only; keep just the scalar metrics.
    numeric_cost = {
        k: float(v)
        for k, v in (cost or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }

    stem = f"{economy}_P{'-'.join(pillar_ids)}_{timestamp}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"

    write_csv(rows, csv_path)
    pkg = EvidencePackageJson(
        economy=economy,
        pillar_ids=pillar_ids,
        generated_at=timestamp,
        model_version=model_version,
        provisions=[_json_provision(r) for r in rows],
        counts={
            "provisions": len(rows),
            "new": sum(1 for r in rows if r.discovery_tag == "NEW"),
            "known": sum(1 for r in rows if r.discovery_tag == "KNOWN"),
        },
        cost=numeric_cost,
    )
    write_json(pkg, json_path)

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "rows": len(rows),
        "violations": violations,
    }


def _json_provision(r: ProvisionRecord) -> dict[str, Any]:
    d = r.model_dump()
    d.update(
        {
            "indicator_id_decimal": _decimal_of(r.indicator_id),
            "pdf_is_scanned": False,
            "retrieval_method": "hybrid_parent_document",
        }
    )
    return d


def _decimal_of(display: str) -> str:
    try:
        from rie_output import to_decimal

        return to_decimal(display)
    except Exception:  # noqa: BLE001
        return display
