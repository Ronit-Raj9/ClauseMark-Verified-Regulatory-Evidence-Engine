"""In-memory adapter fakes — used for tests + dry-run smoke checks.

Every fake implements the corresponding Protocol port. None hits the network or
an LLM. The fakes are deliberately simple but realistic enough to exercise the
full pipeline end-to-end on `data/samples/sample_dpa.txt`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rie_contracts import (
    Claim,
    ClausePattern,
    ConfigRepositoryPort,
    CoverageRecord,
    CoverageState,
    Decomposition,
    DocumentMeta,
    Element,
    ElementType,
    EvidenceSpan,
    GateName,
    GateResult,
    IndicatorConfig,
    Layer1Status,
    LegalRegime,
    OcrEngine,
    PillarConfig,
    RetrievalHit,
    ReviewRecord,
    SourceRegistryEntry,
    SpanRole,
    StructureEdge,
    VerificationReport,
    VerificationStatus,
)

# ════════════════════════════════════════════════════════════════════════════
# FakeRepo — in-memory document/element/claim store
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class FakeRepo:
    documents: dict[str, DocumentMeta] = field(default_factory=dict)
    elements: dict[str, Element] = field(default_factory=dict)
    edges: dict[str, list[StructureEdge]] = field(default_factory=dict)
    claims: dict[str, Claim] = field(default_factory=dict)
    verifications: dict[str, VerificationReport] = field(default_factory=dict)
    coverage: dict[tuple[str, str], CoverageRecord] = field(default_factory=dict)
    reviews: list[ReviewRecord] = field(default_factory=list)

    def save_document(self, meta: DocumentMeta) -> None:
        self.documents[meta.doc_id] = meta

    def save_elements(
        self, doc_id: str, elements: Sequence[Element], edges: Sequence[StructureEdge]
    ) -> None:
        for e in elements:
            self.elements[e.element_id] = e
        self.edges.setdefault(doc_id, []).extend(edges)

    def get_element_text(self, element_id: str) -> str:
        return self.elements[element_id].text

    def get_element(self, element_id: str) -> Element:
        return self.elements[element_id]

    def get_elements(self, element_ids: Sequence[str]) -> Sequence[Element]:
        return [self.elements[i] for i in element_ids if i in self.elements]

    def save_claim(self, claim: Claim) -> None:
        self.claims[claim.claim_id] = claim

    def save_verification(self, report: VerificationReport) -> None:
        self.verifications[report.claim_id] = report

    def save_coverage(self, record: CoverageRecord) -> None:
        self.coverage[(record.jurisdiction, record.indicator_id)] = record

    def save_review(self, review: ReviewRecord) -> None:
        self.reviews.append(review)

    def list_claims_for_review(
        self, jurisdiction: str | None = None, pillar_id: str | None = None
    ) -> Sequence[Claim]:
        out = list(self.claims.values())
        if jurisdiction:
            out = [c for c in out if c.jurisdiction == jurisdiction]
        if pillar_id:
            out = [c for c in out if c.pillar_id == pillar_id]
        return out

    def list_coverage(self, jurisdiction: str | None = None) -> Sequence[CoverageRecord]:
        out = list(self.coverage.values())
        if jurisdiction:
            out = [c for c in out if c.jurisdiction == jurisdiction]
        return out


# ════════════════════════════════════════════════════════════════════════════
# FakeIngest — reads local files from the source registry
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class FakeIngest:
    config: ConfigRepositoryPort

    def load_source_registry(self, jurisdiction: str) -> Sequence[SourceRegistryEntry]:
        return self.config.load_source_registry(jurisdiction)

    def load_document_bytes(self, entry: SourceRegistryEntry) -> tuple[DocumentMeta, bytes]:
        if not entry.local_path:
            raise ValueError(f"FakeIngest needs local_path for {entry.source_id}")
        raw = Path(entry.local_path).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        meta = DocumentMeta(
            doc_id=entry.source_id,
            jurisdiction=entry.jurisdiction,
            title=entry.title,
            document_type=entry.document_type,
            effective_date=entry.effective_date,
            authority_tier=entry.authority_tier,
            source_url=entry.source_url,
            sha256=sha,
            retrieved_at=datetime.now(UTC),
            language=entry.language,
        )
        return meta, raw

    def list_sample_laws(self, samples_dir: Path) -> Sequence[Path]:
        return sorted(samples_dir.glob("*.txt")) + sorted(samples_dir.glob("*.pdf"))


# ════════════════════════════════════════════════════════════════════════════
# FakeExtractor — splits plain text into Section/Paragraph elements
# ════════════════════════════════════════════════════════════════════════════


_SECTION_RE = re.compile(
    r"^(?P<num>Section\s+\d+[A-Za-z]?|Article\s+\d+[A-Za-z]?|Art\.\s+\d+[A-Za-z]?)\.?\s*",
    re.MULTILINE,
)
_CROSS_REF_RE = re.compile(
    r"section\s+(\d+)|sub[- ]?section\s*\((\d+)\)|article\s+(\d+)",
    re.IGNORECASE,
)


@dataclass
class FakeExtractor:
    def extract(
        self, doc_meta: DocumentMeta, raw: bytes
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]:
        text = raw.decode("utf-8", errors="replace")
        elements: list[Element] = []
        edges: list[StructureEdge] = []

        # split into sections by the SECTION header regex
        matches = list(_SECTION_RE.finditer(text))
        if not matches:
            # one big paragraph fallback
            e = Element(
                element_id=f"{doc_meta.doc_id}_p0",
                doc_id=doc_meta.doc_id,
                element_type=ElementType.PARAGRAPH,
                text=text.strip(),
                page=1,
                char_start=0,
                char_end=len(text),
                extraction_confidence=0.99,
                ocr_engine=OcrEngine.NONE,
            )
            return [e], []

        for idx, m in enumerate(matches):
            start = m.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            section_id = f"{doc_meta.doc_id}_s{idx}"
            heading = m.group("num").strip()
            elements.append(
                Element(
                    element_id=section_id,
                    doc_id=doc_meta.doc_id,
                    element_type=ElementType.SECTION,
                    text=section_text,
                    page=1,
                    char_start=start,
                    char_end=end,
                    extraction_confidence=0.99,
                    legal_numbering=heading,
                )
            )
            # cross-references → edges
            for ref in _CROSS_REF_RE.finditer(section_text):
                token = next((g for g in ref.groups() if g), None)
                if token is None:
                    continue
                target_idx = _find_section_idx_by_number(matches, token)
                if target_idx is None or target_idx == idx:
                    continue
                target_id = f"{doc_meta.doc_id}_s{target_idx}"
                edges.append(
                    StructureEdge(
                        from_element=section_id,
                        to_element=target_id,
                        edge_type=__import__("rie_contracts").StructureEdgeType.CROSS_REFERENCE,
                        raw_reference=ref.group(0),
                    )
                )

        return elements, edges


def _find_section_idx_by_number(matches: list[re.Match[str]], token: str) -> int | None:
    for i, m in enumerate(matches):
        if token in m.group("num"):
            return i
    return None


# ════════════════════════════════════════════════════════════════════════════
# FakeRetrieval — naive BM25-ish keyword scorer over all elements
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class FakeRetrieval:
    elements: dict[str, Element] = field(default_factory=dict)
    edges: dict[str, list[StructureEdge]] = field(default_factory=dict)
    docs: dict[str, DocumentMeta] = field(default_factory=dict)

    def index_document(
        self,
        doc_meta: DocumentMeta,
        elements: Sequence[Element],
        edges: Sequence[StructureEdge],
    ) -> None:
        self.docs[doc_meta.doc_id] = doc_meta
        for e in elements:
            self.elements[e.element_id] = e
        self.edges[doc_meta.doc_id] = list(edges)

    def retrieve(
        self, query: str, jurisdiction: str | None, top_k: int = 8
    ) -> Sequence[RetrievalHit]:
        tokens = [t for t in re.split(r"\W+", query.lower()) if len(t) > 3]
        scored: list[tuple[float, Element]] = []
        for e in self.elements.values():
            if (
                jurisdiction
                and self.docs.get(e.doc_id)
                and self.docs[e.doc_id].jurisdiction != jurisdiction
            ):
                continue
            text_lower = e.text.lower()
            score = sum(text_lower.count(t) for t in tokens)
            if score > 0:
                scored.append((float(score), e))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        out: list[RetrievalHit] = []
        for score, e in scored[:top_k]:
            neighbours = self._neighbourhood(e.element_id)
            out.append(
                RetrievalHit(
                    element_id=e.element_id,
                    doc_id=e.doc_id,
                    score=score,
                    parent_element_id=e.element_id,
                    neighbourhood_element_ids=neighbours,
                    snippet=e.text[:240],
                    dense_score=score,
                    sparse_score=score,
                    rerank_score=score,
                )
            )
        return out

    def _neighbourhood(self, element_id: str) -> list[str]:
        e = self.elements.get(element_id)
        if not e:
            return []
        return [
            edge.to_element
            for edge in self.edges.get(e.doc_id, [])
            if edge.from_element == element_id and edge.to_element in self.elements
        ]


# ════════════════════════════════════════════════════════════════════════════
# FakeClassifier — keyword heuristics, returns indicator_id from the choices
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class FakeClassifier:
    get_element: Any  # Callable[[str], Element]

    def classify_clause(
        self,
        clause_element: Element,
        neighbourhood: Sequence[Element],
        pillar: PillarConfig,
        indicator_choices: Sequence[IndicatorConfig],
        n_samples: int = 3,
    ) -> Claim:
        text_lower = clause_element.text.lower()
        scored: list[tuple[int, IndicatorConfig]] = []
        for ind in indicator_choices:
            score = 0
            for lang_kws in ind.positive_keywords.values():
                for kw in lang_kws:
                    score += text_lower.count(kw.lower())
            scored.append((score, ind))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        winner = scored[0][1] if scored and scored[0][0] > 0 else indicator_choices[0]
        pattern = winner.clause_pattern or ClausePattern.OBLIGATION

        span = EvidenceSpan(
            span_id=f"{clause_element.doc_id}#{clause_element.char_start}-{clause_element.char_end}",
            element_id=clause_element.element_id,
            doc_id=clause_element.doc_id,
            char_start=clause_element.char_start,
            char_end=clause_element.char_end,
            role=SpanRole.PRIMARY,
        )
        regime = LegalRegime(
            primary_element_id=clause_element.element_id,
            member_element_ids=[clause_element.element_id] + [n.element_id for n in neighbourhood],
        )
        decomp = Decomposition(
            subject=clause_element.legal_numbering or "subject",
            condition=None,
            constraint=winner.name,
            context=None,
        )
        claim_id = hashlib.md5(
            f"{clause_element.element_id}:{winner.indicator_id}".encode()
        ).hexdigest()
        votes = {winner.indicator_id: max(n_samples - 1, 2)}
        return Claim(
            claim_id=claim_id,
            indicator_id=winner.indicator_id,
            pillar_id=pillar.pillar_id,
            clause_id=clause_element.element_id,
            jurisdiction="SAMPLE",  # overridden by orchestrator from run state
            clause_pattern=pattern,
            decomposition=decomp,
            evidence_spans=[span],
            regime=regime,
            layer1_status=Layer1Status.PENDING_VERIFICATION,
            model_confidence=0.9,
            self_consistency_votes=votes,
            created_at=datetime.now(UTC),
        )


# ════════════════════════════════════════════════════════════════════════════
# FakeVerifier — all gates pass on synthetic claims
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class FakeVerifier:
    def run_gate(
        self,
        gate: GateName,
        claim: Claim,
        get_element_text: Any,
    ) -> GateResult:
        return GateResult(gate=gate, passed=True, detail="fake-ok", ran_at=datetime.now(UTC))

    def verify(
        self,
        claim: Claim,
        get_element_text: Any,
    ) -> VerificationReport:
        gates = [self.run_gate(g, claim, get_element_text) for g in GateName]
        return VerificationReport(
            claim_id=claim.claim_id,
            gates=gates,
            status=VerificationStatus.VERIFIED,
            failure_reasons=[],
        )


# ════════════════════════════════════════════════════════════════════════════
# FakeCoverage — applies 3-state rule
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class FakeCoverage:
    def evaluate(
        self,
        jurisdiction: str,
        indicator_id: str,
        verified_claims: Sequence[Claim],
        gold_recall: float | None,
    ) -> CoverageRecord:
        matches = [
            c
            for c in verified_claims
            if c.indicator_id == indicator_id and c.jurisdiction == jurisdiction
        ]
        if matches:
            return CoverageRecord(
                jurisdiction=jurisdiction,
                indicator_id=indicator_id,
                state=CoverageState.EVIDENCE_FOUND,
                verified_claim_ids=[c.claim_id for c in matches],
            )
        if gold_recall is not None:
            miss = round((1 - gold_recall) * 100)
            return CoverageRecord(
                jurisdiction=jurisdiction,
                indicator_id=indicator_id,
                state=CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS,
                measured_recall=gold_recall,
                reason=f"recall={gold_recall}; miss risk ~{miss}%",
            )
        return CoverageRecord(
            jurisdiction=jurisdiction,
            indicator_id=indicator_id,
            state=CoverageState.INSUFFICIENT_COVERAGE,
            reason="no gold-set recall measured",
        )
