"""Ports — `Protocol` definitions for every adapter seam.

Every adapter implementation lives in its own package and depends ONLY on this
module (plus `rie_contracts.models`). Orchestration wires concrete adapters
behind these `Protocol`s, so each adapter is independently swappable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from rie_contracts.models import (
    Claim,
    CoverageRecord,
    DocumentMeta,
    Element,
    GateName,
    GateResult,
    GoldItem,
    IndicatorConfig,
    PillarConfig,
    RegistryEntry,
    RetrievalHit,
    ReviewRecord,
    SourceRegistryEntry,
    StructureEdge,
    VerificationReport,
)


@runtime_checkable
class IngestPort(Protocol):
    """Load source registries + sample-law files into `DocumentMeta` + raw bytes."""

    def load_source_registry(self, jurisdiction: str) -> Sequence[SourceRegistryEntry]: ...

    def load_document_bytes(self, entry: SourceRegistryEntry) -> tuple[DocumentMeta, bytes]: ...

    def list_sample_laws(self, samples_dir: Path) -> Sequence[Path]: ...


@runtime_checkable
class DocumentExtractorPort(Protocol):
    """Extract `Element`s + structure graph from a document. Routes by document type."""

    def extract(
        self, doc_meta: DocumentMeta, raw: bytes
    ) -> tuple[Sequence[Element], Sequence[StructureEdge]]: ...


@runtime_checkable
class StructureGraphBuilderPort(Protocol):
    """Build cross-reference / definition / proviso edges over extracted elements."""

    def build(self, elements: Sequence[Element]) -> Sequence[StructureEdge]: ...


@runtime_checkable
class VectorStorePort(Protocol):
    """Hybrid (dense + sparse) vector store for child chunks. Qdrant in production."""

    def ensure_collection(self, name: str, dense_dim: int) -> None: ...

    def upsert(
        self,
        collection: str,
        ids: Sequence[str],
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[dict[int, float]],
        payloads: Sequence[dict[str, str | int | float | bool | None]],
    ) -> None: ...

    def search_hybrid(
        self,
        collection: str,
        dense_query: Sequence[float],
        sparse_query: dict[int, float],
        top_k: int,
    ) -> Sequence[tuple[str, float]]: ...


@runtime_checkable
class RerankerPort(Protocol):
    """Cross-encoder reranker. Returns (id, score) sorted desc."""

    def rerank(
        self, query: str, candidates: Sequence[tuple[str, str]]
    ) -> Sequence[tuple[str, float]]: ...


@runtime_checkable
class RetrievalPort(Protocol):
    """Parent-document retrieval: child chunks for matching, parent + neighbourhood for context."""

    def index_document(
        self, doc_meta: DocumentMeta, elements: Sequence[Element], edges: Sequence[StructureEdge]
    ) -> None: ...

    def retrieve(
        self,
        query: str,
        jurisdiction: str | None,
        top_k: int = 8,
    ) -> Sequence[RetrievalHit]: ...


@runtime_checkable
class ClassifierPort(Protocol):
    """LLM classification with constrained decoding. Emits Claim with span IDs, never citations."""

    def classify_clause(
        self,
        clause_element: Element,
        neighbourhood: Sequence[Element],
        pillar: PillarConfig,
        indicator_choices: Sequence[IndicatorConfig],
        n_samples: int = 3,
    ) -> Claim: ...


@runtime_checkable
class VerifierPort(Protocol):
    """Run the 4 gates over a claim. Returns one `GateResult` per gate."""

    def run_gate(
        self,
        gate: GateName,
        claim: Claim,
        get_element_text: ElementTextResolver,
    ) -> GateResult: ...

    def verify(
        self,
        claim: Claim,
        get_element_text: ElementTextResolver,
    ) -> VerificationReport: ...


@runtime_checkable
class CoverageReasonerPort(Protocol):
    """3-state absence reasoning. Carries measured gold-set recall."""

    def evaluate(
        self,
        jurisdiction: str,
        indicator_id: str,
        verified_claims: Sequence[Claim],
        gold_recall: float | None,
    ) -> CoverageRecord: ...


@runtime_checkable
class ConfigRepositoryPort(Protocol):
    """Loads + validates pillar configs, source registries, gold sets."""

    def load_registry(self) -> Sequence[RegistryEntry]: ...

    def load_pillar(self, pillar_id: str) -> PillarConfig: ...

    def load_source_registry(self, jurisdiction: str) -> Sequence[SourceRegistryEntry]: ...

    def load_gold(self, pillar_id: str) -> Sequence[GoldItem]: ...


@runtime_checkable
class DocumentRepositoryPort(Protocol):
    """Persistence for documents, elements, claims, verifications, coverage, reviews."""

    def save_document(self, meta: DocumentMeta) -> None: ...

    def save_elements(
        self, doc_id: str, elements: Sequence[Element], edges: Sequence[StructureEdge]
    ) -> None: ...

    def get_element_text(self, element_id: str) -> str: ...

    def get_element(self, element_id: str) -> Element: ...

    def get_elements(self, element_ids: Sequence[str]) -> Sequence[Element]: ...

    def save_claim(self, claim: Claim) -> None: ...

    def save_verification(self, report: VerificationReport) -> None: ...

    def save_coverage(self, record: CoverageRecord) -> None: ...

    def save_review(self, review: ReviewRecord) -> None: ...

    def list_claims_for_review(
        self, jurisdiction: str | None = None, pillar_id: str | None = None
    ) -> Sequence[Claim]: ...

    def list_coverage(self, jurisdiction: str | None = None) -> Sequence[CoverageRecord]: ...


@runtime_checkable
class HumanReviewQueuePort(Protocol):
    """LangGraph interrupt + review queue."""

    def enqueue(self, claim: Claim, report: VerificationReport) -> str: ...

    def resolve(self, claim_id: str, review: ReviewRecord) -> None: ...

    def pending(self) -> Iterable[tuple[Claim, VerificationReport]]: ...


@runtime_checkable
class EvaluatorPort(Protocol):
    """Gold-set runner + RAGAS metrics."""

    def evaluate_pillar(self, pillar_id: str, claims: Sequence[Claim]) -> dict[str, float]: ...

    def measure_retrieval_recall(
        self, pillar_id: str, retrieved_per_query: Sequence[Sequence[str]]
    ) -> float: ...


# Forward-declared callable type to break a circular dependency between the
# verifier and persistence — verifier asks "what does this element actually say?"
# and the orchestrator passes a function bound to the document repo.
class ElementTextResolver(Protocol):
    def __call__(self, element_id: str) -> str: ...
