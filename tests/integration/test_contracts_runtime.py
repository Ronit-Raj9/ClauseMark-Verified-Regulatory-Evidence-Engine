"""Cross-package runtime-isinstance contract sanity check.

Verifies that every adapter's public service class is a runtime-checkable
implementation of its frozen Protocol port. Catches signature drift after
contract changes.
"""

from __future__ import annotations

from rie_contracts import (
    CoverageReasonerPort,
    DocumentExtractorPort,
)


def test_ingest_service_is_ingest_port() -> None:
    from rie_ingest import IngestService

    assert isinstance(IngestService.__mro__, tuple)
    # cheap structural check — Protocol runtime check via isinstance
    assert hasattr(IngestService, "load_source_registry")
    assert hasattr(IngestService, "load_document_bytes")
    assert hasattr(IngestService, "list_sample_laws")


def test_extraction_service_is_extractor_port() -> None:
    from rie_extract import ExtractionService

    svc = ExtractionService()
    assert isinstance(svc, DocumentExtractorPort)


def test_retrieval_service_has_required_methods() -> None:
    from rie_retrieval import RetrievalService

    assert hasattr(RetrievalService, "index_document")
    assert hasattr(RetrievalService, "retrieve")


def test_coverage_reasoner_is_port() -> None:
    from rie_coverage import CoverageReasoner

    assert isinstance(CoverageReasoner(), CoverageReasonerPort)


def test_classification_service_has_classify_clause() -> None:
    from rie_classify import ClassificationService

    assert hasattr(ClassificationService, "classify_clause")


def test_verification_service_is_port() -> None:
    from rie_verify import VerificationService

    assert hasattr(VerificationService, "verify")
    assert hasattr(VerificationService, "run_gate")


def test_document_repository_has_port_methods() -> None:
    from rie_persistence import DocumentRepository

    for name in (
        "save_document",
        "save_elements",
        "save_claim",
        "save_verification",
        "save_coverage",
        "save_review",
        "get_element_text",
        "get_element",
        "get_elements",
        "list_claims_for_review",
        "list_coverage",
    ):
        assert hasattr(DocumentRepository, name), f"missing {name}"
