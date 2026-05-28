"""Tests for ``rie_ui.api_client.ApiClient`` — respx-mocked round trips."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import respx
from rie_contracts.models import (
    Claim,
    ClausePattern,
    CoverageRecord,
    CoverageState,
    Decomposition,
    EvidenceSpan,
    GateName,
    GateResult,
    Layer1Status,
    Layer2Recommendation,
    LegalRegime,
    ReviewDecision,
    ScoreBand,
    SpanRole,
    VerificationReport,
    VerificationStatus,
)
from rie_ui.api_client import (
    DEFAULT_BASE_URL,
    ApiClient,
    ApiError,
    ClaimSummary,
)

BASE = "http://api.test"
NOW = datetime(2026, 5, 24, 12, 0, 0, tzinfo=UTC)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture()
def client() -> ApiClient:
    transport = httpx.HTTPTransport()
    httpx_client = httpx.Client(base_url=BASE, transport=transport)
    api = ApiClient(BASE, client=httpx_client)
    try:
        yield api
    finally:
        api.close()


def _claim_payload() -> dict[str, Any]:
    span = EvidenceSpan(
        span_id="doc1#10-20",
        element_id="doc1_e1",
        doc_id="doc1",
        char_start=10,
        char_end=20,
        role=SpanRole.PRIMARY,
    )
    claim = Claim(
        claim_id="c-1",
        indicator_id="6.4",
        pillar_id="6",
        clause_id="doc1_art1",
        jurisdiction="SG",
        clause_pattern=ClausePattern.CONDITIONAL_REGIME,
        decomposition=Decomposition(subject="orgs", constraint="adequacy"),
        evidence_spans=[span],
        regime=LegalRegime(primary_element_id="doc1_e1", member_element_ids=["doc1_e1"]),
        layer1_status=Layer1Status.VERIFIED,
        created_at=NOW,
    )
    gates = [
        GateResult(gate=g, passed=True, ran_at=NOW)
        for g in (
            GateName.SPAN_EXISTENCE,
            GateName.VERBATIM_MATCH,
            GateName.ENTAILMENT,
            GateName.SELF_CONSISTENCY,
        )
    ]
    report = VerificationReport(
        claim_id="c-1",
        gates=gates,
        status=VerificationStatus.VERIFIED,
    )
    layer2 = Layer2Recommendation(
        claim_id="c-1",
        indicator_id="6.4",
        recommended_band=ScoreBand.HALF,
        rationale="Conditional adequacy regime present.",
        open_questions=["Is the adequacy list current?"],
        human_confirmation_required=True,
    )
    return {
        "claim": claim.model_dump(mode="json"),
        "verification": report.model_dump(mode="json"),
        "citations": [
            {
                "span_id": span.span_id,
                "doc_id": span.doc_id,
                "element_id": span.element_id,
                "text": "verified text",
                "char_start": span.char_start,
                "char_end": span.char_end,
                "role": span.role.value,
            }
        ],
        "layer2": layer2.model_dump(mode="json"),
    }


# ──────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────


def test_default_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RIE_API_URL", raising=False)
    api = ApiClient()
    try:
        assert api.base_url == DEFAULT_BASE_URL
    finally:
        api.close()


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIE_API_URL", "http://other:9000/")
    api = ApiClient()
    try:
        # Trailing slash stripped.
        assert api.base_url == "http://other:9000"
    finally:
        api.close()


# ──────────────────────────────────────────────────────────────────────
# Runs
# ──────────────────────────────────────────────────────────────────────


@respx.mock(base_url=BASE)
def test_start_run_posts_payload(respx_mock: respx.MockRouter, client: ApiClient) -> None:
    route = respx_mock.post("/v1/runs").mock(
        return_value=httpx.Response(
            201,
            json={
                "run_id": "r-1",
                "jurisdiction": "SG",
                "pillar_ids": ["6", "7"],
                "status": "pending",
                "created_at": NOW.isoformat(),
            },
        )
    )
    out = client.start_run("SG", ["6", "7"])
    assert route.called
    sent = route.calls.last.request
    assert sent.method == "POST"
    body = sent.read().decode()
    assert "SG" in body and '"6"' in body and '"7"' in body
    assert out.run_id == "r-1"
    assert out.status == "pending"
    assert out.pillar_ids == ["6", "7"]


@respx.mock(base_url=BASE)
def test_get_run(respx_mock: respx.MockRouter, client: ApiClient) -> None:
    respx_mock.get("/v1/runs/r-1").mock(
        return_value=httpx.Response(
            200,
            json={
                "run_id": "r-1",
                "jurisdiction": "SG",
                "pillar_ids": ["6"],
                "status": "succeeded",
                "created_at": NOW.isoformat(),
            },
        )
    )
    out = client.get_run("r-1")
    assert out.status == "succeeded"


@respx.mock(base_url=BASE)
def test_list_runs_empty(respx_mock: respx.MockRouter, client: ApiClient) -> None:
    respx_mock.get("/v1/runs").mock(return_value=httpx.Response(200, json=[]))
    assert client.list_runs() == []


# ──────────────────────────────────────────────────────────────────────
# Claims
# ──────────────────────────────────────────────────────────────────────


@respx.mock(base_url=BASE)
def test_list_claims_no_filters(respx_mock: respx.MockRouter, client: ApiClient) -> None:
    route = respx_mock.get("/v1/claims").mock(
        return_value=httpx.Response(
            200,
            json={
                "total": 1,
                "items": [
                    {
                        "claim_id": "c-1",
                        "jurisdiction": "SG",
                        "pillar_id": "6",
                        "indicator_id": "6.4",
                        "clause_id": "doc1_a1",
                        "layer1_status": "verified",
                        "verification_status": "verified",
                        "snippet": "...",
                    }
                ],
            },
        )
    )
    out = client.list_claims()
    assert route.called
    assert len(out) == 1
    assert isinstance(out[0], ClaimSummary)
    assert out[0].claim_id == "c-1"
    assert out[0].layer1_status is Layer1Status.VERIFIED
    # No filter params present.
    assert route.calls.last.request.url.params.multi_items() == []


@respx.mock(base_url=BASE)
def test_list_claims_with_filters(respx_mock: respx.MockRouter, client: ApiClient) -> None:
    route = respx_mock.get("/v1/claims").mock(
        return_value=httpx.Response(200, json={"total": 0, "items": []})
    )
    client.list_claims(
        jurisdiction="SG",
        pillar_id="6",
        indicator_id="6.4",
        status=Layer1Status.FLAGGED,
    )
    params = dict(route.calls.last.request.url.params.multi_items())
    assert params == {
        "jurisdiction": "SG",
        "pillar_id": "6",
        "indicator_id": "6.4",
        "status": "flagged",
    }


@respx.mock(base_url=BASE)
def test_get_claim_returns_detail(respx_mock: respx.MockRouter, client: ApiClient) -> None:
    payload = _claim_payload()
    respx_mock.get("/v1/claims/c-1").mock(return_value=httpx.Response(200, json=payload))
    respx_mock.get("/v1/claims/c-1/reviews").mock(
        return_value=httpx.Response(200, json={"total": 0, "items": []})
    )
    detail = client.get_claim("c-1")
    assert detail.claim.claim_id == "c-1"
    assert len(detail.citations) == 1
    assert detail.citations[0].text == "verified text"
    assert detail.verification is not None
    assert detail.verification.status is VerificationStatus.VERIFIED
    assert detail.layer2 is not None
    assert detail.layer2.recommended_band is ScoreBand.HALF
    assert detail.layer2.human_confirmation_required is True
    assert detail.layer2.open_questions == ["Is the adequacy list current?"]


# ──────────────────────────────────────────────────────────────────────
# Reviews
# ──────────────────────────────────────────────────────────────────────


@respx.mock(base_url=BASE)
def test_submit_review_posts_payload(respx_mock: respx.MockRouter, client: ApiClient) -> None:
    route = respx_mock.post("/v1/reviews").mock(
        return_value=httpx.Response(
            201,
            json={
                "claim_id": "c-1",
                "reviewer": "alice",
                "decision": "correct",
                "corrected_score": "0.5",
                "note": "needs adequacy list",
                "decided_at": NOW.isoformat(),
            },
        )
    )
    out = client.submit_review(
        claim_id="c-1",
        reviewer="alice",
        decision=ReviewDecision.CORRECT,
        corrected_score=ScoreBand.HALF,
        note="needs adequacy list",
    )
    assert route.called
    body = route.calls.last.request.read().decode()
    assert "alice" in body and "correct" in body and "0.5" in body
    assert out.reviewer == "alice"
    assert out.decision is ReviewDecision.CORRECT
    assert out.corrected_score is ScoreBand.HALF


# ──────────────────────────────────────────────────────────────────────
# Coverage
# ──────────────────────────────────────────────────────────────────────


@respx.mock(base_url=BASE)
def test_list_coverage(respx_mock: respx.MockRouter, client: ApiClient) -> None:
    record = CoverageRecord(
        jurisdiction="SG",
        indicator_id="6.4",
        state=CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS,
        measured_recall=0.82,
        reason="not in corpus",
    )
    respx_mock.get("/v1/coverage").mock(
        return_value=httpx.Response(200, json=[record.model_dump(mode="json")])
    )
    out = client.list_coverage(jurisdiction="SG")
    assert len(out) == 1
    assert out[0].state is CoverageState.NO_EVIDENCE_IN_SEARCHED_CORPUS
    assert out[0].measured_recall == 0.82


# ──────────────────────────────────────────────────────────────────────
# Audit
# ──────────────────────────────────────────────────────────────────────


@respx.mock(base_url=BASE)
def test_audit_package_returns_dict(respx_mock: respx.MockRouter, client: ApiClient) -> None:
    payload = {"jurisdiction": "SG", "claims": [], "coverage": []}
    respx_mock.get("/v1/audit/SG").mock(return_value=httpx.Response(200, json=payload))
    out = client.get_audit_package("SG")
    assert out == payload


@respx.mock(base_url=BASE)
def test_audit_package_non_object_raises(respx_mock: respx.MockRouter, client: ApiClient) -> None:
    respx_mock.get("/v1/audit/SG").mock(return_value=httpx.Response(200, json=[1, 2, 3]))
    with pytest.raises(ApiError):
        client.get_audit_package("SG")


# ──────────────────────────────────────────────────────────────────────
# Error handling
# ──────────────────────────────────────────────────────────────────────


@respx.mock(base_url=BASE)
def test_4xx_raises_api_error_with_status(respx_mock: respx.MockRouter, client: ApiClient) -> None:
    respx_mock.get("/v1/claims/nope").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    with pytest.raises(ApiError) as ei:
        client.get_claim("nope")
    assert ei.value.status_code == 404


@respx.mock(base_url=BASE)
def test_5xx_raises_api_error(respx_mock: respx.MockRouter, client: ApiClient) -> None:
    respx_mock.get("/v1/coverage").mock(return_value=httpx.Response(503, text="boom"))
    with pytest.raises(ApiError) as ei:
        client.list_coverage()
    assert ei.value.status_code == 503


@respx.mock(base_url=BASE)
def test_transport_error_raises_api_error(respx_mock: respx.MockRouter, client: ApiClient) -> None:
    respx_mock.get("/v1/coverage").mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(ApiError):
        client.list_coverage()


# ──────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────


@respx.mock(base_url=BASE)
def test_health_returns_dict(respx_mock: respx.MockRouter, client: ApiClient) -> None:
    respx_mock.get("/healthz").mock(return_value=httpx.Response(200, json={"status": "ok"}))
    out = client.health()
    assert out == {"status": "ok"}


# ──────────────────────────────────────────────────────────────────────
# Context-manager
# ──────────────────────────────────────────────────────────────────────


def test_context_manager_closes_owned_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIE_API_URL", "http://x")
    with ApiClient() as api:
        assert api.base_url == "http://x"
    # No assertion needed — closing twice is a no-op.
