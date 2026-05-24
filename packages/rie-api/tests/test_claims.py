"""Claim listing + detail endpoints over the in-memory repo."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from rie_contracts import Claim


@pytest.mark.usefixtures("seed_claim")
def test_list_claims_returns_seeded_claim(client: TestClient) -> None:
    response = client.get("/v1/claims")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["claim_id"] == "claim-001"


@pytest.mark.usefixtures("seed_claim")
def test_list_claims_filter_by_pillar(client: TestClient) -> None:
    response = client.get("/v1/claims", params={"pillar_id": "6"})
    assert response.status_code == 200
    assert response.json()["total"] == 1

    response = client.get("/v1/claims", params={"pillar_id": "7"})
    assert response.status_code == 200
    assert response.json()["total"] == 0


@pytest.mark.usefixtures("seed_claim")
def test_list_claims_filter_by_jurisdiction(client: TestClient) -> None:
    response = client.get("/v1/claims", params={"jurisdiction": "SAMPLE"})
    assert response.json()["total"] == 1

    response = client.get("/v1/claims", params={"jurisdiction": "OTHER"})
    assert response.json()["total"] == 0


def test_get_claim_detail_returns_citations_and_verification(
    client: TestClient, seed_claim: Claim
) -> None:
    response = client.get(f"/v1/claims/{seed_claim.claim_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["claim"]["claim_id"] == seed_claim.claim_id
    assert body["verification"] is not None
    assert body["verification"]["status"] == "flagged"
    # Deterministic citations resolved from stored element text — never empty
    # for our seed.
    assert len(body["citations"]) == 1
    assert body["citations"][0]["text"]


def test_get_unknown_claim_returns_404(client: TestClient) -> None:
    response = client.get("/v1/claims/does-not-exist")
    assert response.status_code == 404
