"""Coverage endpoint: 3-state absence records read back through the API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.usefixtures("seed_coverage")
def test_list_coverage_returns_seeded_record(client: TestClient) -> None:
    response = client.get("/v1/coverage")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    row = body["items"][0]
    assert row["jurisdiction"] == "SAMPLE"
    assert row["indicator_id"] == "6.4"
    assert row["state"] == "no_evidence_in_searched_corpus"
    assert row["measured_recall"] == 0.82


@pytest.mark.usefixtures("seed_coverage")
def test_list_coverage_filter_by_jurisdiction(client: TestClient) -> None:
    hit = client.get("/v1/coverage", params={"jurisdiction": "SAMPLE"})
    assert hit.json()["total"] == 1

    miss = client.get("/v1/coverage", params={"jurisdiction": "ZZ"})
    assert miss.json()["total"] == 0
