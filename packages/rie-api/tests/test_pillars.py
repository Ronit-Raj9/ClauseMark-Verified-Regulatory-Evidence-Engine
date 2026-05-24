"""`/v1/pillars` reads from the real registry — should return all 12 pillars."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_list_pillars_returns_all_twelve(client: TestClient) -> None:
    response = client.get("/v1/pillars")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 12
    ids = {p["pillar_id"] for p in body["items"]}
    # The registry covers pillars 1..12 (as strings, since `pillar_id: str`).
    assert ids == {str(i) for i in range(1, 13)}

    # Spot-check: each item carries the expected fields for the UI.
    sample = body["items"][0]
    assert {
        "pillar_id",
        "pillar_name",
        "cluster",
        "document_profile",
        "status",
        "config_path",
    }.issubset(sample.keys())


def test_get_pillar_returns_indicators(client: TestClient) -> None:
    response = client.get("/v1/pillars/6")
    assert response.status_code == 200
    body = response.json()
    assert body["pillar_id"] == "6"
    assert body["status"] == "built"
    assert isinstance(body["indicators"], list)
    assert len(body["indicators"]) >= 1


def test_get_unknown_pillar_returns_404(client: TestClient) -> None:
    response = client.get("/v1/pillars/9999")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
