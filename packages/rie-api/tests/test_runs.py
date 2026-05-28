"""Run trigger + resume endpoint tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_trigger_run_returns_accepted(client: TestClient) -> None:
    response = client.post(
        "/v1/runs",
        json={"jurisdiction": "SAMPLE", "pillar_ids": ["6"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "accepted"
    assert body["run_id"]
    assert "pillar" in body["detail"]


def test_trigger_run_mints_run_id_when_omitted(client: TestClient) -> None:
    response = client.post(
        "/v1/runs",
        json={"jurisdiction": "SAMPLE", "pillar_ids": ["6"]},
    )
    assert response.status_code == 200
    assert response.json()["run_id"]


def test_resume_run_returns_structured_response(client: TestClient) -> None:
    response = client.post(
        "/v1/runs/no-interrupt/resume",
        json={"decisions": {"claim-001": "accept"}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "no-interrupt"
    assert body["status"] in {"completed", "failed", "interrupted", "unknown"}
    assert "detail" in body
