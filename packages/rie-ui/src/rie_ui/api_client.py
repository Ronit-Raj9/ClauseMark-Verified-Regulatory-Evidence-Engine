"""Typed httpx wrapper around the RIE API.

The UI talks to the API exclusively over HTTP — never in-process imports of
orchestration / persistence. Base URL is taken from ``RIE_API_URL`` (default
``http://localhost:8080``).

Every method returns a Pydantic model from ``rie_contracts``. HTTP errors are
surfaced as :class:`ApiError`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from rie_contracts.models import (
    Claim,
    CoverageRecord,
    Element,
    Layer1Status,
    Layer2Recommendation,
    ReviewDecision,
    ReviewRecord,
    ScoreBand,
    VerificationReport,
    VerificationStatus,
)

DEFAULT_BASE_URL: Final[str] = "http://localhost:8080"
DEFAULT_TIMEOUT: Final[float] = 30.0


class ApiError(RuntimeError):
    """Raised for any non-2xx response or transport-level failure."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ════════════════════════════════════════════════════════════════════════════
# Wire models — request / response envelopes around the frozen contracts.
# ════════════════════════════════════════════════════════════════════════════


class RunRequest(BaseModel):
    """Payload for ``POST /v1/runs``."""

    model_config = ConfigDict(extra="forbid")
    jurisdiction: str = Field(min_length=2, max_length=64)
    pillar_ids: list[str] = Field(default_factory=list)


RunStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]


class RunStatusResponse(BaseModel):
    """Response from ``GET /v1/runs/{run_id}``."""

    model_config = ConfigDict(extra="allow")
    run_id: str
    jurisdiction: str
    pillar_ids: list[str] = Field(default_factory=list)
    status: RunStatus
    created_at: datetime
    updated_at: datetime | None = None
    error: str | None = None
    progress: dict[str, int] | None = None


class ClaimSummary(BaseModel):
    """Compact row for the Claims table — derived server-side."""

    model_config = ConfigDict(extra="allow")
    claim_id: str
    jurisdiction: str
    pillar_id: str
    indicator_id: str
    clause_id: str
    layer1_status: Layer1Status
    verification_status: VerificationStatus | None = None
    snippet: str = ""


class ClaimDetail(BaseModel):
    """Full claim payload for the detail page."""

    model_config = ConfigDict(extra="allow")
    claim: Claim
    document_text: str
    document_id: str
    document_title: str = ""
    elements: list[Element] = Field(default_factory=list)
    verification: VerificationReport | None = None
    layer2: Layer2Recommendation | None = None
    reviews: list[ReviewRecord] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    """Payload for ``POST /v1/reviews``."""

    model_config = ConfigDict(extra="forbid")
    claim_id: str
    reviewer: str
    decision: ReviewDecision
    corrected_score: ScoreBand | None = None
    note: str = ""


_CLAIMS_ADAPTER: Final[TypeAdapter[list[ClaimSummary]]] = TypeAdapter(list[ClaimSummary])
_COVERAGE_ADAPTER: Final[TypeAdapter[list[CoverageRecord]]] = TypeAdapter(list[CoverageRecord])


# ════════════════════════════════════════════════════════════════════════════
# Client
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class _Settings:
    base_url: str
    timeout: float


class ApiClient:
    """Thin typed wrapper. All methods are synchronous (Streamlit-friendly)."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        resolved = (base_url or os.environ.get("RIE_API_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._settings = _Settings(base_url=resolved, timeout=timeout)
        self._client = client or httpx.Client(base_url=resolved, timeout=timeout)
        self._owns_client = client is None

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ApiClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def base_url(self) -> str:
        return self._settings.base_url

    # ──────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
    ) -> Any:
        try:
            resp = self._client.request(method, path, params=params, json=json)
        except httpx.HTTPError as exc:  # pragma: no cover - exercised via respx side effects
            raise ApiError(f"transport error: {exc}") from exc
        if resp.status_code >= 400:
            body = resp.text[:512]
            raise ApiError(
                f"{method} {path} -> {resp.status_code}: {body}",
                status_code=resp.status_code,
            )
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    # ──────────────────────────────────────────────────────────────────────
    # Runs
    # ──────────────────────────────────────────────────────────────────────

    def start_run(self, jurisdiction: str, pillar_ids: list[str]) -> RunStatusResponse:
        payload = RunRequest(jurisdiction=jurisdiction, pillar_ids=pillar_ids).model_dump(
            mode="json"
        )
        data = self._request("POST", "/v1/runs", json=payload)
        return RunStatusResponse.model_validate(data)

    def get_run(self, run_id: str) -> RunStatusResponse:
        data = self._request("GET", f"/v1/runs/{run_id}")
        return RunStatusResponse.model_validate(data)

    def list_runs(self) -> list[RunStatusResponse]:
        data = self._request("GET", "/v1/runs")
        return [RunStatusResponse.model_validate(item) for item in (data or [])]

    # ──────────────────────────────────────────────────────────────────────
    # Claims
    # ──────────────────────────────────────────────────────────────────────

    def list_claims(
        self,
        *,
        jurisdiction: str | None = None,
        pillar_id: str | None = None,
        indicator_id: str | None = None,
        status: Layer1Status | None = None,
    ) -> list[ClaimSummary]:
        params: dict[str, Any] = {}
        if jurisdiction:
            params["jurisdiction"] = jurisdiction
        if pillar_id:
            params["pillar_id"] = pillar_id
        if indicator_id:
            params["indicator_id"] = indicator_id
        if status is not None:
            params["status"] = status.value
        data = self._request("GET", "/v1/claims", params=params or None)
        return _CLAIMS_ADAPTER.validate_python(data or [])

    def get_claim(self, claim_id: str) -> ClaimDetail:
        data = self._request("GET", f"/v1/claims/{claim_id}")
        return ClaimDetail.model_validate(data)

    # ──────────────────────────────────────────────────────────────────────
    # Reviews
    # ──────────────────────────────────────────────────────────────────────

    def submit_review(
        self,
        *,
        claim_id: str,
        reviewer: str,
        decision: ReviewDecision,
        corrected_score: ScoreBand | None = None,
        note: str = "",
    ) -> ReviewRecord:
        payload = ReviewRequest(
            claim_id=claim_id,
            reviewer=reviewer,
            decision=decision,
            corrected_score=corrected_score,
            note=note,
        ).model_dump(mode="json")
        data = self._request("POST", "/v1/reviews", json=payload)
        return ReviewRecord.model_validate(data)

    # ──────────────────────────────────────────────────────────────────────
    # Coverage
    # ──────────────────────────────────────────────────────────────────────

    def list_coverage(self, *, jurisdiction: str | None = None) -> list[CoverageRecord]:
        params: dict[str, Any] = {}
        if jurisdiction:
            params["jurisdiction"] = jurisdiction
        data = self._request("GET", "/v1/coverage", params=params or None)
        return _COVERAGE_ADAPTER.validate_python(data or [])

    # ──────────────────────────────────────────────────────────────────────
    # Audit package
    # ──────────────────────────────────────────────────────────────────────

    def get_audit_package(self, jurisdiction: str) -> dict[str, Any]:
        data = self._request("GET", f"/v1/audit/{jurisdiction}")
        if not isinstance(data, dict):
            raise ApiError(f"audit package for {jurisdiction} is not an object")
        return data

    # ──────────────────────────────────────────────────────────────────────
    # Misc
    # ──────────────────────────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        data = self._request("GET", "/healthz")
        return data if isinstance(data, dict) else {"status": "ok"}


__all__ = [
    "DEFAULT_BASE_URL",
    "ApiClient",
    "ApiError",
    "ClaimDetail",
    "ClaimSummary",
    "ReviewRequest",
    "RunRequest",
    "RunStatus",
    "RunStatusResponse",
]
