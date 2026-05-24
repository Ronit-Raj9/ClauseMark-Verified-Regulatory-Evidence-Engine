"""Pipeline run endpoints.

`POST /v1/runs` triggers a pipeline run by invoking the injected `RunGraph`.
The route is synchronous in the HTTP sense — the orchestration layer may
itself be async / durable / interruptible, but from the API's POV we hand the
request to `RunGraph` and return what it tells us (a run_id + status).

`GET /v1/runs/{run_id}` aggregates per-run stats from persisted claims +
coverage. With no per-run table on the repo today, we report stats scoped to
the jurisdiction passed via query string when available; otherwise we return
totals + an "unknown" run status. This stays useful while keeping the route
contract stable for when a `runs` table arrives.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from fastapi import APIRouter, Depends, Query
from rie_contracts import Claim, Layer1Status
from rie_persistence.repository import DocumentRepository

from rie_api.deps import get_repo, get_run_graph
from rie_api.errors import DependencyUnavailableError
from rie_api.schemas import (
    RunRequest,
    RunResponse,
    RunStatusResponse,
)

router = APIRouter(prefix="/v1/runs", tags=["runs"])


@router.post(
    "",
    response_model=RunResponse,
    summary="Trigger a pipeline run for one jurisdiction",
)
def trigger_run(
    body: RunRequest,
    run_graph=Depends(get_run_graph),
) -> RunResponse:
    # Ensure body has a run_id (orchestration may overwrite, but we default).
    if body.run_id is None:
        body = body.model_copy(update={"run_id": uuid.uuid4().hex})
    try:
        result = run_graph(body)
    except DependencyUnavailableError:
        raise
    except Exception as exc:
        # Surface orchestration failures uniformly without leaking internals.
        return RunResponse(
            run_id=body.run_id or "",
            status="failed",
            detail=f"{type(exc).__name__}: {exc}",
        )
    return result


def _count_status(claims: Sequence[Claim], target: Layer1Status) -> int:
    return sum(1 for c in claims if c.layer1_status == target)


@router.get(
    "/{run_id}",
    response_model=RunStatusResponse,
    summary="Run status + aggregate counts",
)
def get_run_status(
    run_id: str,
    jurisdiction: str | None = Query(default=None, max_length=64),
    repo: DocumentRepository = Depends(get_repo),
) -> RunStatusResponse:
    # We don't (yet) persist a runs table; best-effort stats come from the
    # claims + coverage queried by jurisdiction when caller scopes it.
    from rie_persistence.models import ClaimRow
    from rie_persistence.repository import _to_claim  # type: ignore[attr-defined]
    from sqlalchemy import select

    with repo.session_factory() as session:
        stmt = select(ClaimRow)
        if jurisdiction:
            stmt = stmt.where(ClaimRow.jurisdiction == jurisdiction)
        rows = session.scalars(stmt).all()
        claims = [_to_claim(r, list(r.spans)) for r in rows]

    coverage = list(repo.list_coverage(jurisdiction=jurisdiction))
    return RunStatusResponse(
        run_id=run_id,
        jurisdiction=jurisdiction,
        # We can't yet distinguish "completed" from "still running" without a
        # runs table; report `unknown` so clients don't assume completion.
        status="unknown",
        claim_count=len(claims),
        verified_count=_count_status(claims, Layer1Status.VERIFIED),
        flagged_count=_count_status(claims, Layer1Status.FLAGGED),
        coverage_count=len(coverage),
    )
