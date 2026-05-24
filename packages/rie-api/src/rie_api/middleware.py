"""Request-id middleware, structured logging, and a global exception handler.

Every request gets an `X-Request-ID` (echoed in the response and used as the
correlation id for log lines). Unhandled exceptions are converted to a JSON
error envelope so the client never sees a HTML traceback page.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from rie_api.errors import APIError, http_error_envelope

logger = logging.getLogger("rie_api")

REQUEST_ID_HEADER = "x-request-id"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Tag each request with an id + log start/end with latency in ms."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = rid
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "request.error",
                extra={
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "elapsed_ms": round(elapsed_ms, 2),
                },
            )
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = rid
        logger.info(
            "request.completed",
            extra={
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )
        return response


def install_exception_handlers(app: FastAPI) -> None:
    """Register handlers that render the uniform error envelope."""

    @app.exception_handler(APIError)
    async def _handle_api_error(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=http_error_envelope(exc.code, str(exc.detail), exc.status_code),
        )

    @app.exception_handler(KeyError)
    async def _handle_key_error(_: Request, exc: KeyError) -> JSONResponse:
        # Repositories raise KeyError on missing rows.
        return JSONResponse(
            status_code=404,
            content=http_error_envelope("not_found", str(exc), 404),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("request.unhandled", extra={"exc_type": type(exc).__name__})
        return JSONResponse(
            status_code=500,
            content=http_error_envelope("internal_error", "internal server error", 500),
        )


def configure_logging(level: str = "INFO") -> None:
    """Idempotent logging setup. Plain text formatter — replace with JSON in prod."""
    root = logging.getLogger()
    if root.handlers:
        for h in root.handlers:
            h.setLevel(level)
        root.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)


__all__: tuple[str, ...] = (
    "REQUEST_ID_HEADER",
    "RequestIDMiddleware",
    "configure_logging",
    "install_exception_handlers",
)


# Silences "unused" lint on Any import — kept for downstream typing convenience.
_ANY: Any = None
