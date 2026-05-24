"""Domain exception mapping → HTTP responses.

Routes raise plain Python exceptions; the global handler renders a uniform
JSON error envelope. Adapters' typed exceptions are mapped here so we don't
leak SQLAlchemy / Pydantic internals to the HTTP boundary.
"""

from __future__ import annotations

from fastapi import HTTPException, status


class APIError(HTTPException):
    """Base class for API errors. Carries a stable `code` for clients."""

    code: str = "error"

    def __init__(self, detail: str, status_code: int = status.HTTP_400_BAD_REQUEST) -> None:
        super().__init__(status_code=status_code, detail=detail)


class NotFoundError(APIError):
    code = "not_found"

    def __init__(self, detail: str = "resource not found") -> None:
        super().__init__(detail, status_code=status.HTTP_404_NOT_FOUND)


class ValidationError(APIError):
    code = "validation_error"

    def __init__(self, detail: str) -> None:
        super().__init__(detail, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


class DependencyUnavailableError(APIError):
    code = "dependency_unavailable"

    def __init__(self, detail: str = "downstream dependency unavailable") -> None:
        super().__init__(detail, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


def http_error_envelope(code: str, detail: str, status_code: int) -> dict[str, object]:
    """The on-the-wire shape every error response uses."""
    return {
        "error": {
            "code": code,
            "message": detail,
            "status": status_code,
        }
    }
