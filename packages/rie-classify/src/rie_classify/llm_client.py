"""LLM client Protocol + concrete Ollama / vLLM adapters.

Both concrete clients send the JSON schema with their request so the server
performs grammar-constrained decoding (XGrammar under the hood for vLLM
and Ollama). The client interface is deliberately tiny — only
``complete_json(prompt, schema, temperature, seed)`` — so the service code
stays trivially mockable.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


@runtime_checkable
class LlmClient(Protocol):
    """Minimal LLM-call seam used by the classification service."""

    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        temperature: float,
        seed: int | None,
    ) -> dict[str, Any]: ...


# ════════════════════════════════════════════════════════════════════════════
# Retry policy — shared by both concrete clients.
# ════════════════════════════════════════════════════════════════════════════

_RETRYABLE_HTTP = (httpx.HTTPError, httpx.TimeoutException)


def _retry_decorator() -> Any:
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        retry=retry_if_exception_type(_RETRYABLE_HTTP),
        reraise=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# Ollama — POSTs /api/chat with `format=<schema>` for structured output.
# ════════════════════════════════════════════════════════════════════════════


class OllamaClient:
    """Adapter for Ollama's `/api/chat` endpoint with JSON-schema structured output.

    Ollama uses XGrammar under the hood when you pass a JSON schema via the
    ``format`` field, giving 100% schema compliance.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b-instruct",
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        temperature: float,
        seed: int | None,
    ) -> dict[str, Any]:
        return self._post(prompt, schema, temperature, seed)

    @_retry_decorator()
    def _post(
        self,
        prompt: str,
        schema: dict[str, Any],
        temperature: float,
        seed: int | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": schema,  # XGrammar-enforced schema
            "options": options,
        }
        response = self._client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        body = response.json()
        content = body["message"]["content"]
        if not isinstance(content, str):
            raise ValueError(f"Ollama returned non-string content: {type(content)!r}")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError(f"Ollama JSON output not an object: {type(parsed)!r}")
        return parsed

    def close(self) -> None:
        self._client.close()


# ════════════════════════════════════════════════════════════════════════════
# vLLM — POSTs /v1/chat/completions with `guided_json` extra body.
# ════════════════════════════════════════════════════════════════════════════


class VllmClient:
    """Adapter for vLLM's OpenAI-compatible server with `guided_json` decoding.

    vLLM 0.8.5+ supports XGrammar-backed structured output via the
    ``guided_json`` extra request body field.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self._client = client or httpx.Client(timeout=timeout)

    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        temperature: float,
        seed: int | None,
    ) -> dict[str, Any]:
        return self._post(prompt, schema, temperature, seed)

    @_retry_decorator()
    def _post(
        self,
        prompt: str,
        schema: dict[str, Any],
        temperature: float,
        seed: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "guided_json": schema,  # XGrammar-enforced schema
        }
        if seed is not None:
            payload["seed"] = seed

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError(f"vLLM returned non-string content: {type(content)!r}")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError(f"vLLM JSON output not an object: {type(parsed)!r}")
        return parsed

    def close(self) -> None:
        self._client.close()


__all__ = ["LlmClient", "OllamaClient", "VllmClient"]
