"""VLM client protocol + concrete Ollama / fake implementations.

The VLM-OCR path (see ``systemArchitecture.md`` §5.2) is *never* labelled
"faithful" — a vision-language model **generates** text and can
hallucinate. Every VLM transcription carries:

  • the engine tag :class:`rie_contracts.OcrEngine.VLM`
  • an ``extraction_confidence`` (returned by the client) that downstream
    gates can route on
  • a ``corrected`` flag (``False`` until a human edits the row)

This module owns the network seam only; element synthesis lives in
:mod:`rie_extract.adapters.vlm_ocr`.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — kept as module constants so tests can monkeypatch the env vars.
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_VLM_MODEL = "llama3.2-vision"
DEFAULT_TIMEOUT_SECONDS = 120.0

# A *transcribe* prompt that is deliberately narrow — we ask the VLM to act
# as an OCR engine, not as an analyst. Keeps generated text close to the
# page so the hallucinated-words rate reported in §5.2 stays meaningful.
_TRANSCRIBE_PROMPT = (
    "You are an OCR engine. Transcribe ALL legible text from this page "
    "verbatim. Preserve original line breaks and section headings. Do NOT "
    "summarise, translate, or add commentary. Return only the transcribed "
    "text."
)


@runtime_checkable
class VlmClient(Protocol):
    """Protocol for any client that can transcribe a page image.

    Implementations MUST return a tuple of ``(text, confidence)`` where
    ``confidence`` is in ``[0.0, 1.0]``. A client that has no signal for
    confidence should return ``0.5`` so downstream callers route the page
    for human review rather than silently trusting it.
    """

    def transcribe(self, image_bytes: bytes) -> tuple[str, float]: ...


# ---------------------------------------------------------------------------
# Concrete: Ollama multimodal /api/generate.
# ---------------------------------------------------------------------------


@dataclass
class OllamaVlmClient:
    """Talks to a local Ollama server's ``/api/generate`` multimodal route.

    Confidence is heuristic — Ollama does not return token logprobs over
    the HTTP API in every release, so we approximate confidence by the
    response's ``done_reason`` (a clean stop = higher) and the absence of
    a truncation marker. Conservative by design.
    """

    host: str = DEFAULT_OLLAMA_HOST
    model: str = DEFAULT_VLM_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    prompt: str = _TRANSCRIBE_PROMPT
    _client: httpx.Client | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> OllamaVlmClient:
        """Build a client from ``OLLAMA_HOST`` + ``OLLAMA_VLM_MODEL`` env."""
        host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        model = os.environ.get("OLLAMA_VLM_MODEL", DEFAULT_VLM_MODEL)
        return cls(host=host, model=model)

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.host, timeout=self.timeout_seconds)
        return self._client

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=8.0),
    )
    def transcribe(self, image_bytes: bytes) -> tuple[str, float]:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "images": [encoded],
            "stream": False,
            "options": {"temperature": 0.0},
        }
        resp = self._http().post("/api/generate", json=payload)
        resp.raise_for_status()
        body = resp.json()
        text = str(body.get("response", "")).strip()
        confidence = _confidence_from_ollama_body(body, text)
        return text, confidence


def _confidence_from_ollama_body(body: dict[str, object], text: str) -> float:
    """Map an Ollama generate response to a conservative ``[0, 1]`` score.

    Heuristic: a clean stop (``done_reason == "stop"``) with non-empty
    text → 0.78 (above default OCR threshold). A length / truncation
    finish → 0.55. Empty or malformed → 0.0.
    """
    if not text:
        return 0.0
    done_reason = str(body.get("done_reason", "")).lower()
    if done_reason in {"length", "max_tokens"}:
        return 0.55
    if done_reason in {"", "stop"} or bool(body.get("done", False)):
        return 0.78
    return 0.5


# ---------------------------------------------------------------------------
# Test double: emits a canned transcription. Use in unit tests / CI.
# ---------------------------------------------------------------------------


@dataclass
class FakeVlmClient:
    """Deterministic VLM stand-in for tests.

    Returns ``canned_text`` (or the next item from a per-page list) plus a
    fixed ``confidence`` on every call. Tracks how many pages it has been
    asked about via ``calls`` so tests can assert page count.
    """

    canned_text: str | list[str] = "Section 1. Sample transcription."
    confidence: float = 0.85
    calls: int = 0

    def transcribe(self, image_bytes: bytes) -> tuple[str, float]:
        del image_bytes  # the fake ignores pixels by design
        if isinstance(self.canned_text, list):
            idx = min(self.calls, len(self.canned_text) - 1)
            text = self.canned_text[idx]
        else:
            text = self.canned_text
        self.calls += 1
        return text, self.confidence


__all__ = [
    "DEFAULT_OLLAMA_HOST",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_VLM_MODEL",
    "FakeVlmClient",
    "OllamaVlmClient",
    "VlmClient",
]
