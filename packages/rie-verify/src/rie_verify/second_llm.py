"""Second-LLM verifier for gate 3.

Per §6.5 the entailment gate runs both an NLI cross-encoder AND a second LLM
*of a different model family* than the classifier. Disagreement between the
two never auto-resolves — the claim is flagged for human review.

The Protocol exposes a single boolean question:
    "Is the hypothesis directly supported by the premise?"

The production `OllamaSecondLlm` posts to a local Ollama `/api/generate`
endpoint with a structured Yes/No prompt; the fake returns a preconfigured
verdict for tests.
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

import httpx


@runtime_checkable
class SecondLlmBackend(Protocol):
    """A boolean entailment judge — *yes* iff premise directly supports hypothesis."""

    def judge_entailment(self, premise: str, hypothesis: str) -> bool: ...


_PROMPT_TEMPLATE: Final[str] = """\
You are a strict legal-text entailment verifier. Answer ONLY with a single
word — "Yes" or "No" — and nothing else.

Question: is the HYPOTHESIS directly supported by the PREMISE? Consider only
surface-level, explicit support; do not infer beyond the text.

PREMISE:
{premise}

HYPOTHESIS:
{hypothesis}

Answer (Yes or No):"""


class OllamaSecondLlm:
    """Second-LLM entailment judge backed by a local Ollama server.

    Per §6.5 / implementation.md §13, the second LLM must be a different model
    family than the primary classifier. The orchestrator picks the model name;
    this adapter posts to `{base_url}/api/generate` with `stream=False` and
    returns a strict Yes/No verdict.

    Network failure → `RuntimeError`; the verifier converts any exception into
    a model-gate failure (handled by `VerificationService.run_gate`).
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 30.0,
    ) -> None:
        self._model: str = model
        self._base_url: str = base_url.rstrip("/")
        self._timeout_s: float = timeout_s

    def judge_entailment(self, premise: str, hypothesis: str) -> bool:
        prompt = _PROMPT_TEMPLATE.format(premise=premise, hypothesis=hypothesis)
        url = f"{self._base_url}/api/generate"
        payload: dict[str, object] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            # Constrain to a single, deterministic token; the model still emits
            # the full word but temperature=0 keeps Yes/No stable.
            "options": {"temperature": 0.0, "num_predict": 4},
        }
        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            msg = f"OllamaSecondLlm request failed: {exc}"
            raise RuntimeError(msg) from exc

        text = str(body.get("response", "")).strip().lower()
        # Honest parse — anything other than an unambiguous "yes" is a "no".
        first_token = text.split()[0] if text else ""
        return first_token.startswith("yes")


class FakeSecondLlm:
    """Deterministic second-LLM backend for tests — always returns `verdict`."""

    def __init__(self, verdict: bool) -> None:
        self._verdict: bool = verdict

    def judge_entailment(self, premise: str, hypothesis: str) -> bool:
        return self._verdict
