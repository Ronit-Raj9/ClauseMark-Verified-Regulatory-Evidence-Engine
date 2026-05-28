"""Constrained-decoding classification adapter.

Public surface:
- ``ClassificationService`` — implements ``ClassifierPort``.
- ``build_output_model`` — dynamic Pydantic model with `Literal` enum for
  the indicator id (the anti-hallucination keystone).
- ``LlmClient`` Protocol + ``OllamaClient`` / ``VllmClient`` concrete adapters.
- ``DeterministicFakeLlm`` / ``ScriptedFakeLlm`` — offline fakes for tests.
- ``build_classification_prompt`` — pure prompt builder.
"""

from .fake_llm import DeterministicFakeLlm, ScriptedFakeLlm
from .llm_client import LlmClient, OllamaClient, VllmClient
from .prompt import SYSTEM_INSTRUCTION, build_classification_prompt
from .regime import (
    RegimeAssembler,
    RegimeAssemblyInput,
    build_regime_assembly_input,
    materialise_regime_from_llm_ids,
    merge_llm_regime_hints,
)
from .schema import ClassificationOutputBase, build_output_model
from .service import ClassificationService

__all__ = [
    "SYSTEM_INSTRUCTION",
    "ClassificationOutputBase",
    "ClassificationService",
    "DeterministicFakeLlm",
    "LlmClient",
    "OllamaClient",
    "RegimeAssembler",
    "RegimeAssemblyInput",
    "ScriptedFakeLlm",
    "VllmClient",
    "build_classification_prompt",
    "build_output_model",
    "build_regime_assembly_input",
    "invoke_regime_assembler",
    "materialise_regime_from_llm_ids",
    "merge_llm_regime_hints",
]
