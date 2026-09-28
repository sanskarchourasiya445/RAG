"""
app/llms/llm_factory.py

Responsibility: the ONLY place in the codebase that maps a configured
provider name to a concrete BaseLLM implementation.

This is what makes "switch LLM provider by editing one value in
app/config.py, zero code changes elsewhere" true: main.py / rag_pipeline.py
call get_llm() and receive a BaseLLM - they never import
MistralLLM/GeminiLLM/GroqLLM directly.

Adding a 4th provider later = one new app/llms/<provider>.py file + one
new branch here. No other file in the repository changes.

Canonical provider names
------------------------
  "mistral"   → MistralLLM
  "gemini"    → GeminiLLM
  "groq"      → GroqLLM   (also accepted as "grok" for backwards compatibility)

All names are case-insensitive ("GROQ" == "groq").
"""

from __future__ import annotations

import logging

from app.config import LLM_PROVIDER
from app.llms.base import BaseLLM, LLMGenerationError

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("mistral", "gemini", "groq")

# Backwards-compatibility aliases: old name → canonical name.
# "grok" was an early typo for the Groq provider that found its way into
# tests and documentation. Keeping it as a silent alias avoids breaking
# existing .env files or muscle memory without widening the public API.
_ALIASES: dict[str, str] = {
    "grok": "groq",
}


def get_llm(provider: str | None = None) -> BaseLLM:
    """Instantiate the configured LLM provider.

    Args:
        provider: override the configured provider (mainly useful for
        tests). Defaults to app.config.LLM_PROVIDER.
    """
    raw = (provider or LLM_PROVIDER).lower()
    selected = _ALIASES.get(raw, raw)
    logger.info("Initializing LLM provider: %s", selected)

    if selected == "mistral":
        from app.llms.mistral import MistralLLM

        return MistralLLM()

    if selected == "gemini":
        from app.llms.gemini import GeminiLLM

        return GeminiLLM()

    if selected == "groq":
        from app.llms.groq import GroqLLM

        return GroqLLM()

    raise LLMGenerationError(
        f"Unsupported LLM provider: '{raw}'. Must be one of {SUPPORTED_PROVIDERS} "
        f"(set LLM_PROVIDER in app/config.py)."
    )