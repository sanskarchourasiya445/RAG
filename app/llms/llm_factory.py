"""
app/llms/llm_factory.py

Responsibility: the ONLY place in the codebase that maps a configured
provider name to a concrete BaseLLM implementation.

This is what makes "switch LLM provider by editing one value in
app/config.py, zero code changes elsewhere" true: main.py / rag_pipeline.py
call get_llm() and receive a BaseLLM - they never import
MistralLLM/GeminiLLM/GrokLLM directly.

Adding a 4th provider later = one new app/llms/<provider>.py file + one
new branch here. No other file in the repository changes.
"""

from __future__ import annotations

import logging

from app.config import LLM_PROVIDER
from app.llms.base import BaseLLM, LLMGenerationError

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("mistral", "gemini", "groq")


def get_llm(provider: str | None = None) -> BaseLLM:
    """Instantiate the configured LLM provider.

    Args:
        provider: override the configured provider (mainly useful for
        tests). Defaults to app.config.LLM_PROVIDER.
    """
    selected = (provider or LLM_PROVIDER).lower()
    logger.info("Initializing LLM provider: %s", selected)

    if selected == "mistral":
        from app.llms.mistral import MistralLLM

        return MistralLLM()

    if selected == "gemini":
        from app.llms.gemini import GeminiLLM

        return GeminiLLM()

    if selected == "groq":
        from app.llms.groq import GrokLLM

        return GrokLLM()

    raise LLMGenerationError(
        f"Unsupported LLM provider: '{selected}'. Must be one of {SUPPORTED_PROVIDERS} "
        f"(set LLM_PROVIDER in app/config.py)."
    )