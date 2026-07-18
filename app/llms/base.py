"""
app/llms/base.py

Responsibility: define the contract every LLM provider must implement.
The RAG pipeline depends ONLY on this abstract interface - it never knows
or cares whether the concrete implementation is Mistral, Gemini, or Grok.

This is the abstraction that makes "switch providers via .env, zero code
changes" actually true architecturally (not just at the config level).
Without this, you'd end up with `if provider == "gemini": ...` branches
scattered through the pipeline - the opposite of clean architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMGenerationError(Exception):
    """Raised when an LLM provider fails to generate a response
    (network failure, invalid API key, rate limit, malformed response)."""


class BaseLLM(ABC):
    """Abstract interface for a chat/completion LLM provider.

    Deliberately minimal: a single `generate` method taking a fully-formed
    prompt string and returning a plain string response. Provider-specific
    concerns (message formatting, safety settings, token limits) are
    handled INSIDE each concrete implementation, not leaked into this
    interface or into the pipeline that calls it.
    """

    provider_name: str = "base"

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text response for the given prompt.

        Raises:
            LLMGenerationError: on any failure (network, auth, malformed
            response). Concrete implementations should catch provider-
            specific exceptions and re-raise as LLMGenerationError so the
            pipeline has one exception type to handle, regardless of
            which provider is configured.
        """
        raise NotImplementedError
    