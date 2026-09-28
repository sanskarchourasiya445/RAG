"""
tests/test_llm_factory.py

Contract tests for the LLM factory.

Rules:
  - No real API calls. All providers are instantiated with a fake key via
    monkeypatch so the __init__ key-presence check passes.
  - The factory must map each canonical name to the correct class.
  - The factory must accept "grok" as a backwards-compatibility alias for groq.
  - All name comparisons must be case-insensitive.
  - An unknown provider must raise LLMGenerationError with a clear message.
"""

from __future__ import annotations

import unittest.mock as mock

import pytest

from app.llms.base import LLMGenerationError
from app.llms.llm_factory import get_llm


# ---------------------------------------------------------------------------
# Helpers — patch the SDK clients so no real network calls are made
# ---------------------------------------------------------------------------

def _fake_groq_env(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key-groq")


def _fake_gemini_env(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-gemini")


def _fake_mistral_env(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-mistral")


# ---------------------------------------------------------------------------
# Canonical names
# ---------------------------------------------------------------------------

def test_get_llm_groq_returns_groq_llm(monkeypatch) -> None:
    """Canonical name 'groq' → GroqLLM."""
    _fake_groq_env(monkeypatch)
    from app.llms.groq import GroqLLM

    with mock.patch("app.llms.groq.OpenAI"):
        llm = get_llm("groq")

    assert isinstance(llm, GroqLLM)


def test_get_llm_gemini_returns_gemini_llm(monkeypatch) -> None:
    """Canonical name 'gemini' → GeminiLLM."""
    _fake_gemini_env(monkeypatch)
    from app.llms.gemini import GeminiLLM

    with mock.patch("app.llms.gemini.genai.Client"):
        llm = get_llm("gemini")

    assert isinstance(llm, GeminiLLM)


def test_get_llm_mistral_returns_mistral_llm(monkeypatch) -> None:
    """Canonical name 'mistral' → MistralLLM."""
    _fake_mistral_env(monkeypatch)
    from app.llms.mistral import MistralLLM

    with mock.patch("app.llms.mistral.Mistral"):
        llm = get_llm("mistral")

    assert isinstance(llm, MistralLLM)


# ---------------------------------------------------------------------------
# Backwards-compatibility alias
# ---------------------------------------------------------------------------

def test_get_llm_accepts_grok_alias(monkeypatch) -> None:
    """'grok' is a supported alias for the groq provider."""
    _fake_groq_env(monkeypatch)
    from app.llms.groq import GroqLLM

    with mock.patch("app.llms.groq.OpenAI"):
        llm = get_llm("grok")

    assert isinstance(llm, GroqLLM)


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------

def test_get_llm_is_case_insensitive_groq(monkeypatch) -> None:
    """'GROQ', 'Groq', 'groq' all resolve to GroqLLM."""
    _fake_groq_env(monkeypatch)
    from app.llms.groq import GroqLLM

    with mock.patch("app.llms.groq.OpenAI"):
        for variant in ("GROQ", "Groq", "groq"):
            assert isinstance(get_llm(variant), GroqLLM)


def test_get_llm_is_case_insensitive_grok_alias(monkeypatch) -> None:
    """'GROK' (alias) also resolves to GroqLLM."""
    _fake_groq_env(monkeypatch)
    from app.llms.groq import GroqLLM

    with mock.patch("app.llms.groq.OpenAI"):
        llm = get_llm("GROK")

    assert isinstance(llm, GroqLLM)


# ---------------------------------------------------------------------------
# Unknown provider
# ---------------------------------------------------------------------------

def test_get_llm_unknown_provider_raises_clear_error() -> None:
    """An unsupported provider name raises LLMGenerationError with the name in the message."""
    with pytest.raises(LLMGenerationError) as exc_info:
        get_llm("openai")

    error_text = str(exc_info.value)
    assert "openai" in error_text
    assert "groq" in error_text  # error lists supported providers
