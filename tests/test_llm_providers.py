"""
tests/test_llm_providers.py

Configuration and environment-variable contract tests for all LLM providers.

Goals:
  - Verify each provider reads the correct canonical environment variable.
  - Verify each provider raises a clear, actionable error when the key is missing.
  - Verify the actual API-key value never leaks into error messages or logs.

These tests make NO real API calls. All keys are fake sentinel values.
"""

from __future__ import annotations

import pytest

from app.llms.base import LLMGenerationError

# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class TestGeminiProviderConfig:
    """Gemini provider reads GEMINI_API_KEY (not GOOGLE_API_KEY)."""

    def test_gemini_accepts_canonical_env_var(self, monkeypatch) -> None:
        """GeminiLLM.__init__ must not raise when GEMINI_API_KEY is set.

        We patch google.genai.Client so no real SDK call is made — we only
        test the configuration path.
        """
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-for-testing-only")

        import unittest.mock as mock

        with mock.patch("app.llms.gemini.genai.Client"):
            from app.llms.gemini import GeminiLLM
            # Should not raise - key is present
            llm = GeminiLLM()
            assert llm is not None

    def test_gemini_raises_when_key_missing(self, monkeypatch) -> None:
        """GeminiLLM must raise LLMGenerationError when GEMINI_API_KEY is absent."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        from app.llms.gemini import GeminiLLM

        with pytest.raises(LLMGenerationError) as exc_info:
            GeminiLLM()

        assert "GEMINI_API_KEY" in str(exc_info.value)

    def test_gemini_error_names_correct_variable(self, monkeypatch) -> None:
        """Error message must name GEMINI_API_KEY so the developer knows what to set."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        from app.llms.gemini import GeminiLLM

        with pytest.raises(LLMGenerationError) as exc_info:
            GeminiLLM()

        error_text = str(exc_info.value)
        assert "GEMINI_API_KEY" in error_text
        # Must NOT suggest the wrong variable name
        assert "GOOGLE_API_KEY" not in error_text

    def test_gemini_does_not_leak_key_in_error(self, monkeypatch) -> None:
        """The actual API-key value must never appear in error messages."""
        secret = "super-secret-gemini-key-xyz"
        monkeypatch.setenv("GEMINI_API_KEY", secret)

        import unittest.mock as mock

        # Client instantiation succeeds; generate_content fails at call time.
        mock_client = mock.MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("SDK failure")

        with mock.patch("app.llms.gemini.genai.Client", return_value=mock_client):
            from app.llms.gemini import GeminiLLM

            llm = GeminiLLM()
            with pytest.raises(Exception) as exc_info:
                llm.generate("test prompt")

        assert secret not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

class TestGroqProviderConfig:
    """Groq provider reads GROQ_API_KEY."""

    def test_groq_raises_when_key_missing(self, monkeypatch) -> None:
        """GroqLLM must raise LLMGenerationError when GROQ_API_KEY is absent."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        from app.llms.groq import GroqLLM

        with pytest.raises(LLMGenerationError) as exc_info:
            GroqLLM()

        assert "GROQ_API_KEY" in str(exc_info.value)

    def test_groq_accepts_canonical_env_var(self, monkeypatch) -> None:
        """GroqLLM must not raise when GROQ_API_KEY is set."""
        monkeypatch.setenv("GROQ_API_KEY", "test-groq-key-for-testing-only")

        import unittest.mock as mock
        from openai import OpenAI

        with mock.patch("app.llms.groq.OpenAI"):
            from app.llms.groq import GroqLLM
            llm = GroqLLM()
            assert llm is not None

    def test_groq_does_not_leak_key_in_error(self, monkeypatch) -> None:
        """GROQ_API_KEY value must never appear in error messages."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        from app.llms.groq import GroqLLM

        with pytest.raises(LLMGenerationError) as exc_info:
            GroqLLM()

        assert "test-groq-key-for-testing-only" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------

class TestMistralProviderConfig:
    """Mistral provider reads MISTRAL_API_KEY."""

    def test_mistral_raises_when_key_missing(self, monkeypatch) -> None:
        """MistralLLM must raise LLMGenerationError when MISTRAL_API_KEY is absent."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        from app.llms.mistral import MistralLLM

        with pytest.raises(LLMGenerationError) as exc_info:
            MistralLLM()

        assert "MISTRAL_API_KEY" in str(exc_info.value)

    def test_mistral_accepts_canonical_env_var(self, monkeypatch) -> None:
        """MistralLLM must not raise when MISTRAL_API_KEY is set."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key-for-testing-only")

        import unittest.mock as mock

        with mock.patch("app.llms.mistral.Mistral"):
            from app.llms.mistral import MistralLLM
            llm = MistralLLM()
            assert llm is not None

    def test_mistral_does_not_leak_key_in_error(self, monkeypatch) -> None:
        """MISTRAL_API_KEY value must never appear in error messages."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        from app.llms.mistral import MistralLLM

        with pytest.raises(LLMGenerationError) as exc_info:
            MistralLLM()

        assert "test-mistral-key-for-testing-only" not in str(exc_info.value)
