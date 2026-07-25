from __future__ import annotations

from app.llms.groq import GroqLLM
from app.llms.llm_factory import get_llm


def test_get_llm_accepts_grok_alias(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    llm = get_llm("grok")

    assert isinstance(llm, GroqLLM)


def test_get_llm_is_case_insensitive(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    llm = get_llm("GROK")

    assert isinstance(llm, GroqLLM)
