"""
tests/test_pipeline.py

Unit tests for app/pipeline/rag_pipeline.py.

We deliberately do NOT call pipeline.initialize() here - that method
loads a real embedding model, opens a real Chroma store, and instantiates
a real LLM client, which makes it an integration concern (network/model
downloads), not a fast unit test. Instead we construct a RAGPipeline and
inject fake retriever/llm objects directly, so we can test the
orchestration LOGIC (short-circuit on empty retrieval, timing fields
populated, exception wrapping) in isolation.
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.config import NO_ANSWER_MESSAGE
from app.llms.base import LLMGenerationError
from app.pipeline.rag_pipeline import PipelineError, RAGPipeline


class _FakeRetriever:
    def __init__(self, documents):
        self._documents = documents

    def invoke(self, question: str):
        return self._documents


class _FakeLLM:
    def __init__(self, response: str = "Fake grounded answer.", should_fail: bool = False):
        self._response = response
        self._should_fail = should_fail

    def generate(self, prompt: str) -> str:
        if self._should_fail:
            raise LLMGenerationError("Simulated provider failure")
        return self._response


def _make_pipeline(retriever, llm) -> RAGPipeline:
    pipeline = RAGPipeline()
    pipeline._retriever = retriever  # noqa: SLF001 - intentional test injection
    pipeline.llm = llm
    return pipeline


def test_query_short_circuits_on_empty_retrieval() -> None:
    pipeline = _make_pipeline(_FakeRetriever([]), _FakeLLM())

    result = pipeline.query("What is the medicine stock level?")

    assert result.was_grounded is False
    assert result.answer == NO_ANSWER_MESSAGE
    assert result.sources == []
    assert result.generation_seconds == 0.0


def test_query_returns_grounded_answer_with_sources() -> None:
    docs = [Document(page_content="Stock is 500 units.", metadata={"source_filename": "a.pdf", "page": 1})]
    pipeline = _make_pipeline(_FakeRetriever(docs), _FakeLLM(response="Stock is 500 units."))

    result = pipeline.query("What is the medicine stock level?")

    assert result.was_grounded is True
    assert result.answer == "Stock is 500 units."
    assert result.sources == docs
    assert result.total_seconds >= 0.0


def test_query_wraps_llm_failure_as_pipeline_error() -> None:
    docs = [Document(page_content="Some content.", metadata={"source_filename": "a.pdf", "page": 1})]
    pipeline = _make_pipeline(_FakeRetriever(docs), _FakeLLM(should_fail=True))

    try:
        pipeline.query("Any question")
        assert False, "expected PipelineError to be raised"
    except PipelineError as exc:
        assert "Answer generation failed" in str(exc)


def test_query_raises_if_not_initialized() -> None:
    pipeline = RAGPipeline()  # _retriever is None
    try:
        pipeline.query("question")
        assert False, "expected PipelineError"
    except PipelineError as exc:
        assert "initialize()" in str(exc)