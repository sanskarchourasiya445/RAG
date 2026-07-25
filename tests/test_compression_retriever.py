"""
tests/test_compression_retriever.py

Unit tests for app/retrievers/compression.py.

Uses fakes for both BaseLLM and BaseRetriever, so these tests exercise
ONLY the compression/drop/fallback logic - no real LLM call.
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.config import NO_RELEVANT_CONTENT_MESSAGE
from app.llms.base import BaseLLM, LLMGenerationError
from app.retrievers.base import BaseRetriever
from app.retrievers.compression import CompressionRetriever


class _FakeLLM(BaseLLM):
    """Returns a response per-call, taken in order from `responses`, or
    raises if `raises` is True (in which case every call fails)."""

    provider_name = "fake"

    def __init__(self, responses: list[str] | None = None, raises: bool = False) -> None:
        self._responses = list(responses or [])
        self._raises = raises
        self.prompts_seen: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts_seen.append(prompt)
        if self._raises:
            raise LLMGenerationError("fake LLM failure")
        return self._responses.pop(0)


class _FakeBaseRetriever(BaseRetriever):
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def retrieve(self, query: str) -> list[Document]:
        return self._documents


def _doc(text: str, page: int = 0) -> Document:
    return Document(page_content=text, metadata={"source": "a.pdf", "page": page})


def test_compression_retriever_returns_extracted_excerpts() -> None:
    llm = _FakeLLM(responses=["relevant sentence one", "relevant sentence two"])
    base = _FakeBaseRetriever([_doc("long chunk one", page=0), _doc("long chunk two", page=1)])
    retriever = CompressionRetriever(llm=llm, base_retriever=base)

    result = retriever.retrieve("question")

    assert [doc.page_content for doc in result] == [
        "relevant sentence one",
        "relevant sentence two",
    ]


def test_compression_retriever_preserves_metadata() -> None:
    llm = _FakeLLM(responses=["excerpt"])
    base = _FakeBaseRetriever([_doc("chunk", page=3)])
    retriever = CompressionRetriever(llm=llm, base_retriever=base)

    result = retriever.retrieve("question")

    assert result[0].metadata == {"source": "a.pdf", "page": 3}


def test_compression_retriever_drops_documents_with_no_relevant_content() -> None:
    llm = _FakeLLM(responses=["relevant excerpt", NO_RELEVANT_CONTENT_MESSAGE])
    base = _FakeBaseRetriever([_doc("relevant chunk", page=0), _doc("irrelevant chunk", page=1)])
    retriever = CompressionRetriever(llm=llm, base_retriever=base)

    result = retriever.retrieve("question")

    assert len(result) == 1
    assert result[0].page_content == "relevant excerpt"


def test_compression_retriever_drops_documents_on_empty_output() -> None:
    llm = _FakeLLM(responses=["  ", "kept"])
    base = _FakeBaseRetriever([_doc("chunk a", page=0), _doc("chunk b", page=1)])
    retriever = CompressionRetriever(llm=llm, base_retriever=base)

    result = retriever.retrieve("question")

    assert len(result) == 1
    assert result[0].page_content == "kept"


def test_compression_retriever_falls_back_to_uncompressed_when_all_dropped() -> None:
    llm = _FakeLLM(responses=[NO_RELEVANT_CONTENT_MESSAGE, NO_RELEVANT_CONTENT_MESSAGE])
    original_docs = [_doc("chunk a", page=0), _doc("chunk b", page=1)]
    base = _FakeBaseRetriever(original_docs)
    retriever = CompressionRetriever(llm=llm, base_retriever=base)

    result = retriever.retrieve("question")

    assert result == original_docs


def test_compression_retriever_falls_back_when_llm_fails_for_every_document() -> None:
    llm = _FakeLLM(raises=True)
    original_docs = [_doc("chunk a", page=0)]
    base = _FakeBaseRetriever(original_docs)
    retriever = CompressionRetriever(llm=llm, base_retriever=base)

    result = retriever.retrieve("question")

    assert result == original_docs


def test_compression_retriever_returns_empty_list_when_no_candidates() -> None:
    llm = _FakeLLM(responses=[])
    retriever = CompressionRetriever(llm=llm, base_retriever=_FakeBaseRetriever([]))

    result = retriever.retrieve("question")

    assert result == []


def test_compression_retriever_implements_base_retriever_contract() -> None:
    retriever = CompressionRetriever(llm=_FakeLLM(responses=[]), base_retriever=_FakeBaseRetriever([]))

    assert isinstance(retriever, BaseRetriever)