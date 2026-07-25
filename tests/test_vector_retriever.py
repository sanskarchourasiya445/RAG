"""
tests/test_vector_retriever.py

Unit tests for app/retrievers/vector.py.

Direct successor to the old tests/test_retriever.py - same assertions,
adapted from the free-function API (build_retriever) to the class-based
Strategy Pattern API (VectorRetriever.retrieve). We still don't spin up a
real Chroma + embedding model here (that belongs in an integration test)
- a lightweight fake vector store records how `as_retriever` was called,
so we can assert on the MMR configuration without any network/model
dependency.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.retrievers.vector import RetrieverConfigError, VectorRetriever


class _FakeLangChainRetriever:
    """Stand-in for the object LangChain's `as_retriever()` returns."""

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self.last_query: str | None = None

    def invoke(self, query: str) -> list[Document]:
        self.last_query = query
        return self._documents


class _FakeVectorStore:
    """Minimal stand-in for a Chroma vector store - just records the
    kwargs passed to as_retriever() so tests can assert on them."""

    def __init__(self, documents: list[Document] | None = None) -> None:
        self.last_call_kwargs: dict = {}
        self._documents = documents or []

    def as_retriever(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeLangChainRetriever(self._documents)


def test_vector_retriever_uses_mmr_search_type() -> None:
    store = _FakeVectorStore()
    VectorRetriever(store, top_k=4, fetch_k=20, lambda_mult=0.5)

    assert store.last_call_kwargs["search_type"] == "mmr"


def test_vector_retriever_passes_through_configured_values() -> None:
    store = _FakeVectorStore()
    VectorRetriever(store, top_k=6, fetch_k=30, lambda_mult=0.7)

    search_kwargs = store.last_call_kwargs["search_kwargs"]
    assert search_kwargs["k"] == 6
    assert search_kwargs["fetch_k"] == 30
    assert search_kwargs["lambda_mult"] == 0.7


def test_vector_retriever_rejects_fetch_k_smaller_than_top_k() -> None:
    store = _FakeVectorStore()
    with pytest.raises(RetrieverConfigError):
        VectorRetriever(store, top_k=10, fetch_k=5, lambda_mult=0.5)


def test_vector_retriever_allows_fetch_k_equal_to_top_k() -> None:
    # Not an error, just degenerates toward plain similarity search -
    # valid configuration, even if not the most useful one.
    store = _FakeVectorStore()
    VectorRetriever(store, top_k=5, fetch_k=5, lambda_mult=0.5)
    assert store.last_call_kwargs["search_kwargs"]["k"] == 5


def test_vector_retriever_retrieve_delegates_to_langchain_retriever() -> None:
    documents = [Document(page_content="chunk one", metadata={"source": "a.pdf", "page": 0})]
    store = _FakeVectorStore(documents=documents)
    retriever = VectorRetriever(store)

    result = retriever.retrieve("what is X?")

    assert result == documents


def test_vector_retriever_implements_base_retriever_contract() -> None:
    from app.retrievers.base import BaseRetriever

    store = _FakeVectorStore()
    retriever = VectorRetriever(store)

    assert isinstance(retriever, BaseRetriever)