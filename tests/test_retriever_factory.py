"""
tests/test_retriever_factory.py

Unit tests for app/retrievers/factory.py.

Uses a fake Chroma-like vector store (same fake as test_vector_retriever.py
needs) and a fake BaseLLM, so these tests exercise ONLY the factory's
type-selection logic - no real vector store or LLM.
"""

from __future__ import annotations

import pytest

from app.llms.base import BaseLLM
from app.retrievers.bm25 import BM25Retriever
from app.retrievers.factory import RetrieverFactory, RetrieverFactoryError
from app.retrievers.hybrid import HybridRetriever
from app.retrievers.multi_query import MultiQueryRetriever
from app.retrievers.parent import ParentDocumentRetriever
from app.retrievers.vector import VectorRetriever
from app.vectorstores.parent_store import ParentDocumentStore


class _FakeVectorStore:
    def as_retriever(self, **kwargs):
        return _FakeLangChainRetriever()

    def get(self, include=None):
        # Minimal stand-in for Chroma's collection.get() - enough of a
        # corpus for BM25Retriever.from_documents() to build a real index.
        return {
            "documents": [
                "Python is a popular programming language.",
                "FastAPI is a modern web framework.",
            ],
            "metadatas": [
                {"source": "a.pdf", "page": 0},
                {"source": "a.pdf", "page": 1},
            ],
        }


class _FakeLangChainRetriever:
    def invoke(self, query: str):
        return []


class _FakeLLM(BaseLLM):
    provider_name = "fake"

    def generate(self, prompt: str) -> str:
        return ""


def test_factory_builds_vector_retriever_by_default() -> None:
    retriever = RetrieverFactory.create(_FakeVectorStore(), retriever_type="vector")

    assert isinstance(retriever, VectorRetriever)


def test_factory_builds_multi_query_retriever_when_llm_provided() -> None:
    retriever = RetrieverFactory.create(
        _FakeVectorStore(), llm=_FakeLLM(), retriever_type="multi_query"
    )

    assert isinstance(retriever, MultiQueryRetriever)


def test_factory_raises_for_multi_query_without_llm() -> None:
    with pytest.raises(RetrieverFactoryError):
        RetrieverFactory.create(_FakeVectorStore(), llm=None, retriever_type="multi_query")


def test_factory_raises_for_unsupported_retriever_type() -> None:
    with pytest.raises(RetrieverFactoryError):
        RetrieverFactory.create(_FakeVectorStore(), retriever_type="not_a_real_type")


def test_factory_is_case_insensitive() -> None:
    retriever = RetrieverFactory.create(_FakeVectorStore(), retriever_type="VECTOR")

    assert isinstance(retriever, VectorRetriever)


def test_factory_builds_bm25_retriever() -> None:
    retriever = RetrieverFactory.create(_FakeVectorStore(), retriever_type="bm25")

    assert isinstance(retriever, BM25Retriever)


def test_factory_builds_hybrid_retriever() -> None:
    retriever = RetrieverFactory.create(_FakeVectorStore(), retriever_type="hybrid")

    assert isinstance(retriever, HybridRetriever)


def test_factory_builds_parent_retriever_when_store_provided(tmp_path) -> None:
    parent_store = ParentDocumentStore(tmp_path)
    retriever = RetrieverFactory.create(
        _FakeVectorStore(), retriever_type="parent", parent_store=parent_store
    )

    assert isinstance(retriever, ParentDocumentRetriever)


def test_factory_raises_for_parent_without_parent_store() -> None:
    with pytest.raises(RetrieverFactoryError):
        RetrieverFactory.create(_FakeVectorStore(), retriever_type="parent", parent_store=None)


def test_factory_wraps_retriever_in_reranker_when_enabled(monkeypatch) -> None:
    import app.retrievers.factory as factory_module
    import app.retrievers.reranker as reranker_module
    from app.retrievers.reranker import RerankerRetriever

    monkeypatch.setattr(factory_module, "ENABLE_RERANKER", True)
    monkeypatch.setattr(
        reranker_module, "CrossEncoder", lambda model_name: _FakeCrossEncoderStub()
    )

    retriever = RetrieverFactory.create(_FakeVectorStore(), retriever_type="vector")

    assert isinstance(retriever, RerankerRetriever)


def test_factory_wraps_retriever_in_compression_when_enabled(monkeypatch) -> None:
    import app.retrievers.factory as factory_module
    from app.retrievers.compression import CompressionRetriever

    monkeypatch.setattr(factory_module, "ENABLE_COMPRESSION", True)

    retriever = RetrieverFactory.create(
        _FakeVectorStore(), llm=_FakeLLM(), retriever_type="vector"
    )

    assert isinstance(retriever, CompressionRetriever)


def test_factory_raises_for_compression_without_llm(monkeypatch) -> None:
    import app.retrievers.factory as factory_module

    monkeypatch.setattr(factory_module, "ENABLE_COMPRESSION", True)

    with pytest.raises(RetrieverFactoryError):
        RetrieverFactory.create(_FakeVectorStore(), llm=None, retriever_type="vector")


def test_factory_applies_reranker_before_compression(monkeypatch) -> None:
    # Compression should wrap the (possibly reranked) retriever, not the
    # other way around - i.e. CompressionRetriever's base_retriever should
    # be the RerankerRetriever, confirming stage ORDER: retriever -> rerank -> compress.
    import app.retrievers.factory as factory_module
    import app.retrievers.reranker as reranker_module
    from app.retrievers.compression import CompressionRetriever
    from app.retrievers.reranker import RerankerRetriever

    monkeypatch.setattr(factory_module, "ENABLE_RERANKER", True)
    monkeypatch.setattr(factory_module, "ENABLE_COMPRESSION", True)
    monkeypatch.setattr(
        reranker_module, "CrossEncoder", lambda model_name: _FakeCrossEncoderStub()
    )

    retriever = RetrieverFactory.create(
        _FakeVectorStore(), llm=_FakeLLM(), retriever_type="vector"
    )

    assert isinstance(retriever, CompressionRetriever)
    assert isinstance(retriever._base_retriever, RerankerRetriever)


class _FakeCrossEncoderStub:
    def predict(self, pairs):
        return [0.0] * len(pairs)