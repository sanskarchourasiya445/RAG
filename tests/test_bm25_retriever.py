"""
tests/test_bm25_retriever.py

Unit tests for app/retrievers/bm25.py. Uses a small real in-memory
corpus (BM25 needs actual token statistics, so a fake retriever wouldn't
exercise anything meaningful here) - no vector store or LLM involved.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.retrievers.base import BaseRetriever
from app.retrievers.bm25 import BM25Retriever, BM25RetrieverError

_CORPUS = [
    Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "a.pdf", "page": 0}),
    Document(page_content="Python is a popular programming language for data science.", metadata={"source": "a.pdf", "page": 1}),
    Document(page_content="FastAPI is a modern web framework for building APIs with Python.", metadata={"source": "b.pdf", "page": 0}),
]


def test_bm25_retriever_raises_on_empty_corpus() -> None:
    with pytest.raises(BM25RetrieverError):
        BM25Retriever(documents=[], top_k=3)


def test_bm25_retriever_returns_keyword_relevant_documents() -> None:
    retriever = BM25Retriever(documents=_CORPUS, top_k=2)

    results = retriever.retrieve("Python web framework")

    assert len(results) <= 2
    assert any("FastAPI" in doc.page_content for doc in results)


def test_bm25_retriever_respects_top_k() -> None:
    retriever = BM25Retriever(documents=_CORPUS, top_k=1)

    results = retriever.retrieve("Python")

    assert len(results) == 1


def test_bm25_retriever_implements_base_retriever_contract() -> None:
    retriever = BM25Retriever(documents=_CORPUS, top_k=3)

    assert isinstance(retriever, BaseRetriever)