"""
tests/test_reranker.py

Unit tests for app/retrievers/reranker.py.

The real cross-encoder model is a real (multi-hundred-MB) network
download and a real neural net forward pass - neither belongs in a fast
unit test. Every test here monkeypatches `CrossEncoder` in
app.retrievers.reranker with a small fake that returns scores the test
controls directly, so these tests exercise ONLY RerankerRetriever's own
logic (candidate scoring/sorting/truncation, fallback on failure) - never
a real model.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

import app.retrievers.reranker as reranker_module
from app.retrievers.base import BaseRetriever
from app.retrievers.reranker import RerankerError, RerankerRetriever


class _FakeBaseRetriever(BaseRetriever):
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def retrieve(self, query: str) -> list[Document]:
        return self._documents


class _FakeCrossEncoder:
    """Scores each (query, doc_text) pair using a score the test injects,
    keyed by the document's position in the input list."""

    def __init__(self, model_name: str, scores: list[float] | None = None, raises: bool = False) -> None:
        self.model_name = model_name
        self._scores = scores or []
        self._raises = raises

    def predict(self, pairs):
        if self._raises:
            raise RuntimeError("fake scoring failure")
        return self._scores


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "a.pdf", "page": 0})


def test_reranker_returns_documents_sorted_by_score_descending(monkeypatch) -> None:
    documents = [_doc("low relevance"), _doc("high relevance"), _doc("medium relevance")]
    monkeypatch.setattr(
        reranker_module,
        "CrossEncoder",
        lambda model_name: _FakeCrossEncoder(model_name, scores=[0.1, 0.9, 0.5]),
    )
    retriever = RerankerRetriever(
        base_retriever=_FakeBaseRetriever(documents), model_name="fake-model", final_k=3
    )

    result = retriever.retrieve("query")

    assert [doc.page_content for doc in result] == [
        "high relevance",
        "medium relevance",
        "low relevance",
    ]


def test_reranker_truncates_to_final_k(monkeypatch) -> None:
    documents = [_doc(f"doc {i}") for i in range(5)]
    monkeypatch.setattr(
        reranker_module,
        "CrossEncoder",
        lambda model_name: _FakeCrossEncoder(model_name, scores=[0.1, 0.2, 0.3, 0.4, 0.5]),
    )
    retriever = RerankerRetriever(
        base_retriever=_FakeBaseRetriever(documents), model_name="fake-model", final_k=2
    )

    result = retriever.retrieve("query")

    assert len(result) == 2
    # Highest two scores were docs 4 and 3.
    assert [doc.page_content for doc in result] == ["doc 4", "doc 3"]


def test_reranker_returns_empty_list_when_no_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        reranker_module,
        "CrossEncoder",
        lambda model_name: _FakeCrossEncoder(model_name, scores=[]),
    )
    retriever = RerankerRetriever(
        base_retriever=_FakeBaseRetriever([]), model_name="fake-model", final_k=3
    )

    result = retriever.retrieve("query")

    assert result == []


def test_reranker_falls_back_to_original_order_on_scoring_failure(monkeypatch) -> None:
    documents = [_doc("first"), _doc("second"), _doc("third")]
    monkeypatch.setattr(
        reranker_module,
        "CrossEncoder",
        lambda model_name: _FakeCrossEncoder(model_name, raises=True),
    )
    retriever = RerankerRetriever(
        base_retriever=_FakeBaseRetriever(documents), model_name="fake-model", final_k=2
    )

    result = retriever.retrieve("query")

    # Falls back to the base retriever's original order, truncated to final_k.
    assert [doc.page_content for doc in result] == ["first", "second"]


def test_reranker_raises_rerankererror_when_model_fails_to_load(monkeypatch) -> None:
    def _raise_on_construct(model_name):
        raise RuntimeError("could not load model")

    monkeypatch.setattr(reranker_module, "CrossEncoder", _raise_on_construct)

    with pytest.raises(RerankerError):
        RerankerRetriever(
            base_retriever=_FakeBaseRetriever([]), model_name="broken-model", final_k=3
        )


def test_reranker_implements_base_retriever_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        reranker_module,
        "CrossEncoder",
        lambda model_name: _FakeCrossEncoder(model_name, scores=[]),
    )
    retriever = RerankerRetriever(
        base_retriever=_FakeBaseRetriever([]), model_name="fake-model", final_k=3
    )

    assert isinstance(retriever, BaseRetriever)