"""
tests/test_hybrid_retriever.py

Unit tests for app/retrievers/hybrid.py. Uses fake BaseRetriever
implementations returning fixed, ordered document lists, so these tests
exercise ONLY the RRF fusion math - no real vector store or BM25 index.
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.retrievers.base import BaseRetriever
from app.retrievers.hybrid import HybridRetriever


class _FakeOrderedRetriever(BaseRetriever):
    """Always returns the same fixed, pre-ranked list, regardless of query."""

    def __init__(self, ordered_documents: list[Document]) -> None:
        self._ordered_documents = ordered_documents

    def retrieve(self, query: str) -> list[Document]:
        return self._ordered_documents


def _doc(text: str, page: int) -> Document:
    return Document(page_content=text, metadata={"source": "a.pdf", "page": page})


def test_hybrid_retriever_ranks_document_found_by_both_lists_highest() -> None:
    shared = _doc("shared chunk", page=0)
    vector_only = _doc("vector only", page=1)
    bm25_only = _doc("bm25 only", page=2)

    vector = _FakeOrderedRetriever([shared, vector_only])
    bm25 = _FakeOrderedRetriever([shared, bm25_only])

    retriever = HybridRetriever(
        vector_retriever=vector,
        bm25_retriever=bm25,
        vector_weight=0.5,
        bm25_weight=0.5,
        top_k=3,
    )

    results = retriever.retrieve("query")

    # shared appears in both lists -> its fused RRF score is the sum of
    # two contributions, so it must rank above anything found in only one list.
    assert results[0].page_content == "shared chunk"


def test_hybrid_retriever_respects_top_k() -> None:
    vector = _FakeOrderedRetriever([_doc("v1", 1), _doc("v2", 2), _doc("v3", 3)])
    bm25 = _FakeOrderedRetriever([_doc("b1", 4)])

    retriever = HybridRetriever(
        vector_retriever=vector, bm25_retriever=bm25, vector_weight=0.7, bm25_weight=0.3, top_k=2
    )

    results = retriever.retrieve("query")

    assert len(results) == 2


def test_hybrid_retriever_higher_weight_wins_a_tied_rank() -> None:
    # Same document, rank 0 in both lists individually would tie - but with
    # unequal weights, whichever list is weighted higher should NOT change
    # which single document ranks first when it's the only candidate;
    # instead we check that a bm25-only top document loses to a
    # vector-only top document when vector is weighted much higher.
    vector_top = _doc("vector top", page=1)
    bm25_top = _doc("bm25 top", page=2)

    vector = _FakeOrderedRetriever([vector_top])
    bm25 = _FakeOrderedRetriever([bm25_top])

    retriever = HybridRetriever(
        vector_retriever=vector, bm25_retriever=bm25, vector_weight=0.9, bm25_weight=0.1, top_k=2
    )

    results = retriever.retrieve("query")

    assert results[0].page_content == "vector top"


def test_hybrid_retriever_deduplicates_by_source_and_page() -> None:
    same_chunk_v = _doc("same text", page=0)
    same_chunk_b = _doc("same text", page=0)

    vector = _FakeOrderedRetriever([same_chunk_v])
    bm25 = _FakeOrderedRetriever([same_chunk_b])

    retriever = HybridRetriever(
        vector_retriever=vector, bm25_retriever=bm25, vector_weight=0.5, bm25_weight=0.5, top_k=5
    )

    results = retriever.retrieve("query")

    assert len(results) == 1


def test_hybrid_retriever_implements_base_retriever_contract() -> None:
    retriever = HybridRetriever(
        vector_retriever=_FakeOrderedRetriever([]),
        bm25_retriever=_FakeOrderedRetriever([]),
        vector_weight=0.5,
        bm25_weight=0.5,
        top_k=3,
    )
    assert isinstance(retriever, BaseRetriever)