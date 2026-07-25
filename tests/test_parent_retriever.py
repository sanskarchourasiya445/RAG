"""
tests/test_parent_retriever.py

Unit tests for app/retrievers/parent.py. Uses a fake child retriever and
a real ParentDocumentStore backed by tmp_path, so these tests exercise
the actual "look up and dedupe by parent_id" logic without a real vector
store or embedding model.
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.retrievers.base import BaseRetriever
from app.retrievers.parent import ParentDocumentRetriever
from app.vectorstores.parent_store import ParentDocumentStore


class _FakeChildRetriever(BaseRetriever):
    def __init__(self, child_matches: list[Document]) -> None:
        self._child_matches = child_matches

    def retrieve(self, query: str) -> list[Document]:
        return self._child_matches


def _child(parent_id: str, text: str = "child text") -> Document:
    return Document(page_content=text, metadata={"parent_id": parent_id})


def test_parent_retriever_returns_parent_for_matching_child(tmp_path) -> None:
    parent_store = ParentDocumentStore(tmp_path)
    parent_store.save_all(
        [Document(page_content="full parent context", metadata={"parent_id": "p1"})]
    )
    child_retriever = _FakeChildRetriever([_child("p1")])
    retriever = ParentDocumentRetriever(child_retriever=child_retriever, parent_store=parent_store)

    result = retriever.retrieve("query")

    assert len(result) == 1
    assert result[0].page_content == "full parent context"


def test_parent_retriever_deduplicates_multiple_children_of_same_parent(tmp_path) -> None:
    parent_store = ParentDocumentStore(tmp_path)
    parent_store.save_all([Document(page_content="parent text", metadata={"parent_id": "p1"})])
    child_retriever = _FakeChildRetriever([_child("p1", "child A"), _child("p1", "child B")])
    retriever = ParentDocumentRetriever(child_retriever=child_retriever, parent_store=parent_store)

    result = retriever.retrieve("query")

    assert len(result) == 1  # both children map to the same parent


def test_parent_retriever_returns_multiple_distinct_parents(tmp_path) -> None:
    parent_store = ParentDocumentStore(tmp_path)
    parent_store.save_all(
        [
            Document(page_content="parent one", metadata={"parent_id": "p1"}),
            Document(page_content="parent two", metadata={"parent_id": "p2"}),
        ]
    )
    child_retriever = _FakeChildRetriever([_child("p1"), _child("p2")])
    retriever = ParentDocumentRetriever(child_retriever=child_retriever, parent_store=parent_store)

    result = retriever.retrieve("query")

    assert {doc.page_content for doc in result} == {"parent one", "parent two"}


def test_parent_retriever_skips_child_with_missing_parent(tmp_path) -> None:
    parent_store = ParentDocumentStore(tmp_path)  # empty - nothing saved
    child_retriever = _FakeChildRetriever([_child("does-not-exist")])
    retriever = ParentDocumentRetriever(child_retriever=child_retriever, parent_store=parent_store)

    result = retriever.retrieve("query")

    assert result == []


def test_parent_retriever_implements_base_retriever_contract(tmp_path) -> None:
    parent_store = ParentDocumentStore(tmp_path)
    retriever = ParentDocumentRetriever(
        child_retriever=_FakeChildRetriever([]), parent_store=parent_store
    )

    assert isinstance(retriever, BaseRetriever)