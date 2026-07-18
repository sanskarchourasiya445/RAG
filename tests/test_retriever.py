"""
tests/test_retriever.py

Unit tests for app/retrievers/retriever.py.

We don't spin up a real Chroma + embedding model here (that belongs in
an integration test, not a fast unit test) - instead we use a lightweight
fake vector store that records how `as_retriever` was called, so we can
assert on the MMR configuration we build without any network/model
dependency.
"""

from __future__ import annotations

import pytest

from app.retrievers.retriever import RetrieverConfigError, build_retriever


class _FakeVectorStore:
    """Minimal stand-in for a Chroma vector store - just records the
    kwargs passed to as_retriever() so tests can assert on them."""

    def __init__(self) -> None:
        self.last_call_kwargs: dict = {}

    def as_retriever(self, **kwargs):
        self.last_call_kwargs = kwargs
        return f"fake_retriever:{kwargs}"


def test_build_retriever_uses_mmr_search_type() -> None:
    store = _FakeVectorStore()
    build_retriever(store, top_k=4, fetch_k=20, lambda_mult=0.5)

    assert store.last_call_kwargs["search_type"] == "mmr"


def test_build_retriever_passes_through_configured_values() -> None:
    store = _FakeVectorStore()
    build_retriever(store, top_k=6, fetch_k=30, lambda_mult=0.7)

    search_kwargs = store.last_call_kwargs["search_kwargs"]
    assert search_kwargs["k"] == 6
    assert search_kwargs["fetch_k"] == 30
    assert search_kwargs["lambda_mult"] == 0.7


def test_build_retriever_rejects_fetch_k_smaller_than_top_k() -> None:
    store = _FakeVectorStore()
    with pytest.raises(RetrieverConfigError):
        build_retriever(store, top_k=10, fetch_k=5, lambda_mult=0.5)


def test_build_retriever_allows_fetch_k_equal_to_top_k() -> None:
    # Not an error, just degenerates toward plain similarity search -
    # valid configuration, even if not the most useful one.
    store = _FakeVectorStore()
    build_retriever(store, top_k=5, fetch_k=5, lambda_mult=0.5)
    assert store.last_call_kwargs["search_kwargs"]["k"] == 5