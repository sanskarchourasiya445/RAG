"""
tests/test_parent_store.py

Unit tests for app/vectorstores/parent_store.py. Uses pytest's tmp_path
fixture for a real (throwaway) directory - this is a filesystem-backed
store, so a fake/in-memory double would test nothing meaningful about
its actual persistence behavior.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.vectorstores.parent_store import ParentDocumentStore, ParentStoreError


def test_parent_store_save_and_get_roundtrip(tmp_path) -> None:
    store = ParentDocumentStore(tmp_path)
    doc = Document(page_content="parent text", metadata={"parent_id": "abc123", "source": "a.pdf"})

    store.save_all([doc])
    result = store.get("abc123")

    assert result is not None
    assert result.page_content == "parent text"
    assert result.metadata["source"] == "a.pdf"


def test_parent_store_get_returns_none_for_missing_id(tmp_path) -> None:
    store = ParentDocumentStore(tmp_path)

    assert store.get("does-not-exist") is None


def test_parent_store_raises_when_parent_id_missing(tmp_path) -> None:
    store = ParentDocumentStore(tmp_path)
    doc = Document(page_content="text", metadata={})  # no parent_id

    with pytest.raises(ParentStoreError):
        store.save_all([doc])


def test_parent_store_creates_its_directory(tmp_path) -> None:
    target = tmp_path / "nested" / "parent_store"
    assert not target.exists()

    ParentDocumentStore(target)

    assert target.exists()


def test_parent_store_persists_across_instances(tmp_path) -> None:
    ParentDocumentStore(tmp_path).save_all(
        [Document(page_content="persisted", metadata={"parent_id": "p1"})]
    )

    # A brand new store instance pointed at the same directory should
    # see what a previous instance saved - this is what makes parent
    # chunks survive a pipeline restart.
    reopened = ParentDocumentStore(tmp_path)
    result = reopened.get("p1")

    assert result is not None
    assert result.page_content == "persisted"