"""
tests/test_parent_child_chunker.py

Unit tests for app/chunkers/parent_child_chunker.py.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.chunkers.parent_child_chunker import ChunkConfigError, split_into_parent_child

_LONG_TEXT = (
    "Paragraph one has several sentences. It talks about apples. "
    "Apples are a common fruit.\n\n"
    "Paragraph two talks about oranges instead. Oranges are citrus. "
    "Citrus fruits are acidic.\n\n"
    "Paragraph three is about bananas. Bananas are tropical. "
    "They grow in bunches."
) * 3  # long enough to actually split into multiple parent chunks


def test_split_into_parent_child_raises_when_child_not_smaller_than_parent() -> None:
    with pytest.raises(ChunkConfigError):
        split_into_parent_child(
            [Document(page_content="text", metadata={})],
            parent_chunk_size=200,
            parent_chunk_overlap=0,
            child_chunk_size=200,
            child_chunk_overlap=0,
        )


def test_split_into_parent_child_returns_empty_for_empty_input() -> None:
    parents, children = split_into_parent_child(
        [], parent_chunk_size=500, parent_chunk_overlap=0, child_chunk_size=100, child_chunk_overlap=0
    )
    assert parents == []
    assert children == []


def test_split_into_parent_child_every_child_has_a_matching_parent() -> None:
    doc = Document(page_content=_LONG_TEXT, metadata={"source": "a.pdf", "page": 0})
    parents, children = split_into_parent_child(
        [doc],
        parent_chunk_size=300,
        parent_chunk_overlap=0,
        child_chunk_size=80,
        child_chunk_overlap=0,
    )

    assert len(parents) > 1  # source text was long enough to need multiple parents
    assert len(children) > len(parents)  # each parent split into multiple children

    parent_ids = {p.metadata["parent_id"] for p in parents}
    for child in children:
        assert child.metadata["parent_id"] in parent_ids


def test_split_into_parent_child_preserves_source_metadata() -> None:
    doc = Document(page_content=_LONG_TEXT, metadata={"source": "a.pdf", "page": 2})
    parents, children = split_into_parent_child(
        [doc],
        parent_chunk_size=300,
        parent_chunk_overlap=0,
        child_chunk_size=80,
        child_chunk_overlap=0,
    )

    for parent in parents:
        assert parent.metadata["source"] == "a.pdf"
        assert parent.metadata["page"] == 2
    for child in children:
        assert child.metadata["source"] == "a.pdf"
        assert child.metadata["page"] == 2


def test_split_into_parent_child_assigns_unique_parent_ids() -> None:
    doc = Document(page_content=_LONG_TEXT, metadata={"source": "a.pdf", "page": 0})
    parents, _ = split_into_parent_child(
        [doc],
        parent_chunk_size=300,
        parent_chunk_overlap=0,
        child_chunk_size=80,
        child_chunk_overlap=0,
    )

    ids = [p.metadata["parent_id"] for p in parents]
    assert len(ids) == len(set(ids))