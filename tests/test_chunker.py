"""
tests/test_chunker.py

Unit tests for app/chunkers/recursive_chunker.py.

These tests use plain in-memory Document objects (no PDF I/O at all) -
chunking logic should be fully testable in isolation from loading.
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.chunkers.recursive_chunker import chunk_documents


def _make_doc(text: str, source: str = "sample.pdf", page: int = 0) -> Document:
    return Document(page_content=text, metadata={"source_filename": source, "page": page})


def test_chunk_documents_empty_list_returns_empty_list() -> None:
    assert chunk_documents([]) == []


def test_chunk_documents_splits_long_text() -> None:
    long_text = "Sentence one. " * 500  # comfortably longer than default chunk_size
    doc = _make_doc(long_text)

    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        # allow slight overshoot since the splitter respects separators
        # over hard character limits
        assert len(chunk.page_content) <= 250


def test_chunk_documents_preserves_source_metadata() -> None:
    doc = _make_doc("Some short PHC supply chain content.", source="phc_report.pdf", page=3)

    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)

    assert all(c.metadata["source_filename"] == "phc_report.pdf" for c in chunks)
    assert all(c.metadata["page"] == 3 for c in chunks)


def test_chunk_documents_assigns_incrementing_chunk_index_per_source() -> None:
    long_text = "Chunk content here. " * 100
    doc = _make_doc(long_text, source="report.pdf")

    chunks = chunk_documents([doc], chunk_size=150, chunk_overlap=10)

    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_documents_short_text_produces_single_chunk() -> None:
    doc = _make_doc("A very short sentence.")
    chunks = chunk_documents([doc], chunk_size=1000, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].page_content == "A very short sentence."