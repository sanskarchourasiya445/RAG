"""
tests/test_chunk_identity.py

Unit tests for deterministic chunk identity.

Proves:
  - Same source + page + chunk_index → same ID (determinism).
  - Same source + same page + DIFFERENT chunk_index → different IDs (no collision).
  - Different pages → different IDs.
  - Different sources → different IDs.
  - Parent-child chunker produces stable parent_ids across two identical ingestion runs.
  - document_key() includes chunk_index so retriever deduplication is correct.
"""

from __future__ import annotations

import hashlib

import pytest
from langchain_core.documents import Document

from app.chunkers.recursive_chunker import chunk_documents, make_chunk_id
from app.chunkers.parent_child_chunker import split_into_parent_child
from app.retrievers.utils import document_key


# ---------------------------------------------------------------------------
# make_chunk_id — the core identity function
# ---------------------------------------------------------------------------

class TestMakeChunkId:
    def test_same_inputs_same_id(self) -> None:
        """Determinism: identical inputs always produce the same ID."""
        a = make_chunk_id("report.pdf", 3, 0)
        b = make_chunk_id("report.pdf", 3, 0)
        assert a == b

    def test_different_chunk_index_different_id(self) -> None:
        """Uniqueness: two chunks from the same source/page must not collide."""
        id_0 = make_chunk_id("report.pdf", 3, 0)
        id_1 = make_chunk_id("report.pdf", 3, 1)
        id_2 = make_chunk_id("report.pdf", 3, 2)
        assert id_0 != id_1
        assert id_1 != id_2

    def test_different_pages_different_id(self) -> None:
        """Two chunks with identical index but different pages are different."""
        id_p2 = make_chunk_id("report.pdf", 2, 0)
        id_p3 = make_chunk_id("report.pdf", 3, 0)
        assert id_p2 != id_p3

    def test_different_sources_different_id(self) -> None:
        """Two chunks with same page/index from different files are different."""
        id_a = make_chunk_id("doc_a.pdf", 0, 0)
        id_b = make_chunk_id("doc_b.pdf", 0, 0)
        assert id_a != id_b

    def test_id_is_32_hex_chars(self) -> None:
        """The ID must be exactly 32 lowercase hex characters (128-bit prefix)."""
        chunk_id = make_chunk_id("report.pdf", 0, 0)
        assert len(chunk_id) == 32
        assert all(c in "0123456789abcdef" for c in chunk_id)

    def test_id_matches_expected_sha256(self) -> None:
        """Output matches the documented formula sha256('<source>:<page>:<index>')[:32]."""
        expected = hashlib.sha256(b"doc.pdf:5:3").hexdigest()[:32]
        assert make_chunk_id("doc.pdf", 5, 3) == expected


# ---------------------------------------------------------------------------
# chunk_documents integration — IDs are reproducible across two runs
# ---------------------------------------------------------------------------

class TestChunkDocumentsIdentity:
    def _make_doc(self, text: str, source: str = "sample.pdf", page: int = 0) -> Document:
        return Document(
            page_content=text,
            metadata={"source_filename": source, "page": page},
        )

    def test_chunk_index_is_assigned(self) -> None:
        """chunk_documents must assign chunk_index to every chunk."""
        long_text = "Sentence. " * 200
        doc = self._make_doc(long_text)
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_chunk_indices_are_sequential(self) -> None:
        """chunk_index values start at 0 and increment by 1."""
        long_text = "Sentence. " * 200
        doc = self._make_doc(long_text)
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_repeated_ingestion_same_ids(self) -> None:
        """Two identical ingestion runs produce identical chunk IDs."""
        long_text = "Repeated content. " * 100
        doc = self._make_doc(long_text)

        chunks_run1 = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        chunks_run2 = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

        ids_run1 = [
            make_chunk_id(
                c.metadata["source_filename"],
                c.metadata["page"],
                c.metadata["chunk_index"],
            )
            for c in chunks_run1
        ]
        ids_run2 = [
            make_chunk_id(
                c.metadata["source_filename"],
                c.metadata["page"],
                c.metadata["chunk_index"],
            )
            for c in chunks_run2
        ]

        assert ids_run1 == ids_run2

    def test_no_id_collision_across_pages(self) -> None:
        """Chunks from different pages must have distinct IDs even with same chunk_index."""
        text = "Content. " * 100
        doc_p0 = self._make_doc(text, page=0)
        doc_p1 = self._make_doc(text, page=1)

        chunks_p0 = chunk_documents([doc_p0], chunk_size=200, chunk_overlap=20)
        chunks_p1 = chunk_documents([doc_p1], chunk_size=200, chunk_overlap=20)

        ids_p0 = {
            make_chunk_id(c.metadata["source_filename"], c.metadata["page"], c.metadata["chunk_index"])
            for c in chunks_p0
        }
        ids_p1 = {
            make_chunk_id(c.metadata["source_filename"], c.metadata["page"], c.metadata["chunk_index"])
            for c in chunks_p1
        }

        # No overlap between the two pages' ID sets
        assert ids_p0.isdisjoint(ids_p1)


# ---------------------------------------------------------------------------
# parent_child_chunker — deterministic parent_id
# ---------------------------------------------------------------------------

class TestParentChildIdentity:
    def _make_doc(self, text: str, source: str = "book.pdf", page: int = 0) -> Document:
        return Document(
            page_content=text,
            metadata={"source_filename": source, "page": page},
        )

    def test_parent_id_is_deterministic(self) -> None:
        """Two identical ingestion runs produce the same parent_id values."""
        text = "Parent content. " * 200
        docs = [self._make_doc(text)]

        parents1, _ = split_into_parent_child(docs, parent_chunk_size=800, child_chunk_size=200)
        parents2, _ = split_into_parent_child(docs, parent_chunk_size=800, child_chunk_size=200)

        ids1 = [p.metadata["parent_id"] for p in parents1]
        ids2 = [p.metadata["parent_id"] for p in parents2]
        assert ids1 == ids2

    def test_children_share_parent_id(self) -> None:
        """Every child chunk must carry the parent_id of the parent it belongs to."""
        text = "Some text. " * 300
        docs = [self._make_doc(text)]
        parents, children = split_into_parent_child(docs, parent_chunk_size=800, child_chunk_size=200)

        parent_ids = {p.metadata["parent_id"] for p in parents}
        child_parent_ids = {c.metadata["parent_id"] for c in children}
        # Every parent_id referenced by a child must exist as a parent
        assert child_parent_ids.issubset(parent_ids)

    def test_different_sources_different_parent_ids(self) -> None:
        """Parents from different files must not share IDs."""
        text = "Content. " * 200
        doc_a = self._make_doc(text, source="file_a.pdf")
        doc_b = self._make_doc(text, source="file_b.pdf")

        parents_a, _ = split_into_parent_child([doc_a], parent_chunk_size=800, child_chunk_size=200)
        parents_b, _ = split_into_parent_child([doc_b], parent_chunk_size=800, child_chunk_size=200)

        ids_a = {p.metadata["parent_id"] for p in parents_a}
        ids_b = {p.metadata["parent_id"] for p in parents_b}
        assert ids_a.isdisjoint(ids_b)


# ---------------------------------------------------------------------------
# document_key — retriever deduplication key includes chunk_index
# ---------------------------------------------------------------------------

class TestDocumentKey:
    def test_same_page_different_chunk_index_different_keys(self) -> None:
        """Chunks from the same source/page but different chunk_index must have different keys."""
        doc_a = Document(
            page_content="chunk A text",
            metadata={"source": "doc.pdf", "page": 3, "chunk_index": 0},
        )
        doc_b = Document(
            page_content="chunk B text",
            metadata={"source": "doc.pdf", "page": 3, "chunk_index": 1},
        )
        assert document_key(doc_a) != document_key(doc_b)

    def test_same_inputs_same_key(self) -> None:
        """Identical metadata produces the same key (determinism for deduplication)."""
        doc1 = Document(
            page_content="text",
            metadata={"source": "doc.pdf", "page": 0, "chunk_index": 2},
        )
        doc2 = Document(
            page_content="different text",
            metadata={"source": "doc.pdf", "page": 0, "chunk_index": 2},
        )
        # Key is based on metadata, not content
        assert document_key(doc1) == document_key(doc2)

    def test_key_without_chunk_index_falls_back_to_source_page(self) -> None:
        """When chunk_index is absent, key is (source, page) — backwards compatible."""
        doc = Document(
            page_content="text",
            metadata={"source": "doc.pdf", "page": 5},
        )
        assert document_key(doc) == ("doc.pdf", 5)

    def test_key_without_source_falls_back_to_content(self) -> None:
        """When source/page are absent, key falls back to raw page_content."""
        doc = Document(page_content="unique content here", metadata={})
        assert document_key(doc) == ("unique content here",)
