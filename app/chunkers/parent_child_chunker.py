"""
app/chunkers/parent_child_chunker.py

Responsibility: split page-level Documents into PARENT chunks (larger,
returned to the LLM for context) and CHILD chunks (smaller, embedded for
precise vector search), with every child tagged with the id of the
parent it came from.

Why two splitters instead of one?
Vector search works best on small, focused chunks - a large chunk dilutes
the embedding with unrelated content, hurting similarity match quality.
But a small chunk often lacks enough surrounding context for the LLM to
answer well on its own. The Parent Document Retriever pattern gets both:
similarity search matches against small, precise CHILD chunks, but the
larger PARENT chunk each match belongs to is what's actually returned and
sent to the LLM (see app/retrievers/parent.py).

Why a separate module from app/chunkers/recursive_chunker.py?
recursive_chunker.py implements the single-level chunking used by every
retriever strategy EXCEPT "parent". Keeping this two-level splitting
logic in its own file means RETRIEVER_TYPE="parent" is the only thing
that pulls this module in - the shared chunker used by every other
strategy never needs a parent/child branch inside it.
"""

from __future__ import annotations

import hashlib
import logging
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)

logger = logging.getLogger(__name__)


class ChunkConfigError(Exception):
    """Raised when parent/child chunk sizes are configured inconsistently."""


def split_into_parent_child(
    documents: List[Document],
    parent_chunk_size: int = PARENT_CHUNK_SIZE,
    parent_chunk_overlap: int = PARENT_CHUNK_OVERLAP,
    child_chunk_size: int = CHILD_CHUNK_SIZE,
    child_chunk_overlap: int = CHILD_CHUNK_OVERLAP,
) -> Tuple[List[Document], List[Document]]:
    """Split `documents` into (parent_chunks, child_chunks).

    Every parent chunk gets a freshly generated "parent_id" in its
    metadata; every child chunk split from that parent gets the SAME
    "parent_id" in its metadata. That shared id is how
    ParentDocumentRetriever maps a vector-search hit (a child) back to
    the parent it should actually return.

    Raises:
        ChunkConfigError: if child_chunk_size >= parent_chunk_size - a
            child chunk that isn't meaningfully smaller than its parent
            defeats the entire point of this splitting strategy.
    """
    if child_chunk_size >= parent_chunk_size:
        raise ChunkConfigError(
            f"child_chunk_size ({child_chunk_size}) must be smaller than "
            f"parent_chunk_size ({parent_chunk_size}) - otherwise child "
            f"chunks provide no precision benefit over the parent."
        )

    if not documents:
        logger.warning("split_into_parent_child called with an empty document list.")
        return [], []

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_chunks = parent_splitter.split_documents(documents)

    parents_with_ids: List[Document] = []
    child_chunks: List[Document] = []

    # Track a per-source parent counter so the ID is stable:
    # parent N from file X always gets the same hash regardless of
    # which other files are being ingested in the same batch.
    per_source_parent_counter: dict[str, int] = {}

    for parent in parent_chunks:
        source = parent.metadata.get("source_filename") or parent.metadata.get("source", "")
        page = parent.metadata.get("page", 0)
        parent_idx = per_source_parent_counter.get(source, 0)
        per_source_parent_counter[source] = parent_idx + 1

        # Deterministic parent_id: same document ingested twice → same ID.
        raw = f"{source}:{page}:{parent_idx}"
        parent_id = hashlib.sha256(raw.encode()).hexdigest()[:32]

        parent.metadata["parent_id"] = parent_id
        parents_with_ids.append(parent)

        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent_id
        child_chunks.extend(children)

    logger.info(
        "Split %d page(s) into %d parent chunk(s) and %d child chunk(s) "
        "(parent_size=%d, child_size=%d).",
        len(documents),
        len(parents_with_ids),
        len(child_chunks),
        parent_chunk_size,
        child_chunk_size,
    )
    return parents_with_ids, child_chunks