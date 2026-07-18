"""
app/chunkers/recursive_chunker.py

Responsibility: split page-level Documents into retrieval-sized chunks.

Why RecursiveCharacterTextSplitter specifically?
It tries a prioritized list of separators (paragraph -> line -> sentence ->
word -> character) and only falls back to a harder split when a softer one
doesn't fit within chunk_size. This keeps chunks semantically coherent
instead of slicing mid-sentence, which matters a lot for retrieval quality
and for how well the LLM can ground an answer in a single chunk.

Design note: chunk_size/overlap are NOT hardcoded here. They default to
the values in app/config.py so tuning retrieval quality is a one-line
config change, not a change scattered across files.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


def build_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Factory for the splitter. Kept as a function (not a module-level
    singleton) so tests can construct splitters with different parameters
    without mutating global state."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split page-level documents into retrieval-sized chunks.

    Metadata (source_filename, page, ingested_at_utc) is propagated
    automatically by LangChain's splitter, since it copies the parent
    Document's metadata onto every resulting chunk. We verify that here
    so a silent metadata-loss regression fails loudly in logs/tests
    instead of surfacing as "why don't citations show page numbers"
    three weeks later.
    """
    if not documents:
        logger.warning("chunk_documents called with an empty document list.")
        return []

    splitter = build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    missing_metadata = [c for c in chunks if "source_filename" not in c.metadata]
    if missing_metadata:
        logger.error(
            "%d chunk(s) lost 'source_filename' metadata during splitting.",
            len(missing_metadata),
        )

    # Give every chunk a stable, human-readable index within its source file.
    # Useful for debugging retrieval results and for deduplication logic
    # in the vector store layer.
    per_source_counter: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source_filename", "unknown")
        idx = per_source_counter.get(source, 0)
        chunk.metadata["chunk_index"] = idx
        per_source_counter[source] = idx + 1

    logger.info(
        "Split %d page(s) into %d chunk(s) (chunk_size=%d, overlap=%d).",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks