"""
app/retrievers/utils.py

Responsibility: small, pure helper functions shared across retriever
strategies. Kept separate from any single retriever so MultiQueryRetriever
today - and HybridRetriever / rerank merging later (Version 3+) - can
reuse the same deduplication logic instead of each reimplementing it.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document


def document_key(document: Document) -> tuple:
    """Build a stable identity key for a Document.

    Uses (source, page, chunk_index) when all three metadata fields are
    available. chunk_index is assigned by the chunker (recursive_chunker.py)
    to distinguish multiple chunks that come from the same source/page —
    without it, two chunks from page 3 of the same PDF would collide and one
    would be silently dropped during RRF deduplication.

    Falls back to (source, page) when chunk_index is absent (e.g. parent
    chunks, which represent a full page-level split and are not sub-indexed),
    and then to raw page_content for loaders that don't set source/page at all.

    Public (not just an internal helper for deduplicate_documents below)
    because HybridRetriever (app/retrievers/hybrid.py) needs the same
    notion of "is this the same document" to match a chunk's vector-search
    rank against its BM25 rank during Reciprocal Rank Fusion.
    """
    source = document.metadata.get("source")
    page = document.metadata.get("page")
    if source is not None and page is not None:
        chunk_index = document.metadata.get("chunk_index")
        if chunk_index is not None:
            return (source, page, chunk_index)
        return (source, page)
    return (document.page_content,)


def deduplicate_documents(documents: List[Document]) -> List[Document]:
    """Remove duplicate documents while preserving first-seen order.

    Order matters here: documents from the ORIGINAL query are placed
    first by callers (see MultiQueryRetriever), so when a duplicate is
    found, the copy retrieved for the original query - not a rephrased
    variant - is the one kept.
    """
    seen: set[tuple] = set()
    unique: List[Document] = []
    for document in documents:
        key = document_key(document)
        if key in seen:
            continue
        seen.add(key)
        unique.append(document)
    return unique