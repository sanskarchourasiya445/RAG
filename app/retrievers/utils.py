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

    Prefers (source, page) metadata when available - two chunks from the
    same source/page are almost certainly the same chunk retrieved twice
    (via different query variants, or by both halves of a hybrid search).
    Falls back to the raw page_content when that metadata isn't present
    (e.g. loaders that don't set it), so identity still works, just at
    exact-text granularity instead.

    Public (not just an internal helper for deduplicate_documents below)
    because HybridRetriever (app/retrievers/hybrid.py) needs the same
    notion of "is this the same document" to match a chunk's vector-search
    rank against its BM25 rank during Reciprocal Rank Fusion.
    """
    source = document.metadata.get("source")
    page = document.metadata.get("page")
    if source is not None and page is not None:
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