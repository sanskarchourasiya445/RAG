"""
app/retrievers/bm25.py

Responsibility: keyword-based (sparse) retrieval via BM25, exposed
through the BaseRetriever interface - this is the sparse half of hybrid
search (app/retrievers/hybrid.py fuses this with VectorRetriever's dense
results).

Why LangChain's BM25Retriever (langchain_community) instead of a
from-scratch implementation?
Unlike MultiQueryRetriever (which needed our own BaseLLM - a deliberately
non-LangChain type), BM25 has no such conflict: it's pure text statistics
over a fixed corpus, no LLM involved. LangChain's BM25Retriever wraps the
well-tested `rank_bm25` library and needs nothing beyond a List[Document],
so reusing it here is the SAME pattern VectorRetriever already uses for
Chroma (wrap an existing, well-tested LangChain retriever behind
BaseRetriever) - not a new architectural exception.

Why build the index from the FULL corpus at construction time, not
per-query?
BM25 scoring depends on corpus-wide statistics (inverse document
frequency: how rare is this term across ALL documents) - there's no
per-query equivalent of vector search's "fetch_k candidates, then
narrow". So the index is built once, eagerly, when this retriever is
constructed (see RetrieverFactory._build_bm25_retriever, which fetches
every chunk via app.vectorstores.chroma_store.get_all_documents), and
every retrieve() call reuses that same in-memory index.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_community.retrievers import BM25Retriever as _LangChainBM25Retriever
from langchain_core.documents import Document

from app.retrievers.base import BaseRetriever, RetrieverError

logger = logging.getLogger(__name__)


class BM25RetrieverError(RetrieverError):
    """Raised when the BM25 index cannot be built (e.g. empty corpus)."""


class BM25Retriever(BaseRetriever):
    """Sparse, keyword-based retriever over a fixed document corpus."""

    def __init__(self, documents: List[Document], top_k: int) -> None:
        """
        Args:
            documents: the FULL corpus to index (not just a candidate
                subset) - every chunk currently in the vector store.
            top_k: how many documents retrieve() returns per query.

        Raises:
            BM25RetrieverError: if `documents` is empty - there's nothing
            meaningful to build a keyword index over.
        """
        if not documents:
            raise BM25RetrieverError(
                "Cannot build a BM25 index from an empty document corpus."
            )

        logger.info(
            "Building BM25 index over %d document(s) (top_k=%d)",
            len(documents),
            top_k,
        )
        self._langchain_retriever = _LangChainBM25Retriever.from_documents(documents)
        self._langchain_retriever.k = top_k

    def retrieve(self, query: str) -> List[Document]:
        """Return up to `top_k` documents for `query`, ranked by BM25 score."""
        return self._langchain_retriever.invoke(query)