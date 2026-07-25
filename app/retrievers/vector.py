"""
app/retrievers/vector.py

Responsibility: turn a Chroma vector store into a configured retriever
using MMR (Maximal Marginal Relevance) search, exposed through the
BaseRetriever strategy interface.

This is a direct move of the former app/retrievers/retriever.py content
into the Strategy Pattern - the MMR logic itself is UNCHANGED (same
validation, same defaults, same LangChain call), only its shape has
changed: a free function (`build_retriever`) becomes a class
(`VectorRetriever`) implementing `BaseRetriever.retrieve()`, so the
pipeline (and other retrievers, like MultiQueryRetriever) can depend on
the abstraction instead of a concrete LangChain type.

Why MMR instead of plain similarity search?
Plain top-K similarity can return several near-duplicate chunks (e.g. the
same statistic repeated across a summary and a detail section), wasting
context budget on redundant evidence. MMR first fetches a larger candidate
pool (fetch_k) by similarity, then greedily selects `top_k` of them while
penalizing chunks that are too similar to ones already selected - so the
final set is both relevant AND diverse.

lambda_mult controls the relevance/diversity tradeoff:
  1.0 -> pure relevance (behaves like plain similarity search)
  0.0 -> pure diversity (may sacrifice relevance for variety)
  0.5 -> balanced (a reasonable default)

Common beginner mistake: setting fetch_k == top_k. MMR needs a bigger
candidate pool than the final result size, or there's nothing to diversify
between - it silently degenerates into plain similarity search.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import FETCH_K, LAMBDA_MULT, TOP_K
from app.retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)


class RetrieverConfigError(Exception):
    """Raised when retriever configuration is invalid (e.g. fetch_k < top_k)."""


class VectorRetriever(BaseRetriever):
    """MMR-configured retriever over an existing Chroma vector store.

    This is the baseline retrieval strategy every other strategy in this
    package is built on top of (MultiQueryRetriever wraps one, HybridRetriever
    will combine one with BM25, etc) - so its behavior deliberately stays
    simple and well-tested.
    """

    def __init__(
        self,
        vector_store: Chroma,
        top_k: int = TOP_K,
        fetch_k: int = FETCH_K,
        lambda_mult: float = LAMBDA_MULT,
    ) -> None:
        """Build an MMR-configured retriever from an existing vector store.

        Raises:
            RetrieverConfigError: if fetch_k < top_k (MMR would have no
            candidate pool left to diversify over).
        """
        if fetch_k < top_k:
            raise RetrieverConfigError(
                f"fetch_k ({fetch_k}) must be >= top_k ({top_k}) for MMR to "
                f"have a meaningful candidate pool to diversify over."
            )

        self._top_k = top_k
        self._fetch_k = fetch_k
        self._lambda_mult = lambda_mult

        logger.info(
            "Building MMR retriever (top_k=%d, fetch_k=%d, lambda_mult=%.2f)",
            top_k,
            fetch_k,
            lambda_mult,
        )

        self._langchain_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
            },
        )

    def retrieve(self, query: str) -> List[Document]:
        """Return up to `top_k` documents for `query`, selected by MMR."""
        return self._langchain_retriever.invoke(query)