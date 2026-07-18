"""
app/retrievers/retriever.py

Responsibility: turn a Chroma vector store into a configured retriever.
This is where retrieval STRATEGY lives, isolated from both the vector
store (storage concern) and the pipeline (orchestration concern).

Why MMR (Maximal Marginal Relevance) instead of plain similarity search?
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

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from app.config import FETCH_K, LAMBDA_MULT, TOP_K

logger = logging.getLogger(__name__)


class RetrieverConfigError(Exception):
    """Raised when retriever configuration is invalid (e.g. fetch_k < top_k)."""


def build_retriever(
    vector_store: Chroma,
    top_k: int = TOP_K,
    fetch_k: int = FETCH_K,
    lambda_mult: float = LAMBDA_MULT,
) -> VectorStoreRetriever:
    """Build an MMR-configured retriever from an existing vector store."""
    if fetch_k < top_k:
        raise RetrieverConfigError(
            f"fetch_k ({fetch_k}) must be >= top_k ({top_k}) for MMR to have "
            f"a meaningful candidate pool to diversify over."
        )

    logger.info(
        "Building MMR retriever (top_k=%d, fetch_k=%d, lambda_mult=%.2f)",
        top_k,
        fetch_k,
        lambda_mult,
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )
    return retriever