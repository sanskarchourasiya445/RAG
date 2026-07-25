"""
app/retrievers/reranker.py

Responsibility: re-rank an already-retrieved candidate set using a
cross-encoder model, returning a smaller, more precisely-ordered final
set. Wraps ANY BaseRetriever - it doesn't care whether the candidates
came from vector search, BM25, hybrid fusion, or multi-query expansion.

Why a cross-encoder, and why AFTER retrieval rather than instead of it?
A cross-encoder scores a (query, document) pair directly - jointly
attending over both texts in one forward pass - which is far more
accurate than comparing two independently-computed embeddings (what
vector similarity does), but it doesn't scale: you can't cross-encode a
query against millions of documents at query time. So the standard
two-stage pattern is: use a cheap/fast strategy (vector/BM25/hybrid) to
pull a broad candidate pool (optimizing for RECALL), then use a slower,
more accurate cross-encoder to re-score just that pool (optimizing for
PRECISION) and keep only the best few.

Toggle: ENABLE_RERANKER in app/config.py. When disabled,
RetrieverFactory never constructs this class at all - zero overhead,
since loading a cross-encoder model has a real, non-trivial cost.

Composition, not inheritance: like MultiQueryRetriever and
HybridRetriever, this class fetches no documents of its own - it wraps
another BaseRetriever and adds re-scoring around its results.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.retrievers.base import BaseRetriever, RetrieverError

logger = logging.getLogger(__name__)


class RerankerError(RetrieverError):
    """Raised when the cross-encoder model fails to load."""


class RerankerRetriever(BaseRetriever):
    """Wraps another BaseRetriever and re-ranks its results with a
    cross-encoder before returning the final top_n."""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        model_name: str,
        final_k: int,
    ) -> None:
        """
        Args:
            base_retriever: supplies the candidate pool to re-rank. Should
                return MORE documents than final_k, or reranking has
                nothing meaningful to narrow down from - see
                RetrieverFactory, which builds the wrapped strategy with
                RERANK_CANDIDATE_K (wider) instead of TOP_K when
                ENABLE_RERANKER is True.
            model_name: HuggingFace cross-encoder model id, e.g.
                "cross-encoder/ms-marco-MiniLM-L-6-v2". Configurable
                (app.config.CROSS_ENCODER_MODEL) so the accuracy/latency
                tradeoff can be tuned without code changes.
            final_k: how many documents to return after reranking
                (app.config.FINAL_K).

        Raises:
            RerankerError: if the cross-encoder model fails to load
                (bad model name, no network access, etc).
        """
        self._base_retriever = base_retriever
        self._final_k = final_k

        try:
            logger.info("Loading cross-encoder reranker model: %s", model_name)
            self._model = CrossEncoder(model_name)
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(
                f"Failed to load cross-encoder model '{model_name}': {exc}"
            ) from exc

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve candidates from the wrapped retriever, re-score them
        with the cross-encoder, and return the top `final_k` by score."""
        candidates = self._base_retriever.retrieve(query)

        if not candidates:
            return []

        pairs = [(query, document.page_content) for document in candidates]

        try:
            scores = self._model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            # Reranking is a precision IMPROVEMENT, not a hard requirement
            # to get an answer - if scoring fails at query time (OOM,
            # unexpected input, etc), fall back to the base retriever's
            # original order rather than raising and losing the answer
            # entirely.
            logger.warning(
                "Cross-encoder scoring failed (%s) - falling back to the "
                "base retriever's original order, truncated to final_k.",
                exc,
            )
            return candidates[: self._final_k]

        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [document for document, _ in ranked[: self._final_k]]