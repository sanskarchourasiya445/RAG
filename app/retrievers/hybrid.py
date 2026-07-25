"""
app/retrievers/hybrid.py

Responsibility: combine dense (vector) and sparse (BM25) retrieval into
one ranked result list, using weighted Reciprocal Rank Fusion (RRF).

Why RRF instead of merging raw scores?
Vector similarity scores (cosine/L2 distance) and BM25 scores live on
completely different, incompatible scales - there's no principled way to
add "0.82 cosine similarity" to "14.3 BM25 score". RRF sidesteps this by
fusing RANKS instead of raw scores: a document's fused score is
1/(RRF_K + rank) for wherever it placed in each list, summed across
lists. Only relative order within each list matters, so the two
different scoring scales stop being a problem - this is why the
technique is standard for combining sparse + dense retrieval.

HYBRID_VECTOR_WEIGHT / HYBRID_BM25_WEIGHT let one list's ranks count for
more than the other's in the fused score (a weighted variant of the
plain RRF paper). E.g. weighting vector search higher for conceptual
questions, or BM25 higher when exact keyword/terminology matches matter
more (product codes, acronyms, proper nouns embeddings often blur
together).

A document that appears in only one of the two lists is NOT penalized to
zero for the list it's missing from - it simply gets no contribution
from that list, which is exactly the desired behavior (BM25 and vector
search have different blind spots; a chunk found by only one of them is
still a real candidate, just a weaker one than a chunk both agree on).
"""

from __future__ import annotations

import logging
from typing import Dict, List

from langchain_core.documents import Document

from app.retrievers.base import BaseRetriever
from app.retrievers.utils import document_key

logger = logging.getLogger(__name__)

# Standard RRF smoothing constant from the original Reciprocal Rank
# Fusion paper (Cormack et al.). Large enough that a document's EXACT
# rank matters less than simply appearing near the top of a list, which
# is what makes RRF robust to the two lists using unrelated scoring
# scales in the first place.
RRF_K = 60


class HybridRetriever(BaseRetriever):
    """Fuses a vector retriever and a BM25 retriever via weighted RRF.

    Composition, not inheritance: this class fetches no documents of its
    own - it delegates to the two BaseRetriever instances it wraps and
    only adds the rank-fusion logic around their combined results.
    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        vector_weight: float,
        bm25_weight: float,
        top_k: int,
    ) -> None:
        """
        Args:
            vector_retriever: dense retrieval strategy (typically VectorRetriever).
            bm25_retriever: sparse retrieval strategy (BM25Retriever).
            vector_weight: RRF weight applied to the vector list's ranks.
            bm25_weight: RRF weight applied to the BM25 list's ranks.
            top_k: how many fused results retrieve() returns.
        """
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._top_k = top_k

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve from both strategies and return the top_k fused results."""
        vector_results = self._vector_retriever.retrieve(query)
        bm25_results = self._bm25_retriever.retrieve(query)

        scores: Dict[tuple, float] = {}
        documents_by_key: Dict[tuple, Document] = {}

        for rank, document in enumerate(vector_results):
            key = document_key(document)
            documents_by_key[key] = document
            scores[key] = scores.get(key, 0.0) + self._rrf_contribution(rank, self._vector_weight)

        for rank, document in enumerate(bm25_results):
            key = document_key(document)
            documents_by_key.setdefault(key, document)
            scores[key] = scores.get(key, 0.0) + self._rrf_contribution(rank, self._bm25_weight)

        ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [documents_by_key[key] for key in ranked_keys[: self._top_k]]

    @staticmethod
    def _rrf_contribution(rank: int, weight: float) -> float:
        """A single list's contribution to a document's fused RRF score.
        `rank` is 0-indexed (position in the list); RRF conventionally
        uses 1-indexed ranks, hence the `+ 1` below."""
        return weight * (1.0 / (RRF_K + rank + 1))