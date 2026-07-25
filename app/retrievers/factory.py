"""
app/retrievers/factory.py

Responsibility: the ONLY place in the codebase that maps a configured
RETRIEVER_TYPE to a concrete BaseRetriever implementation. Mirrors
app/llms/llm_factory.py (get_llm) exactly - same pattern, same reason:
"switch strategy by editing one value in app/config.py, zero code changes
elsewhere" is only true if exactly one file knows about concrete types.

The pipeline calls RetrieverFactory.create(...) and receives a
BaseRetriever - it never imports VectorRetriever/MultiQueryRetriever
directly.

Adding a new retriever type later (Version 3+: "hybrid", "parent", ...) =
one new app/retrievers/<name>.py file + one new branch here. No other
file in the repository changes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_chroma import Chroma

from app.config import (
    CROSS_ENCODER_MODEL,
    ENABLE_COMPRESSION,
    ENABLE_RERANKER,
    FETCH_K,
    FINAL_K,
    HYBRID_BM25_WEIGHT,
    HYBRID_VECTOR_WEIGHT,
    LAMBDA_MULT,
    MULTI_QUERY_COUNT,
    RERANK_CANDIDATE_K,
    RETRIEVER_TYPE,
    TOP_K,
)
from app.llms.base import BaseLLM
from app.retrievers.base import BaseRetriever

if TYPE_CHECKING:
    from app.vectorstores.parent_store import ParentDocumentStore

logger = logging.getLogger(__name__)

SUPPORTED_RETRIEVER_TYPES = ("vector", "multi_query", "bm25", "hybrid", "parent")


class RetrieverFactoryError(Exception):
    """Raised when the configured retriever type is invalid, or required
    dependencies (e.g. an LLM for multi_query) are missing."""


class RetrieverFactory:
    """Builds the configured BaseRetriever. Stateless - a static factory,
    not an instance, since building a retriever needs no state of its own
    beyond the arguments passed in for this one call."""

    @staticmethod
    def create(
        vector_store: Chroma,
        llm: BaseLLM | None = None,
        retriever_type: str | None = None,
        parent_store: "ParentDocumentStore | None" = None,
    ) -> BaseRetriever:
        """Build the configured retriever strategy, optionally wrapped in
        a cross-encoder reranker and/or contextual compression.

        Args:
            vector_store: an already-initialized Chroma store. Every
                supported strategy is built on top of vector search, so
                this is always required.
            llm: required only for strategies/wrappers that need one
                (currently "multi_query", for query-variant generation,
                and contextual compression). Pass the pipeline's
                already-instantiated BaseLLM - the factory does not
                create its own.
            retriever_type: override the configured type (mainly useful
                for tests). Defaults to app.config.RETRIEVER_TYPE.
            parent_store: required only for retriever_type="parent" -
                where each matched child chunk's parent is looked up.
                The pipeline owns its lifecycle (it's tied to ingestion,
                same as the vector store itself) and passes it in here.

        Raises:
            RetrieverFactoryError: unsupported retriever_type, or a
                required dependency (llm, parent_store) wasn't provided
                for the selected type/wrapper.
        """
        # Local imports: keeps this module free of every concrete
        # retriever's import cost (and their transitive dependencies)
        # until the specific type is actually requested.
        from app.retrievers.vector import VectorRetriever

        selected = (retriever_type or RETRIEVER_TYPE).lower()
        logger.info(
            "Building retriever via factory: type=%s (reranker=%s, compression=%s)",
            selected,
            ENABLE_RERANKER,
            ENABLE_COMPRESSION,
        )

        # When reranking is enabled, the base strategy should surface a
        # WIDER candidate pool (RERANK_CANDIDATE_K) for the cross-encoder
        # to narrow down from - the final "how many documents actually
        # reach the LLM" decision becomes the reranker's job (FINAL_K),
        # not the base strategy's TOP_K. When reranking is disabled, the
        # base strategy's own TOP_K is the final answer, same as before
        # this version existed.
        candidate_k = RERANK_CANDIDATE_K if ENABLE_RERANKER else TOP_K

        vector_retriever = VectorRetriever(
            vector_store,
            top_k=candidate_k,
            fetch_k=FETCH_K,
            lambda_mult=LAMBDA_MULT,
        )

        if selected == "vector":
            retriever: BaseRetriever = vector_retriever

        elif selected == "multi_query":
            from app.retrievers.multi_query import MultiQueryRetriever

            if llm is None:
                raise RetrieverFactoryError(
                    "retriever_type='multi_query' requires an llm instance "
                    "to generate query variants, but none was provided."
                )
            retriever = MultiQueryRetriever(
                llm=llm,
                base_retriever=vector_retriever,
                query_count=MULTI_QUERY_COUNT,
            )

        elif selected == "bm25":
            retriever = RetrieverFactory._build_bm25_retriever(vector_store, candidate_k)

        elif selected == "hybrid":
            from app.retrievers.hybrid import HybridRetriever

            bm25_retriever = RetrieverFactory._build_bm25_retriever(vector_store, candidate_k)
            retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=bm25_retriever,
                vector_weight=HYBRID_VECTOR_WEIGHT,
                bm25_weight=HYBRID_BM25_WEIGHT,
                top_k=candidate_k,
            )

        elif selected == "parent":
            from app.retrievers.parent import ParentDocumentRetriever

            if parent_store is None:
                raise RetrieverFactoryError(
                    "retriever_type='parent' requires a parent_store instance "
                    "(the vector store must be populated with CHILD chunks - "
                    "see RAGPipeline.initialize), but none was provided."
                )
            # vector_retriever above is built over the vector store as-is -
            # for retriever_type="parent" that store was populated with
            # CHILD chunks (not the single-level chunks every other
            # strategy uses), so it's already a "child retriever" here,
            # no separate construction needed.
            retriever = ParentDocumentRetriever(
                child_retriever=vector_retriever,
                parent_store=parent_store,
            )

        else:
            raise RetrieverFactoryError(
                f"Unsupported retriever_type: '{selected}'. Must be one of "
                f"{SUPPORTED_RETRIEVER_TYPES} (set RETRIEVER_TYPE in app/config.py)."
            )

        if ENABLE_RERANKER:
            from app.retrievers.reranker import RerankerRetriever

            retriever = RerankerRetriever(
                base_retriever=retriever,
                model_name=CROSS_ENCODER_MODEL,
                final_k=FINAL_K,
            )

        if ENABLE_COMPRESSION:
            from app.retrievers.compression import CompressionRetriever

            if llm is None:
                raise RetrieverFactoryError(
                    "ENABLE_COMPRESSION=True requires an llm instance to "
                    "extract relevant excerpts, but none was provided."
                )
            retriever = CompressionRetriever(llm=llm, base_retriever=retriever)

        return retriever

    @staticmethod
    def _build_bm25_retriever(vector_store: Chroma, top_k: int) -> BaseRetriever:
        """Shared by both the "bm25" and "hybrid" branches: fetch the full
        corpus out of the vector store and build a BM25 index over it.
        Kept as one method so both call sites can never drift out of sync
        on how that corpus is fetched.

        `top_k` is passed in (not read from config directly) so callers
        can supply the wider RERANK_CANDIDATE_K when reranking is enabled,
        same as the vector retriever above."""
        from app.retrievers.bm25 import BM25Retriever
        from app.vectorstores.chroma_store import get_all_documents

        documents = get_all_documents(vector_store)
        return BM25Retriever(documents, top_k=top_k)