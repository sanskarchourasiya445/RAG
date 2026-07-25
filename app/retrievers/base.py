"""
app/retrievers/base.py

Responsibility: define the Strategy Pattern contract every retriever
implementation must satisfy. The pipeline depends ONLY on this
abstraction - it never imports VectorRetriever, MultiQueryRetriever, or
any other concrete retriever directly.

This mirrors app/llms/base.py (BaseLLM): a single deliberately-minimal
method so swapping retrieval strategies is a config change
(RETRIEVER_TYPE in app/config.py), never a pipeline code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class RetrieverError(Exception):
    """Raised when a retriever fails to produce results (bad config,
    upstream vector store failure, LLM failure during query expansion,
    etc). Concrete retrievers should catch provider/library-specific
    exceptions and re-raise as RetrieverError so the pipeline has one
    exception type to handle, regardless of which strategy is configured.
    """


class BaseRetriever(ABC):
    """Abstract interface for a retrieval strategy.

    Deliberately minimal: a single `retrieve(query) -> List[Document]`
    contract. Everything about HOW documents are found (MMR similarity,
    query expansion, hybrid fusion, reranking, ...) is an implementation
    detail hidden behind this one method.
    """

    @abstractmethod
    def retrieve(self, query: str) -> List[Document]:
        """Return the documents relevant to `query`.

        Raises:
            RetrieverError: on any failure that prevents retrieval from
            completing (e.g. malformed configuration, upstream failure).
            Returning an empty list is NOT an error - it's a valid
            "nothing relevant found" result that the pipeline already
            handles (see RAGPipeline.query's zero-result short-circuit).
        """
        raise NotImplementedError