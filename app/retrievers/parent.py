"""
app/retrievers/parent.py

Responsibility: retrieve documents via similarity search over small CHILD
chunks, but return the larger PARENT chunk each matched child belongs to
- giving vector search the precision of small chunks and the LLM the
context of larger ones.

Why wrap a child-indexed VectorRetriever instead of implementing
similarity search here directly?
Same "wrap an existing, well-tested strategy" pattern as
MultiQueryRetriever / HybridRetriever / RerankerRetriever: this class
adds NO new similarity-search logic of its own. It delegates all actual
matching to a VectorRetriever built over a vector store populated with
CHILD chunks (see app/chunkers/parent_child_chunker.py and
RAGPipeline.initialize, which chunks documents differently specifically
for retriever_type="parent"), and only adds the "look up each match's
parent, deduplicated" step around it.
"""

from __future__ import annotations

import logging
from typing import List, Set

from langchain_core.documents import Document

from app.retrievers.base import BaseRetriever
from app.vectorstores.parent_store import ParentDocumentStore

logger = logging.getLogger(__name__)


class ParentDocumentRetriever(BaseRetriever):
    """Wraps a child-indexed VectorRetriever; returns each match's parent
    chunk instead of the child chunk that actually matched."""

    def __init__(
        self,
        child_retriever: BaseRetriever,
        parent_store: ParentDocumentStore,
    ) -> None:
        """
        Args:
            child_retriever: a retriever built over CHILD chunks (small,
                embedded for precise similarity search) - typically a
                VectorRetriever over a vector store populated via
                split_into_parent_child's child_chunks.
            parent_store: where the larger parent chunk for each child's
                metadata["parent_id"] is looked up.
        """
        self._child_retriever = child_retriever
        self._parent_store = parent_store

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve matching CHILD chunks, then return their PARENT
        chunks - deduplicated, keeping each parent at the rank of its
        first (closest) matching child."""
        child_matches = self._child_retriever.retrieve(query)

        seen_parent_ids: Set[str] = set()
        parents: List[Document] = []

        for child in child_matches:
            parent_id = child.metadata.get("parent_id")
            if not parent_id or parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_id)

            parent = self._parent_store.get(parent_id)
            if parent is None:
                # Store and vector index have drifted out of sync (e.g. the
                # parent store directory was cleared but the vector store
                # wasn't rebuilt) - skip this one match rather than fail
                # the whole query over a single missing parent.
                logger.warning(
                    "Matched child chunk referenced parent_id=%r, but no "
                    "parent document was found in the parent store - "
                    "skipping it.",
                    parent_id,
                )
                continue
            parents.append(parent)

        return parents