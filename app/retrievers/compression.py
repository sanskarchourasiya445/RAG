"""
app/retrievers/compression.py

Responsibility: compress each retrieved document down to only the
content relevant to the query, before it reaches the answer-generation
LLM. Wraps ANY BaseRetriever - typically applied as the LAST stage
(after retrieval, and after reranking if enabled), immediately before
the final RAG prompt is built.

Why NOT LangChain's ContextualCompressionRetriever + LLMChainExtractor?
Same reason as MultiQueryRetriever (app/retrievers/multi_query.py):
LLMChainExtractor.from_llm() requires a LangChain BaseLanguageModel/
Runnable, and this project's BaseLLM is deliberately not that - it's a
minimal generate(prompt) -> str contract so providers stay swappable
with zero LangChain-specific glue in app/llms/. So this class implements
the same "ask the LLM to extract only the relevant span" algorithm
directly against our own BaseLLM and BaseRetriever abstractions.

Why compress at all, given reranking already picked the best chunks?
Reranking (app/retrievers/reranker.py) selects the best whole CHUNKS -
it doesn't shrink them. A selected chunk can still contain a few
sentences that answer the question and several paragraphs that don't.
Compression asks the LLM to extract just the relevant sentences from
each chunk, so the final prompt is denser signal and less filler - which
helps both token cost and answer accuracy (less chance of the model
getting distracted by unrelated nearby text in the same chunk).

Fail-soft behavior:
  - If compressing one document fails (LLM error) or the LLM finds
    nothing relevant in it, that document is DROPPED, not kept raw -
    keeping raw/failed content would silently defeat compression's
    entire purpose.
  - If EVERY document ends up dropped (over-aggressive compression, or
    the LLM having a bad run), we fall back to the ORIGINAL, uncompressed
    candidate set - so a base retriever that DID find something never
    ends up handing query() zero context because of a compression stage
    that was only ever meant to refine, not gate, the result.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document

from app.config import NO_RELEVANT_CONTENT_MESSAGE
from app.llms.base import BaseLLM, LLMGenerationError
from app.prompts.compression_prompt import build_compression_prompt
from app.retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)


class CompressionRetriever(BaseRetriever):
    """Wraps a BaseRetriever and compresses each of its results to the
    query-relevant excerpt only.

    Composition, not inheritance: like MultiQueryRetriever and
    RerankerRetriever, this class fetches no documents of its own - it
    delegates to the wrapped `base_retriever` and adds per-document
    compression around its results.
    """

    def __init__(self, llm: BaseLLM, base_retriever: BaseRetriever) -> None:
        """
        Args:
            llm: used to extract the relevant excerpt from each document.
            base_retriever: supplies the candidate documents to compress -
                typically the (possibly reranked) output of another
                strategy in this package.
        """
        self._llm = llm
        self._base_retriever = base_retriever

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve candidates from the wrapped retriever, compress each
        one to its query-relevant excerpt, and drop any that compress to
        nothing.

        Falls back to the original, uncompressed candidates if
        compression drops every single one - see module docstring.
        """
        candidates = self._base_retriever.retrieve(query)
        if not candidates:
            return []

        compressed: List[Document] = []
        for document in candidates:
            excerpt = self._compress_one(query, document)
            if excerpt is not None:
                compressed.append(Document(page_content=excerpt, metadata=document.metadata))

        if not compressed:
            logger.warning(
                "Contextual compression dropped every candidate for this "
                "query - falling back to the uncompressed candidate set."
            )
            return candidates

        return compressed

    def _compress_one(self, query: str, document: Document) -> Optional[str]:
        """Return the query-relevant excerpt of `document`, or None if
        compression failed or found nothing relevant."""
        prompt = build_compression_prompt(question=query, document_text=document.page_content)

        try:
            raw_output = self._llm.generate(prompt)
        except LLMGenerationError:
            logger.warning(
                "Compression failed for one document (LLM generation "
                "error) - dropping it rather than keeping raw content."
            )
            return None

        excerpt = raw_output.strip()
        if not excerpt or excerpt == NO_RELEVANT_CONTENT_MESSAGE:
            return None
        return excerpt