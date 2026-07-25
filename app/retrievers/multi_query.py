"""
app/retrievers/multi_query.py

Responsibility: generate several semantically-varied rephrasings of the
user's question using the configured LLM, run the wrapped retriever
against each variant, and merge + deduplicate the combined results.

Why this is a custom implementation, not LangChain's built-in
MultiQueryRetriever.from_llm()?
LangChain's MultiQueryRetriever.from_llm() requires a LangChain
BaseLanguageModel/Runnable to generate query variants. This project's LLM
abstraction (app.llms.base.BaseLLM) is deliberately NOT a LangChain type -
it's a minimal `generate(prompt: str) -> str` contract specifically so any
provider (Mistral/Gemini/Groq) can be swapped via app/llms/llm_factory.py
with zero LangChain-specific glue code anywhere else in the codebase.
Wrapping BaseLLM in a LangChain Runnable purely to satisfy
MultiQueryRetriever.from_llm's type signature would leak a LangChain
dependency back into app/llms/ for one retriever's benefit - so instead
this class implements the same "expand -> retrieve -> merge" algorithm
directly against our own BaseLLM and BaseRetriever abstractions. Same
technique as LangChain's version, zero added framework coupling.

Fallback behavior (why this can never make things WORSE than plain
vector search):
  - If query-variant generation returns zero usable lines (LLM failure,
    empty/malformed output), we log a warning and fall back to a single
    retrieve() call on the original query via the wrapped retriever.
  - The original question is always included alongside the generated
    variants (not replaced by them) - multi-query is a recall BOOST on
    top of normal retrieval, never a substitute for it.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document

from app.llms.base import BaseLLM, LLMGenerationError
from app.prompts.multi_query_prompt import build_multi_query_prompt
from app.retrievers.base import BaseRetriever, RetrieverError
from app.retrievers.utils import deduplicate_documents

logger = logging.getLogger(__name__)


class MultiQueryRetriever(BaseRetriever):
    """Wraps a BaseRetriever and expands the query before retrieving.

    Composition, not inheritance: this class does not implement its own
    document-fetching logic - it delegates every actual retrieve() call
    to the `base_retriever` it wraps (typically a VectorRetriever) and
    only adds the "generate variants, merge results" behavior around it.
    """

    def __init__(
        self,
        llm: BaseLLM,
        base_retriever: BaseRetriever,
        query_count: int,
    ) -> None:
        """
        Args:
            llm: used to generate alternative phrasings of the query.
            base_retriever: the retriever actually queried for each variant
                (and for the fallback path when expansion fails).
            query_count: how many LLM-generated variants to request, in
                addition to the original question. Configured via
                app.config.MULTI_QUERY_COUNT - never hardcoded here.
        """
        self._llm = llm
        self._base_retriever = base_retriever
        self._query_count = query_count

    def retrieve(self, query: str) -> List[Document]:
        """Expand `query` into several variants, retrieve for each, and
        return the deduplicated union of all results.

        The original query's results are placed first, so that when the
        same chunk is retrieved via multiple variants, the copy kept by
        deduplicate_documents() is the one from the original phrasing.
        """
        variants = self._generate_variants(query)

        all_documents: List[Document] = []
        for variant in [query, *variants]:
            try:
                documents = self._base_retriever.retrieve(variant)
            except RetrieverError:
                # One bad variant shouldn't sink the whole expansion -
                # skip it and keep merging results from the others.
                logger.warning(
                    "Retrieval failed for query variant %r - skipping it.",
                    variant,
                )
                continue
            all_documents.extend(documents)

        return deduplicate_documents(all_documents)

    def _generate_variants(self, query: str) -> List[str]:
        """Ask the LLM for `query_count` rephrasings of `query`.

        Returns an empty list (never raises) on any generation or
        parsing failure - retrieve() treats that as "no variants
        available" and falls back to the original query alone.
        """
        prompt = build_multi_query_prompt(question=query, query_count=self._query_count)

        try:
            raw_output = self._llm.generate(prompt)
        except LLMGenerationError:
            logger.warning(
                "Multi-query expansion failed (LLM generation error) - "
                "falling back to the original query only.",
            )
            return []

        variants = [line.strip() for line in raw_output.splitlines() if line.strip()]

        if not variants:
            logger.warning(
                "Multi-query expansion produced no usable variants - "
                "falling back to the original query only.",
            )

        return variants[: self._query_count]