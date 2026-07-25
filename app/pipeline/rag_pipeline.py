"""
Responsibility: ORCHESTRATION ONLY. This is the one file allowed to know
about every other module (loaders, chunkers, embeddings, vector store,
retriever, LLM, prompt). main.py should never import any of those
directly - it only ever talks to RAGPipeline.

Why a class instead of a bag of module-level functions?
The pipeline holds STATE that is expensive to (re)create: the loaded
embedding model, the open vector store connection, the configured
retriever, the instantiated LLM client. Building these once at startup
and reusing them per-query is both a performance requirement and a
correctness one (re-checking "does the collection exist" on every
question would be wasteful and confusing).

Two-layer grounding guard (deliberate, not redundant):
  1. If retrieval returns literally zero chunks, short-circuit BEFORE
     calling the LLM at all - saves a network call/cost and is an
     unambiguous "nothing to answer from" case.
  2. If chunks come back but are low-relevance, we still let the LLM's
     grounded prompt (rag_prompt.py) make the final call, because
     "looks irrelevant" is a fuzzy judgment call best made by the model
     that can actually read the content, not by a hardcoded similarity
     threshold that would need constant re-tuning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

from app.chunkers.parent_child_chunker import split_into_parent_child
from app.chunkers.recursive_chunker import chunk_documents
from app.config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    DOCUMENTS_DIR,
    NO_ANSWER_MESSAGE,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PARENT_STORE_DIR,
    RETRIEVER_TYPE,
    ensure_dirs,
)
from app.embeddings.embedding_model import get_embedding_model
from app.llms.base import BaseLLM, LLMGenerationError
from app.llms.llm_factory import get_llm
from app.loaders.pdf_loader import LoadResult, load_all_pdfs
from app.prompts.rag_prompt import build_rag_prompt
from app.retrievers.base import BaseRetriever
from app.retrievers.factory import RetrieverFactory
from app.utils.timer import Timer
from app.vectorstores.chroma_store import (
    VectorStoreError,
    get_or_create_vector_store,
)
from app.vectorstores.parent_store import ParentDocumentStore

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Top-level exception for pipeline initialization/query failures.
    Wraps lower-level exceptions (VectorStoreError, LLMGenerationError,
    PDFLoadError) so main.py has one exception type to catch at the
    console boundary."""


@dataclass
class QueryResult:
    """Everything the console layer (main.py) needs to render the chat
    experience the spec asks for: answer, sources, and timings."""

    answer: str
    sources: List[Document]
    generation_seconds: float
    total_seconds: float
    was_grounded: bool  # False if we short-circuited on zero retrieved chunks


@dataclass
class IngestionReport:
    """Startup console report data (files loaded, pages, chunks, etc)."""

    files_loaded: List[str] = field(default_factory=list)
    files_failed: List[str] = field(default_factory=list)
    total_pages: int = 0
    total_chunks: int = 0
    loaded_from_existing_store: bool = False


class RAGPipeline:
    """Owns the full lifecycle: startup (ingest-or-load) and per-query
    (retrieve -> prompt -> generate)."""

    def __init__(self) -> None:
        self.llm: BaseLLM
        self._retriever: BaseRetriever | None = None
        self.ingestion_report = IngestionReport()

    def initialize(self) -> IngestionReport:
        """Run once at application startup. Loads the embedding model,
        loads-or-creates the vector store, builds the retriever, and
        instantiates the configured LLM. Returns a report for console
        display."""
        logger.info("Initializing RAG pipeline...")
        ensure_dirs()

        embedding_model = get_embedding_model()

        # Only used for retriever_type="parent" - built unconditionally
        # (cheap: just ensures a directory exists) whenever that strategy
        # is configured, since it's needed whether we're ingesting fresh
        # documents below OR loading an already-populated vector store
        # (the parent chunks living on disk here are what make either
        # path work).
        parent_store: ParentDocumentStore | None = None
        if RETRIEVER_TYPE == "parent":
            parent_store = ParentDocumentStore(PARENT_STORE_DIR)

        # We only actually load/chunk PDFs if no existing vector store is
        # found - vector_store module handles that check. We pass a
        # closure so the (potentially expensive) load+chunk work is
        # skipped entirely when a persisted collection already exists.
        load_result_holder: dict[str, LoadResult] = {}

        def _load_and_chunk() -> List[Document]:
            result = load_all_pdfs(DOCUMENTS_DIR)
            load_result_holder["result"] = result
            if result.total_pages == 0:
                raise PipelineError(
                    "No pages were loaded from any PDF - cannot build a "
                    "vector store from zero documents. Add PDFs to the "
                    "documents folder and restart."
                )

            if RETRIEVER_TYPE == "parent":
                # The vector store must be populated with CHILD chunks for
                # this strategy - the PARENT chunks are persisted separately
                # (parent_store), never embedded/indexed themselves. See
                # app/retrievers/parent.py for how a child match is mapped
                # back to its parent at query time.
                parents, children = split_into_parent_child(
                    result.documents,
                    parent_chunk_size=PARENT_CHUNK_SIZE,
                    parent_chunk_overlap=PARENT_CHUNK_OVERLAP,
                    child_chunk_size=CHILD_CHUNK_SIZE,
                    child_chunk_overlap=CHILD_CHUNK_OVERLAP,
                )
                assert parent_store is not None  # guaranteed by the RETRIEVER_TYPE check above
                parent_store.save_all(parents)
                self.ingestion_report.total_chunks = len(children)
                return children

            chunks = chunk_documents(result.documents)
            self.ingestion_report.total_chunks = len(chunks)
            return chunks

        try:
            vector_store = get_or_create_vector_store(
                embedding_model=embedding_model,
                documents_provider=_load_and_chunk,
            )
        except VectorStoreError as exc:
            raise PipelineError(f"Vector store initialization failed: {exc}") from exc

        if "result" in load_result_holder:
            result = load_result_holder["result"]
            self.ingestion_report.files_loaded = result.files_loaded
            self.ingestion_report.files_failed = result.files_failed
            self.ingestion_report.total_pages = result.total_pages
            self.ingestion_report.loaded_from_existing_store = False
        else:
            self.ingestion_report.loaded_from_existing_store = True

        # LLM is built BEFORE the retriever: some strategies (e.g.
        # "multi_query") need an already-instantiated BaseLLM to generate
        # query variants, and the factory takes it as a plain argument
        # rather than constructing its own - one LLM instance per
        # pipeline, shared by both generation and retrieval.
        try:
            self.llm = get_llm()
        except LLMGenerationError as exc:
            raise PipelineError(f"LLM initialization failed: {exc}") from exc

        self._retriever = RetrieverFactory.create(
            vector_store, llm=self.llm, parent_store=parent_store
        )

        logger.info("RAG pipeline initialized successfully.")
        return self.ingestion_report

    def query(self, question: str) -> QueryResult:
        """Answer a single question: retrieve -> prompt -> generate."""
        if self._retriever is None:
            raise PipelineError("Pipeline.initialize() must be called before query().")

        with Timer() as total_timer:
            logger.info("Retrieving context for question: %r", question)
            documents = self._retriever.retrieve(question)

            # Layer 1 guard: zero retrieved chunks -> short-circuit, no LLM call.
            if not documents:
                logger.warning("No documents retrieved for question: %r", question)
                return QueryResult(
                    answer=NO_ANSWER_MESSAGE,
                    sources=[],
                    generation_seconds=0.0,
                    total_seconds=total_timer.elapsed_seconds,
                    was_grounded=False,
                )

            prompt = build_rag_prompt(question=question, documents=documents)

            with Timer() as gen_timer:
                try:
                    answer = self.llm.generate(prompt)
                except LLMGenerationError as exc:
                    raise PipelineError(f"Answer generation failed: {exc}") from exc

        return QueryResult(
            answer=answer.strip(),
            sources=documents,
            generation_seconds=gen_timer.elapsed_seconds,
            total_seconds=total_timer.elapsed_seconds,
            was_grounded=True,
        )