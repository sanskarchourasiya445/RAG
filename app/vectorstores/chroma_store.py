"""
app/vectorstores/chroma_store.py

Responsibility: the ONLY file in the repo that imports/knows about
ChromaDB directly. Every other module talks to "the vector store" through
the functions defined here. Swapping Chroma for another vector database
(Qdrant, Weaviate, pgvector...) means rewriting this file only.

Startup logic (as specified):
  - If the persisted Chroma collection already has vectors -> load it.
  - Otherwise -> caller must run ingestion to create it.

Why check collection COUNT and not just directory existence?
A Chroma persist directory can exist on disk (e.g. from a previous run
that crashed mid-ingestion) while containing zero actual vectors. Treating
"directory exists" as "safe to skip ingestion" would silently start the
app with an empty, useless retriever. Checking the real document count
avoids that failure mode.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import CHROMA_COLLECTION_NAME, VECTOR_DB_DIR

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised for any failure creating, loading, or writing to the vector store."""


def _open_store(embedding_model: Embeddings) -> Chroma:
    """Open (or lazily create) the on-disk Chroma collection. This alone
    does not guarantee the collection has data - see vector_store_has_data()."""
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(VECTOR_DB_DIR),
    )


def vector_store_exists_with_data(embedding_model: Embeddings) -> bool:
    """Returns True only if a Chroma collection exists AND actually
    contains vectors. This is what startup uses to decide 'load' vs
    'create from scratch'."""
    try:
        store = _open_store(embedding_model)
        count = store._collection.count()  # noqa: SLF001 - Chroma doesn't
        # expose a public count() on the LangChain wrapper as of this
        # writing; this is a deliberate, documented use of a private
        # attribute rather than reimplementing collection introspection.
        logger.info("Existing Chroma collection found with %d vector(s).", count)
        return count > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not inspect existing Chroma collection: %s", exc)
        return False


def create_vector_store(documents: List[Document], embedding_model: Embeddings) -> Chroma:
    """Create a brand-new persisted Chroma collection from chunked documents."""
    if not documents:
        raise VectorStoreError("Cannot create a vector store from an empty document list.")

    try:
        logger.info(
            "Creating new Chroma collection '%s' with %d document(s)...",
            CHROMA_COLLECTION_NAME,
            len(documents),
        )
        store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=str(VECTOR_DB_DIR),
        )
        logger.info("Vector store created and persisted at %s", VECTOR_DB_DIR)
        return store
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to create vector store: %s", exc)
        raise VectorStoreError(f"Failed to create vector store: {exc}") from exc


def load_vector_store(embedding_model: Embeddings) -> Chroma:
    """Load an existing persisted Chroma collection."""
    try:
        store = _open_store(embedding_model)
        logger.info("Loaded existing Chroma collection from %s", VECTOR_DB_DIR)
        return store
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load vector store: %s", exc)
        raise VectorStoreError(f"Failed to load vector store: {exc}") from exc


def get_all_documents(store: Chroma) -> List[Document]:
    """Fetch every chunk currently stored in the collection, reconstructed
    as LangChain Documents.

    Why this exists: sparse/keyword retrieval (BM25, app/retrievers/bm25.py)
    needs the full corpus text up front to build its index - unlike vector
    search, there's no equivalent of "similarity search a subset of the
    store". This is the one place that knows HOW to pull that corpus back
    out of Chroma (via the underlying collection's .get()), so BM25Retriever
    itself stays vector-store-agnostic.
    """
    try:
        raw = store.get(include=["documents", "metadatas"])
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(
            f"Failed to fetch documents from vector store: {exc}"
        ) from exc

    texts = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    return [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(texts, metadatas)
    ]


def get_or_create_vector_store(
    embedding_model: Embeddings,
    documents_provider: Optional[callable] = None,
) -> Chroma:
    """
    The single entry point the rest of the app should call.

    documents_provider: a zero-arg callable that returns List[Document] to
    embed, invoked ONLY if no existing collection is found. Passed as a
    callable (not eagerly-computed documents) so that when a collection
    already exists, we never pay the cost of loading+chunking PDFs at all.
    """
    if vector_store_exists_with_data(embedding_model):
        return load_vector_store(embedding_model)

    if documents_provider is None:
        raise VectorStoreError(
            "No existing vector store found, and no documents_provider was "
            "given to build one from scratch."
        )

    documents = documents_provider()
    return create_vector_store(documents, embedding_model)