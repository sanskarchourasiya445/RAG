"""
app/embeddings/embedding_model.py

Responsibility: provide a single, cached embedding model instance that
conforms to LangChain's `Embeddings` interface (embed_documents / embed_query).

Why local sentence-transformers instead of a paid API?
- Zero marginal cost per document/query - important for a project that
  ingests potentially hundreds of PHC documents repeatedly during dev.
- No API key required, no network dependency for embeddings specifically
  (only the LLM call needs network + a key).
- BAAI/bge-small-en-v1.5 is a strong small-footprint retrieval model:
  384-dim vectors, fast on CPU, good enough for hackathon/production-lite use.

Tradeoff: quality is good but not frontier-hosted-API quality, and CPU
inference is slower than a hosted embeddings endpoint for very large corpora.
For a district-scale PHC document set (dozens-hundreds of PDFs) this is the
right tradeoff.

Why a singleton/cache instead of instantiating per call?
Loading a sentence-transformers model involves reading model weights from
disk (or downloading them once) - this takes real time. Ingestion embeds
many chunks and query-time embeds one query at a time; both should reuse
the exact same loaded model instance.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_DEVICE, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingModelError(Exception):
    """Raised when the embedding model fails to load (e.g. missing
    dependency, corrupted cache, out-of-memory)."""


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a cached, ready-to-use embedding model.

    lru_cache(maxsize=1) gives us a process-wide singleton without the
    boilerplate of a manual module-level global + None-check. Since this
    function takes no arguments, there's only ever one cache entry.
    """
    try:
        logger.info(
            "Loading embedding model '%s' on device '%s' (first call may "
            "download weights - this can take a minute)...",
            EMBEDDING_MODEL,
            EMBEDDING_DEVICE,
        )
        model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            # normalize_embeddings=True is recommended for bge-* models:
            # they're trained/evaluated with cosine similarity on
            # L2-normalized vectors. Skipping this silently degrades
            # retrieval quality without throwing any error.
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully.")
        return model
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load embedding model: %s", exc)
        raise EmbeddingModelError(
            f"Could not load embedding model '{EMBEDDING_MODEL}': {exc}"
        ) from exc