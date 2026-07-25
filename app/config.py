"""
app/config.py

Single source of configuration for Mini-RAG.

This is a plain Python module, not a validated settings class - for a
learning project, a flat file of constants is easier to read top to
bottom and easier to change (just edit a value and re-run) than a
pydantic-settings layer with its own indirection.

Note: LLM API keys are NOT stored here. Each provider file in app/llms/
(mistral.py, gemini.py, grok.py) loads its own key directly from the
environment via python-dotenv. Keys are secrets and shouldn't sit next
to plain, non-secret settings like chunk size - so they're kept local
to the file that actually uses them.

To change any behavior of Mini-RAG (chunk size, retriever settings,
which LLM provider to use, etc.), edit the values below.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
# PROJECT_ROOT = the mini-rag/ folder itself (one level up from app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = PROJECT_ROOT / "documents"
VECTOR_DB_DIR = PROJECT_ROOT / "chroma_db"

SUPPORTED_FILE_EXTENSIONS = (".pdf",)


def ensure_dirs() -> None:
    """Create the documents/ and chroma_db/ folders if they don't exist yet.
    Called once at startup so ingestion never fails on a missing folder."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# ---------------------------------------------------------------------
# Embeddings (local, free - no API key needed)
# ---------------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DEVICE = "cpu"  # "cpu" or "cuda"

# ---------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------
CHROMA_COLLECTION_NAME = "mini_rag_collection"

# ---------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------
# Which retrieval strategy RetrieverFactory builds. One of: "vector",
# "multi_query", "bm25", "hybrid", "parent" (see app/retrievers/factory.py
# for the full mapping).
RETRIEVER_TYPE = "hybrid"

# MMR (Maximal Marginal Relevance) settings - used by every strategy,
# since each one is built on top of a VectorRetriever.
TOP_K = 5          # how many chunks to return
FETCH_K = 20       # how many candidates MMR picks from before diversifying
LAMBDA_MULT = 0.5  # 1.0 = pure relevance, 0.0 = pure diversity

# multi_query strategy: how many LLM-generated rephrasings of the
# question to retrieve with, IN ADDITION to the original question.
MULTI_QUERY_COUNT = 3

# hybrid strategy: RRF weights for combining vector search + BM25.
# Must not both be 0 - see HybridRetriever for the fusion logic.
HYBRID_VECTOR_WEIGHT = 0.7
HYBRID_BM25_WEIGHT = 0.3

# ---------------------------------------------------------------------
# Reranking (cross-encoder) - optional second stage applied AFTER
# whichever RETRIEVER_TYPE above runs. Toggle independent of retriever
# strategy: any of vector/multi_query/bm25/hybrid can be reranked.
# ---------------------------------------------------------------------
ENABLE_RERANKER = True
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATE_K = 10  # candidates the base strategy fetches before reranking
FINAL_K = 5              # how many documents survive reranking, sent to the LLM

# ---------------------------------------------------------------------
# Contextual Compression - optional final stage, applied AFTER retrieval
# and (if enabled) reranking. Compresses each surviving document down to
# only the excerpt relevant to the question, before it reaches the
# answer-generation LLM.
# ---------------------------------------------------------------------
ENABLE_COMPRESSION = True
NO_RELEVANT_CONTENT_MESSAGE = "NO_RELEVANT_CONTENT"

# ---------------------------------------------------------------------
# Parent Document Retriever ("parent" strategy) - indexes small CHILD
# chunks for precise vector search, but returns the larger PARENT chunk
# each match belongs to, for more context per answer. Only used when
# RETRIEVER_TYPE = "parent" - every other strategy uses CHUNK_SIZE /
# CHUNK_OVERLAP above instead.
# ---------------------------------------------------------------------
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50
PARENT_STORE_DIR = PROJECT_ROOT / "parent_store"

# ---------------------------------------------------------------------
# LLM provider selection
# ---------------------------------------------------------------------
# One of: "mistral", "gemini", "groq"
LLM_PROVIDER = "groq"

# Default model per provider. Each provider file imports its own constant
# below - the API key is loaded separately (see app/llms/<provider>.py).
MISTRAL_MODEL = "mistral-large-latest"
GEMINI_MODEL = "gemini-1.5-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"
# ---------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------
NO_ANSWER_MESSAGE = "I could not find the answer in the supplied documents."