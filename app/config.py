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
# Retriever (MMR - Maximal Marginal Relevance)
# ---------------------------------------------------------------------
RETRIEVER_TYPE = "mmr"
TOP_K = 4          # how many chunks to return
FETCH_K = 20       # how many candidates MMR picks from before diversifying
LAMBDA_MULT = 0.5  # 1.0 = pure relevance, 0.0 = pure diversity

# ---------------------------------------------------------------------
# LLM provider selection
# ---------------------------------------------------------------------
# One of: "mistral", "gemini", "grok"
LLM_PROVIDER = "mistral"

# Default model per provider. Each provider file imports its own constant
# below - the API key is loaded separately (see app/llms/<provider>.py).
MISTRAL_MODEL = "mistral-large-latest"
GEMINI_MODEL = "gemini-1.5-flash"
GROK_MODEL = "grok-beta"

# ---------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------
NO_ANSWER_MESSAGE = "I could not find the answer in the supplied documents."