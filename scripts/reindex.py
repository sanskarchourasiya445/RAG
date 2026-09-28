"""
scripts/reindex.py

Explicit reindex command: clears the local vector index and rebuilds it
from scratch by running the full ingestion pipeline.

Usage
-----
    python scripts/reindex.py          # prompts for confirmation
    python scripts/reindex.py --yes    # skips confirmation (CI / scripting)

What this does
--------------
1. Deletes the local Chroma vector index (data/chroma/).
2. Deletes the local parent-document store (data/parent_store/), if present.
3. Re-creates both directories.
4. Runs the full ingestion pipeline (load PDFs → chunk → embed → store).
5. Reports the result.

What this does NOT do
---------------------
- Does NOT touch documents/ — your source PDFs are safe.
- Does NOT commit or push anything to Git.
- Does NOT modify app/config.py or any source code.

When to use this
----------------
- After changing CHUNK_SIZE, CHUNK_OVERLAP, or EMBEDDING_MODEL in config.
- After adding or removing PDFs from documents/.
- After an interrupted ingestion left a partial/corrupt index.
- Whenever you want a guaranteed-clean rebuild.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys

# Ensure the project root is importable when called as a script.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    DOCUMENTS_DIR,
    PARENT_STORE_DIR,
    VECTOR_DB_DIR,
    ensure_dirs,
)
from app.embeddings.embedding_model import get_embedding_model
from app.loaders.pdf_loader import load_all_pdfs
from app.vectorstores.chroma_store import VectorStoreError, create_vector_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("reindex")


# ---------------------------------------------------------------------------
# Internal helpers — kept small; all heavy logic lives in the application
# ---------------------------------------------------------------------------

def _confirm_interactive() -> bool:
    """Prompt the user and return True only if they explicitly confirm."""
    print()
    print("=" * 60)
    print("  REINDEX — This will delete your local vector index")
    print("=" * 60)
    print(f"  Vector store : {VECTOR_DB_DIR}")
    print(f"  Parent store : {PARENT_STORE_DIR}")
    print(f"  Source docs  : {DOCUMENTS_DIR}  ← NOT deleted")
    print("=" * 60)
    print()
    answer = input("  Type 'yes' to continue, anything else to cancel: ").strip().lower()
    return answer == "yes"


def _wipe_runtime_indexes() -> None:
    """Delete the vector store and parent store directories."""
    for path in (VECTOR_DB_DIR, PARENT_STORE_DIR):
        if path.exists():
            logger.info("Removing %s ...", path)
            shutil.rmtree(path)
        else:
            logger.info("Directory does not exist, skipping: %s", path)


def _run_ingestion() -> int:
    """Load PDFs, chunk, embed, and write to Chroma. Returns chunk count."""
    from app.config import (
        CHILD_CHUNK_OVERLAP,
        CHILD_CHUNK_SIZE,
        PARENT_CHUNK_OVERLAP,
        PARENT_CHUNK_SIZE,
        RETRIEVER_TYPE,
    )

    result = load_all_pdfs(DOCUMENTS_DIR)

    if result.total_pages == 0:
        logger.error(
            "No pages were loaded from any PDF in %s. "
            "Add PDF files to that directory and re-run reindex.",
            DOCUMENTS_DIR,
        )
        return 0

    logger.info(
        "Loaded %d page(s) from %d file(s).",
        result.total_pages,
        len(result.files_loaded),
    )

    embedding_model = get_embedding_model()

    if RETRIEVER_TYPE == "parent":
        from app.chunkers.parent_child_chunker import split_into_parent_child
        from app.vectorstores.parent_store import ParentDocumentStore

        parents, children = split_into_parent_child(
            result.documents,
            parent_chunk_size=PARENT_CHUNK_SIZE,
            parent_chunk_overlap=PARENT_CHUNK_OVERLAP,
            child_chunk_size=CHILD_CHUNK_SIZE,
            child_chunk_overlap=CHILD_CHUNK_OVERLAP,
        )
        parent_store = ParentDocumentStore(PARENT_STORE_DIR)
        parent_store.save_all(parents)
        logger.info("Saved %d parent chunk(s) to %s.", len(parents), PARENT_STORE_DIR)
        create_vector_store(children, embedding_model)
        return len(children)

    from app.chunkers.recursive_chunker import chunk_documents

    chunks = chunk_documents(result.documents)
    create_vector_store(chunks, embedding_model)
    return len(chunks)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Delete and rebuild the local RAG vector index from source PDFs. "
            "Source documents in documents/ are never deleted."
        )
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt (useful in scripts/CI).",
    )
    args = parser.parse_args()

    if not args.yes:
        if not _confirm_interactive():
            print("\n  Cancelled — no changes were made.")
            sys.exit(0)

    print()
    logger.info("Starting reindex...")

    _wipe_runtime_indexes()
    ensure_dirs()

    try:
        chunk_count = _run_ingestion()
    except VectorStoreError as exc:
        logger.error("Ingestion failed: %s", exc)
        sys.exit(1)

    if chunk_count == 0:
        logger.warning("Reindex completed but no chunks were written.")
        sys.exit(1)

    print()
    print("=" * 60)
    print(f"  Reindex complete — {chunk_count} chunk(s) written to index.")
    print(f"  Vector store: {VECTOR_DB_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
