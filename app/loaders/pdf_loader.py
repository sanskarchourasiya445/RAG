"""
app/loaders/pdf_loader.py

Responsibility: turn PDF files on disk into LangChain `Document` objects
with clean, useful metadata. Nothing in this file knows about chunking,
embeddings, or the vector store - it only knows how to read PDFs.

Why loop per-file instead of langchain_community.document_loaders.DirectoryLoader?
1. Per-file error isolation: one corrupted/encrypted PDF should not abort
   ingestion of the other 49 documents.
2. Per-file metadata enrichment (source filename, ingestion timestamp,
   page count) is easier to attach explicitly than to retrofit after a
   bulk load.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.config import DOCUMENTS_DIR, SUPPORTED_FILE_EXTENSIONS

logger = logging.getLogger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF cannot be read at all (corrupted, encrypted, etc).
    Callers decide whether to skip-and-continue or abort."""


@dataclass
class LoadResult:
    """Summary of a full ingestion-folder load, used for the startup
    console report the spec asks for (files loaded, pages loaded, failures)."""

    documents: List[Document]
    files_loaded: List[str]
    files_failed: List[str]
    total_pages: int


def discover_pdf_files(documents_dir: Path = DOCUMENTS_DIR) -> List[Path]:
    """Return every supported file inside the documents folder, sorted for
    deterministic ingestion order (important for reproducible chunk IDs)."""
    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Documents folder does not exist: {documents_dir}. "
            f"Create it and add PDF files before running ingestion."
        )

    files = [
        p
        for p in sorted(documents_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_FILE_EXTENSIONS
    ]
    return files


def load_single_pdf(file_path: Path) -> List[Document]:
    """Load one PDF into a list of per-page Documents with enriched metadata.

    Raises PDFLoadError on any failure so the caller can decide what to do
    (skip and continue, in our case).
    """
    try:
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
    except Exception as exc:  # noqa: BLE001 - we deliberately want to catch
        # ANY failure here (corrupted file, missing pdf backend, permission
        # error) and convert it into our own domain exception, so upstream
        # code only needs to know about PDFLoadError, not every possible
        # pypdf/langchain exception type.
        raise PDFLoadError(f"Failed to load '{file_path.name}': {exc}") from exc

    if not pages:
        raise PDFLoadError(f"'{file_path.name}' produced zero pages (empty or scanned-image PDF?).")

    ingestion_time = _dt.datetime.utcnow().isoformat()
    for page in pages:
        # LangChain's PyPDFLoader already sets metadata["page"] and
        # metadata["source"]; we enrich rather than overwrite.
        page.metadata.update(
            {
                "source_filename": file_path.name,
                "ingested_at_utc": ingestion_time,
            }
        )

    return pages


def load_all_pdfs(documents_dir: Path = DOCUMENTS_DIR) -> LoadResult:
    """Load every PDF in the documents folder. Resilient to individual
    file failures - logs and skips rather than raising, so a single bad
    PDF doesn't block ingestion of everything else."""
    pdf_paths = discover_pdf_files(documents_dir)

    if not pdf_paths:
        logger.warning("No PDF files found in %s", documents_dir)

    all_documents: List[Document] = []
    files_loaded: List[str] = []
    files_failed: List[str] = []

    for path in pdf_paths:
        try:
            logger.info("Loading PDF: %s", path.name)
            pages = load_single_pdf(path)
            all_documents.extend(pages)
            files_loaded.append(path.name)
            logger.info("Loaded %d page(s) from %s", len(pages), path.name)
        except PDFLoadError as exc:
            logger.error("Skipping '%s': %s", path.name, exc)
            files_failed.append(path.name)

    return LoadResult(
        documents=all_documents,
        files_loaded=files_loaded,
        files_failed=files_failed,
        total_pages=len(all_documents),
    )