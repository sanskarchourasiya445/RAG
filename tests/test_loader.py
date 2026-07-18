"""
tests/test_loader.py

Unit tests for app/loaders/pdf_loader.py.

Design note: we don't ship a real PDF fixture into the repo (binary
fixtures rot and bloat git history). Instead we generate a minimal valid
PDF on the fly with reportlab-free raw bytes... but since that's fragile,
we instead test the RESILIENCE behavior (missing folder, corrupted file)
using plain text files disguised with a .pdf extension, which is exactly
the "invalid/corrupted PDF" failure mode the spec asks us to handle -
PyPDFLoader will fail to parse it just as it would a truly corrupted file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.loaders.pdf_loader import (
    PDFLoadError,
    discover_pdf_files,
    load_all_pdfs,
    load_single_pdf,
)


def test_discover_pdf_files_raises_on_missing_folder(tmp_path: Path) -> None:
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        discover_pdf_files(missing_dir)


def test_discover_pdf_files_filters_by_extension(tmp_path: Path) -> None:
    (tmp_path / "doc1.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "notes.txt").write_text("not a pdf")
    (tmp_path / "doc2.pdf").write_bytes(b"%PDF-1.4 fake")

    found = discover_pdf_files(tmp_path)
    names = sorted(p.name for p in found)

    assert names == ["doc1.pdf", "doc2.pdf"]


def test_load_single_pdf_raises_pdfloaderror_on_corrupted_file(tmp_path: Path) -> None:
    corrupted = tmp_path / "corrupted.pdf"
    corrupted.write_text("this is not valid PDF content at all")

    with pytest.raises(PDFLoadError):
        load_single_pdf(corrupted)


def test_load_all_pdfs_skips_corrupted_and_continues(tmp_path: Path) -> None:
    # One corrupted "PDF", zero valid ones - load_all_pdfs must not raise,
    # it should report the failure and return an empty-but-valid result.
    (tmp_path / "bad.pdf").write_text("not a real pdf")

    result = load_all_pdfs(tmp_path)

    assert result.files_failed == ["bad.pdf"]
    assert result.files_loaded == []
    assert result.total_pages == 0


def test_discover_pdf_files_empty_folder_returns_empty_list(tmp_path: Path) -> None:
    assert discover_pdf_files(tmp_path) == []