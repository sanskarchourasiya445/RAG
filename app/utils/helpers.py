"""
app/utils/helpers.py

Small, stateless, cross-cutting helpers used by more than one module.
Rule for what belongs here: no domain ownership (no loading/chunking/LLM
logic), used in 2+ unrelated places. Everything else belongs in the
module that owns that responsibility - this file must not become a
dumping ground for pipeline logic.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document


def format_sources_for_console(documents: List[Document]) -> str:
    """Render retrieved chunks as a human-readable summary for console
    output during chat (source filename + page, as required by the spec).
    """
    if not documents:
        return "  (no sources retrieved)"

    lines = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source_filename", "unknown")
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:80].replace("\n", " ").strip()
        lines.append(f"  [{i}] {source} (page {page}) — \"{preview}...\"")
    return "\n".join(lines)


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    """Tiny helper for console messages like '1 file' vs '3 files'."""
    plural = plural or f"{singular}s"
    return f"{count} {singular if count == 1 else plural}"