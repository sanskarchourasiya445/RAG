"""
app/vectorstores/parent_store.py

Responsibility: persist and retrieve PARENT documents by id, for use by
ParentDocumentRetriever (app/retrievers/parent.py). This is the "parent
half" of parent/child chunking - vector search only ever indexes CHILD
chunks (see app/chunkers/parent_child_chunker.py); this store is where
the larger parent chunk a matching child came from gets looked up.

Why a JSON file per parent document, not a real key-value database?
Consistent with the rest of this project's local-first, dependency-light
approach (Chroma persisted to a local folder, no Docker, no external
services): one JSON file per parent id under PARENT_STORE_DIR is
trivially inspectable, trivially backed up, and needs zero new
dependencies. At Mini-RAG's scale (a handful of PDFs) this is plenty; a
production system with a large, frequently-updated corpus would swap
this for a real KV store (Redis, SQLite, ...) behind this same
save_all()/get() interface - nothing else in the codebase would need to
change.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ParentStoreError(Exception):
    """Raised when a parent document can't be written to or read from disk."""


class ParentDocumentStore:
    """File-backed store: one JSON file per parent document, named
    `<parent_id>.json`, under `directory`."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    def save_all(self, parents: Iterable[Document]) -> None:
        """Persist every parent document, keyed by its own
        metadata["parent_id"] (set by
        app.chunkers.parent_child_chunker.split_into_parent_child).

        Raises:
            ParentStoreError: if any parent is missing "parent_id", or a
                write fails.
        """
        parents = list(parents)
        for parent in parents:
            parent_id = parent.metadata.get("parent_id")
            if not parent_id:
                raise ParentStoreError(
                    "Cannot save a parent document with no 'parent_id' in "
                    "its metadata - was it produced by split_into_parent_child()?"
                )
            payload = {"page_content": parent.page_content, "metadata": parent.metadata}
            try:
                self._path_for(parent_id).write_text(json.dumps(payload), encoding="utf-8")
            except OSError as exc:
                raise ParentStoreError(
                    f"Failed to write parent document '{parent_id}': {exc}"
                ) from exc

        logger.info("Saved %d parent document(s) to %s", len(parents), self._directory)

    def get(self, parent_id: str) -> Optional[Document]:
        """Return the parent document for `parent_id`, or None if it was
        never saved (e.g. store and vector index have drifted out of sync).

        Raises:
            ParentStoreError: if the file exists but can't be read/parsed
                (corrupt data, not a missing-key case).
        """
        path = self._path_for(parent_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ParentStoreError(
                f"Failed to read parent document '{parent_id}': {exc}"
            ) from exc
        return Document(page_content=payload["page_content"], metadata=payload["metadata"])

    def _path_for(self, parent_id: str) -> Path:
        return self._directory / f"{parent_id}.json"