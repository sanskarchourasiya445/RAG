"""
Thin console entrypoint. Contains ZERO business logic - it only:
  1. Initializes the pipeline and prints the startup report.
  2. Runs a REPL loop that calls pipeline.query() and renders results.

Why keep this deliberately thin?
Every meaningful capability here already exists on RAGPipeline. That
means when you later wrap this same pipeline in a FastAPI route
(POST /query) or a React-facing API, you import RAGPipeline and call
the same two methods - you are not rewriting business logic that got
tangled into a console script.
"""

from __future__ import annotations

import logging
import sys

from app.config import (
    EMBEDDING_MODEL,
    FETCH_K,
    LAMBDA_MULT,
    LLM_PROVIDER,
    RETRIEVER_TYPE,
    TOP_K,
)
from app.pipeline.rag_pipeline import IngestionReport, PipelineError, RAGPipeline
from app.utils.helpers import format_sources_for_console, pluralize

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def print_startup_report(report: IngestionReport) -> None:
    print("\n" + "=" * 60)
    print("MINI-RAG - STARTUP REPORT")
    print("=" * 60)

    if report.loaded_from_existing_store:
        print("Vector database status : loaded existing collection")
    else:
        print(f"PDFs loaded             : {pluralize(len(report.files_loaded), 'file')}")
        if report.files_loaded:
            for f in report.files_loaded:
                print(f"   - {f}")
        if report.files_failed:
            print(f"PDFs failed              : {pluralize(len(report.files_failed), 'file')}")
            for f in report.files_failed:
                print(f"   - {f} (see logs for details)")
        print(f"Pages loaded             : {report.total_pages}")
        print(f"Chunks created           : {report.total_chunks}")
        print("Vector database status  : created and persisted")

    print(f"Embedding model          : {EMBEDDING_MODEL}")
    print(f"Retriever                : {RETRIEVER_TYPE} "
          f"(top_k={TOP_K}, fetch_k={FETCH_K}, lambda={LAMBDA_MULT})")
    print(f"Selected LLM provider    : {LLM_PROVIDER}")
    print("Retriever initialized    : yes")
    print("=" * 60)
    print("Mini-RAG is ready. Type your question, or 'exit' to quit.")
    print("=" * 60 + "\n")


def run_repl(pipeline: RAGPipeline) -> None:
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Mini-RAG. Goodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Exiting Mini-RAG. Goodbye!")
            break

        try:
            result = pipeline.query(question)
        except PipelineError as exc:
            logger.error("Query failed: %s", exc)
            print(f"\n[Error] {exc}\n")
            continue

        print("\nRetrieved sources:")
        print(format_sources_for_console(result.sources))
        print(f"\nAnswer: {result.answer}")
        print(
            f"\n(generation time: {result.generation_seconds:.2f}s | "
            f"total query time: {result.total_seconds:.2f}s)\n"
        )


def main() -> int:
    pipeline = RAGPipeline()
    try:
        report = pipeline.initialize()
    except PipelineError as exc:
        logger.error("Pipeline initialization failed: %s", exc)
        print(f"\n[Fatal] Mini-RAG failed to start: {exc}\n")
        return 1

    print_startup_report(report)
    run_repl(pipeline)
    return 0


if __name__ == "__main__":
    sys.exit(main())