"""
app/prompts/rag_prompt.py

Responsibility: the single template used to turn (retrieved context,
user question) into a final prompt string sent to whichever LLM is
configured. Provider-agnostic - lives above the app/llms/ abstraction.

Why an explicit "if unknown, say X" instruction?
LLMs are trained to be maximally helpful, which means their default
behavior when context doesn't contain an answer is to fall back on
general world knowledge and hallucinate something plausible. A grounded
RAG system needs an explicit, literal escape hatch phrase - without one,
"be grounded" instructions alone are unreliable.

NO_ANSWER_MESSAGE is imported (not retyped) from app.config so the
prompt's instruction and the pipeline's fallback-detection logic can
never drift out of sync with each other.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from app.config import NO_ANSWER_MESSAGE

RAG_PROMPT_TEMPLATE = """You are a careful, factual assistant answering questions using ONLY the context provided below.

RULES:
1. Answer using ONLY information found in the context. Do not use outside knowledge.
2. If the context does not contain enough information to answer, respond with EXACTLY this sentence and nothing else:
   "{no_answer_message}"
3. Give a detailed, well-structured explanation using all relevant information from the context.
4. Include definitions, important concepts, explanations, advantages, disadvantages, examples, and applications whenever they are available in the context.
5. Organize the answer with headings and bullet points when appropriate.
6. If multiple retrieved chunks contain different pieces of information, combine them into one complete answer.
7. Do not mention "the context" or "the documents" in your answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


def _format_context(documents: List[Document]) -> str:
    """Turn retrieved chunks into a single context block, tagging each
    chunk with its source and page so the LLM's answer stays traceable
    back to a specific document/page (even though we strip source-talk
    from the final answer text itself per rule #4 above - traceability
    happens at the console/citation layer, in rag_pipeline.py)."""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source_filename", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[Chunk {i} | source: {source} | page: {page}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_rag_prompt(question: str, documents: List[Document]) -> str:
    """Build the final prompt string to send to the LLM."""
    context = _format_context(documents) if documents else "(no relevant context retrieved)"
    return RAG_PROMPT_TEMPLATE.format(
        no_answer_message=NO_ANSWER_MESSAGE,
        context=context,
        question=question,
    )