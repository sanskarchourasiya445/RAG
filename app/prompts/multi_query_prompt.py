"""
app/prompts/multi_query_prompt.py

Responsibility: the single template used to ask the configured LLM for
N alternative phrasings of the user's question, for use by
MultiQueryRetriever (app/retrievers/multi_query.py).

Kept as its own function - mirroring rag_prompt.py's build_rag_prompt -
so the wording can be tuned or tested independently of the retriever
logic that consumes its output.

Why "one per line, no numbering/bullets"?
The retriever parses the LLM's raw text output by splitting on newlines
(see app/retrievers/multi_query.py). Numbered lists, bullet markers, or
commentary would either pollute a variant string (degrading its
embedding quality) or require extra parsing/regex just to strip them
back out - asking for the bare format up front avoids both.
"""

from __future__ import annotations

MULTI_QUERY_PROMPT_TEMPLATE = """You are an assistant that helps a document retrieval system find more relevant results.

Given the question below, generate {query_count} different versions of it. Each version should preserve the original meaning but vary in wording, phrasing, level of detail, or perspective, so that a vector similarity search using these versions is more likely to find relevant passages that the original phrasing alone might miss.

Return ONLY the {query_count} questions, one per line. Do not number them, do not use bullet points, and do not add any explanation or commentary.

Original question: {question}
"""


def build_multi_query_prompt(question: str, query_count: int) -> str:
    """Build the prompt string sent to the LLM to generate query variants."""
    return MULTI_QUERY_PROMPT_TEMPLATE.format(
        query_count=query_count,
        question=question,
    )