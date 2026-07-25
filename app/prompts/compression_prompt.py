"""
app/prompts/compression_prompt.py

Responsibility: the single template used to ask the configured LLM to
extract only the query-relevant portion of a single retrieved document,
for use by CompressionRetriever (app/retrievers/compression.py).

Kept as its own function - mirroring rag_prompt.py's build_rag_prompt and
multi_query_prompt.py's build_multi_query_prompt - so the wording can be
tuned or tested independently of the retriever logic that consumes it.

Why "copy verbatim, don't summarize"?
Compression's job is to REMOVE irrelevant sentences, not to rewrite the
relevant ones. If the LLM paraphrased instead of extracting, the final
answer-generation prompt could end up citing/trusting content that
subtly drifted from the source chunk's actual wording - defeating the
whole point of grounding answers in retrieved text.

Why a literal sentinel string for "nothing relevant"?
Same reasoning as NO_ANSWER_MESSAGE in rag_prompt.py: an LLM asked to
"extract the relevant part" will often produce SOME text even when
nothing in the document actually answers the question, unless given an
explicit, literal escape hatch to signal "nothing here." The sentinel
(NO_RELEVANT_CONTENT_MESSAGE) lives in app/config.py - both this prompt
and CompressionRetriever's detection logic import the same constant, so
instruction and detection can't drift out of sync with each other.
"""

from __future__ import annotations

from app.config import NO_RELEVANT_CONTENT_MESSAGE

COMPRESSION_PROMPT_TEMPLATE = """Given the question and a piece of text below, extract ONLY the sentences or phrases from the text that are directly relevant to answering the question.

RULES:
1. Copy the relevant portions VERBATIM - do not summarize, paraphrase, or add any information not present in the text.
2. If NONE of the text is relevant to the question, respond with EXACTLY this and nothing else:
   "{no_relevant_content_message}"

QUESTION:
{question}

TEXT:
{document_text}

RELEVANT EXCERPT:
"""


def build_compression_prompt(question: str, document_text: str) -> str:
    """Build the prompt string sent to the LLM to compress one document."""
    return COMPRESSION_PROMPT_TEMPLATE.format(
        no_relevant_content_message=NO_RELEVANT_CONTENT_MESSAGE,
        question=question,
        document_text=document_text,
    )