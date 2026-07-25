"""
tests/test_multi_query_retriever.py

Unit tests for app/retrievers/multi_query.py.

Uses fakes for both BaseLLM and BaseRetriever so these tests exercise
ONLY the expansion/merge logic - no real LLM call, no real vector store.
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.llms.base import BaseLLM, LLMGenerationError
from app.retrievers.base import BaseRetriever, RetrieverError
from app.retrievers.multi_query import MultiQueryRetriever


class _FakeLLM(BaseLLM):
    """Returns a fixed multi-line string of variants, or raises/returns
    garbage, depending on what the test wants to exercise."""

    provider_name = "fake"

    def __init__(self, response: str | None = None, raises: bool = False) -> None:
        self._response = response
        self._raises = raises
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        if self._raises:
            raise LLMGenerationError("fake LLM failure")
        return self._response or ""


class _FakeBaseRetriever(BaseRetriever):
    """Returns documents keyed by query string; raises for queries not
    in its lookup table, to simulate a partial failure."""

    def __init__(self, documents_by_query: dict[str, list[Document]]) -> None:
        self._documents_by_query = documents_by_query
        self.queries_received: list[str] = []

    def retrieve(self, query: str) -> list[Document]:
        self.queries_received.append(query)
        if query not in self._documents_by_query:
            raise RetrieverError(f"no fake result configured for query: {query!r}")
        return self._documents_by_query[query]


def _doc(text: str, source: str = "a.pdf", page: int = 0) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


def test_multi_query_retriever_queries_original_plus_generated_variants() -> None:
    llm = _FakeLLM(response="variant one\nvariant two")
    base = _FakeBaseRetriever(
        {
            "original question": [_doc("chunk A")],
            "variant one": [_doc("chunk B")],
            "variant two": [_doc("chunk C")],
        }
    )
    retriever = MultiQueryRetriever(llm=llm, base_retriever=base, query_count=2)

    retriever.retrieve("original question")

    assert base.queries_received == ["original question", "variant one", "variant two"]


def test_multi_query_retriever_merges_and_deduplicates_results() -> None:
    llm = _FakeLLM(response="variant one")
    shared_chunk = _doc("same chunk", source="a.pdf", page=1)
    base = _FakeBaseRetriever(
        {
            "original question": [shared_chunk],
            "variant one": [shared_chunk, _doc("unique chunk", source="a.pdf", page=2)],
        }
    )
    retriever = MultiQueryRetriever(llm=llm, base_retriever=base, query_count=1)

    result = retriever.retrieve("original question")

    assert len(result) == 2  # shared_chunk counted once, plus the unique one


def test_multi_query_retriever_respects_query_count_cap() -> None:
    llm = _FakeLLM(response="v1\nv2\nv3\nv4\nv5")
    base = _FakeBaseRetriever({"q": [], "v1": [], "v2": []})
    retriever = MultiQueryRetriever(llm=llm, base_retriever=base, query_count=2)

    retriever.retrieve("q")

    # original + only the first 2 variants, even though the LLM returned 5
    assert base.queries_received == ["q", "v1", "v2"]


def test_multi_query_retriever_falls_back_to_original_query_on_llm_failure() -> None:
    llm = _FakeLLM(raises=True)
    base = _FakeBaseRetriever({"q": [_doc("only result")]})
    retriever = MultiQueryRetriever(llm=llm, base_retriever=base, query_count=3)

    result = retriever.retrieve("q")

    assert base.queries_received == ["q"]
    assert len(result) == 1


def test_multi_query_retriever_falls_back_on_empty_llm_output() -> None:
    llm = _FakeLLM(response="   \n   ")  # whitespace-only lines
    base = _FakeBaseRetriever({"q": [_doc("only result")]})
    retriever = MultiQueryRetriever(llm=llm, base_retriever=base, query_count=3)

    result = retriever.retrieve("q")

    assert base.queries_received == ["q"]
    assert len(result) == 1


def test_multi_query_retriever_skips_variant_that_fails_but_keeps_others() -> None:
    llm = _FakeLLM(response="good variant\nbad variant")
    base = _FakeBaseRetriever(
        {
            "q": [_doc("original result", page=0)],
            "good variant": [_doc("good result", page=1)],
            # "bad variant" deliberately NOT in the lookup table -> raises RetrieverError
        }
    )
    retriever = MultiQueryRetriever(llm=llm, base_retriever=base, query_count=2)

    result = retriever.retrieve("q")

    assert len(result) == 2  # original + good variant; bad variant's failure was swallowed


def test_multi_query_retriever_implements_base_retriever_contract() -> None:
    retriever = MultiQueryRetriever(
        llm=_FakeLLM(response=""),
        base_retriever=_FakeBaseRetriever({}),
        query_count=1,
    )
    assert isinstance(retriever, BaseRetriever)