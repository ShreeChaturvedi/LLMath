"""Tests for the retrieval tool wrapper."""

from llmath.retrieval.theorem_kb import TheoremSnippet
from llmath.tools.retrieval_tool import RetrieveTool


class DummyKB:
    def __init__(self, results):
        self._results = results

    def get_theorems(self, query, k=3):
        return self._results


def test_retrieve_tool_formats_snippets():
    snippets = [
        TheoremSnippet(
            idx=0,
            score=0.9,
            title="Product Rule",
            snippet="d(fg)=f'g+fg'.",
            full_text="d(fg)=f'g+fg'.",
        ),
        TheoremSnippet(
            idx=1,
            score=0.8,
            title="Chain Rule",
            snippet="d(f(g(x)))=f'(g(x))g'(x).",
            full_text="d(f(g(x)))=f'(g(x))g'(x).",
        ),
    ]
    tool = RetrieveTool(DummyKB(snippets), default_k=2)

    result = tool.execute("derivative rules")
    assert result.success is True
    assert "[T1] (idx=0) (score=0.900) Product Rule:" in result.output
    assert "[T2] (idx=1) (score=0.800) Chain Rule:" in result.output


def test_retrieve_tool_handles_empty_results():
    tool = RetrieveTool(DummyKB([]), default_k=2)
    result = tool.execute("no matches")
    assert result.success is True
    assert result.output == "No theorems found."
