"""Tests for the prompt builder."""

import pytest

from llmath.prompts.builder import build_baseline_prompt, build_math_prompt
from llmath.retrieval.theorem_kb import TheoremSnippet


@pytest.fixture
def sample_theorems():
    """Create sample theorem snippets for testing."""
    return [
        TheoremSnippet(
            idx=0,
            score=0.95,
            title="Product Rule",
            snippet="The derivative of fg is f'g + fg'.",
            full_text="The derivative of fg is f'g + fg'. This is a fundamental rule.",
        ),
        TheoremSnippet(
            idx=1,
            score=0.85,
            title="Chain Rule",
            snippet="The derivative of f(g(x)) is f'(g(x)) * g'(x).",
            full_text="The derivative of f(g(x)) is f'(g(x)) * g'(x).",
        ),
    ]


class TestBuildMathPrompt:
    """Tests for build_math_prompt function."""

    def test_includes_question(self):
        question = "What is the derivative of x^2?"
        prompt = build_math_prompt(question, [])
        assert question in prompt

    def test_includes_system_header(self):
        prompt = build_math_prompt("Test question", [])
        assert "mathematical assistant" in prompt

    def test_includes_answer_instructions(self):
        prompt = build_math_prompt("Test question", [])
        assert "proof-style solution" in prompt

    def test_formats_theorems_with_labels(self, sample_theorems):
        prompt = build_math_prompt("Test", sample_theorems)
        assert "[T1]" in prompt
        assert "[T2]" in prompt
        assert "Product Rule" in prompt
        assert "Chain Rule" in prompt

    def test_includes_theorem_snippets(self, sample_theorems):
        prompt = build_math_prompt("Test", sample_theorems)
        assert "f'g + fg'" in prompt

    def test_formats_sympy_context(self, sample_theorems):
        sympy_ctx = [
            "diff(x**2, x) -> 2*x",
            "solve(x**2 - 1) -> [-1, 1]",
        ]
        prompt = build_math_prompt("Test", sample_theorems, sympy_ctx)
        assert "[S1]" in prompt
        assert "[S2]" in prompt
        assert "2*x" in prompt
        assert "[-1, 1]" in prompt

    def test_handles_empty_theorems(self):
        prompt = build_math_prompt("Test question", [])
        assert "None." in prompt

    def test_handles_empty_sympy(self, sample_theorems):
        prompt = build_math_prompt("Test", sample_theorems, [])
        assert "Symbolic tool results (SymPy):" in prompt
        assert "None." in prompt

    def test_handles_none_sympy(self, sample_theorems):
        prompt = build_math_prompt("Test", sample_theorems, None)
        assert "Symbolic tool results (SymPy):" in prompt


class TestBuildBaselinePrompt:
    """Tests for build_baseline_prompt function."""

    def test_includes_question(self):
        question = "What is 2 + 2?"
        prompt = build_baseline_prompt(question)
        assert question in prompt

    def test_mentions_no_tools(self):
        prompt = build_baseline_prompt("Test")
        assert "DO NOT reference any external tools" in prompt

    def test_no_theorem_references(self):
        prompt = build_baseline_prompt("Test")
        assert "[T" not in prompt
        assert "[S" not in prompt


class TestPromptStructure:
    """Tests for overall prompt structure."""

    def test_prompt_order(self, sample_theorems):
        sympy_ctx = ["simplify(x + x) -> 2*x"]
        prompt = build_math_prompt("Test", sample_theorems, sympy_ctx)

        # Check that sections appear in correct order
        question_pos = prompt.find("Question:")
        theorems_pos = prompt.find("Retrieved theorems")
        sympy_pos = prompt.find("Symbolic tool results")
        instructions_pos = prompt.find("Write a self-contained")

        assert question_pos < theorems_pos
        assert theorems_pos < sympy_pos
        assert sympy_pos < instructions_pos

    def test_prompt_not_empty(self, sample_theorems):
        prompt = build_math_prompt("Test", sample_theorems)
        assert len(prompt) > 100  # Should have substantial content
