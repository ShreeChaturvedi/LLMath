"""Integration tests for the agent pipeline.

These tests verify that the components work together correctly.
Most tests use mocks to avoid requiring GPU resources.
"""

from unittest.mock import Mock

import pytest

from llmath.config import LLMathConfig
from llmath.prompts.builder import build_math_prompt
from llmath.prompts.orchestrator import OrchestratorResult, ToolOrchestrator
from llmath.retrieval.base import SearchResult
from llmath.retrieval.theorem_kb import TheoremKB, TheoremSnippet
from llmath.tools.registry import create_default_registry


@pytest.fixture
def mock_retriever():
    """Create a mock retriever for testing."""
    retriever = Mock()
    retriever.ds = Mock()
    retriever.ds.column_names = ["text", "title"]
    retriever.text_field = "text"

    # Mock search results
    retriever.search.return_value = [
        SearchResult(idx=0, score=0.95, text="The product rule states..."),
        SearchResult(idx=1, score=0.85, text="The chain rule states..."),
    ]

    # Mock dataset rows
    retriever.ds.__getitem__ = Mock(
        side_effect=lambda i: {
            "text": f"Full text for entry {i}",
            "title": f"Theorem {i}",
        }
    )

    return retriever


@pytest.fixture
def mock_theorems():
    """Create sample theorem snippets."""
    return [
        TheoremSnippet(
            idx=0,
            score=0.95,
            title="Product Rule",
            snippet="The derivative of fg is f'g + fg'.",
            full_text="The derivative of fg is f'g + fg'. This is fundamental.",
        ),
        TheoremSnippet(
            idx=1,
            score=0.85,
            title="Chain Rule",
            snippet="The derivative of f(g(x)) is f'(g(x)) * g'(x).",
            full_text="The derivative of f(g(x)) is f'(g(x)) * g'(x).",
        ),
    ]


class TestToolRegistry:
    """Integration tests for the tool registry."""

    def test_registry_dispatch_and_execute(self):
        """Test that registry correctly dispatches and executes commands."""
        registry = create_default_registry()

        # Test solve command
        result = registry.execute("solve: x**2 - 1 = 0")
        assert result.success
        assert "[-1, 1]" in result.output

        # Test diff command
        result = registry.execute("diff: x**2")
        assert result.success
        assert "2*x" in result.output

        # Test integrate command
        result = registry.execute("integrate: 2*x")
        assert result.success
        assert "x**2" in result.output

        # Test simplify (default)
        result = registry.execute("(x**2 - 1) / (x - 1)")
        assert result.success
        assert "x + 1" in result.output

    def test_registry_format_result(self):
        """Test result formatting for prompts."""
        registry = create_default_registry()

        formatted = registry.format_result("solve: x**2 - 1")
        assert "solve" in formatted
        assert "[-1, 1]" in formatted


class TestToolOrchestrator:
    """Integration tests for the orchestrator."""

    def test_orchestrator_combines_retrieval_and_tools(self, mock_retriever):
        """Test that orchestrator integrates retrieval and tools."""
        kb = TheoremKB(mock_retriever)
        registry = create_default_registry()
        orchestrator = ToolOrchestrator(kb, registry)

        result = orchestrator.solve_with_tools(
            question="Solve x**2 - 1 = 0",
            k=2,
            sympy_expressions=["solve: x**2 - 1 = 0"],
        )

        assert isinstance(result, OrchestratorResult)
        assert result.question == "Solve x**2 - 1 = 0"
        assert len(result.theorems) == 2
        assert len(result.sympy_context) == 1
        assert "[-1, 1]" in result.sympy_context[0]
        assert len(result.prompt) > 100

    def test_orchestrator_handles_empty_inputs(self, mock_retriever):
        """Test orchestrator with no SymPy expressions."""
        kb = TheoremKB(mock_retriever)
        registry = create_default_registry()
        orchestrator = ToolOrchestrator(kb, registry)

        result = orchestrator.solve_with_tools(
            question="What is continuity?",
            k=2,
            sympy_expressions=[],
        )

        assert result.sympy_context == []
        assert "None." in result.prompt or "Symbolic tool results" in result.prompt


class TestPromptBuilding:
    """Integration tests for prompt construction."""

    def test_full_prompt_structure(self, mock_theorems):
        """Test that complete prompts have correct structure."""
        sympy_ctx = [
            "solve(x**2-1) -> [-1, 1]",
            "diff(x**2, x) -> 2*x",
        ]

        prompt = build_math_prompt(
            question="Find the roots of x**2 - 1",
            theorems=mock_theorems,
            sympy_context=sympy_ctx,
        )

        # Check all sections present
        assert "mathematical assistant" in prompt
        assert "Find the roots" in prompt
        assert "[T1]" in prompt
        assert "[T2]" in prompt
        assert "Product Rule" in prompt
        assert "[S1]" in prompt
        assert "[S2]" in prompt
        assert "[-1, 1]" in prompt
        assert "proof-style" in prompt

    def test_prompt_section_ordering(self, mock_theorems):
        """Test that prompt sections appear in correct order."""
        prompt = build_math_prompt(
            question="Test",
            theorems=mock_theorems,
            sympy_context=["result1"],
        )

        q_pos = prompt.find("Question:")
        t_pos = prompt.find("Retrieved theorems")
        s_pos = prompt.find("Symbolic tool results")
        i_pos = prompt.find("Write a self-contained")

        assert q_pos < t_pos < s_pos < i_pos


class TestEndToEndMocked:
    """End-to-end tests with mocked model."""

    def test_agent_pipeline_mocked(self, mock_retriever):
        """Test full pipeline with mocked model."""
        from llmath.agent.math_agent import AgentResult, MathAgent
        from llmath.prompts.orchestrator import ToolOrchestrator
        from llmath.retrieval.theorem_kb import TheoremKB
        from llmath.tools.registry import create_default_registry

        # Setup components
        kb = TheoremKB(mock_retriever)
        registry = create_default_registry()
        orchestrator = ToolOrchestrator(kb, registry)

        # Mock the model
        mock_model = Mock()
        mock_gen_result = Mock()
        mock_gen_result.text = "This is the mocked answer."
        mock_gen_result.raw_text = "This is the mocked answer."
        mock_gen_result.input_tokens = 100
        mock_gen_result.output_tokens = 50
        mock_model.generate.return_value = mock_gen_result

        # Create agent
        agent = MathAgent(orchestrator, mock_model)

        # Run agent
        result = agent.run(
            question="What is 2+2?",
            sympy_expressions=["simplify: 2+2"],
        )

        assert isinstance(result, AgentResult)
        assert result.question == "What is 2+2?"
        assert result.answer == "This is the mocked answer."
        assert len(result.theorems) == 2
        assert len(result.sympy_context) == 1
        mock_model.generate.assert_called_once()


class TestConfigIntegration:
    """Test configuration integration."""

    def test_config_flows_to_components(self):
        """Test that config values are used by components."""
        config = LLMathConfig()

        # Verify default values
        assert config.agent.default_k == 5
        assert config.generation.max_new_tokens == 512

        # Modify and verify
        config.agent.default_k = 3
        assert config.agent.default_k == 3
