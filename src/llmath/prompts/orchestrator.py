"""Tool orchestrator for the math agent.

Coordinates retrieval and symbolic tools to build complete prompts.
"""

import logging
from dataclasses import dataclass

from ..retrieval.theorem_kb import TheoremKB, TheoremSnippet
from ..tools.registry import ToolRegistry
from .builder import build_math_prompt

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Result from the orchestrator.

    Contains all intermediate results from retrieval and tool execution,
    plus the final constructed prompt.

    Attributes:
        question: The original question.
        theorems: Retrieved theorem snippets.
        sympy_context: Formatted tool results.
        prompt: The complete prompt ready for the LLM.
    """

    question: str
    theorems: list[TheoremSnippet]
    sympy_context: list[str]
    prompt: str


class ToolOrchestrator:
    """Orchestrates retrieval and tool execution to build prompts.

    Combines the theorem knowledge base with the tool registry to
    process questions end-to-end before sending to the LLM.

    Attributes:
        kb: TheoremKB for retrieval.
        tools: ToolRegistry for symbolic computation.
    """

    def __init__(self, kb: TheoremKB, tool_registry: ToolRegistry) -> None:
        """Initialize the orchestrator.

        Args:
            kb: Theorem knowledge base for retrieval.
            tool_registry: Registry of available tools.
        """
        self.kb = kb
        self.tools = tool_registry

    def solve_with_tools(
        self,
        question: str,
        k: int = 5,
        sympy_expressions: list[str] | None = None,
    ) -> OrchestratorResult:
        """Orchestrate retrieval and symbolic tools.

        Args:
            question: The math question to answer.
            k: Number of theorems to retrieve.
            sympy_expressions: List of tool commands to execute
                (e.g., ["solve: x**2-1", "diff: sin(x)"]).

        Returns:
            OrchestratorResult with all intermediate data and the prompt.
        """
        sympy_expressions = sympy_expressions or []

        # 1. Retrieve theorems
        logger.debug("Retrieving theorems for question.")
        theorems = self.kb.get_theorems(question, k=k)

        # 2. Execute SymPy tools
        sympy_context: list[str] = []
        for expr in sympy_expressions:
            logger.debug("Executing tool expression: %s", expr)
            formatted = self.tools.format_result(expr.strip())
            sympy_context.append(formatted)

        # 3. Build prompt
        logger.debug(
            "Building prompt with %s theorems and %s tool results.",
            len(theorems),
            len(sympy_context),
        )
        prompt = build_math_prompt(question, theorems, sympy_context)

        return OrchestratorResult(
            question=question,
            theorems=theorems,
            sympy_context=sympy_context,
            prompt=prompt,
        )


def create_orchestrator(
    retriever,
    agent_config=None,
) -> ToolOrchestrator:
    """Convenience function to create an orchestrator.

    Args:
        retriever: NaturalProofsRetriever instance.
        agent_config: Optional AgentConfig for the TheoremKB.

    Returns:
        Configured ToolOrchestrator.
    """
    from ..tools.registry import create_default_registry

    kb = TheoremKB(retriever, config=agent_config)
    tools = create_default_registry()

    return ToolOrchestrator(kb, tools)
