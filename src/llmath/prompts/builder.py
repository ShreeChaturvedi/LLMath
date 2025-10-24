"""Prompt builder for the math agent.

Constructs prompts that combine:
- The original question
- Retrieved theorem snippets
- Symbolic tool outputs
- Instructions for proof-style answers
"""

from textwrap import indent

from ..retrieval.theorem_kb import TheoremSnippet
from .templates import ANSWER_INSTRUCTIONS, BASELINE_PROMPT_TEMPLATE, SYSTEM_HEADER


def build_math_prompt(
    question: str,
    theorems: list[TheoremSnippet],
    sympy_context: list[str] | None = None,
) -> str:
    """Construct the full prompt for the math agent.

    Args:
        question: The mathematical question to answer.
        theorems: List of retrieved theorem snippets.
        sympy_context: List of symbolic computation results
            (e.g., "solve(x**2-1) -> [-1, 1]").

    Returns:
        Formatted prompt string ready for the LLM.
    """
    sympy_context = sympy_context or []

    # Question block
    q_block = f"Question:\n{question.strip()}\n\n"

    # Theorems block
    if theorems:
        theorem_blocks = []
        for i, t in enumerate(theorems, 1):
            label = f"[T{i}]"
            title = t.title.strip()
            snippet = t.snippet.strip()
            block = f"{label}  Title: {title}\n"
            block += indent(snippet, "    ")
            theorem_blocks.append(block)
        theorems_block = (
            "Retrieved theorems / definitions:\n" + "\n\n".join(theorem_blocks) + "\n\n"
        )
    else:
        theorems_block = "Retrieved theorems / definitions:\nNone.\n\n"

    # SymPy tool context block
    if sympy_context:
        sym_lines = []
        for i, info in enumerate(sympy_context, 1):
            label = f"[S{i}]"
            sym_lines.append(f"{label}  {info}")
        sym_block = "Symbolic tool results (SymPy):\n" + "\n".join(sym_lines) + "\n\n"
    else:
        sym_block = "Symbolic tool results (SymPy):\nNone.\n\n"

    return SYSTEM_HEADER + q_block + theorems_block + sym_block + ANSWER_INSTRUCTIONS


def build_baseline_prompt(question: str) -> str:
    """Build a simple prompt without retrieval or tools.

    Used for baseline comparison to measure the effect of RAG and tools.

    Args:
        question: The mathematical question.

    Returns:
        Simple prompt string without context.
    """
    return BASELINE_PROMPT_TEMPLATE.format(question=question.strip())
