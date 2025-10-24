"""Evaluation module for LLMath - baseline comparison and metrics."""

from .baseline import BaselineModel, answer_without_tools
from .comparison import (
    ComparisonResult,
    compare_baseline_and_agent,
    print_summary,
    summarize_results,
)
from .theoremqa import (
    TheoremQAExample,
    extract_final_answer,
    is_correct,
    load_theoremqa,
)

__all__ = [
    "BaselineModel",
    "answer_without_tools",
    "ComparisonResult",
    "compare_baseline_and_agent",
    "summarize_results",
    "print_summary",
    "TheoremQAExample",
    "load_theoremqa",
    "extract_final_answer",
    "is_correct",
]
