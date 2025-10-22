"""Evaluation module for LLMath - baseline comparison and metrics."""

from .baseline import BaselineModel, answer_without_tools
from .comparison import (
    ComparisonResult,
    compare_baseline_and_agent,
    summarize_results,
    print_summary,
)

__all__ = [
    "BaselineModel",
    "answer_without_tools",
    "ComparisonResult",
    "compare_baseline_and_agent",
    "summarize_results",
    "print_summary",
]
