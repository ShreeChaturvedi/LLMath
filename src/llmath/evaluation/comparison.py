"""Comparison utilities for evaluating agent vs baseline.

Provides functions for running side-by-side comparisons and
analyzing the results.
"""

from collections import Counter
from dataclasses import dataclass, field

from ..agent.math_agent import MathAgent
from ..inference.deepseek import DeepSeekMathModel
from .baseline import BaselineModel


@dataclass
class ComparisonResult:
    """Result from comparing baseline and agent on a question.

    Attributes:
        question: The original question.
        sympy_expressions: SymPy commands used by the agent.
        baseline_answer: The baseline model's answer.
        agent_answer: The agent's answer.
        theorem_titles: Titles of retrieved theorems.
        sympy_context: Formatted SymPy outputs.
        label: Manual evaluation label (e.g., "agent_better").
        notes: Additional notes about the comparison.
    """

    question: str
    sympy_expressions: list[str]
    baseline_answer: str
    agent_answer: str
    theorem_titles: list[str] = field(default_factory=list)
    sympy_context: list[str] = field(default_factory=list)
    label: str | None = None
    notes: str = ""

    def __str__(self) -> str:
        """Format as readable comparison."""
        lines = [
            "=" * 60,
            "QUESTION:",
            self.question,
            "",
            "[BASELINE]",
            self.baseline_answer,
            "",
            "[AGENT]",
            f"Theorems: {', '.join(self.theorem_titles) or 'None'}",
            f"SymPy: {', '.join(self.sympy_context) or 'None'}",
            "",
            self.agent_answer,
        ]
        if self.label:
            lines.extend(["", f"Label: {self.label}"])
        if self.notes:
            lines.extend([f"Notes: {self.notes}"])
        lines.append("=" * 60)
        return "\n".join(lines)


def compare_baseline_and_agent(
    eval_items: list[dict],
    agent: MathAgent,
    model: DeepSeekMathModel,
    max_new_tokens: int = 256,
    verbose: bool = True,
) -> list[ComparisonResult]:
    """Run baseline vs agent comparison on evaluation items.

    For each item, runs both the baseline model and the full agent,
    collecting answers and metadata for analysis.

    Args:
        eval_items: List of dicts with 'question' and optional
            'sympy_expressions' and 'k' keys.
        agent: The configured MathAgent.
        model: The DeepSeekMathModel (used for baseline).
        max_new_tokens: Maximum tokens for generation.
        verbose: Whether to print results as they're generated.

    Returns:
        List of ComparisonResult objects.
    """
    baseline = BaselineModel(model)
    results: list[ComparisonResult] = []

    for i, item in enumerate(eval_items, start=1):
        question = item["question"]
        sympy_expressions = item.get("sympy_expressions", []) or []
        k = item.get("k", 5)

        if verbose:
            print("=" * 60)
            print(f"Example {i}")
            print(f"QUESTION: {question}")

        # Baseline answer
        baseline_answer = baseline.answer(
            question,
            max_new_tokens=max_new_tokens,
        )
        if verbose:
            print("\n[BASELINE]")
            print(baseline_answer)

        # Agent answer
        agent_result = agent.run(
            question=question,
            sympy_expressions=sympy_expressions,
            k=k,
            max_new_tokens=max_new_tokens,
        )
        if verbose:
            print("\n[AGENT]")
            print(f"Theorems: {[t.title for t in agent_result.theorems]}")
            print(f"SymPy: {agent_result.sympy_context}")
            print(agent_result.answer)
            print()

        results.append(
            ComparisonResult(
                question=question,
                sympy_expressions=sympy_expressions,
                baseline_answer=baseline_answer,
                agent_answer=agent_result.answer,
                theorem_titles=[t.title for t in agent_result.theorems],
                sympy_context=list(agent_result.sympy_context),
            )
        )

    return results


def summarize_results(results: list[ComparisonResult]) -> dict:
    """Summarize comparison results by label counts.

    Args:
        results: List of labeled ComparisonResult objects.

    Returns:
        Dict with 'counts' (Counter of labels), 'total', and
        'labeled' (number with non-None labels).
    """
    labels = [r.label for r in results if r.label is not None]
    counts = Counter(labels)

    return {
        "counts": dict(counts),
        "total": len(results),
        "labeled": len(labels),
    }


def print_summary(results: list[ComparisonResult]) -> None:
    """Print a formatted summary of comparison results.

    Args:
        results: List of comparison results to summarize.
    """
    summary = summarize_results(results)

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total examples: {summary['total']}")
    print(f"Labeled: {summary['labeled']}")
    print()
    print("Label distribution:")
    for label, count in sorted(summary["counts"].items()):
        pct = 100 * count / summary["labeled"] if summary["labeled"] > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")
    print()

    for i, r in enumerate(results, start=1):
        print(f"[{i}] {r.question[:50]}...")
        print(f"    Label: {r.label or 'unlabeled'}")
        if r.notes:
            print(f"    Notes: {r.notes}")
        print()
