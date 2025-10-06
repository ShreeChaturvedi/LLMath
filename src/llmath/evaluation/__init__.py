─────┬──────────────────────────────────────────────────────────────────────────
     │ STDIN
─────┼──────────────────────────────────────────────────────────────────────────
   1 │ """Evaluation module for LLMath - baseline comparison and metrics."""
   2 │ 
   3 │ from .baseline import BaselineModel, answer_without_tools
   4 │ from .comparison import ComparisonResult, compare_baseline_and_agent, summarize_results
   5 │ 
   6 │ __all__ = [
   7 │     "BaselineModel",
   8 │     "answer_without_tools",
   9 │     "ComparisonResult",
  10 │     "compare_baseline_and_agent",
  11 │     "summarize_results",
  12 │ ]
─────┴──────────────────────────────────────────────────────────────────────────
