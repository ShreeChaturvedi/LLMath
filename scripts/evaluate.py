#!/usr/bin/env python3
"""Run evaluation comparing baseline vs agent.

Usage:
    python scripts/evaluate.py [--config configs/default.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Default evaluation questions
DEFAULT_EVAL_ITEMS = [
    {
        "question": "Prove that the derivative of x**2*sin(x) is 2*x*sin(x) + x**2*cos(x).",
        "sympy_expressions": ["diff: x**2*sin(x)"],
    },
    {
        "question": "Show that the sum of two continuous functions on R is continuous.",
        "sympy_expressions": [],
    },
    {
        "question": "Solve the equation x**2 - 1 = 0 and justify your steps.",
        "sympy_expressions": ["solve: x**2 - 1 = 0"],
    },
    {
        "question": "Compute the derivative of sin(x)*cos(x) and simplify your answer.",
        "sympy_expressions": ["diff: sin(x)*cos(x)"],
    },
    {
        "question": "Find the indefinite integral of 2*x and explain your reasoning.",
        "sympy_expressions": ["integrate: 2*x"],
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline vs agent"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to trained LoRA adapters (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/eval_results.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens per response",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    from llmath.config import load_config
    from llmath.retrieval import NaturalProofsRetriever
    from llmath.retrieval.theorem_kb import TheoremKB
    from llmath.prompts.orchestrator import create_orchestrator
    from llmath.inference.deepseek import DeepSeekMathModel
    from llmath.agent import MathAgent
    from llmath.evaluation import compare_baseline_and_agent, print_summary

    # Load config
    config = load_config(args.config)
    if args.adapter_path:
        config.model.adapter_path = Path(args.adapter_path)

    print("=" * 60)
    print("LLMath Evaluation")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Adapter: {config.model.adapter_path or 'None (base model)'}")
    print(f"Output: {args.output}")
    print()

    # Initialize components
    print("Loading retriever...")
    retriever = NaturalProofsRetriever(rebuild_index=False)

    print("Loading model...")
    model = DeepSeekMathModel.from_config(
        config.model,
        config.generation,
    )

    print("Creating orchestrator...")
    orchestrator = create_orchestrator(retriever, config.agent)

    print("Creating agent...")
    agent = MathAgent(orchestrator, model, config.agent)

    # Run evaluation
    print("\nRunning evaluation...")
    print("=" * 60)

    results = compare_baseline_and_agent(
        eval_items=DEFAULT_EVAL_ITEMS,
        agent=agent,
        model=model,
        max_new_tokens=args.max_tokens,
        verbose=not args.quiet,
    )

    # Print summary
    print_summary(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_data = [
        {
            "question": r.question,
            "sympy_expressions": r.sympy_expressions,
            "baseline_answer": r.baseline_answer,
            "agent_answer": r.agent_answer,
            "theorem_titles": r.theorem_titles,
            "sympy_context": r.sympy_context,
            "label": r.label,
            "notes": r.notes,
        }
        for r in results
    ]

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
