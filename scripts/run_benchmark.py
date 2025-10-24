#!/usr/bin/env python3
"""Run TheoremQA benchmark across multiple modes."""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class ExampleResult:
    question: str
    gold_answer: str
    prediction: str
    correct: bool
    input_tokens: int
    output_tokens: int
    tool_calls: list[dict] = field(default_factory=list)
    retrieved_ids: list[int] = field(default_factory=list)
    iterations: int | None = None


def parse_modes(value: str) -> list[str]:
    return [mode.strip() for mode in value.split(",") if mode.strip()]


def parse_tool_name(command: str) -> str:
    if ":" in command:
        return command.split(":", 1)[0].strip()
    return command.strip()


def tool_call_useful(tool_name: str, observation: str) -> bool:
    if observation.startswith("[error]"):
        return False
    if tool_name == "retrieve":
        return "No theorems found." not in observation
    return True


def extract_retrieved_ids(observation: str) -> list[int]:
    ids = []
    for part in observation.split():
        cleaned = part.strip("()")
        if cleaned.startswith("idx="):
            try:
                ids.append(int(cleaned.replace("idx=", "")))
            except ValueError:
                continue
    return ids


def main():
    parser = argparse.ArgumentParser(description="Run TheoremQA benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional local path to TheoremQA JSON/JSONL",
    )
    parser.add_argument(
        "--dataset",
        "--dataset-name",
        dest="dataset_name",
        type=str,
        default="theoremqa",
        help="HuggingFace dataset name if no local path is provided",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="baseline,rag,manual,autonomous",
        help="Comma-separated list of modes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional LoRA adapter path",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens per response",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    from llmath.agent import MathAgent, create_react_agent
    from llmath.config import load_config
    from llmath.evaluation.theoremqa import (
        extract_final_answer,
        is_correct,
        load_theoremqa,
    )
    from llmath.inference.deepseek import DeepSeekMathModel
    from llmath.prompts.builder import build_baseline_prompt, build_math_prompt
    from llmath.prompts.orchestrator import create_orchestrator
    from llmath.retrieval import NaturalProofsRetriever
    from llmath.retrieval.theorem_kb import TheoremKB

    config = load_config(args.config)
    if args.adapter_path:
        config.model.adapter_path = Path(args.adapter_path)

    examples = load_theoremqa(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
        limit=args.limit,
    )

    modes = parse_modes(args.modes)

    retriever = NaturalProofsRetriever(rebuild_index=False)
    kb = TheoremKB(retriever, config=config.agent)
    model = DeepSeekMathModel.from_config(config.model, config.generation)
    orchestrator = create_orchestrator(retriever, config.agent)
    manual_agent = MathAgent(orchestrator, model, config.agent)
    react_agent = create_react_agent(
        retriever=retriever,
        model_config=config.model,
        generation_config=config.generation,
        agent_config=config.agent,
        react_config=config.react,
    )

    results: dict[str, dict] = {}

    for mode in modes:
        mode_results: list[ExampleResult] = []
        tool_calls_total = 0
        tool_calls_useful = 0
        retrieval_recall_total = 0.0
        retrieval_recall_count = 0
        iterations_total = 0
        token_total = 0
        correct_total = 0

        for example in examples:
            question = example.question
            gold = example.answer
            sympy_expressions = example.metadata.get("sympy_expressions", [])
            if not isinstance(sympy_expressions, list):
                sympy_expressions = [str(sympy_expressions)]
            gold_ids = example.metadata.get("theorem_ids") or example.metadata.get(
                "theorem_indices"
            )

            if mode == "baseline":
                prompt = build_baseline_prompt(question)
                gen = model.generate(
                    prompt=prompt,
                    max_new_tokens=args.max_tokens,
                )
                prediction = gen.text
                input_tokens = gen.input_tokens
                output_tokens = gen.output_tokens
                retrieved_ids: list[int] = []
                tool_calls: list[dict] = []
                iterations = None
            elif mode == "rag":
                theorems = kb.get_theorems(question, k=config.agent.default_k)
                prompt = build_math_prompt(question, theorems, [])
                gen = model.generate(
                    prompt=prompt,
                    max_new_tokens=args.max_tokens,
                )
                prediction = gen.text
                input_tokens = gen.input_tokens
                output_tokens = gen.output_tokens
                retrieved_ids = [t.idx for t in theorems]
                tool_calls = []
                iterations = None
            elif mode == "manual":
                agent_result = manual_agent.run(
                    question=question,
                    sympy_expressions=sympy_expressions,
                    k=config.agent.default_k,
                    max_new_tokens=args.max_tokens,
                )
                prediction = agent_result.answer
                input_tokens = agent_result.input_tokens
                output_tokens = agent_result.output_tokens
                retrieved_ids = [t.idx for t in agent_result.theorems]
                tool_calls = []
                iterations = None
            elif mode == "autonomous":
                react_result = react_agent.run(question)
                prediction = react_result.answer or ""
                input_tokens = react_result.input_tokens
                output_tokens = react_result.output_tokens
                retrieved_ids = []
                tool_calls = []
                for step in react_result.steps:
                    if step.tool:
                        tool_calls.append(
                            {"tool": step.tool, "observation": step.observation or ""}
                        )
                    if step.tool and step.observation:
                        tool_name = parse_tool_name(step.tool)
                        if tool_name == "retrieve":
                            retrieved_ids.extend(extract_retrieved_ids(step.observation))
                iterations = react_result.iterations
                iterations_total += iterations
            else:
                raise ValueError(f"Unknown mode: {mode}")

            final_answer = extract_final_answer(prediction)
            correct = is_correct(final_answer, gold)

            tool_calls_total += len(tool_calls)
            for call in tool_calls:
                tool_name = parse_tool_name(call["tool"])
                if tool_call_useful(tool_name, call["observation"]):
                    tool_calls_useful += 1

            if gold_ids:
                gold_set = set(gold_ids)
                retrieved_set = set(retrieved_ids)
                if gold_set:
                    recall = len(gold_set & retrieved_set) / len(gold_set)
                    retrieval_recall_total += recall
                    retrieval_recall_count += 1

            token_total += input_tokens + output_tokens
            if correct:
                correct_total += 1

            mode_results.append(
                ExampleResult(
                    question=question,
                    gold_answer=gold,
                    prediction=final_answer,
                    correct=correct,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tool_calls=tool_calls,
                    retrieved_ids=retrieved_ids,
                    iterations=iterations,
                )
            )

        total_examples = len(mode_results)
        accuracy = correct_total / total_examples if total_examples else 0.0
        tool_precision = tool_calls_useful / tool_calls_total if tool_calls_total else None
        retrieval_recall = (
            retrieval_recall_total / retrieval_recall_count if retrieval_recall_count else None
        )
        avg_iterations = None
        if total_examples and mode == "autonomous":
            avg_iterations = iterations_total / total_examples
        token_efficiency = token_total / correct_total if correct_total else None

        results[mode] = {
            "metrics": {
                "accuracy": accuracy,
                "tool_call_precision": tool_precision,
                "retrieval_recall_at_k": retrieval_recall,
                "avg_iterations": avg_iterations,
                "token_efficiency": token_efficiency,
                "total_examples": total_examples,
            },
            "examples": [r.__dict__ for r in mode_results],
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(
            {
                "dataset": {
                    "name": args.dataset_name,
                    "path": args.dataset_path,
                    "split": args.split,
                },
                "modes": results,
            },
            f,
            indent=2,
        )

    print(f"Benchmark results saved to: {output_path}")


if __name__ == "__main__":
    main()
