"""Utilities for loading and evaluating TheoremQA."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable

from datasets import load_dataset


@dataclass
class TheoremQAExample:
    """Single TheoremQA example."""

    question: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _extract_field(item: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        if key in item and item[key]:
            return str(item[key]).strip()
    return None


def _normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("$", "")
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"[^0-9a-zA-Z.\-\/ ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_final_answer(text: str) -> str:
    """Extract a final answer string from model output."""
    match = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match[-1].strip()

    for marker in ("Final Answer:", "Answer:", "Conclusion:"):
        if marker in text:
            return text.split(marker, 1)[-1].strip()

    return text.strip()


def is_correct(prediction: str, gold: str, tol: float = 1e-6) -> bool:
    """Return True if prediction matches gold answer."""
    pred_norm = _normalize_answer(prediction)
    gold_norm = _normalize_answer(gold)
    if pred_norm == gold_norm:
        return True

    try:
        pred_val = float(pred_norm)
        gold_val = float(gold_norm)
        return abs(pred_val - gold_val) <= tol
    except ValueError:
        return False


def load_theoremqa(
    dataset_path: str | Path | None = None,
    dataset_name: str = "theoremqa",
    split: str = "test",
    limit: int | None = None,
) -> list[TheoremQAExample]:
    """Load TheoremQA from a local file or HuggingFace dataset."""
    if dataset_path:
        path = Path(dataset_path)
        if path.is_dir():
            json_path = path / "theoremqa.json"
            jsonl_path = path / "theoremqa.jsonl"
            if json_path.exists():
                path = json_path
            elif jsonl_path.exists():
                path = jsonl_path
            else:
                raise FileNotFoundError(f"No theoremqa.json or theoremqa.jsonl in {path}")

        if not path.exists():
            raise FileNotFoundError(f"TheoremQA dataset not found: {path}")

        if path.suffix == ".jsonl":
            items = []
            with path.open("r") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
        else:
            with path.open("r") as f:
                items = json.load(f)

        if isinstance(items, dict) and "data" in items:
            items = items["data"]
    else:
        dataset = load_dataset(dataset_name, split=split)
        items = list(dataset)

    examples: list[TheoremQAExample] = []
    for item in items[: limit or len(items)]:
        question = _extract_field(item, ("question", "query", "problem"))
        answer = _extract_field(item, ("answer", "final_answer", "solution", "output"))
        if question is None or answer is None:
            continue
        metadata = {
            k: v
            for k, v in item.items()
            if k
            not in {
                "question",
                "query",
                "problem",
                "answer",
                "final_answer",
                "solution",
                "output",
            }
        }
        examples.append(TheoremQAExample(question=question, answer=answer, metadata=metadata))

    return examples
