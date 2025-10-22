"""Synthetic ReAct trace generation for training."""

import random

from datasets import Dataset

from ..prompts.react_templates import build_react_system_prompt


def _pick_field(cols: list[str], candidates: tuple[str, ...], default: str | None = None) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return default


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip().replace("\\n", "\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def build_react_examples(
    ds: Dataset,
    text_field: str,
    max_examples: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """Build synthetic ReAct training examples from NaturalProofs."""
    cols = list(ds.column_names)

    statement_field = _pick_field(
        cols,
        candidates=("statement", "goal", "theorem", "page", "text"),
    )
    proof_field = _pick_field(
        cols,
        candidates=("proof", "proof_text", "text"),
    )
    title_field = _pick_field(
        cols,
        candidates=("title", "name", "source", "section", "chapter"),
    )

    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    indices = indices[:max_examples]

    tool_names = ["retrieve", "simplify", "solve", "diff", "integrate"]
    system_prompt = build_react_system_prompt(tool_names)

    examples: list[dict] = []
    for idx in indices:
        row = ds[idx]

        statement = ""
        if statement_field is not None:
            statement = str(row[statement_field]).strip()
        if not statement:
            statement = str(row[text_field]).strip()

        proof_text = ""
        if proof_field is not None:
            proof_text = str(row[proof_field]).strip()
        if not proof_text:
            proof_text = str(row[text_field]).strip()

        title = None
        if title_field is not None:
            title = str(row[title_field]).strip()
        if not title:
            title = f"NaturalProofs entry {idx}"

        question = statement
        query = " ".join(statement.split()[:12])
        snippet = _truncate(str(row[text_field]), 220)

        observation = f"[T1] (idx={idx}) (score=1.000) {title}: {snippet}"
        answer = (
            f"Using [T1], {_truncate(proof_text, 300)}\n"
            "Conclusion: The statement follows from the cited theorem."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question:\n{question}"},
            {
                "role": "assistant",
                "content": (
                    "<think>Retrieve a relevant theorem.</think>"
                    f"<tool>retrieve: {query}</tool>"
                ),
            },
            {"role": "user", "content": f"<observe>{observation}</observe>"},
            {
                "role": "assistant",
                "content": (
                    "<think>Use the retrieved theorem to answer.</think>"
                    f"<answer>{answer}</answer>"
                ),
            },
        ]

        examples.append({"messages": messages})

    return examples


def create_react_dataset(
    ds: Dataset,
    text_field: str,
    max_examples: int = 1000,
    seed: int = 42,
) -> Dataset:
    """Create a HuggingFace Dataset for ReAct training."""
    examples = build_react_examples(ds, text_field, max_examples, seed)
    return Dataset.from_list(examples)
