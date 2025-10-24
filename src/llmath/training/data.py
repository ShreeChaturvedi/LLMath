"""Data utilities for building SFT datasets from NaturalProofs.

Provides functions for creating instruction-response pairs from
the NaturalProofs dataset for fine-tuning.
"""

import random

from datasets import Dataset


def _pick_field(
    cols: list[str], candidates: tuple[str, ...], default: str | None = None
) -> str | None:
    """Find the first matching field from candidates in columns."""
    for c in candidates:
        if c in cols:
            return c
    return default


def build_sft_examples(
    ds: Dataset,
    text_field: str,
    max_examples: int = 300,
    seed: int = 42,
) -> list[dict]:
    """Build instruction-response pairs from NaturalProofs.

    Creates simple proof-style examples for supervised fine-tuning.
    The examples teach the model to produce structured answers with:
    1. Theorems Used
    2. Proof
    3. Conclusion

    Args:
        ds: The NaturalProofs dataset.
        text_field: The field containing the main text content.
        max_examples: Maximum number of examples to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with 'prompt' and 'response' keys.
    """
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

    # Shuffle and limit indices
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    indices = indices[:max_examples]

    examples = []

    for idx in indices:
        row = ds[idx]

        # Get statement with fallbacks
        statement = ""
        if statement_field is not None:
            statement = str(row[statement_field]).strip()
        if not statement:
            statement = str(row[text_field]).strip()

        # Get proof with fallbacks
        proof_text = ""
        if proof_field is not None:
            proof_text = str(row[proof_field]).strip()
        if not proof_text:
            proof_text = str(row[text_field]).strip()

        # Get title with fallback
        title = None
        if title_field is not None:
            title = str(row[title_field]).strip()
        if not title:
            title = f"NaturalProofs entry {idx}"

        # Build instruction prompt
        prompt = (
            "You are a mathematical assistant. Prove the following statement in a clear, "
            "concise, proof-style format.\n\n"
            f"Title: {title}\n"
            f"Statement:\n{statement}\n\n"
            "Follow this structure:\n"
            "1. Theorems Used: list any named results you rely on.\n"
            "2. Proof: short step-by-step argument.\n"
            "3. Conclusion: one sentence summarizing the result.\n"
        )

        # Build target response
        response = (
            f"Theorems Used:\n"
            f"- From the NaturalProofs corpus: {title}\n\n"
            f"Proof:\n{proof_text}\n\n"
            f"Conclusion:\nTherefore, the statement above holds as required."
        )

        examples.append(
            {
                "prompt": prompt,
                "response": response,
            }
        )

    return examples


def create_sft_dataset(
    ds: Dataset,
    text_field: str,
    max_examples: int = 300,
    seed: int = 42,
) -> Dataset:
    """Create a HuggingFace Dataset from SFT examples.

    Convenience wrapper around build_sft_examples that returns
    a properly formatted Dataset object.

    Args:
        ds: The NaturalProofs dataset.
        text_field: The field containing the main text content.
        max_examples: Maximum number of examples to generate.
        seed: Random seed for reproducibility.

    Returns:
        HuggingFace Dataset with 'prompt' and 'response' columns.
    """
    examples = build_sft_examples(ds, text_field, max_examples, seed)
    return Dataset.from_list(examples)
