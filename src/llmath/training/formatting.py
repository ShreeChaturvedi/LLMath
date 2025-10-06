"""Formatting utilities for SFT training.

Provides functions for formatting datasets according to the
DeepSeek chat template and tokenization.
"""

from typing import Callable

from transformers import PreTrainedTokenizer


def format_for_deepseek(
    examples: dict[str, list[str]],
    tokenizer: PreTrainedTokenizer,
) -> list[str]:
    """Format examples using DeepSeek's chat template.

    Converts prompt/response pairs into the chat format expected
    by DeepSeek-Math for instruction tuning.

    Args:
        examples: Dict with 'prompt' and 'response' lists.
        tokenizer: The tokenizer with chat template support.

    Returns:
        List of formatted text strings.
    """
    texts = []
    for prompt, response in zip(examples["prompt"], examples["response"]):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        texts.append(text)
    return texts


def tokenize_batch(
    examples: dict[str, list[str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 1024,
) -> dict[str, list[list[int]]]:
    """Tokenize a batch of examples for training.

    Formats examples using the chat template, then tokenizes with
    padding and truncation.

    Args:
        examples: Dict with 'prompt' and 'response' lists.
        tokenizer: The tokenizer to use.
        max_length: Maximum sequence length.

    Returns:
        Dict with 'input_ids', 'attention_mask', and 'labels'.
    """
    texts = format_for_deepseek(examples, tokenizer)
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    # For causal LM, labels are the same as input_ids
    enc["labels"] = enc["input_ids"].copy()
    return enc


def create_tokenize_function(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 1024,
) -> Callable:
    """Create a tokenization function for dataset.map().

    Returns a function that can be passed to Dataset.map() for
    batch tokenization.

    Args:
        tokenizer: The tokenizer to use.
        max_length: Maximum sequence length.

    Returns:
        Function suitable for Dataset.map(batched=True).
    """
    def tokenize_fn(examples):
        return tokenize_batch(examples, tokenizer, max_length)
    return tokenize_fn
