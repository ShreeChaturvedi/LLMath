"""Training module for LLMath - LoRA fine-tuning utilities."""

from typing import TYPE_CHECKING

# Heavy imports (datasets, transformers, peft) are lazy-loaded
if TYPE_CHECKING:
    from .data import build_sft_examples, create_sft_dataset
    from .formatting import (
        create_tokenize_function,
        format_for_deepseek,
        tokenize_batch,
    )
    from .react_data import build_react_examples, create_react_dataset
    from .trainer import create_trainer, train_lora

__all__ = [
    "build_sft_examples",
    "create_sft_dataset",
    "build_react_examples",
    "create_react_dataset",
    "format_for_deepseek",
    "tokenize_batch",
    "create_tokenize_function",
    "create_trainer",
    "train_lora",
]


def __getattr__(name: str):
    """Lazy import for heavy dependencies."""
    if name in ("build_sft_examples", "create_sft_dataset"):
        from .data import build_sft_examples, create_sft_dataset

        return {"build_sft_examples": build_sft_examples, "create_sft_dataset": create_sft_dataset}[name]
    if name in ("format_for_deepseek", "tokenize_batch", "create_tokenize_function"):
        from .formatting import create_tokenize_function, format_for_deepseek, tokenize_batch

        return {"format_for_deepseek": format_for_deepseek, "tokenize_batch": tokenize_batch, "create_tokenize_function": create_tokenize_function}[name]
    if name in ("build_react_examples", "create_react_dataset"):
        from .react_data import build_react_examples, create_react_dataset

        return {"build_react_examples": build_react_examples, "create_react_dataset": create_react_dataset}[name]
    if name in ("create_trainer", "train_lora"):
        from .trainer import create_trainer, train_lora

        return {"create_trainer": create_trainer, "train_lora": train_lora}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
