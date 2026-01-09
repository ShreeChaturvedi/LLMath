"""Inference module for LLMath - model loading and generation."""

from typing import TYPE_CHECKING

from .generation import clean_model_output, truncate_at_stop_sequences

# Heavy imports (peft, transformers) are lazy-loaded
if TYPE_CHECKING:
    from .deepseek import DeepSeekMathModel, GenerationResult
    from .model_loader import (
        create_lora_config,
        create_lora_model,
        load_base_model,
        load_tokenizer,
        load_trained_model,
    )

__all__ = [
    "load_tokenizer",
    "load_base_model",
    "create_lora_config",
    "create_lora_model",
    "load_trained_model",
    "clean_model_output",
    "truncate_at_stop_sequences",
    "DeepSeekMathModel",
    "GenerationResult",
]


def __getattr__(name: str):
    """Lazy import for heavy dependencies."""
    if name in ("DeepSeekMathModel", "GenerationResult"):
        from .deepseek import DeepSeekMathModel, GenerationResult

        if name == "DeepSeekMathModel":
            return DeepSeekMathModel
        return GenerationResult
    if name in (
        "load_tokenizer",
        "load_base_model",
        "create_lora_config",
        "create_lora_model",
        "load_trained_model",
    ):
        from . import model_loader

        return getattr(model_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
