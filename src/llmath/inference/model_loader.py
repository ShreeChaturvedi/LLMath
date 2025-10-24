"""Model loading utilities.

Provides functions for loading DeepSeek-Math models with optional
quantization and LoRA adapters.
"""

import logging

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..config import ModelConfig

logger = logging.getLogger(__name__)


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load and configure the tokenizer.

    Args:
        model_name: HuggingFace model name or path.

    Returns:
        Configured AutoTokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(config: ModelConfig) -> AutoModelForCausalLM:
    """Load the base model with optional 4-bit quantization.

    Args:
        config: Model configuration.

    Returns:
        Loaded model (potentially quantized).
    """
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
            logger.info("Loaded model in 4-bit quantization")
            return model
        except ImportError as e:
            logger.warning(f"4-bit quantization unavailable: {e}")

    # Fallback: no quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    logger.info("Loaded model without quantization")
    return model


def create_lora_config(config: ModelConfig) -> LoraConfig:
    """Create LoRA configuration for training.

    Args:
        config: Model configuration with LoRA settings.

    Returns:
        LoraConfig instance.
    """
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def create_lora_model(base_model: AutoModelForCausalLM, config: ModelConfig):
    """Wrap base model with LoRA adapters.

    Args:
        base_model: The base language model.
        config: Model configuration with LoRA settings.

    Returns:
        Model with LoRA adapters (PeftModel).
    """
    base_model = prepare_model_for_kbit_training(base_model)
    lora_config = create_lora_config(config)
    return get_peft_model(base_model, lora_config)


def load_trained_model(
    config: ModelConfig,
    tokenizer: AutoTokenizer | None = None,
) -> tuple[PeftModel | AutoModelForCausalLM, AutoTokenizer]:
    """Load model with pre-trained LoRA adapters.

    Args:
        config: Model configuration.
        tokenizer: Optional pre-loaded tokenizer.

    Returns:
        Tuple of (model, tokenizer).
    """
    if tokenizer is None:
        tokenizer = load_tokenizer(config.model_name)

    base = load_base_model(config)
    base.generation_config.pad_token_id = tokenizer.eos_token_id

    if config.adapter_path:
        model = PeftModel.from_pretrained(base, str(config.adapter_path))
        model.eval()
        logger.info(f"Loaded LoRA adapters from {config.adapter_path}")
        return model, tokenizer

    return base, tokenizer
