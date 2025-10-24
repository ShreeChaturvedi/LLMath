"""Training utilities for LoRA fine-tuning.

Provides functions for configuring and running LoRA training
on DeepSeek-Math models.
"""

import logging
from pathlib import Path

from datasets import Dataset
from peft import PeftModel
from transformers import PreTrainedTokenizer, Trainer, TrainingArguments

from ..config import TrainingConfig

logger = logging.getLogger(__name__)


def create_training_args(
    config: TrainingConfig,
    output_dir: str | Path,
) -> TrainingArguments:
    """Create TrainingArguments from configuration.

    Args:
        config: Training configuration.
        output_dir: Directory for checkpoints and logs.

    Returns:
        Configured TrainingArguments.
    """
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        optim=config.optimizer,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler,
        report_to="none",
    )


def prepare_model_for_training(model: PeftModel) -> PeftModel:
    """Prepare a PEFT model for LoRA training.

    Sets the model to training mode and ensures only LoRA
    parameters are trainable.

    Args:
        model: The PEFT model to prepare.

    Returns:
        The prepared model.
    """
    model.train()
    model.config.use_cache = False

    # Ensure only LoRA parameters are trainable
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def create_trainer(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    config: TrainingConfig,
    output_dir: str | Path,
    eval_dataset: Dataset | None = None,
) -> Trainer:
    """Create a configured Trainer for LoRA fine-tuning.

    Args:
        model: The PEFT model to train.
        tokenizer: The tokenizer (for saving).
        train_dataset: Tokenized training dataset.
        config: Training configuration.
        output_dir: Directory for outputs.
        eval_dataset: Optional evaluation dataset.

    Returns:
        Configured Trainer instance.
    """
    model = prepare_model_for_training(model)
    training_args = create_training_args(config, output_dir)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )


def train_lora(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    config: TrainingConfig,
    output_dir: str | Path,
    eval_dataset: Dataset | None = None,
) -> Path:
    """Run LoRA fine-tuning and save the adapters.

    Complete training pipeline that:
    1. Creates the trainer
    2. Runs training
    3. Saves the model and tokenizer

    Args:
        model: The PEFT model to train.
        tokenizer: The tokenizer.
        train_dataset: Tokenized training dataset.
        config: Training configuration.
        output_dir: Directory for outputs.
        eval_dataset: Optional evaluation dataset.

    Returns:
        Path to the saved adapter directory.
    """
    output_dir = Path(output_dir)

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=config,
        output_dir=output_dir,
        eval_dataset=eval_dataset,
    )

    logger.info("Starting LoRA fine-tuning...")
    trainer.train()

    # Save final checkpoint
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir
