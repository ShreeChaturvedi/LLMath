#!/usr/bin/env python3
"""Train LoRA adapters on NaturalProofs dataset.

Usage:
    python scripts/train_lora.py [--config configs/default.yaml]

Requires GPU with bitsandbytes support for best results.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Train LoRA adapters for DeepSeek-Math"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/deepseek_math_sft_lora",
        help="Output directory for trained adapters",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=300,
        help="Maximum training examples to use",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sft",
        choices=["sft", "react"],
        help="Training mode: sft or react",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    args = parser.parse_args()

    from llmath.config import load_config
    from llmath.retrieval import NaturalProofsRetriever
    from llmath.inference.model_loader import (
        load_tokenizer,
        load_base_model,
        create_lora_model,
    )
    from llmath.training import (
        create_sft_dataset,
        create_react_dataset,
        create_tokenize_function,
        train_lora,
    )

    # Load configuration
    config = load_config(args.config)
    if args.epochs:
        config.training.num_epochs = args.epochs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LLMath LoRA Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Max examples: {args.max_examples}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Mode: {args.mode}")
    print(f"Max seq length: {config.training.max_seq_length}")
    print()

    # Load retriever to get dataset
    print("Loading NaturalProofs dataset...")
    retriever = NaturalProofsRetriever(rebuild_index=False)

    # Build dataset
    if args.mode == "react":
        print("Building ReAct dataset...")
        sft_dataset = create_react_dataset(
            retriever.ds,
            retriever.text_field,
            max_examples=args.max_examples,
        )
    else:
        print("Building SFT dataset...")
        sft_dataset = create_sft_dataset(
            retriever.ds,
            retriever.text_field,
            max_examples=args.max_examples,
        )
    print(f"Dataset size: {len(sft_dataset)}")

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = load_tokenizer(config.model.model_name)
    base_model = load_base_model(config.model)
    model = create_lora_model(base_model, config.model)
    model.print_trainable_parameters()

    # Tokenize dataset
    print("\nTokenizing dataset...")
    tokenize_fn = create_tokenize_function(
        tokenizer,
        max_length=config.training.max_seq_length,
    )
    tokenized_dataset = sft_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=sft_dataset.column_names,
    )

    # Train
    print("\nStarting training...")
    train_lora(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        config=config.training,
        output_dir=output_dir,
    )

    print(f"\nTraining complete! Adapters saved to: {output_dir}")


if __name__ == "__main__":
    main()
