─────┬──────────────────────────────────────────────────────────────────────────
     │ STDIN
─────┼──────────────────────────────────────────────────────────────────────────
   1 │ """Training module for LLMath - LoRA fine-tuning utilities."""
   2 │ 
   3 │ from .data import build_sft_examples, create_sft_dataset
   4 │ from .formatting import format_for_deepseek, tokenize_batch
   5 │ from .trainer import create_trainer, train_lora
   6 │ 
   7 │ __all__ = [
   8 │     "build_sft_examples",
   9 │     "create_sft_dataset",
  10 │     "format_for_deepseek",
  11 │     "tokenize_batch",
  12 │     "create_trainer",
  13 │     "train_lora",
  14 │ ]
─────┴──────────────────────────────────────────────────────────────────────────
