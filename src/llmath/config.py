"""
Configuration management for LLMath.

Uses Pydantic for validation and YAML for external config files.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding model."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = True


class RetrieverConfig(BaseModel):
    """Configuration for the NaturalProofs retriever."""

    dataset_name: str = "wellecks/naturalproofs-gen"
    dataset_split: str = "train"
    index_path: Path = Path("data/naturalproofs_faiss.index")
    meta_path: Path = Path("data/naturalproofs_meta.json")
    text_field_candidates: list[str] = Field(
        default=["statement", "text", "page", "theorem", "content"]
    )
    title_field_candidates: list[str] = Field(
        default=["title", "theorem", "name", "source", "section", "chapter"]
    )


class ModelConfig(BaseModel):
    """Configuration for the DeepSeek-Math model."""

    model_name: str = "deepseek-ai/deepseek-math-7b-rl"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    adapter_path: Path | None = None


class GenerationConfig(BaseModel):
    """Configuration for text generation."""

    max_new_tokens: int = 512
    min_new_tokens: int = 120
    temperature: float = 0.6
    top_p: float = 0.9
    repetition_penalty: float = 1.05


class AgentConfig(BaseModel):
    """Configuration for the math agent."""

    default_k: int = 5
    max_snippet_chars: int = 400


class ReActConfig(BaseModel):
    """Configuration for the ReAct agent loop."""

    max_iterations: int = 8
    max_tokens_per_step: int = 256
    retrieval_k: int = 3
    enable_self_verification: bool = True


class TrainingConfig(BaseModel):
    """Configuration for LoRA fine-tuning."""

    output_dir: Path = Path("data/deepseek_math_sft_lora")
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_epochs: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    max_seq_length: int = 2048
    max_examples: int = 300
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2
    optimizer: str = "adamw_torch"
    lr_scheduler: str = "cosine"


class LLMathConfig(BaseModel):
    """Root configuration for LLMath."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    react: ReActConfig = Field(default_factory=ReActConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def load_config(path: Path | str | None = None) -> LLMathConfig:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to YAML config file. If None, returns default config.

    Returns:
        LLMathConfig instance with loaded or default values.
    """
    if path is None:
        return LLMathConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return LLMathConfig(**data)


def _convert_paths(obj):
    """Convert Path objects to strings for YAML serialization."""
    if isinstance(obj, dict):
        return {k: _convert_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_paths(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_config(config: LLMathConfig, path: Path | str) -> None:
    """Save configuration to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = _convert_paths(config.model_dump())

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
