"""Tests for the configuration module."""

import tempfile
from pathlib import Path

import pytest

from llmath.config import (
    LLMathConfig,
    EmbeddingConfig,
    RetrieverConfig,
    ModelConfig,
    GenerationConfig,
    AgentConfig,
    load_config,
    save_config,
)


class TestDefaultConfig:
    """Test default configuration values."""

    def test_default_config_creates_valid_instance(self):
        config = LLMathConfig()
        assert config is not None
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.retriever, RetrieverConfig)
        assert isinstance(config.model, ModelConfig)

    def test_embedding_defaults(self):
        config = EmbeddingConfig()
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.batch_size == 64
        assert config.normalize is True

    def test_retriever_defaults(self):
        config = RetrieverConfig()
        assert config.dataset_name == "wellecks/naturalproofs-gen"
        assert config.dataset_split == "train"
        assert "text" in config.text_field_candidates

    def test_model_defaults(self):
        config = ModelConfig()
        assert "deepseek" in config.model_name.lower()
        assert config.load_in_4bit is True
        assert config.lora_r == 16

    def test_generation_defaults(self):
        config = GenerationConfig()
        assert config.max_new_tokens == 512
        assert 0 < config.temperature < 1


class TestConfigLoading:
    """Test YAML config loading and saving."""

    def test_load_config_returns_defaults_when_no_path(self):
        config = load_config()
        assert config == LLMathConfig()

    def test_load_config_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_save_and_load_roundtrip(self):
        original = LLMathConfig()
        original.agent.default_k = 10  # Modify a value

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"
            save_config(original, path)

            loaded = load_config(path)
            assert loaded.agent.default_k == 10
            assert loaded.embedding == original.embedding


class TestConfigValidation:
    """Test configuration validation."""

    def test_invalid_temperature_type_raises(self):
        with pytest.raises(Exception):
            GenerationConfig(temperature="not_a_number")

    def test_config_accepts_partial_override(self):
        config = LLMathConfig(
            agent=AgentConfig(default_k=10)
        )
        assert config.agent.default_k == 10
        assert config.embedding.batch_size == 64  # Default preserved
