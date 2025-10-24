"""
LLMath - Theorem-aware mathematical proof assistant with retrieval-augmented generation.

This package provides tools for mathematical reasoning by combining:
- Retrieval-augmented generation over the NaturalProofs corpus
- Symbolic computation via SymPy
- Fine-tuned DeepSeek-Math model for proof-style answers
"""

__version__ = "0.1.0"
__author__ = "Shree Chaturvedi"

# Configuration
from .config import (
    AgentConfig,
    EmbeddingConfig,
    GenerationConfig,
    LLMathConfig,
    ModelConfig,
    ReActConfig,
    RetrieverConfig,
    TrainingConfig,
    load_config,
)

# Core components (lazy imports to avoid heavy dependencies at import time)


def __getattr__(name):
    """Lazy import heavy components."""
    if name == "NaturalProofsRetriever":
        from .retrieval import NaturalProofsRetriever

        return NaturalProofsRetriever
    elif name == "TheoremKB":
        from .retrieval.theorem_kb import TheoremKB

        return TheoremKB
    elif name == "MathAgent":
        from .agent import MathAgent

        return MathAgent
    elif name == "create_math_agent":
        from .agent import create_math_agent

        return create_math_agent
    elif name == "ReActAgent":
        from .agent import ReActAgent

        return ReActAgent
    elif name == "create_react_agent":
        from .agent import create_react_agent

        return create_react_agent
    elif name == "create_agent":
        from .agent import create_agent

        return create_agent
    elif name == "DeepSeekMathModel":
        from .inference import DeepSeekMathModel

        return DeepSeekMathModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Configuration
    "LLMathConfig",
    "EmbeddingConfig",
    "RetrieverConfig",
    "ModelConfig",
    "GenerationConfig",
    "AgentConfig",
    "ReActConfig",
    "TrainingConfig",
    "load_config",
    # Core components (lazy)
    "NaturalProofsRetriever",
    "TheoremKB",
    "MathAgent",
    "create_math_agent",
    "ReActAgent",
    "create_react_agent",
    "create_agent",
    "DeepSeekMathModel",
]
