"""Pytest configuration and fixtures for LLMath tests."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_config():
    """Provide a minimal test configuration."""
    from llmath.config import LLMathConfig, RetrieverConfig

    return LLMathConfig(
        retriever=RetrieverConfig(
            index_path=Path("tests/fixtures/test_index.faiss"),
            meta_path=Path("tests/fixtures/test_meta.json"),
        )
    )


@pytest.fixture
def sample_question():
    """Provide a sample math question for testing."""
    return "Prove that the derivative of x**2*sin(x) is 2*x*sin(x) + x**2*cos(x)."


@pytest.fixture
def sample_sympy_expressions():
    """Provide sample SymPy expression commands."""
    return [
        "diff: x**2*sin(x)",
        "solve: x**2 - 1 = 0",
        "simplify: (x**2 - 1) / (x - 1)",
    ]
