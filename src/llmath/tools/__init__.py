"""Tools module for LLMath - symbolic computation and utilities."""

from typing import TYPE_CHECKING

from .base import BaseTool, ToolResult
from .registry import ToolRegistry, create_default_registry, create_react_registry
from .sympy_tools import (
    DifferentiateTool,
    IntegrateTool,
    SimplifyTool,
    SolveTool,
    differentiate_expr,
    integrate_expr,
    simplify_expr,
    solve_equation,
)

# RetrieveTool requires faiss; lazy import to avoid breaking tests
if TYPE_CHECKING:
    from .retrieval_tool import RetrieveTool

__all__ = [
    "BaseTool",
    "ToolResult",
    "SimplifyTool",
    "SolveTool",
    "DifferentiateTool",
    "IntegrateTool",
    "RetrieveTool",
    "simplify_expr",
    "solve_equation",
    "differentiate_expr",
    "integrate_expr",
    "ToolRegistry",
    "create_default_registry",
    "create_react_registry",
]


def __getattr__(name: str):
    """Lazy import for heavy dependencies."""
    if name == "RetrieveTool":
        from .retrieval_tool import RetrieveTool

        return RetrieveTool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
