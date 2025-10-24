"""Abstract base classes for tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result from a tool execution.

    Attributes:
        success: Whether the tool executed successfully.
        output: The output string if successful.
        error: Error message if unsuccessful.
    """

    success: bool
    output: str | None = None
    error: str | None = None

    def __str__(self) -> str:
        if self.success:
            return self.output or ""
        return f"[error] {self.error}"


class BaseTool(ABC):
    """Abstract base class for computational tools.

    Tools are callable components that perform specific operations
    (e.g., symbolic computation, lookup, verification).

    Subclasses must define:
        - name: Unique identifier for the tool
        - description: Human-readable description
        - execute(): Perform the tool's operation
    """

    name: str
    description: str

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given arguments.

        Args:
            *args: Positional arguments specific to the tool.
            **kwargs: Keyword arguments specific to the tool.

        Returns:
            ToolResult indicating success/failure and output.
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> ToolResult:
        """Allow calling the tool directly."""
        return self.execute(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
