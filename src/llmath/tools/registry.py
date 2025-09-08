"""Tool registry for dynamic dispatch of tool calls."""

from typing import Optional

from .base import BaseTool, ToolResult


class ToolRegistry:
    """Registry for dynamically dispatching tool calls.

    Allows registering tools by name and executing them via string commands.

    Example:
        >>> registry = create_default_registry()
        >>> result = registry.execute("diff: x**2*sin(x)")
        >>> print(result.output)
        x*(x*cos(x) + 2*sin(x))
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool in the registry.

        Args:
            tool: Tool instance to register. Uses tool.name as the key.
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            name: Name of the tool to retrieve.

        Returns:
            Tool instance or None if not found.
        """
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names.
        """
        return list(self._tools.keys())

    def dispatch(self, command: str) -> tuple[str, str]:
        """Parse a command string into tool name and arguments.

        Commands have the format "tool_name: arguments" (e.g., "solve: x**2 - 1").
        If no prefix is provided, defaults to "simplify".

        Args:
            command: Command string to parse.

        Returns:
            Tuple of (tool_name, args).
        """
        command = command.strip()
        if ":" in command:
            name, args = command.split(":", 1)
            return name.strip(), args.strip()
        return "simplify", command

    def execute(self, command: str) -> ToolResult:
        """Parse and execute a command string.

        Args:
            command: Command string (e.g., "diff: x**2*sin(x)").

        Returns:
            ToolResult from the executed tool.
        """
        tool_name, args = self.dispatch(command)
        tool = self.get(tool_name)

        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")

        return tool.execute(args)

    def format_result(self, command: str) -> str:
        """Execute a command and format the result as a string.

        Args:
            command: Command string to execute.

        Returns:
            Formatted string like "diff(x**2, x) -> 2*x".
        """
        tool_name, args = self.dispatch(command)
        result = self.execute(command)

        if result.success:
            return f"{tool_name}({args}) -> {result.output}"
        return f"{tool_name}({args}) -> [error: {result.error}]"


def create_default_registry() -> ToolRegistry:
    """Create a registry with all built-in SymPy tools.

    Returns:
        ToolRegistry with simplify, solve, diff, and integrate tools.
    """
    from .sympy_tools import (
        SimplifyTool,
        SolveTool,
        DifferentiateTool,
        IntegrateTool,
    )

    registry = ToolRegistry()
    registry.register(SimplifyTool())
    registry.register(SolveTool())
    registry.register(DifferentiateTool())
    registry.register(IntegrateTool())

    return registry
