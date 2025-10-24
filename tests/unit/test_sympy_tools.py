"""Tests for SymPy tools."""

from llmath.tools.sympy_tools import (
    SimplifyTool,
    SolveTool,
    DifferentiateTool,
    IntegrateTool,
    simplify_expr,
    solve_equation,
    differentiate_expr,
    integrate_expr,
)
from llmath.tools.registry import ToolRegistry, create_default_registry


class TestSimplifyTool:
    """Tests for the simplify tool."""

    def test_simplify_basic_fraction(self):
        tool = SimplifyTool()
        result = tool.execute("(x**2 - 1) / (x - 1)")
        assert result.success
        assert result.output == "x + 1"

    def test_simplify_trig_identity(self):
        tool = SimplifyTool()
        result = tool.execute("sin(x)**2 + cos(x)**2")
        assert result.success
        assert result.output == "1"

    def test_simplify_invalid_syntax(self):
        tool = SimplifyTool()
        result = tool.execute("invalid(((")
        assert not result.success
        assert result.error is not None


class TestSolveTool:
    """Tests for the solve tool."""

    def test_solve_quadratic_with_equals(self):
        tool = SolveTool()
        result = tool.execute("x**2 - 1 = 0")
        assert result.success
        assert "-1" in result.output
        assert "1" in result.output

    def test_solve_quadratic_implicit_zero(self):
        tool = SolveTool()
        result = tool.execute("x**2 - 4")
        assert result.success
        assert "-2" in result.output
        assert "2" in result.output

    def test_solve_linear(self):
        tool = SolveTool()
        result = tool.execute("2*x + 4 = 0")
        assert result.success
        assert "-2" in result.output

    def test_solve_no_solution(self):
        tool = SolveTool()
        result = tool.execute("x**2 + 1 = 0")
        assert result.success
        # Complex solutions or empty list


class TestDifferentiateTool:
    """Tests for the differentiate tool."""

    def test_diff_polynomial(self):
        tool = DifferentiateTool()
        result = tool.execute("x**3 + 3*x")
        assert result.success
        assert "3*x**2" in result.output
        assert "3" in result.output

    def test_diff_product(self):
        tool = DifferentiateTool()
        result = tool.execute("x**2*sin(x)")
        assert result.success
        # Result should contain both terms from product rule

    def test_diff_trig(self):
        tool = DifferentiateTool()
        result = tool.execute("sin(x)")
        assert result.success
        assert "cos" in result.output


class TestIntegrateTool:
    """Tests for the integrate tool."""

    def test_integrate_polynomial(self):
        tool = IntegrateTool()
        result = tool.execute("2*x")
        assert result.success
        assert "x**2" in result.output

    def test_integrate_trig(self):
        tool = IntegrateTool()
        result = tool.execute("cos(x)")
        assert result.success
        assert "sin" in result.output

    def test_integrate_constant(self):
        tool = IntegrateTool()
        result = tool.execute("5")
        assert result.success
        assert "5*x" in result.output


class TestLegacyFunctions:
    """Tests for legacy function-based API."""

    def test_simplify_expr_function(self):
        result = simplify_expr("(x**2 - 1) / (x - 1)")
        assert result == "x + 1"

    def test_solve_equation_function(self):
        result = solve_equation("x**2 - 1 = 0")
        assert "-1" in result
        assert "1" in result

    def test_differentiate_expr_function(self):
        result = differentiate_expr("x**3")
        assert "3*x**2" in result

    def test_integrate_expr_function(self):
        result = integrate_expr("2*x")
        assert "x**2" in result


class TestToolRegistry:
    """Tests for the tool registry."""

    def test_create_default_registry(self):
        registry = create_default_registry()
        assert "simplify" in registry.list_tools()
        assert "solve" in registry.list_tools()
        assert "diff" in registry.list_tools()
        assert "integrate" in registry.list_tools()

    def test_dispatch_with_prefix(self):
        registry = ToolRegistry()
        name, args = registry.dispatch("solve: x**2 - 1")
        assert name == "solve"
        assert args == "x**2 - 1"

    def test_dispatch_without_prefix(self):
        registry = ToolRegistry()
        name, args = registry.dispatch("x**2 + 1")
        assert name == "simplify"
        assert args == "x**2 + 1"

    def test_execute_via_registry(self):
        registry = create_default_registry()
        result = registry.execute("diff: x**2")
        assert result.success
        assert "2*x" in result.output

    def test_unknown_tool_error(self):
        registry = ToolRegistry()
        result = registry.execute("unknown_tool: x")
        assert not result.success
        assert "Unknown tool" in result.error

    def test_format_result(self):
        registry = create_default_registry()
        formatted = registry.format_result("simplify: x + x")
        assert "simplify" in formatted
        assert "2*x" in formatted
        assert "->" in formatted
