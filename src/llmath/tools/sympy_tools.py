"""SymPy-based symbolic computation tools.

Provides tools for:
- Expression simplification
- Equation solving
- Differentiation
- Integration
"""

import logging

from sympy import Eq, diff, integrate, simplify, solve, symbols, sympify
from sympy.core.expr import Expr

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


def _to_expr(expr_str: str) -> Expr:
    """Best-effort conversion from string to SymPy expression.

    Args:
        expr_str: String representation of mathematical expression.

    Returns:
        SymPy Expr object.
    """
    return sympify(expr_str, convert_xor=True)


class SimplifyTool(BaseTool):
    """Tool for simplifying algebraic expressions.

    Example:
        >>> tool = SimplifyTool()
        >>> result = tool.execute("(x**2 - 1) / (x - 1)")
        >>> result.output
        'x + 1'
    """

    name = "simplify"
    description = "Simplify an algebraic expression"

    def execute(self, expr_str: str) -> ToolResult:
        """Simplify the given expression.

        Args:
            expr_str: Expression to simplify (e.g., "(x**2 - 1) / (x - 1)").

        Returns:
            ToolResult with simplified expression or error.
        """
        try:
            expr = _to_expr(expr_str)
            result = str(simplify(expr))
            logger.debug("Simplified expression.")
            return ToolResult(success=True, output=result)
        except Exception as e:
            logger.exception("Simplify tool failed.")
            return ToolResult(success=False, error=f"simplify error: {e}")


class SolveTool(BaseTool):
    """Tool for solving equations.

    Example:
        >>> tool = SolveTool()
        >>> result = tool.execute("x**2 - 1 = 0")
        >>> result.output
        '[-1, 1]'
    """

    name = "solve"
    description = "Solve an equation for a given symbol"

    def execute(self, equation_str: str, symbol: str = "x") -> ToolResult:
        """Solve the equation for the specified symbol.

        Args:
            equation_str: Equation to solve. Can include '=' or be interpreted
                as equal to zero (e.g., "x**2 - 1" means x**2 - 1 = 0).
            symbol: Variable to solve for (default: "x").

        Returns:
            ToolResult with solutions or error.
        """
        try:
            x = symbols(symbol)

            if "=" in equation_str:
                left, right = equation_str.split("=", 1)
                eq = Eq(_to_expr(left), _to_expr(right))
            else:
                eq = Eq(_to_expr(equation_str), 0)

            sol = solve(eq, x)
            logger.debug("Solved equation.")
            return ToolResult(success=True, output=str(sol))
        except Exception as e:
            logger.exception("Solve tool failed.")
            return ToolResult(success=False, error=f"solve error: {e}")


class DifferentiateTool(BaseTool):
    """Tool for computing derivatives.

    Example:
        >>> tool = DifferentiateTool()
        >>> result = tool.execute("sin(x) * x**2")
        >>> result.output
        'x*(x*cos(x) + 2*sin(x))'
    """

    name = "diff"
    description = "Differentiate expression with respect to a symbol"

    def execute(self, expr_str: str, symbol: str = "x") -> ToolResult:
        """Compute the derivative of the expression.

        Args:
            expr_str: Expression to differentiate.
            symbol: Variable to differentiate with respect to (default: "x").

        Returns:
            ToolResult with derivative or error.
        """
        try:
            x = symbols(symbol)
            expr = _to_expr(expr_str)
            d = diff(expr, x)
            result = str(simplify(d))
            logger.debug("Differentiated expression.")
            return ToolResult(success=True, output=result)
        except Exception as e:
            logger.exception("Differentiate tool failed.")
            return ToolResult(success=False, error=f"differentiate error: {e}")


class IntegrateTool(BaseTool):
    """Tool for computing indefinite integrals.

    Example:
        >>> tool = IntegrateTool()
        >>> result = tool.execute("2*x")
        >>> result.output
        'x**2'
    """

    name = "integrate"
    description = "Compute indefinite integral with respect to a symbol"

    def execute(self, expr_str: str, symbol: str = "x") -> ToolResult:
        """Compute the indefinite integral of the expression.

        Args:
            expr_str: Expression to integrate.
            symbol: Variable to integrate with respect to (default: "x").

        Returns:
            ToolResult with integral or error.
        """
        try:
            x = symbols(symbol)
            expr = _to_expr(expr_str)
            integral_val = integrate(expr, x)
            logger.debug("Integrated expression.")
            return ToolResult(success=True, output=str(integral_val))
        except Exception as e:
            logger.exception("Integrate tool failed.")
            return ToolResult(success=False, error=f"integrate error: {e}")


# Legacy function-based API for backward compatibility
def simplify_expr(expr_str: str) -> str:
    """Simplify an algebraic expression (legacy function API)."""
    result = SimplifyTool().execute(expr_str)
    return result.output if result.success else f"[sympy_error] {result.error}"


def solve_equation(equation_str: str, symbol: str = "x") -> str:
    """Solve an equation (legacy function API)."""
    result = SolveTool().execute(equation_str, symbol)
    return result.output if result.success else f"[sympy_error] {result.error}"


def differentiate_expr(expr_str: str, symbol: str = "x") -> str:
    """Differentiate an expression (legacy function API)."""
    result = DifferentiateTool().execute(expr_str, symbol)
    return result.output if result.success else f"[sympy_error] {result.error}"


def integrate_expr(expr_str: str, symbol: str = "x") -> str:
    """Integrate an expression (legacy function API)."""
    result = IntegrateTool().execute(expr_str, symbol)
    return result.output if result.success else f"[sympy_error] {result.error}"
