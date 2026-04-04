"""Safe math evaluation tool — restricts eval to arithmetic and approved functions."""

import ast
import math
import operator
from typing import Any

from langchain_core.tools import tool

# Whitelist of safe operators and functions the calculator may use
_SAFE_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_SAFE_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.expr) -> float:
    """Recursively evaluate an AST node using only whitelisted operations.

    Args:
        node: Parsed AST expression node.

    Returns:
        Numeric result of the expression.

    Raises:
        ValueError: If the expression contains a disallowed operation or name.
    """
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Disallowed constant type: {type(node.value)}")
        return float(node.value)

    if isinstance(node, ast.Name):
        if node.id not in _SAFE_FUNCTIONS:
            raise ValueError(f"Disallowed name: '{node.id}'")
        value = _SAFE_FUNCTIONS[node.id]
        if callable(value):
            raise ValueError(f"'{node.id}' is a function, not a constant")
        return float(value)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS or not callable(_SAFE_FUNCTIONS[func_name]):
            raise ValueError(f"Disallowed function: '{func_name}'")
        args = [_safe_eval(arg) for arg in node.args]
        return float(_SAFE_FUNCTIONS[func_name](*args))

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Disallowed binary operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return float(_SAFE_OPERATORS[op_type](left, right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)  # type: ignore[assignment]
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Disallowed unary operator: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return float(_SAFE_OPERATORS[op_type](operand))

    raise ValueError(f"Disallowed AST node type: {type(node).__name__}")


@tool
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression and return the numeric result.

    Supports basic arithmetic (+, -, *, /, **, %, //), and a restricted set
    of math functions: abs, round, sqrt, log, log10, exp, ceil, floor.
    Constants pi and e are available. No arbitrary Python execution is possible.

    Args:
        expression: A mathematical expression string, e.g. '(1.22 ** 3 - 1) * 100'
            or 'sqrt(2) * log(10)'.

    Returns:
        String representation of the numeric result, or an error message.

    Examples:
        calculate("(1 + 0.18) ** 5 - 1")  -> "1.2388..."  (5yr CAGR from 18% annual)
        calculate("sqrt(252) * 0.15")       -> "2.38..."    (annualised vol)
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        return f"Syntax error in expression: {exc}"

    try:
        result = _safe_eval(tree.body)
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"Calculation error: {exc}"

    return str(result)
