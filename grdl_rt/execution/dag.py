# -*- coding: utf-8 -*-
"""
DAG Utilities — Safe condition evaluator and topological sort.

Provides a restricted expression evaluator for conditional step execution
and a level-aware topological sort for determining parallel execution
groups in a workflow DAG.

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-11
"""

# Standard library
import ast
import operator
from typing import Any, Dict


# Allowed comparison operators
_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

# Allowed binary operators
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}


def evaluate_condition(expression: str, context: Dict[str, Any]) -> bool:
    """Safely evaluate a condition expression against a context.

    Uses Python's ``ast`` module to parse the expression and walks
    the tree manually, only allowing safe operations (comparisons,
    boolean logic, arithmetic, attribute/key access).  Function calls,
    imports, and other potentially unsafe constructs are rejected.

    Parameters
    ----------
    expression : str
        A Python expression string, e.g. ``"metadata.band_count > 1"``.
    context : Dict[str, Any]
        Variables available in the expression.  Dotted attribute
        access (``metadata.band_count``) resolves via dict key
        lookup first, then ``getattr`` as fallback.

    Returns
    -------
    bool
        The boolean result of evaluating the expression.

    Raises
    ------
    ValueError
        If the expression contains disallowed constructs (function
        calls, imports, assignments, etc.).
    """
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Invalid condition expression: {e}") from e

    result = _eval_node(tree.body, context)
    return bool(result)


def _eval_node(node: ast.AST, context: Dict[str, Any]) -> Any:
    """Recursively evaluate an AST node against the context."""

    # Constants: numbers, strings, booleans, None
    if isinstance(node, ast.Constant):
        return node.value

    # Variable names
    if isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        # Support Python builtins True/False/None as names (Python 3.7 compat)
        if node.id == 'True':
            return True
        if node.id == 'False':
            return False
        if node.id == 'None':
            return None
        raise ValueError(
            f"Unknown variable '{node.id}' in condition expression"
        )

    # Dotted attribute access: metadata.band_count
    if isinstance(node, ast.Attribute):
        value = _eval_node(node.value, context)
        attr = node.attr
        # Try dict key first, then getattr
        if isinstance(value, dict):
            if attr in value:
                return value[attr]
        return getattr(value, attr)

    # Subscript: metadata["key"]
    if isinstance(node, ast.Subscript):
        value = _eval_node(node.value, context)
        key = _eval_node(node.slice, context)
        return value[key]

    # Comparisons: a > b, a == b, a in b, etc.
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, context)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, context)
            op_func = _CMP_OPS.get(type(op))
            if op_func is None:
                raise ValueError(
                    f"Unsupported comparison operator: {type(op).__name__}"
                )
            if not op_func(left, right):
                return False
            left = right
        return True

    # Boolean operators: and, or
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_eval_node(v, context) for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(_eval_node(v, context) for v in node.values)
        raise ValueError(
            f"Unsupported boolean operator: {type(node.op).__name__}"
        )

    # Unary operators: not, -, +
    if isinstance(node, ast.UnaryOp):
        op_func = _UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(
                f"Unsupported unary operator: {type(node.op).__name__}"
            )
        return op_func(_eval_node(node.operand, context))

    # Binary operators: +, -, *, /, //, %, **
    if isinstance(node, ast.BinOp):
        op_func = _BIN_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(
                f"Unsupported binary operator: {type(node.op).__name__}"
            )
        left = _eval_node(node.left, context)
        right = _eval_node(node.right, context)
        return op_func(left, right)

    # Ternary: a if cond else b
    if isinstance(node, ast.IfExp):
        test = _eval_node(node.test, context)
        if test:
            return _eval_node(node.body, context)
        return _eval_node(node.orelse, context)

    # Tuple/List literals (for 'in' checks)
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt, context) for elt in node.elts)
    if isinstance(node, ast.List):
        return [_eval_node(elt, context) for elt in node.elts]

    # Reject everything else (Call, Lambda, Import, etc.)
    raise ValueError(
        f"Disallowed construct in condition expression: "
        f"{type(node).__name__}"
    )
