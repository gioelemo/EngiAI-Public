"""
Arithmetic tools for mathematical operations.
"""

from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: first integer
        b: second integer

    Returns:
        The sum of a and b
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: first integer
        b: second integer

    Returns:
        The product of a and b
    """
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers.

    Args:
        a: first integer (numerator)
        b: second integer (denominator)

    Returns:
        The quotient of a divided by b

    Raises:
        ZeroDivisionError: If b is zero
    """
    if b == 0:
        raise ZeroDivisionError
    return a / b
