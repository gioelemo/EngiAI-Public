"""
Code execution tools for running Python code.

This module provides tools for executing Python code using a REPL environment,
allowing agents to run code snippets and see their output.
"""

from langchain_core.tools import tool
from langchain_experimental.tools.python.tool import PythonREPLTool

# Initialize the Python REPL tool
_python_repl = PythonREPLTool()


@tool
def execute_python_code(code: str) -> str:
    """
    Execute Python code and return the output.

    This tool runs Python code in a REPL environment and captures the output.
    Use this when you need to:
    - Perform calculations
    - Test code snippets
    - Run data analysis
    - Generate plots or visualizations
    - Execute any Python code

    Args:
        code: Python code to execute (as a string)

    Returns:
        str: The output from executing the code (stdout/stderr)

    Example:
        >>> result = execute_python_code("print(1+1)")
        >>> print(result)
        2

        >>> result = execute_python_code("import numpy as np\\nprint(np.array([1,2,3]).mean())")
        >>> print(result)
        2.0

    Note:
        - Code is executed in a persistent REPL session, so variables persist between calls
        - Standard libraries are available, but external packages need to be installed
        - Be careful with file operations and system commands
    """
    try:
        result = _python_repl.invoke(code)
        return str(result)
    except Exception as e:
        return f"Error executing code: {e!s}"


@tool
def execute_python_expression(expression: str) -> str:
    """
    Evaluate a Python expression and return the result.

    This is a convenience tool for evaluating single expressions.
    Use this for quick calculations or evaluations.

    Args:
        expression: Python expression to evaluate

    Returns:
        str: The result of evaluating the expression

    Example:
        >>> result = execute_python_expression("2 + 2")
        >>> print(result)
        4

        >>> result = execute_python_expression("[i**2 for i in range(5)]")
        >>> print(result)
        [0, 1, 4, 9, 16]
    """
    try:
        code = f"print({expression})"
        result = _python_repl.invoke(code)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e!s}"
