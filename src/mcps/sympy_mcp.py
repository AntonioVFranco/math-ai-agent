"""
SymPy Mathematical Computing Protocol (MCP) Server

This module implements a comprehensive MCP server for the SymPy library,
providing advanced symbolic computation capabilities for the math-ai-agent.

Author: MathBoardAI Agent Team
Task ID: MCP-001
"""

import asyncio
import logging
import random
import traceback
from typing import Any, Dict, List, Optional, Union
from functools import wraps
import time

try:
    from asyncio import timeout as asyncio_timeout
except ImportError:
    from asyncio_timeout import timeout as asyncio_timeout

import sympy as sp
from sympy import (
    symbols, solve, simplify, integrate, diff, latex, Matrix,
    sympify, SympifyError, Float, I, oo, nan, zoo,
    parse_expr, Eq, sqrt, sin, cos, tan, exp, log, pi, E
)
from sympy.core.sympify import SympifyError
from sympy.matrices.common import NonInvertibleMatrixError

# MCP imports
try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    import mcp.server.stdio
except ImportError:
    # Fallback for development/testing without full MCP
    class Server:
        def __init__(self, name: str, version: str):
            self.name = name
            self.version = version
            self.tools = {}
        
        def list_tools(self):
            return list(self.tools.values())
        
        def call_tool(self, name: str, arguments: Dict):
            return self.tools[name](**arguments)
    
    class Tool:
        def __init__(self, name: str, description: str, inputSchema: Dict):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema
    
    class TextContent:
        def __init__(self, type: str, text: str):
            self.type = type
            self.text = text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global constants
DEFAULT_TIMEOUT = 30.0  # seconds
MAX_MEMORY_MB = 100
MAX_VARIABLES = 20


def timeout_handler(timeout_seconds: float = DEFAULT_TIMEOUT):
    """Decorator to add timeout functionality to async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                async with asyncio_timeout(timeout_seconds):
                    return await func(*args, **kwargs)
            except asyncio.TimeoutError:
                logger.error(f"Timeout occurred in {func.__name__} after {timeout_seconds}s")
                return {
                    "result_latex": "",
                    "result_sympy": "",
                    "status": "error",
                    "error_message": f"Operation timed out after {timeout_seconds} seconds"
                }
        return wrapper
    return decorator


def error_handler(func):
    """Decorator to handle common SymPy errors gracefully."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except SympifyError as e:
            logger.error(f"SympifyError in {func.__name__}: {str(e)}")
            return {
                "result_latex": "",
                "result_sympy": "",
                "status": "error",
                "error_message": f"Invalid mathematical expression: {str(e)}"
            }
        except ValueError as e:
            logger.error(f"ValueError in {func.__name__}: {str(e)}")
            return {
                "result_latex": "",
                "result_sympy": "",
                "status": "error",
                "error_message": f"Invalid input value: {str(e)}"
            }
        except NonInvertibleMatrixError as e:
            logger.error(f"NonInvertibleMatrixError in {func.__name__}: {str(e)}")
            return {
                "result_latex": "",
                "result_sympy": "",
                "status": "error",
                "error_message": f"Matrix is not invertible: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "result_latex": "",
                "result_sympy": "",
                "status": "error",
                "error_message": f"Unexpected error: {str(e)}"
            }
    return wrapper


class SymPyMCPServer:
    """
    SymPy Mathematical Computing Protocol Server
    
    Provides advanced symbolic computation capabilities through MCP interface.
    Supports equation solving, expression simplification, calculus operations,
    matrix manipulations, and numerical verification.
    """
    
    def __init__(self, name: str = "sympy-mcp", version: str = "1.0.0"):
        """
        Initialize the SymPy MCP Server.
        
        Args:
            name: Server name identifier
            version: Server version string
        """
        self.server = Server(name, version)
        self.name = name
        self.version = version
        self._setup_tools()
        logger.info(f"Initialized {name} v{version}")
    
    def _setup_tools(self) -> None:
        """Register all available mathematical tools with the MCP server."""
        # Define tool schemas
        expression_schema = {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression as a string"
                }
            },
            "required": ["expression"]
        }
        
        derivative_schema = {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to differentiate"
                },
                "variable": {
                    "type": "string",
                    "description": "Variable to differentiate with respect to"
                }
            },
            "required": ["expression", "variable"]
        }
        
        integral_schema = {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to integrate"
                },
                "variable": {
                    "type": "string",
                    "description": "Variable to integrate with respect to"
                },
                "lower_limit": {
                    "type": "string",
                    "description": "Lower limit for definite integral (optional)"
                },
                "upper_limit": {
                    "type": "string",
                    "description": "Upper limit for definite integral (optional)"
                }
            },
            "required": ["expression", "variable"]
        }
        
        matrix_schema = {
            "type": "object",
            "properties": {
                "matrix_data": {
                    "type": "string",
                    "description": "Matrix data as string representation"
                },
                "operation": {
                    "type": "string",
                    "description": "Matrix operation: det, inv, transpose, eigenvals, eigenvects"
                }
            },
            "required": ["matrix_data", "operation"]
        }
        
        verification_schema = {
            "type": "object",
            "properties": {
                "original_expression": {
                    "type": "string",
                    "description": "Original mathematical expression"
                },
                "symbolic_result": {
                    "type": "string",
                    "description": "Symbolic result to verify"
                },
                "num_tests": {
                    "type": "integer",
                    "description": "Number of random test points (default: 10)"
                }
            },
            "required": ["original_expression", "symbolic_result"]
        }
        
        # Register tools
        tools = [
            Tool("solve_equation", "Solve algebraic and transcendental equations", expression_schema),
            Tool("simplify_expression", "Simplify mathematical expressions", expression_schema),
            Tool("compute_derivative", "Calculate derivatives of expressions", derivative_schema),
            Tool("compute_integral", "Calculate integrals of expressions", integral_schema),
            Tool("matrix_operations", "Perform matrix operations", matrix_schema),
            Tool("numerical_verification", "Numerically verify symbolic results", verification_schema),
            Tool("to_latex", "Convert expression to LaTeX format", expression_schema)
        ]
        
        for tool in tools:
            self.server.tools[tool.name] = tool
    
    def to_latex(self, expr: Union[sp.Basic, str]) -> str:
        """
        Convert a SymPy expression to LaTeX format.
        
        Args:
            expr: SymPy expression or string representation
            
        Returns:
            LaTeX formatted string
        """
        try:
            if isinstance(expr, str):
                expr = sympify(expr)
            return latex(expr)
        except Exception as e:
            logger.error(f"LaTeX conversion error: {str(e)}")
            return str(expr)
    
    @timeout_handler(DEFAULT_TIMEOUT)
    @error_handler
    async def solve_equation(self, expression: str) -> Dict[str, Any]:
        """
        Solve algebraic and transcendental equations.
        
        Args:
            expression: Mathematical equation as string (e.g., "x**2 - 4 = 0")
            
        Returns:
            Dictionary containing solutions in both SymPy and LaTeX formats
        """
        logger.info(f"Solving equation: {expression}")
        
        # Parse the expression
        if '=' in expression:
            left, right = expression.split('=', 1)
            equation = Eq(sympify(left.strip()), sympify(right.strip()))
        else:
            equation = sympify(expression)
        
        # Find all symbols in the equation
        free_symbols = equation.free_symbols
        
        if len(free_symbols) > MAX_VARIABLES:
            return {
                "result_latex": "",
                "result_sympy": "",
                "status": "error",
                "error_message": f"Too many variables ({len(free_symbols)}). Maximum allowed: {MAX_VARIABLES}"
            }
        
        # Solve the equation
        solutions = solve(equation, free_symbols)
        
        # Format results
        if isinstance(solutions, dict):
            result_latex = "\\{" + ", ".join([f"{self.to_latex(k)}: {self.to_latex(v)}" 
                                            for k, v in solutions.items()]) + "\\}"
        elif isinstance(solutions, list):
            result_latex = "\\{" + ", ".join([self.to_latex(sol) for sol in solutions]) + "\\}"
        else:
            result_latex = self.to_latex(solutions)
        
        return {
            "result_latex": result_latex,
            "result_sympy": str(solutions),
            "status": "success",
            "error_message": "",
            "free_symbols": [str(sym) for sym in free_symbols],
            "num_solutions": len(solutions) if isinstance(solutions, (list, dict)) else 1
        }
    
    @timeout_handler(DEFAULT_TIMEOUT)
    @error_handler
    async def simplify_expression(self, expression: str) -> Dict[str, Any]:
        """
        Simplify mathematical expressions.
        
        Args:
            expression: Mathematical expression as string
            
        Returns:
            Dictionary containing simplified expression in both SymPy and LaTeX formats
        """
        logger.info(f"Simplifying expression: {expression}")
        
        # Parse and simplify
        expr = sympify(expression)
        simplified = simplify(expr)
        
        return {
            "result_latex": self.to_latex(simplified),
            "result_sympy": str(simplified),
            "status": "success",
            "error_message": "",
            "original_latex": self.to_latex(expr),
            "original_sympy": str(expr),
            "is_simplified": str(simplified) != str(expr)
        }
    
    @timeout_handler(DEFAULT_TIMEOUT)
    @error_handler
    async def compute_derivative(self, expression: str, variable: str) -> Dict[str, Any]:
        """
        Calculate derivatives of expressions.
        
        Args:
            expression: Mathematical expression to differentiate
            variable: Variable to differentiate with respect to
            
        Returns:
            Dictionary containing derivative in both SymPy and LaTeX formats
        """
        logger.info(f"Computing derivative of {expression} with respect to {variable}")
        
        # Parse expression and variable
        expr = sympify(expression)
        var = symbols(variable)
        
        # Check if variable is in the expression
        if var not in expr.free_symbols:
            logger.warning(f"Variable {variable} not found in expression {expression}")
        
        # Compute derivative
        derivative = diff(expr, var)
        simplified_derivative = simplify(derivative)
        
        return {
            "result_latex": self.to_latex(simplified_derivative),
            "result_sympy": str(simplified_derivative),
            "status": "success",
            "error_message": "",
            "original_latex": self.to_latex(expr),
            "variable": variable,
            "unsimplified_derivative": str(derivative)
        }
    
    @timeout_handler(DEFAULT_TIMEOUT)
    @error_handler
    async def compute_integral(self, expression: str, variable: str, 
                             lower_limit: Optional[str] = None, 
                             upper_limit: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate integrals of expressions.
        
        Args:
            expression: Mathematical expression to integrate
            variable: Variable to integrate with respect to
            lower_limit: Lower limit for definite integral (optional)
            upper_limit: Upper limit for definite integral (optional)
            
        Returns:
            Dictionary containing integral in both SymPy and LaTeX formats
        """
        logger.info(f"Computing integral of {expression} with respect to {variable}")
        
        # Parse expression and variable
        expr = sympify(expression)
        var = symbols(variable)
        
        # Compute integral
        if lower_limit is not None and upper_limit is not None:
            # Definite integral
            lower = sympify(lower_limit)
            upper = sympify(upper_limit)
            result = integrate(expr, (var, lower, upper))
            integral_type = "definite"
        else:
            # Indefinite integral
            result = integrate(expr, var)
            integral_type = "indefinite"
        
        # Simplify result
        simplified_result = simplify(result)
        
        return {
            "result_latex": self.to_latex(simplified_result),
            "result_sympy": str(simplified_result),
            "status": "success",
            "error_message": "",
            "original_latex": self.to_latex(expr),
            "variable": variable,
            "integral_type": integral_type,
            "limits": {
                "lower": lower_limit,
                "upper": upper_limit
            } if integral_type == "definite" else None
        }
    
    @timeout_handler(DEFAULT_TIMEOUT)
    @error_handler
    async def matrix_operations(self, matrix_data: str, operation: str) -> Dict[str, Any]:
        """
        Perform matrix operations.
        
        Args:
            matrix_data: Matrix data as string representation
            operation: Operation to perform (det, inv, transpose, eigenvals, eigenvects)
            
        Returns:
            Dictionary containing matrix operation result
        """
        logger.info(f"Performing matrix operation: {operation}")
        
        # Parse matrix data
        try:
            # Handle different matrix input formats
            if matrix_data.startswith('[[') and matrix_data.endswith(']]'):
                # List of lists format
                matrix_list = eval(matrix_data)
                matrix = Matrix(matrix_list)
            else:
                # Try to parse as SymPy Matrix
                matrix = sympify(matrix_data)
                if not isinstance(matrix, Matrix):
                    raise ValueError("Input is not a valid matrix")
        except Exception as e:
            return {
                "result_latex": "",
                "result_sympy": "",
                "status": "error",
                "error_message": f"Invalid matrix format: {str(e)}"
            }
        
        # Perform the requested operation
        result = None
        
        if operation == "det":
            result = matrix.det()
        elif operation == "inv":
            result = matrix.inv()
        elif operation == "transpose":
            result = matrix.T
        elif operation == "eigenvals":
            result = matrix.eigenvals()
        elif operation == "eigenvects":
            result = matrix.eigenvects()
        else:
            return {
                "result_latex": "",
                "result_sympy": "",
                "status": "error",
                "error_message": f"Unsupported operation: {operation}"
            }
        
        return {
            "result_latex": self.to_latex(result),
            "result_sympy": str(result),
            "status": "success",
            "error_message": "",
            "original_matrix_latex": self.to_latex(matrix),
            "operation": operation,
            "matrix_shape": f"{matrix.rows}x{matrix.cols}"
        }
    
    @timeout_handler(DEFAULT_TIMEOUT)
    @error_handler
    async def numerical_verification(self, original_expression: str, 
                                   symbolic_result: str, 
                                   num_tests: int = 10) -> Dict[str, Any]:
        """
        Numerically verify symbolic results by substituting random values.
        
        Args:
            original_expression: Original mathematical expression
            symbolic_result: Symbolic result to verify
            num_tests: Number of random test points to use
            
        Returns:
            Dictionary containing verification results
        """
        logger.info(f"Verifying symbolic result with {num_tests} test points")
        
        # Parse expressions
        original = sympify(original_expression)
        result = sympify(symbolic_result)
        
        # Get all symbols from both expressions
        all_symbols = original.free_symbols.union(result.free_symbols)
        
        if not all_symbols:
            # No variables to substitute - just compare directly
            original_val = complex(original.evalf())
            result_val = complex(result.evalf())
            is_equal = abs(original_val - result_val) < 1e-10
            
            return {
                "result_latex": "\\text{" + ("Verified" if is_equal else "Failed") + "}",
                "result_sympy": str(is_equal),
                "status": "success",
                "error_message": "",
                "verification_passed": is_equal,
                "num_tests": 1,
                "test_results": [{"values": {}, "original": original_val, "result": result_val, "match": is_equal}]
            }
        
        # Perform numerical tests
        test_results = []
        passed_tests = 0
        
        for i in range(num_tests):
            # Generate random values for each symbol
            test_values = {}
            for sym in all_symbols:
                # Generate random value between -10 and 10
                test_values[sym] = random.uniform(-10, 10)
            
            try:
                # Substitute values and evaluate
                original_val = complex(original.subs(test_values).evalf())
                result_val = complex(result.subs(test_values).evalf())
                
                # Check if values are approximately equal
                tolerance = 1e-10
                is_match = abs(original_val - result_val) < tolerance
                
                if is_match:
                    passed_tests += 1
                
                test_results.append({
                    "values": {str(k): v for k, v in test_values.items()},
                    "original": original_val,
                    "result": result_val,
                    "match": is_match
                })
                
            except Exception as e:
                logger.warning(f"Test {i+1} failed with error: {str(e)}")
                test_results.append({
                    "values": {str(k): v for k, v in test_values.items()},
                    "error": str(e),
                    "match": False
                })
        
        verification_passed = passed_tests == num_tests
        success_rate = passed_tests / num_tests
        
        return {
            "result_latex": f"\\text{{Verification: {passed_tests}/{num_tests} tests passed}}",
            "result_sympy": str(verification_passed),
            "status": "success",
            "error_message": "",
            "verification_passed": verification_passed,
            "success_rate": success_rate,
            "num_tests": num_tests,
            "passed_tests": passed_tests,
            "test_results": test_results[:5]  # Return only first 5 for brevity
        }
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a specific tool with given arguments.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        method_map = {
            "solve_equation": self.solve_equation,
            "simplify_expression": self.simplify_expression,
            "compute_derivative": self.compute_derivative,
            "compute_integral": self.compute_integral,
            "matrix_operations": self.matrix_operations,
            "numerical_verification": self.numerical_verification,
            "to_latex": self._to_latex_tool
        }
        
        if name not in method_map:
            return {
                "result_latex": "",
                "result_sympy": "",
                "status": "error",
                "error_message": f"Unknown tool: {name}"
            }
        
        return await method_map[name](**arguments)
    
    async def _to_latex_tool(self, expression: str) -> Dict[str, Any]:
        """Tool wrapper for to_latex method."""
        try:
            latex_result = self.to_latex(expression)
            return {
                "result_latex": latex_result,
                "result_sympy": expression,
                "status": "success",
                "error_message": ""
            }
        except Exception as e:
            return {
                "result_latex": "",
                "result_sympy": "",
                "status": "error",
                "error_message": str(e)
            }
    
    def list_tools(self) -> List[Tool]:
        """Return list of available tools."""
        return list(self.server.tools.values())


# Async main function for running the server
async def main():
    """Main entry point for the SymPy MCP Server."""
    server = SymPyMCPServer()
    logger.info(f"Starting {server.name} v{server.version}")
    
    # In a real MCP implementation, this would start the server
    # For now, we'll just demonstrate the server capabilities
    logger.info("SymPy MCP Server ready")
    
    # Example usage
    test_result = await server.solve_equation("x**2 - 4")
    logger.info(f"Test result: {test_result}")


if __name__ == "__main__":
    asyncio.run(main())