"""
SymPy MCP Client

A client module for interacting with the SymPy MCP server.
Provides a convenient interface for testing and integration purposes.

Author: Math AI Agent Team
Task ID: MCP-001
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import json

from .sympy_mcp import SymPyMCPServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SymPyMCPClient:
    """
    Client for interacting with the SymPy MCP Server.
    
    This client provides a convenient interface for calling SymPy mathematical
    operations through the MCP protocol. It can be used for testing, integration,
    and as a reference implementation for other clients.
    """
    
    def __init__(self, server: Optional[SymPyMCPServer] = None):
        """
        Initialize the SymPy MCP Client.
        
        Args:
            server: Optional SymPy MCP Server instance. If None, creates a new one.
        """
        self.server = server or SymPyMCPServer()
        logger.info(f"Initialized SymPy MCP Client with server {self.server.name}")
    
    async def solve_equation(self, expression: str) -> Dict[str, Any]:
        """
        Solve an equation using the MCP server.
        
        Args:
            expression: Mathematical equation as string
            
        Returns:
            Dictionary containing solution results
        """
        return await self.server.call_tool("solve_equation", {"expression": expression})
    
    async def simplify_expression(self, expression: str) -> Dict[str, Any]:
        """
        Simplify an expression using the MCP server.
        
        Args:
            expression: Mathematical expression as string
            
        Returns:
            Dictionary containing simplification results
        """
        return await self.server.call_tool("simplify_expression", {"expression": expression})
    
    async def compute_derivative(self, expression: str, variable: str) -> Dict[str, Any]:
        """
        Compute derivative using the MCP server.
        
        Args:
            expression: Mathematical expression to differentiate
            variable: Variable to differentiate with respect to
            
        Returns:
            Dictionary containing derivative results
        """
        return await self.server.call_tool("compute_derivative", {
            "expression": expression,
            "variable": variable
        })
    
    async def compute_integral(self, expression: str, variable: str,
                             lower_limit: Optional[str] = None,
                             upper_limit: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute integral using the MCP server.
        
        Args:
            expression: Mathematical expression to integrate
            variable: Variable to integrate with respect to
            lower_limit: Lower limit for definite integral (optional)
            upper_limit: Upper limit for definite integral (optional)
            
        Returns:
            Dictionary containing integral results
        """
        args = {
            "expression": expression,
            "variable": variable
        }
        if lower_limit is not None:
            args["lower_limit"] = lower_limit
        if upper_limit is not None:
            args["upper_limit"] = upper_limit
            
        return await self.server.call_tool("compute_integral", args)
    
    async def matrix_operations(self, matrix_data: str, operation: str) -> Dict[str, Any]:
        """
        Perform matrix operations using the MCP server.
        
        Args:
            matrix_data: Matrix data as string representation
            operation: Operation to perform
            
        Returns:
            Dictionary containing matrix operation results
        """
        return await self.server.call_tool("matrix_operations", {
            "matrix_data": matrix_data,
            "operation": operation
        })
    
    async def numerical_verification(self, original_expression: str,
                                   symbolic_result: str,
                                   num_tests: int = 10) -> Dict[str, Any]:
        """
        Numerically verify symbolic results using the MCP server.
        
        Args:
            original_expression: Original mathematical expression
            symbolic_result: Symbolic result to verify
            num_tests: Number of random test points
            
        Returns:
            Dictionary containing verification results
        """
        return await self.server.call_tool("numerical_verification", {
            "original_expression": original_expression,
            "symbolic_result": symbolic_result,
            "num_tests": num_tests
        })
    
    async def to_latex(self, expression: str) -> Dict[str, Any]:
        """
        Convert expression to LaTeX using the MCP server.
        
        Args:
            expression: Mathematical expression as string
            
        Returns:
            Dictionary containing LaTeX conversion results
        """
        return await self.server.call_tool("to_latex", {"expression": expression})
    
    def list_available_tools(self) -> List[str]:
        """
        Get list of available tools from the server.
        
        Returns:
            List of tool names
        """
        tools = self.server.list_tools()
        return [tool.name for tool in tools]
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary containing tool information or None if not found
        """
        tools = self.server.list_tools()
        for tool in tools:
            if tool.name == tool_name:
                return {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
        return None
    
    async def batch_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform multiple operations in batch.
        
        Args:
            operations: List of operation dictionaries with 'tool' and 'arguments' keys
            
        Returns:
            List of results for each operation
        """
        results = []
        for op in operations:
            try:
                result = await self.server.call_tool(op['tool'], op['arguments'])
                results.append(result)
            except Exception as e:
                results.append({
                    "result_latex": "",
                    "result_sympy": "",
                    "status": "error",
                    "error_message": f"Batch operation failed: {str(e)}"
                })
        return results
    
    def format_result(self, result: Dict[str, Any], format_type: str = "latex") -> str:
        """
        Format a result for display.
        
        Args:
            result: Result dictionary from MCP server
            format_type: Format type ("latex", "sympy", "json")
            
        Returns:
            Formatted result string
        """
        if result["status"] == "error":
            return f"Error: {result['error_message']}"
        
        if format_type == "latex":
            return result.get("result_latex", "")
        elif format_type == "sympy":
            return result.get("result_sympy", "")
        elif format_type == "json":
            return json.dumps(result, indent=2)
        else:
            return str(result)
    
    async def test_connection(self) -> bool:
        """
        Test the connection to the MCP server.
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            result = await self.simplify_expression("2*x + 3*x")
            return result["status"] == "success"
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False


# Convenience functions for direct usage
async def solve(expression: str) -> Dict[str, Any]:
    """Convenience function to solve an equation."""
    client = SymPyMCPClient()
    return await client.solve_equation(expression)


async def simplify(expression: str) -> Dict[str, Any]:
    """Convenience function to simplify an expression."""
    client = SymPyMCPClient()
    return await client.simplify_expression(expression)


async def derivative(expression: str, variable: str) -> Dict[str, Any]:
    """Convenience function to compute a derivative."""
    client = SymPyMCPClient()
    return await client.compute_derivative(expression, variable)


async def integral(expression: str, variable: str, 
                  lower_limit: Optional[str] = None,
                  upper_limit: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to compute an integral."""
    client = SymPyMCPClient()
    return await client.compute_integral(expression, variable, lower_limit, upper_limit)


async def matrix_op(matrix_data: str, operation: str) -> Dict[str, Any]:
    """Convenience function for matrix operations."""
    client = SymPyMCPClient()
    return await client.matrix_operations(matrix_data, operation)


async def verify(original_expression: str, symbolic_result: str, 
                num_tests: int = 10) -> Dict[str, Any]:
    """Convenience function for numerical verification."""
    client = SymPyMCPClient()
    return await client.numerical_verification(original_expression, symbolic_result, num_tests)


# Example usage
async def main():
    """Example client usage."""
    client = SymPyMCPClient()
    
    # Test connection
    print("Testing connection...")
    if await client.test_connection():
        print("✓ Connection successful")
    else:
        print("✗ Connection failed")
        return
    
    # List available tools
    print("\nAvailable tools:")
    for tool_name in client.list_available_tools():
        print(f"  - {tool_name}")
    
    # Example operations
    print("\nExample operations:")
    
    # Solve equation
    print("\n1. Solving x^2 - 4 = 0:")
    result = await client.solve_equation("x**2 - 4")
    print(f"   LaTeX: {result['result_latex']}")
    print(f"   SymPy: {result['result_sympy']}")
    
    # Simplify expression
    print("\n2. Simplifying 2*x + 3*x:")
    result = await client.simplify_expression("2*x + 3*x")
    print(f"   LaTeX: {result['result_latex']}")
    print(f"   SymPy: {result['result_sympy']}")
    
    # Compute derivative
    print("\n3. Derivative of x^2 + 2*x + 1:")
    result = await client.compute_derivative("x**2 + 2*x + 1", "x")
    print(f"   LaTeX: {result['result_latex']}")
    print(f"   SymPy: {result['result_sympy']}")
    
    # Compute integral
    print("\n4. Integral of 2*x:")
    result = await client.compute_integral("2*x", "x")
    print(f"   LaTeX: {result['result_latex']}")
    print(f"   SymPy: {result['result_sympy']}")


if __name__ == "__main__":
    asyncio.run(main())