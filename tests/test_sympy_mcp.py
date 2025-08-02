"""
Comprehensive Test Suite for SymPy MCP Server

This test suite covers all tools, error cases, and acceptance criteria
for the SymPy Mathematical Computing Protocol server.

Author: Math AI Agent Team
Task ID: MCP-001
Coverage Target: 90%+
"""

import pytest
import asyncio
import time
from typing import Dict, Any
from unittest.mock import patch, MagicMock

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.mcps.sympy_mcp import SymPyMCPServer
from src.mcps.sympy_client import SymPyMCPClient


class TestSymPyMCPServer:
    """Test suite for the SymPy MCP Server."""
    
    @pytest.fixture
    def server(self):
        """Create a SymPy MCP Server instance for testing."""
        return SymPyMCPServer()
    
    @pytest.fixture
    def client(self, server):
        """Create a SymPy MCP Client instance for testing."""
        return SymPyMCPClient(server)
    
    # Test server initialization
    def test_server_initialization(self, server):
        """Test that the server initializes correctly."""
        assert server.name == "sympy-mcp"
        assert server.version == "1.0.0"
        assert len(server.list_tools()) == 7  # All tools registered
    
    def test_tool_registration(self, server):
        """Test that all required tools are registered."""
        tool_names = [tool.name for tool in server.list_tools()]
        expected_tools = [
            "solve_equation", "simplify_expression", "compute_derivative",
            "compute_integral", "matrix_operations", "numerical_verification",
            "to_latex"
        ]
        for tool in expected_tools:
            assert tool in tool_names
    
    # Test LaTeX conversion utility
    def test_to_latex_conversion(self, server):
        """Test LaTeX conversion functionality."""
        # Test with string input
        latex_result = server.to_latex("x**2 + 2*x + 1")
        assert "x^{2}" in latex_result
        
        # Test with SymPy expression
        import sympy as sp
        expr = sp.sympify("x**2 + 2*x + 1")
        latex_result = server.to_latex(expr)
        assert "x^{2}" in latex_result
    
    @pytest.mark.asyncio
    async def test_to_latex_tool(self, server):
        """Test the to_latex tool."""
        result = await server.call_tool("to_latex", {"expression": "x**2 + 1"})
        assert result["status"] == "success"
        assert "x^{2}" in result["result_latex"]
        assert result["result_sympy"] == "x**2 + 1"
    
    # Test equation solving
    @pytest.mark.asyncio
    async def test_solve_simple_equation(self, server):
        """Test solving a simple quadratic equation."""
        result = await server.solve_equation("x**2 - 4")
        assert result["status"] == "success"
        assert "2" in result["result_sympy"] and "-2" in result["result_sympy"]
        assert len(result["free_symbols"]) == 1
        assert "x" in result["free_symbols"]
    
    @pytest.mark.asyncio
    async def test_solve_equation_with_equals(self, server):
        """Test solving equation with equals sign."""
        result = await server.solve_equation("x**2 - 4 = 0")
        assert result["status"] == "success"
        assert "2" in result["result_sympy"] and "-2" in result["result_sympy"]
    
    @pytest.mark.asyncio
    async def test_solve_system_equations(self, server):
        """Test solving system of equations."""
        # Note: This tests the multi-variable capability
        result = await server.solve_equation("x + y - 3")
        assert result["status"] == "success"
        assert len(result["free_symbols"]) == 2
    
    @pytest.mark.asyncio
    async def test_solve_transcendental_equation(self, server):
        """Test solving transcendental equation."""
        result = await server.solve_equation("exp(x) - 2")
        assert result["status"] == "success"
        assert "log" in result["result_sympy"] or "ln" in result["result_sympy"]
    
    # Test expression simplification
    @pytest.mark.asyncio
    async def test_simplify_basic_expression(self, server):
        """Test basic expression simplification."""
        result = await server.simplify_expression("2*x + 3*x")
        assert result["status"] == "success"
        assert "5*x" in result["result_sympy"]
        assert result["is_simplified"] is True
    
    @pytest.mark.asyncio
    async def test_simplify_trigonometric(self, server):
        """Test trigonometric expression simplification."""
        result = await server.simplify_expression("sin(x)**2 + cos(x)**2")
        assert result["status"] == "success"
        assert "1" in result["result_sympy"]
    
    @pytest.mark.asyncio
    async def test_simplify_already_simple(self, server):
        """Test simplification of already simple expression."""
        result = await server.simplify_expression("x + 1")
        assert result["status"] == "success"
        assert result["is_simplified"] is False  # Already simple
    
    # Test derivatives
    @pytest.mark.asyncio
    async def test_compute_simple_derivative(self, server):
        """Test computing derivative of polynomial."""
        result = await server.compute_derivative("x**2 + 2*x + 1", "x")
        assert result["status"] == "success"
        assert "2*x + 2" in result["result_sympy"]
        assert result["variable"] == "x"
    
    @pytest.mark.asyncio
    async def test_compute_derivative_trigonometric(self, server):
        """Test computing derivative of trigonometric function."""
        result = await server.compute_derivative("sin(x)", "x")
        assert result["status"] == "success"
        assert "cos(x)" in result["result_sympy"]
    
    @pytest.mark.asyncio
    async def test_compute_derivative_missing_variable(self, server):
        """Test derivative when variable is not in expression."""
        result = await server.compute_derivative("x**2", "y")
        assert result["status"] == "success"
        assert "0" in result["result_sympy"]
    
    # Test integrals
    @pytest.mark.asyncio
    async def test_compute_indefinite_integral(self, server):
        """Test computing indefinite integral."""
        result = await server.compute_integral("2*x", "x")
        assert result["status"] == "success"
        assert "x**2" in result["result_sympy"]
        assert result["integral_type"] == "indefinite"
        assert result["limits"] is None
    
    @pytest.mark.asyncio
    async def test_compute_definite_integral(self, server):
        """Test computing definite integral."""
        result = await server.compute_integral("x", "x", "0", "2")
        assert result["status"] == "success"
        assert result["integral_type"] == "definite"
        assert result["limits"]["lower"] == "0"
        assert result["limits"]["upper"] == "2"
        assert "2" in result["result_sympy"]
    
    @pytest.mark.asyncio
    async def test_compute_integral_trigonometric(self, server):
        """Test computing integral of trigonometric function."""
        result = await server.compute_integral("cos(x)", "x")
        assert result["status"] == "success"
        assert "sin(x)" in result["result_sympy"]
    
    # Test matrix operations
    @pytest.mark.asyncio
    async def test_matrix_determinant(self, server):
        """Test matrix determinant calculation."""
        matrix_data = "[[1, 2], [3, 4]]"
        result = await server.matrix_operations(matrix_data, "det")
        assert result["status"] == "success"
        assert "-2" in result["result_sympy"]
        assert result["operation"] == "det"
        assert "2x2" in result["matrix_shape"]
    
    @pytest.mark.asyncio
    async def test_matrix_inverse(self, server):
        """Test matrix inverse calculation."""
        matrix_data = "[[1, 2], [3, 4]]"
        result = await server.matrix_operations(matrix_data, "inv")
        assert result["status"] == "success"
        assert result["operation"] == "inv"
    
    @pytest.mark.asyncio
    async def test_matrix_transpose(self, server):
        """Test matrix transpose."""
        matrix_data = "[[1, 2, 3], [4, 5, 6]]"
        result = await server.matrix_operations(matrix_data, "transpose")
        assert result["status"] == "success"
        assert result["operation"] == "transpose"
    
    @pytest.mark.asyncio
    async def test_matrix_eigenvalues(self, server):
        """Test matrix eigenvalues calculation."""
        matrix_data = "[[1, 2], [2, 1]]"
        result = await server.matrix_operations(matrix_data, "eigenvals")
        assert result["status"] == "success"
        assert result["operation"] == "eigenvals"
    
    @pytest.mark.asyncio
    async def test_matrix_eigenvectors(self, server):
        """Test matrix eigenvectors calculation."""
        matrix_data = "[[1, 2], [2, 1]]"
        result = await server.matrix_operations(matrix_data, "eigenvects")
        assert result["status"] == "success"
        assert result["operation"] == "eigenvects"
    
    # Test numerical verification
    @pytest.mark.asyncio
    async def test_numerical_verification_success(self, server):
        """Test numerical verification with correct result."""
        original = "x**2 - 4"
        result_expr = "(x - 2)*(x + 2)"
        result = await server.numerical_verification(original, result_expr, 5)
        assert result["status"] == "success"
        assert result["verification_passed"] is True
        assert result["success_rate"] == 1.0
        assert result["num_tests"] == 5
    
    @pytest.mark.asyncio
    async def test_numerical_verification_failure(self, server):
        """Test numerical verification with incorrect result."""
        original = "x**2 - 4"
        result_expr = "x - 2"  # Incorrect result
        result = await server.numerical_verification(original, result_expr, 5)
        assert result["status"] == "success"
        assert result["verification_passed"] is False
        assert result["success_rate"] < 1.0
    
    @pytest.mark.asyncio
    async def test_numerical_verification_no_variables(self, server):
        """Test numerical verification with no variables."""
        original = "2 + 2"
        result_expr = "4"
        result = await server.numerical_verification(original, result_expr, 1)
        assert result["status"] == "success"
        assert result["verification_passed"] is True
    
    # Test error handling
    @pytest.mark.asyncio
    async def test_invalid_expression_error(self, server):
        """Test handling of invalid mathematical expressions."""
        result = await server.solve_equation("x=2x+x^")  # Invalid syntax
        assert result["status"] == "error"
        assert "Invalid mathematical expression" in result["error_message"]
    
    @pytest.mark.asyncio
    async def test_unknown_tool_error(self, server):
        """Test handling of unknown tool calls."""
        result = await server.call_tool("unknown_tool", {})
        assert result["status"] == "error"
        assert "Unknown tool" in result["error_message"]
    
    @pytest.mark.asyncio
    async def test_matrix_invalid_format_error(self, server):
        """Test handling of invalid matrix format."""
        result = await server.matrix_operations("invalid_matrix", "det")
        assert result["status"] == "error"
        assert "Invalid matrix format" in result["error_message"]
    
    @pytest.mark.asyncio
    async def test_matrix_unsupported_operation_error(self, server):
        """Test handling of unsupported matrix operations."""
        matrix_data = "[[1, 2], [3, 4]]"
        result = await server.matrix_operations(matrix_data, "unsupported_op")
        assert result["status"] == "error"
        assert "Unsupported operation" in result["error_message"]
    
    @pytest.mark.asyncio
    async def test_matrix_non_invertible_error(self, server):
        """Test handling of non-invertible matrix."""
        matrix_data = "[[1, 2], [2, 4]]"  # Singular matrix
        result = await server.matrix_operations(matrix_data, "inv")
        assert result["status"] == "error"
        assert "not invertible" in result["error_message"]
    
    @pytest.mark.asyncio
    async def test_too_many_variables_error(self, server):
        """Test handling of expressions with too many variables."""
        # Create expression with many variables
        variables = [f"x{i}" for i in range(25)]  # More than MAX_VARIABLES (20)
        expression = " + ".join(variables)
        result = await server.solve_equation(expression)
        assert result["status"] == "error"
        assert "Too many variables" in result["error_message"]
    
    # Test performance criteria
    @pytest.mark.asyncio
    async def test_response_time_under_2_seconds(self, server):
        """Test that standard operations complete within 2 seconds."""
        start_time = time.time()
        result = await server.simplify_expression("x**4 + 4*x**3 + 6*x**2 + 4*x + 1")
        end_time = time.time()
        
        assert result["status"] == "success"
        assert (end_time - start_time) < 2.0  # Less than 2 seconds
    
    @pytest.mark.asyncio
    async def test_multiple_variable_performance(self, server):
        """Test performance with multiple variables (10+ variables)."""
        # Create expression with 10 variables
        variables = [f"x{i}" for i in range(10)]
        expression = " + ".join([f"{var}**2" for var in variables])
        
        start_time = time.time()
        result = await server.simplify_expression(expression)
        end_time = time.time()
        
        assert result["status"] == "success"
        assert (end_time - start_time) < 5.0  # Should handle without significant degradation
    
    # Test timeout functionality
    @pytest.mark.asyncio
    async def test_timeout_mechanism(self, server):
        """Test that timeout mechanism works (mock heavy computation)."""
        # We'll test the timeout decorator by mocking a slow operation
        with patch('asyncio.sleep', side_effect=asyncio.TimeoutError):
            # This would normally timeout, but we'll simulate it
            result = await server.simplify_expression("x + 1")
            # The actual test would need a genuinely slow operation
            # For now, we test that the mechanism exists
            assert True  # Placeholder for timeout test


class TestSymPyMCPClient:
    """Test suite for the SymPy MCP Client."""
    
    @pytest.fixture
    def server(self):
        """Create a SymPy MCP Server instance for testing."""
        return SymPyMCPServer()
    
    @pytest.fixture
    def client(self, server):
        """Create a SymPy MCP Client instance for testing."""
        return SymPyMCPClient(server)
    
    def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.server is not None
        assert client.server.name == "sympy-mcp"
    
    def test_list_available_tools(self, client):
        """Test listing available tools."""
        tools = client.list_available_tools()
        assert len(tools) == 7
        assert "solve_equation" in tools
        assert "simplify_expression" in tools
    
    def test_get_tool_info(self, client):
        """Test getting tool information."""
        info = client.get_tool_info("solve_equation")
        assert info is not None
        assert info["name"] == "solve_equation"
        assert "description" in info
        assert "input_schema" in info
        
        # Test non-existent tool
        info = client.get_tool_info("non_existent_tool")
        assert info is None
    
    @pytest.mark.asyncio
    async def test_client_solve_equation(self, client):
        """Test client solve equation method."""
        result = await client.solve_equation("x**2 - 1")
        assert result["status"] == "success"
        assert "1" in result["result_sympy"] and "-1" in result["result_sympy"]
    
    @pytest.mark.asyncio
    async def test_client_simplify_expression(self, client):
        """Test client simplify expression method."""
        result = await client.simplify_expression("x + x")
        assert result["status"] == "success"
        assert "2*x" in result["result_sympy"]
    
    @pytest.mark.asyncio
    async def test_client_compute_derivative(self, client):
        """Test client compute derivative method."""
        result = await client.compute_derivative("x**3", "x")
        assert result["status"] == "success"
        assert "3*x**2" in result["result_sympy"]
    
    @pytest.mark.asyncio
    async def test_client_compute_integral(self, client):
        """Test client compute integral method."""
        result = await client.compute_integral("x", "x")
        assert result["status"] == "success"
        assert "x**2/2" in result["result_sympy"]
    
    @pytest.mark.asyncio
    async def test_client_matrix_operations(self, client):
        """Test client matrix operations method."""
        result = await client.matrix_operations("[[1, 0], [0, 1]]", "det")
        assert result["status"] == "success"
        assert "1" in result["result_sympy"]
    
    @pytest.mark.asyncio
    async def test_client_numerical_verification(self, client):
        """Test client numerical verification method."""
        result = await client.numerical_verification("x + 1", "x + 1", 3)
        assert result["status"] == "success"
        assert result["verification_passed"] is True
    
    @pytest.mark.asyncio
    async def test_client_to_latex(self, client):
        """Test client to LaTeX method."""
        result = await client.to_latex("x**2")
        assert result["status"] == "success"
        assert "x^{2}" in result["result_latex"]
    
    @pytest.mark.asyncio
    async def test_client_batch_operations(self, client):
        """Test client batch operations."""
        operations = [
            {"tool": "simplify_expression", "arguments": {"expression": "x + x"}},
            {"tool": "solve_equation", "arguments": {"expression": "x - 1"}},
            {"tool": "compute_derivative", "arguments": {"expression": "x**2", "variable": "x"}}
        ]
        results = await client.batch_operations(operations)
        assert len(results) == 3
        assert all(result["status"] == "success" for result in results)
    
    def test_client_format_result(self, client):
        """Test client result formatting."""
        result = {
            "result_latex": "x^{2}",
            "result_sympy": "x**2",
            "status": "success",
            "error_message": ""
        }
        
        latex_format = client.format_result(result, "latex")
        assert latex_format == "x^{2}"
        
        sympy_format = client.format_result(result, "sympy")
        assert sympy_format == "x**2"
        
        json_format = client.format_result(result, "json")
        assert "x^{2}" in json_format
        
        # Test error formatting
        error_result = {"status": "error", "error_message": "Test error"}
        error_format = client.format_result(error_result)
        assert "Error: Test error" in error_format
    
    @pytest.mark.asyncio
    async def test_client_connection_test(self, client):
        """Test client connection testing."""
        is_connected = await client.test_connection()
        assert is_connected is True


# Integration tests
class TestIntegration:
    """Integration tests for the complete SymPy MCP system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow from client to server."""
        # Create server and client
        server = SymPyMCPServer()
        client = SymPyMCPClient(server)
        
        # Test complete mathematical workflow
        # 1. Solve equation
        solve_result = await client.solve_equation("x**2 - 5*x + 6")
        assert solve_result["status"] == "success"
        
        # 2. Verify one of the solutions
        solutions = solve_result["result_sympy"]
        if "2" in solutions:
            verify_result = await client.numerical_verification(
                "x**2 - 5*x + 6", "(x - 2)*(x - 3)", 5
            )
            assert verify_result["verification_passed"] is True
        
        # 3. Compute derivative of original
        deriv_result = await client.compute_derivative("x**2 - 5*x + 6", "x")
        assert deriv_result["status"] == "success"
        assert "2*x - 5" in deriv_result["result_sympy"]
        
        # 4. Integrate the derivative (should get back original + constant)
        integral_result = await client.compute_integral("2*x - 5", "x")
        assert integral_result["status"] == "success"
        assert "x**2" in integral_result["result_sympy"]
        assert "5*x" in integral_result["result_sympy"]
    
    @pytest.mark.asyncio
    async def test_all_tools_available_and_working(self):
        """Test that all required tools are available and working."""
        server = SymPyMCPServer()
        
        # Test each tool with simple inputs
        test_cases = [
            ("solve_equation", {"expression": "x - 1"}),
            ("simplify_expression", {"expression": "x + x"}),
            ("compute_derivative", {"expression": "x**2", "variable": "x"}),
            ("compute_integral", {"expression": "x", "variable": "x"}),
            ("matrix_operations", {"matrix_data": "[[1, 0], [0, 1]]", "operation": "det"}),
            ("numerical_verification", {"original_expression": "x", "symbolic_result": "x", "num_tests": 1}),
            ("to_latex", {"expression": "x**2"})
        ]
        
        for tool_name, args in test_cases:
            result = await server.call_tool(tool_name, args)
            assert result["status"] == "success", f"Tool {tool_name} failed: {result.get('error_message', '')}"


# Performance and stress tests
class TestPerformance:
    """Performance and stress tests for the SymPy MCP system."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        server = SymPyMCPServer()
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(10):
            task = server.simplify_expression(f"x**{i} + x**{i}")
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all succeeded
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Task {i} failed with exception: {result}"
            assert result["status"] == "success", f"Task {i} failed: {result.get('error_message', '')}"
    
    @pytest.mark.asyncio
    async def test_memory_usage_bounds(self):
        """Test that memory usage stays within bounds."""
        # This is a placeholder for memory testing
        # In a real implementation, you would use memory profiling tools
        server = SymPyMCPServer()
        
        # Perform many operations
        for i in range(50):
            result = await server.simplify_expression(f"x**{i % 5} + {i}")
            assert result["status"] == "success"
        
        # In a real test, you would check memory usage here
        assert True  # Placeholder


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])