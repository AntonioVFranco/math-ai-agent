"""
SymPy MCP Usage Examples

This script demonstrates how to use the SymPy MCP client to perform
various mathematical operations through the MCP protocol.

Author: Math AI Agent Team
Task ID: MCP-001
"""

import asyncio
import sys
import os

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.mcps.sympy_client import SymPyMCPClient
from src.mcps.sympy_mcp import SymPyMCPServer


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_result(operation: str, result: dict, show_details: bool = True):
    """Print a formatted result."""
    print(f"\n{operation}:")
    print(f"  Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"  LaTeX:  {result['result_latex']}")
        print(f"  SymPy:  {result['result_sympy']}")
        
        if show_details:
            # Print additional details if available
            for key, value in result.items():
                if key not in ['status', 'result_latex', 'result_sympy', 'error_message']:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
    else:
        print(f"  Error: {result['error_message']}")


async def demonstrate_equation_solving():
    """Demonstrate equation solving capabilities."""
    print_section("EQUATION SOLVING")
    
    client = SymPyMCPClient()
    
    # Test cases for equation solving
    equations = [
        ("Simple Quadratic", "x**2 - 4 = 0"),
        ("Cubic Equation", "x**3 - 6*x**2 + 11*x - 6 = 0"),
        ("Transcendental", "exp(x) - 3 = 0"),
        ("System (implicit)", "x**2 + y**2 - 25"),
        ("Trigonometric", "sin(x) - 0.5 = 0"),
        ("Rational", "1/x - 2 = 0")
    ]
    
    for name, equation in equations:
        try:
            result = await client.solve_equation(equation)
            print_result(f"{name}: {equation}", result)
        except Exception as e:
            print(f"Error solving {name}: {e}")


async def demonstrate_expression_simplification():
    """Demonstrate expression simplification capabilities."""
    print_section("EXPRESSION SIMPLIFICATION")
    
    client = SymPyMCPClient()
    
    # Test cases for simplification
    expressions = [
        ("Basic Algebra", "2*x + 3*x - x"),
        ("Trigonometric Identity", "sin(x)**2 + cos(x)**2"),
        ("Factoring", "x**2 - 4"),
        ("Rational Expression", "(x**2 - 1)/(x - 1)"),
        ("Logarithmic", "log(x) + log(y)"),
        ("Complex Expression", "sqrt(x**2 + 2*x + 1)"),
        ("Exponential", "exp(x) * exp(y)"),
        ("Already Simple", "x + 1")
    ]
    
    for name, expression in expressions:
        try:
            result = await client.simplify_expression(expression)
            print_result(f"{name}: {expression}", result)
        except Exception as e:
            print(f"Error simplifying {name}: {e}")


async def demonstrate_calculus():
    """Demonstrate calculus operations (derivatives and integrals)."""
    print_section("CALCULUS OPERATIONS")
    
    client = SymPyMCPClient()
    
    print("\n--- DERIVATIVES ---")
    
    # Test cases for derivatives
    derivative_cases = [
        ("Polynomial", "x**3 + 2*x**2 - 5*x + 3", "x"),
        ("Trigonometric", "sin(x) * cos(x)", "x"),
        ("Exponential", "exp(x**2)", "x"),
        ("Logarithmic", "log(x**2 + 1)", "x"),
        ("Chain Rule", "sin(x**2 + 1)", "x"),
        ("Partial Derivative", "x**2 + y**2", "x"),
        ("Product Rule", "x * sin(x)", "x")
    ]
    
    for name, expression, variable in derivative_cases:
        try:
            result = await client.compute_derivative(expression, variable)
            print_result(f"d/d{variable} [{expression}]", result, False)
        except Exception as e:
            print(f"Error computing derivative {name}: {e}")
    
    print("\n--- INTEGRALS ---")
    
    # Test cases for integrals
    integral_cases = [
        ("Polynomial", "x**2 + 2*x + 1", "x", None, None),
        ("Trigonometric", "sin(x)", "x", None, None),
        ("Exponential", "exp(x)", "x", None, None),
        ("Rational", "1/x", "x", None, None),
        ("Definite Integral", "x", "x", "0", "2"),
        ("Definite Trigonometric", "cos(x)", "x", "0", "pi/2")
    ]
    
    for name, expression, variable, lower, upper in integral_cases:
        try:
            result = await client.compute_integral(expression, variable, lower, upper)
            if lower and upper:
                print_result(f"∫[{lower} to {upper}] {expression} d{variable}", result, False)
            else:
                print_result(f"∫ {expression} d{variable}", result, False)
        except Exception as e:
            print(f"Error computing integral {name}: {e}")


async def demonstrate_matrix_operations():
    """Demonstrate matrix operations."""
    print_section("MATRIX OPERATIONS")
    
    client = SymPyMCPClient()
    
    # Test matrices
    matrices = [
        ("2x2 Identity", "[[1, 0], [0, 1]]"),
        ("2x2 General", "[[1, 2], [3, 4]]"),
        ("3x3 Symmetric", "[[1, 2, 3], [2, 4, 5], [3, 5, 6]]"),
        ("Singular Matrix", "[[1, 2], [2, 4]]")
    ]
    
    operations = ["det", "transpose", "eigenvals", "eigenvects", "inv"]
    
    for matrix_name, matrix_data in matrices:
        print(f"\n--- {matrix_name} ---")
        print(f"Matrix: {matrix_data}")
        
        for operation in operations:
            try:
                result = await client.matrix_operations(matrix_data, operation)
                if result['status'] == 'success':
                    print(f"  {operation.capitalize()}: {result['result_sympy']}")
                else:
                    print(f"  {operation.capitalize()}: Error - {result['error_message']}")
            except Exception as e:
                print(f"  Error computing {operation}: {e}")


async def demonstrate_numerical_verification():
    """Demonstrate numerical verification of symbolic results."""
    print_section("NUMERICAL VERIFICATION")
    
    client = SymPyMCPClient()
    
    # Test cases for verification
    verification_cases = [
        ("Correct Factorization", "x**2 - 4", "(x - 2)*(x + 2)", True),
        ("Incorrect Factorization", "x**2 - 4", "x - 2", False),
        ("Trigonometric Identity", "sin(x)**2 + cos(x)**2", "1", True),
        ("Derivative Check", "x**3", "3*x**2", False),  # This should fail as they're not equal
        ("Algebraic Manipulation", "2*x + 3*x", "5*x", True),
        ("Constant Expression", "2 + 2", "4", True)
    ]
    
    for name, original, result_expr, expected_pass in verification_cases:
        try:
            result = await client.numerical_verification(original, result_expr, 5)
            status = "✓ PASS" if result['verification_passed'] else "✗ FAIL"
            expected = "✓" if expected_pass else "✗"
            match = "✓" if result['verification_passed'] == expected_pass else "✗"
            
            print(f"\n{name}:")
            print(f"  Original: {original}")
            print(f"  Result:   {result_expr}")
            print(f"  Status:   {status} (Expected: {expected}) {match}")
            print(f"  Success Rate: {result['success_rate']:.2%}")
            if result['num_tests'] > 0:
                print(f"  Tests Passed: {result['passed_tests']}/{result['num_tests']}")
                
        except Exception as e:
            print(f"Error verifying {name}: {e}")


async def demonstrate_latex_conversion():
    """Demonstrate LaTeX conversion capabilities."""
    print_section("LATEX CONVERSION")
    
    client = SymPyMCPClient()
    
    # Test cases for LaTeX conversion
    expressions = [
        ("Simple Polynomial", "x**2 + 2*x + 1"),
        ("Fraction", "x/y + 1/2"),
        ("Square Root", "sqrt(x**2 + 1)"),
        ("Trigonometric", "sin(x) + cos(y)"),
        ("Summation", "sum(x**i, (i, 1, n))"),
        ("Greek Letters", "alpha + beta*gamma"),
        ("Integral", "integrate(x**2, x)"),
        ("Matrix", "Matrix([[1, 2], [3, 4]])")
    ]
    
    for name, expression in expressions:
        try:
            result = await client.to_latex(expression)
            print(f"\n{name}:")
            print(f"  Expression: {expression}")
            print(f"  LaTeX:      {result['result_latex']}")
        except Exception as e:
            print(f"Error converting {name}: {e}")


async def demonstrate_batch_operations():
    """Demonstrate batch operations."""
    print_section("BATCH OPERATIONS")
    
    client = SymPyMCPClient()
    
    # Define a batch of operations
    operations = [
        {
            "tool": "solve_equation",
            "arguments": {"expression": "x**2 - 1"}
        },
        {
            "tool": "simplify_expression",
            "arguments": {"expression": "x + x + x"}
        },
        {
            "tool": "compute_derivative",
            "arguments": {"expression": "x**2", "variable": "x"}
        },
        {
            "tool": "compute_integral",
            "arguments": {"expression": "2*x", "variable": "x"}
        },
        {
            "tool": "matrix_operations",
            "arguments": {"matrix_data": "[[1, 2], [3, 4]]", "operation": "det"}
        }
    ]
    
    print("Executing batch operations...")
    try:
        results = await client.batch_operations(operations)
        
        for i, (operation, result) in enumerate(zip(operations, results)):
            print(f"\nOperation {i+1}: {operation['tool']}")
            print(f"  Arguments: {operation['arguments']}")
            print(f"  Status: {result['status']}")
            if result['status'] == 'success':
                print(f"  Result: {result['result_sympy']}")
            else:
                print(f"  Error: {result['error_message']}")
                
    except Exception as e:
        print(f"Error in batch operations: {e}")


async def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    print_section("ERROR HANDLING")
    
    client = SymPyMCPClient()
    
    # Test cases that should produce errors
    error_cases = [
        ("Invalid Expression", "solve_equation", {"expression": "x=2x+x^"}),
        ("Unknown Tool", "unknown_tool", {"expression": "x + 1"}),
        ("Missing Arguments", "compute_derivative", {"expression": "x**2"}),
        ("Invalid Matrix", "matrix_operations", {"matrix_data": "not_a_matrix", "operation": "det"}),
        ("Unsupported Operation", "matrix_operations", {"matrix_data": "[[1, 2], [3, 4]]", "operation": "unknown"}),
        ("Division by Zero", "simplify_expression", {"expression": "1/0"}),
        ("Non-invertible Matrix", "matrix_operations", {"matrix_data": "[[1, 2], [2, 4]]", "operation": "inv"})
    ]
    
    for name, tool, arguments in error_cases:
        try:
            result = await client.server.call_tool(tool, arguments)
            print(f"\n{name}:")
            print(f"  Tool: {tool}")
            print(f"  Arguments: {arguments}")
            print(f"  Status: {result['status']}")
            print(f"  Error: {result['error_message']}")
        except Exception as e:
            print(f"\n{name}:")
            print(f"  Unexpected Exception: {e}")


async def demonstrate_performance_features():
    """Demonstrate performance-related features."""
    print_section("PERFORMANCE FEATURES")
    
    client = SymPyMCPClient()
    
    print("Testing response time for standard operations...")
    
    import time
    
    # Test response times
    test_cases = [
        ("Simple Solve", "solve_equation", {"expression": "x**2 - 9"}),
        ("Complex Simplify", "simplify_expression", {"expression": "((x+1)**3 - (x-1)**3)/2"}),
        ("Derivative", "compute_derivative", {"expression": "sin(x**2)*cos(x)", "variable": "x"}),
        ("Integral", "compute_integral", {"expression": "x*exp(x)", "variable": "x"})
    ]
    
    for name, tool, arguments in test_cases:
        start_time = time.time()
        result = await client.server.call_tool(tool, arguments)
        end_time = time.time()
        
        duration = end_time - start_time
        status = "✓" if duration < 2.0 else "⚠"
        
        print(f"\n{name}:")
        print(f"  Duration: {duration:.3f}s {status}")
        print(f"  Status: {result['status']}")
        if result['status'] == 'success':
            print(f"  Result: {result['result_sympy'][:50]}...")
    
    # Test with multiple variables
    print(f"\nTesting with multiple variables (10 variables)...")
    multi_var_expr = " + ".join([f"x{i}**2" for i in range(10)])
    
    start_time = time.time()
    result = await client.simplify_expression(multi_var_expr)
    end_time = time.time()
    
    duration = end_time - start_time
    status = "✓" if duration < 5.0 else "⚠"
    
    print(f"  Expression: {multi_var_expr}")
    print(f"  Duration: {duration:.3f}s {status}")
    print(f"  Status: {result['status']}")


async def main():
    """Main demonstration function."""
    print("SymPy MCP Usage Examples")
    print("=" * 60)
    print("This script demonstrates all capabilities of the SymPy MCP server.")
    print("Each section shows different mathematical operations and features.")
    
    # Test connection first
    client = SymPyMCPClient()
    print("\nTesting MCP server connection...")
    if await client.test_connection():
        print("✓ MCP server connection successful")
    else:
        print("✗ MCP server connection failed")
        return
    
    # List available tools
    print(f"\nAvailable tools: {', '.join(client.list_available_tools())}")
    
    # Run all demonstrations
    demonstrations = [
        demonstrate_equation_solving,
        demonstrate_expression_simplification,
        demonstrate_calculus,
        demonstrate_matrix_operations,
        demonstrate_numerical_verification,
        demonstrate_latex_conversion,
        demonstrate_batch_operations,
        demonstrate_error_handling,
        demonstrate_performance_features
    ]
    
    for demo in demonstrations:
        try:
            await demo()
        except Exception as e:
            print(f"\nError in {demo.__name__}: {e}")
    
    print_section("DEMONSTRATION COMPLETE")
    print("All SymPy MCP features have been demonstrated.")
    print("Check the output above for results of each operation.")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())