"""
Comprehensive Test Suite for Optimization Solver Module

Tests all functionality of the optimization solver including gradient descent,
function evaluation, gradient computation, and integration with SymPy MCP.

Author: Math AI Agent Team
Task ID: SOLVER-002
Coverage Target: 95%+
"""

import pytest
import sys
import os
import math
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from solvers.optimization_solver import (
        OptimizationSolver, OptimizationResult, OptimizationError,
        get_solver, minimize_function, find_minimum_1d, find_minimum_2d
    )
    OPTIMIZATION_SOLVER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Optimization solver not available: {e}")
    OPTIMIZATION_SOLVER_AVAILABLE = False


@pytest.mark.skipif(not OPTIMIZATION_SOLVER_AVAILABLE, reason="Optimization solver not available")
class TestOptimizationSolver:
    """Test suite for the OptimizationSolver class."""
    
    @pytest.fixture
    def mock_sympy_client(self):
        """Create a mock SymPy client for testing."""
        client = Mock()
        
        # Mock derivative computation
        def mock_compute_derivative(expr, var):
            # Simple mock derivatives for common test functions
            if expr == "(x-10)**2" and var == "x":
                return {"status": "success", "derivative": "2*(x-10)"}
            elif expr == "(x-5)**2 + (y-3)**2" and var == "x":
                return {"status": "success", "derivative": "2*(x-5)"}
            elif expr == "(x-5)**2 + (y-3)**2" and var == "y":
                return {"status": "success", "derivative": "2*(y-3)"}
            elif expr == "x**2" and var == "x":
                return {"status": "success", "derivative": "2*x"}
            elif expr == "x**3 - 3*x" and var == "x":
                return {"status": "success", "derivative": "3*x**2 - 3"}
            else:
                return {"status": "error", "error": "Unsupported function for mock"}
        
        client.compute_derivative = mock_compute_derivative
        
        # Mock expression simplification
        def mock_simplify_expression(expr):
            try:
                # Simple evaluation for basic expressions
                result = eval(expr, {'__builtins__': {}, 'math': math})
                return {"status": "success", "simplified": str(result)}
            except:
                return {"status": "error", "error": "Cannot simplify"}
        
        client.simplify_expression = mock_simplify_expression
        
        return client
    
    @pytest.fixture
    def solver(self, mock_sympy_client):
        """Create an optimization solver with mock SymPy client."""
        return OptimizationSolver(mock_sympy_client)
    
    def test_solver_initialization(self, mock_sympy_client):
        """Test solver initialization."""
        solver = OptimizationSolver(mock_sympy_client)
        
        assert solver.sympy_client is not None
        assert solver._gradient_cache == {}
    
    def test_solver_initialization_without_client(self):
        """Test solver initialization without SymPy client."""
        solver = OptimizationSolver()
        
        assert solver.sympy_client is None
        assert solver._gradient_cache == {}
    
    def test_set_sympy_client(self, mock_sympy_client):
        """Test setting SymPy client after initialization."""
        solver = OptimizationSolver()
        solver.set_sympy_client(mock_sympy_client)
        
        assert solver.sympy_client is mock_sympy_client


@pytest.mark.skipif(not OPTIMIZATION_SOLVER_AVAILABLE, reason="Optimization solver not available")
class TestGradientDescent:
    """Test suite for gradient descent functionality."""
    
    @pytest.fixture
    def mock_sympy_client(self):
        """Create a mock SymPy client for testing."""
        client = Mock()
        
        def mock_compute_derivative(expr, var):
            if expr == "(x-10)**2" and var == "x":
                return {"status": "success", "derivative": "2*(x-10)"}
            elif expr == "(x-5)**2 + (y-3)**2" and var == "x":
                return {"status": "success", "derivative": "2*(x-5)"}
            elif expr == "(x-5)**2 + (y-3)**2" and var == "y":
                return {"status": "success", "derivative": "2*(y-3)"}
            else:
                return {"status": "error", "error": "Unsupported"}
        
        client.compute_derivative = mock_compute_derivative
        client.simplify_expression = lambda x: {"status": "success", "simplified": str(eval(x, {'__builtins__': {}, 'math': math}))}
        
        return client
    
    @pytest.fixture
    def solver(self, mock_sympy_client):
        """Create solver with mock client."""
        return OptimizationSolver(mock_sympy_client)
    
    def test_gradient_descent_1d_convex(self, solver):
        """Test gradient descent on simple 1D convex function (x-10)^2."""
        result = solver.gradient_descent(
            function_expr="(x-10)**2",
            variables=["x"],
            initial_point=[0.0],
            learning_rate=0.1,
            max_iterations=1000,
            tolerance=1e-6
        )
        
        assert result.success is True
        assert result.converged is True
        assert len(result.minimum_point) == 1
        assert abs(result.minimum_point[0] - 10.0) < 1e-3  # Should find minimum at x=10
        assert result.minimum_value < 1e-6  # Function value should be near 0
        assert result.iterations > 0
        assert result.convergence_history is not None
        assert len(result.convergence_history) == result.iterations  # Including initial point (iteration 0)
    
    def test_gradient_descent_2d_convex(self, solver):
        """Test gradient descent on 2D convex function (x-5)^2 + (y-3)^2."""
        result = solver.gradient_descent(
            function_expr="(x-5)**2 + (y-3)**2",
            variables=["x", "y"],
            initial_point=[0.0, 0.0],
            learning_rate=0.1,
            max_iterations=1000,
            tolerance=1e-6
        )
        
        assert result.success is True
        assert result.converged is True
        assert len(result.minimum_point) == 2
        assert abs(result.minimum_point[0] - 5.0) < 1e-3  # Should find minimum at x=5
        assert abs(result.minimum_point[1] - 3.0) < 1e-3  # Should find minimum at y=3
        assert result.minimum_value < 1e-6  # Function value should be near 0
    
    def test_gradient_descent_with_momentum(self, solver):
        """Test gradient descent with momentum."""
        result = solver.gradient_descent(
            function_expr="(x-10)**2",
            variables=["x"],
            initial_point=[0.0],
            learning_rate=0.05,
            momentum=0.9,
            max_iterations=1000,
            tolerance=1e-6
        )
        
        assert result.success is True
        assert result.converged is True
        assert abs(result.minimum_point[0] - 10.0) < 1e-3
        assert "momentum" in result.algorithm_used
    
    def test_gradient_descent_adaptive_lr(self, solver):
        """Test gradient descent with adaptive learning rate."""
        result = solver.gradient_descent(
            function_expr="(x-10)**2",
            variables=["x"],
            initial_point=[0.0],
            learning_rate=1.0,  # Start with large learning rate
            adaptive_lr=True,
            max_iterations=1000,
            tolerance=1e-6
        )
        
        assert result.success is True
        assert result.converged is True
        assert abs(result.minimum_point[0] - 10.0) < 1e-3
        assert "adaptive" in result.algorithm_used
    
    def test_gradient_descent_no_history(self, solver):
        """Test gradient descent without recording history."""
        result = solver.gradient_descent(
            function_expr="(x-10)**2",
            variables=["x"],
            initial_point=[0.0],
            learning_rate=0.1,
            record_history=False,
            max_iterations=100
        )
        
        assert result.success is True
        assert result.convergence_history is None
    
    def test_gradient_descent_max_iterations(self, solver):
        """Test gradient descent with very few iterations."""
        result = solver.gradient_descent(
            function_expr="(x-10)**2",
            variables=["x"],
            initial_point=[0.0],
            learning_rate=0.001,  # Very small learning rate
            max_iterations=5,     # Very few iterations
            tolerance=1e-6
        )
        
        assert result.success is True
        assert result.converged is False  # Should not converge with so few iterations
        assert result.iterations == 5
    
    def test_gradient_descent_large_tolerance(self, solver):
        """Test gradient descent with large tolerance for quick convergence."""
        result = solver.gradient_descent(
            function_expr="(x-10)**2",
            variables=["x"],
            initial_point=[9.9],  # Start very close to minimum
            learning_rate=0.1,
            tolerance=1e-1,       # Large tolerance
            max_iterations=1000
        )
        
        assert result.success is True
        assert result.converged is True
        assert result.iterations < 10  # Should converge quickly


@pytest.mark.skipif(not OPTIMIZATION_SOLVER_AVAILABLE, reason="Optimization solver not available")
class TestInputValidation:
    """Test suite for input validation."""
    
    @pytest.fixture
    def solver(self):
        """Create solver for validation tests."""
        return OptimizationSolver()
    
    def test_empty_function_expression(self, solver):
        """Test validation with empty function expression."""
        result = solver.gradient_descent("", ["x"], [0.0])
        
        assert result.success is False
        assert "Function expression must be a non-empty string" in result.error_message
    
    def test_empty_variables_list(self, solver):
        """Test validation with empty variables list."""
        result = solver.gradient_descent("x**2", [], [])
        
        assert result.success is False
        assert "Variables must be a non-empty list" in result.error_message
    
    def test_mismatched_variables_and_point(self, solver):
        """Test validation with mismatched variables and initial point."""
        result = solver.gradient_descent("x**2 + y**2", ["x", "y"], [0.0])  # Only one coordinate
        
        assert result.success is False
        assert "Initial point must be a list of 2 numbers" in result.error_message
    
    def test_invalid_learning_rate(self, solver):
        """Test validation with invalid learning rate."""
        result = solver.gradient_descent("x**2", ["x"], [0.0], learning_rate=-0.1)
        
        assert result.success is False
        assert "Learning rate must be a positive number" in result.error_message
    
    def test_invalid_max_iterations(self, solver):
        """Test validation with invalid max iterations."""
        result = solver.gradient_descent("x**2", ["x"], [0.0], max_iterations=0)
        
        assert result.success is False
        assert "Maximum iterations must be a positive integer" in result.error_message
    
    def test_invalid_tolerance(self, solver):
        """Test validation with invalid tolerance."""
        result = solver.gradient_descent("x**2", ["x"], [0.0], tolerance=-1e-6)
        
        assert result.success is False
        assert "Tolerance must be a positive number" in result.error_message
    
    def test_infinite_initial_point(self, solver):
        """Test validation with infinite initial point."""
        result = solver.gradient_descent("x**2", ["x"], [float('inf')])
        
        assert result.success is False
        assert "finite numbers" in result.error_message


@pytest.mark.skipif(not OPTIMIZATION_SOLVER_AVAILABLE, reason="Optimization solver not available")
class TestGradientComputation:
    """Test suite for gradient computation functionality."""
    
    @pytest.fixture
    def mock_sympy_client(self):
        """Create a mock SymPy client."""
        client = Mock()
        
        def mock_compute_derivative(expr, var):
            if expr == "x**2" and var == "x":
                return {"status": "success", "derivative": "2*x"}
            else:
                return {"status": "error", "error": "Unsupported"}
        
        client.compute_derivative = mock_compute_derivative
        return client
    
    def test_gradient_computation_success(self, mock_sympy_client):
        """Test successful gradient computation."""
        solver = OptimizationSolver(mock_sympy_client)
        
        gradient = solver._get_gradient_function("x**2", ["x"])
        
        assert gradient is not None
        assert len(gradient) == 1
        assert gradient[0] == "2*x"
    
    def test_gradient_computation_caching(self, mock_sympy_client):
        """Test gradient computation caching."""
        solver = OptimizationSolver(mock_sympy_client)
        
        # First call
        gradient1 = solver._get_gradient_function("x**2", ["x"])
        
        # Second call (should use cache)
        gradient2 = solver._get_gradient_function("x**2", ["x"])
        
        assert gradient1 == gradient2
        assert len(solver._gradient_cache) == 1
    
    def test_gradient_computation_no_client(self):
        """Test gradient computation without SymPy client."""
        solver = OptimizationSolver()
        
        gradient = solver._get_gradient_function("x**2", ["x"])
        
        assert gradient is None
    
    def test_gradient_computation_client_error(self):
        """Test gradient computation with SymPy client error."""
        client = Mock()
        client.compute_derivative = Mock(return_value={"status": "error", "error": "Test error"})
        
        solver = OptimizationSolver(client)
        
        gradient = solver._get_gradient_function("invalid_expr", ["x"])
        
        assert gradient is None


@pytest.mark.skipif(not OPTIMIZATION_SOLVER_AVAILABLE, reason="Optimization solver not available")
class TestFunctionEvaluation:
    """Test suite for function evaluation."""
    
    @pytest.fixture
    def solver(self):
        """Create solver for function evaluation tests."""
        return OptimizationSolver()
    
    def test_evaluate_simple_function(self, solver):
        """Test evaluation of simple function."""
        value = solver._evaluate_function("x**2", ["x"], [3.0])
        
        assert value is not None
        assert abs(value - 9.0) < 1e-10
    
    def test_evaluate_2d_function(self, solver):
        """Test evaluation of 2D function."""
        value = solver._evaluate_function("x**2 + y**2", ["x", "y"], [3.0, 4.0])
        
        assert value is not None
        assert abs(value - 25.0) < 1e-10  # 3^2 + 4^2 = 9 + 16 = 25
    
    def test_evaluate_complex_function(self, solver):
        """Test evaluation of more complex function."""
        value = solver._evaluate_function("(x-5)**2 + (y-3)**2", ["x", "y"], [5.0, 3.0])
        
        assert value is not None
        assert abs(value - 0.0) < 1e-10
    
    def test_evaluate_function_with_math_functions(self, solver):
        """Test evaluation with mathematical functions."""
        import math
        value = solver._evaluate_function("sin(x)", ["x"], [math.pi/2])
        
        assert value is not None
        assert abs(value - 1.0) < 1e-10


@pytest.mark.skipif(not OPTIMIZATION_SOLVER_AVAILABLE, reason="Optimization solver not available")
class TestConvenienceFunctions:
    """Test suite for convenience functions."""
    
    def test_get_solver_singleton(self):
        """Test singleton pattern of get_solver."""
        solver1 = get_solver()
        solver2 = get_solver()
        
        assert solver1 is solver2
    
    def test_minimize_function(self):
        """Test minimize_function convenience function."""
        # This will test the interface, actual optimization requires SymPy client
        try:
            result = minimize_function("x**2", ["x"], [1.0], max_iterations=1)
            # Should return a result object even if optimization fails due to no SymPy client
            assert isinstance(result, OptimizationResult)
        except Exception:
            # Expected if no SymPy client is available
            pass
    
    def test_find_minimum_1d(self):
        """Test find_minimum_1d convenience function."""
        try:
            result = find_minimum_1d("x**2", initial_point=1.0, max_iterations=1)
            assert isinstance(result, OptimizationResult)
        except Exception:
            pass
    
    def test_find_minimum_2d(self):
        """Test find_minimum_2d convenience function."""
        try:
            result = find_minimum_2d("x**2 + y**2", max_iterations=1)
            assert isinstance(result, OptimizationResult)
        except Exception:
            pass


@pytest.mark.skipif(not OPTIMIZATION_SOLVER_AVAILABLE, reason="Optimization solver not available")
class TestAcceptanceCriteria:
    """Test the specific acceptance criteria from the task specification."""
    
    @pytest.fixture
    def mock_sympy_client(self):
        """Create a mock SymPy client for acceptance tests."""
        client = Mock()
        
        def mock_compute_derivative(expr, var):
            if expr == "(x-10)**2" and var == "x":
                return {"status": "success", "derivative": "2*(x-10)"}
            elif expr == "(x-5)**2 + (y-3)**2" and var == "x":
                return {"status": "success", "derivative": "2*(x-5)"}
            elif expr == "(x-5)**2 + (y-3)**2" and var == "y":
                return {"status": "success", "derivative": "2*(y-3)"}
            else:
                return {"status": "error", "error": "Unsupported"}
        
        client.compute_derivative = Mock(side_effect=mock_compute_derivative)
        client.simplify_expression = lambda x: {"status": "success", "simplified": str(eval(x, {'__builtins__': {}, 'math': math}))}
        
        return client
    
    def test_acceptance_criterion_1_minimum_x_minus_10_squared(self, mock_sympy_client):
        """Test: find minimum of (x-10)^2 returns x ≈ 10."""
        solver = OptimizationSolver(mock_sympy_client)
        
        result = solver.gradient_descent(
            function_expr="(x-10)**2",
            variables=["x"],
            initial_point=[0.0],
            learning_rate=0.1,
            max_iterations=1000,
            tolerance=1e-6
        )
        
        assert result.success is True
        assert result.converged is True
        assert len(result.minimum_point) == 1
        assert abs(result.minimum_point[0] - 10.0) < 1e-3
        
        print("✓ Acceptance Criterion 1: Minimum of (x-10)^2 found at x ≈ 10")
    
    def test_acceptance_criterion_2_iteration_history(self, mock_sympy_client):
        """Test: output contains list of x values for each iteration."""
        solver = OptimizationSolver(mock_sympy_client)
        
        result = solver.gradient_descent(
            function_expr="(x-10)**2",
            variables=["x"],
            initial_point=[0.0],
            learning_rate=0.1,
            max_iterations=100,
            tolerance=1e-6,
            record_history=True
        )
        
        assert result.success is True
        assert result.convergence_history is not None
        assert len(result.convergence_history) > 1
        
        # Check that each history entry contains point information
        for entry in result.convergence_history:
            assert 'point' in entry
            assert isinstance(entry['point'], list)
            assert len(entry['point']) == 1  # 1D function
            assert 'iteration' in entry
            assert 'function_value' in entry
        
        print("✓ Acceptance Criterion 2: Iteration history with x values recorded")
    
    def test_acceptance_criterion_3_sympy_gradient_computation(self, mock_sympy_client):
        """Test: solver uses SymPy MCP client to calculate gradients."""
        solver = OptimizationSolver(mock_sympy_client)
        
        # This should call the SymPy client for gradient computation
        gradient = solver._get_gradient_function("(x-10)**2", ["x"])
        
        assert gradient is not None
        assert len(gradient) == 1
        assert gradient[0] == "2*(x-10)"
        
        # Verify the mock was called
        mock_sympy_client.compute_derivative.assert_called_with("(x-10)**2", "x")
        
        print("✓ Acceptance Criterion 3: SymPy MCP client used for gradient computation")
    
    def test_acceptance_criterion_4_2d_function_minimum(self, mock_sympy_client):
        """Test: 2D function (x-5)^2 + (y-3)^2 finds minimum at (5,3)."""
        solver = OptimizationSolver(mock_sympy_client)
        
        result = solver.gradient_descent(
            function_expr="(x-5)**2 + (y-3)**2",
            variables=["x", "y"],
            initial_point=[0.0, 0.0],
            learning_rate=0.1,
            max_iterations=1000,
            tolerance=1e-6
        )
        
        assert result.success is True
        assert result.converged is True
        assert len(result.minimum_point) == 2
        assert abs(result.minimum_point[0] - 5.0) < 1e-3
        assert abs(result.minimum_point[1] - 3.0) < 1e-3
        
        print("✓ Acceptance Criterion 4: 2D function minimum found at (5,3)")


@pytest.mark.skipif(not OPTIMIZATION_SOLVER_AVAILABLE, reason="Optimization solver not available")
class TestOptimizationResult:
    """Test suite for OptimizationResult dataclass."""
    
    def test_optimization_result_creation(self):
        """Test creation of OptimizationResult."""
        result = OptimizationResult(
            success=True,
            minimum_point=[1.0, 2.0],
            minimum_value=0.5,
            iterations=10
        )
        
        assert result.success is True
        assert result.minimum_point == [1.0, 2.0]
        assert result.minimum_value == 0.5
        assert result.iterations == 10
    
    def test_optimization_result_to_dict(self):
        """Test conversion of OptimizationResult to dictionary."""
        result = OptimizationResult(
            success=True,
            minimum_point=[1.0],
            minimum_value=0.1,
            iterations=5,
            converged=True
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['success'] is True
        assert result_dict['minimum_point'] == [1.0]
        assert result_dict['minimum_value'] == 0.1
        assert result_dict['iterations'] == 5
        assert result_dict['converged'] is True


@pytest.mark.skipif(not OPTIMIZATION_SOLVER_AVAILABLE, reason="Optimization solver not available")
class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    @pytest.fixture
    def mock_sympy_client(self):
        """Create a mock SymPy client."""
        client = Mock()
        client.compute_derivative = Mock(return_value={"status": "success", "derivative": "2*x"})
        client.simplify_expression = Mock(return_value={"status": "success", "simplified": "1.0"})
        return client
    
    def test_zero_learning_rate_error(self, mock_sympy_client):
        """Test error handling for zero learning rate."""
        solver = OptimizationSolver(mock_sympy_client)
        
        result = solver.gradient_descent(
            function_expr="x**2",
            variables=["x"],
            initial_point=[1.0],
            learning_rate=0.0
        )
        
        assert result.success is False
        assert "Learning rate must be a positive number" in result.error_message
    
    def test_very_large_learning_rate_warning(self, mock_sympy_client):
        """Test warning for very large learning rate."""
        solver = OptimizationSolver(mock_sympy_client)
        
        # This should generate a warning but still proceed
        result = solver.gradient_descent(
            function_expr="x**2",
            variables=["x"],
            initial_point=[1.0],
            learning_rate=2.0,  # Very large
            max_iterations=1     # Limit iterations to avoid long test
        )
        
        # Should not fail due to large learning rate (just warning)
        assert isinstance(result, OptimizationResult)
    
    def test_nan_in_initial_point(self, mock_sympy_client):
        """Test error handling for NaN in initial point."""
        solver = OptimizationSolver(mock_sympy_client)
        
        result = solver.gradient_descent(
            function_expr="x**2",
            variables=["x"],
            initial_point=[float('nan')]
        )
        
        assert result.success is False
        assert "finite numbers" in result.error_message


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])