"""
Calculus and Optimization Solver Module

This module provides numerical optimization capabilities with a focus on 
gradient descent algorithms. It enables finding minima of functions and 
visualizing the optimization process, which is fundamental for machine 
learning model training.

Author: MathBoardAI Agent Team
Task ID: SOLVER-002
"""

import math
import time
import logging
from typing import List, Dict, Any, Union, Optional, Tuple, Callable
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationError(Exception):
    """Custom exception for optimization-related errors."""
    pass


@dataclass
class OptimizationResult:
    """Result container for optimization operations."""
    success: bool
    minimum_point: Optional[List[float]] = None
    minimum_value: Optional[float] = None
    iterations: int = 0
    convergence_history: Optional[List[Dict[str, Any]]] = None
    final_gradient_norm: Optional[float] = None
    converged: bool = False
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    algorithm_used: str = "gradient_descent"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'success': self.success,
            'minimum_point': self.minimum_point,
            'minimum_value': self.minimum_value,
            'iterations': self.iterations,
            'convergence_history': self.convergence_history,
            'final_gradient_norm': self.final_gradient_norm,
            'converged': self.converged,
            'error_message': self.error_message,
            'execution_time_ms': self.execution_time_ms,
            'algorithm_used': self.algorithm_used
        }


class OptimizationSolver:
    """
    A comprehensive optimization solver focused on gradient descent algorithms.
    
    This solver can minimize scalar functions of one or multiple variables using
    gradient descent and its variants. It's particularly useful for understanding
    the mathematical foundations of machine learning optimization.
    """
    
    def __init__(self, sympy_client=None):
        """
        Initialize the optimization solver.
        
        Args:
            sympy_client: SymPy MCP client for symbolic differentiation
        """
        self.sympy_client = sympy_client
        self._gradient_cache = {}  # Cache for computed gradients
        
        logger.info("Optimization solver initialized")
    
    def set_sympy_client(self, sympy_client):
        """Set the SymPy MCP client."""
        self.sympy_client = sympy_client
        logger.info("SymPy client set for optimization solver")
    
    def gradient_descent(self, 
                        function_expr: str,
                        variables: List[str],
                        initial_point: List[float],
                        learning_rate: float = 0.1,
                        max_iterations: int = 1000,
                        tolerance: float = 1e-6,
                        adaptive_lr: bool = False,
                        momentum: float = 0.0,
                        record_history: bool = True) -> OptimizationResult:
        """
        Find the minimum of a function using gradient descent.
        
        This is the main optimization function that implements the gradient descent
        algorithm with various enhancements like adaptive learning rate and momentum.
        
        Args:
            function_expr: String representation of the function to minimize
            variables: List of variable names (e.g., ['x'] or ['x', 'y'])
            initial_point: Starting point for optimization
            learning_rate: Step size for gradient descent
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance (gradient norm threshold)
            adaptive_lr: Whether to use adaptive learning rate
            momentum: Momentum factor (0.0 = no momentum)
            record_history: Whether to record optimization history
            
        Returns:
            OptimizationResult containing the optimization results
            
        Examples:
            >>> solver = OptimizationSolver()
            >>> result = solver.gradient_descent("(x-10)**2", ["x"], [0.0])
            >>> print(f"Minimum at x = {result.minimum_point[0]:.6f}")
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_gradient_descent_inputs(
                function_expr, variables, initial_point, learning_rate, 
                max_iterations, tolerance
            )
            
            # Get gradient function using SymPy
            gradient_func = self._get_gradient_function(function_expr, variables)
            if gradient_func is None:
                return OptimizationResult(
                    success=False,
                    error_message="Failed to compute gradient function"
                )
            
            # Initialize optimization variables
            current_point = initial_point.copy()
            velocity = [0.0] * len(variables)  # For momentum
            history = [] if record_history else None
            
            # Initial function evaluation
            initial_value = self._evaluate_function(function_expr, variables, current_point)
            best_point = current_point.copy()
            best_value = initial_value
            
            if record_history:
                history.append({
                    'iteration': 0,
                    'point': current_point.copy(),
                    'function_value': initial_value,
                    'gradient_norm': None,
                    'learning_rate': learning_rate
                })
            
            logger.info(f"Starting gradient descent optimization")
            logger.info(f"Initial point: {current_point}, Initial value: {initial_value:.6f}")
            
            # Main optimization loop
            for iteration in range(1, max_iterations + 1):
                # Compute gradient at current point
                gradient = self._evaluate_gradient(gradient_func, variables, current_point)
                if gradient is None:
                    return OptimizationResult(
                        success=False,
                        error_message=f"Failed to evaluate gradient at iteration {iteration}"
                    )
                
                # Calculate gradient norm for convergence check
                gradient_norm = math.sqrt(sum(g**2 for g in gradient))
                
                # Check for convergence
                if gradient_norm < tolerance:
                    logger.info(f"Converged at iteration {iteration} with gradient norm {gradient_norm:.8f}")
                    converged = True
                    break
                
                # Adaptive learning rate (simple backtracking)
                current_lr = learning_rate
                if adaptive_lr:
                    current_lr = self._adaptive_learning_rate(
                        function_expr, variables, current_point, gradient, learning_rate
                    )
                
                # Update point using gradient descent with momentum
                for i in range(len(variables)):
                    # Momentum update
                    velocity[i] = momentum * velocity[i] - current_lr * gradient[i]
                    current_point[i] += velocity[i]
                
                # Evaluate function at new point
                current_value = self._evaluate_function(function_expr, variables, current_point)
                
                # Track best point found so far
                if current_value < best_value:
                    best_point = current_point.copy()
                    best_value = current_value
                
                # Record history
                if record_history:
                    history.append({
                        'iteration': iteration,
                        'point': current_point.copy(),
                        'function_value': current_value,
                        'gradient': gradient.copy(),
                        'gradient_norm': gradient_norm,
                        'learning_rate': current_lr
                    })
                
                # Log progress periodically
                if iteration % 100 == 0 or iteration <= 10:
                    logger.debug(f"Iteration {iteration}: point={current_point}, "
                               f"value={current_value:.6f}, gradient_norm={gradient_norm:.8f}")
            
            else:
                # Maximum iterations reached
                logger.warning(f"Maximum iterations ({max_iterations}) reached without convergence")
                converged = False
                gradient_norm = math.sqrt(sum(g**2 for g in gradient)) if 'gradient' in locals() else None
            
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            
            # Use best point found during optimization
            final_point = best_point
            final_value = best_value
            final_gradient = self._evaluate_gradient(gradient_func, variables, final_point)
            final_gradient_norm = math.sqrt(sum(g**2 for g in final_gradient)) if final_gradient else None
            
            logger.info(f"Optimization completed in {execution_time_ms:.2f}ms")
            logger.info(f"Final point: {final_point}, Final value: {final_value:.6f}")
            
            return OptimizationResult(
                success=True,
                minimum_point=final_point,
                minimum_value=final_value,
                iterations=iteration if converged else max_iterations,
                convergence_history=history,
                final_gradient_norm=final_gradient_norm,
                converged=converged,
                execution_time_ms=execution_time_ms,
                algorithm_used=f"gradient_descent{'_momentum' if momentum > 0 else ''}{'_adaptive' if adaptive_lr else ''}"
            )
            
        except OptimizationError as e:
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            
            logger.error(f"Optimization error: {e}")
            return OptimizationResult(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )
        
        except Exception as e:
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            
            logger.error(f"Unexpected error in gradient descent: {e}")
            return OptimizationResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                execution_time_ms=execution_time_ms
            )
    
    def _get_gradient_function(self, function_expr: str, variables: List[str]) -> Optional[List[str]]:
        """
        Compute the gradient of a function using SymPy MCP client.
        
        Args:
            function_expr: String representation of the function
            variables: List of variable names
            
        Returns:
            List of gradient expressions as strings, or None if computation fails
        """
        try:
            # Check cache first
            cache_key = (function_expr, tuple(variables))
            if cache_key in self._gradient_cache:
                logger.debug("Using cached gradient")
                return self._gradient_cache[cache_key]
            
            if not self.sympy_client:
                raise OptimizationError("SymPy client not available for gradient computation")
            
            gradient_expressions = []
            
            # Compute partial derivative for each variable
            for var in variables:
                logger.debug(f"Computing partial derivative with respect to {var}")
                
                result = self.sympy_client.compute_derivative(function_expr, var)
                
                if not result or result.get('status') != 'success':
                    error_msg = result.get('error', 'Unknown error') if result else 'No result'
                    raise OptimizationError(f"Failed to compute derivative w.r.t. {var}: {error_msg}")
                
                derivative_expr = result.get('derivative')
                if not derivative_expr:
                    raise OptimizationError(f"No derivative expression returned for variable {var}")
                
                gradient_expressions.append(derivative_expr)
                logger.debug(f"∂/∂{var}: {derivative_expr}")
            
            # Cache the result
            self._gradient_cache[cache_key] = gradient_expressions
            
            logger.info(f"Successfully computed gradient with {len(gradient_expressions)} components")
            return gradient_expressions
            
        except Exception as e:
            logger.error(f"Error computing gradient: {e}")
            return None
    
    def _evaluate_function(self, function_expr: str, variables: List[str], point: List[float]) -> Optional[float]:
        """
        Evaluate a function at a specific point.
        
        Args:
            function_expr: String representation of the function
            variables: List of variable names
            point: Point at which to evaluate the function
            
        Returns:
            Function value at the point, or None if evaluation fails
        """
        try:
            # Create substitution dictionary
            substitutions = {var: val for var, val in zip(variables, point)}
            
            # Use SymPy to evaluate the expression
            if self.sympy_client:
                # Create a substituted expression
                substituted_expr = function_expr
                for var, val in substitutions.items():
                    substituted_expr = substituted_expr.replace(var, str(val))
                
                # Use SymPy to evaluate
                result = self.sympy_client.simplify_expression(substituted_expr)
                if result and result.get('status') == 'success':
                    simplified = result.get('simplified')
                    try:
                        return float(simplified)
                    except (ValueError, TypeError):
                        # Try to evaluate as a mathematical expression
                        return eval(simplified, {'__builtins__': {}, 'math': math})
            
            # Fallback: direct evaluation with Python
            # Replace variable names with values
            eval_expr = function_expr
            for var, val in substitutions.items():
                eval_expr = eval_expr.replace(var, str(val))
            
            # Replace common mathematical functions
            eval_expr = eval_expr.replace('^', '**')
            
            # Safe evaluation
            allowed_names = {
                '__builtins__': {},
                'math': math,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'exp': math.exp,
                'sqrt': math.sqrt,
                'abs': abs,
                'pow': pow
            }
            
            return eval(eval_expr, allowed_names)
            
        except Exception as e:
            logger.error(f"Error evaluating function at {point}: {e}")
            return None
    
    def _evaluate_gradient(self, gradient_exprs: List[str], variables: List[str], point: List[float]) -> Optional[List[float]]:
        """
        Evaluate gradient expressions at a specific point.
        
        Args:
            gradient_exprs: List of gradient expressions
            variables: List of variable names
            point: Point at which to evaluate gradients
            
        Returns:
            List of gradient values, or None if evaluation fails
        """
        try:
            gradient_values = []
            
            for i, grad_expr in enumerate(gradient_exprs):
                value = self._evaluate_function(grad_expr, variables, point)
                if value is None:
                    logger.error(f"Failed to evaluate gradient component {i} at {point}")
                    return None
                gradient_values.append(value)
            
            return gradient_values
            
        except Exception as e:
            logger.error(f"Error evaluating gradient at {point}: {e}")
            return None
    
    def _adaptive_learning_rate(self, function_expr: str, variables: List[str], 
                               current_point: List[float], gradient: List[float], 
                               base_lr: float) -> float:
        """
        Compute adaptive learning rate using simple backtracking line search.
        
        Args:
            function_expr: Function expression
            variables: Variable names
            current_point: Current optimization point
            gradient: Current gradient
            base_lr: Base learning rate
            
        Returns:
            Adjusted learning rate
        """
        try:
            current_value = self._evaluate_function(function_expr, variables, current_point)
            if current_value is None:
                return base_lr
            
            # Try the base learning rate first
            lr = base_lr
            alpha = 0.5  # Backtracking factor
            c1 = 1e-4    # Sufficient decrease parameter
            
            for _ in range(5):  # Maximum 5 backtracking steps
                # Compute new point with current learning rate
                new_point = [current_point[i] - lr * gradient[i] for i in range(len(variables))]
                new_value = self._evaluate_function(function_expr, variables, new_point)
                
                if new_value is None:
                    lr *= alpha
                    continue
                
                # Check sufficient decrease condition (Armijo condition)
                gradient_dot_direction = sum(g**2 for g in gradient)  # Since direction = -gradient
                expected_decrease = c1 * lr * gradient_dot_direction
                
                if new_value <= current_value - expected_decrease:
                    return lr
                
                # Reduce learning rate
                lr *= alpha
            
            return lr
            
        except Exception as e:
            logger.debug(f"Error in adaptive learning rate computation: {e}")
            return base_lr
    
    def _validate_gradient_descent_inputs(self, function_expr: str, variables: List[str], 
                                        initial_point: List[float], learning_rate: float,
                                        max_iterations: int, tolerance: float) -> None:
        """
        Validate inputs for gradient descent.
        
        Args:
            function_expr: Function expression to validate
            variables: Variable names to validate
            initial_point: Initial point to validate
            learning_rate: Learning rate to validate
            max_iterations: Maximum iterations to validate
            tolerance: Tolerance to validate
            
        Raises:
            OptimizationError: If any input is invalid
        """
        if not function_expr or not isinstance(function_expr, str):
            raise OptimizationError("Function expression must be a non-empty string")
        
        if not variables or not isinstance(variables, list):
            raise OptimizationError("Variables must be a non-empty list")
        
        if not all(isinstance(var, str) and var.strip() for var in variables):
            raise OptimizationError("All variables must be non-empty strings")
        
        if not isinstance(initial_point, list) or len(initial_point) != len(variables):
            raise OptimizationError(f"Initial point must be a list of {len(variables)} numbers")
        
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in initial_point):
            raise OptimizationError("All initial point coordinates must be finite numbers")
        
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise OptimizationError("Learning rate must be a positive number")
        
        if learning_rate > 1.0:
            logger.warning(f"Large learning rate ({learning_rate}) may cause instability")
        
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise OptimizationError("Maximum iterations must be a positive integer")
        
        if max_iterations > 100000:
            logger.warning(f"Large number of iterations ({max_iterations}) may take long time")
        
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            raise OptimizationError("Tolerance must be a positive number")
    
    def find_critical_points(self, function_expr: str, variables: List[str], 
                           search_region: List[Tuple[float, float]] = None,
                           num_starting_points: int = 10) -> Dict[str, Any]:
        """
        Find critical points of a function using multiple random starting points.
        
        Args:
            function_expr: Function expression
            variables: Variable names
            search_region: List of (min, max) tuples for each variable
            num_starting_points: Number of random starting points to try
            
        Returns:
            Dictionary containing found critical points and analysis
        """
        try:
            import random
            
            if search_region is None:
                search_region = [(-10.0, 10.0)] * len(variables)
            
            if len(search_region) != len(variables):
                raise OptimizationError("Search region must match number of variables")
            
            critical_points = []
            unique_points = []
            
            logger.info(f"Searching for critical points with {num_starting_points} starting points")
            
            for i in range(num_starting_points):
                # Generate random starting point
                start_point = [
                    random.uniform(region[0], region[1]) 
                    for region in search_region
                ]
                
                # Run optimization
                result = self.gradient_descent(
                    function_expr, variables, start_point,
                    learning_rate=0.01, max_iterations=1000, tolerance=1e-8
                )
                
                if result.success and result.converged:
                    # Check if this is a new unique point
                    is_unique = True
                    for existing_point in unique_points:
                        if all(abs(a - b) < 1e-4 for a, b in zip(result.minimum_point, existing_point)):
                            is_unique = False
                            break
                    
                    if is_unique:
                        unique_points.append(result.minimum_point)
                        critical_points.append({
                            'point': result.minimum_point,
                            'value': result.minimum_value,
                            'starting_point': start_point,
                            'iterations': result.iterations
                        })
            
            # Sort by function value
            critical_points.sort(key=lambda x: x['value'])
            
            return {
                'success': True,
                'critical_points': critical_points,
                'num_points_found': len(critical_points),
                'global_minimum': critical_points[0] if critical_points else None,
                'search_info': {
                    'num_starting_points': num_starting_points,
                    'search_region': search_region,
                    'function_expr': function_expr,
                    'variables': variables
                }
            }
            
        except Exception as e:
            logger.error(f"Error in critical point search: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }


# Global solver instance
_global_solver = None

def get_solver(sympy_client=None) -> OptimizationSolver:
    """Get a global optimization solver instance (singleton pattern)."""
    global _global_solver
    if _global_solver is None:
        _global_solver = OptimizationSolver(sympy_client)
    elif sympy_client and not _global_solver.sympy_client:
        _global_solver.set_sympy_client(sympy_client)
    return _global_solver


# Convenience functions for direct usage
def minimize_function(function_expr: str, variables: List[str], initial_point: List[float],
                     **kwargs) -> OptimizationResult:
    """
    Convenience function to minimize a function using gradient descent.
    
    Args:
        function_expr: Function to minimize
        variables: Variable names
        initial_point: Starting point
        **kwargs: Additional arguments for gradient_descent
        
    Returns:
        OptimizationResult
    """
    solver = get_solver()
    return solver.gradient_descent(function_expr, variables, initial_point, **kwargs)


def find_minimum_1d(function_expr: str, variable: str = 'x', initial_point: float = 0.0,
                   **kwargs) -> OptimizationResult:
    """
    Convenience function for 1D function minimization.
    
    Args:
        function_expr: 1D function expression
        variable: Variable name (default 'x')
        initial_point: Starting point
        **kwargs: Additional arguments
        
    Returns:
        OptimizationResult
    """
    return minimize_function(function_expr, [variable], [initial_point], **kwargs)


def find_minimum_2d(function_expr: str, variables: List[str] = None, 
                   initial_point: List[float] = None, **kwargs) -> OptimizationResult:
    """
    Convenience function for 2D function minimization.
    
    Args:
        function_expr: 2D function expression
        variables: Variable names (default ['x', 'y'])
        initial_point: Starting point (default [0.0, 0.0])
        **kwargs: Additional arguments
        
    Returns:
        OptimizationResult
    """
    if variables is None:
        variables = ['x', 'y']
    if initial_point is None:
        initial_point = [0.0, 0.0]
    
    return minimize_function(function_expr, variables, initial_point, **kwargs)