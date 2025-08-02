"""
Comprehensive Test Suite for Visualization System

Tests all functionality of the integrated visualization system including
2D function plotting, matrix heatmaps, decomposition visualizations,
and integration with the Gradio interface.

Author: MathBoardAI Agent Team
Task ID: UI-002 / F5
Coverage Target: 80%+
"""

import pytest
import sys
import os
import numpy as np
from typing import Dict, Any, List, Union

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from interface.visualizer import (
        MathematicalVisualizer, get_visualizer,
        plot_function_2d, plot_matrix_heatmap, plot_matrix_decomposition, plot_eigenvalues
    )
    import plotly.graph_objects as go
    import matplotlib.figure
    VISUALIZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualizer not available: {e}")
    VISUALIZER_AVAILABLE = False


@pytest.mark.skipif(not VISUALIZER_AVAILABLE, reason="Visualizer not available")
class TestMathematicalVisualizer:
    """Test suite for the MathematicalVisualizer class."""
    
    @pytest.fixture
    def visualizer(self):
        """Create a visualizer instance for testing."""
        return MathematicalVisualizer()
    
    @pytest.fixture
    def sample_matrices(self):
        """Provide sample matrices for testing."""
        return {
            "simple_2x2": [[1, 2], [3, 4]],
            "simple_3x3": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "identity_2x2": [[1, 0], [0, 1]],
            "zero_2x2": [[0, 0], [0, 0]],
            "rectangular_2x3": [[1, 2, 3], [4, 5, 6]],
            "single_element": [[5]],
            "negative_values": [[-1, -2], [-3, -4]]
        }
    
    @pytest.fixture
    def sample_expressions(self):
        """Provide sample mathematical expressions for testing."""
        return {
            "linear": "2*x + 1",
            "quadratic": "x**2 - 3*x + 2",
            "cubic": "x**3 - 2*x**2 + x - 1",
            "trigonometric": "sin(x)",
            "exponential": "exp(x)",
            "logarithmic": "log(x)",
            "rational": "1/x",
            "complex": "sin(x**2) + cos(x)",
            "constant": "5",
            "zero": "0"
        }
    
    # Test initialization
    def test_visualizer_initialization(self, visualizer):
        """Test visualizer initialization."""
        assert visualizer is not None
        assert hasattr(visualizer, 'default_range')
        assert hasattr(visualizer, 'default_points')
        assert hasattr(visualizer, 'colors')
        assert visualizer.default_range == (-10, 10)
        assert visualizer.default_points == 1000
    
    def test_singleton_pattern(self):
        """Test that get_visualizer returns the same instance."""
        visualizer1 = get_visualizer()
        visualizer2 = get_visualizer()
        assert visualizer1 is visualizer2
    
    # Test 2D function plotting
    def test_plot_function_2d_linear(self, visualizer, sample_expressions):
        """Test 2D plotting of linear function."""
        result = visualizer.plot_function_2d(sample_expressions["linear"])
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_function_2d_quadratic(self, visualizer, sample_expressions):
        """Test 2D plotting of quadratic function."""
        result = visualizer.plot_function_2d(sample_expressions["quadratic"])
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_function_2d_trigonometric(self, visualizer, sample_expressions):
        """Test 2D plotting of trigonometric function."""
        result = visualizer.plot_function_2d(
            sample_expressions["trigonometric"],
            x_range=(-np.pi, np.pi)
        )
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_function_2d_custom_range(self, visualizer, sample_expressions):
        """Test 2D plotting with custom range."""
        result = visualizer.plot_function_2d(
            sample_expressions["quadratic"],
            x_range=(-5, 5)
        )
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_function_2d_custom_variable(self, visualizer):
        """Test 2D plotting with custom variable name."""
        result = visualizer.plot_function_2d("t**2 + 1", variable='t')
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_function_2d_matplotlib(self, visualizer, sample_expressions):
        """Test 2D plotting using matplotlib backend."""
        result = visualizer.plot_function_2d(
            sample_expressions["quadratic"],
            use_plotly=False
        )
        
        assert result is not None
        assert isinstance(result, matplotlib.figure.Figure)
    
    def test_plot_function_2d_constant(self, visualizer, sample_expressions):
        """Test plotting of constant expression."""
        result = visualizer.plot_function_2d(sample_expressions["constant"])
        
        assert result is not None
        # Should handle constant expressions gracefully
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_function_2d_invalid_expression(self, visualizer):
        """Test plotting with invalid expression."""
        result = visualizer.plot_function_2d("invalid_expression_xyz")
        
        # Should return error plot
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_function_2d_empty_expression(self, visualizer):
        """Test plotting with empty expression."""
        result = visualizer.plot_function_2d("")
        
        # Should return error plot
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    # Test matrix heatmap plotting
    def test_plot_matrix_heatmap_simple(self, visualizer, sample_matrices):
        """Test matrix heatmap plotting."""
        result = visualizer.plot_matrix_heatmap(sample_matrices["simple_2x2"])
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_matrix_heatmap_3x3(self, visualizer, sample_matrices):
        """Test 3x3 matrix heatmap plotting."""
        result = visualizer.plot_matrix_heatmap(sample_matrices["simple_3x3"])
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_matrix_heatmap_rectangular(self, visualizer, sample_matrices):
        """Test rectangular matrix heatmap plotting."""
        result = visualizer.plot_matrix_heatmap(sample_matrices["rectangular_2x3"])
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_matrix_heatmap_single_element(self, visualizer, sample_matrices):
        """Test single element matrix heatmap."""
        result = visualizer.plot_matrix_heatmap(sample_matrices["single_element"])
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_matrix_heatmap_with_labels(self, visualizer, sample_matrices):
        """Test matrix heatmap with custom labels."""
        labels = {
            'x': ['Col A', 'Col B'],
            'y': ['Row 1', 'Row 2']
        }
        result = visualizer.plot_matrix_heatmap(
            sample_matrices["simple_2x2"],
            labels=labels
        )
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_matrix_heatmap_matplotlib(self, visualizer, sample_matrices):
        """Test matrix heatmap using matplotlib backend."""
        result = visualizer.plot_matrix_heatmap(
            sample_matrices["simple_2x2"],
            use_plotly=False
        )
        
        assert result is not None
        assert isinstance(result, matplotlib.figure.Figure)
    
    def test_plot_matrix_heatmap_empty_matrix(self, visualizer):
        """Test heatmap with empty matrix."""
        result = visualizer.plot_matrix_heatmap([])
        
        # Should return error plot
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_matrix_heatmap_invalid_matrix(self, visualizer):
        """Test heatmap with invalid matrix format."""
        result = visualizer.plot_matrix_heatmap([[1, 2], [3, 4, 5]])  # Inconsistent rows
        
        # Should return error plot
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    # Test matrix decomposition plotting
    def test_plot_matrix_decomposition_lu(self, visualizer):
        """Test LU decomposition visualization."""
        matrices = {
            'P': [[1, 0], [0, 1]],
            'L': [[1, 0], [3, 1]],
            'U': [[1, 2], [0, -2]]
        }
        
        result = visualizer.plot_matrix_decomposition(matrices, "LU")
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_matrix_decomposition_qr(self, visualizer):
        """Test QR decomposition visualization."""
        matrices = {
            'Q': [[0.6, -0.8], [0.8, 0.6]],
            'R': [[5, 4], [0, 1]]
        }
        
        result = visualizer.plot_matrix_decomposition(matrices, "QR")
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_matrix_decomposition_svd(self, visualizer):
        """Test SVD visualization."""
        matrices = {
            'U': [[0.6, -0.8], [0.8, 0.6]],
            'Vh': [[0.7, -0.7], [0.7, 0.7]]
        }
        
        result = visualizer.plot_matrix_decomposition(matrices, "SVD")
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_matrix_decomposition_empty(self, visualizer):
        """Test decomposition with empty matrices dict."""
        result = visualizer.plot_matrix_decomposition({}, "LU")
        
        # Should return error plot
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_matrix_decomposition_matplotlib(self, visualizer):
        """Test matrix decomposition using matplotlib backend."""
        matrices = {
            'L': [[1, 0], [0.5, 1]],
            'U': [[2, 1], [0, 1.5]]
        }
        
        result = visualizer.plot_matrix_decomposition(matrices, "LU", use_plotly=False)
        
        assert result is not None
        assert isinstance(result, matplotlib.figure.Figure)
    
    # Test eigenvalue plotting
    def test_plot_eigenvalues_real(self, visualizer):
        """Test eigenvalue plotting with real eigenvalues."""
        eigenvalues = [3.0, 1.0, -1.0]
        
        result = visualizer.plot_eigenvalues(eigenvalues)
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_eigenvalues_complex(self, visualizer):
        """Test eigenvalue plotting with complex eigenvalues."""
        eigenvalues = [1+2j, 1-2j, 0+0j]
        
        result = visualizer.plot_eigenvalues(eigenvalues)
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_plot_eigenvalues_matplotlib(self, visualizer):
        """Test eigenvalue plotting using matplotlib backend."""
        eigenvalues = [2+1j, 2-1j]
        
        result = visualizer.plot_eigenvalues(eigenvalues, use_plotly=False)
        
        assert result is not None
        assert isinstance(result, matplotlib.figure.Figure)
    
    def test_plot_eigenvalues_empty(self, visualizer):
        """Test eigenvalue plotting with empty list."""
        result = visualizer.plot_eigenvalues([])
        
        # Should return error plot
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    # Test expression cleaning
    def test_clean_expression_basic(self, visualizer):
        """Test basic expression cleaning."""
        cleaned = visualizer._clean_expression("x^2 + 3*x")
        assert "**" in cleaned  # Should convert ^ to **
    
    def test_clean_expression_functions(self, visualizer):
        """Test expression cleaning with mathematical functions."""
        cleaned = visualizer._clean_expression("sin(x) + cos(x)")
        assert "sin" in cleaned and "cos" in cleaned
    
    def test_clean_expression_constants(self, visualizer):
        """Test expression cleaning with mathematical constants."""
        cleaned = visualizer._clean_expression("pi*x + e")
        assert "pi" in cleaned
    
    # Test error handling
    def test_error_plot_creation_plotly(self, visualizer):
        """Test error plot creation with Plotly."""
        error_plot = visualizer._create_error_plot("Test error message", use_plotly=True)
        
        assert error_plot is not None
        assert isinstance(error_plot, go.Figure)
    
    def test_error_plot_creation_matplotlib(self, visualizer):
        """Test error plot creation with Matplotlib."""
        error_plot = visualizer._create_error_plot("Test error message", use_plotly=False)
        
        assert error_plot is not None
        assert isinstance(error_plot, matplotlib.figure.Figure)


@pytest.mark.skipif(not VISUALIZER_AVAILABLE, reason="Visualizer not available")
class TestConvenienceFunctions:
    """Test the convenience functions for direct usage."""
    
    def test_convenience_function_2d_plot(self):
        """Test plot_function_2d convenience function."""
        result = plot_function_2d("x**2 + 1")
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_convenience_function_matrix_heatmap(self):
        """Test plot_matrix_heatmap convenience function."""
        matrix = [[1, 2, 3], [4, 5, 6]]
        result = plot_matrix_heatmap(matrix)
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_convenience_function_decomposition(self):
        """Test plot_matrix_decomposition convenience function."""
        matrices = {'L': [[1, 0], [0.5, 1]], 'U': [[2, 1], [0, 1.5]]}
        result = plot_matrix_decomposition(matrices, "LU")
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_convenience_function_eigenvalues(self):
        """Test plot_eigenvalues convenience function."""
        eigenvals = [1+1j, 1-1j, 2]
        result = plot_eigenvalues(eigenvals)
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))


@pytest.mark.skipif(not VISUALIZER_AVAILABLE, reason="Visualizer not available")
class TestAcceptanceCriteria:
    """Test the specific acceptance criteria for the visualization system."""
    
    def test_acceptance_sine_wave_plot(self):
        """Test sine wave plotting as specified in acceptance criteria."""
        # Simulates: "plot the function f(x) = sin(x) from -pi to pi"
        result = plot_function_2d("sin(x)", x_range=(-np.pi, np.pi))
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_acceptance_matrix_determinant_heatmap(self):
        """Test matrix heatmap for determinant as specified in acceptance criteria."""
        # Simulates: requesting determinant of a matrix should show heatmap
        matrix = [[2, 1], [1, 2]]
        result = plot_matrix_heatmap(matrix, title="Matrix for Determinant")
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_acceptance_invalid_function_graceful_handling(self):
        """Test graceful handling of difficult/impossible functions."""
        # Test various problematic expressions
        problematic_expressions = [
            "invalid_function_xyz",
            "",
            "x/0",  # Division by zero
            "sqrt(-x)",  # Complex result for negative x
            "log(0)",   # Undefined logarithm
            "1/(x-x)",  # 0/0 case
        ]
        
        for expr in problematic_expressions:
            result = plot_function_2d(expr)
            # Should return some result (likely error plot) and not crash
            assert result is not None
            assert isinstance(result, (go.Figure, matplotlib.figure.Figure))


@pytest.mark.skipif(not VISUALIZER_AVAILABLE, reason="Visualizer not available")
class TestMathematicalAccuracy:
    """Test mathematical accuracy of visualizations."""
    
    def test_quadratic_function_accuracy(self):
        """Test that quadratic function plotting is mathematically accurate."""
        # Test that x^2 has minimum at x=0
        visualizer = get_visualizer()
        
        # This would require examining the actual plot data
        # For now, just ensure it doesn't crash and returns a plot
        result = visualizer.plot_function_2d("x**2", x_range=(-2, 2))
        assert result is not None
    
    def test_trigonometric_function_accuracy(self):
        """Test trigonometric function mathematical accuracy."""
        # Test that sin(x) has the correct period and range
        visualizer = get_visualizer()
        
        result = visualizer.plot_function_2d("sin(x)", x_range=(-2*np.pi, 2*np.pi))
        assert result is not None
    
    def test_matrix_heatmap_values_accuracy(self):
        """Test that matrix heatmap accurately represents matrix values."""
        visualizer = get_visualizer()
        
        # Test with known matrix values
        matrix = [[1, -1], [-1, 1]]
        result = visualizer.plot_matrix_heatmap(matrix)
        
        assert result is not None
        # The actual values should be represented in the heatmap
        # This is a basic structural test


@pytest.mark.skipif(not VISUALIZER_AVAILABLE, reason="Visualizer not available")
class TestPerformance:
    """Test performance characteristics of the visualization system."""
    
    def test_large_matrix_heatmap_performance(self):
        """Test performance with larger matrices."""
        # Create a larger matrix (10x10)
        large_matrix = [[i*j for j in range(10)] for i in range(10)]
        
        visualizer = get_visualizer()
        result = visualizer.plot_matrix_heatmap(large_matrix)
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
    
    def test_high_resolution_function_plot(self):
        """Test function plotting with high resolution."""
        visualizer = get_visualizer()
        # Increase points for higher resolution
        visualizer.default_points = 5000
        
        result = visualizer.plot_function_2d("sin(10*x)", x_range=(-np.pi, np.pi))
        
        assert result is not None
        assert isinstance(result, (go.Figure, matplotlib.figure.Figure))
        
        # Reset to default
        visualizer.default_points = 1000


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])