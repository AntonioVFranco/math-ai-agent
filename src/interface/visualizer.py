"""
Integrated Visualization System

A comprehensive visualization module that generates and displays interactive plots
and diagrams for mathematical problems and solutions. This system enhances user
understanding by providing visual representations within the Gradio interface.

Author: Math AI Agent Team
Task ID: UI-002 / F5
Integration: Called by engine.py, displayed in app.py
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import sympy as sp
from typing import Optional, Dict, Any, List, Tuple, Union
import warnings
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class MathematicalVisualizer:
    """
    Comprehensive visualization system for mathematical expressions and data.
    
    This visualizer creates interactive plots and diagrams to enhance user
    understanding of mathematical concepts, solutions, and relationships.
    """
    
    def __init__(self):
        """Initialize the visualizer with default settings."""
        # Default plotting parameters
        self.default_range = (-10, 10)
        self.default_points = 1000
        self.figure_size = (10, 6)
        self.dpi = 100
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd'
        }
        
        # Plot style
        plt.style.use('default')
        
        logger.info("Mathematical Visualizer initialized successfully")
    
    def plot_function_2d(self, expression: str, variable: str = 'x', 
                        x_range: Tuple[float, float] = None,
                        title: str = None, use_plotly: bool = True) -> Optional[Union[go.Figure, matplotlib.figure.Figure]]:
        """
        Generate a 2D plot of a mathematical function.
        
        Args:
            expression: Mathematical expression string (e.g., "x**2 - 3*x + 2")
            variable: Variable name (default: 'x')
            x_range: Tuple of (min, max) for x-axis range
            title: Plot title (auto-generated if None)
            use_plotly: Use Plotly (True) or Matplotlib (False)
            
        Returns:
            Plot object compatible with Gradio's gr.Plot component
        """
        try:
            logger.info(f"Creating 2D function plot for: {expression}")
            
            # Set default range if not provided
            if x_range is None:
                x_range = self.default_range
            
            # Parse the mathematical expression
            try:
                # Clean and prepare expression
                cleaned_expr = self._clean_expression(expression)
                
                # Create sympy symbol and expression
                x = sp.Symbol(variable)
                expr = sp.sympify(cleaned_expr)
                
                # Validate that expression contains the variable
                if x not in expr.free_symbols and not expr.is_number:
                    # Try to find any symbols in the expression
                    symbols = expr.free_symbols
                    if symbols:
                        # Use the first symbol found
                        x = list(symbols)[0]
                    else:
                        # Constant expression
                        logger.warning(f"Expression '{expression}' is constant, creating horizontal line")
                
            except (sp.SympifyError, ValueError) as e:
                logger.error(f"Failed to parse expression '{expression}': {e}")
                return self._create_error_plot(f"Invalid expression: {expression}", use_plotly)
            
            # Generate x values
            x_vals = np.linspace(x_range[0], x_range[1], self.default_points)
            
            # Evaluate the expression
            try:
                # Convert sympy expression to numpy-compatible function
                f = sp.lambdify(x, expr, 'numpy')
                y_vals = f(x_vals)
                
                # Handle complex results
                if np.iscomplexobj(y_vals):
                    logger.warning(f"Expression '{expression}' produces complex values, taking real part")
                    y_vals = np.real(y_vals)
                
                # Handle infinite/NaN values
                finite_mask = np.isfinite(y_vals)
                if not np.any(finite_mask):
                    return self._create_error_plot(f"Expression '{expression}' produces no finite values", use_plotly)
                
            except Exception as e:
                logger.error(f"Failed to evaluate expression '{expression}': {e}")
                return self._create_error_plot(f"Cannot evaluate expression: {expression}", use_plotly)
            
            # Create title if not provided
            if title is None:
                title = f"f({variable}) = {expression}"
            
            # Create the plot
            if use_plotly:
                return self._create_plotly_2d_plot(x_vals, y_vals, title, variable, expression)
            else:
                return self._create_matplotlib_2d_plot(x_vals, y_vals, title, variable, expression)
                
        except Exception as e:
            logger.error(f"Unexpected error in plot_function_2d: {e}")
            return self._create_error_plot(f"Plotting error: {str(e)}", use_plotly)
    
    def plot_matrix_heatmap(self, matrix: List[List[float]], title: str = None,
                           labels: Dict[str, List[str]] = None,
                           use_plotly: bool = True) -> Optional[Union[go.Figure, matplotlib.figure.Figure]]:
        """
        Generate a heatmap visualization of a matrix.
        
        Args:
            matrix: 2D list representing the matrix
            title: Plot title (auto-generated if None)
            labels: Dictionary with 'x' and 'y' keys for axis labels
            use_plotly: Use Plotly (True) or Matplotlib (False)
            
        Returns:
            Plot object compatible with Gradio's gr.Plot component
        """
        try:
            logger.info(f"Creating matrix heatmap for {len(matrix)}x{len(matrix[0]) if matrix else 0} matrix")
            
            # Validate matrix
            if not matrix or not matrix[0]:
                return self._create_error_plot("Empty matrix provided", use_plotly)
            
            # Convert to numpy array
            try:
                np_matrix = np.array(matrix, dtype=float)
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to convert matrix to numpy array: {e}")
                return self._create_error_plot("Invalid matrix format", use_plotly)
            
            # Validate matrix shape
            if np_matrix.ndim != 2:
                return self._create_error_plot("Matrix must be 2-dimensional", use_plotly)
            
            rows, cols = np_matrix.shape
            
            # Create title if not provided
            if title is None:
                title = f"Matrix Heatmap ({rows}×{cols})"
            
            # Prepare labels
            if labels is None:
                labels = {
                    'x': [f"Col {i+1}" for i in range(cols)],
                    'y': [f"Row {i+1}" for i in range(rows)]
                }
            
            # Create the plot
            if use_plotly:
                return self._create_plotly_heatmap(np_matrix, title, labels)
            else:
                return self._create_matplotlib_heatmap(np_matrix, title, labels)
                
        except Exception as e:
            logger.error(f"Unexpected error in plot_matrix_heatmap: {e}")
            return self._create_error_plot(f"Heatmap error: {str(e)}", use_plotly)
    
    def plot_matrix_decomposition(self, matrices: Dict[str, List[List[float]]], 
                                 decomposition_type: str,
                                 use_plotly: bool = True) -> Optional[Union[go.Figure, matplotlib.figure.Figure]]:
        """
        Visualize matrix decomposition results (LU, QR, SVD, etc.).
        
        Args:
            matrices: Dictionary of matrices (e.g., {'P': [[...]], 'L': [[...]], 'U': [[...]]})
            decomposition_type: Type of decomposition ('LU', 'QR', 'SVD', etc.)
            use_plotly: Use Plotly (True) or Matplotlib (False)
            
        Returns:
            Plot object with subplots for each matrix
        """
        try:
            logger.info(f"Creating {decomposition_type} decomposition visualization")
            
            if not matrices:
                return self._create_error_plot("No matrices provided for decomposition", use_plotly)
            
            # Filter out non-matrix entries
            valid_matrices = {}
            for name, matrix in matrices.items():
                if isinstance(matrix, list) and matrix and isinstance(matrix[0], list):
                    try:
                        np_matrix = np.array(matrix, dtype=float)
                        if np_matrix.ndim == 2:
                            valid_matrices[name] = np_matrix
                    except:
                        continue
            
            if not valid_matrices:
                return self._create_error_plot("No valid matrices found in decomposition", use_plotly)
            
            if use_plotly:
                return self._create_plotly_decomposition_plot(valid_matrices, decomposition_type)
            else:
                return self._create_matplotlib_decomposition_plot(valid_matrices, decomposition_type)
                
        except Exception as e:
            logger.error(f"Error in plot_matrix_decomposition: {e}")
            return self._create_error_plot(f"Decomposition plot error: {str(e)}", use_plotly)
    
    def plot_eigenvalues(self, eigenvalues: List[complex], eigenvectors: List[List[complex]] = None,
                        use_plotly: bool = True) -> Optional[Union[go.Figure, matplotlib.figure.Figure]]:
        """
        Visualize eigenvalues in the complex plane.
        
        Args:
            eigenvalues: List of eigenvalues (can be complex)
            eigenvectors: Optional list of eigenvectors
            use_plotly: Use Plotly (True) or Matplotlib (False)
            
        Returns:
            Plot object showing eigenvalues in complex plane
        """
        try:
            logger.info(f"Creating eigenvalue visualization for {len(eigenvalues)} eigenvalues")
            
            if not eigenvalues:
                return self._create_error_plot("No eigenvalues provided", use_plotly)
            
            # Convert to numpy array
            evals = np.array(eigenvalues, dtype=complex)
            
            # Extract real and imaginary parts
            real_parts = np.real(evals)
            imag_parts = np.imag(evals)
            
            if use_plotly:
                return self._create_plotly_eigenvalue_plot(real_parts, imag_parts)
            else:
                return self._create_matplotlib_eigenvalue_plot(real_parts, imag_parts)
                
        except Exception as e:
            logger.error(f"Error in plot_eigenvalues: {e}")
            return self._create_error_plot(f"Eigenvalue plot error: {str(e)}", use_plotly)
    
    def _clean_expression(self, expression: str) -> str:
        """Clean and prepare mathematical expression for parsing."""
        # Replace common mathematical notation
        replacements = {
            '^': '**',  # Power notation
            'sin': 'sin',
            'cos': 'cos',
            'tan': 'tan',
            'log': 'log',
            'ln': 'log',
            'sqrt': 'sqrt',
            'abs': 'Abs',
            'pi': 'pi',
            'e': 'E'
        }
        
        cleaned = expression
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned
    
    def _create_plotly_2d_plot(self, x_vals: np.ndarray, y_vals: np.ndarray, 
                              title: str, variable: str, expression: str) -> go.Figure:
        """Create a 2D function plot using Plotly."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name=f'f({variable}) = {expression}',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=variable,
            yaxis_title=f'f({variable})',
            template='plotly_white',
            width=800,
            height=500,
            showlegend=True
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def _create_matplotlib_2d_plot(self, x_vals: np.ndarray, y_vals: np.ndarray,
                                  title: str, variable: str, expression: str) -> matplotlib.figure.Figure:
        """Create a 2D function plot using Matplotlib."""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        ax.plot(x_vals, y_vals, color=self.colors['primary'], linewidth=2, 
                label=f'f({variable}) = {expression}')
        
        ax.set_title(title)
        ax.set_xlabel(variable)
        ax.set_ylabel(f'f({variable})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def _create_plotly_heatmap(self, matrix: np.ndarray, title: str, 
                              labels: Dict[str, List[str]]) -> go.Figure:
        """Create a matrix heatmap using Plotly."""
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=labels['x'],
            y=labels['y'],
            colorscale='RdBu_r',
            showscale=True,
            text=np.round(matrix, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            width=600,
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def _create_matplotlib_heatmap(self, matrix: np.ndarray, title: str,
                                  labels: Dict[str, List[str]]) -> matplotlib.figure.Figure:
        """Create a matrix heatmap using Matplotlib."""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set labels
        ax.set_xticks(range(len(labels['x'])))
        ax.set_yticks(range(len(labels['y'])))
        ax.set_xticklabels(labels['x'])
        ax.set_yticklabels(labels['y'])
        
        # Add text annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(title)
        plt.tight_layout()
        return fig
    
    def _create_plotly_decomposition_plot(self, matrices: Dict[str, np.ndarray], 
                                         decomposition_type: str) -> go.Figure:
        """Create decomposition visualization using Plotly."""
        n_matrices = len(matrices)
        cols = min(3, n_matrices)
        rows = (n_matrices + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=list(matrices.keys()),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for i, (name, matrix) in enumerate(matrices.items()):
            row = i // cols + 1
            col = i % cols + 1
            
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    colorscale='RdBu_r',
                    showscale=(i == 0),  # Only show scale for first plot
                    text=np.round(matrix, 3),
                    texttemplate='%{text}',
                    textfont={"size": 8}
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=f"{decomposition_type} Decomposition",
            width=800,
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def _create_matplotlib_decomposition_plot(self, matrices: Dict[str, np.ndarray],
                                            decomposition_type: str) -> matplotlib.figure.Figure:
        """Create decomposition visualization using Matplotlib."""
        n_matrices = len(matrices)
        cols = min(3, n_matrices)
        rows = (n_matrices + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), dpi=self.dpi)
        if n_matrices == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (name, matrix) in enumerate(matrices.items()):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto')
            ax.set_title(f'Matrix {name}')
            
            # Add text annotations for small matrices
            if matrix.shape[0] <= 5 and matrix.shape[1] <= 5:
                for r in range(matrix.shape[0]):
                    for c in range(matrix.shape[1]):
                        ax.text(c, r, f'{matrix[r, c]:.3f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # Hide unused subplots
        for i in range(n_matrices, rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        fig.suptitle(f"{decomposition_type} Decomposition")
        plt.tight_layout()
        return fig
    
    def _create_plotly_eigenvalue_plot(self, real_parts: np.ndarray, 
                                      imag_parts: np.ndarray) -> go.Figure:
        """Create eigenvalue plot using Plotly."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=real_parts,
            y=imag_parts,
            mode='markers',
            marker=dict(
                size=10,
                color=self.colors['primary'],
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            name='Eigenvalues',
            text=[f'λ = {r:.3f} + {i:.3f}i' for r, i in zip(real_parts, imag_parts)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Eigenvalues in Complex Plane',
            xaxis_title='Real Part',
            yaxis_title='Imaginary Part',
            template='plotly_white',
            width=600,
            height=500,
            showlegend=True
        )
        
        # Add axes lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    def _create_matplotlib_eigenvalue_plot(self, real_parts: np.ndarray,
                                         imag_parts: np.ndarray) -> matplotlib.figure.Figure:
        """Create eigenvalue plot using Matplotlib."""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        ax.scatter(real_parts, imag_parts, color=self.colors['primary'], 
                  s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        ax.set_title('Eigenvalues in Complex Plane')
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.grid(True, alpha=0.3)
        
        # Add axes lines
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Annotate points
        for i, (r, im) in enumerate(zip(real_parts, imag_parts)):
            ax.annotate(f'λ{i+1}', (r, im), xytext=(5, 5), 
                       textcoords='offset points', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def _create_error_plot(self, error_message: str, use_plotly: bool = True) -> Union[go.Figure, matplotlib.figure.Figure]:
        """Create an error plot when visualization fails."""
        if use_plotly:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Visualization Error:<br>{error_message}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red"),
                align="center"
            )
            fig.update_layout(
                title="Visualization Error",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                template='plotly_white',
                width=600,
                height=400
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            ax.text(0.5, 0.5, f"Visualization Error:\n{error_message}",
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_title("Visualization Error")
            ax.axis('off')
            return fig


# Global visualizer instance for efficiency
_global_visualizer = None


def get_visualizer() -> MathematicalVisualizer:
    """Get a global visualizer instance (singleton pattern)."""
    global _global_visualizer
    if _global_visualizer is None:
        _global_visualizer = MathematicalVisualizer()
    return _global_visualizer


# Convenience functions for direct usage
def plot_function_2d(expression: str, variable: str = 'x', 
                    x_range: Tuple[float, float] = None,
                    title: str = None, use_plotly: bool = True) -> Optional[Union[go.Figure, matplotlib.figure.Figure]]:
    """Generate a 2D plot of a mathematical function."""
    visualizer = get_visualizer()
    return visualizer.plot_function_2d(expression, variable, x_range, title, use_plotly)


def plot_matrix_heatmap(matrix: List[List[float]], title: str = None,
                       labels: Dict[str, List[str]] = None,
                       use_plotly: bool = True) -> Optional[Union[go.Figure, matplotlib.figure.Figure]]:
    """Generate a heatmap visualization of a matrix."""
    visualizer = get_visualizer()
    return visualizer.plot_matrix_heatmap(matrix, title, labels, use_plotly)


def plot_matrix_decomposition(matrices: Dict[str, List[List[float]]], 
                             decomposition_type: str,
                             use_plotly: bool = True) -> Optional[Union[go.Figure, matplotlib.figure.Figure]]:
    """Visualize matrix decomposition results."""
    visualizer = get_visualizer()
    return visualizer.plot_matrix_decomposition(matrices, decomposition_type, use_plotly)


def plot_eigenvalues(eigenvalues: List[complex], eigenvectors: List[List[complex]] = None,
                    use_plotly: bool = True) -> Optional[Union[go.Figure, matplotlib.figure.Figure]]:
    """Visualize eigenvalues in the complex plane."""
    visualizer = get_visualizer()
    return visualizer.plot_eigenvalues(eigenvalues, eigenvectors, use_plotly)