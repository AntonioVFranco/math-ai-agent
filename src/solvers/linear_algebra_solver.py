"""
High-Performance Linear Algebra Solver

A dedicated numerical linear algebra solver using NumPy and SciPy libraries.
Provides optimized matrix operations and advanced decompositions for AI/ML
applications including dimensionality reduction, neural network analysis,
and optimization problems.

Author: MathBoardAI Agent Team
Task ID: SOLVER-001
Field: Numerical Linear Algebra & Matrix Theory
"""

import logging
import time
from typing import List, Dict, Any, Tuple, Optional, Union
import warnings

import numpy as np
import scipy.linalg
from numpy.linalg import LinAlgError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress numpy warnings for cleaner output
warnings.filterwarnings('ignore', category=np.ComplexWarning)


class LinearAlgebraSolver:
    """
    High-performance numerical linear algebra solver.
    
    This solver provides optimized implementations of core matrix operations
    and advanced decompositions using NumPy and SciPy's highly optimized
    computational backends.
    
    Key Features:
    - Numerical stability through SciPy implementations
    - Comprehensive error handling for edge cases
    - Performance optimization for AI/ML applications
    - Mathematical verification capabilities
    """
    
    def __init__(self):
        """Initialize the linear algebra solver."""
        self.operations_count = 0
        self.total_computation_time = 0.0
        logger.info("High-Performance Linear Algebra Solver initialized")
    
    def _validate_matrix(self, matrix: List[List[float]], 
                        operation_name: str,
                        require_square: bool = False,
                        min_size: Optional[int] = None) -> Tuple[bool, Optional[str], Optional[np.ndarray]]:
        """
        Validate input matrix and convert to NumPy array.
        
        Args:
            matrix: Input matrix as list of lists
            operation_name: Name of the operation for error messages
            require_square: Whether the matrix must be square
            min_size: Minimum required size for the matrix
            
        Returns:
            Tuple of (is_valid, error_message, numpy_array)
        """
        try:
            # Convert to numpy array
            np_matrix = np.array(matrix, dtype=float)
            
            # Check if it's actually a matrix (2D)
            if np_matrix.ndim != 2:
                return False, f"Input must be a 2D matrix for {operation_name}", None
            
            # Check for empty matrix
            if np_matrix.size == 0:
                return False, f"Matrix cannot be empty for {operation_name}", None
            
            # Check minimum size requirement
            if min_size and (np_matrix.shape[0] < min_size or np_matrix.shape[1] < min_size):
                return False, f"Matrix must be at least {min_size}x{min_size} for {operation_name}", None
            
            # Check square requirement
            if require_square and np_matrix.shape[0] != np_matrix.shape[1]:
                return False, f"Matrix must be square for {operation_name}. Got shape {np_matrix.shape}", None
            
            # Check for invalid values
            if not np.isfinite(np_matrix).all():
                return False, f"Matrix contains invalid values (NaN or infinity) for {operation_name}", None
            
            return True, None, np_matrix
            
        except (ValueError, TypeError) as e:
            return False, f"Invalid matrix format for {operation_name}: {str(e)}", None
    
    def compute_determinant(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        Calculate the determinant of a square matrix.
        
        Args:
            matrix: Square matrix as list of lists
            
        Returns:
            Dictionary containing determinant value or error information
        """
        start_time = time.time()
        
        try:
            # Validate input
            is_valid, error_msg, np_matrix = self._validate_matrix(
                matrix, "determinant", require_square=True, min_size=1
            )
            
            if not is_valid:
                return {
                    "success": False,
                    "error": error_msg,
                    "operation": "determinant"
                }
            
            # Compute determinant using SciPy for numerical stability
            det_value = scipy.linalg.det(np_matrix)
            
            # Handle near-zero determinants
            if abs(det_value) < 1e-12:
                det_value = 0.0
            
            end_time = time.time()
            computation_time = (end_time - start_time) * 1000
            
            self.operations_count += 1
            self.total_computation_time += computation_time
            
            return {
                "success": True,
                "determinant": float(det_value),
                "matrix_shape": np_matrix.shape,
                "operation": "determinant",
                "computation_time_ms": computation_time,
                "is_singular": abs(det_value) < 1e-12
            }
            
        except LinAlgError as e:
            return {
                "success": False,
                "error": f"Linear algebra error computing determinant: {str(e)}",
                "operation": "determinant"
            }
        except Exception as e:
            logger.error(f"Unexpected error in determinant computation: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error computing determinant: {str(e)}",
                "operation": "determinant"
            }
    
    def compute_inverse(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        Calculate the inverse of a square matrix.
        
        Args:
            matrix: Square matrix as list of lists
            
        Returns:
            Dictionary containing inverse matrix or error information
        """
        start_time = time.time()
        
        try:
            # Validate input
            is_valid, error_msg, np_matrix = self._validate_matrix(
                matrix, "matrix inverse", require_square=True, min_size=1
            )
            
            if not is_valid:
                return {
                    "success": False,
                    "error": error_msg,
                    "operation": "inverse"
                }
            
            # Check if matrix is singular before attempting inversion
            det = scipy.linalg.det(np_matrix)
            if abs(det) < 1e-12:
                return {
                    "success": False,
                    "error": "Matrix is singular and cannot be inverted (determinant ≈ 0)",
                    "operation": "inverse",
                    "determinant": float(det)
                }
            
            # Compute inverse using SciPy for numerical stability
            inverse_matrix = scipy.linalg.inv(np_matrix)
            
            end_time = time.time()
            computation_time = (end_time - start_time) * 1000
            
            self.operations_count += 1
            self.total_computation_time += computation_time
            
            return {
                "success": True,
                "inverse": inverse_matrix.tolist(),
                "matrix_shape": np_matrix.shape,
                "operation": "inverse",
                "computation_time_ms": computation_time,
                "condition_number": float(np.linalg.cond(np_matrix))
            }
            
        except LinAlgError as e:
            return {
                "success": False,
                "error": f"Matrix is singular and cannot be inverted: {str(e)}",
                "operation": "inverse"
            }
        except Exception as e:
            logger.error(f"Unexpected error in inverse computation: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error computing inverse: {str(e)}",
                "operation": "inverse"
            }
    
    def lu_decomposition(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        Perform LU decomposition with partial pivoting: PA = LU.
        
        Args:
            matrix: Input matrix as list of lists
            
        Returns:
            Dictionary containing P, L, U matrices or error information
        """
        start_time = time.time()
        
        try:
            # Validate input
            is_valid, error_msg, np_matrix = self._validate_matrix(
                matrix, "LU decomposition", require_square=True, min_size=1
            )
            
            if not is_valid:
                return {
                    "success": False,
                    "error": error_msg,
                    "operation": "lu_decomposition"
                }
            
            # Perform LU decomposition with partial pivoting
            P, L, U = scipy.linalg.lu(np_matrix)
            
            end_time = time.time()
            computation_time = (end_time - start_time) * 1000
            
            self.operations_count += 1
            self.total_computation_time += computation_time
            
            # Verify decomposition accuracy
            reconstruction_error = np.linalg.norm(P @ L @ U - np_matrix)
            
            return {
                "success": True,
                "P": P.tolist(),  # Permutation matrix
                "L": L.tolist(),  # Lower triangular matrix
                "U": U.tolist(),  # Upper triangular matrix
                "matrix_shape": np_matrix.shape,
                "operation": "lu_decomposition",
                "computation_time_ms": computation_time,
                "reconstruction_error": float(reconstruction_error)
            }
            
        except LinAlgError as e:
            return {
                "success": False,
                "error": f"LU decomposition failed: {str(e)}",
                "operation": "lu_decomposition"
            }
        except Exception as e:
            logger.error(f"Unexpected error in LU decomposition: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error in LU decomposition: {str(e)}",
                "operation": "lu_decomposition"
            }
    
    def qr_decomposition(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        Perform QR decomposition: A = QR.
        
        Args:
            matrix: Input matrix as list of lists
            
        Returns:
            Dictionary containing Q and R matrices or error information
        """
        start_time = time.time()
        
        try:
            # Validate input
            is_valid, error_msg, np_matrix = self._validate_matrix(
                matrix, "QR decomposition", require_square=False, min_size=1
            )
            
            if not is_valid:
                return {
                    "success": False,
                    "error": error_msg,
                    "operation": "qr_decomposition"
                }
            
            # Perform QR decomposition
            Q, R = scipy.linalg.qr(np_matrix)
            
            end_time = time.time()
            computation_time = (end_time - start_time) * 1000
            
            self.operations_count += 1
            self.total_computation_time += computation_time
            
            # Verify decomposition accuracy
            reconstruction_error = np.linalg.norm(Q @ R - np_matrix)
            
            # Check orthogonality of Q
            orthogonality_error = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))
            
            return {
                "success": True,
                "Q": Q.tolist(),  # Orthonormal matrix
                "R": R.tolist(),  # Upper triangular matrix
                "matrix_shape": np_matrix.shape,
                "Q_shape": Q.shape,
                "R_shape": R.shape,
                "operation": "qr_decomposition",
                "computation_time_ms": computation_time,
                "reconstruction_error": float(reconstruction_error),
                "orthogonality_error": float(orthogonality_error)
            }
            
        except LinAlgError as e:
            return {
                "success": False,
                "error": f"QR decomposition failed: {str(e)}",
                "operation": "qr_decomposition"
            }
        except Exception as e:
            logger.error(f"Unexpected error in QR decomposition: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error in QR decomposition: {str(e)}",
                "operation": "qr_decomposition"
            }
    
    def eigen_decomposition(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        Compute eigenvalues and eigenvectors of a square matrix.
        
        Args:
            matrix: Square matrix as list of lists
            
        Returns:
            Dictionary containing eigenvalues and eigenvectors or error information
        """
        start_time = time.time()
        
        try:
            # Validate input
            is_valid, error_msg, np_matrix = self._validate_matrix(
                matrix, "eigenvalue decomposition", require_square=True, min_size=1
            )
            
            if not is_valid:
                return {
                    "success": False,
                    "error": error_msg,
                    "operation": "eigen_decomposition"
                }
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = scipy.linalg.eig(np_matrix)
            
            end_time = time.time()
            computation_time = (end_time - start_time) * 1000
            
            self.operations_count += 1
            self.total_computation_time += computation_time
            
            # Sort eigenvalues and eigenvectors by eigenvalue magnitude (descending)
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues_sorted = eigenvalues[idx]
            eigenvectors_sorted = eigenvectors[:, idx]
            
            # Separate real and imaginary parts if complex
            if np.iscomplexobj(eigenvalues_sorted):
                eigenvalues_real = eigenvalues_sorted.real.tolist()
                eigenvalues_imag = eigenvalues_sorted.imag.tolist()
                eigenvalues_complex = True
            else:
                eigenvalues_real = eigenvalues_sorted.real.tolist()
                eigenvalues_imag = None
                eigenvalues_complex = False
            
            if np.iscomplexobj(eigenvectors_sorted):
                eigenvectors_real = eigenvectors_sorted.real.tolist()
                eigenvectors_imag = eigenvectors_sorted.imag.tolist()
                eigenvectors_complex = True
            else:
                eigenvectors_real = eigenvectors_sorted.real.tolist()
                eigenvectors_imag = None
                eigenvectors_complex = False
            
            # Compute spectral radius (largest eigenvalue magnitude)
            spectral_radius = float(np.max(np.abs(eigenvalues)))
            
            return {
                "success": True,
                "eigenvalues_real": eigenvalues_real,
                "eigenvalues_imag": eigenvalues_imag,
                "eigenvectors_real": eigenvectors_real,
                "eigenvectors_imag": eigenvectors_imag,
                "eigenvalues_complex": eigenvalues_complex,
                "eigenvectors_complex": eigenvectors_complex,
                "spectral_radius": spectral_radius,
                "matrix_shape": np_matrix.shape,
                "operation": "eigen_decomposition",
                "computation_time_ms": computation_time
            }
            
        except LinAlgError as e:
            return {
                "success": False,
                "error": f"Eigenvalue decomposition failed: {str(e)}",
                "operation": "eigen_decomposition"
            }
        except Exception as e:
            logger.error(f"Unexpected error in eigenvalue decomposition: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error in eigenvalue decomposition: {str(e)}",
                "operation": "eigen_decomposition"
            }
    
    def svd(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """
        Perform Singular Value Decomposition: A = U * S * Vh.
        
        Args:
            matrix: Input matrix as list of lists
            
        Returns:
            Dictionary containing U, S, and Vh matrices or error information
        """
        start_time = time.time()
        
        try:
            # Validate input
            is_valid, error_msg, np_matrix = self._validate_matrix(
                matrix, "SVD", require_square=False, min_size=1
            )
            
            if not is_valid:
                return {
                    "success": False,
                    "error": error_msg,
                    "operation": "svd"
                }
            
            # Perform SVD
            U, s, Vh = scipy.linalg.svd(np_matrix)
            
            end_time = time.time()
            computation_time = (end_time - start_time) * 1000
            
            self.operations_count += 1
            self.total_computation_time += computation_time
            
            # Verify decomposition accuracy
            # Reconstruct the original matrix
            S_full = np.zeros(np_matrix.shape)
            S_full[:min(np_matrix.shape), :min(np_matrix.shape)] = np.diag(s)
            reconstruction_error = np.linalg.norm(U @ S_full @ Vh - np_matrix)
            
            # Compute matrix rank and condition number
            rank = np.sum(s > 1e-12 * s[0])  # Numerical rank
            condition_number = s[0] / s[-1] if s[-1] > 1e-12 else np.inf
            
            return {
                "success": True,
                "U": U.tolist(),           # Left singular vectors
                "s": s.tolist(),           # Singular values (1D array)
                "Vh": Vh.tolist(),         # Right singular vectors (transposed)
                "matrix_shape": np_matrix.shape,
                "U_shape": U.shape,
                "s_shape": s.shape,
                "Vh_shape": Vh.shape,
                "operation": "svd",
                "computation_time_ms": computation_time,
                "reconstruction_error": float(reconstruction_error),
                "rank": int(rank),
                "condition_number": float(condition_number),
                "largest_singular_value": float(s[0]),
                "smallest_singular_value": float(s[-1])
            }
            
        except LinAlgError as e:
            return {
                "success": False,
                "error": f"SVD failed: {str(e)}",
                "operation": "svd"
            }
        except Exception as e:
            logger.error(f"Unexpected error in SVD: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error in SVD: {str(e)}",
                "operation": "svd"
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get solver performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        avg_time = (self.total_computation_time / self.operations_count 
                   if self.operations_count > 0 else 0)
        
        return {
            "operations_performed": self.operations_count,
            "total_computation_time_ms": self.total_computation_time,
            "average_computation_time_ms": avg_time
        }


# Global solver instance for efficiency
_global_solver = None


def get_solver() -> LinearAlgebraSolver:
    """Get a global solver instance (singleton pattern)."""
    global _global_solver
    if _global_solver is None:
        _global_solver = LinearAlgebraSolver()
    return _global_solver


# Convenience functions for direct usage
def compute_determinant(matrix: List[List[float]]) -> Dict[str, Any]:
    """Compute determinant of a square matrix."""
    solver = get_solver()
    return solver.compute_determinant(matrix)


def compute_inverse(matrix: List[List[float]]) -> Dict[str, Any]:
    """Compute inverse of a square matrix."""
    solver = get_solver()
    return solver.compute_inverse(matrix)


def lu_decomposition(matrix: List[List[float]]) -> Dict[str, Any]:
    """Perform LU decomposition of a matrix."""
    solver = get_solver()
    return solver.lu_decomposition(matrix)


def qr_decomposition(matrix: List[List[float]]) -> Dict[str, Any]:
    """Perform QR decomposition of a matrix."""
    solver = get_solver()
    return solver.qr_decomposition(matrix)


def eigen_decomposition(matrix: List[List[float]]) -> Dict[str, Any]:
    """Compute eigenvalues and eigenvectors of a square matrix."""
    solver = get_solver()
    return solver.eigen_decomposition(matrix)


def svd(matrix: List[List[float]]) -> Dict[str, Any]:
    """Perform Singular Value Decomposition of a matrix."""
    solver = get_solver()
    return solver.svd(matrix)