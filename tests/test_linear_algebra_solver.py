"""
Comprehensive Test Suite for Linear Algebra Solver

Tests all functionality of the high-performance linear algebra solver including
matrix operations, decompositions, error handling, and mathematical verification.

Author: MathBoardAI Agent Team
Task ID: SOLVER-001
Coverage Target: 90%+
"""

import pytest
import sys
import os
import numpy as np
from typing import List, Dict, Any

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from solvers.linear_algebra_solver import (
        LinearAlgebraSolver, get_solver,
        compute_determinant, compute_inverse, lu_decomposition,
        qr_decomposition, eigen_decomposition, svd
    )
    SOLVER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Linear algebra solver not available: {e}")
    SOLVER_AVAILABLE = False


@pytest.mark.skipif(not SOLVER_AVAILABLE, reason="Linear algebra solver not available")
class TestLinearAlgebraSolver:
    """Test suite for the LinearAlgebraSolver class."""
    
    @pytest.fixture
    def solver(self):
        """Create a solver instance for testing."""
        return LinearAlgebraSolver()
    
    @pytest.fixture
    def test_matrices(self):
        """Provide standard test matrices for various operations."""
        return {
            # 2x2 invertible matrix
            "simple_2x2": [[1, 2], [3, 4]],
            
            # 2x2 singular matrix (determinant = 0)
            "singular_2x2": [[1, 2], [2, 4]],
            
            # 3x3 invertible matrix
            "simple_3x3": [[1, 2, 3], [0, 4, 5], [0, 0, 6]],
            
            # 3x3 symmetric positive definite matrix
            "symmetric_3x3": [[4, 1, 2], [1, 3, 1], [2, 1, 5]],
            
            # 2x3 rectangular matrix
            "rectangular_2x3": [[1, 2, 3], [4, 5, 6]],
            
            # Identity matrices
            "identity_2x2": [[1, 0], [0, 1]],
            "identity_3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            
            # Complex eigenvalue matrix
            "rotation_2x2": [[0, -1], [1, 0]],  # 90-degree rotation
            
            # Near-singular matrix (high condition number)
            "ill_conditioned": [[1, 1], [1, 1.0001]]
        }
    
    # Test determinant computation
    def test_determinant_simple_2x2(self, solver, test_matrices):
        """Test determinant computation for simple 2x2 matrix."""
        result = solver.compute_determinant(test_matrices["simple_2x2"])
        
        assert result["success"] is True
        assert abs(result["determinant"] - (-2)) < 1e-10  # det([[1,2],[3,4]]) = -2
        assert result["matrix_shape"] == (2, 2)
        assert "computation_time_ms" in result
    
    def test_determinant_singular_matrix(self, solver, test_matrices):
        """Test determinant computation for singular matrix."""
        result = solver.compute_determinant(test_matrices["singular_2x2"])
        
        assert result["success"] is True
        assert abs(result["determinant"]) < 1e-10  # Should be 0 (or very close)
        assert result["is_singular"] is True
    
    def test_determinant_identity_matrix(self, solver, test_matrices):
        """Test determinant of identity matrix."""
        result = solver.compute_determinant(test_matrices["identity_3x3"])
        
        assert result["success"] is True
        assert abs(result["determinant"] - 1.0) < 1e-10
        assert result["is_singular"] is False
    
    def test_determinant_non_square_matrix(self, solver, test_matrices):
        """Test determinant with non-square matrix (should fail)."""
        result = solver.compute_determinant(test_matrices["rectangular_2x3"])
        
        assert result["success"] is False
        assert "square" in result["error"].lower()
    
    # Test matrix inverse computation
    def test_inverse_simple_2x2(self, solver, test_matrices):
        """Test matrix inverse computation and verification."""
        matrix = test_matrices["simple_2x2"]
        result = solver.compute_inverse(matrix)
        
        assert result["success"] is True
        assert "inverse" in result
        
        # Mathematical verification: A * A^(-1) = I
        A = np.array(matrix)
        A_inv = np.array(result["inverse"])
        identity_check = A @ A_inv
        
        # Should be very close to identity matrix
        expected_identity = np.eye(A.shape[0])
        assert np.allclose(identity_check, expected_identity, atol=1e-10)
    
    def test_inverse_singular_matrix(self, solver, test_matrices):
        """Test inverse computation for singular matrix (should fail)."""
        result = solver.compute_inverse(test_matrices["singular_2x2"])
        
        assert result["success"] is False
        assert "singular" in result["error"].lower()
        assert "determinant" in result  # Should provide determinant info
    
    def test_inverse_identity_matrix(self, solver, test_matrices):
        """Test inverse of identity matrix."""
        result = solver.compute_inverse(test_matrices["identity_2x2"])
        
        assert result["success"] is True
        
        # Inverse of identity should be identity
        inverse_matrix = np.array(result["inverse"])
        expected_identity = np.eye(2)
        assert np.allclose(inverse_matrix, expected_identity, atol=1e-10)
    
    # Test LU decomposition
    def test_lu_decomposition_simple_matrix(self, solver, test_matrices):
        """Test LU decomposition with mathematical verification."""
        matrix = test_matrices["simple_3x3"]
        result = solver.lu_decomposition(matrix)
        
        assert result["success"] is True
        assert "P" in result and "L" in result and "U" in result
        
        # Mathematical verification: P @ L @ U = A
        A = np.array(matrix)
        P = np.array(result["P"])
        L = np.array(result["L"])
        U = np.array(result["U"])
        
        reconstructed = P @ L @ U
        assert np.allclose(reconstructed, A, atol=1e-10)
        
        # Verify L is lower triangular with unit diagonal
        assert np.allclose(np.triu(L, k=1), 0, atol=1e-10)  # Upper part is zero
        assert np.allclose(np.diag(L), 1, atol=1e-10)  # Diagonal is ones
        
        # Verify U is upper triangular
        assert np.allclose(np.tril(U, k=-1), 0, atol=1e-10)  # Lower part is zero
    
    def test_lu_decomposition_symmetric_matrix(self, solver, test_matrices):
        """Test LU decomposition for symmetric matrix."""
        result = solver.lu_decomposition(test_matrices["symmetric_3x3"])
        
        assert result["success"] is True
        assert result["reconstruction_error"] < 1e-10
    
    # Test QR decomposition
    def test_qr_decomposition_square_matrix(self, solver, test_matrices):
        """Test QR decomposition with mathematical verification."""
        matrix = test_matrices["simple_3x3"]
        result = solver.qr_decomposition(matrix)
        
        assert result["success"] is True
        assert "Q" in result and "R" in result
        
        # Mathematical verification: Q @ R = A
        A = np.array(matrix)
        Q = np.array(result["Q"])
        R = np.array(result["R"])
        
        reconstructed = Q @ R
        assert np.allclose(reconstructed, A, atol=1e-10)
        
        # Verify Q is orthonormal: Q^T @ Q = I
        QTQ = Q.T @ Q
        expected_identity = np.eye(Q.shape[1])
        assert np.allclose(QTQ, expected_identity, atol=1e-10)
        
        # Verify R is upper triangular
        assert np.allclose(np.tril(R, k=-1), 0, atol=1e-10)
    
    def test_qr_decomposition_rectangular_matrix(self, solver, test_matrices):
        """Test QR decomposition for rectangular matrix."""
        result = solver.qr_decomposition(test_matrices["rectangular_2x3"])
        
        assert result["success"] is True
        assert result["Q_shape"] == (2, 2)  # Q should be 2x2
        assert result["R_shape"] == (2, 3)  # R should be 2x3
        assert result["orthogonality_error"] < 1e-10
    
    # Test eigenvalue decomposition
    def test_eigen_decomposition_symmetric_matrix(self, solver, test_matrices):
        """Test eigenvalue decomposition for symmetric matrix."""
        result = solver.eigen_decomposition(test_matrices["symmetric_3x3"])
        
        assert result["success"] is True
        assert "eigenvalues_real" in result
        assert "eigenvectors_real" in result
        
        # For symmetric matrices, eigenvalues should be real
        assert not result["eigenvalues_complex"]
        assert not result["eigenvectors_complex"]
        
        # Verify eigenvalue/eigenvector relationship: A*v = λ*v
        A = np.array(test_matrices["symmetric_3x3"])
        eigenvals = np.array(result["eigenvalues_real"])
        eigenvecs = np.array(result["eigenvectors_real"])
        
        for i in range(len(eigenvals)):
            λ = eigenvals[i]
            v = eigenvecs[:, i]
            
            # Check A*v = λ*v
            Av = A @ v
            λv = λ * v
            assert np.allclose(Av, λv, atol=1e-10)
    
    def test_eigen_decomposition_rotation_matrix(self, solver, test_matrices):
        """Test eigenvalue decomposition for rotation matrix (complex eigenvalues)."""
        result = solver.eigen_decomposition(test_matrices["rotation_2x2"])
        
        assert result["success"] is True
        assert result["eigenvalues_complex"] is True  # Should have complex eigenvalues
        
        # Rotation matrix should have eigenvalues with magnitude 1
        eigenvals_real = np.array(result["eigenvalues_real"])
        eigenvals_imag = np.array(result["eigenvalues_imag"])
        eigenvals_complex = eigenvals_real + 1j * eigenvals_imag
        
        magnitudes = np.abs(eigenvals_complex)
        assert np.allclose(magnitudes, 1.0, atol=1e-10)
    
    def test_eigen_decomposition_identity_matrix(self, solver, test_matrices):
        """Test eigenvalue decomposition for identity matrix."""
        result = solver.eigen_decomposition(test_matrices["identity_2x2"])
        
        assert result["success"] is True
        
        # Identity matrix should have all eigenvalues = 1
        eigenvals = np.array(result["eigenvalues_real"])
        assert np.allclose(eigenvals, 1.0, atol=1e-10)
        assert result["spectral_radius"] == 1.0
    
    # Test SVD
    def test_svd_square_matrix(self, solver, test_matrices):
        """Test SVD with mathematical verification."""
        matrix = test_matrices["simple_2x2"]
        result = solver.svd(matrix)
        
        assert result["success"] is True
        assert "U" in result and "s" in result and "Vh" in result
        
        # Mathematical verification: U @ diag(s) @ Vh = A
        A = np.array(matrix)
        U = np.array(result["U"])
        s = np.array(result["s"])
        Vh = np.array(result["Vh"])
        
        # Reconstruct matrix
        S_matrix = np.diag(s)
        reconstructed = U @ S_matrix @ Vh
        assert np.allclose(reconstructed, A, atol=1e-10)
        
        # Verify U and Vh are orthonormal
        assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-10)
        assert np.allclose(Vh @ Vh.T, np.eye(Vh.shape[0]), atol=1e-10)
        
        # Verify singular values are non-negative and sorted
        assert np.all(s >= 0)
        assert np.all(s[:-1] >= s[1:])  # Descending order
    
    def test_svd_rectangular_matrix(self, solver, test_matrices):
        """Test SVD for rectangular matrix."""
        result = solver.svd(test_matrices["rectangular_2x3"])
        
        assert result["success"] is True
        assert result["U_shape"] == (2, 2)
        assert result["s_shape"] == (2,)  # min(2,3) = 2
        assert result["Vh_shape"] == (3, 3)
        assert result["reconstruction_error"] < 1e-10
    
    def test_svd_rank_detection(self, solver, test_matrices):
        """Test SVD rank detection for singular matrix."""
        result = solver.svd(test_matrices["singular_2x2"])
        
        assert result["success"] is True
        assert result["rank"] == 1  # Singular matrix should have rank 1
        assert result["condition_number"] == np.inf  # Should be infinite
    
    # Test acceptance criteria examples
    def test_acceptance_criteria_svd(self, solver):
        """Test the specific acceptance criteria example: svd([[1, 2], [3, 4]])."""
        matrix = [[1, 2], [3, 4]]
        result = solver.svd(matrix)
        
        assert result["success"] is True
        assert "U" in result
        assert "s" in result  
        assert "Vh" in result
        
        # Verify the decomposition is mathematically correct
        A = np.array(matrix)
        U = np.array(result["U"])
        s = np.array(result["s"])
        Vh = np.array(result["Vh"])
        
        S_matrix = np.diag(s)
        reconstructed = U @ S_matrix @ Vh
        assert np.allclose(reconstructed, A, atol=1e-10)
    
    def test_acceptance_criteria_singular_inverse(self, solver):
        """Test the specific acceptance criteria: compute_inverse([[1, 2], [1, 2]])."""
        singular_matrix = [[1, 2], [1, 2]]
        result = solver.compute_inverse(singular_matrix)
        
        assert result["success"] is False
        assert "singular" in result["error"].lower()
        assert "cannot be inverted" in result["error"].lower()
    
    # Test error handling
    def test_empty_matrix_error(self, solver):
        """Test handling of empty matrix."""
        result = solver.compute_determinant([])
        assert result["success"] is False
        assert "empty" in result["error"].lower()
    
    def test_invalid_matrix_format(self, solver):
        """Test handling of invalid matrix format."""
        # Non-rectangular matrix
        invalid_matrix = [[1, 2], [3, 4, 5]]
        result = solver.compute_determinant(invalid_matrix)
        assert result["success"] is False
    
    def test_nan_values_error(self, solver):
        """Test handling of matrices with NaN values."""
        nan_matrix = [[1, float('nan')], [3, 4]]
        result = solver.compute_determinant(nan_matrix)
        assert result["success"] is False
        assert "invalid values" in result["error"].lower()
    
    def test_infinity_values_error(self, solver):
        """Test handling of matrices with infinity values."""
        inf_matrix = [[1, 2], [float('inf'), 4]]
        result = solver.compute_determinant(inf_matrix)
        assert result["success"] is False
        assert "invalid values" in result["error"].lower()
    
    # Test solver statistics
    def test_solver_statistics(self, solver, test_matrices):
        """Test solver performance statistics tracking."""
        initial_stats = solver.get_statistics()
        initial_count = initial_stats["operations_performed"]
        
        # Perform some operations
        solver.compute_determinant(test_matrices["simple_2x2"])
        solver.compute_inverse(test_matrices["simple_2x2"])
        
        final_stats = solver.get_statistics()
        
        assert final_stats["operations_performed"] == initial_count + 2
        assert final_stats["total_computation_time_ms"] > initial_stats["total_computation_time_ms"]
        assert final_stats["average_computation_time_ms"] >= 0
    
    # Test singleton pattern
    def test_global_solver_singleton(self):
        """Test that the global solver singleton works correctly."""
        solver1 = get_solver()
        solver2 = get_solver()
        
        assert solver1 is solver2  # Should be the same instance


@pytest.mark.skipif(not SOLVER_AVAILABLE, reason="Linear algebra solver not available")
class TestConvenienceFunctions:
    """Test the convenience functions for direct usage."""
    
    def test_convenience_function_determinant(self):
        """Test compute_determinant convenience function."""
        matrix = [[1, 2], [3, 4]]
        result = compute_determinant(matrix)
        
        assert result["success"] is True
        assert abs(result["determinant"] - (-2)) < 1e-10
    
    def test_convenience_function_inverse(self):
        """Test compute_inverse convenience function."""
        matrix = [[1, 2], [3, 4]]
        result = compute_inverse(matrix)
        
        assert result["success"] is True
        assert "inverse" in result
    
    def test_convenience_function_lu(self):
        """Test lu_decomposition convenience function."""
        matrix = [[1, 2], [3, 4]]
        result = lu_decomposition(matrix)
        
        assert result["success"] is True
        assert "P" in result and "L" in result and "U" in result
    
    def test_convenience_function_qr(self):
        """Test qr_decomposition convenience function."""
        matrix = [[1, 2], [3, 4]]
        result = qr_decomposition(matrix)
        
        assert result["success"] is True
        assert "Q" in result and "R" in result
    
    def test_convenience_function_eigen(self):
        """Test eigen_decomposition convenience function."""
        matrix = [[1, 2], [2, 1]]  # Symmetric matrix
        result = eigen_decomposition(matrix)
        
        assert result["success"] is True
        assert "eigenvalues_real" in result
    
    def test_convenience_function_svd(self):
        """Test svd convenience function."""
        matrix = [[1, 2], [3, 4]]
        result = svd(matrix)
        
        assert result["success"] is True
        assert "U" in result and "s" in result and "Vh" in result


@pytest.mark.skipif(not SOLVER_AVAILABLE, reason="Linear algebra solver not available")
class TestMathematicalVerification:
    """Comprehensive mathematical verification tests."""
    
    def test_lu_decomposition_verification(self):
        """Verify LU decomposition satisfies P @ L @ U = A."""
        # Test with multiple matrices
        test_matrices = [
            [[4, 3], [2, 1]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 10]],
            [[2, -1, 0], [1, 0, 2], [3, 1, 1]]
        ]
        
        for matrix in test_matrices:
            result = lu_decomposition(matrix)
            assert result["success"] is True
            
            A = np.array(matrix)
            P = np.array(result["P"])
            L = np.array(result["L"])
            U = np.array(result["U"])
            
            # Verify P @ L @ U = A
            reconstructed = P @ L @ U
            assert np.allclose(reconstructed, A, atol=1e-12), f"LU verification failed for matrix {matrix}"
    
    def test_qr_decomposition_verification(self):
        """Verify QR decomposition satisfies Q @ R = A and Q^T @ Q = I."""
        test_matrices = [
            [[1, 2], [3, 4]],
            [[1, 0, 1], [2, 1, 0], [0, 1, 2]],
            [[1, 2, 3], [4, 5, 6]]  # Rectangular
        ]
        
        for matrix in test_matrices:
            result = qr_decomposition(matrix)
            assert result["success"] is True
            
            A = np.array(matrix)
            Q = np.array(result["Q"])
            R = np.array(result["R"])
            
            # Verify Q @ R = A
            reconstructed = Q @ R
            assert np.allclose(reconstructed, A, atol=1e-12), f"QR reconstruction failed for matrix {matrix}"
            
            # Verify Q is orthonormal
            QTQ = Q.T @ Q
            expected_identity = np.eye(Q.shape[1])
            assert np.allclose(QTQ, expected_identity, atol=1e-12), f"Q orthogonality failed for matrix {matrix}"
    
    def test_svd_verification(self):
        """Verify SVD satisfies U @ diag(s) @ Vh = A."""
        test_matrices = [
            [[1, 2], [3, 4]],
            [[1, 0, 0], [0, 2, 0], [0, 0, 3]],  # Diagonal
            [[1, 2, 3], [4, 5, 6]]  # Rectangular
        ]
        
        for matrix in test_matrices:
            result = svd(matrix)
            assert result["success"] is True
            
            A = np.array(matrix)
            U = np.array(result["U"])
            s = np.array(result["s"])
            Vh = np.array(result["Vh"])
            
            # Reconstruct the matrix
            m, n = A.shape
            S = np.zeros((m, n))
            S[:min(m, n), :min(m, n)] = np.diag(s)
            reconstructed = U @ S @ Vh
            
            assert np.allclose(reconstructed, A, atol=1e-12), f"SVD reconstruction failed for matrix {matrix}"
    
    def test_eigenvalue_verification(self):
        """Verify eigenvalue decomposition satisfies A @ v = λ @ v."""
        # Use symmetric matrices to ensure real eigenvalues
        test_matrices = [
            [[3, 1], [1, 3]],  # Simple symmetric
            [[4, -2], [-2, 1]],  # Another symmetric
            [[1, 0, 0], [0, 2, 0], [0, 0, 3]]  # Diagonal
        ]
        
        for matrix in test_matrices:
            result = eigen_decomposition(matrix)
            assert result["success"] is True
            
            A = np.array(matrix)
            eigenvals = np.array(result["eigenvalues_real"])
            eigenvecs = np.array(result["eigenvectors_real"])
            
            # Verify A @ v = λ @ v for each eigenvalue/eigenvector pair
            for i in range(len(eigenvals)):
                λ = eigenvals[i]
                v = eigenvecs[:, i]
                
                Av = A @ v
                λv = λ * v
                
                # Allow for sign ambiguity in eigenvectors
                if not np.allclose(Av, λv, atol=1e-10):
                    # Try with negated eigenvector
                    λv_neg = λ * (-v)
                    assert np.allclose(Av, λv_neg, atol=1e-10), f"Eigenvalue verification failed for matrix {matrix}, eigenvalue {λ}"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])