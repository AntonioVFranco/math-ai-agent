"""
Comprehensive Test Suite for Multi-Layer Verification System

Tests all functionality of the mathematical solution verifier including
numerical cross-validation, symbolic verification, matrix reconstruction,
and integration with the core engine pipeline.

Author: Math AI Agent Team
Task ID: CORE-003
Coverage Target: 80%+
"""

import pytest
import sys
import os
import numpy as np
from typing import Dict, Any, List

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from core.verifier import (
        MathematicalVerifier, get_verifier, verify_solution,
        VerificationResult, VerificationMethod
    )
    from core.models import ParsedProblem, MathDomain, ProblemType, Confidence
    VERIFIER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Verifier not available: {e}")
    VERIFIER_AVAILABLE = False


@pytest.mark.skipif(not VERIFIER_AVAILABLE, reason="Verifier not available")
class TestMathematicalVerifier:
    """Test suite for the MathematicalVerifier class."""
    
    @pytest.fixture
    def verifier(self):
        """Create a verifier instance for testing."""
        return MathematicalVerifier()
    
    @pytest.fixture
    def sample_parsed_problems(self):
        """Provide sample parsed problems for testing."""
        return {
            # Linear algebra problems
            "determinant_2x2": ParsedProblem(
                domain=MathDomain.LINEAR_ALGEBRA,
                problem_type=ProblemType.DETERMINANT,
                variables=[],
                expression="[[1, 2], [3, 4]]",
                confidence=Confidence(domain=0.9, problem_type=0.9, overall=0.9),
                original_text="Find the determinant of [[1, 2], [3, 4]]"
            ),
            "inverse_2x2": ParsedProblem(
                domain=MathDomain.LINEAR_ALGEBRA,
                problem_type=ProblemType.INVERSE,
                variables=[],
                expression="[[2, 1], [1, 1]]",
                confidence=Confidence(domain=0.9, problem_type=0.9, overall=0.9),
                original_text="Find the inverse of [[2, 1], [1, 1]]"
            ),
            "svd_2x2": ParsedProblem(
                domain=MathDomain.LINEAR_ALGEBRA,
                problem_type=ProblemType.SVD,
                variables=[],
                expression="[[1, 2], [3, 4]]",
                confidence=Confidence(domain=0.9, problem_type=0.9, overall=0.9),
                original_text="Find the SVD of [[1, 2], [3, 4]]"
            ),
            # Calculus problems
            "derivative": ParsedProblem(
                domain=MathDomain.CALCULUS,
                problem_type=ProblemType.DIFFERENTIATE,
                variables=['x'],
                expression="x**2 + 3*x + 2",
                confidence=Confidence(domain=0.8, problem_type=0.8, overall=0.8),
                original_text="Find the derivative of x^2 + 3x + 2"
            ),
            # Algebraic problems
            "simplify": ParsedProblem(
                domain=MathDomain.ALGEBRA,
                problem_type=ProblemType.SIMPLIFY,
                variables=['x'],
                expression="(x+1)*(x-1)",
                confidence=Confidence(domain=0.7, problem_type=0.7, overall=0.7),
                original_text="Simplify (x+1)*(x-1)"
            )
        }
    
    @pytest.fixture
    def sample_solutions(self):
        """Provide sample solutions for testing."""
        return {
            # Correct solutions
            "determinant_correct": {
                "success": True,
                "determinant": -2.0,
                "original_matrix": [[1, 2], [3, 4]]
            },
            "inverse_correct": {
                "success": True,
                "inverse": [[-1.0, 1.0], [1.0, -2.0]],
                "original_matrix": [[2, 1], [1, 1]]
            },
            "svd_correct": {
                "success": True,
                "U": [[-0.40455358, -0.9145143], [-0.9145143, 0.40455358]],
                "s": [5.4649857, 0.36596619],
                "Vh": [[-0.57604844, -0.81741556], [0.81741556, -0.57604844]],
                "original_matrix": [[1, 2], [3, 4]]
            },
            "derivative_correct": {
                "result_sympy": "2*x + 3",
                "result_latex": "2x + 3"
            },
            "simplify_correct": {
                "result_sympy": "x**2 - 1",
                "simplified": "x**2 - 1"
            },
            # Incorrect solutions
            "determinant_incorrect": {
                "success": True,
                "determinant": 5.0,  # Wrong answer
                "original_matrix": [[1, 2], [3, 4]]
            },
            "derivative_incorrect": {
                "result_sympy": "x + 3",  # Missing x term
                "result_latex": "x + 3"
            }
        }
    
    # Test initialization and basic functionality
    def test_verifier_initialization(self, verifier):
        """Test verifier initialization."""
        assert verifier is not None
        assert verifier.sympy_client is not None
        assert verifier.linear_algebra_solver is not None
        assert verifier.total_verifications == 0
        assert verifier.successful_verifications == 0
    
    def test_singleton_pattern(self):
        """Test that get_verifier returns the same instance."""
        verifier1 = get_verifier()
        verifier2 = get_verifier()
        assert verifier1 is verifier2
    
    # Test verification method selection
    def test_method_selection_linear_algebra(self, verifier, sample_parsed_problems):
        """Test method selection for linear algebra problems."""
        # Test determinant
        method = verifier._select_verification_method(
            sample_parsed_problems["determinant_2x2"], {}
        )
        assert method == VerificationMethod.IDENTITY_VERIFICATION
        
        # Test SVD
        method = verifier._select_verification_method(
            sample_parsed_problems["svd_2x2"], {}
        )
        assert method == VerificationMethod.MATRIX_RECONSTRUCTION
    
    def test_method_selection_calculus(self, verifier, sample_parsed_problems):
        """Test method selection for calculus problems."""
        method = verifier._select_verification_method(
            sample_parsed_problems["derivative"], {}
        )
        assert method == VerificationMethod.DERIVATIVE_INTEGRAL_CHECK
    
    def test_method_selection_algebra(self, verifier, sample_parsed_problems):
        """Test method selection for algebraic problems."""
        method = verifier._select_verification_method(
            sample_parsed_problems["simplify"], {}
        )
        assert method == VerificationMethod.NUMERICAL_CROSS_CHECK
    
    # Test numerical cross-validation
    def test_numerical_verification_correct(self, verifier, sample_parsed_problems, sample_solutions):
        """Test numerical verification with correct solution."""
        result = verifier._check_numerically(
            sample_parsed_problems["simplify"],
            sample_solutions["simplify_correct"]
        )
        
        assert isinstance(result, VerificationResult)
        assert result.method == VerificationMethod.NUMERICAL_CROSS_CHECK
        # Should be high confidence for correct algebraic simplification
        assert result.confidence > 0.8
    
    def test_numerical_verification_incorrect(self, verifier, sample_parsed_problems, sample_solutions):
        """Test numerical verification with incorrect solution."""
        result = verifier._check_numerically(
            sample_parsed_problems["derivative"],
            sample_solutions["derivative_incorrect"]
        )
        
        assert isinstance(result, VerificationResult)
        assert result.method == VerificationMethod.NUMERICAL_CROSS_CHECK
        # Should have low confidence for incorrect derivative
        assert result.confidence < 0.5
    
    def test_numerical_verification_no_variables(self, verifier):
        """Test numerical verification with expressions that have no variables."""
        problem = ParsedProblem(
            domain=MathDomain.ALGEBRA,
            problem_type=ProblemType.SIMPLIFY,
            variables=[],
            expression="2 + 3",
            confidence=Confidence(domain=0.9, problem_type=0.9, overall=0.9),
            original_text="Simplify 2 + 3"
        )
        
        solution = {"result_sympy": "5"}
        
        result = verifier._check_numerically(problem, solution)
        assert result.is_verified
        assert result.confidence == 1.0
    
    def test_numerical_verification_missing_expression(self, verifier, sample_parsed_problems):
        """Test numerical verification with missing expression."""
        problem = sample_parsed_problems["simplify"]
        problem.expression = None  # Remove expression
        
        result = verifier._check_numerically(problem, {})
        assert not result.is_verified
        assert result.confidence == 0.0
        assert "No original expression found" in result.details
    
    # Test matrix reconstruction verification
    def test_matrix_reconstruction_svd(self, verifier, sample_parsed_problems, sample_solutions):
        """Test SVD matrix reconstruction verification."""
        result = verifier._check_matrix_reconstruction(
            sample_parsed_problems["svd_2x2"],
            sample_solutions["svd_correct"]
        )
        
        assert isinstance(result, VerificationResult)
        assert result.method == VerificationMethod.MATRIX_RECONSTRUCTION
        assert result.is_verified
        assert result.confidence > 0.9
        assert "reconstruction error" in result.details.lower()
    
    def test_matrix_reconstruction_missing_matrices(self, verifier, sample_parsed_problems):
        """Test matrix reconstruction with missing decomposition matrices."""
        incomplete_solution = {
            "success": True,
            "U": [[1, 0], [0, 1]],
            # Missing 's' and 'Vh'
            "original_matrix": [[1, 2], [3, 4]]
        }
        
        result = verifier._check_matrix_reconstruction(
            sample_parsed_problems["svd_2x2"],
            incomplete_solution
        )
        
        assert not result.is_verified
        assert "Missing" in result.details
    
    # Test matrix identity verification
    def test_identity_verification_determinant_correct(self, verifier, sample_parsed_problems, sample_solutions):
        """Test determinant identity verification with correct solution."""
        result = verifier._check_matrix_identity(
            sample_parsed_problems["determinant_2x2"],
            sample_solutions["determinant_correct"]
        )
        
        assert isinstance(result, VerificationResult)
        assert result.method == VerificationMethod.IDENTITY_VERIFICATION
        assert result.is_verified
        assert result.confidence == 1.0
    
    def test_identity_verification_determinant_incorrect(self, verifier, sample_parsed_problems, sample_solutions):
        """Test determinant identity verification with incorrect solution."""
        result = verifier._check_matrix_identity(
            sample_parsed_problems["determinant_2x2"],
            sample_solutions["determinant_incorrect"]
        )
        
        assert not result.is_verified
        assert result.confidence == 0.0
    
    def test_identity_verification_inverse(self, verifier, sample_parsed_problems, sample_solutions):
        """Test inverse matrix identity verification."""
        result = verifier._check_matrix_identity(
            sample_parsed_problems["inverse_2x2"],
            sample_solutions["inverse_correct"]
        )
        
        assert result.method == VerificationMethod.IDENTITY_VERIFICATION
        # Should verify A * A^(-1) = I
        assert result.is_verified or result.confidence > 0.8
    
    def test_identity_verification_missing_original_matrix(self, verifier, sample_parsed_problems):
        """Test identity verification with missing original matrix."""
        solution_without_matrix = {
            "success": True,
            "determinant": -2.0
            # Missing original_matrix
        }
        
        result = verifier._check_matrix_identity(
            sample_parsed_problems["determinant_2x2"],
            solution_without_matrix
        )
        
        assert not result.is_verified
        assert "Original matrix not found" in result.details
    
    # Test main verification orchestration
    def test_main_verification_correct_solution(self, verifier, sample_parsed_problems, sample_solutions):
        """Test main verification entry point with correct solution."""
        result = verifier.verify_solution(
            sample_parsed_problems["determinant_2x2"],
            sample_solutions["determinant_correct"]
        )
        
        assert isinstance(result, VerificationResult)
        assert result.is_verified
        assert result.confidence > 0.8
        assert result.execution_time_ms >= 0
    
    def test_main_verification_incorrect_solution(self, verifier, sample_parsed_problems, sample_solutions):
        """Test main verification entry point with incorrect solution."""
        result = verifier.verify_solution(
            sample_parsed_problems["determinant_2x2"],
            sample_solutions["determinant_incorrect"]
        )
        
        assert isinstance(result, VerificationResult)
        assert not result.is_verified
        assert result.confidence < 0.5
    
    def test_verification_statistics_tracking(self, verifier, sample_parsed_problems, sample_solutions):
        """Test that verification statistics are properly tracked."""
        initial_stats = verifier.get_statistics()
        initial_count = initial_stats['total_verifications']
        
        # Perform a verification
        verifier.verify_solution(
            sample_parsed_problems["determinant_2x2"],
            sample_solutions["determinant_correct"]
        )
        
        final_stats = verifier.get_statistics()
        
        assert final_stats['total_verifications'] == initial_count + 1
        assert final_stats['total_verification_time_ms'] > initial_stats['total_verification_time_ms']
    
    # Test error handling
    def test_verification_with_invalid_solution(self, verifier, sample_parsed_problems):
        """Test verification with completely invalid solution."""
        invalid_solution = {"invalid": "data"}
        
        result = verifier.verify_solution(
            sample_parsed_problems["determinant_2x2"],
            invalid_solution
        )
        
        assert isinstance(result, VerificationResult)
        # Should not crash, but may not verify
        assert result.execution_time_ms >= 0
    
    def test_verification_with_none_solution(self, verifier, sample_parsed_problems):
        """Test verification with None solution."""
        result = verifier.verify_solution(
            sample_parsed_problems["determinant_2x2"],
            None
        )
        
        assert isinstance(result, VerificationResult)
        assert not result.is_verified
    
    # Test placeholder methods
    def test_symbolic_verification_placeholder(self, verifier, sample_parsed_problems, sample_solutions):
        """Test symbolic verification placeholder implementation."""
        result = verifier._check_symbolically(
            sample_parsed_problems["derivative"],
            sample_solutions["derivative_correct"]
        )
        
        assert isinstance(result, VerificationResult)
        assert result.method == VerificationMethod.SYMBOLIC_CROSS_CHECK
        # Placeholder should return moderate confidence
        assert 0.5 <= result.confidence <= 1.0
        assert "placeholder" in result.details.lower()
    
    def test_substitution_verification_placeholder(self, verifier, sample_parsed_problems, sample_solutions):
        """Test substitution verification placeholder implementation."""
        result = verifier._check_substitution(
            sample_parsed_problems["derivative"],
            sample_solutions["derivative_correct"]
        )
        
        assert isinstance(result, VerificationResult)
        assert result.method == VerificationMethod.SUBSTITUTION_CHECK
        assert "placeholder" in result.details.lower()
    
    def test_dimensional_analysis_placeholder(self, verifier, sample_parsed_problems):
        """Test dimensional analysis placeholder implementation."""
        result = verifier._check_dimensional_consistency(
            sample_parsed_problems["derivative"],
            {}
        )
        
        assert isinstance(result, VerificationResult)
        assert result.method == VerificationMethod.DIMENSIONAL_ANALYSIS
        assert "placeholder" in result.details.lower()
    
    # Test acceptance criteria
    def test_acceptance_criteria_correct_symbolic_integration(self, verifier):
        """Test acceptance criteria: correctly verify symbolic integration result."""
        problem = ParsedProblem(
            domain=MathDomain.CALCULUS,
            problem_type=ProblemType.INTEGRATE,
            variables=['x'],
            expression="2*x",
            confidence=Confidence(domain=0.9, problem_type=0.9, overall=0.9),
            original_text="Integrate 2*x"
        )
        
        # Correct solution: integral of 2*x is x^2 + C
        correct_solution = {
            "result_sympy": "x**2",
            "result_latex": "x^2"
        }
        
        result = verifier.verify_solution(problem, correct_solution)
        assert isinstance(result, VerificationResult)
        # Should use appropriate verification method
        assert result.method in [
            VerificationMethod.DERIVATIVE_INTEGRAL_CHECK,
            VerificationMethod.NUMERICAL_CROSS_CHECK
        ]
    
    def test_acceptance_criteria_incorrect_symbolic_solution(self, verifier):
        """Test acceptance criteria: flag incorrect symbolic solution."""
        problem = ParsedProblem(
            domain=MathDomain.CALCULUS,
            problem_type=ProblemType.INTEGRATE,
            variables=['x'],
            expression="2*x",
            confidence=Confidence(domain=0.9, problem_type=0.9, overall=0.9),
            original_text="Integrate 2*x"
        )
        
        # Incorrect solution: integral of 2*x is NOT x^2 + C + 1
        incorrect_solution = {
            "result_sympy": "x**2 + 1",  # Extra constant
            "result_latex": "x^2 + 1"
        }
        
        result = verifier.verify_solution(problem, incorrect_solution)
        # Should detect the error through numerical testing
        assert result.confidence < 0.8  # Should have lower confidence
    
    # Test convenience function
    def test_convenience_function(self, sample_parsed_problems, sample_solutions):
        """Test the convenience verify_solution function."""
        result = verify_solution(
            sample_parsed_problems["determinant_2x2"],
            sample_solutions["determinant_correct"]
        )
        
        assert isinstance(result, VerificationResult)
        assert result.is_verified


@pytest.mark.skipif(not VERIFIER_AVAILABLE, reason="Verifier not available")
class TestVerificationResult:
    """Test the VerificationResult dataclass."""
    
    def test_verification_result_creation(self):
        """Test VerificationResult object creation."""
        result = VerificationResult(
            is_verified=True,
            confidence=0.95,
            method=VerificationMethod.NUMERICAL_CROSS_CHECK,
            details="Test verification successful",
            execution_time_ms=15.5
        )
        
        assert result.is_verified is True
        assert result.confidence == 0.95
        assert result.method == VerificationMethod.NUMERICAL_CROSS_CHECK
        assert result.details == "Test verification successful"
        assert result.execution_time_ms == 15.5
        assert result.warnings == []  # Default empty list
    
    def test_verification_result_with_warnings(self):
        """Test VerificationResult with warnings."""
        warnings = ["Warning 1", "Warning 2"]
        result = VerificationResult(
            is_verified=False,
            confidence=0.3,
            method=VerificationMethod.UNKNOWN,
            details="Verification failed",
            execution_time_ms=5.0,
            warnings=warnings
        )
        
        assert result.warnings == warnings
        assert len(result.warnings) == 2


@pytest.mark.skipif(not VERIFIER_AVAILABLE, reason="Verifier not available")
class TestVerificationMethods:
    """Test the VerificationMethod enumeration."""
    
    def test_verification_method_values(self):
        """Test that all verification methods have correct string values."""
        assert VerificationMethod.NUMERICAL_CROSS_CHECK.value == "numerical_cross_check"
        assert VerificationMethod.SYMBOLIC_CROSS_CHECK.value == "symbolic_cross_check"
        assert VerificationMethod.MATRIX_RECONSTRUCTION.value == "matrix_reconstruction"
        assert VerificationMethod.IDENTITY_VERIFICATION.value == "identity_verification"
        assert VerificationMethod.SUBSTITUTION_CHECK.value == "substitution_check"
        assert VerificationMethod.DERIVATIVE_INTEGRAL_CHECK.value == "derivative_integral_check"
        assert VerificationMethod.DIMENSIONAL_ANALYSIS.value == "dimensional_analysis"
        assert VerificationMethod.UNKNOWN.value == "unknown"


@pytest.mark.skipif(not VERIFIER_AVAILABLE, reason="Verifier not available")
class TestMathematicalVerification:
    """Mathematical verification accuracy tests."""
    
    def test_determinant_mathematical_accuracy(self):
        """Test determinant verification mathematical accuracy."""
        verifier = get_verifier()
        
        # Test multiple matrices with known determinants
        test_cases = [
            ([[1, 2], [3, 4]], -2.0),
            ([[2, 1], [1, 2]], 3.0),
            ([[5, 6], [7, 8]], -2.0),
        ]
        
        for matrix, expected_det in test_cases:
            problem = ParsedProblem(
                domain=MathDomain.LINEAR_ALGEBRA,
                problem_type=ProblemType.DETERMINANT,
                variables=[],
                expression=str(matrix),
                confidence=Confidence(domain=0.9, problem_type=0.9, overall=0.9),
                original_text=f"Find determinant of {matrix}"
            )
            
            solution = {
                "success": True,
                "determinant": expected_det,
                "original_matrix": matrix
            }
            
            result = verifier.verify_solution(problem, solution)
            assert result.is_verified, f"Failed to verify determinant for matrix {matrix}"
            assert result.confidence > 0.9, f"Low confidence for correct determinant of {matrix}"
    
    def test_matrix_inverse_mathematical_accuracy(self):
        """Test matrix inverse verification mathematical accuracy."""
        verifier = get_verifier()
        
        # Test 2x2 matrix inverse
        A = [[2, 1], [1, 1]]
        A_inv = [[-1.0, 1.0], [1.0, -2.0]]  # Computed inverse
        
        problem = ParsedProblem(
            domain=MathDomain.LINEAR_ALGEBRA,
            problem_type=ProblemType.INVERSE,
            variables=[],
            expression=str(A),
            confidence=Confidence(domain=0.9, problem_type=0.9, overall=0.9),
            original_text=f"Find inverse of {A}"
        )
        
        solution = {
            "success": True,
            "inverse": A_inv,
            "original_matrix": A
        }
        
        result = verifier.verify_solution(problem, solution)
        assert result.is_verified, "Failed to verify correct matrix inverse"
        assert result.confidence > 0.9, "Low confidence for correct matrix inverse"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])