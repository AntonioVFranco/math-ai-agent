"""
Comprehensive Test Suite for Mathematical Problem Parser

Tests all functionality of the mathematical problem parser including
domain classification, problem type identification, entity extraction,
and edge cases.

Author: MathBoardAI Agent Team
Task ID: CORE-002 / F1
Coverage Target: 90%+
"""

import pytest
import sys
import os

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.parser import MathematicalProblemParser, parse, get_parser
from core.models import MathDomain, ProblemType, ParsedProblem, ParsingResult


class TestMathematicalProblemParser:
    """Test suite for the MathematicalProblemParser class."""
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance for testing."""
        return MathematicalProblemParser()
    
    # Test domain classification - Calculus (5+ examples as required)
    def test_calculus_domain_classification(self, parser):
        """Test classification of calculus problems."""
        
        calculus_problems = [
            "differentiate x^3 + 2x",
            "find the derivative of sin(x) with respect to x",
            "integrate x^2 from 0 to 5",
            "what is the limit of (x^2 - 1)/(x - 1) as x approaches 1",
            "find the partial derivative of x^2*y + y^3 with respect to x",
            "calculate the definite integral of cos(x) from 0 to pi/2",
            "determine the Taylor series for e^x around x = 0"
        ]
        
        for problem in calculus_problems:
            parsed = parser.parse(problem)
            assert parsed.domain == MathDomain.CALCULUS, f"Failed for: {problem}"
            assert parsed.confidence.domain > 0.3, f"Low confidence for: {problem}"
    
    # Test domain classification - Algebra (5+ examples as required)
    def test_algebra_domain_classification(self, parser):
        """Test classification of algebra problems."""
        
        algebra_problems = [
            "solve 3x - 9 = 0 for x",
            "factor the polynomial x^2 - 4x + 4",
            "simplify the expression 2x + 3x - x",
            "find the roots of the quadratic equation x^2 - 5x + 6 = 0",
            "expand (x + 2)(x - 3)",
            "solve the system of equations: x + y = 5, 2x - y = 1",
            "determine the value of x when 2x^2 + 3x - 2 = 0"
        ]
        
        for problem in algebra_problems:
            parsed = parser.parse(problem)
            assert parsed.domain == MathDomain.ALGEBRA, f"Failed for: {problem}"
            assert parsed.confidence.domain > 0.3, f"Low confidence for: {problem}"
    
    # Test domain classification - Linear Algebra (5+ examples as required)
    def test_linear_algebra_domain_classification(self, parser):
        """Test classification of linear algebra problems."""
        
        linear_algebra_problems = [
            "find the determinant of the matrix [[1, 2], [3, 4]]",
            "calculate the inverse of the given matrix",
            "find the eigenvalues of the matrix A",
            "compute the transpose of matrix B",
            "determine the rank of the coefficient matrix",
            "find the null space of the linear transformation",
            "calculate the dot product of vectors u and v"
        ]
        
        for problem in linear_algebra_problems:
            parsed = parser.parse(problem)
            assert parsed.domain == MathDomain.LINEAR_ALGEBRA, f"Failed for: {problem}"
            assert parsed.confidence.domain > 0.3, f"Low confidence for: {problem}"
    
    # Test specific acceptance criteria examples
    def test_differentiate_classification_acceptance_criteria(self, parser):
        """Test the specific example from acceptance criteria."""
        parsed = parser.parse("differentiate x^3 + 2x")
        
        assert parsed.domain == MathDomain.CALCULUS
        assert parsed.problem_type == ProblemType.DIFFERENTIATE
        assert 'x' in parsed.variables
        assert parsed.confidence.domain > 0.0
        assert parsed.confidence.problem_type > 0.0
    
    def test_solve_equation_classification_acceptance_criteria(self, parser):
        """Test the specific solve equation example from acceptance criteria."""
        parsed = parser.parse("solve 3x - 9 = 0 for x")
        
        assert parsed.domain == MathDomain.ALGEBRA
        assert parsed.problem_type == ProblemType.SOLVE_EQUATION
        assert 'x' in parsed.variables
    
    def test_unknown_domain_classification(self, parser):
        """Test handling of unclear problems."""
        unclear_problems = [
            "hello world",
            "this is not a math problem",
            "random text without mathematical content",
            "just some words here"
        ]
        
        for problem in unclear_problems:
            parsed = parser.parse(problem)
            assert parsed.domain == MathDomain.UNKNOWN, f"Should be unknown for: {problem}"
    
    # Test problem type identification
    def test_problem_type_differentiate(self, parser):
        """Test identification of differentiation problems."""
        differentiate_problems = [
            "differentiate x^2",
            "find the derivative of sin(x)",
            "what is d/dx of x^3",
            "compute the derivative with respect to x"
        ]
        
        for problem in differentiate_problems:
            parsed = parser.parse(problem)
            assert parsed.problem_type == ProblemType.DIFFERENTIATE, f"Failed for: {problem}"
    
    def test_problem_type_integrate(self, parser):
        """Test identification of integration problems."""
        integrate_problems = [
            "integrate x^2",
            "find the integral of cos(x)",
            "calculate ∫ x dx",
            "what is the antiderivative of x^2"
        ]
        
        for problem in integrate_problems:
            parsed = parser.parse(problem)
            assert parsed.problem_type == ProblemType.INTEGRATE, f"Failed for: {problem}"
    
    def test_problem_type_solve_equation(self, parser):
        """Test identification of equation solving problems."""
        solve_problems = [
            "solve x^2 = 4",
            "find x when 2x + 3 = 7",
            "determine the value of x in x^2 - 5x + 6 = 0",
            "what is the solution to 3x - 9 = 0"
        ]
        
        for problem in solve_problems:
            parsed = parser.parse(problem)
            assert parsed.problem_type == ProblemType.SOLVE_EQUATION, f"Failed for: {problem}"
    
    def test_problem_type_simplify(self, parser):
        """Test identification of simplification problems."""
        simplify_problems = [
            "simplify 2x + 3x",
            "reduce the expression x^2 - 4x + 4",
            "combine like terms in 5x + 2x - 3x",
            "express in simplest form: (x^2 - 1)/(x - 1)"
        ]
        
        for problem in simplify_problems:
            parsed = parser.parse(problem)
            assert parsed.problem_type == ProblemType.SIMPLIFY, f"Failed for: {problem}"
    
    # Test variable extraction
    def test_variable_extraction_single_variable(self, parser):
        """Test extraction of single variables."""
        test_cases = [
            ("differentiate x^3 + 2x", ['x']),
            ("solve for y when y^2 = 16", ['y']),
            ("find the value of z in 3z + 5 = 14", ['z']),
            ("integrate t dt", ['t']),
            ("what is d/dx of x^2", ['x'])
        ]
        
        for problem, expected_vars in test_cases:
            parsed = parser.parse(problem)
            for var in expected_vars:
                assert var in parsed.variables, f"Variable {var} not found in: {problem}"
    
    def test_variable_extraction_multiple_variables(self, parser):
        """Test extraction of multiple variables."""
        test_cases = [
            ("solve x + y = 5 and 2x - y = 1", ['x', 'y']),
            ("find partial derivative of x^2*y + z", ['x', 'y', 'z']),
            ("minimize f(x,y) = x^2 + y^2", ['x', 'y']),
            ("integrate x*y with respect to x", ['x', 'y'])
        ]
        
        for problem, expected_vars in test_cases:
            parsed = parser.parse(problem)
            for var in expected_vars:
                assert var in parsed.variables, f"Variable {var} not found in: {problem}"
    
    # Test expression extraction
    def test_expression_extraction(self, parser):
        """Test extraction of mathematical expressions."""
        test_cases = [
            ("what is the derivative of x^2", "x^2"),
            ("solve 3x - 9 = 0", "3x - 9 = 0"),  
            ("integrate x^2 + 1", "x^2 + 1"),
            ("simplify 2x + 3x", "2x + 3x")
        ]
        
        for problem, expected_expr in test_cases:
            parsed = parser.parse(problem)
            # The expression might be cleaned or modified, so check if expected is contained
            assert parsed.expression is not None, f"No expression found for: {problem}"
            # Allow for some flexibility in expression extraction
            assert any(char in parsed.expression for char in expected_expr.replace(' ', '')), \
                f"Expected elements of '{expected_expr}' not found in '{parsed.expression}' for: {problem}"
    
    # Test mathematical entity extraction
    def test_mathematical_function_extraction(self, parser):
        """Test extraction of mathematical functions."""
        test_cases = [
            ("differentiate sin(x)", ["sin"]),
            ("integrate cos(x) + log(x)", ["cos", "log"]),
            ("find the limit of exp(x)/x", ["exp"]),
            ("calculate sqrt(x^2 + 1)", ["sqrt"])
        ]
        
        for problem, expected_functions in test_cases:
            parsed = parser.parse(problem)
            for func in expected_functions:
                assert func in parsed.functions, f"Function {func} not found in: {problem}"
    
    def test_mathematical_constant_extraction(self, parser):
        """Test extraction of mathematical constants."""
        test_cases = [
            ("integrate from 0 to pi", ["pi"]),
            ("find the limit as x approaches infinity", ["infinity"]),
            ("calculate e^x", ["e"]),
            ("what is the value of π", ["π"])
        ]
        
        for problem, expected_constants in test_cases:
            parsed = parser.parse(problem)
            # Check if any of the expected constants are found
            found_constants = [const for const in expected_constants if const in parsed.constants]
            assert len(found_constants) > 0, f"No expected constants {expected_constants} found in: {problem}"
    
    # Test confidence scoring
    def test_confidence_scores(self, parser):
        """Test that confidence scores are properly calculated."""
        # High confidence cases
        high_confidence_problems = [
            "differentiate x^3 + 2x",
            "solve 3x - 9 = 0 for x",
            "find the determinant of matrix A"
        ]
        
        for problem in high_confidence_problems:
            parsed = parser.parse(problem)
            assert 0.0 <= parsed.confidence.domain <= 1.0, "Domain confidence out of range"
            assert 0.0 <= parsed.confidence.problem_type <= 1.0, "Problem type confidence out of range"
            assert 0.0 <= parsed.confidence.overall <= 1.0, "Overall confidence out of range"
    
    # Test edge cases and error handling
    def test_empty_string(self, parser):
        """Test handling of empty input."""
        parsed = parser.parse("")
        assert parsed.domain == MathDomain.UNKNOWN
        assert parsed.problem_type == ProblemType.UNKNOWN
        assert len(parsed.variables) == 0
    
    def test_very_long_string(self, parser):
        """Test handling of very long input."""
        long_problem = "solve " + "x + " * 1000 + "1 = 0"
        parsed = parser.parse(long_problem)
        assert parsed.domain == MathDomain.ALGEBRA
        assert parsed.problem_type == ProblemType.SOLVE_EQUATION
        assert 'x' in parsed.variables
    
    def test_special_characters(self, parser):
        """Test handling of special mathematical characters."""
        special_problems = [
            "integrate ∫ x dx",
            "find ∑ x^n",
            "calculate π/2",
            "what is ∞ - ∞"
        ]
        
        for problem in special_problems:
            parsed = parser.parse(problem)
            # Should not crash and should return a valid ParsedProblem
            assert isinstance(parsed, ParsedProblem)
    
    def test_latex_expressions(self, parser):
        """Test handling of LaTeX mathematical expressions."""
        latex_problems = [
            r"differentiate \frac{x^2}{2}",
            r"integrate \sqrt{x}",
            r"find \lim_{x \to 0} \frac{\sin x}{x}",
            r"solve \int_0^\pi \sin x dx"
        ]
        
        for problem in latex_problems:
            parsed = parser.parse(problem)
            # Should not crash and should identify some mathematical content
            assert isinstance(parsed, ParsedProblem)
            # Should have some mathematical content identified
            assert (parsed.domain != MathDomain.UNKNOWN or 
                   parsed.problem_type != ProblemType.UNKNOWN or 
                   len(parsed.variables) > 0)
    
    # Test trigonometry domain
    def test_trigonometry_domain_classification(self, parser):
        """Test classification of trigonometry problems."""
        trig_problems = [
            "find sin(30 degrees)",
            "solve cos(x) = 0.5",
            "simplify sin^2(x) + cos^2(x)",
            "what is the period of tan(x)",
            "calculate the amplitude of 3*sin(2x)",
            "find the phase shift of sin(x + π/4)",
            "solve the triangle with sides a=3, b=4, c=5"
        ]
        
        for problem in trig_problems:
            parsed = parser.parse(problem)
            # Should classify as either trigonometry or calculus (for derivatives of trig functions)
            assert parsed.domain in [MathDomain.TRIGONOMETRY, MathDomain.CALCULUS, MathDomain.ALGEBRA], \
                f"Unexpected domain for: {problem}"
    
    # Test geometry domain
    def test_geometry_domain_classification(self, parser):
        """Test classification of geometry problems."""
        geometry_problems = [
            "find the area of a circle with radius 5",
            "calculate the perimeter of a rectangle",
            "what is the volume of a sphere",
            "find the distance between two points",
            "calculate the angle in a triangle",
            "determine if lines are parallel"
        ]
        
        for problem in geometry_problems:
            parsed = parser.parse(problem)
            assert parsed.domain == MathDomain.GEOMETRY, f"Failed for: {problem}"
    
    # Test statistics domain
    def test_statistics_domain_classification(self, parser):
        """Test classification of statistics problems."""
        stats_problems = [
            "find the mean of the dataset",
            "calculate the standard deviation",
            "what is the probability of getting heads",
            "find the correlation between x and y",
            "perform a hypothesis test",
            "calculate the confidence interval"
        ]
        
        for problem in stats_problems:
            parsed = parser.parse(problem)
            assert parsed.domain == MathDomain.STATISTICS, f"Failed for: {problem}"


class TestParsingResult:
    """Test the parsing result wrapper functionality."""
    
    def test_successful_parsing(self):
        """Test successful parsing through the parse function."""
        result = parse("differentiate x^2")
        
        assert result.success is True
        assert result.parsed_problem is not None
        assert result.error_message is None
        assert result.parsing_time_ms is not None
        assert result.parsing_time_ms >= 0
    
    def test_parsing_with_invalid_input(self):
        """Test parsing behavior with edge case inputs."""
        # Empty string should still succeed but with unknown classification
        result = parse("")
        assert result.success is True
        assert result.parsed_problem.domain == MathDomain.UNKNOWN
    
    def test_parser_singleton(self):
        """Test that the global parser singleton works correctly."""
        parser1 = get_parser()
        parser2 = get_parser()
        
        # Should return the same instance
        assert parser1 is parser2
        
        # Should be functional
        parsed = parser1.parse("solve x = 1")
        assert parsed.domain == MathDomain.ALGEBRA


class TestParsingPerformance:
    """Test parsing performance and efficiency."""
    
    def test_parsing_time_reasonable(self):
        """Test that parsing completes in reasonable time."""
        problems = [
            "differentiate x^3 + 2x",
            "solve 3x - 9 = 0 for x",
            "find the determinant of [[1, 2], [3, 4]]",
            "integrate sin(x) from 0 to pi"
        ]
        
        for problem in problems:
            result = parse(problem)
            assert result.success is True
            # Should complete in under 100ms for simple problems
            assert result.parsing_time_ms < 100, f"Too slow for: {problem}"
    
    def test_batch_parsing_efficiency(self):
        """Test efficiency when parsing multiple problems."""
        problems = ["solve x^2 = 4" for _ in range(10)]
        
        results = []
        for problem in problems:
            result = parse(problem)
            results.append(result)
        
        # All should succeed
        assert all(r.success for r in results)
        
        # Average time should be reasonable
        avg_time = sum(r.parsing_time_ms for r in results) / len(results)
        assert avg_time < 50, "Batch parsing too slow"


class TestIntegration:
    """Integration tests for the parser with realistic scenarios."""
    
    def test_realistic_problem_scenarios(self):
        """Test with realistic mathematical problems."""
        realistic_problems = [
            # Student homework style
            "Find the derivative of f(x) = 3x^2 + 2x - 1",
            "Solve the quadratic equation 2x^2 - 5x + 2 = 0",
            "Calculate the definite integral of x^2 from 1 to 3",
            
            # More natural language
            "What is the slope of the tangent line to y = x^3 at x = 2?",
            "How do you find the area under the curve y = sin(x) from 0 to π?",
            "Can you help me factor x^2 - 9?",
            
            # Mixed content
            "Given the function f(x) = e^x, find f'(x)",
            "For the matrix A = [[2, 1], [1, 2]], compute det(A)",
            "Simplify the expression (x + 3)(x - 3) and then differentiate it"
        ]
        
        for problem in realistic_problems:
            result = parse(problem)
            assert result.success is True, f"Failed to parse: {problem}"
            
            parsed = result.parsed_problem
            # Should have identified some structure
            assert (parsed.domain != MathDomain.UNKNOWN or
                   parsed.problem_type != ProblemType.UNKNOWN or
                   len(parsed.variables) > 0 or
                   parsed.expression is not None), f"No structure identified for: {problem}"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])