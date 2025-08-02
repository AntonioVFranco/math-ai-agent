"""
Mathematical Problem Parser

A foundational, rule-based parser that analyzes raw text strings containing
mathematical problems. Classifies domain, identifies intent, and extracts
key mathematical entities.

Author: Math AI Agent Team
Task ID: CORE-002 / F1
"""

import re
import time
import logging
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter

from .models import (
    ParsedProblem, ParsingResult, MathDomain, ProblemType,
    MathEntity, Confidence
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathematicalProblemParser:
    """
    A rule-based parser for mathematical problems.
    
    This parser analyzes raw text to classify mathematical domains,
    identify problem types, and extract mathematical entities like
    variables, expressions, and operators.
    """
    
    def __init__(self):
        """Initialize the parser with predefined patterns and keywords."""
        self._initialize_patterns()
        self._initialize_keywords()
        
    def _initialize_patterns(self) -> None:
        """Initialize regex patterns for entity extraction."""
        
        # Variable patterns (single letters, optionally with subscripts/superscripts)
        self.variable_pattern = re.compile(r'\b[a-zA-Z](?:[_\d]*|\^[^\s\w]*)?\b')
        
        # Mathematical expression patterns
        self.expression_patterns = {
            'polynomial': re.compile(r'[a-zA-Z]\^?\d*\s*[\+\-]\s*\d*[a-zA-Z]*\^?\d*'),
            'fraction': re.compile(r'\d*[a-zA-Z]*\s*/\s*\d*[a-zA-Z]*'),
            'equation': re.compile(r'.*=.*'),
            'function_call': re.compile(r'[a-zA-Z]+\([^)]+\)'),
            'integral': re.compile(r'∫|\\int'),
            'derivative': re.compile(r'd/d[a-zA-Z]|\\frac{d}{d[a-zA-Z]}'),
            'summation': re.compile(r'∑|\\sum'),
            'limit': re.compile(r'lim|\\lim'),
            'matrix': re.compile(r'\[\[.*?\]\]|\[\s*\[.*?\]\s*,\s*\[.*?\]\s*\]|matrix\s*\([^)]+\)|\[\s*\d+.*?\d+\s*\]'),
        }
        
        # Number patterns
        self.number_pattern = re.compile(r'-?\d*\.?\d+(?:[eE][+-]?\d+)?')
        
        # Mathematical operators
        self.operator_pattern = re.compile(r'[\+\-\*/\^=<>≤≥≠∈∉∪∩]')
        
        # Mathematical functions
        self.function_pattern = re.compile(r'\b(?:sin|cos|tan|log|ln|exp|sqrt|abs|max|min|det|inv)\b')
        
        # Mathematical constants
        self.constant_pattern = re.compile(r'\b(?:pi|π|e|i|∞|infinity)\b')
        
    def _initialize_keywords(self) -> None:
        """Initialize keyword mappings for domain and problem type classification."""
        
        # Domain classification keywords
        self.domain_keywords = {
            MathDomain.CALCULUS: {
                'differentiate', 'derivative', 'integrate', 'integral', 'limit', 'continuous',
                'series', 'taylor', 'partial', 'gradient', 'divergence', 'curl', 'flux',
                'd/dx', 'dy/dx', '∫', '∂', 'lim', 'sup', 'inf'
            },
            MathDomain.LINEAR_ALGEBRA: {
                'matrix', 'vector', 'eigenvalue', 'eigenvector', 'determinant', 'inverse',
                'transpose', 'rank', 'null', 'space', 'basis', 'dimension', 'orthogonal',
                'projection', 'span', 'linear', 'transformation', 'det', 'trace',
                'lu decomposition', 'qr decomposition', 'svd', 'singular value decomposition',
                'eigendecomposition', 'diagonalization', 'cholesky', 'gram-schmidt',
                'orthonormal', 'orthogonalization', 'condition number', 'spectral radius',
                'frobenius norm', 'singular values', 'left singular vectors', 'right singular vectors'
            },
            MathDomain.ALGEBRA: {
                'solve', 'equation', 'system', 'factor', 'expand', 'simplify', 'polynomial',
                'quadratic', 'linear', 'root', 'zero', 'coefficient', 'variable', 'unknown'
            },
            MathDomain.TRIGONOMETRY: {
                'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'asin', 'acos', 'atan',
                'sinh', 'cosh', 'tanh', 'angle', 'triangle', 'radian', 'degree',
                'amplitude', 'period', 'phase', 'trigonometric'
            },
            MathDomain.GEOMETRY: {
                'point', 'line', 'circle', 'triangle', 'rectangle', 'polygon', 'area',
                'perimeter', 'volume', 'surface', 'angle', 'distance', 'coordinate',
                'geometric', 'congruent', 'similar', 'parallel', 'perpendicular'
            },
            MathDomain.STATISTICS: {
                'mean', 'median', 'mode', 'variance', 'standard', 'deviation', 'probability',
                'distribution', 'normal', 'binomial', 'poisson', 'correlation', 'regression',
                'hypothesis', 'test', 'confidence', 'interval', 'sample', 'population',
                'statistics', 'statistical', 'data', 'average', 'std', 'pdf', 'pmf', 'cdf',
                'cumulative', 'density', 'mass', 'function', 'random', 'variates', 'generate',
                't-test', 'ttest', 'p-value', 'pvalue', 'significance', 'alpha', 'null hypothesis',
                'alternative hypothesis', 'two-sample', 'independent', 'welch', 'equal variance', 
                'descriptive stats', 'descriptive statistics', 'summary statistics', 'percentile',
                'quartile', 'iqr', 'interquartile', 'skewness', 'kurtosis', 'shapiro', 'normality',
                'uniform', 'exponential', 'gamma', 'beta', 'chi-square', 'chi2', 'student',
                'f-distribution', 'lognormal', 'weibull', 'pareto', 'kolmogorov', 'jarque', 'bera',
                'correlation coefficient', 'pearson', 'spearman', 'kendall', 'covariance',
                'bayesian', 'prior', 'posterior', 'likelihood', 'frequentist', 'estimation',
                'maximum likelihood', 'mle', 'anova', 'chi-square test', 'goodness of fit',
                'contingency table', 'independence test', 'homogeneity test', 'effect size',
                'cohens d', 'power analysis', 'sample size', 'central limit theorem', 'clt'
            },
            MathDomain.OPTIMIZATION: {
                'minimize', 'maximize', 'optimization', 'optimize', 'minimum', 'maximum',
                'gradient descent', 'gradient', 'descent', 'find minimum', 'find maximum',
                'critical points', 'critical point', 'extrema', 'local minimum', 'local maximum',
                'global minimum', 'global maximum', 'cost function', 'loss function', 'objective function',
                'learning rate', 'convergence', 'iteration', 'iterative', 'numerical optimization',
                'steepest descent', 'line search', 'backtracking', 'armijo', 'wolfe conditions',
                'momentum', 'adaptive learning rate', 'adam', 'rmsprop', 'adagrad',
                'convex optimization', 'concave', 'convex', 'quadratic programming', 'linear programming',
                'constrained optimization', 'unconstrained optimization', 'lagrange multipliers',
                'karush kuhn tucker', 'kkt conditions', 'penalty method', 'barrier method',
                'newton method', 'quasi-newton', 'bfgs', 'l-bfgs', 'conjugate gradient',
                'trust region', 'levenberg marquardt', 'gauss newton', 'machine learning',
                'neural network training', 'backpropagation', 'stochastic gradient descent', 'sgd',
                'batch gradient descent', 'mini-batch', 'online learning', 'reinforcement learning'
            },
            MathDomain.NUMBER_THEORY: {
                'prime', 'composite', 'factor', 'divisible', 'gcd', 'lcm', 'modular',
                'congruent', 'remainder', 'integer', 'rational', 'irrational', 'real',
                'complex', 'number', 'theory'
            }
        }
        
        # Problem type classification keywords
        self.problem_type_keywords = {
            ProblemType.SOLVE_EQUATION: {
                'solve', 'find', 'determine', 'calculate', 'what', 'value', 'root', 'zero', 'solution'
            },
            ProblemType.DIFFERENTIATE: {
                'differentiate', 'derivative', 'find the derivative', 'derive', 'd/dx', 'rate of change'
            },
            ProblemType.INTEGRATE: {
                'integrate', 'integral', 'antiderivative', 'area under', 'definite integral', '∫'
            },
            ProblemType.SIMPLIFY: {
                'simplify', 'reduce', 'combine', 'condense', 'express in simplest form'
            },
            ProblemType.FACTOR: {
                'factor', 'factorize', 'factorization', 'factor completely'
            },
            ProblemType.EXPAND: {
                'expand', 'multiply out', 'distribute', 'foil'
            },
            ProblemType.EVALUATE: {
                'evaluate', 'compute', 'calculate', 'find the value'
            },
            ProblemType.MATRIX_OPERATIONS: {
                'transpose', 'multiply matrices', 'matrix', 'matrices', 'rank', 'trace',
                'matrix multiplication', 'matrix addition', 'matrix subtraction'
            },
            ProblemType.DETERMINANT: {
                'determinant', 'det', 'find the determinant', 'compute the determinant',
                'calculate the determinant', 'what is the determinant'
            },
            ProblemType.INVERSE: {
                'inverse', 'find the inverse', 'compute the inverse', 'calculate the inverse',
                'matrix inverse', 'invert the matrix', 'what is the inverse'
            },
            ProblemType.LU_DECOMPOSITION: {
                'lu decomposition', 'lu factorization', 'perform lu', 'find the lu',
                'lu decompose', 'lower upper decomposition'
            },
            ProblemType.QR_DECOMPOSITION: {
                'qr decomposition', 'qr factorization', 'perform qr', 'find the qr',
                'qr decompose', 'gram schmidt', 'orthogonal decomposition'
            },
            ProblemType.SVD: {
                'svd', 'singular value decomposition', 'find the svd', 'perform svd',
                'singular values', 'compute svd', 'calculate svd'
            },
            ProblemType.EIGEN_DECOMPOSITION: {
                'eigenvalue', 'eigenvector', 'eigendecomposition', 'diagonalize',
                'find eigenvalues', 'compute eigenvalues', 'find eigenvectors',
                'spectral decomposition', 'eigenanalysis'
            },
            ProblemType.LIMIT: {
                'limit', 'approach', 'tends to', 'as x approaches', 'lim'
            },
            ProblemType.VERIFY: {
                'verify', 'check', 'prove', 'show that', 'demonstrate'
            },
            ProblemType.DESCRIPTIVE_STATISTICS: {
                'calculate', 'find', 'compute', 'descriptive statistics', 'summary statistics',
                'mean', 'median', 'mode', 'variance', 'standard deviation', 'std dev',
                'average', 'quartile', 'percentile', 'iqr', 'skewness', 'kurtosis',
                'minimum', 'maximum', 'range', 'statistics for', 'stats for', 'analyze data'
            },
            ProblemType.HYPOTHESIS_TEST: {
                'hypothesis test', 'test the hypothesis', 'statistical test', 'significance test',
                'null hypothesis', 'alternative hypothesis', 'reject', 'accept', 'fail to reject',
                'statistical significance', 'p-value', 'alpha level', 'critical value'
            },
            ProblemType.T_TEST: {
                't-test', 'ttest', 't test', 'two sample t-test', 'independent t-test',
                'welch test', 'student t-test', 'paired t-test', 'one sample t-test',
                'compare means', 'difference between means', 'equal variances'
            },
            ProblemType.DISTRIBUTION_ANALYSIS: {
                'distribution', 'probability distribution', 'pdf', 'pmf', 'cdf',
                'normal distribution', 'binomial distribution', 'poisson distribution',
                'uniform distribution', 'exponential distribution', 'gamma distribution',
                'beta distribution', 'chi-square distribution', 'student t distribution',
                'f distribution', 'probability density', 'cumulative distribution',
                'distribution parameters', 'fit distribution'
            },
            ProblemType.RANDOM_GENERATION: {
                'generate', 'random', 'sample', 'simulate', 'random variates',
                'random numbers', 'random sample', 'monte carlo', 'simulation',
                'generate from distribution', 'random draw', 'sampling'
            },
            ProblemType.CORRELATION_ANALYSIS: {
                'correlation', 'correlation coefficient', 'pearson correlation',
                'spearman correlation', 'kendall correlation', 'covariance',
                'relationship', 'association', 'linear relationship', 'correlation test'
            },
            ProblemType.NORMALITY_TEST: {
                'normality test', 'test for normality', 'normal distribution test',
                'shapiro wilk', 'shapiro test', 'kolmogorov smirnov', 'jarque bera',
                'anderson darling', 'qq plot', 'is normally distributed'
            },
            ProblemType.MINIMIZE: {
                'minimize', 'find minimum', 'find the minimum', 'minimize function',
                'minimize the function', 'find minimum value', 'minimum of', 'min',
                'reduce', 'lowest value', 'smallest value', 'optimization problem'
            },
            ProblemType.MAXIMIZE: {
                'maximize', 'find maximum', 'find the maximum', 'maximize function',
                'maximize the function', 'find maximum value', 'maximum of', 'max',
                'highest value', 'largest value', 'peak value'
            },
            ProblemType.GRADIENT_DESCENT: {
                'gradient descent', 'use gradient descent', 'apply gradient descent',
                'gradient descent algorithm', 'steepest descent', 'descent method',
                'iterative optimization', 'gradient-based optimization'
            },
            ProblemType.FIND_CRITICAL_POINTS: {
                'critical points', 'find critical points', 'critical point',
                'stationary points', 'extrema', 'local extrema', 'inflection points',
                'where gradient is zero', 'where derivative is zero'
            },
            ProblemType.OPTIMIZATION: {
                'optimize', 'optimization', 'optimal', 'best value', 'optimal solution',
                'optimization problem', 'constrained optimization', 'unconstrained optimization',
                'numerical optimization', 'convex optimization', 'nonlinear optimization'
            }
        }
    
    def _classify_domain(self, text: str) -> Tuple[MathDomain, float]:
        """
        Classify the mathematical domain of the problem.
        
        Args:
            text: The problem text to classify
            
        Returns:
            Tuple of (domain, confidence_score)
        """
        text_lower = text.lower()
        domain_scores = {}
        
        # Score each domain based on keyword matches
        for domain, keywords in self.domain_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in text_lower:
                    # Weight longer, more specific keywords higher
                    weight = len(keyword.split()) * 2 if len(keyword.split()) > 1 else 1
                    score += weight
                    matched_keywords.append(keyword)
            
            # Bonus for multiple different keywords
            if len(matched_keywords) > 1:
                score += len(matched_keywords) * 0.5
                
            domain_scores[domain] = score
        
        # Find the domain with the highest score
        if not domain_scores or max(domain_scores.values()) == 0:
            return MathDomain.UNKNOWN, 0.0
        
        best_domain = max(domain_scores, key=domain_scores.get)
        max_score = domain_scores[best_domain]
        
        # Calculate confidence (normalize by reasonable maximum score)
        max_possible_score = 10  # Reasonable maximum for a single domain
        confidence = min(1.0, max_score / max_possible_score)
        
        return best_domain, confidence
    
    def _identify_problem_type(self, text: str) -> Tuple[ProblemType, float]:
        """
        Identify the type of mathematical problem.
        
        Args:
            text: The problem text to analyze
            
        Returns:
            Tuple of (problem_type, confidence_score)
        """
        text_lower = text.lower()
        type_scores = {}
        
        # Score each problem type based on keyword matches
        for problem_type, keywords in self.problem_type_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in text_lower:
                    # Weight longer, more specific phrases higher
                    weight = len(keyword.split()) * 2 if len(keyword.split()) > 1 else 1
                    score += weight
                    matched_keywords.append(keyword)
            
            # Bonus for multiple matches
            if len(matched_keywords) > 1:
                score += len(matched_keywords) * 0.3
                
            type_scores[problem_type] = score
        
        # Check for mathematical patterns that indicate problem type
        if re.search(r'=', text):
            type_scores[ProblemType.SOLVE_EQUATION] = type_scores.get(ProblemType.SOLVE_EQUATION, 0) + 2
        
        if re.search(r'd/d[a-zA-Z]|derivative|differentiate', text_lower):
            type_scores[ProblemType.DIFFERENTIATE] = type_scores.get(ProblemType.DIFFERENTIATE, 0) + 3
        
        if re.search(r'∫|integral|integrate', text_lower):
            type_scores[ProblemType.INTEGRATE] = type_scores.get(ProblemType.INTEGRATE, 0) + 3
        
        # Linear algebra specific patterns
        if re.search(r'\bdet\b|\bdeterminant\b', text_lower):
            type_scores[ProblemType.DETERMINANT] = type_scores.get(ProblemType.DETERMINANT, 0) + 4
        
        if re.search(r'\binverse\b|\binvert\b', text_lower):
            type_scores[ProblemType.INVERSE] = type_scores.get(ProblemType.INVERSE, 0) + 4
        
        if re.search(r'\blu\s+decomposition\b|\blu\s+factorization\b', text_lower):
            type_scores[ProblemType.LU_DECOMPOSITION] = type_scores.get(ProblemType.LU_DECOMPOSITION, 0) + 5
        
        if re.search(r'\bqr\s+decomposition\b|\bqr\s+factorization\b', text_lower):
            type_scores[ProblemType.QR_DECOMPOSITION] = type_scores.get(ProblemType.QR_DECOMPOSITION, 0) + 5
        
        if re.search(r'\bsvd\b|\bsingular\s+value\s+decomposition\b', text_lower):
            type_scores[ProblemType.SVD] = type_scores.get(ProblemType.SVD, 0) + 5
        
        if re.search(r'\beigenvalue\b|\beigenvector\b|\beigendecomposition\b', text_lower):
            type_scores[ProblemType.EIGEN_DECOMPOSITION] = type_scores.get(ProblemType.EIGEN_DECOMPOSITION, 0) + 4
        
        # Statistical analysis patterns
        if re.search(r'\bt-test\b|\bttest\b|\bt test\b', text_lower):
            type_scores[ProblemType.T_TEST] = type_scores.get(ProblemType.T_TEST, 0) + 5
        
        if re.search(r'\bmean\b|\bmedian\b|\bmode\b|\bvariance\b|\bstandard deviation\b', text_lower):
            type_scores[ProblemType.DESCRIPTIVE_STATISTICS] = type_scores.get(ProblemType.DESCRIPTIVE_STATISTICS, 0) + 3
        
        if re.search(r'\bhypothesis test\b|\bp-value\b|\bnull hypothesis\b', text_lower):
            type_scores[ProblemType.HYPOTHESIS_TEST] = type_scores.get(ProblemType.HYPOTHESIS_TEST, 0) + 4
        
        if re.search(r'\bdistribution\b|\bpdf\b|\bpmf\b|\bcdf\b|\bnormal\b|\bbinomial\b|\bpoisson\b', text_lower):
            type_scores[ProblemType.DISTRIBUTION_ANALYSIS] = type_scores.get(ProblemType.DISTRIBUTION_ANALYSIS, 0) + 3
        
        if re.search(r'\bgenerate\b|\brandom\b|\bsample\b|\bsimulate\b|\brandom variates\b', text_lower):
            type_scores[ProblemType.RANDOM_GENERATION] = type_scores.get(ProblemType.RANDOM_GENERATION, 0) + 3
        
        if re.search(r'\bcorrelation\b|\bcorrelation coefficient\b|\bpearson\b|\bspearman\b', text_lower):
            type_scores[ProblemType.CORRELATION_ANALYSIS] = type_scores.get(ProblemType.CORRELATION_ANALYSIS, 0) + 4
        
        if re.search(r'\bnormality test\b|\bshapiro\b|\bkolmogorov\b|\bnormally distributed\b', text_lower):
            type_scores[ProblemType.NORMALITY_TEST] = type_scores.get(ProblemType.NORMALITY_TEST, 0) + 4
        
        # Optimization patterns
        if re.search(r'\bminimize\b|\bfind minimum\b|\bfind the minimum\b', text_lower):
            type_scores[ProblemType.MINIMIZE] = type_scores.get(ProblemType.MINIMIZE, 0) + 4
        
        if re.search(r'\bmaximize\b|\bfind maximum\b|\bfind the maximum\b', text_lower):
            type_scores[ProblemType.MAXIMIZE] = type_scores.get(ProblemType.MAXIMIZE, 0) + 4
        
        if re.search(r'\bgradient descent\b|\buse gradient descent\b|\bapply gradient descent\b', text_lower):
            type_scores[ProblemType.GRADIENT_DESCENT] = type_scores.get(ProblemType.GRADIENT_DESCENT, 0) + 5
        
        if re.search(r'\bcritical points\b|\bfind critical points\b|\bstationary points\b', text_lower):
            type_scores[ProblemType.FIND_CRITICAL_POINTS] = type_scores.get(ProblemType.FIND_CRITICAL_POINTS, 0) + 4
        
        if re.search(r'\boptimize\b|\boptimization\b|\boptimal\b|\boptimization problem\b', text_lower):
            type_scores[ProblemType.OPTIMIZATION] = type_scores.get(ProblemType.OPTIMIZATION, 0) + 3
        
        # Matrix pattern detection
        if re.search(r'\[\[.*?\]\]|\[\s*\[.*?\]\s*,\s*\[.*?\]\s*\]', text):
            # Detected matrix format, boost linear algebra scores
            if 'determinant' in text_lower or 'det' in text_lower:
                type_scores[ProblemType.DETERMINANT] = type_scores.get(ProblemType.DETERMINANT, 0) + 3
            elif 'inverse' in text_lower:
                type_scores[ProblemType.INVERSE] = type_scores.get(ProblemType.INVERSE, 0) + 3
            elif 'svd' in text_lower or 'singular' in text_lower:
                type_scores[ProblemType.SVD] = type_scores.get(ProblemType.SVD, 0) + 3
            elif 'lu' in text_lower:
                type_scores[ProblemType.LU_DECOMPOSITION] = type_scores.get(ProblemType.LU_DECOMPOSITION, 0) + 3
            elif 'qr' in text_lower:
                type_scores[ProblemType.QR_DECOMPOSITION] = type_scores.get(ProblemType.QR_DECOMPOSITION, 0) + 3
            elif 'eigen' in text_lower:
                type_scores[ProblemType.EIGEN_DECOMPOSITION] = type_scores.get(ProblemType.EIGEN_DECOMPOSITION, 0) + 3
            else:
                type_scores[ProblemType.MATRIX_OPERATIONS] = type_scores.get(ProblemType.MATRIX_OPERATIONS, 0) + 2
        
        # Find the problem type with the highest score
        if not type_scores or max(type_scores.values()) == 0:
            return ProblemType.UNKNOWN, 0.0
        
        best_type = max(type_scores, key=type_scores.get)
        max_score = type_scores[best_type]
        
        # Calculate confidence
        max_possible_score = 8  # Reasonable maximum for a single problem type
        confidence = min(1.0, max_score / max_possible_score)
        
        return best_type, confidence
    
    def _extract_entities(self, text: str) -> Tuple[List[str], Optional[str], List[MathEntity]]:
        """
        Extract mathematical entities from the text.
        
        Args:
            text: The problem text to analyze
            
        Returns:
            Tuple of (variables, main_expression, detailed_entities)
        """
        entities = []
        
        # Extract variables
        variables = []
        for match in self.variable_pattern.finditer(text):
            var = match.group()
            # Filter out common English words and mathematical functions
            if (len(var) == 1 and var.isalpha() and 
                var.lower() not in {'a', 'i', 'o', 'e', 'is', 'of', 'to', 'in', 'it', 'be', 'or'}):
                variables.append(var)
                entities.append(MathEntity(
                    entity=var,
                    entity_type="variable",
                    position=match.start(),
                    context=text[max(0, match.start()-10):match.end()+10]
                ))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variables = []
        for var in variables:
            if var not in seen:
                seen.add(var)
                unique_variables.append(var)
        
        # Extract main mathematical expression
        main_expression = self._extract_main_expression(text)
        
        # Extract mathematical functions
        for match in self.function_pattern.finditer(text):
            func = match.group()
            entities.append(MathEntity(
                entity=func,
                entity_type="function",
                position=match.start(),
                context=text[max(0, match.start()-5):match.end()+5]
            ))
        
        # Extract mathematical constants
        for match in self.constant_pattern.finditer(text):
            const = match.group()
            entities.append(MathEntity(
                entity=const,
                entity_type="constant",
                position=match.start(),
                context=text[max(0, match.start()-5):match.end()+5]
            ))
        
        # Extract operators
        for match in self.operator_pattern.finditer(text):
            op = match.group()
            entities.append(MathEntity(
                entity=op,
                entity_type="operator",
                position=match.start(),
                context=text[max(0, match.start()-3):match.end()+3]
            ))
        
        return unique_variables, main_expression, entities
    
    def _extract_main_expression(self, text: str) -> Optional[str]:
        """
        Extract the main mathematical expression from natural language text.
        
        Args:
            text: The problem text
            
        Returns:
            The main mathematical expression or None if not found
        """
        # Remove common natural language phrases
        cleaned_text = text
        
        # Patterns to remove natural language context but preserve mathematical expressions
        natural_language_patterns = [
            r'\b(?:what is the (?:derivative|integral) of)\s+',
            r'\b(?:find the (?:derivative|integral) of)\s+',
            r'\b(?:what is|find|calculate|determine|solve for)\s+',
            r'\b(?:please|can you|could you|would you)\s*',
            r'\b(?:with respect to \w+|where)\b'
        ]
        
        for pattern in natural_language_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # Look for mathematical expressions
        for pattern_name, pattern in self.expression_patterns.items():
            match = pattern.search(cleaned_text)
            if match:
                expression = match.group().strip()
                if len(expression) > 2:  # Minimum meaningful expression length
                    return expression
        
        # If no specific pattern matches, try to extract anything that looks mathematical
        # Look for sequences with variables, numbers, and operators
        math_expression_pattern = re.compile(r'[a-zA-Z]\^?\d*[\+\-\*/\^]?[a-zA-Z0-9\^]*|[a-zA-Z]\d*')
        matches = math_expression_pattern.findall(cleaned_text)
        
        if matches:
            # Find the longest meaningful match
            best_match = max(matches, key=len)
            if len(best_match.strip()) > 0:
                return best_match.strip()
        
        # Last resort: look for simple variable patterns
        simple_var_pattern = re.compile(r'\b[a-zA-Z]\^?\d*\b')
        var_matches = simple_var_pattern.findall(cleaned_text)
        if var_matches:
            return max(var_matches, key=len)
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess the input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        cleaned = ' '.join(text.split())
        
        # Convert common LaTeX symbols to text
        latex_replacements = {
            r'\\frac\{([^}]+)\}\{([^}]+)\}': r'(\1)/(\2)',
            r'\\sqrt\{([^}]+)\}': r'sqrt(\1)',
            r'\\int': '∫',
            r'\\sum': '∑',
            r'\\lim': 'lim',
            r'\\pi': 'π',
            r'\\infty': '∞'
        }
        
        for latex, replacement in latex_replacements.items():
            cleaned = re.sub(latex, replacement, cleaned)
        
        return cleaned
    
    def parse(self, problem_text: str) -> ParsedProblem:
        """
        Parse a mathematical problem text and extract structured information.
        
        Args:
            problem_text: Raw text of the mathematical problem
            
        Returns:
            ParsedProblem object containing structured information
        """
        # Clean the input text
        cleaned_text = self._clean_text(problem_text)
        
        # Classify domain
        domain, domain_confidence = self._classify_domain(cleaned_text)
        
        # Identify problem type
        problem_type, type_confidence = self._identify_problem_type(cleaned_text)
        
        # Extract entities
        variables, expression, entities = self._extract_entities(cleaned_text)
        
        # Separate entities by type
        functions = [e.entity for e in entities if e.entity_type == "function"]
        constants = [e.entity for e in entities if e.entity_type == "constant"]
        operators = [e.entity for e in entities if e.entity_type == "operator"]
        
        # Create confidence object
        confidence = Confidence(
            domain=domain_confidence,
            problem_type=type_confidence,
            overall=(domain_confidence + type_confidence) / 2
        )
        
        # Create parsing notes
        parsing_notes = []
        if domain_confidence < 0.5:
            parsing_notes.append(f"Low confidence in domain classification ({domain_confidence:.2f})")
        if type_confidence < 0.5:
            parsing_notes.append(f"Low confidence in problem type identification ({type_confidence:.2f})")
        if not variables and not expression:
            parsing_notes.append("No clear mathematical variables or expressions found")
        
        # Create the parsed problem object
        parsed_problem = ParsedProblem(
            domain=domain,
            problem_type=problem_type,
            variables=variables,
            expression=expression,
            constants=list(set(constants)),  # Remove duplicates
            functions=list(set(functions)),  # Remove duplicates  
            operators=list(set(operators)),  # Remove duplicates
            entities=entities,
            original_text=problem_text,
            cleaned_text=cleaned_text,
            confidence=confidence,
            parsing_notes=parsing_notes,
            metadata={
                'parser_version': '1.0.0',
                'total_entities': len(entities),
                'text_length': len(problem_text),
                'cleaned_text_length': len(cleaned_text)
            }
        )
        
        return parsed_problem


# Convenience function for direct parsing
def parse(problem_text: str) -> ParsingResult:
    """
    Parse a mathematical problem text and return a result object.
    
    Args:
        problem_text: Raw text of the mathematical problem
        
    Returns:
        ParsingResult object containing the parsed problem or error information
    """
    start_time = time.time()
    
    try:
        parser = MathematicalProblemParser()
        parsed_problem = parser.parse(problem_text)
        
        end_time = time.time()
        parsing_time_ms = (end_time - start_time) * 1000
        
        return ParsingResult.success_result(parsed_problem, parsing_time_ms)
        
    except Exception as e:
        end_time = time.time()
        parsing_time_ms = (end_time - start_time) * 1000
        
        logger.error(f"Error parsing problem: {str(e)}")
        return ParsingResult.error_result(str(e), parsing_time_ms)


# Create a global parser instance for efficiency
_global_parser = None

def get_parser() -> MathematicalProblemParser:
    """Get a global parser instance (singleton pattern)."""
    global _global_parser
    if _global_parser is None:
        _global_parser = MathematicalProblemParser()
    return _global_parser