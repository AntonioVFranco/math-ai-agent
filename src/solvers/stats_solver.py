"""
Probability and Statistics Solver Module

This module provides comprehensive statistical analysis capabilities including
descriptive statistics, probability distributions, hypothesis testing, and
random variate generation using SciPy.stats and NumPy.

Author: MathBoardAI Agent Team
Task ID: SOLVER-003
"""

import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any, Union, Optional, Tuple
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class StatisticsError(Exception):
    """Custom exception for statistics-related errors."""
    pass


def validate_data(data: List[Union[int, float]], min_length: int = 1) -> None:
    """
    Validate input data for statistical operations.
    
    Args:
        data: List of numerical data
        min_length: Minimum required length
        
    Raises:
        StatisticsError: If data is invalid
    """
    if not isinstance(data, list):
        raise StatisticsError(f"Data must be a list, got {type(data)}")
    
    if len(data) < min_length:
        raise StatisticsError(f"Data must contain at least {min_length} values, got {len(data)}")
    
    if not all(isinstance(x, (int, float)) and not np.isnan(x) for x in data):
        raise StatisticsError("All data values must be finite numbers")


def validate_distribution(dist_name: str, params: Dict[str, float]) -> None:
    """
    Validate distribution name and parameters.
    
    Args:
        dist_name: Name of the distribution
        params: Distribution parameters
        
    Raises:
        StatisticsError: If distribution or parameters are invalid
    """
    # List of supported distributions
    supported_distributions = [
        'norm', 'binom', 'poisson', 'uniform', 'exponential', 'gamma',
        'beta', 'chi2', 't', 'f', 'lognorm', 'weibull', 'pareto'
    ]
    
    if dist_name not in supported_distributions:
        raise StatisticsError(f"Unsupported distribution: {dist_name}. "
                            f"Supported distributions: {supported_distributions}")
    
    if not isinstance(params, dict):
        raise StatisticsError("Parameters must be provided as a dictionary")
    
    # Validate specific distribution parameters
    if dist_name == 'norm':
        if 'scale' in params and params['scale'] <= 0:
            raise StatisticsError("Normal distribution scale (std) must be positive")
    
    elif dist_name == 'binom':
        if 'n' not in params or 'p' not in params:
            raise StatisticsError("Binomial distribution requires 'n' and 'p' parameters")
        if not (0 <= params['p'] <= 1):
            raise StatisticsError("Binomial probability 'p' must be between 0 and 1")
        if params['n'] < 0 or not isinstance(params['n'], int):
            raise StatisticsError("Binomial 'n' must be a non-negative integer")
    
    elif dist_name == 'poisson':
        if 'mu' not in params:
            raise StatisticsError("Poisson distribution requires 'mu' parameter")
        if params['mu'] <= 0:
            raise StatisticsError("Poisson 'mu' must be positive")


def calculate_descriptive_stats(data: List[Union[int, float]]) -> Dict[str, Any]:
    """
    Calculate comprehensive descriptive statistics for a dataset.
    
    Args:
        data: List of numerical values
        
    Returns:
        Dictionary containing descriptive statistics
        
    Examples:
        >>> calculate_descriptive_stats([1, 2, 3, 4, 5])
        {'mean': 3.0, 'median': 3.0, 'variance': 2.0, 'std_dev': 1.414...}
    """
    try:
        # Validate input
        validate_data(data, min_length=1)
        
        # Convert to numpy array for calculations
        np_data = np.array(data, dtype=float)
        
        # Calculate basic statistics
        result = {
            'count': len(data),
            'mean': float(np.mean(np_data)),
            'median': float(np.median(np_data)),
            'mode': None,  # Will calculate separately
            'variance': float(np.var(np_data, ddof=1)) if len(data) > 1 else 0.0,
            'std_dev': float(np.std(np_data, ddof=1)) if len(data) > 1 else 0.0,
            'min': float(np.min(np_data)),
            'max': float(np.max(np_data)),
            'range': float(np.max(np_data) - np.min(np_data)),
            'q1': float(np.percentile(np_data, 25)),
            'q3': float(np.percentile(np_data, 75)),
            'iqr': float(np.percentile(np_data, 75) - np.percentile(np_data, 25)),
            'skewness': float(stats.skew(np_data)),
            'kurtosis': float(stats.kurtosis(np_data)),
            'sum': float(np.sum(np_data))
        }
        
        # Calculate mode (most frequent value)
        try:
            mode_result = stats.mode(np_data, keepdims=True)
            if len(mode_result.mode) > 0:
                result['mode'] = float(mode_result.mode[0])
                result['mode_count'] = int(mode_result.count[0])
        except Exception:
            result['mode'] = None
            result['mode_count'] = 0
        
        # Additional percentiles
        result['percentiles'] = {
            '10th': float(np.percentile(np_data, 10)),
            '25th': result['q1'],
            '50th': result['median'],
            '75th': result['q3'],
            '90th': float(np.percentile(np_data, 90)),
            '95th': float(np.percentile(np_data, 95)),
            '99th': float(np.percentile(np_data, 99))
        }
        
        # Add metadata
        result['success'] = True
        result['message'] = f"Descriptive statistics calculated for {len(data)} data points"
        
        logger.info(f"Calculated descriptive statistics for {len(data)} data points")
        return result
        
    except StatisticsError as e:
        logger.error(f"Statistics error in descriptive stats: {e}")
        return {
            'success': False,
            'error': str(e),
            'error_type': 'StatisticsError'
        }
    except Exception as e:
        logger.error(f"Unexpected error in descriptive stats: {e}")
        return {
            'success': False,
            'error': f"Unexpected error: {str(e)}",
            'error_type': type(e).__name__
        }


def get_distribution_details(dist_name: str, params: Dict[str, float], 
                           points: List[Union[int, float]]) -> Dict[str, Any]:
    """
    Calculate PDF/PMF and CDF for a given distribution at specific points.
    
    Args:
        dist_name: Name of the distribution (e.g., 'norm', 'binom', 'poisson')
        params: Distribution parameters as a dictionary
        points: List of points to evaluate
        
    Returns:
        Dictionary containing PDF/PMF, CDF values, and distribution info
        
    Examples:
        >>> get_distribution_details('norm', {'loc': 0, 'scale': 1}, [0, 1, 2])
        {'pdf': [0.398, 0.242, 0.054], 'cdf': [0.5, 0.841, 0.977], ...}
    """
    try:
        # Validate inputs
        validate_distribution(dist_name, params)
        validate_data(points, min_length=1)
        
        # Get the distribution object
        dist = getattr(stats, dist_name)
        
        # Convert points to numpy array
        points_array = np.array(points, dtype=float)
        
        # Calculate PDF/PMF and CDF
        try:
            if hasattr(dist, 'pmf'):
                # Discrete distribution - use PMF
                pdf_values = dist.pmf(points_array, **params)
                prob_type = 'pmf'
            else:
                # Continuous distribution - use PDF
                pdf_values = dist.pdf(points_array, **params)
                prob_type = 'pdf'
            
            cdf_values = dist.cdf(points_array, **params)
            
        except Exception as e:
            raise StatisticsError(f"Error calculating distribution values: {str(e)}")
        
        # Get distribution statistics
        try:
            mean, var = dist.stats(**params, moments='mv')
            mean = float(mean) if not np.isnan(mean) else None
            var = float(var) if not np.isnan(var) else None
        except Exception:
            mean, var = None, None
        
        result = {
            'distribution': dist_name,
            'parameters': params,
            'points': points,
            prob_type: [float(x) for x in pdf_values],
            'cdf': [float(x) for x in cdf_values],
            'distribution_stats': {
                'mean': mean,
                'variance': var,
                'std_dev': float(np.sqrt(var)) if var is not None and var >= 0 else None
            },
            'success': True,
            'message': f"Distribution details calculated for {dist_name} at {len(points)} points"
        }
        
        logger.info(f"Calculated {dist_name} distribution details for {len(points)} points")
        return result
        
    except StatisticsError as e:
        logger.error(f"Statistics error in distribution details: {e}")
        return {
            'success': False,
            'error': str(e),
            'error_type': 'StatisticsError'
        }
    except Exception as e:
        logger.error(f"Unexpected error in distribution details: {e}")
        return {
            'success': False,
            'error': f"Unexpected error: {str(e)}",
            'error_type': type(e).__name__
        }


def perform_t_test(sample_a: List[Union[int, float]], 
                   sample_b: List[Union[int, float]], 
                   equal_var: bool = True) -> Dict[str, Any]:
    """
    Perform an independent two-sample t-test.
    
    Args:
        sample_a: First sample data
        sample_b: Second sample data
        equal_var: Whether to assume equal variances (Welch's t-test if False)
        
    Returns:
        Dictionary containing t-statistic, p-value, and test details
        
    Examples:
        >>> perform_t_test([1, 2, 3], [4, 5, 6], equal_var=True)
        {'t_statistic': -3.674, 'p_value': 0.021, 'degrees_freedom': 4, ...}
    """
    try:
        # Validate inputs
        validate_data(sample_a, min_length=2)
        validate_data(sample_b, min_length=2)
        
        # Convert to numpy arrays
        a = np.array(sample_a, dtype=float)
        b = np.array(sample_b, dtype=float)
        
        # Check for sufficient variation
        if np.var(a) == 0 and np.var(b) == 0:
            raise StatisticsError("Both samples have zero variance - t-test not meaningful")
        
        # Perform t-test
        try:
            t_stat, p_value = stats.ttest_ind(a, b, equal_var=equal_var)
        except Exception as e:
            raise StatisticsError(f"Error performing t-test: {str(e)}")
        
        # Calculate degrees of freedom
        if equal_var:
            df = len(a) + len(b) - 2
        else:
            # Welch's t-test degrees of freedom
            var_a = np.var(a, ddof=1)
            var_b = np.var(b, ddof=1)
            n_a, n_b = len(a), len(b)
            
            numerator = (var_a/n_a + var_b/n_b)**2
            denominator = (var_a/n_a)**2/(n_a-1) + (var_b/n_b)**2/(n_b-1)
            df = numerator / denominator
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1)) / (len(a)+len(b)-2))
        cohens_d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_size_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_size_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_size_interpretation = "medium"
        else:
            effect_size_interpretation = "large"
        
        # Interpret p-value
        alpha_levels = [0.001, 0.01, 0.05, 0.1]
        significance_level = "not significant"
        for alpha in alpha_levels:
            if p_value < alpha:
                significance_level = f"significant at α = {alpha}"
                break
        
        result = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'degrees_freedom': float(df),
            'equal_variances_assumed': equal_var,
            'sample_sizes': {'sample_a': len(a), 'sample_b': len(b)},
            'sample_means': {'sample_a': float(np.mean(a)), 'sample_b': float(np.mean(b))},
            'sample_std_devs': {'sample_a': float(np.std(a, ddof=1)), 'sample_b': float(np.std(b, ddof=1))},
            'effect_size': {
                'cohens_d': float(cohens_d),
                'interpretation': effect_size_interpretation
            },
            'interpretation': {
                'significance_level': significance_level,
                'null_hypothesis': "The means of the two populations are equal",
                'alternative_hypothesis': "The means of the two populations are not equal"
            },
            'success': True,
            'message': f"Two-sample t-test completed (equal_var={equal_var})"
        }
        
        logger.info(f"Performed t-test: t={t_stat:.4f}, p={p_value:.4f}")
        return result
        
    except StatisticsError as e:
        logger.error(f"Statistics error in t-test: {e}")
        return {
            'success': False,
            'error': str(e),
            'error_type': 'StatisticsError'
        }
    except Exception as e:
        logger.error(f"Unexpected error in t-test: {e}")
        return {
            'success': False,
            'error': f"Unexpected error: {str(e)}",
            'error_type': type(e).__name__
        }


def generate_random_variates(dist_name: str, params: Dict[str, float], 
                           size: int) -> Dict[str, Any]:
    """
    Generate random variates from a specified distribution.
    
    Args:
        dist_name: Name of the distribution
        params: Distribution parameters
        size: Number of random variates to generate
        
    Returns:
        Dictionary containing generated samples and statistics
        
    Examples:
        >>> generate_random_variates('norm', {'loc': 0, 'scale': 1}, 100)
        {'samples': [...], 'sample_stats': {...}, 'theoretical_stats': {...}}
    """
    try:
        # Validate inputs
        validate_distribution(dist_name, params)
        
        if not isinstance(size, int) or size <= 0:
            raise StatisticsError("Size must be a positive integer")
        
        if size > 1000000:
            raise StatisticsError("Sample size too large (max: 1,000,000)")
        
        # Get the distribution object
        dist = getattr(stats, dist_name)
        
        # Generate random variates
        try:
            samples = dist.rvs(size=size, **params)
            
            # Ensure we have a list of floats
            if np.isscalar(samples):
                samples = [float(samples)]
            else:
                samples = [float(x) for x in samples]
                
        except Exception as e:
            raise StatisticsError(f"Error generating random variates: {str(e)}")
        
        # Calculate sample statistics
        sample_stats = calculate_descriptive_stats(samples)
        
        # Get theoretical distribution statistics
        try:
            theoretical_mean, theoretical_var = dist.stats(**params, moments='mv')
            theoretical_mean = float(theoretical_mean) if not np.isnan(theoretical_mean) else None
            theoretical_var = float(theoretical_var) if not np.isnan(theoretical_var) else None
            theoretical_std = float(np.sqrt(theoretical_var)) if theoretical_var is not None and theoretical_var >= 0 else None
        except Exception:
            theoretical_mean, theoretical_var, theoretical_std = None, None, None
        
        result = {
            'distribution': dist_name,
            'parameters': params,
            'sample_size': size,
            'samples': samples,
            'sample_stats': {
                'mean': sample_stats.get('mean'),
                'std_dev': sample_stats.get('std_dev'),
                'min': sample_stats.get('min'),
                'max': sample_stats.get('max'),
                'median': sample_stats.get('median')
            } if sample_stats.get('success') else None,
            'theoretical_stats': {
                'mean': theoretical_mean,
                'variance': theoretical_var,
                'std_dev': theoretical_std
            },
            'convergence_check': None,  # Will add if sample size is large enough
            'success': True,
            'message': f"Generated {size} random variates from {dist_name} distribution"
        }
        
        # Add convergence check for large samples
        if size >= 30 and sample_stats.get('success') and theoretical_mean is not None:
            sample_mean = sample_stats.get('mean')
            mean_diff = abs(sample_mean - theoretical_mean)
            expected_std_error = theoretical_std / np.sqrt(size) if theoretical_std else None
            
            result['convergence_check'] = {
                'sample_mean': sample_mean,
                'theoretical_mean': theoretical_mean,
                'difference': mean_diff,
                'standard_error': expected_std_error,
                'within_2_se': mean_diff < (2 * expected_std_error) if expected_std_error else None
            }
        
        logger.info(f"Generated {size} random variates from {dist_name} distribution")
        return result
        
    except StatisticsError as e:
        logger.error(f"Statistics error in random variate generation: {e}")
        return {
            'success': False,
            'error': str(e),
            'error_type': 'StatisticsError'
        }
    except Exception as e:
        logger.error(f"Unexpected error in random variate generation: {e}")
        return {
            'success': False,
            'error': f"Unexpected error: {str(e)}",
            'error_type': type(e).__name__
        }


# Additional utility functions for enhanced statistical analysis

def perform_normality_test(data: List[Union[int, float]], 
                         test_type: str = 'shapiro') -> Dict[str, Any]:
    """
    Perform normality tests on data.
    
    Args:
        data: List of numerical values
        test_type: Type of test ('shapiro', 'kstest', 'jarque_bera')
        
    Returns:
        Dictionary containing test results
    """
    try:
        validate_data(data, min_length=3)
        
        np_data = np.array(data, dtype=float)
        
        if test_type == 'shapiro':
            if len(data) > 5000:
                raise StatisticsError("Shapiro-Wilk test limited to 5000 samples")
            stat, p_value = stats.shapiro(np_data)
            test_name = "Shapiro-Wilk"
            
        elif test_type == 'kstest':
            stat, p_value = stats.kstest(np_data, 'norm', args=(np.mean(np_data), np.std(np_data, ddof=1)))
            test_name = "Kolmogorov-Smirnov"
            
        elif test_type == 'jarque_bera':
            stat, p_value = stats.jarque_bera(np_data)
            test_name = "Jarque-Bera"
            
        else:
            raise StatisticsError(f"Unsupported test type: {test_type}")
        
        # Interpret results
        is_normal_005 = p_value > 0.05
        is_normal_001 = p_value > 0.01
        
        return {
            'test_name': test_name,
            'statistic': float(stat),
            'p_value': float(p_value),
            'is_normal_alpha_0.05': is_normal_005,
            'is_normal_alpha_0.01': is_normal_001,
            'interpretation': f"Data {'appears' if is_normal_005 else 'does not appear'} normally distributed (α = 0.05)",
            'success': True
        }
        
    except StatisticsError as e:
        return {'success': False, 'error': str(e), 'error_type': 'StatisticsError'}
    except Exception as e:
        return {'success': False, 'error': f"Unexpected error: {str(e)}", 'error_type': type(e).__name__}


def calculate_correlation(x_data: List[Union[int, float]], 
                        y_data: List[Union[int, float]], 
                        method: str = 'pearson') -> Dict[str, Any]:
    """
    Calculate correlation between two variables.
    
    Args:
        x_data: First variable data
        y_data: Second variable data  
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Dictionary containing correlation coefficient and p-value
    """
    try:
        validate_data(x_data, min_length=2)
        validate_data(y_data, min_length=2)
        
        if len(x_data) != len(y_data):
            raise StatisticsError("x_data and y_data must have the same length")
        
        x = np.array(x_data, dtype=float)
        y = np.array(y_data, dtype=float)
        
        if method == 'pearson':
            corr, p_value = stats.pearsonr(x, y)
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(x, y)
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(x, y)
        else:
            raise StatisticsError(f"Unsupported correlation method: {method}")
        
        # Interpret correlation strength
        abs_corr = abs(corr)
        if abs_corr < 0.1:
            strength = "negligible"
        elif abs_corr < 0.3:
            strength = "weak"
        elif abs_corr < 0.5:
            strength = "moderate"
        elif abs_corr < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        return {
            'method': method,
            'correlation_coefficient': float(corr),
            'p_value': float(p_value),
            'strength': strength,
            'direction': 'positive' if corr > 0 else 'negative' if corr < 0 else 'none',
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
            'sample_size': len(x_data),
            'success': True
        }
        
    except StatisticsError as e:
        return {'success': False, 'error': str(e), 'error_type': 'StatisticsError'}
    except Exception as e:
        return {'success': False, 'error': f"Unexpected error: {str(e)}", 'error_type': type(e).__name__}