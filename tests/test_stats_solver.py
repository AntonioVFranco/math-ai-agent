"""
Comprehensive Test Suite for Statistics Solver Module

Tests all functionality of the statistics solver including descriptive statistics,
probability distributions, hypothesis testing, and random variate generation.

Author: Math AI Agent Team
Task ID: SOLVER-003
Coverage Target: 95%+
"""

import pytest
import sys
import os
import numpy as np
from typing import List, Dict, Any

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from solvers.stats_solver import (
        calculate_descriptive_stats, get_distribution_details,
        perform_t_test, generate_random_variates,
        perform_normality_test, calculate_correlation,
        StatisticsError, validate_data, validate_distribution
    )
    import scipy.stats as stats
    STATS_SOLVER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Stats solver not available: {e}")
    STATS_SOLVER_AVAILABLE = False


@pytest.mark.skipif(not STATS_SOLVER_AVAILABLE, reason="Stats solver not available")
class TestDescriptiveStatistics:
    """Test suite for descriptive statistics calculations."""
    
    def test_simple_list_known_values(self):
        """Test descriptive stats with known values [1, 2, 3, 4, 5]."""
        data = [1, 2, 3, 4, 5]
        result = calculate_descriptive_stats(data)
        
        assert result['success'] is True
        assert result['count'] == 5
        assert result['mean'] == 3.0
        assert result['median'] == 3.0
        assert abs(result['variance'] - 2.5) < 1e-10  # Sample variance
        assert abs(result['std_dev'] - np.sqrt(2.5)) < 1e-10
        assert result['min'] == 1.0
        assert result['max'] == 5.0
        assert result['range'] == 4.0
        assert result['sum'] == 15.0
    
    def test_single_value(self):
        """Test with single value."""
        data = [42]
        result = calculate_descriptive_stats(data)
        
        assert result['success'] is True
        assert result['count'] == 1
        assert result['mean'] == 42.0
        assert result['median'] == 42.0
        assert result['variance'] == 0.0  # Only one value
        assert result['std_dev'] == 0.0
        assert result['min'] == 42.0
        assert result['max'] == 42.0
        assert result['range'] == 0.0
    
    def test_identical_values(self):
        """Test with all identical values."""
        data = [5, 5, 5, 5, 5]
        result = calculate_descriptive_stats(data)
        
        assert result['success'] is True
        assert result['mean'] == 5.0
        assert result['median'] == 5.0
        assert result['variance'] == 0.0
        assert result['std_dev'] == 0.0
        assert result['mode'] == 5.0
        assert result['mode_count'] == 5
    
    def test_negative_values(self):
        """Test with negative values."""
        data = [-3, -1, 0, 1, 3]
        result = calculate_descriptive_stats(data)
        
        assert result['success'] is True
        assert result['mean'] == 0.0
        assert result['median'] == 0.0
        assert result['min'] == -3.0
        assert result['max'] == 3.0
    
    def test_floating_point_values(self):
        """Test with floating point values."""
        data = [1.1, 2.2, 3.3, 4.4, 5.5]
        result = calculate_descriptive_stats(data)
        
        assert result['success'] is True
        assert abs(result['mean'] - 3.3) < 1e-10
        assert abs(result['median'] - 3.3) < 1e-10
    
    def test_percentiles(self):
        """Test percentile calculations."""
        data = list(range(1, 101))  # 1 to 100
        result = calculate_descriptive_stats(data)
        
        assert result['success'] is True
        assert abs(result['percentiles']['25th'] - 25.75) < 1  # Approximate
        assert abs(result['percentiles']['50th'] - 50.5) < 1
        assert abs(result['percentiles']['75th'] - 75.25) < 1
    
    def test_empty_list_error(self):
        """Test error handling for empty list."""
        result = calculate_descriptive_stats([])
        
        assert result['success'] is False
        assert 'error' in result
        assert result['error_type'] == 'StatisticsError'
    
    def test_invalid_data_types(self):
        """Test error handling for invalid data types."""
        result = calculate_descriptive_stats([1, 2, "invalid", 4])
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_nan_values(self):
        """Test error handling for NaN values."""
        result = calculate_descriptive_stats([1, 2, float('nan'), 4])
        
        assert result['success'] is False
        assert 'finite numbers' in result['error']


@pytest.mark.skipif(not STATS_SOLVER_AVAILABLE, reason="Stats solver not available")
class TestDistributionDetails:
    """Test suite for probability distribution calculations."""
    
    def test_normal_distribution(self):
        """Test normal distribution calculations."""
        result = get_distribution_details('norm', {'loc': 0, 'scale': 1}, [0, 1, -1])
        
        assert result['success'] is True
        assert result['distribution'] == 'norm'
        assert 'pdf' in result
        assert 'cdf' in result
        assert len(result['pdf']) == 3
        assert len(result['cdf']) == 3
        
        # Test known values for standard normal
        assert abs(result['cdf'][0] - 0.5) < 1e-10  # CDF at 0 should be 0.5
        assert abs(result['pdf'][0] - stats.norm.pdf(0)) < 1e-10
    
    def test_binomial_distribution(self):
        """Test binomial distribution calculations."""
        result = get_distribution_details('binom', {'n': 10, 'p': 0.5}, [0, 5, 10])
        
        assert result['success'] is True
        assert result['distribution'] == 'binom'
        assert 'pmf' in result  # Should use PMF for discrete distribution
        assert 'cdf' in result
        
        # Test known values
        expected_pmf_5 = stats.binom.pmf(5, n=10, p=0.5)
        assert abs(result['pmf'][1] - expected_pmf_5) < 1e-10
    
    def test_poisson_distribution(self):
        """Test Poisson distribution calculations."""
        result = get_distribution_details('poisson', {'mu': 3}, [0, 3, 6])
        
        assert result['success'] is True
        assert result['distribution'] == 'poisson'
        assert 'pmf' in result
        assert result['distribution_stats']['mean'] == 3.0  # Poisson mean equals mu
    
    def test_invalid_distribution_name(self):
        """Test error handling for invalid distribution name."""
        result = get_distribution_details('invalid_dist', {}, [1, 2, 3])
        
        assert result['success'] is False
        assert 'Unsupported distribution' in result['error']
    
    def test_invalid_distribution_parameters(self):
        """Test error handling for invalid parameters."""
        # Negative scale for normal distribution
        result = get_distribution_details('norm', {'loc': 0, 'scale': -1}, [0])
        
        assert result['success'] is False
        assert 'scale' in result['error']
    
    def test_binomial_invalid_parameters(self):
        """Test binomial distribution parameter validation."""
        # Invalid probability
        result = get_distribution_details('binom', {'n': 10, 'p': 1.5}, [5])
        assert result['success'] is False
        
        # Missing parameters
        result = get_distribution_details('binom', {'n': 10}, [5])
        assert result['success'] is False
    
    def test_empty_points_list(self):
        """Test error handling for empty points list."""
        result = get_distribution_details('norm', {'loc': 0, 'scale': 1}, [])
        
        assert result['success'] is False
        assert 'at least 1 values' in result['error']


@pytest.mark.skipif(not STATS_SOLVER_AVAILABLE, reason="Stats solver not available")
class TestTTest:
    """Test suite for t-test functionality."""
    
    def test_basic_t_test(self):
        """Test basic t-test with known result."""
        # Simple case where we know the result
        sample_a = [1, 2, 3]
        sample_b = [4, 5, 6]
        
        result = perform_t_test(sample_a, sample_b, equal_var=True)
        
        assert result['success'] is True
        assert 't_statistic' in result
        assert 'p_value' in result
        assert 'degrees_freedom' in result
        assert result['degrees_freedom'] == 4  # n1 + n2 - 2 = 3 + 3 - 2 = 4
        assert result['equal_variances_assumed'] is True
        
        # The t-statistic should be negative (sample_a mean < sample_b mean)
        assert result['t_statistic'] < 0
    
    def test_welch_t_test(self):
        """Test Welch's t-test (unequal variances)."""
        sample_a = [1, 2, 3, 4, 5]
        sample_b = [10, 20, 30]
        
        result = perform_t_test(sample_a, sample_b, equal_var=False)
        
        assert result['success'] is True
        assert result['equal_variances_assumed'] is False
        # Welch's t-test should have different (typically non-integer) degrees of freedom
        assert isinstance(result['degrees_freedom'], float)
    
    def test_identical_samples(self):
        """Test t-test with identical samples."""
        sample_a = [1, 2, 3, 4, 5]
        sample_b = [1, 2, 3, 4, 5]
        
        result = perform_t_test(sample_a, sample_b)
        
        assert result['success'] is True
        assert abs(result['t_statistic']) < 1e-10  # Should be essentially 0
        assert result['p_value'] > 0.99  # Should be very close to 1
    
    def test_effect_size_calculation(self):
        """Test Cohen's d effect size calculation."""
        # Create samples with known effect size
        sample_a = [1, 2, 3, 4, 5]
        sample_b = [3, 4, 5, 6, 7]  # Mean difference of 2
        
        result = perform_t_test(sample_a, sample_b)
        
        assert result['success'] is True
        assert 'effect_size' in result
        assert 'cohens_d' in result['effect_size']
        assert 'interpretation' in result['effect_size']
        
        # Cohen's d should be positive and substantial
        assert result['effect_size']['cohens_d'] > 0
    
    def test_insufficient_data_error(self):
        """Test error handling for insufficient data."""
        result = perform_t_test([1], [2], equal_var=True)
        
        assert result['success'] is False
        assert 'at least 2 values' in result['error']
    
    def test_empty_samples_error(self):
        """Test error handling for empty samples."""
        result = perform_t_test([], [1, 2, 3])
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_zero_variance_samples(self):
        """Test handling of zero variance samples."""
        sample_a = [5, 5, 5]  # Zero variance
        sample_b = [5, 5, 5]  # Zero variance
        
        result = perform_t_test(sample_a, sample_b)
        
        assert result['success'] is False
        assert 'zero variance' in result['error']
    
    def test_sample_statistics_included(self):
        """Test that sample statistics are included in results."""
        sample_a = [1, 2, 3]
        sample_b = [4, 5, 6]
        
        result = perform_t_test(sample_a, sample_b)
        
        assert result['success'] is True
        assert 'sample_sizes' in result
        assert 'sample_means' in result
        assert 'sample_std_devs' in result
        assert result['sample_sizes']['sample_a'] == 3
        assert result['sample_sizes']['sample_b'] == 3
        assert result['sample_means']['sample_a'] == 2.0
        assert result['sample_means']['sample_b'] == 5.0


@pytest.mark.skipif(not STATS_SOLVER_AVAILABLE, reason="Stats solver not available")
class TestRandomVariates:
    """Test suite for random variate generation."""
    
    def test_normal_random_variates(self):
        """Test normal distribution random variate generation."""
        result = generate_random_variates('norm', {'loc': 0, 'scale': 1}, 100)
        
        assert result['success'] is True
        assert result['distribution'] == 'norm'
        assert result['sample_size'] == 100
        assert len(result['samples']) == 100
        assert 'sample_stats' in result
        assert 'theoretical_stats' in result
        
        # Sample mean should be close to theoretical mean for large samples
        if result['sample_stats'] and result['theoretical_stats']['mean'] is not None:
            mean_diff = abs(result['sample_stats']['mean'] - result['theoretical_stats']['mean'])
            assert mean_diff < 0.5  # Should be reasonably close
    
    def test_binomial_random_variates(self):
        """Test binomial distribution random variate generation."""
        result = generate_random_variates('binom', {'n': 10, 'p': 0.5}, 50)
        
        assert result['success'] is True
        assert result['distribution'] == 'binom'
        assert len(result['samples']) == 50
        
        # All samples should be integers between 0 and 10
        for sample in result['samples']:
            assert 0 <= sample <= 10
            assert float(sample).is_integer()
    
    def test_poisson_random_variates(self):
        """Test Poisson distribution random variate generation."""
        result = generate_random_variates('poisson', {'mu': 3}, 50)
        
        assert result['success'] is True
        assert result['distribution'] == 'poisson'
        
        # All samples should be non-negative integers
        for sample in result['samples']:
            assert sample >= 0
            assert float(sample).is_integer()
    
    def test_convergence_check(self):
        """Test convergence check for large samples."""
        result = generate_random_variates('norm', {'loc': 5, 'scale': 2}, 1000)
        
        assert result['success'] is True
        assert 'convergence_check' in result
        assert result['convergence_check'] is not None
        
        conv = result['convergence_check']
        assert 'sample_mean' in conv
        assert 'theoretical_mean' in conv
        assert 'difference' in conv
        assert 'standard_error' in conv
        
        # For large samples, should usually be within 2 standard errors
        assert conv['within_2_se'] is True or conv['difference'] < 0.5
    
    def test_invalid_size_error(self):
        """Test error handling for invalid sample size."""
        result = generate_random_variates('norm', {'loc': 0, 'scale': 1}, 0)
        
        assert result['success'] is False
        assert 'positive integer' in result['error']
    
    def test_large_size_error(self):
        """Test error handling for excessively large sample size."""
        result = generate_random_variates('norm', {'loc': 0, 'scale': 1}, 2000000)
        
        assert result['success'] is False
        assert 'too large' in result['error']
    
    def test_invalid_distribution_error(self):
        """Test error handling for invalid distribution."""
        result = generate_random_variates('invalid_dist', {}, 10)
        
        assert result['success'] is False
        assert 'Unsupported distribution' in result['error']
    
    def test_single_variate(self):
        """Test generation of single random variate."""
        result = generate_random_variates('norm', {'loc': 0, 'scale': 1}, 1)
        
        assert result['success'] is True
        assert len(result['samples']) == 1
        assert isinstance(result['samples'][0], float)


@pytest.mark.skipif(not STATS_SOLVER_AVAILABLE, reason="Stats solver not available")
class TestAdditionalFunctions:
    """Test suite for additional statistical functions."""
    
    def test_normality_test_shapiro(self):
        """Test Shapiro-Wilk normality test."""
        # Generate normal data
        np.random.seed(42)
        normal_data = list(np.random.normal(0, 1, 50))
        
        result = perform_normality_test(normal_data, 'shapiro')
        
        assert result['success'] is True
        assert result['test_name'] == 'Shapiro-Wilk'
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'is_normal_alpha_0.05' in result
    
    def test_normality_test_invalid_type(self):
        """Test error handling for invalid normality test type."""
        result = perform_normality_test([1, 2, 3, 4, 5], 'invalid_test')
        
        assert result['success'] is False
        assert 'Unsupported test type' in result['error']
    
    def test_correlation_pearson(self):
        """Test Pearson correlation calculation."""
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 4, 6, 8, 10]  # Perfect positive correlation
        
        result = calculate_correlation(x_data, y_data, 'pearson')
        
        assert result['success'] is True
        assert result['method'] == 'pearson'
        assert abs(result['correlation_coefficient'] - 1.0) < 1e-10
        assert result['strength'] == 'very strong'
        assert result['direction'] == 'positive'
    
    def test_correlation_negative(self):
        """Test negative correlation."""
        x_data = [1, 2, 3, 4, 5]
        y_data = [10, 8, 6, 4, 2]  # Perfect negative correlation
        
        result = calculate_correlation(x_data, y_data, 'pearson')
        
        assert result['success'] is True
        assert abs(result['correlation_coefficient'] - (-1.0)) < 1e-10
        assert result['direction'] == 'negative'
    
    def test_correlation_different_lengths(self):
        """Test error handling for different length arrays."""
        result = calculate_correlation([1, 2, 3], [1, 2], 'pearson')
        
        assert result['success'] is False
        assert 'same length' in result['error']
    
    def test_correlation_spearman(self):
        """Test Spearman correlation calculation."""
        x_data = [1, 2, 3, 4, 5]
        y_data = [1, 4, 9, 16, 25]  # Monotonic but not linear
        
        result = calculate_correlation(x_data, y_data, 'spearman')
        
        assert result['success'] is True
        assert result['method'] == 'spearman'
        assert result['correlation_coefficient'] == 1.0  # Perfect rank correlation


@pytest.mark.skipif(not STATS_SOLVER_AVAILABLE, reason="Stats solver not available")
class TestValidationFunctions:
    """Test suite for validation functions."""
    
    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        # Should not raise exception
        validate_data([1, 2, 3, 4, 5], min_length=3)
    
    def test_validate_data_too_short(self):
        """Test data validation with insufficient data."""
        with pytest.raises(StatisticsError) as exc_info:
            validate_data([1, 2], min_length=3)
        assert "at least 3 values" in str(exc_info.value)
    
    def test_validate_data_invalid_type(self):
        """Test data validation with non-list input."""
        with pytest.raises(StatisticsError) as exc_info:
            validate_data("not a list")
        assert "must be a list" in str(exc_info.value)
    
    def test_validate_distribution_valid(self):
        """Test distribution validation with valid parameters."""
        # Should not raise exception
        validate_distribution('norm', {'loc': 0, 'scale': 1})
    
    def test_validate_distribution_invalid_name(self):
        """Test distribution validation with invalid name."""
        with pytest.raises(StatisticsError) as exc_info:
            validate_distribution('invalid', {})
        assert "Unsupported distribution" in str(exc_info.value)
    
    def test_validate_distribution_invalid_params(self):
        """Test distribution validation with invalid parameters."""
        with pytest.raises(StatisticsError) as exc_info:
            validate_distribution('norm', {'scale': -1})
        assert "scale" in str(exc_info.value)


@pytest.mark.skipif(not STATS_SOLVER_AVAILABLE, reason="Stats solver not available")
class TestAcceptanceCriteria:
    """Test the specific acceptance criteria from the task specification."""
    
    def test_acceptance_criterion_1_descriptive_stats(self):
        """Test: calculate mean, median, and variance of [1, 2, 3, 4, 5]."""
        result = calculate_descriptive_stats([1, 2, 3, 4, 5])
        
        assert result['success'] is True
        assert result['mean'] == 3.0
        assert result['median'] == 3.0
        assert result['variance'] == 2.5  # Sample variance
        
        print("✓ Acceptance Criterion 1: Descriptive stats calculation works correctly")
    
    def test_acceptance_criterion_2_t_test(self):
        """Test: t-test between [1,2,3] and [4,5,6] returns valid results."""
        result = perform_t_test([1, 2, 3], [4, 5, 6])
        
        assert result['success'] is True
        assert 't_statistic' in result
        assert 'p_value' in result
        assert isinstance(result['t_statistic'], float)
        assert isinstance(result['p_value'], float)
        assert 0 <= result['p_value'] <= 1
        
        print("✓ Acceptance Criterion 2: T-test returns valid t-statistic and p-value")
    
    def test_acceptance_criterion_3_insufficient_t_test_data(self):
        """Test: structured error for t-test on [1] and [2]."""
        result = perform_t_test([1], [2])
        
        assert result['success'] is False
        assert 'error' in result
        assert 'error_type' in result
        assert result['error_type'] == 'StatisticsError'
        
        print("✓ Acceptance Criterion 3: Proper error handling for insufficient data")


@pytest.mark.skipif(not STATS_SOLVER_AVAILABLE, reason="Stats solver not available")
class TestIntegrationScenarios:
    """Test realistic statistical analysis scenarios."""
    
    def test_complete_data_analysis_workflow(self):
        """Test a complete data analysis workflow."""
        # Simulate experimental data
        control_group = [12.3, 11.8, 13.1, 12.7, 11.9, 12.4, 13.2, 12.1, 12.8, 13.0]
        treatment_group = [14.1, 13.8, 14.5, 13.9, 14.3, 14.0, 13.7, 14.2, 14.4, 13.6]
        
        # Step 1: Descriptive statistics
        control_stats = calculate_descriptive_stats(control_group)
        treatment_stats = calculate_descriptive_stats(treatment_group)
        
        assert control_stats['success'] is True
        assert treatment_stats['success'] is True
        
        # Step 2: Normality tests
        control_normality = perform_normality_test(control_group, 'shapiro')
        treatment_normality = perform_normality_test(treatment_group, 'shapiro')
        
        assert control_normality['success'] is True
        assert treatment_normality['success'] is True
        
        # Step 3: T-test comparison
        t_test_result = perform_t_test(control_group, treatment_group)
        
        assert t_test_result['success'] is True
        
        # Step 4: Effect size should be substantial
        assert t_test_result['effect_size']['cohens_d'] > 1.0  # Large effect
        
        print("✓ Complete data analysis workflow successful")
    
    def test_distribution_modeling_scenario(self):
        """Test distribution modeling and random generation."""
        # Model a normal distribution
        params = {'loc': 100, 'scale': 15}  # IQ scores
        test_points = [85, 100, 115, 130]  # Various IQ levels
        
        # Get distribution details
        dist_details = get_distribution_details('norm', params, test_points)
        
        assert dist_details['success'] is True
        assert len(dist_details['pdf']) == 4
        assert len(dist_details['cdf']) == 4
        
        # Generate random samples
        random_samples = generate_random_variates('norm', params, 1000)
        
        assert random_samples['success'] is True
        assert len(random_samples['samples']) == 1000
        
        # Check convergence
        sample_mean = random_samples['sample_stats']['mean']
        assert abs(sample_mean - 100) < 5  # Should be close to theoretical mean
        
        print("✓ Distribution modeling scenario successful")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])