Statistics Tutorial
===================

This tutorial demonstrates how to solve various statistics and probability problems using the Math AI Agent. The system provides comprehensive support for descriptive statistics, hypothesis testing, probability distributions, and statistical analysis.

Overview of Statistics Features
--------------------------------

The Math AI Agent supports:

* **Descriptive Statistics**: Mean, median, mode, variance, standard deviation, quartiles
* **Probability Distributions**: Normal, binomial, Poisson, exponential, and more
* **Hypothesis Testing**: t-tests, chi-square tests, ANOVA, normality tests
* **Correlation Analysis**: Pearson, Spearman, and other correlation measures
* **Statistical Visualizations**: Histograms, box plots, Q-Q plots, distribution curves
* **Random Sampling**: Generate samples from various distributions

Basic Descriptive Statistics
-----------------------------

Central Tendency and Spread
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**
Calculate basic statistics for a dataset.

**Input:**
.. code-block:: text

   Calculate the mean, median, mode, and standard deviation of: [2, 4, 4, 4, 5, 5, 7, 9]

**Expected Solution Process:**
1. **Mean**: Sum all values and divide by count: (2+4+4+4+5+5+7+9)/8 = 5.0
2. **Median**: Middle value of sorted data: (4+5)/2 = 4.5
3. **Mode**: Most frequent value: 4 (appears 3 times)
4. **Standard Deviation**: √(Σ(x-μ)²/n) = √(10/8) = 1.12
5. **Additional statistics**: Range, quartiles, skewness, kurtosis

**What You'll Learn:**
- When to use mean vs. median
- How outliers affect different measures
- Interpretation of standard deviation

**Advanced Example:**
.. code-block:: text

   Analyze the distribution of exam scores: [85, 92, 78, 96, 88, 76, 91, 84, 89, 93, 87, 82]

Quartiles and Box Plots
~~~~~~~~~~~~~~~~~~~~~~

**Problem:**
Understand data distribution using quartiles.

**Input:**
.. code-block:: text

   Find the five-number summary and create a box plot for: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

**Expected Solution Process:**
1. **Minimum**: 1
2. **Q1 (25th percentile)**: 5
3. **Median (Q2, 50th percentile)**: 10
4. **Q3 (75th percentile)**: 15
5. **Maximum**: 19
6. **IQR**: Q3 - Q1 = 10
7. **Outlier detection**: Values beyond Q1-1.5×IQR or Q3+1.5×IQR

**What the Visualization Shows:**
- Box spans from Q1 to Q3
- Line inside box shows median
- Whiskers extend to min/max (or outlier boundaries)
- Individual outliers plotted as points

Probability Distributions
-------------------------

Normal Distribution
~~~~~~~~~~~~~~~~~~~

**Problem:**
Work with the most important continuous distribution.

**Input:**
.. code-block:: text

   For a normal distribution with mean=100 and std=15, find P(X < 115)

**Expected Solution Process:**
1. Standardize: Z = (115-100)/15 = 1.0
2. Use standard normal table: P(Z < 1.0) = 0.8413
3. Interpret: About 84.13% of values are below 115
4. Show the distribution curve with shaded area

**Practical Applications:**
- IQ scores (μ=100, σ=15)
- Heights, weights, measurement errors
- Quality control limits

**Advanced Normal Problems:**
.. code-block:: text

   Find the 95th percentile of a normal distribution with μ=50, σ=10

.. code-block:: text

   What percentage of values fall within 2 standard deviations of the mean?

Binomial Distribution
~~~~~~~~~~~~~~~~~~~~

**Problem:**
Handle discrete probability with fixed number of trials.

**Input:**
.. code-block:: text

   A coin is flipped 10 times. What's the probability of getting exactly 6 heads?

**Expected Solution Process:**
1. Identify parameters: n=10, p=0.5, k=6
2. Apply binomial formula: P(X=k) = C(n,k) × p^k × (1-p)^(n-k)
3. Calculate: C(10,6) × (0.5)^6 × (0.5)^4 = 210 × (0.5)^10 = 0.2051
4. Show probability mass function
5. Calculate mean (np) and variance (np(1-p))

**Real-World Example:**
.. code-block:: text

   In a quality control process, 5% of items are defective. 
   In a batch of 20 items, what's the probability of finding 0, 1, or 2 defective items?

Poisson Distribution
~~~~~~~~~~~~~~~~~~~

**Problem:**
Model rare events occurring over time or space.

**Input:**
.. code-block:: text

   Emails arrive at an average rate of 3 per hour. 
   What's the probability of receiving exactly 5 emails in the next hour?

**Expected Solution Process:**
1. Identify parameter: λ = 3 (average rate)
2. Apply Poisson formula: P(X=k) = (λ^k × e^(-λ))/k!
3. Calculate: P(X=5) = (3^5 × e^(-3))/5! = (243 × 0.0498)/120 = 0.1008
4. Show probability mass function
5. Note that mean = variance = λ for Poisson

**Applications You'll Explore:**
- Customer arrivals at a service counter
- Defects in manufacturing
- Website visits per minute
- Radioactive decay events

Hypothesis Testing
------------------

One-Sample t-Test
~~~~~~~~~~~~~~~~~

**Problem:**
Test if a sample mean differs significantly from a hypothesized value.

**Input:**
.. code-block:: text

   Test if the mean of [23, 25, 28, 22, 24, 26, 27, 25] is significantly different from 25

**Expected Solution Process:**
1. **State hypotheses**: H₀: μ = 25 vs H₁: μ ≠ 25
2. **Calculate sample statistics**: x̄ = 25, s = 2.14, n = 8
3. **Compute test statistic**: t = (x̄ - μ₀)/(s/√n) = (25-25)/(2.14/√8) = 0
4. **Find p-value**: P(|t₇| > 0) = 1.0
5. **Conclusion**: Fail to reject H₀; no significant difference

**What You'll Understand:**
- Null and alternative hypotheses
- Type I and Type II errors
- p-values and significance levels
- When to use t-test vs z-test

Two-Sample t-Test
~~~~~~~~~~~~~~~~~

**Problem:**
Compare means of two independent groups.

**Input:**
.. code-block:: text

   Compare test scores: Group A = [85, 88, 92, 78, 91] and Group B = [82, 79, 84, 77, 80]

**Expected Solution Process:**
1. **State hypotheses**: H₀: μ₁ = μ₂ vs H₁: μ₁ ≠ μ₂
2. **Calculate sample statistics**: x̄₁ = 86.8, s₁ = 5.26, n₁ = 5; x̄₂ = 80.4, s₂ = 2.88, n₂ = 5
3. **Check equal variances assumption** (F-test or Levene's test)
4. **Compute pooled standard error**
5. **Calculate t-statistic and p-value**
6. **Draw conclusion** about difference in means

**Variants You'll Learn:**
- Welch's t-test (unequal variances)
- Paired t-test (dependent samples)
- One-tailed vs. two-tailed tests

Chi-Square Tests
~~~~~~~~~~~~~~~

**Problem:**
Test relationships between categorical variables.

**Input:**
.. code-block:: text

   Test independence between smoking status and lung disease:
   Observed data: [[10, 5], [15, 20]] (rows: smoker/non-smoker, cols: disease/no disease)

**Expected Solution Process:**
1. **State hypotheses**: H₀: smoking and disease are independent
2. **Calculate expected frequencies**: E_ij = (row_i × col_j)/total
3. **Compute chi-square statistic**: χ² = Σ((O_ij - E_ij)²/E_ij)
4. **Find critical value** with appropriate degrees of freedom
5. **Compare and conclude**

**Goodness of Fit Test:**
.. code-block:: text

   Test if a die is fair: observed frequencies [8, 12, 10, 15, 9, 6] for faces 1-6

Normality Testing
----------------

Shapiro-Wilk Test
~~~~~~~~~~~~~~~~~

**Problem:**
Test if data comes from a normal distribution.

**Input:**
.. code-block:: text

   Test normality of: [2.3, 1.8, 2.1, 2.5, 1.9, 2.2, 2.4, 2.0, 2.3, 2.1]

**Expected Solution Process:**
1. **Null hypothesis**: Data comes from normal distribution
2. **Calculate W statistic** using ranks and expected order statistics
3. **Find p-value** from Shapiro-Wilk table
4. **Interpret results**: p > 0.05 suggests normality

**Visual Checks:**
- Q-Q plot (quantile-quantile plot)
- Histogram with normal overlay
- Box plot for symmetry

**Alternative Tests:**
- Kolmogorov-Smirnov test
- Anderson-Darling test
- Jarque-Bera test

Correlation Analysis
--------------------

Pearson Correlation
~~~~~~~~~~~~~~~~~~

**Problem:**
Measure linear relationship between two continuous variables.

**Input:**
.. code-block:: text

   Calculate correlation between height and weight: 
   heights = [65, 67, 70, 72, 68] and weights = [120, 140, 160, 180, 150]

**Expected Solution Process:**
1. **Calculate means**: x̄ = 68.4, ȳ = 150
2. **Compute covariance**: Cov(X,Y) = Σ(x_i - x̄)(y_i - ȳ)/(n-1)
3. **Calculate standard deviations**: s_x, s_y
4. **Correlation coefficient**: r = Cov(X,Y)/(s_x × s_y)
5. **Test significance**: t = r√(n-2)/√(1-r²)

**What You'll Learn:**
- r = 1: perfect positive correlation
- r = -1: perfect negative correlation  
- r = 0: no linear correlation
- r² interpretation (explained variance)

Spearman Rank Correlation
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**
Measure monotonic relationship using ranks.

**Input:**
.. code-block:: text

   Calculate Spearman correlation for ordinal data:
   ranks1 = [1, 2, 3, 4, 5] and ranks2 = [2, 1, 4, 3, 5]

**Expected Solution Process:**
1. **Convert to ranks** (if not already ranked)
2. **Calculate rank differences**: d_i = rank1_i - rank2_i
3. **Apply formula**: ρ = 1 - (6Σd_i²)/(n(n²-1))
4. **Test significance**

**When to Use Spearman:**
- Ordinal data
- Non-linear but monotonic relationships
- Presence of outliers
- Non-normal distributions

Regression Analysis
-------------------

Simple Linear Regression
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**
Fit a line to predict one variable from another.

**Input:**
.. code-block:: text

   Fit a regression line for advertising spend vs sales:
   advertising = [1, 2, 3, 4, 5] and sales = [2, 4, 5, 7, 8]

**Expected Solution Process:**
1. **Calculate slope**: b₁ = Σ(x_i - x̄)(y_i - ȳ)/Σ(x_i - x̄)²
2. **Calculate intercept**: b₀ = ȳ - b₁x̄
3. **Form equation**: ŷ = b₀ + b₁x
4. **Calculate R²**: proportion of variance explained
5. **Test coefficient significance**
6. **Check residuals** for model assumptions

**Model Diagnostics:**
- Residual plots
- Normal probability plot of residuals
- Durbin-Watson test for autocorrelation

Multiple Regression
~~~~~~~~~~~~~~~~~~~

**Problem:**
Predict using multiple predictor variables.

**Input:**
.. code-block:: text

   Predict house price from size and age:
   prices = [200, 250, 180, 300, 220]
   sizes = [1500, 1800, 1200, 2200, 1600]  
   ages = [10, 5, 15, 2, 8]

**Matrix Approach:**
1. **Set up design matrix X** with intercept column
2. **Use normal equations**: β = (X'X)⁻¹X'y
3. **Calculate fitted values** and residuals
4. **Compute R², adjusted R²**
5. **F-test for overall significance**

Advanced Statistical Concepts
-----------------------------

Analysis of Variance (ANOVA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**
Compare means across multiple groups.

**Input:**
.. code-block:: text

   Test if three teaching methods have different effectiveness:
   Method A: [85, 88, 92, 78, 91]
   Method B: [82, 79, 84, 77, 80]  
   Method C: [90, 93, 89, 95, 87]

**Expected Solution Process:**
1. **State hypotheses**: H₀: μ₁ = μ₂ = μ₃
2. **Calculate group means** and overall mean
3. **Compute sum of squares**: SSB (between), SSW (within), SST (total)
4. **Form F-statistic**: F = MSB/MSW
5. **Compare to critical value**
6. **Post-hoc analysis** if significant

Confidence Intervals
~~~~~~~~~~~~~~~~~~~~

**Problem:**
Estimate population parameters with uncertainty quantification.

**Input:**
.. code-block:: text

   Find 95% confidence interval for population mean from sample: [12, 15, 18, 14, 16, 13, 17]

**Expected Solution Process:**
1. **Calculate sample statistics**: x̄, s, n
2. **Find t-critical value**: t_{α/2,n-1}
3. **Calculate margin of error**: ME = t × (s/√n)
4. **Form interval**: x̄ ± ME
5. **Interpret**: "We are 95% confident that μ is between..."

**Intervals for Different Parameters:**
- Population proportion
- Difference of means
- Ratio of variances

Statistical Visualizations
--------------------------

Distribution Plots
~~~~~~~~~~~~~~~~~

The system automatically generates appropriate visualizations:

**Histograms:**
- Show data distribution shape
- Overlay theoretical distributions
- Highlight unusual observations

**Box Plots:**
- Compare multiple groups
- Identify outliers visually
- Show distribution quartiles

**Q-Q Plots:**
- Check distributional assumptions
- Compare sample to theoretical quantiles
- Assess normality graphically

**Scatter Plots:**
- Show relationships between variables
- Add regression lines
- Identify influential points

Time Series and Sampling
------------------------

Random Sampling
~~~~~~~~~~~~~~~

**Problem:**
Generate samples from various distributions.

**Input:**
.. code-block:: text

   Generate 100 random values from a normal distribution with mean=50, std=10

**What You'll Get:**
- Generated sample values
- Verification that sample statistics match theoretical values
- Histogram of generated values
- Comparison to theoretical distribution

**Other Distributions:**
.. code-block:: text

   Generate samples from exponential distribution with rate=2

.. code-block:: text

   Simulate 50 Bernoulli trials with p=0.3

Central Limit Theorem Demonstration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:**
Show how sample means approach normality.

**Input:**
.. code-block:: text

   Demonstrate CLT: take 1000 samples of size 30 from uniform(0,1) and plot sample means

**Expected Demonstration:**
1. Generate many small samples from non-normal distribution
2. Calculate mean of each sample
3. Show distribution of sample means
4. Compare to theoretical normal distribution
5. Vary sample size to see effect

Interactive Examples
--------------------

Progressive Difficulty Levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Beginner:**

1. **Basic statistics:**
   ``Calculate mean and standard deviation of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]``

2. **Simple probability:**
   ``What's P(X ≤ 0) for standard normal distribution?``

**Intermediate:**

3. **Hypothesis testing:**
   ``Test if sample [20, 22, 19, 21, 23] has mean significantly different from 20``

4. **Correlation:**
   ``Find correlation between study hours [2, 4, 6, 8] and grades [70, 80, 85, 95]``

**Advanced:**

5. **Regression analysis:**
   ``Fit multiple regression predicting salary from education and experience``

6. **ANOVA:**
   ``Compare effectiveness of three different treatments using ANOVA``

Real-World Applications
-----------------------

Quality Control
~~~~~~~~~~~~~~

**Control Charts:**
- Monitor process mean and variability
- Detect when process goes out of control
- Set control limits using 3-sigma rule

**Acceptance Sampling:**
- Decide whether to accept or reject batches
- Balance Type I and Type II errors
- Optimize sample size and acceptance criteria

A/B Testing
~~~~~~~~~~~

**Web Analytics:**
- Compare conversion rates between website versions
- Calculate required sample size for desired power
- Account for multiple testing problems

**Medical Trials:**
- Compare treatment effectiveness
- Ensure ethical early stopping rules
- Handle missing data appropriately

Market Research
~~~~~~~~~~~~~~

**Survey Analysis:**
- Handle sampling bias and non-response
- Construct confidence intervals for proportions
- Test differences between demographic groups

**Customer Satisfaction:**
- Analyze Likert scale data
- Compare satisfaction across time periods
- Identify key drivers of satisfaction

Common Statistical Pitfalls
---------------------------

**Correlation vs. Causation:**
- High correlation doesn't imply causation
- Consider confounding variables
- Use experimental design for causal inference

**Multiple Testing:**
- Adjust significance levels for multiple comparisons
- Use Bonferroni or FDR corrections
- Be aware of data dredging

**Sample Size:**
- Too small: low power, unreliable results
- Calculate required sample size before collecting data
- Consider effect size, not just statistical significance

**Assumptions:**
- Check normality, independence, equal variances
- Use appropriate tests for your data type
- Consider non-parametric alternatives

Best Practices
--------------

**Data Exploration:**
- Always plot your data first
- Check for outliers and anomalies
- Understand the data generation process

**Model Selection:**
- Start with simple models
- Validate assumptions
- Use cross-validation for model comparison

**Interpretation:**
- Report confidence intervals, not just point estimates
- Consider practical significance vs. statistical significance
- Acknowledge limitations and assumptions

**Communication:**
- Use clear, non-technical language
- Provide context for statistical measures
- Include uncertainty in conclusions

Next Steps
----------

After mastering statistical fundamentals:

* **Machine Learning**: Apply statistics to predictive modeling
* **Time Series Analysis**: Handle temporal dependencies
* **Bayesian Statistics**: Learn probabilistic approaches
* **Advanced Topics**: Survival analysis, mixed models, non-parametrics

Statistics is the science of learning from data. The Math AI Agent helps you apply statistical methods correctly and interpret results meaningfully.

**Happy analyzing!** 📊📈