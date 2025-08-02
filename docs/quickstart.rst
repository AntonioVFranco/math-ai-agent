Quick Start Guide
=================

Welcome to Math AI Agent! This guide will help you get started with solving mathematical problems in just a few minutes.

What is Math AI Agent?
----------------------

Math AI Agent is a sophisticated AI-powered system that solves complex mathematical problems across multiple domains including:

* **Algebra**: Equation solving, polynomial operations
* **Calculus**: Derivatives, integrals, limits
* **Linear Algebra**: Matrix operations, eigenvalues, decompositions
* **Optimization**: Gradient descent, critical points
* **Statistics**: Descriptive stats, hypothesis testing, distributions

The system combines the power of **OpenAI GPT-4o** for natural language understanding with **SymPy** for precise symbolic mathematics, providing step-by-step solutions with verification.

Getting Started
---------------

Step 1: Launch the Application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to use Math AI Agent is through the web interface:

**Using Docker (Recommended):**

.. code-block:: bash

   # Clone the repository
   git clone <repository-url>
   cd math-ai-agent

   # Launch with Docker Compose
   docker-compose up

   # Open your browser to http://localhost:7860

**Local Installation:**

.. code-block:: bash

   # Install Python dependencies
   pip install -r requirements.txt

   # Run the application
   python src/interface/app.py

   # Open your browser to http://localhost:7860

Step 2: Configure Your API Key
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Get an OpenAI API Key:**
   
   * Visit `OpenAI Platform <https://platform.openai.com/api-keys>`_
   * Sign in to your account (or create one)
   * Click "Create new secret key"
   * Copy the key (it starts with ``sk-``)

2. **Enter Your API Key:**
   
   * In the web interface, click on "Settings / Configurações"
   * Paste your API key in the "OpenAI API Key" field
   * Your key is only used for this session and never stored

.. important::
   Make sure your OpenAI API key has access to GPT-4o and sufficient credits for usage.

Step 3: Solve Your First Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's start with a simple example:

1. **Enter a Mathematical Problem:**

   In the text area, type:
   
   .. code-block:: text
   
      Solve the equation: x² - 5x + 6 = 0

2. **Click "Resolver" (Solve):**

   The system will:
   
   * Parse your problem
   * Create a solution plan
   * Execute calculations using SymPy
   * Verify the results
   * Present a complete solution

3. **Review the Results:**

   You'll see:
   
   * **Step-by-step solution** with mathematical reasoning
   * **Final answer** clearly highlighted
   * **Verification status** (✅ if verified)
   * **Visualizations** (when relevant)

Example Problems to Try
-----------------------

Here are some examples to help you explore different mathematical domains:

Basic Algebra
~~~~~~~~~~~~~

.. code-block:: text

   Solve for x: 2x + 5 = 13

.. code-block:: text

   Factor the polynomial: x² + 7x + 12

.. code-block:: text

   Simplify: (x² - 4)/(x - 2)

Calculus
~~~~~~~~

.. code-block:: text

   Find the derivative of f(x) = x³ + 2x² - 5x + 1

.. code-block:: text

   Compute the integral: ∫(2x + 3)dx

.. code-block:: text

   Evaluate: ∫₀^π sin(x)dx

Linear Algebra
~~~~~~~~~~~~~~

.. code-block:: text

   Calculate the determinant of the matrix: [[1, 2], [3, 4]]

.. code-block:: text

   Find the inverse of: [[2, 1], [1, 1]]

.. code-block:: text

   Compute the eigenvalues of: [[3, 1], [0, 2]]

Optimization
~~~~~~~~~~~~

.. code-block:: text

   Find the minimum of f(x) = x² - 4x + 5

.. code-block:: text

   Use gradient descent to minimize: (x-3)² + (y-2)²

Statistics
~~~~~~~~~~

.. code-block:: text

   Calculate the mean, median, and standard deviation of: [2, 4, 4, 4, 5, 5, 7, 9]

.. code-block:: text

   Perform a t-test comparing two samples: [1,2,3,4,5] and [2,3,4,5,6]

Advanced Features
-----------------

LaTeX Support
~~~~~~~~~~~~~

You can use LaTeX notation in your problems:

.. code-block:: text

   Solve: $\frac{d}{dx}[\sin(x^2)] = ?$

.. code-block:: text

   Compute: $\int_0^{\infty} e^{-x^2} dx$

The system will properly interpret and render LaTeX expressions.

Natural Language Input
~~~~~~~~~~~~~~~~~~~~~~

The system understands problems described in plain English:

.. code-block:: text

   What is the area under the curve y = x² from x = 0 to x = 2?

.. code-block:: text

   If I have a circle with radius 5, what is its area and circumference?

.. code-block:: text

   Find where the function f(x) = x³ - 6x² + 9x - 4 has its maximum and minimum points.

Visualizations
~~~~~~~~~~~~~~

The system automatically generates relevant visualizations:

* **Function plots** for calculus problems
* **Matrix heatmaps** for linear algebra
* **Optimization paths** for gradient descent
* **Statistical distributions** for probability problems

Understanding the Output
------------------------

Solution Structure
~~~~~~~~~~~~~~~~~~

Each solution typically includes:

1. **Problem Restatement**: Clarification of what's being solved
2. **Solution Approach**: High-level strategy explanation
3. **Step-by-Step Solution**: Detailed mathematical steps
4. **Final Answer**: Clearly highlighted result
5. **Verification Status**: Confidence in the solution accuracy

Verification Levels
~~~~~~~~~~~~~~~~~~~

Solutions are automatically verified with confidence levels:

* **✅ High Confidence (90%+)**: Multiple verification methods passed
* **✅ Medium Confidence (70-90%)**: Some verification methods passed  
* **⚠️ Low Confidence (50-70%)**: Limited verification possible
* **❌ Not Verified (<50%)**: Unable to verify or verification failed

Processing Information
~~~~~~~~~~~~~~~~~~~~~~

At the bottom of each solution, you'll see:

* **⏱️ Processing Time**: How long the solution took
* **🤖 OpenAI API Calls**: Number of GPT-4o requests made
* **🧮 SymPy Calculations**: Number of symbolic computations
* **🎯 Parsing Confidence**: How well the problem was understood

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**"Invalid API Key" Error:**
   * Check that your key starts with ``sk-``
   * Verify the key is active on OpenAI's platform
   * Ensure you have sufficient credits

**"Problem parsing failed" Error:**
   * Try rephrasing your problem more clearly
   * Use standard mathematical notation
   * Break complex problems into smaller parts

**Slow Response Times:**
   * Complex problems take longer to solve
   * Check your internet connection
   * Large problems may require more processing time

**No Visualization Displayed:**
   * Not all problems have visualizations
   * Check the "Visualização" tab
   * Some plots may take time to render

Getting Better Results
~~~~~~~~~~~~~~~~~~~~~~

**Be Specific:**
   * Clear problem statements get better results
   * Include all necessary information
   * Specify what you want to find

**Use Standard Notation:**
   * Standard mathematical symbols work best
   * LaTeX is supported for complex expressions
   * Avoid ambiguous notation

**Break Down Complex Problems:**
   * Split multi-part problems into steps
   * Solve prerequisites first
   * Build up to complex solutions

Tips for Success
-----------------

Effective Problem Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Start Simple**: Begin with basic problems to understand the system
* **Be Precise**: Use exact mathematical language when possible
* **Provide Context**: Include units, domains, and constraints when relevant
* **Check Input**: Review your problem before submitting

Making the Most of Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Explore Domains**: Try problems from different mathematical areas
* **Use Visualizations**: Check the visualization tab for graphs and plots
* **Read Verification**: Pay attention to verification status and warnings
* **Learn from Steps**: Study the step-by-step solutions to learn methods

Example Workflow
~~~~~~~~~~~~~~~~

Here's a typical workflow for solving a calculus problem:

1. **Enter Problem**: "Find the critical points of f(x) = x³ - 6x² + 9x"

2. **Review Parsing**: Check that the system understood your function correctly

3. **Examine Solution**: Read through the derivative calculation and critical point finding

4. **Check Verification**: Ensure the solution is verified (✅)

5. **View Visualization**: Look at the function plot showing critical points

6. **Understand Results**: Note that critical points are at x = 1 and x = 3

Next Steps
----------

Now that you're familiar with the basics:

* **Explore Tutorials**: Check out domain-specific tutorials for detailed examples
* **Try Complex Problems**: Experiment with multi-step mathematical problems
* **Read Documentation**: Dive deeper into the system architecture and features
* **Join the Community**: Share your experiences and get help from other users

**Happy problem solving!** 🧮✨

.. note::
   This system is designed to be a powerful mathematical assistant. While it provides highly accurate solutions with verification, always double-check critical calculations for important applications.

Need Help?
----------

* **FAQ**: Check the :doc:`faq` for common questions
* **Tutorials**: Explore :doc:`tutorials/linear_algebra`, :doc:`tutorials/optimization`, and :doc:`tutorials/statistics`
* **Technical Docs**: See the :doc:`developer_guide` for advanced usage
* **Issues**: Report problems on GitHub