API Reference
=============

This section provides comprehensive documentation for all public APIs in the Math AI Agent system. The documentation is automatically generated from docstrings in the source code.

Core Engine
-----------

The core engine module contains the main orchestration logic for mathematical problem solving.

.. automodule:: src.core.engine
   :members:
   :undoc-members:
   :show-inheritance:

Data Models
-----------

Core data structures and type definitions used throughout the system.

.. automodule:: src.core.models
   :members:
   :undoc-members:
   :show-inheritance:

Problem Parser
--------------

Mathematical problem parsing and analysis functionality.

.. automodule:: src.core.parser
   :members:
   :undoc-members:
   :show-inheritance:

Verification System
-------------------

Multi-layer solution verification and validation.

.. automodule:: src.core.verifier
   :members:
   :undoc-members:
   :show-inheritance:

Solvers
-------

Domain-specific mathematical solvers for different problem types.

Linear Algebra Solver
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: src.solvers.linear_algebra_solver
   :members:
   :undoc-members:
   :show-inheritance:

Optimization Solver
~~~~~~~~~~~~~~~~~~~

.. automodule:: src.solvers.optimization_solver
   :members:
   :undoc-members:
   :show-inheritance:

Statistics Solver
~~~~~~~~~~~~~~~~~

.. automodule:: src.solvers.stats_solver
   :members:
   :undoc-members:
   :show-inheritance:

MCP Clients
-----------

Model Context Protocol clients for external mathematical computation services.

SymPy MCP Client
~~~~~~~~~~~~~~~~

.. automodule:: src.mcps.sympy_client
   :members:
   :undoc-members:
   :show-inheritance:

SymPy MCP Server
~~~~~~~~~~~~~~~~

.. automodule:: src.mcps.sympy_mcp
   :members:
   :undoc-members:
   :show-inheritance:

User Interface
--------------

Web interface and visualization components.

Gradio Application
~~~~~~~~~~~~~~~~~~

.. automodule:: src.interface.app
   :members:
   :undoc-members:
   :show-inheritance:

Visualization Engine
~~~~~~~~~~~~~~~~~~~~

.. automodule:: src.interface.visualizer
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

Common utility functions and helper modules.

.. automodule:: src.utils
   :members:
   :undoc-members:
   :show-inheritance:

Function Index
--------------

Core Functions
~~~~~~~~~~~~~~

**Main Entry Points:**

.. autofunction:: src.core.engine.execute_solution_pipeline

**Engine Methods:**

.. autofunction:: src.core.engine.MathAIEngine.__init__
.. autofunction:: src.core.engine.MathAIEngine.execute_solution_pipeline
.. autofunction:: src.core.engine.MathAIEngine.get_statistics

**Parser Functions:**

.. autofunction:: src.core.parser.get_parser
.. autofunction:: src.core.parser.MathematicalProblemParser.parse

**Verification Functions:**

.. autofunction:: src.core.verifier.get_verifier
.. autofunction:: src.core.verifier.MathematicalVerifier.verify_solution

Solver Functions
~~~~~~~~~~~~~~~~

**Linear Algebra:**

.. autofunction:: src.solvers.linear_algebra_solver.get_solver
.. autofunction:: src.solvers.linear_algebra_solver.LinearAlgebraSolver.compute_determinant
.. autofunction:: src.solvers.linear_algebra_solver.LinearAlgebraSolver.compute_inverse
.. autofunction:: src.solvers.linear_algebra_solver.LinearAlgebraSolver.lu_decomposition
.. autofunction:: src.solvers.linear_algebra_solver.LinearAlgebraSolver.qr_decomposition
.. autofunction:: src.solvers.linear_algebra_solver.LinearAlgebraSolver.eigen_decomposition
.. autofunction:: src.solvers.linear_algebra_solver.LinearAlgebraSolver.svd

**Optimization:**

.. autofunction:: src.solvers.optimization_solver.get_solver
.. autofunction:: src.solvers.optimization_solver.OptimizationSolver.gradient_descent
.. autofunction:: src.solvers.optimization_solver.OptimizationSolver.find_critical_points

**Statistics:**

.. autofunction:: src.solvers.stats_solver.calculate_descriptive_stats
.. autofunction:: src.solvers.stats_solver.perform_t_test
.. autofunction:: src.solvers.stats_solver.calculate_correlation
.. autofunction:: src.solvers.stats_solver.perform_normality_test

**SymPy MCP:**

.. autofunction:: src.mcps.sympy_client.SymPyMCPClient.solve_equation
.. autofunction:: src.mcps.sympy_client.SymPyMCPClient.simplify_expression
.. autofunction:: src.mcps.sympy_client.SymPyMCPClient.compute_derivative
.. autofunction:: src.mcps.sympy_client.SymPyMCPClient.compute_integral

Interface Functions
~~~~~~~~~~~~~~~~~~~

**Application:**

.. autofunction:: src.interface.app.create_interface
.. autofunction:: src.interface.app.launch_app
.. autofunction:: src.interface.app.process_mathematical_problem

**Visualization:**

.. autofunction:: src.interface.visualizer.get_visualizer
.. autofunction:: src.interface.visualizer.MathematicalVisualizer.plot_function_2d
.. autofunction:: src.interface.visualizer.MathematicalVisualizer.plot_function_3d
.. autofunction:: src.interface.visualizer.MathematicalVisualizer.plot_matrix_heatmap

Class Hierarchy
---------------

Core Classes
~~~~~~~~~~~~

.. inheritance-diagram:: src.core.engine.MathAIEngine src.core.parser.MathematicalProblemParser src.core.verifier.MathematicalVerifier
   :parts: 1

Data Model Classes
~~~~~~~~~~~~~~~~~~

.. inheritance-diagram:: src.core.models.ParsedProblem src.core.models.PipelineResult src.core.models.ExecutionStep src.core.models.VerificationResult
   :parts: 1

Solver Classes
~~~~~~~~~~~~~~

.. inheritance-diagram:: src.solvers.linear_algebra_solver.LinearAlgebraSolver src.solvers.optimization_solver.OptimizationSolver src.mcps.sympy_client.SymPyMCPClient
   :parts: 1

Interface Classes
~~~~~~~~~~~~~~~~

.. inheritance-diagram:: src.interface.visualizer.MathematicalVisualizer
   :parts: 1

Enumerations
------------

The system uses several enumerations to define standardized values:

**Mathematical Domains:**

.. autoclass:: src.core.models.MathDomain
   :members:
   :undoc-members:

**Problem Types:**

.. autoclass:: src.core.models.ProblemType
   :members:
   :undoc-members:

**Verification Methods:**

.. autoclass:: src.core.verifier.VerificationMethod
   :members:
   :undoc-members:

Type Definitions
----------------

Common type aliases and custom types used throughout the system:

.. autodata:: src.core.models.Matrix
   :annotation: List[List[float]]

.. autodata:: src.core.models.Vector
   :annotation: List[float]

.. autodata:: src.core.models.OptimizationHistory
   :annotation: List[Dict[str, Any]]

Constants
---------

System-wide constants and configuration values:

.. autodata:: src.core.models.DEFAULT_CONFIDENCE_THRESHOLD
   :annotation: float

.. autodata:: src.core.models.MAX_ITERATIONS
   :annotation: int

.. autodata:: src.core.models.TOLERANCE
   :annotation: float

Exception Classes
-----------------

Custom exceptions for error handling:

.. autoexception:: src.core.models.MathAIError
   :members:

.. autoexception:: src.core.models.ParsingError
   :members:

.. autoexception:: src.core.models.SolverError
   :members:

.. autoexception:: src.core.models.VerificationError
   :members:

Examples
--------

Basic Usage Examples
~~~~~~~~~~~~~~~~~~~~

**Solving a Simple Equation:**

.. code-block:: python

   from src.core.engine import execute_solution_pipeline

   # Solve a quadratic equation
   problem = "Solve x^2 - 5x + 6 = 0"
   api_key = "your-openai-api-key"
   
   result = execute_solution_pipeline(problem, api_key)
   
   if result.success:
       print(result.final_answer)
       if result.plot_object:
           result.plot_object.show()
   else:
       print(f"Error: {result.error_message}")

**Using Individual Solvers:**

.. code-block:: python

   from src.solvers.linear_algebra_solver import get_solver

   # Compute matrix determinant
   solver = get_solver()
   matrix = [[1, 2], [3, 4]]
   result = solver.compute_determinant(matrix)
   
   print(f"Determinant: {result['determinant']}")

**Custom Verification:**

.. code-block:: python

   from src.core.verifier import get_verifier
   from src.core.models import ParsedProblem, MathDomain, ProblemType

   # Create a parsed problem
   parsed = ParsedProblem(
       original_text="x^2 = 4",
       domain=MathDomain.ALGEBRA,
       problem_type=ProblemType.SOLVE_EQUATION,
       expression="x^2 - 4",
       variables=['x']
   )
   
   # Verify solution
   verifier = get_verifier()
   solution = {'x': [2, -2]}
   verification = verifier.verify_solution(parsed, solution)
   
   print(f"Verified: {verification.is_verified}")
   print(f"Confidence: {verification.confidence:.2%}")

Advanced Usage Examples
~~~~~~~~~~~~~~~~~~~~~~~

**Custom Optimization with Visualization:**

.. code-block:: python

   from src.solvers.optimization_solver import get_solver
   from src.mcps.sympy_client import SymPyMCPClient
   from src.interface.visualizer import get_visualizer

   # Set up optimization
   sympy_client = SymPyMCPClient()
   solver = get_solver(sympy_client)
   visualizer = get_visualizer()
   
   # Optimize function
   result = solver.gradient_descent(
       function_expr="(x-3)**2 + (y-2)**2",
       variables=['x', 'y'],
       initial_point=[0.0, 0.0],
       record_history=True
   )
   
   # Visualize optimization path
   if result.success and result.convergence_history:
       x_path = [entry['point'][0] for entry in result.convergence_history]
       y_path = [entry['point'][1] for entry in result.convergence_history]
       func_values = [entry['function_value'] for entry in result.convergence_history]
       
       plot = visualizer.plot_optimization_path_2d(
           "(x-3)**2 + (y-2)**2",
           ['x', 'y'],
           x_path,
           y_path,
           func_values,
           title="Optimization Path"
       )
       plot.show()

Version Information
-------------------

.. autodata:: src.__version__
   :annotation: str

For complete examples and tutorials, see the :doc:`tutorials/linear_algebra`, :doc:`tutorials/optimization`, and :doc:`tutorials/statistics` sections.