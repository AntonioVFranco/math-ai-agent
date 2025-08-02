Developer Guide
===============

This guide covers everything you need to know to set up, develop, and contribute to the Math AI Agent project.

Prerequisites
-------------

Before you begin, ensure you have the following installed:

* **Python 3.8 or higher**
* **Docker** (recommended for consistent development environment)
* **Git** for version control
* **OpenAI API Key** with GPT-4o access

Setup Instructions
------------------

Docker Setup (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project includes a complete Docker setup for consistent development environments:

.. code-block:: bash

   # Clone the repository
   git clone <repository-url>
   cd math-ai-agent

   # Build the Docker image
   docker build -t math-ai-agent .

   # Run the container
   docker run -p 7860:7860 -e OPENAI_API_KEY=your_api_key_here math-ai-agent

   # Or use docker-compose for development
   docker-compose up --build

The Docker setup includes:

* All Python dependencies pre-installed
* SymPy MCP server configuration
* Development tools (pytest, black, mypy)
* Port mapping for the Gradio interface

Local Development Setup
~~~~~~~~~~~~~~~~~~~~~~~

For local development without Docker:

.. code-block:: bash

   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-docs.txt  # For documentation

   # Set up environment variables
   cp .env.example .env
   # Edit .env and add your OpenAI API key

   # Install the package in development mode
   pip install -e .

Project Structure
-----------------

Understanding the project structure is crucial for effective development:

.. code-block:: text

   math-ai-agent/
   ├── src/                          # Main source code
   │   ├── core/                     # Core engine components
   │   │   ├── __init__.py
   │   │   ├── engine.py            # Main orchestration engine
   │   │   ├── models.py            # Data models and types
   │   │   ├── parser.py            # Problem parsing logic
   │   │   └── verifier.py          # Solution verification
   │   ├── interface/               # User interface components
   │   │   ├── __init__.py
   │   │   ├── app.py              # Gradio web interface
   │   │   └── visualizer.py       # Plotting and visualization
   │   ├── mcps/                    # MCP client implementations
   │   │   ├── __init__.py
   │   │   ├── sympy_client.py     # SymPy MCP client
   │   │   └── sympy_mcp.py        # MCP server implementation
   │   ├── solvers/                 # Domain-specific solvers
   │   │   ├── __init__.py
   │   │   ├── linear_algebra_solver.py
   │   │   ├── optimization_solver.py
   │   │   └── stats_solver.py
   │   └── utils/                   # Utility functions
   │       └── __init__.py
   ├── tests/                       # Test suite
   │   ├── __init__.py
   │   ├── benchmark_runner.py     # Performance benchmarks
   │   ├── evaluators/             # Solution evaluators
   │   └── test_*.py              # Unit and integration tests
   ├── docs/                        # Documentation (Sphinx)
   ├── prompts/                     # LLM prompt templates
   ├── data/                        # Test data and benchmarks
   ├── scripts/                     # Utility scripts
   ├── docker-compose.yml          # Docker Compose configuration
   ├── Dockerfile                  # Docker image definition
   ├── requirements.txt            # Python dependencies
   └── requirements-docs.txt       # Documentation dependencies

Development Workflow
--------------------

Code Style and Standards
~~~~~~~~~~~~~~~~~~~~~~~~~

We maintain high code quality standards:

**Python Style Guide**
   * Follow PEP 8 conventions
   * Use type hints throughout
   * Write comprehensive docstrings (Google style)
   * Maximum line length: 100 characters

**Code Formatting**
   .. code-block:: bash

      # Format code with Black
      black src/ tests/

      # Sort imports with isort
      isort src/ tests/

      # Check types with mypy
      mypy src/

**Documentation Standards**
   * All public functions must have docstrings
   * Use Google-style docstrings
   * Include examples in docstrings where helpful
   * Update documentation when adding new features

Git Workflow
~~~~~~~~~~~~

We use a standard Git workflow with feature branches:

.. code-block:: bash

   # Create a new feature branch
   git checkout -b feature/your-feature-name

   # Make your changes and commit
   git add .
   git commit -m "feat: add new optimization algorithm"

   # Push and create a pull request
   git push origin feature/your-feature-name

**Commit Message Convention**
   * ``feat:``: New features
   * ``fix:``: Bug fixes
   * ``docs:``: Documentation changes
   * ``test:``: Adding or updating tests
   * ``refactor:``: Code refactoring
   * ``style:``: Code style changes

Testing
-------

Running Tests
~~~~~~~~~~~~~

The project includes comprehensive test coverage:

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage report
   pytest --cov=src --cov-report=html

   # Run specific test file
   pytest tests/test_engine.py

   # Run tests matching a pattern
   pytest -k "test_integration"

   # Run tests with verbose output
   pytest -v

Test Structure
~~~~~~~~~~~~~~

Tests are organized by component:

* **Unit Tests**: Test individual functions and classes
* **Integration Tests**: Test component interactions
* **End-to-End Tests**: Test complete workflows
* **Benchmark Tests**: Performance and accuracy benchmarks

**Example Test Structure:**

.. code-block:: python

   # tests/test_engine.py
   import pytest
   from src.core.engine import MathAIEngine

   class TestMathAIEngine:
       def setup_method(self):
           """Set up test fixtures."""
           self.api_key = "test-key"
           self.engine = MathAIEngine(self.api_key)

       def test_initialization(self):
           """Test engine initialization."""
           assert self.engine is not None
           assert self.engine.total_problems_solved == 0

       @pytest.mark.integration
       def test_complete_pipeline(self):
           """Test complete problem-solving pipeline."""
           problem = "Solve x^2 - 4 = 0"
           result = self.engine.execute_solution_pipeline(problem, self.api_key)
           assert result.success is True

Running Benchmarks
~~~~~~~~~~~~~~~~~~

Performance benchmarks help ensure the system maintains acceptable performance:

.. code-block:: bash

   # Run benchmark suite
   python tests/benchmark_runner.py

   # Run specific benchmark
   python scripts/run_benchmark.sh

   # Generate benchmark report
   python tests/benchmark_runner.py --report

Adding New Features
-------------------

Adding a New Solver
~~~~~~~~~~~~~~~~~~~

To add a new mathematical domain solver:

1. **Create the Solver Class**

   .. code-block:: python

      # src/solvers/new_domain_solver.py
      from typing import Dict, Any, List
      from ..core.models import SolverResult

      class NewDomainSolver:
          """Solver for new mathematical domain."""
          
          def __init__(self):
              """Initialize the solver."""
              pass
          
          def solve_problem(self, expression: str, **kwargs) -> SolverResult:
              """
              Solve a problem in the new domain.
              
              Args:
                  expression: Mathematical expression to solve
                  **kwargs: Additional solver parameters
                  
              Returns:
                  SolverResult containing the solution
              """
              # Implementation here
              pass

2. **Register the Solver**

   Update ``src/core/engine.py`` to include your solver:

   .. code-block:: python

      from ..solvers.new_domain_solver import NewDomainSolver

      class MathAIEngine:
          def __init__(self, openai_api_key: str):
              # ... existing initialization
              self.new_domain_solver = NewDomainSolver()

3. **Add Problem Type Detection**

   Update ``src/core/models.py`` and ``src/core/parser.py``:

   .. code-block:: python

      # In models.py
      class ProblemType(Enum):
          # ... existing types
          NEW_DOMAIN_PROBLEM = "new_domain_problem"

      # In parser.py - add detection logic

4. **Add Tool Execution**

   Update the ``_execute_tool_call`` method in ``engine.py``:

   .. code-block:: python

      elif tool == 'new_domain_solver':
          return self._execute_new_domain_tool_call(step_number, step)

5. **Write Tests**

   Create comprehensive tests in ``tests/test_new_domain_solver.py``

Adding New Problem Types
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Extend the Enum**

   .. code-block:: python

      # src/core/models.py
      class ProblemType(Enum):
          NEW_PROBLEM_TYPE = "new_problem_type"

2. **Update Parser Patterns**

   Add detection patterns in ``src/core/parser.py``

3. **Add Solver Support**

   Ensure appropriate solvers can handle the new problem type

4. **Update Visualization**

   Add visualization logic in ``src/interface/visualizer.py`` if needed

Contributing Guidelines
-----------------------

Pull Request Process
~~~~~~~~~~~~~~~~~~~~

1. **Fork the Repository**
   Create a fork of the main repository

2. **Create Feature Branch**
   Create a new branch for your feature or bug fix

3. **Implement Changes**
   * Write clean, well-documented code
   * Add appropriate tests
   * Update documentation if needed

4. **Run Quality Checks**

   .. code-block:: bash

      # Run tests
      pytest

      # Check code style
      black --check src/ tests/
      isort --check-only src/ tests/

      # Type checking
      mypy src/

5. **Submit Pull Request**
   * Provide clear description of changes
   * Reference any related issues
   * Ensure all checks pass

Code Review Guidelines
~~~~~~~~~~~~~~~~~~~~~~

When reviewing code, consider:

* **Functionality**: Does the code work as intended?
* **Testing**: Are there adequate tests?
* **Documentation**: Is the code well-documented?
* **Performance**: Are there any performance implications?
* **Security**: Are there any security concerns?
* **Maintainability**: Is the code readable and maintainable?

Release Process
---------------

Version Management
~~~~~~~~~~~~~~~~~~

We use semantic versioning (MAJOR.MINOR.PATCH):

* **MAJOR**: Breaking changes
* **MINOR**: New features (backward compatible)
* **PATCH**: Bug fixes (backward compatible)

Creating a Release
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Update version in setup.py and __init__.py
   # Create and push tag
   git tag -a v2.0.0 -m "Release version 2.0.0"
   git push origin v2.0.0

   # Build and test release
   python setup.py sdist bdist_wheel
   twine check dist/*

Debugging and Troubleshooting
------------------------------

Common Issues
~~~~~~~~~~~~~

**Import Errors**
   * Ensure you're in the project directory
   * Activate your virtual environment
   * Install dependencies: ``pip install -r requirements.txt``

**API Key Issues**
   * Verify your OpenAI API key is valid
   * Check key has sufficient credits
   * Ensure key has GPT-4o access

**Docker Issues**
   * Rebuild image: ``docker build --no-cache -t math-ai-agent .``
   * Check port conflicts: ``docker ps``
   * View logs: ``docker logs <container_id>``

Debugging Tools
~~~~~~~~~~~~~~~

**Logging**
   The system uses Python's logging module extensively:

   .. code-block:: python

      import logging
      logging.basicConfig(level=logging.DEBUG)

**Profiling**
   For performance analysis:

   .. code-block:: bash

      # Profile a specific function
      python -m cProfile -o profile.stats script.py

**Memory Debugging**
   Use memory profiler for memory leak detection:

   .. code-block:: bash

      pip install memory-profiler
      python -m memory_profiler script.py

Development Best Practices
---------------------------

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

* **Cache Results**: Cache expensive computations
* **Lazy Loading**: Load resources only when needed
* **Parallel Processing**: Use async/await for I/O operations
* **Memory Management**: Clean up large objects when done

Security Considerations
~~~~~~~~~~~~~~~~~~~~~~

* **Input Validation**: Always validate user input
* **API Key Protection**: Never log or expose API keys
* **Output Sanitization**: Sanitize all outputs to prevent XSS
* **Dependency Management**: Keep dependencies updated

Error Handling
~~~~~~~~~~~~~~

* **Graceful Degradation**: Provide fallbacks when components fail
* **User-Friendly Messages**: Convert technical errors to user-friendly messages
* **Comprehensive Logging**: Log errors with sufficient context
* **Recovery Strategies**: Implement retry logic where appropriate

Getting Help
------------

If you need help with development:

* **Documentation**: Start with this guide and the API reference
* **Issues**: Check existing issues on GitHub
* **Community**: Join our developer discussions
* **Code Review**: Request code review from maintainers

Remember: good code is not just working code, but code that is readable, maintainable, and well-tested. Happy coding!