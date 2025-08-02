System Architecture
==================

The Math AI Agent is built on a modular, pipeline-based architecture that ensures reliability, maintainability, and extensibility. This document provides a comprehensive overview of the system's architectural components and their interactions.

Architecture Overview
---------------------

The system follows a layered architecture with clear separation of concerns:

.. mermaid::

   graph TB
       subgraph "User Interface Layer"
           UI[Gradio Web Interface]
       end
       
       subgraph "Core Engine Layer"
           ENGINE[Math AI Engine]
           PARSER[Problem Parser]
           SYNTH[Answer Synthesizer]
       end
       
       subgraph "Solver Layer"
           LA[Linear Algebra Solver]
           OPT[Optimization Solver]
           STATS[Statistics Solver]
           SYMPY[SymPy MCP Client]
       end
       
       subgraph "Verification & Visualization"
           VERIFY[Multi-layer Verifier]
           VIZ[Visualization Engine]
       end
       
       subgraph "External Services"
           OPENAI[OpenAI GPT-4o]
           MCP[SymPy MCP Server]
       end
       
       UI --> ENGINE
       ENGINE --> PARSER
       ENGINE --> LA
       ENGINE --> OPT
       ENGINE --> STATS
       ENGINE --> SYMPY
       ENGINE --> VERIFY
       ENGINE --> VIZ
       ENGINE --> SYNTH
       ENGINE --> OPENAI
       SYMPY --> MCP

Data Flow Architecture
----------------------

The complete problem-solving pipeline follows this data flow:

.. graphviz::

   digraph G {
       rankdir=TB;
       node [shape=box, style=rounded];
       
       // Input phase
       "User Problem" -> "Problem Parser";
       "Problem Parser" -> "Parsed Problem Object";
       
       // Planning phase
       "Parsed Problem Object" -> "OpenAI Planning";
       "OpenAI Planning" -> "Solution Plan";
       
       // Execution phase
       "Solution Plan" -> "Tool Execution Router";
       "Tool Execution Router" -> "Linear Algebra Solver";
       "Tool Execution Router" -> "Optimization Solver";
       "Tool Execution Router" -> "Statistics Solver";
       "Tool Execution Router" -> "SymPy MCP Client";
       
       // Results aggregation
       "Linear Algebra Solver" -> "Execution Results";
       "Optimization Solver" -> "Execution Results";
       "Statistics Solver" -> "Execution Results";
       "SymPy MCP Client" -> "Execution Results";
       
       // Verification and visualization
       "Execution Results" -> "Solution Verifier";
       "Execution Results" -> "Visualization Engine";
       
       // Final synthesis
       "Solution Verifier" -> "Answer Synthesizer";
       "Visualization Engine" -> "Answer Synthesizer";
       "Answer Synthesizer" -> "Final Solution";
   }

Core Components
---------------

Math AI Engine (``src/core/engine.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The central orchestrator that coordinates all system components. Key responsibilities:

* **Pipeline Management**: Orchestrates the complete solution workflow
* **Component Integration**: Manages interactions between all system modules
* **Error Handling**: Provides robust error handling and recovery
* **Performance Monitoring**: Tracks execution times and API usage

**Key Methods:**

* ``execute_solution_pipeline(problem_text, api_key)`` - Main entry point
* ``_parse_problem(problem_text)`` - Problem parsing coordination
* ``_create_solution_plan(parsed_problem)`` - OpenAI-powered planning
* ``_execute_plan(plan_steps)`` - Tool execution coordination

Problem Parser (``src/core/parser.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyzes mathematical problems and extracts structured information:

* **Domain Classification**: Identifies mathematical domain (algebra, calculus, etc.)
* **Problem Type Detection**: Determines specific problem type (integration, matrix operations, etc.)
* **Variable Extraction**: Identifies mathematical variables and expressions
* **Confidence Scoring**: Provides confidence metrics for parsing accuracy

**Key Classes:**

* ``MathematicalProblemParser`` - Main parser class
* ``ParsedProblem`` - Structured problem representation
* ``MathDomain`` - Enumeration of supported mathematical domains

Solver Layer
~~~~~~~~~~~~

The solver layer contains specialized modules for different mathematical domains:

**Linear Algebra Solver** (``src/solvers/linear_algebra_solver.py``)
   * Matrix operations (determinant, inverse, decompositions)
   * Eigenvalue/eigenvector computations
   * Numerical linear algebra algorithms

**Optimization Solver** (``src/solvers/optimization_solver.py``)
   * Gradient descent algorithms
   * Critical point finding
   * Multi-variable optimization
   * Convergence tracking and visualization

**Statistics Solver** (``src/solvers/stats_solver.py``)
   * Descriptive statistics
   * Hypothesis testing
   * Distribution analysis
   * Correlation analysis

**SymPy MCP Client** (``src/mcps/sympy_client.py``)
   * Symbolic mathematics interface
   * Equation solving
   * Calculus operations (derivatives, integrals)
   * Expression simplification

Verification System (``src/core/verifier.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-layer verification system ensuring solution accuracy:

**Verification Methods:**

1. **Numerical Verification**: Test solutions with random values
2. **Symbolic Verification**: Algebraic validation using SymPy
3. **Dimensional Analysis**: Check unit consistency
4. **Boundary Testing**: Validate edge cases
5. **Cross-validation**: Compare results across different methods

**Verification Confidence Levels:**

* **High (0.9+)**: Multiple verification methods passed
* **Medium (0.7-0.9)**: Some verification methods passed
* **Low (0.5-0.7)**: Limited verification possible
* **Unknown (<0.5)**: Unable to verify

Visualization Engine (``src/interface/visualizer.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generates interactive visualizations for mathematical problems:

**Supported Visualizations:**

* **Function Plots**: 2D and 3D function visualization
* **Matrix Heatmaps**: Visual representation of matrices
* **Optimization Paths**: Gradient descent convergence visualization
* **Statistical Plots**: Histograms, scatter plots, distribution curves
* **Eigenvalue Plots**: Complex plane eigenvalue visualization

User Interface Layer
--------------------

Gradio Web Interface (``src/interface/app.py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provides a user-friendly web interface with:

* **Responsive Design**: Mobile-friendly interface
* **Real-time Processing**: Live problem solving with progress feedback
* **LaTeX Rendering**: Beautiful mathematical notation display
* **Error Handling**: User-friendly error messages and troubleshooting
* **Security**: Secure API key handling

Communication Patterns
-----------------------

The system uses several communication patterns:

**1. Pipeline Pattern**
   Sequential processing through well-defined stages with clear data contracts.

**2. Strategy Pattern**
   Different solvers implement common interfaces, allowing dynamic solver selection.

**3. Observer Pattern**
   Components notify the engine of completion status and results.

**4. Factory Pattern**
   Solver factories create appropriate solver instances based on problem type.

External Dependencies
---------------------

**OpenAI GPT-4o Integration**
   * **Purpose**: Natural language understanding and solution planning
   * **Communication**: REST API calls
   * **Error Handling**: Rate limiting, timeout handling, fallback strategies

**SymPy MCP Server**
   * **Purpose**: Symbolic mathematics computations
   * **Communication**: MCP (Model Context Protocol)
   * **Benefits**: Isolated execution, precise symbolic computation

Security Architecture
---------------------

**API Key Management**
   * Keys are never stored persistently
   * Secure transmission using HTTPS
   * Session-based key handling

**Input Validation**
   * Mathematical expression sanitization
   * Input length and complexity limits
   * Malicious input detection and prevention

**Output Sanitization**
   * LaTeX output validation
   * HTML content escaping
   * Safe visualization rendering

Performance Considerations
--------------------------

**Optimization Strategies**

1. **Caching**: Cache frequently used computations
2. **Parallel Processing**: Execute independent calculations concurrently
3. **Lazy Loading**: Load heavy components only when needed
4. **Resource Management**: Monitor memory usage and cleanup resources

**Scalability Design**

* **Stateless Architecture**: Each request is independent
* **Horizontal Scaling**: Multiple instances can run concurrently
* **Resource Isolation**: Docker containerization for deployment
* **Load Balancing**: Support for multiple backend instances

Extensibility Framework
-----------------------

The architecture supports easy extension through:

**Adding New Solvers**
   1. Implement the ``BaseSolver`` interface
   2. Register the solver in the engine's routing logic
   3. Add appropriate problem type detection
   4. Include verification methods

**New Problem Types**
   1. Extend the ``ProblemType`` enumeration
   2. Update parser patterns
   3. Create solver mappings
   4. Add visualization support

**New Verification Methods**
   1. Implement the ``VerificationMethod`` interface
   2. Register in the verification system
   3. Define confidence scoring logic

Quality Assurance
-----------------

**Testing Strategy**
   * Unit tests for individual components
   * Integration tests for pipeline functionality
   * End-to-end tests for complete workflows
   * Performance benchmarks

**Code Quality**
   * Type hints throughout the codebase
   * Comprehensive docstrings
   * Automated code formatting (Black, isort)
   * Static analysis (mypy, pylint)

**Monitoring & Observability**
   * Execution time tracking
   * API usage monitoring
   * Error rate tracking
   * Performance metrics collection

This architecture ensures the Math AI Agent is robust, maintainable, and capable of handling complex mathematical problems while providing accurate, well-verified solutions.