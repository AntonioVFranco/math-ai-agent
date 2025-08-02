Adding New Modules
==================

This tutorial walks you through the process of extending the MathBoardAI Agent system by adding new solvers, problem types, and integration components. We'll use a concrete example to demonstrate the complete process.

Tutorial: Adding a New Solver
------------------------------

In this tutorial, we'll add a new "Geometry Solver" that can handle geometric problems like calculating areas, perimeters, and solving geometric equations.

Step 1: Plan Your Solver
~~~~~~~~~~~~~~~~~~~~~~~~~

Before writing code, plan what your solver will do:

**Geometry Solver Requirements:**
- Calculate areas of shapes (circle, triangle, rectangle, etc.)
- Calculate perimeters/circumferences
- Solve geometric equations (e.g., Pythagorean theorem)
- Handle coordinate geometry problems
- Support both exact and numerical results

**Integration Points:**
- Problem parser needs to recognize geometry problems
- Engine needs to route geometry problems to the new solver
- Verification system needs to validate geometric solutions
- Visualizer should support geometric plots

Step 2: Create the Solver Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create the new solver file:

.. code-block:: python

   # src/solvers/geometry_solver.py
   """
   Geometry Solver for MathBoardAI Agent
   
   Handles geometric calculations including areas, perimeters,
   coordinate geometry, and geometric theorem applications.
   """
   
   import math
   import logging
   from typing import Dict, Any, List, Optional, Tuple, Union
   from dataclasses import dataclass
   
   logger = logging.getLogger(__name__)
   
   
   @dataclass
   class GeometricShape:
       """Represents a geometric shape with its properties."""
       shape_type: str
       parameters: Dict[str, float]
       area: Optional[float] = None
       perimeter: Optional[float] = None
       
   
   class GeometrySolver:
       """
       Solver for geometric problems including area, perimeter,
       and coordinate geometry calculations.
       """
       
       def __init__(self):
           """Initialize the geometry solver."""
           self.supported_shapes = {
               'circle', 'triangle', 'rectangle', 'square', 
               'parallelogram', 'trapezoid', 'polygon'
           }
           logger.info("Geometry solver initialized")
       
       def calculate_circle_properties(self, radius: float) -> Dict[str, Any]:
           """
           Calculate area and circumference of a circle.
           
           Args:
               radius: Circle radius
               
           Returns:
               Dictionary containing area and circumference
           """
           try:
               if radius <= 0:
                   return {
                       'success': False,
                       'error': 'Radius must be positive',
                       'error_type': 'InvalidInput'
                   }
               
               area = math.pi * radius ** 2
               circumference = 2 * math.pi * radius
               
               return {
                   'success': True,
                   'shape': 'circle',
                   'radius': radius,
                   'area': area,
                   'area_exact': f'π × {radius}²',
                   'area_latex': f'\\pi \\times {radius}^2',
                   'circumference': circumference,
                   'circumference_exact': f'2π × {radius}',
                   'circumference_latex': f'2\\pi \\times {radius}',
                   'diameter': 2 * radius
               }
               
           except Exception as e:
               logger.error(f"Circle calculation failed: {str(e)}")
               return {
                   'success': False,
                   'error': f'Circle calculation failed: {str(e)}',
                   'error_type': type(e).__name__
               }
       
       def calculate_triangle_properties(self, side_a: float, side_b: float, 
                                       side_c: Optional[float] = None,
                                       base: Optional[float] = None,
                                       height: Optional[float] = None) -> Dict[str, Any]:
           """
           Calculate area and perimeter of a triangle.
           
           Args:
               side_a: First side length
               side_b: Second side length  
               side_c: Third side length (for general triangle)
               base: Base length (for area calculation with height)
               height: Height (for area calculation with base)
               
           Returns:
               Dictionary containing triangle properties
           """
           try:
               # Validate inputs
               if side_a <= 0 or side_b <= 0:
                   return {
                       'success': False,
                       'error': 'Side lengths must be positive',
                       'error_type': 'InvalidInput'
                   }
               
               result = {
                   'success': True,
                   'shape': 'triangle',
                   'side_a': side_a,
                   'side_b': side_b
               }
               
               # Calculate area and perimeter based on available information
               if base is not None and height is not None:
                   # Area using base and height
                   area = 0.5 * base * height
                   result.update({
                       'base': base,
                       'height': height,
                       'area': area,
                       'area_formula': '(1/2) × base × height',
                       'area_latex': '\\frac{1}{2} \\times \\text{base} \\times \\text{height}'
                   })
               
               elif side_c is not None:
                   # Complete triangle - use Heron's formula
                   if side_c <= 0:
                       return {
                           'success': False,
                           'error': 'All side lengths must be positive',
                           'error_type': 'InvalidInput'
                       }
                   
                   # Check triangle inequality
                   if not (side_a + side_b > side_c and 
                          side_a + side_c > side_b and 
                          side_b + side_c > side_a):
                       return {
                           'success': False,
                           'error': 'Triangle inequality not satisfied',
                           'error_type': 'InvalidTriangle'
                       }
                   
                   # Calculate using Heron's formula
                   s = (side_a + side_b + side_c) / 2  # semi-perimeter
                   area = math.sqrt(s * (s - side_a) * (s - side_b) * (s - side_c))
                   perimeter = side_a + side_b + side_c
                   
                   result.update({
                       'side_c': side_c,
                       'perimeter': perimeter,
                       'semi_perimeter': s,
                       'area': area,
                       'area_formula': "√[s(s-a)(s-b)(s-c)] where s = (a+b+c)/2",
                       'area_latex': '\\sqrt{s(s-a)(s-b)(s-c)}'
                   })
               
               elif side_a == side_b:
                   # Isosceles right triangle assumption
                   area = 0.5 * side_a * side_b
                   perimeter = side_a + side_b + math.sqrt(side_a**2 + side_b**2)
                   
                   result.update({
                       'triangle_type': 'isosceles_right',
                       'area': area,
                       'perimeter': perimeter,
                       'hypotenuse': math.sqrt(side_a**2 + side_b**2),
                       'area_formula': '(1/2) × a × b',
                       'area_latex': '\\frac{1}{2} \\times a \\times b'
                   })
               
               return result
               
           except Exception as e:
               logger.error(f"Triangle calculation failed: {str(e)}")
               return {
                   'success': False,
                   'error': f'Triangle calculation failed: {str(e)}',
                   'error_type': type(e).__name__
               }
       
       def calculate_rectangle_properties(self, length: float, width: float) -> Dict[str, Any]:
           """Calculate area and perimeter of a rectangle."""
           try:
               if length <= 0 or width <= 0:
                   return {
                       'success': False,
                       'error': 'Length and width must be positive',
                       'error_type': 'InvalidInput'
                   }
               
               area = length * width
               perimeter = 2 * (length + width)
               diagonal = math.sqrt(length**2 + width**2)
               
               return {
                   'success': True,
                   'shape': 'rectangle',
                   'length': length,
                   'width': width,
                   'area': area,
                   'perimeter': perimeter,
                   'diagonal': diagonal,
                   'area_formula': 'length × width',
                   'area_latex': '\\text{length} \\times \\text{width}',
                   'perimeter_formula': '2 × (length + width)',
                   'perimeter_latex': '2 \\times (\\text{length} + \\text{width})'
               }
               
           except Exception as e:
               logger.error(f"Rectangle calculation failed: {str(e)}")
               return {
                   'success': False,
                   'error': f'Rectangle calculation failed: {str(e)}',
                   'error_type': type(e).__name__
               }
       
       def solve_pythagorean_theorem(self, a: Optional[float] = None,
                                   b: Optional[float] = None,
                                   c: Optional[float] = None) -> Dict[str, Any]:
           """
           Solve for missing side in right triangle using Pythagorean theorem.
           
           Args:
               a: First leg (can be None to solve for)
               b: Second leg (can be None to solve for)
               c: Hypotenuse (can be None to solve for)
               
           Returns:
               Dictionary containing the solution
           """
           try:
               # Count how many values we have
               given_values = sum(x is not None for x in [a, b, c])
               
               if given_values != 2:
                   return {
                       'success': False,
                       'error': 'Exactly two values must be provided',
                       'error_type': 'InvalidInput'
                   }
               
               result = {
                   'success': True,
                   'theorem': 'Pythagorean theorem',
                   'formula': 'a² + b² = c²',
                   'formula_latex': 'a^2 + b^2 = c^2'
               }
               
               if a is None:
                   # Solve for a: a = √(c² - b²)
                   if c <= b:
                       return {
                           'success': False,
                           'error': 'Hypotenuse must be greater than other side',
                           'error_type': 'InvalidTriangle'
                       }
                   a = math.sqrt(c**2 - b**2)
                   result.update({
                       'solving_for': 'a',
                       'given': {'b': b, 'c': c},
                       'solution': a,
                       'solution_formula': '√(c² - b²)',
                       'solution_latex': '\\sqrt{c^2 - b^2}'
                   })
               
               elif b is None:
                   # Solve for b: b = √(c² - a²)
                   if c <= a:
                       return {
                           'success': False,
                           'error': 'Hypotenuse must be greater than other side',
                           'error_type': 'InvalidTriangle'
                       }
                   b = math.sqrt(c**2 - a**2)
                   result.update({
                       'solving_for': 'b',
                       'given': {'a': a, 'c': c},
                       'solution': b,
                       'solution_formula': '√(c² - a²)',
                       'solution_latex': '\\sqrt{c^2 - a^2}'
                   })
               
               else:  # c is None
                   # Solve for c: c = √(a² + b²)
                   c = math.sqrt(a**2 + b**2)
                   result.update({
                       'solving_for': 'c',
                       'given': {'a': a, 'b': b},
                       'solution': c,
                       'solution_formula': '√(a² + b²)',
                       'solution_latex': '\\sqrt{a^2 + b^2}'
                   })
               
               # Add final verification
               result.update({
                   'a': a, 'b': b, 'c': c,
                   'verification': abs(a**2 + b**2 - c**2) < 1e-10
               })
               
               return result
               
           except Exception as e:
               logger.error(f"Pythagorean theorem calculation failed: {str(e)}")
               return {
                   'success': False,
                   'error': f'Pythagorean calculation failed: {str(e)}',
                   'error_type': type(e).__name__
               }
       
       def calculate_distance_between_points(self, x1: float, y1: float,
                                           x2: float, y2: float) -> Dict[str, Any]:
           """Calculate distance between two points in 2D coordinate system."""
           try:
               distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
               
               return {
                   'success': True,
                   'problem_type': 'coordinate_geometry',
                   'point1': (x1, y1),
                   'point2': (x2, y2),
                   'distance': distance,
                   'formula': '√[(x₂-x₁)² + (y₂-y₁)²]',
                   'formula_latex': '\\sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}',
                   'calculation': f'√[({x2}-{x1})² + ({y2}-{y1})²]',
                   'calculation_latex': f'\\sqrt{{({x2}-{x1})^2 + ({y2}-{y1})^2}}'
               }
               
           except Exception as e:
               logger.error(f"Distance calculation failed: {str(e)}")
               return {
                   'success': False,
                   'error': f'Distance calculation failed: {str(e)}',
                   'error_type': type(e).__name__
               }
   
   
   def get_solver() -> GeometrySolver:
       """Get a configured geometry solver instance."""
       return GeometrySolver()

Step 3: Update the Data Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add new problem types and domains to the models:

.. code-block:: python

   # In src/core/models.py - add to existing enums
   
   class MathDomain(Enum):
       # ... existing domains
       GEOMETRY = "geometry"
   
   class ProblemType(Enum):
       # ... existing types
       # Geometry problem types
       AREA_CALCULATION = "area_calculation"
       PERIMETER_CALCULATION = "perimeter_calculation"  
       PYTHAGOREAN_THEOREM = "pythagorean_theorem"
       COORDINATE_GEOMETRY = "coordinate_geometry"
       GEOMETRIC_PROPERTIES = "geometric_properties"

Step 4: Update the Problem Parser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add recognition patterns for geometry problems:

.. code-block:: python

   # In src/core/parser.py - add to _detect_problem_type method
   
   def _detect_problem_type(self, text: str, domain: MathDomain) -> ProblemType:
       """Detect the specific type of mathematical problem."""
       text_lower = text.lower()
       
       # ... existing detection logic
       
       # Geometry problem detection
       if domain == MathDomain.GEOMETRY:
           if any(word in text_lower for word in ['area', 'surface area']):
               return ProblemType.AREA_CALCULATION
           elif any(word in text_lower for word in ['perimeter', 'circumference']):
               return ProblemType.PERIMETER_CALCULATION
           elif any(word in text_lower for word in ['pythagorean', 'right triangle', 'hypotenuse']):
               return ProblemType.PYTHAGOREAN_THEOREM
           elif any(word in text_lower for word in ['distance', 'coordinate', 'point']):
               return ProblemType.COORDINATE_GEOMETRY
           else:
               return ProblemType.GEOMETRIC_PROPERTIES
       
       # ... rest of existing logic
   
   def _detect_domain(self, text: str) -> MathDomain:
       """Detect the mathematical domain of the problem."""
       text_lower = text.lower()
       
       # ... existing domain detection
       
       # Geometry keywords
       geometry_keywords = [
           'area', 'perimeter', 'circumference', 'triangle', 'circle', 'rectangle',
           'square', 'polygon', 'pythagorean', 'hypotenuse', 'coordinate', 'distance',
           'geometric', 'geometry', 'shape', 'radius', 'diameter', 'length', 'width'
       ]
       
       if any(keyword in text_lower for keyword in geometry_keywords):
           return MathDomain.GEOMETRY
       
       # ... rest of existing logic

Step 5: Update the Engine
~~~~~~~~~~~~~~~~~~~~~~~~~

Integrate the new solver into the main engine:

.. code-block:: python

   # In src/core/engine.py - add import and initialization
   
   try:
       from ..solvers.geometry_solver import get_solver as get_geometry_solver
   except ImportError:
       from solvers.geometry_solver import get_solver as get_geometry_solver
   
   class MathAIEngine:
       def __init__(self, openai_api_key: str):
           # ... existing initialization
           self.geometry_solver = get_geometry_solver()
   
       def _execute_tool_call(self, step_number: int, step: Dict) -> ExecutionStep:
           """Execute a single tool call step."""
           try:
               tool = step.get('tool', '')
               # ... existing tool routing
               elif tool == 'geometry_solver':
                   return self._execute_geometry_tool_call(step_number, step)
               # ... rest of method
   
       def _execute_geometry_tool_call(self, step_number: int, step: Dict) -> ExecutionStep:
           """Execute a geometry solver tool call."""
           try:
               command = step.get('command', '')
               args = step.get('args', {})
               description = step.get('description', '')
               
               # Map commands to geometry solver methods
               command_mapping = {
                   'calculate_circle_properties': self.geometry_solver.calculate_circle_properties,
                   'calculate_triangle_properties': self.geometry_solver.calculate_triangle_properties,
                   'calculate_rectangle_properties': self.geometry_solver.calculate_rectangle_properties,
                   'solve_pythagorean_theorem': self.geometry_solver.solve_pythagorean_theorem,
                   'calculate_distance_between_points': self.geometry_solver.calculate_distance_between_points
               }
               
               if command not in command_mapping:
                   return ExecutionStep(
                       step_number=step_number,
                       step_type='tool_call',
                       tool='geometry_solver',
                       command=command,
                       args=args,
                       success=False,
                       error_message=f"Unknown geometry command: {command}"
                   )
               
               # Execute the tool call
               method = command_mapping[command]
               result = method(**args)
               
               return ExecutionStep(
                   step_number=step_number,
                   step_type='tool_call',
                   content=description,
                   tool='geometry_solver',
                   command=command,
                   args=args,
                   result=result,
                   success=result.get('success', False)
               )
               
           except Exception as e:
               logger.error(f"Geometry tool call execution failed: {str(e)}")
               return ExecutionStep(
                   step_number=step_number,
                   step_type='tool_call',
                   tool='geometry_solver',
                   command=step.get('command', ''),
                   args=step.get('args', {}),
                   success=False,
                   error_message=str(e)
               )

Step 6: Update the Prompt Templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update the planning prompt to include geometry solver:

.. code-block:: text

   # In prompts/planning_prompt.txt - add to Available tools section
   
   Available tools: 
   - SymPy MCP: solve_equation, simplify_expression, compute_derivative, compute_integral, matrix_operations, numerical_verification, to_latex
   - Linear Algebra Solver: compute_determinant, compute_inverse, lu_decomposition, qr_decomposition, eigen_decomposition, svd
   - Optimization Solver: gradient_descent, find_critical_points
   - Geometry Solver: calculate_circle_properties, calculate_triangle_properties, calculate_rectangle_properties, solve_pythagorean_theorem, calculate_distance_between_points
   
   For geometry problems (areas, perimeters, coordinate geometry, Pythagorean theorem), use the geometry_solver tool.

Step 7: Add Visualization Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extend the visualizer for geometric shapes:

.. code-block:: python

   # In src/interface/visualizer.py - add new methods
   
   def plot_geometric_shape(self, shape_data: Dict[str, Any], title: str = "Geometric Shape") -> Any:
       """Plot geometric shapes based on calculation results."""
       try:
           import plotly.graph_objects as go
           import numpy as np
           
           shape_type = shape_data.get('shape', '')
           
           if shape_type == 'circle':
               return self._plot_circle(shape_data, title)
           elif shape_type == 'triangle':
               return self._plot_triangle(shape_data, title)
           elif shape_type == 'rectangle':
               return self._plot_rectangle(shape_data, title)
           else:
               return None
               
       except Exception as e:
           self.logger.error(f"Geometric shape plotting failed: {e}")
           return None
   
   def _plot_circle(self, shape_data: Dict[str, Any], title: str) -> Any:
       """Plot a circle with its properties."""
       import plotly.graph_objects as go
       import numpy as np
       
       radius = shape_data.get('radius', 1)
       
       # Create circle points
       theta = np.linspace(0, 2*np.pi, 100)
       x = radius * np.cos(theta)
       y = radius * np.sin(theta)
       
       fig = go.Figure()
       
       # Plot circle
       fig.add_trace(go.Scatter(
           x=x, y=y,
           mode='lines',
           name=f'Circle (r={radius})',
           line=dict(color='blue', width=3)
       ))
       
       # Add center point
       fig.add_trace(go.Scatter(
           x=[0], y=[0],
           mode='markers',
           name='Center',
           marker=dict(color='red', size=8)
       ))
       
       # Add radius line
       fig.add_trace(go.Scatter(
           x=[0, radius], y=[0, 0],
           mode='lines+text',
           name='Radius',
           line=dict(color='red', dash='dash'),
           text=[None, f'r={radius}'],
           textposition='middle right'
       ))
       
       fig.update_layout(
           title=f"{title}<br>Area: {shape_data.get('area', 0):.2f}, Circumference: {shape_data.get('circumference', 0):.2f}",
           xaxis_title="X",
           yaxis_title="Y",
           showlegend=True,
           aspectratio=dict(x=1, y=1),
           xaxis=dict(scaleanchor="y", scaleratio=1)
       )
       
       return fig

Step 8: Add Verification Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extend the verifier to handle geometry problems:

.. code-block:: python

   # In src/core/verifier.py - add to _verify_by_domain method
   
   def _verify_by_domain(self, parsed_problem: ParsedProblem, 
                        solution_data: Dict[str, Any]) -> VerificationResult:
       """Verify solution based on mathematical domain."""
       
       # ... existing domain verification
       
       elif parsed_problem.domain == MathDomain.GEOMETRY:
           return self._verify_geometry_solution(parsed_problem, solution_data)
   
   def _verify_geometry_solution(self, parsed_problem: ParsedProblem,
                                solution_data: Dict[str, Any]) -> VerificationResult:
       """Verify geometric calculations."""
       try:
           start_time = time.time()
           warnings = []
           
           shape_type = solution_data.get('shape', '')
           
           if shape_type == 'circle':
               return self._verify_circle_calculation(solution_data, warnings, start_time)
           elif shape_type == 'triangle':
               return self._verify_triangle_calculation(solution_data, warnings, start_time)
           elif 'pythagorean' in solution_data.get('theorem', '').lower():
               return self._verify_pythagorean_calculation(solution_data, warnings, start_time)
           else:
               # Basic verification for other geometric calculations
               end_time = time.time()
               return VerificationResult(
                   is_verified=True,
                   confidence=0.7,  # Medium confidence for unspecific verification
                   method=VerificationMethod.BASIC_CHECK,
                   details="Basic geometric calculation verification passed",
                   execution_time_ms=(end_time - start_time) * 1000,
                   warnings=warnings
               )
               
       except Exception as e:
           self.logger.error(f"Geometry verification failed: {e}")
           end_time = time.time()
           return VerificationResult(
               is_verified=False,
               confidence=0.0,
               method=VerificationMethod.UNKNOWN,
               details=f"Verification failed: {str(e)}",
               execution_time_ms=(end_time - start_time) * 1000,
               warnings=[f"Verification error: {str(e)}"]
           )

Step 9: Write Comprehensive Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create tests for your new solver:

.. code-block:: python

   # tests/test_geometry_solver.py
   import pytest
   import math
   from src.solvers.geometry_solver import GeometrySolver, get_solver
   
   
   class TestGeometrySolver:
       def setup_method(self):
           """Set up test fixtures."""
           self.solver = get_solver()
       
       def test_circle_properties(self):
           """Test circle area and circumference calculations."""
           result = self.solver.calculate_circle_properties(5.0)
           
           assert result['success'] is True
           assert result['shape'] == 'circle'
           assert result['radius'] == 5.0
           assert abs(result['area'] - (math.pi * 25)) < 1e-10
           assert abs(result['circumference'] - (10 * math.pi)) < 1e-10
           assert result['diameter'] == 10.0
       
       def test_circle_invalid_radius(self):
           """Test circle calculation with invalid radius."""
           result = self.solver.calculate_circle_properties(-1.0)
           
           assert result['success'] is False
           assert 'positive' in result['error'].lower()
       
       def test_triangle_heron_formula(self):
           """Test triangle area using Heron's formula."""
           # 3-4-5 right triangle
           result = self.solver.calculate_triangle_properties(3.0, 4.0, 5.0)
           
           assert result['success'] is True
           assert result['shape'] == 'triangle'
           assert result['perimeter'] == 12.0
           assert abs(result['area'] - 6.0) < 1e-10  # Should be 6
       
       def test_triangle_inequality(self):
           """Test triangle inequality validation."""
           result = self.solver.calculate_triangle_properties(1.0, 2.0, 5.0)
           
           assert result['success'] is False
           assert 'inequality' in result['error'].lower()
       
       def test_pythagorean_solve_hypotenuse(self):
           """Test solving for hypotenuse in Pythagorean theorem."""
           result = self.solver.solve_pythagorean_theorem(a=3.0, b=4.0)
           
           assert result['success'] is True
           assert result['solving_for'] == 'c'
           assert abs(result['solution'] - 5.0) < 1e-10
           assert result['verification'] is True
       
       def test_pythagorean_solve_leg(self):
           """Test solving for leg in Pythagorean theorem."""
           result = self.solver.solve_pythagorean_theorem(a=3.0, c=5.0)
           
           assert result['success'] is True
           assert result['solving_for'] == 'b'
           assert abs(result['solution'] - 4.0) < 1e-10
           assert result['verification'] is True
       
       def test_distance_between_points(self):
           """Test distance calculation between two points."""
           result = self.solver.calculate_distance_between_points(0, 0, 3, 4)
           
           assert result['success'] is True
           assert result['point1'] == (0, 0)
           assert result['point2'] == (3, 4)
           assert abs(result['distance'] - 5.0) < 1e-10
       
       def test_rectangle_properties(self):
           """Test rectangle area and perimeter calculations."""
           result = self.solver.calculate_rectangle_properties(4.0, 3.0)
           
           assert result['success'] is True
           assert result['shape'] == 'rectangle'
           assert result['area'] == 12.0
           assert result['perimeter'] == 14.0
           assert abs(result['diagonal'] - 5.0) < 1e-10
   
   
   # Integration tests
   class TestGeometryIntegration:
       def test_engine_integration(self):
           """Test geometry solver integration with main engine."""
           from src.core.engine import MathAIEngine
           
           # This would require a valid API key in practice
           # engine = MathAIEngine("test-api-key")
           # assert hasattr(engine, 'geometry_solver')
           pass

Step 10: Update Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add your solver to the documentation:

.. code-block:: rst

   # In docs/api_reference.rst
   
   Geometry Solver
   ~~~~~~~~~~~~~~~
   
   .. automodule:: src.solvers.geometry_solver
      :members:
      :undoc-members:
      :show-inheritance:

Create a tutorial for geometry problems:

.. code-block:: rst

   # docs/tutorials/geometry.rst
   
   Geometry Tutorial
   =================
   
   This tutorial demonstrates how to solve various geometric problems using the MathBoardAI Agent.
   
   Basic Shape Calculations
   ------------------------
   
   **Circle Problems:**
   
   .. code-block:: text
   
      Calculate the area and circumference of a circle with radius 5.
   
   **Triangle Problems:**
   
   .. code-block:: text
   
      Find the area of a triangle with sides 3, 4, and 5.
   
   **Pythagorean Theorem:**
   
   .. code-block:: text
   
      In a right triangle, if one leg is 3 and the hypotenuse is 5, what is the other leg?

Step 11: Testing the Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test your new solver end-to-end:

.. code-block:: bash

   # Run all tests including your new ones
   pytest tests/test_geometry_solver.py -v
   
   # Test the complete integration
   python -c "
   from src.core.engine import execute_solution_pipeline
   result = execute_solution_pipeline(
       'Calculate the area of a circle with radius 3', 
       'your-api-key'
   )
   print(result.final_answer)
   "

Best Practices for Module Development
-------------------------------------

Code Quality Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

1. **Type Hints**: Use comprehensive type hints
2. **Error Handling**: Implement robust error handling
3. **Logging**: Add appropriate logging throughout
4. **Documentation**: Write clear docstrings and comments
5. **Testing**: Achieve high test coverage

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Caching**: Cache expensive calculations
2. **Validation**: Validate inputs early and clearly
3. **Memory**: Clean up large objects when done
4. **Complexity**: Consider algorithmic complexity

Integration Checklist
~~~~~~~~~~~~~~~~~~~~~~

Before submitting your new module:

- [ ] Solver class implemented with comprehensive methods
- [ ] Problem types added to models
- [ ] Parser updated to recognize new problem types
- [ ] Engine integration completed
- [ ] Visualization support added
- [ ] Verification methods implemented
- [ ] Comprehensive tests written
- [ ] Documentation updated
- [ ] Integration tests passing
- [ ] Performance benchmarks acceptable

Common Pitfalls to Avoid
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Missing Error Handling**: Always handle edge cases
2. **Inconsistent Interfaces**: Follow established patterns
3. **Poor Test Coverage**: Test both success and failure cases
4. **Missing Documentation**: Document all public methods
5. **Performance Issues**: Profile your code for bottlenecks

Advanced Integration Topics
---------------------------

Custom Verification Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For complex mathematical domains, you may need custom verification:

.. code-block:: python

   def _verify_custom_calculation(self, solution_data: Dict[str, Any]) -> bool:
       """Custom verification logic for specialized calculations."""
       # Implement domain-specific verification logic
       pass

Dynamic Solver Selection
~~~~~~~~~~~~~~~~~~~~~~~

For complex problems spanning multiple domains:

.. code-block:: python

   def _select_optimal_solver(self, parsed_problem: ParsedProblem) -> str:
       """Select the best solver based on problem characteristics."""
       # Implement intelligent solver selection logic
       pass

This comprehensive tutorial demonstrates how to extend the MathBoardAI Agent with new mathematical capabilities while maintaining code quality, testing standards, and integration consistency.