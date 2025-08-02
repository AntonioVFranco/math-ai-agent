"""
Pydantic Models for Mathematical Problem Parser

This module defines the data structures used by the mathematical problem parser
to represent parsed problems in a structured format.

Author: Math AI Agent Team  
Task ID: CORE-002 / F1
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class MathDomain(str, Enum):
    """Enumeration of mathematical domains."""
    ALGEBRA = "algebra"
    CALCULUS = "calculus" 
    LINEAR_ALGEBRA = "linear_algebra"
    TRIGONOMETRY = "trigonometry"
    STATISTICS = "statistics"
    OPTIMIZATION = "optimization"
    NUMBER_THEORY = "number_theory"
    GEOMETRY = "geometry"
    UNKNOWN = "unknown"


class ProblemType(str, Enum):
    """Enumeration of mathematical problem types."""
    # Algebraic operations
    SOLVE_EQUATION = "solve_equation"
    SOLVE_SYSTEM = "solve_system"
    SIMPLIFY = "simplify"
    FACTOR = "factor"
    EXPAND = "expand"
    
    # Calculus operations
    DIFFERENTIATE = "differentiate"
    INTEGRATE = "integrate"
    LIMIT = "limit"
    SERIES = "series"
    
    # Linear algebra operations
    MATRIX_OPERATIONS = "matrix_operations"
    EIGENVALUES = "eigenvalues"
    DETERMINANT = "determinant"
    INVERSE = "inverse"
    LU_DECOMPOSITION = "lu_decomposition"
    QR_DECOMPOSITION = "qr_decomposition"
    SVD = "svd"
    EIGEN_DECOMPOSITION = "eigen_decomposition"
    
    # Trigonometric operations
    TRIGONOMETRIC_SIMPLIFY = "trigonometric_simplify"
    TRIGONOMETRIC_SOLVE = "trigonometric_solve"
    
    # Statistical operations
    DESCRIPTIVE_STATISTICS = "descriptive_statistics"
    HYPOTHESIS_TEST = "hypothesis_test"
    DISTRIBUTION_ANALYSIS = "distribution_analysis"
    RANDOM_GENERATION = "random_generation"
    CORRELATION_ANALYSIS = "correlation_analysis"
    NORMALITY_TEST = "normality_test"
    T_TEST = "t_test"
    
    # Optimization operations
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    GRADIENT_DESCENT = "gradient_descent"
    FIND_CRITICAL_POINTS = "find_critical_points"
    OPTIMIZATION = "optimization"
    
    # General operations
    EVALUATE = "evaluate"
    PLOT = "plot"
    VERIFY = "verify"
    UNKNOWN = "unknown"


class Confidence(BaseModel):
    """Confidence scores for parser predictions."""
    domain: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in domain classification")
    problem_type: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in problem type identification")
    overall: float = Field(0.0, ge=0.0, le=1.0, description="Overall parsing confidence")
    
    def calculate_overall(self):
        """Calculate overall confidence as average of domain and problem_type confidence."""
        return (self.domain + self.problem_type) / 2


class MathEntity(BaseModel):
    """Represents a mathematical entity found in the problem text."""
    entity: str = Field(..., description="The mathematical entity (variable, constant, function)")
    entity_type: str = Field(..., description="Type of entity: variable, constant, function, operator")
    position: int = Field(..., description="Starting position in the original text")
    context: Optional[str] = Field(None, description="Surrounding context of the entity")


class ParsedProblem(BaseModel):
    """
    Structured representation of a parsed mathematical problem.
    
    This model contains all the information extracted from a raw text
    mathematical problem, including domain classification, problem type,
    variables, expressions, and confidence scores.
    """
    
    # Core classification
    domain: MathDomain = Field(
        MathDomain.UNKNOWN,
        description="Mathematical domain of the problem"
    )
    
    problem_type: ProblemType = Field(
        ProblemType.UNKNOWN,
        description="Type of mathematical operation requested"
    )
    
    # Mathematical entities
    variables: List[str] = Field(
        default_factory=list,
        description="List of variables found in the problem"
    )
    
    expression: Optional[str] = Field(
        None,
        description="The core mathematical expression extracted from the text"
    )
    
    # Additional mathematical entities
    constants: List[str] = Field(
        default_factory=list,
        description="List of mathematical constants found (pi, e, etc.)"
    )
    
    functions: List[str] = Field(
        default_factory=list,  
        description="List of mathematical functions found (sin, cos, log, etc.)"
    )
    
    operators: List[str] = Field(
        default_factory=list,
        description="List of mathematical operators found (+, -, *, /, ^, etc.)"
    )
    
    # Detailed entity information
    entities: List[MathEntity] = Field(
        default_factory=list,
        description="Detailed information about all mathematical entities"
    )
    
    # Original text and processing info
    original_text: str = Field(
        ...,
        description="Original problem text"
    )
    
    cleaned_text: Optional[str] = Field(
        None,
        description="Cleaned version of the original text"
    )
    
    # Confidence and metadata
    confidence: Confidence = Field(
        default_factory=Confidence,
        description="Confidence scores for parser predictions"
    )
    
    parsing_notes: List[str] = Field(
        default_factory=list,
        description="Notes about the parsing process, warnings, or ambiguities"
    )
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the parsing process"
    )
    
    @validator('variables')
    def validate_variables(cls, v):
        """Ensure variables are unique and properly formatted."""
        if v:
            # Remove duplicates while preserving order
            seen = set()
            unique_vars = []
            for var in v:
                if var not in seen:
                    seen.add(var)
                    unique_vars.append(var)
            return unique_vars
        return v
    
    @validator('expression')
    def validate_expression(cls, v):
        """Basic validation of mathematical expression."""
        if v:
            # Remove extra whitespace
            return ' '.join(v.split())
        return v
    
    def add_parsing_note(self, note: str) -> None:
        """Add a note about the parsing process."""
        if note not in self.parsing_notes:
            self.parsing_notes.append(note)
    
    def set_confidence(self, domain_conf: float, problem_type_conf: float) -> None:
        """Set confidence scores for the parsing results."""
        self.confidence.domain = max(0.0, min(1.0, domain_conf))
        self.confidence.problem_type = max(0.0, min(1.0, problem_type_conf))
        # overall confidence is automatically calculated by the validator
    
    def has_clear_intent(self, threshold: float = 0.7) -> bool:
        """Check if the parsing has clear intent based on confidence scores."""
        return self.confidence.overall >= threshold
    
    def is_solvable(self) -> bool:
        """Check if the problem appears to be solvable based on parsed information."""
        return (
            self.domain != MathDomain.UNKNOWN and
            self.problem_type != ProblemType.UNKNOWN and
            (self.expression is not None or len(self.variables) > 0)
        )
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the parsed problem."""
        summary_parts = []
        
        if self.domain != MathDomain.UNKNOWN:
            summary_parts.append(f"Domain: {self.domain.value}")
        
        if self.problem_type != ProblemType.UNKNOWN:
            summary_parts.append(f"Type: {self.problem_type.value}")
        
        if self.variables:
            summary_parts.append(f"Variables: {', '.join(self.variables)}")
        
        if self.expression:
            summary_parts.append(f"Expression: {self.expression}")
        
        summary_parts.append(f"Confidence: {self.confidence.overall:.2f}")
        
        return " | ".join(summary_parts)


class ParsingResult(BaseModel):
    """
    Result container for the parsing operation.
    
    This model wraps the ParsedProblem with additional information
    about the parsing process, including success status and any errors.
    """
    
    success: bool = Field(..., description="Whether parsing was successful")
    
    parsed_problem: Optional[ParsedProblem] = Field(
        None,
        description="The parsed problem if parsing was successful"
    )
    
    error_message: Optional[str] = Field(
        None,
        description="Error message if parsing failed"
    )
    
    parsing_time_ms: Optional[float] = Field(
        None,
        description="Time taken for parsing in milliseconds"
    )
    
    parser_version: str = Field(
        "1.0.0",
        description="Version of the parser used"
    )
    
    @classmethod
    def success_result(cls, parsed_problem: ParsedProblem, parsing_time_ms: float = None) -> 'ParsingResult':
        """Create a successful parsing result."""
        return cls(
            success=True,
            parsed_problem=parsed_problem,
            parsing_time_ms=parsing_time_ms
        )
    
    @classmethod
    def error_result(cls, error_message: str, parsing_time_ms: float = None) -> 'ParsingResult':
        """Create an error parsing result."""
        return cls(
            success=False,
            error_message=error_message,
            parsing_time_ms=parsing_time_ms
        )


# Type aliases for convenience
MathDomainType = MathDomain
ProblemTypeType = ProblemType